"""
Just-Printing-Server: 极简打印中继服务
FastAPI后端主文件

技术约束：
- 无数据库、无用户体系、无持久化存储
- 所有文件仅存内存或临时目录，请求结束立即销毁
- 所有日志仅输出到 stdout
- 所有配置通过环境变量注入

API端点：
- POST /auth: 认证
- POST /upload: 文件上传
- GET /preview.pdf: PDF预览
- GET /printer/status: 打印机状态
- POST /print: 打印投递
- POST /cancel: 取消会话
"""


import os
import io
import struct
import logging
import asyncio
import aiohttp
import tempfile
import mimetypes
from typing import Dict, Optional, List, Tuple
from datetime import datetime
from dataclasses import dataclass, field
from fastapi import FastAPI, HTTPException, Depends, Header, Query, status, UploadFile, File, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse, HTMLResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from PIL import Image
import img2pdf
import PyPDF2
from pyipp import IPP
from pyipp.exceptions import IPPError

# 配置日志
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="[%(asctime)s] [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# 创建FastAPI应用
app = FastAPI(
    title="Just-Printing-Server",
    description="极简打印中继服务",
    version="1.0.0"
)

# 限流器配置
limiter = Limiter(key_func=get_remote_address, default_limits=[])

# 自定义限流错误处理
async def rate_limit_exceeded_handler(request, exc):
    return JSONResponse(
        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
        content={"error": "请求过于频繁，请稍后再试"}
    )

# 限流器集成
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, rate_limit_exceeded_handler)

# 会话数据定义
@dataclass
class SessionData:
    token: str = ""
    pdf_bytes: bytes = b""
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_accessed: datetime = field(default_factory=datetime.utcnow)
    files: List[dict] = field(default_factory=list)

# 全局会话存储
sessions: Dict[str, SessionData] = {}

# 会话管理函数
def create_session(token: str) -> SessionData:
    """创建新会话，如果已存在则覆盖"""
    session = SessionData(token=token)
    sessions[token] = session
    logger.info(f"创建会话: {token[:8]}...")
    return session

def get_session(token: str, max_age_seconds: int = 1800) -> Optional[SessionData]:
    """获取会话并更新最后访问时间，如果会话过期则删除"""
    session = sessions.get(token)
    if not session:
        return None
    # 检查过期
    now = datetime.utcnow()
    age = (now - session.last_accessed).total_seconds()
    if age > max_age_seconds:
        logger.info(f"会话过期: {token[:8]}... (未活动 {age:.0f}秒)")
        del sessions[token]
        return None
    # 更新最后访问时间和token字段
    session.last_accessed = now
    session.token = token
    logger.debug(f"访问会话: {token[:8]}...")
    return session

def delete_session(token: str) -> bool:
    """删除会话，返回是否成功"""
    if token in sessions:
        del sessions[token]
        logger.info(f"删除会话: {token[:8]}...")
        return True
    return False

def cleanup_sessions(max_age_seconds: int = 1800) -> int:
    """清理超过指定时间未活动的会话，返回清理数量"""
    if not sessions:
        return 0
    now = datetime.utcnow()
    expired_tokens = []
    for token, session in sessions.items():
        age = (now - session.last_accessed).total_seconds()
        if age > max_age_seconds:
            expired_tokens.append(token)
    for token in expired_tokens:
        del sessions[token]
        logger.info(f"清理过期会话: {token[:8]}... (未活动 {max_age_seconds}秒)")
    return len(expired_tokens)

# 认证依赖
security = HTTPBearer(auto_error=False)

async def verify_token(
    authorization: Optional[HTTPAuthorizationCredentials] = Depends(security),
    token_query: Optional[str] = Query(None, alias="token")
) -> str:
    """
    验证token，支持两种方式：
    1. Authorization头: Bearer <token>
    2. 查询参数: ?token=<token>

    返回验证通过的token字符串，否则抛出HTTPException 401
    """
    # 优先使用Authorization头
    token = None
    if authorization and authorization.credentials:
        token = authorization.credentials
    elif token_query:
        token = token_query

    if not token:
        logger.warning("认证失败：未提供token")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"error": "未提供访问令牌"}
        )

    # 验证token是否等于环境变量ACCESS_TOKEN
    if token != config.ACCESS_TOKEN:
        logger.warning(f"认证失败：无效token (提供: {token[:8]}...)")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"error": "无效的访问令牌"}
        )

    logger.info(f"认证成功：token {token[:8]}...")
    return token

# 会话依赖函数
async def require_session(token: str = Depends(verify_token)) -> SessionData:
    """依赖函数：验证token并返回会话，如果不存在则返回404"""
    session = get_session(token)
    if not session:
        logger.warning(f"会话不存在: {token[:8]}...")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"error": "会话不存在或已过期"}
        )
    return session

async def get_or_create_session(token: str = Depends(verify_token)) -> SessionData:
    """依赖函数：验证token并返回会话，如果不存在则创建"""
    session = get_session(token)
    if not session:
        session = create_session(token)
        logger.info(f"为新token创建会话: {token[:8]}...")
    else:
        # 确保会话有token字段（兼容之前创建的会话）
        session.token = token
        session.last_accessed = datetime.utcnow()
    return session

# 环境变量配置
class Config:
    PRINTER_IPP_URL = os.getenv("PRINTER_IPP_URL", "")
    PRINTER_NAME = os.getenv("PRINTER_NAME", "Unknown Printer")
    ACCESS_TOKEN = os.getenv("ACCESS_TOKEN", "")
    MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "50"))
    RATE_LIMIT_PER_IP = os.getenv("RATE_LIMIT_PER_IP", "5/minute")
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

config = Config()

# 全局并发控制信号量
conversion_semaphore = asyncio.Semaphore(2)

# 支持的文件扩展名和MIME类型
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".pdf"}
ALLOWED_MIME_TYPES = {
    "image/jpeg",
    "image/png",
    "application/pdf"
}

# 文件验证函数
def validate_file(file: UploadFile) -> Tuple[bool, str]:
    """验证文件大小、类型和安全性，返回 (是否有效, 错误信息)"""
    # 检查文件大小
    max_size_bytes = config.MAX_UPLOAD_MB * 1024 * 1024
    file.file.seek(0, 2)  # 移动到文件末尾
    file_size = file.file.tell()
    file.file.seek(0)  # 重置文件指针
    if file_size > max_size_bytes:
        return False, f"文件大小超过限制 ({config.MAX_UPLOAD_MB}MB)"

    # 检查文件扩展名
    filename = file.filename or ""
    ext = os.path.splitext(filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        return False, "不支持的文件格式，仅支持 JPG/PNG/PDF"

    # 检查MIME类型（基于扩展名）
    mime_type, _ = mimetypes.guess_type(filename)
    if mime_type and mime_type not in ALLOWED_MIME_TYPES:
        return False, "不支持的文件类型，仅支持 JPG/PNG/PDF"

    # 文件名安全性检查
    safe_filename = os.path.basename(filename)
    if safe_filename != filename:
        return False, "文件名包含非法路径字符"

    return True, ""

# 文件转换函数
async def convert_to_pdf(file: UploadFile) -> bytes:
    """将上传文件转换为PDF字节流"""
    filename = file.filename or ""
    ext = os.path.splitext(filename)[1].lower()

    # 读取文件内容
    file.file.seek(0)
    file_bytes = await file.read()

    # 如果是PDF，直接返回
    if ext == ".pdf":
        return file_bytes

    # 图片转PDF
    try:
        # 使用asyncio.to_thread处理CPU密集型操作
        def convert_image():
            image = Image.open(io.BytesIO(file_bytes))
            # 转换为RGB模式（兼容所有图片）
            if image.mode in ("RGBA", "P", "LA"):
                image = image.convert("RGB")
            elif image.mode != "RGB":
                image = image.convert("RGB")
            # 使用临时文件保存图像，然后使用img2pdf转换
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
                image.save(tmp, format='JPEG', quality=95)
                tmp_path = tmp.name
            try:
                pdf_bytes = img2pdf.convert(tmp_path)
            finally:
                os.unlink(tmp_path)
            return pdf_bytes

        pdf_bytes = await asyncio.to_thread(convert_image)
        return pdf_bytes
    except Exception as e:
        logger.error(f"图片转PDF失败: {e}")
        raise ValueError(f"图片转换失败: {str(e)}")

# PDF合并函数
def merge_pdfs(pdf_list: List[bytes]) -> bytes:
    """合并多个PDF字节流为一个PDF"""
    if not pdf_list:
        return b""

    merger = PyPDF2.PdfMerger()

    for i, pdf_bytes in enumerate(pdf_list):
        try:
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_bytes))
            merger.append(pdf_reader)
        except Exception as e:
            logger.error(f"合并第{i+1}个PDF失败: {e}")
            raise ValueError(f"PDF合并失败: {str(e)}")

    output = io.BytesIO()
    merger.write(output)
    merger.close()

    return output.getvalue()

# 获取PDF页数
def get_pdf_page_count(pdf_bytes: bytes) -> int:
    """计算PDF的页数"""
    if not pdf_bytes:
        return 0
    try:
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_bytes))
        return len(pdf_reader.pages)
    except:
        return 0

# 数据模型
class AuthRequest(BaseModel):
    token: str

class PrintRequest(BaseModel):
    copies: int = 1
    sides: str = "one-sided"  # "one-sided" or "two-sided"
    color_mode: str = "monochrome"  # "monochrome" or "color"

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    try:
        with open("index.html", "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Frontend file not found")

@app.get("/health")
async def health_check():
    """健康检查端点"""
    return {"status": "healthy"}

@app.post("/auth")
async def authenticate(auth_request: AuthRequest):
    """
    认证端点，验证token是否有效

    请求体: {"token": "string"}
    响应: 200 成功，401 无效token
    """
    token = auth_request.token

    if not token:
        logger.warning("认证失败：空token")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"error": "未提供访问令牌"}
        )

    if token != config.ACCESS_TOKEN:
        logger.warning(f"认证失败：无效token (提供: {token[:8]}...)")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"error": "无效的访问令牌"}
        )

    logger.info(f"认证成功：token {token[:8]}...")
    return {"status": "authenticated", "message": "认证成功"}

@app.post("/upload")
@limiter.limit(config.RATE_LIMIT_PER_IP)
async def upload_file(
    request: Request,
    file: UploadFile = File(..., description="上传的文件（JPG/PNG/PDF）"),
    session: SessionData = Depends(get_or_create_session)
):
    """
    文件上传端点
    接收文件，验证大小和类型，转换为PDF并合并到会话的PDF中
    """
    logger.info(f"开始处理文件上传: {file.filename}")

    # 1. 文件验证
    is_valid, error_msg = validate_file(file)
    if not is_valid:
        logger.warning(f"文件验证失败: {file.filename} - {error_msg}")
        # 根据错误类型确定状态码
        if "文件大小超过限制" in error_msg:
            status_code = status.HTTP_413_REQUEST_ENTITY_TOO_LARGE
        else:
            status_code = status.HTTP_400_BAD_REQUEST
        raise HTTPException(
            status_code=status_code,
            detail={"error": error_msg}
        )

    # 2. 使用信号量控制并发转换
    async with conversion_semaphore:
        try:
            # 3. 转换为PDF
            pdf_bytes = await convert_to_pdf(file)

            # 4. 创建文件信息（包含PDF字节）
            file_info = {
                "filename": file.filename,
                "size": len(pdf_bytes),
                "pages": get_pdf_page_count(pdf_bytes),
                "uploaded_at": datetime.utcnow().isoformat(),
                "pdf_bytes": pdf_bytes  # 存储单个文件的PDF字节
            }

            # 5. 重新生成合并PDF（包括新文件）
            # 收集所有文件的PDF字节
            all_pdf_bytes = []
            for f in session.files:
                if "pdf_bytes" in f:
                    all_pdf_bytes.append(f["pdf_bytes"])
            all_pdf_bytes.append(pdf_bytes)  # 添加当前文件

            merged_pdf = merge_pdfs(all_pdf_bytes)

            # 6. 更新会话
            session.files.append(file_info)
            session.pdf_bytes = merged_pdf
            session.last_accessed = datetime.utcnow()

            # 7. 计算总页数和大小
            total_pages = get_pdf_page_count(merged_pdf)
            total_size = len(merged_pdf)

            logger.info(f"文件上传成功: {file.filename}, 总页数: {total_pages}, 总大小: {total_size}字节")

            # 8. 返回响应
            return {
                "pages": total_pages,
                "size": total_size,
                "preview_url": f"/preview.pdf?token={session.token}"
            }

        except ValueError as e:
            logger.error(f"文件转换失败: {file.filename} - {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail={"error": f"文件转换失败: {str(e)}"}
            )
        except Exception as e:
            logger.error(f"上传处理异常: {file.filename} - {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail={"error": "服务器内部错误"}
            )

@app.get("/preview.pdf")
async def preview_pdf(
    session: SessionData = Depends(require_session)
):
    """
    预览会话中合并的PDF

    支持两种认证方式：
    1. 查询参数: ?token=<token>
    2. Authorization头: Bearer <token>

    返回PDF字节流，Content-Type: application/pdf
    如果会话不存在或PDF为空，返回404错误
    """
    logger.info(f"预览PDF请求: token {session.token[:8]}...")

    # 检查PDF数据是否为空
    if not session.pdf_bytes:
        logger.warning(f"PDF为空: token {session.token[:8]}...")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"error": "PDF为空，请先上传文件"}
        )

    # 创建字节流响应
    return StreamingResponse(
        io.BytesIO(session.pdf_bytes),
        media_type="application/pdf",
        headers={
            "Content-Disposition": f'inline; filename="preview.pdf"',
            "Content-Length": str(len(session.pdf_bytes)),
            "Cache-Control": "no-cache"
        }
    )


@app.get("/printer/status")
async def get_printer_status(token: str = Depends(verify_token)):
    """
    查询IPP打印机状态

    返回:
    - 在线: {"status": "online"}
    - 离线: {"status": "offline", "error": "错误详情"}

    需要认证: Authorization头或token查询参数
    """
    logger.info(f"查询打印机状态: token {token[:8]}...")

    # 检查打印机URL是否配置
    if not config.PRINTER_IPP_URL:
        logger.error("打印机未配置: PRINTER_IPP_URL 环境变量为空")
        return {"status": "offline", "error": "打印机未配置"}

    try:
        # 使用异步客户端连接打印机
        async with IPP(config.PRINTER_IPP_URL) as ipp:
            printer = await ipp.printer()
            logger.info(f"打印机在线: {config.PRINTER_NAME} ({config.PRINTER_IPP_URL})")
            return {"status": "online"}  # 已修改，不返回具体名称
    except asyncio.TimeoutError:
        logger.warning(f"打印机连接超时: {config.PRINTER_NAME} ({config.PRINTER_IPP_URL})")
        return {"status": "offline", "error": "连接打印机超时（5秒）"}
    except Exception as e:
        logger.warning(f"打印机连接失败: {config.PRINTER_NAME} ({config.PRINTER_IPP_URL}) - {str(e)}")
        return {"status": "offline", "error": f"打印机连接失败: {str(e)}"}


def build_ipp_print_request(printer_uri: str, pdf_bytes: bytes, copies: int, sides: str, color_mode: str) -> bytes:
    """
    手动构建标准的 IPP Print-Job 协议二进制 Payload
    完全避免第三方库对打印属性支持不全的问题
    """
    # IPP/1.1 (0x0101), Operation: Print-Job (0x0002), Request ID: 1 (0x00000001)
    req = bytearray(b'\x01\x01\x00\x02\x00\x00\x00\x01')
    
    # 1. Operation Attributes Group (0x01)
    req.append(0x01)
    
    # attributes-charset = utf-8
    req.extend(b'\x47\x00\x12attributes-charset\x00\x05utf-8')
    # attributes-natural-language = en-us
    req.extend(b'\x48\x00\x1battributes-natural-language\x00\x05en-us')
    
    # printer-uri
    req.extend(b'\x45\x00\x0bprinter-uri')
    req.extend(struct.pack('>H', len(printer_uri)))
    req.extend(printer_uri.encode('utf-8'))
    
    # requesting-user-name = fastapi
    req.extend(b'\x42\x00\x14requesting-user-name\x00\x07fastapi')
    
    # document-format = application/pdf
    req.extend(b'\x49\x00\x0fdocument-format\x00\x0fapplication/pdf')
    
    # 2. Job Attributes Group (0x02)
    req.append(0x02)
    
    # copies (integer)
    req.extend(b'\x21\x00\x06copies\x00\x04')
    req.extend(struct.pack('>I', copies))
    
    # sides (keyword)
    req.extend(b'\x44\x00\x05sides')
    req.extend(struct.pack('>H', len(sides)))
    req.extend(sides.encode('utf-8'))
    
    # print-color-mode (keyword)
    req.extend(b'\x44\x00\x10print-color-mode')
    req.extend(struct.pack('>H', len(color_mode)))
    req.extend(color_mode.encode('utf-8'))
    
    # 3. End of Attributes (0x03)
    req.append(0x03)
    
    # 4. Document Content (追加真实的 PDF 二进制数据)
    req.extend(pdf_bytes)
    
    return bytes(req)

@app.post("/print")
async def print_document(
    print_request: PrintRequest,
    session: SessionData = Depends(require_session)
):
    """
    打印投递端点
    直接通过 HTTP POST 构建原生 IPP 数据包发给打印机，不依赖任何宿主机 CUPS/lp 命令
    """
    logger.info(f"打印请求: token {session.token[:8]}..., copies={print_request.copies}, sides={print_request.sides}, color_mode={print_request.color_mode}")

    # 1. 验证参数
    if print_request.copies < 1 or print_request.copies > 99:
        raise HTTPException(status_code=400, detail={"error": "份数必须在1到99之间"})
    if print_request.sides not in ["one-sided", "two-sided"]:
        raise HTTPException(status_code=400, detail={"error": "单双面设置必须是 'one-sided' 或 'two-sided'"})
    if print_request.color_mode not in ["monochrome", "color"]:
        raise HTTPException(status_code=400, detail={"error": "色彩模式必须是 'monochrome' 或 'color'"})

    # 2. 检查PDF数据
    if not session.pdf_bytes:
        raise HTTPException(status_code=404, detail={"error": "PDF为空，请先上传文件"})

    # 3. 读取并转换打印机 URI
    # 从 .env 读取 ipp://localhost:631/printers/LJ4000D
    printer_uri = os.getenv("PRINTER_IPP_URL", "ipp://localhost:631/printers/default")
    
    # aiohttp 走的是原生 TCP 请求，需要将 ipp:// 转换为 http://，ipps:// 转换为 https://
    http_url = printer_uri.replace("ipp://", "http://").replace("ipps://", "https://")

    # 4. 映射 IPP 标准属性关键字
    ipp_sides = "two-sided-long-edge" if print_request.sides == "two-sided" else "one-sided"
    ipp_color = "monochrome" if print_request.color_mode == "monochrome" else "color"

    # 5. 构建 IPP 协议 Payload
    ipp_payload = build_ipp_print_request(
        printer_uri=printer_uri,
        pdf_bytes=session.pdf_bytes,
        copies=print_request.copies,
        sides=ipp_sides,
        color_mode=ipp_color
    )

    try:
        # 设置超时
        timeout = aiohttp.ClientTimeout(total=30)
        async with aiohttp.ClientSession(timeout=timeout) as http_session:
            logger.info(f"正在发送 IPP 打印任务到: {http_url} (URI: {printer_uri})")
            
            # IPP 协议固定使用的 Content-Type
            headers = {"Content-Type": "application/ipp"}
            
            # 执行纯净的网络 POST 请求
            async with http_session.post(http_url, data=ipp_payload, headers=headers) as resp:
                response_bytes = await resp.read()

                if resp.status != 200:
                    logger.error(f"HTTP 请求失败, Status: {resp.status}")
                    raise HTTPException(status_code=503, detail=f"打印机/服务响应异常 HTTP {resp.status}")

                # 解析 IPP 响应状态 (响应包的前四个字节：2字节版本 + 2字节状态码)
                if len(response_bytes) >= 4:
                    status_code = struct.unpack('>H', response_bytes[2:4])[0]
                    # IPP 规范中 0x0000~0x0007 均代表各类“成功”状态
                    if status_code in (0x0000, 0x0001, 0x0002):
                        logger.info(f"打印任务提交成功, IPP Status Code: 0x{status_code:04x}")
                    else:
                        logger.error(f"打印机拒绝任务, IPP Status Code: 0x{status_code:04x}")
                        raise HTTPException(status_code=503, detail=f"打印被拒绝 (IPP 状态码 0x{status_code:04x})")
                else:
                    logger.error("响应的数据不符合 IPP 格式")
                    raise HTTPException(status_code=503, detail="打印机返回了无效数据")

    except asyncio.TimeoutError:
        logger.error("打印请求网络超时")
        raise HTTPException(status_code=503, detail={"error": "打印机网络连接超时"})
    except aiohttp.ClientError as e:
        logger.error(f"网络连接失败: {str(e)}")
        raise HTTPException(status_code=503, detail={"error": f"无法连接到打印机网络: {str(e)}"})
    except Exception as e:
        logger.error(f"未知打印异常: {str(e)}")
        raise HTTPException(status_code=500, detail={"error": f"服务器内部打印异常: {str(e)}"})
    finally:
        # 无论成功与否，销毁当前会话状态，并且由于不再写文件，也不需要清理 tempfile 了
        delete_session(session.token)

    return {"status": "success", "message": "打印任务已提交到打印队列"}


@app.post("/cancel")
async def cancel_session(
    token: str = Depends(verify_token)
):
    """
    取消当前会话，清空上传的文件和合并PDF

    无论会话是否存在，都返回200成功状态
    如果会话存在则销毁，如果不存在也不返回错误
    """
    deleted = delete_session(token)

    if deleted:
        logger.info(f"会话已取消: token {token[:8]}...")
        return {"status": "cancelled", "message": "会话已取消"}
    else:
        logger.info(f"取消操作: token {token[:8]}... (会话不存在)")
        return {"status": "cancelled", "message": "会话不存在或已被取消"}


# 其他端点将在后续模块中添加

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=3001)
