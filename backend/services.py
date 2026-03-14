"""
服务模块 - 包含所有业务逻辑服务

按功能领域用注释分隔：
1. 会话管理
2. 认证
3. 文件处理
4. 打印服务
"""

import os
import io
import struct
import secrets
import logging
import asyncio
import tempfile
import mimetypes
from typing import Dict, Optional, List, Tuple
from datetime import datetime
from dataclasses import dataclass, field

from fastapi import Depends, HTTPException, Query, status, UploadFile
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from PIL import Image
import img2pdf
import PyPDF2
from pyipp import IPP

from .config import config

# 配置日志
logging.basicConfig(
    level=config.LOG_LEVEL,
    format="[%(asctime)s] [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


# ============================================================================
# 1. 会话管理
# ============================================================================

@dataclass
class SessionData:
    """会话数据类"""
    token: str = ""
    pdf_bytes: bytes = b""
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_accessed: datetime = field(default_factory=datetime.utcnow)
    files: List[dict] = field(default_factory=list)


# 全局会话存储
sessions: Dict[str, SessionData] = {}


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


# ============================================================================
# 2. 认证
# ============================================================================

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

    # 验证token是否等于环境变量ACCESS_TOKEN (使用 timing-safe 比较)
    if not secrets.compare_digest(token, config.ACCESS_TOKEN):
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


# ============================================================================
# 3. 文件处理
# ============================================================================

# 全局并发控制信号量
conversion_semaphore = asyncio.Semaphore(2)

# 支持的文件扩展名和MIME类型
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".pdf"}
ALLOWED_MIME_TYPES = {
    "image/jpeg",
    "image/png",
    "application/pdf"
}


def validate_file(file: UploadFile) -> Tuple[bool, str]:
    """验证文件大小、类型和安全性，返回 (是否有效, 错误信息)"""
    # 检查文件大小
    max_size_bytes = config.MAX_UPLOAD_MB * 1024 * 1024
    file.file.seek(0, 2)  # 移动到文件末尾
    file_size = file.file.tell()
    file.file.seek(0)  # 重置文件指针（供后续读取使用）
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


def get_merged_pdf(session: SessionData) -> bytes:
    """
    从会话中的单个文件动态生成合并PDF
    仅在需要时（如预览、打印）才执行合并操作
    """
    if not session.files:
        return b""

    # 收集所有文件的PDF字节
    pdf_list = []
    for f in session.files:
        if "pdf_bytes" in f and f["pdf_bytes"]:
            pdf_list.append(f["pdf_bytes"])

    return merge_pdfs(pdf_list)


def get_pdf_page_count(pdf_bytes: bytes) -> int:
    """计算PDF的页数"""
    if not pdf_bytes:
        return 0
    try:
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_bytes))
        return len(pdf_reader.pages)
    except Exception:
        return 0


# ============================================================================
# 4. 打印服务
# ============================================================================

def build_ipp_print_request(printer_uri: str, pdf_bytes: bytes, copies: int, sides: str, color_mode: str) -> bytes:
    """
    手动构建标准的 IPP Print-Job 协议二进制 Payload
    完全避免第三方库对打印属性支持不全的问题
    """

    # ============================================================================
    # IPP 协议常量定义
    # 参考: RFC 8010 (IPP/1.1) 和 RFC 8011 (IPP Model)
    # ============================================================================

    # IPP 版本和操作码
    IPP_VERSION_1_1 = 0x0101          # IPP/1.1
    IPP_OP_PRINT_JOB = 0x0002         # Print-Job 操作
    IPP_REQUEST_ID = 1                 # 请求 ID

    # IPP 属性组标签 (Group Tags)
    IPP_GROUP_OPERATION = 0x01         # Operation Attributes Group
    IPP_GROUP_JOB = 0x02               # Job Attributes Group
    IPP_GROUP_END = 0x03               # End of Attributes Group

    # IPP 属性类型标签 (Value Tags) - out-of-band 和文本类型
    IPP_TAG_CHARSET = 0x47             # attribute-charset (charset)
    IPP_TAG_NATURAL_LANG = 0x48        # attributes-natural-language (naturalLanguage)
    IPP_TAG_URI = 0x45                 # printer-uri (uri)
    IPP_TAG_NAME = 0x42                # requesting-user-name (name)
    IPP_TAG_KEYWORD = 0x44             # sides, print-color-mode (keyword)
    IPP_TAG_MIME_TYPE = 0x49           # document-format (mimeMediaType)
    IPP_TAG_INTEGER = 0x21             # copies (integer)

    # ============================================================================
    # 构建 IPP 请求
    # ============================================================================

    # IPP/1.1 (0x0101), Operation: Print-Job (0x0002), Request ID: 1 (0x00000001)
    req = bytearray(b'\x01\x01\x00\x02\x00\x00\x00\x01')

    # 1. Operation Attributes Group (0x01)
    req.append(IPP_GROUP_OPERATION)

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
    req.append(IPP_GROUP_JOB)

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
    req.append(IPP_GROUP_END)

    # 4. Document Content (追加真实的 PDF 二进制数据)
    req.extend(pdf_bytes)

    return bytes(req)


async def get_printer_status(token: str) -> dict:
    """
    查询IPP打印机状态

    返回:
    - 在线: {"status": "online"}
    - 离线: {"status": "offline", "error": "错误详情"}
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
            return {"status": "online"}  # 不返回具体名称
    except asyncio.TimeoutError:
        logger.warning(f"打印机连接超时: {config.PRINTER_NAME} ({config.PRINTER_IPP_URL})")
        return {"status": "offline", "error": "连接打印机超时（5秒）"}
    except Exception as e:
        logger.warning(f"打印机连接失败: {config.PRINTER_NAME} ({config.PRINTER_IPP_URL}) - {str(e)}")
        return {"status": "offline", "error": f"打印机连接失败: {str(e)}"}
