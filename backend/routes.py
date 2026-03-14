"""
API 路由模块 - 包含所有 API 端点
"""

import io
import asyncio
import struct
import secrets
from datetime import datetime
from typing import List

from fastapi import FastAPI, HTTPException, Depends, Header, Query, status, UploadFile, File, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import aiohttp

from .config import config
from .models import AuthRequest, PrintRequest
from .services import (
    SessionData,
    verify_token,
    require_session,
    get_or_create_session,
    delete_session,
    validate_file,
    convert_to_pdf,
    merge_pdfs,
    get_merged_pdf,
    get_pdf_page_count,
    conversion_semaphore,
    get_printer_status,
    get_printer_capabilities,
    build_ipp_print_request,
    get_session,
)

# 配置日志
import logging
logging.basicConfig(
    level=config.LOG_LEVEL,
    format="[%(asctime)s] [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


# 限流器配置
limiter = Limiter(key_func=get_remote_address, default_limits=[])

# 自定义限流错误处理
async def rate_limit_exceeded_handler(request, exc):
    return JSONResponse(
        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
        content={"error": "请求过于频繁，请稍后再试"}
    )


def create_routes(app: FastAPI) -> None:
    """为应用创建和注册所有路由"""

    # 限流器集成
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, rate_limit_exceeded_handler)

    # 全局 aiohttp.ClientSession (按需创建)
    async def get_http_session() -> aiohttp.ClientSession:
        """获取或创建全局 HTTP session"""
        if not hasattr(app.state, 'http_session') or app.state.http_session is None:
            timeout = aiohttp.ClientTimeout(total=30)
            app.state.http_session = aiohttp.ClientSession(timeout=timeout)
        return app.state.http_session

    # ============================================================================
    # 页面和健康检查
    # ============================================================================

    @app.get("/", response_class=HTMLResponse)
    async def serve_frontend():
        """服务前端页面"""
        # 尝试从缓存读取，如果缓存为空则加载
        if not hasattr(serve_frontend, '_cached_html'):
            try:
                index_path = config.FRONTEND_PATH + "/index.html"
                with open(index_path, "r", encoding="utf-8") as f:
                    serve_frontend._cached_html = f.read()
            except FileNotFoundError:
                raise HTTPException(status_code=404, detail="Frontend file not found")
        return serve_frontend._cached_html

    @app.get("/health")
    async def health_check():
        """健康检查端点"""
        return {"status": "healthy"}

    # ============================================================================
    # 认证端点
    # ============================================================================

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

        if not secrets.compare_digest(token, config.ACCESS_TOKEN):
            logger.warning(f"认证失败：无效token (提供: {token[:8]}...)")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail={"error": "无效的访问令牌"}
            )

        logger.info(f"认证成功：token {token[:8]}...")
        return {"status": "authenticated", "message": "认证成功"}

    # ============================================================================
    # 文件上传端点
    # ============================================================================

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

                # 4. 创建文件信息（仅存储单个文件的PDF字节）
                file_info = {
                    "filename": file.filename,
                    "size": len(pdf_bytes),
                    "pages": get_pdf_page_count(pdf_bytes),
                    "uploaded_at": datetime.utcnow().isoformat(),
                    "pdf_bytes": pdf_bytes
                }

                # 5. 更新会话（不再实时合并，只存储单个文件）
                session.files.append(file_info)
                # 不再设置 session.pdf_bytes，合并在预览/打印时按需进行
                session.last_accessed = datetime.utcnow()

                # 6. 计算总页数（从所有文件计算）
                total_pages = sum(f.get("pages", 0) for f in session.files)

                logger.info(f"文件上传成功: {file.filename}, 当前总页数: {total_pages}")

                # 7. 返回响应
                return {
                    "pages": total_pages,
                    "size": sum(f.get("size", 0) for f in session.files),
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

    # ============================================================================
    # PDF 预览端点
    # ============================================================================

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

        # 动态生成合并PDF（按需合并）
        merged_pdf = get_merged_pdf(session)

        # 检查PDF数据是否为空
        if not merged_pdf:
            logger.warning(f"PDF为空: token {session.token[:8]}...")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={"error": "PDF为空，请先上传文件"}
            )

        # 创建字节流响应
        return StreamingResponse(
            io.BytesIO(merged_pdf),
            media_type="application/pdf",
            headers={
                "Content-Disposition": f'inline; filename="preview.pdf"',
                "Content-Length": str(len(merged_pdf)),
                "Cache-Control": "no-cache"
            }
        )

    # ============================================================================
    # 打印机状态端点
    # ============================================================================

    @app.get("/printer/status")
    async def get_printer_status_endpoint(token: str = Depends(verify_token)):
        """
        查询IPP打印机状态

        返回:
        - 在线: {"status": "online"}
        - 离线: {"status": "offline", "error": "错误详情"}

        需要认证: Authorization头或token查询参数
        """
        return await get_printer_status(token)

    # ============================================================================
    # 打印机能力端点
    # ============================================================================

    @app.get("/printer/capabilities")
    async def get_printer_capabilities_endpoint(token: str = Depends(verify_token)):
        """
        获取IPP打印机能力（支持的打印选项）

        返回:
        - 成功: {
            "media": ["iso-a4", "iso-a3", "letter", ...],
            "sides": ["one-sided", "two-sided-long-edge", ...],
            "color_mode": ["monochrome", "color"],
            "print_quality": ["draft", "normal", "high"],
            "orientation": ["portrait", "landscape"]
          }
        - 离线: {"status": "offline", "error": "错误详情"}

        需要认证: Authorization头或token查询参数
        """
        return await get_printer_capabilities(token)

    # ============================================================================
    # 打印投递端点
    # ============================================================================

    @app.post("/print")
    async def print_document(
        print_request: PrintRequest,
        session: SessionData = Depends(require_session)
    ):
        """
        打印投递端点
        直接通过 HTTP POST 构建原生 IPP 数据包发给打印机，不依赖任何宿主机 CUPS/lp 命令
        """
        logger.info(f"打印请求: token {session.token[:8]}..., copies={print_request.copies}, sides={print_request.sides}, color_mode={print_request.color_mode}, media={print_request.media}, quality={print_request.print_quality}, orientation={print_request.orientation}")

        # 1. 验证参数
        if print_request.copies < 1 or print_request.copies > 99:
            raise HTTPException(status_code=400, detail={"error": "份数必须在1到99之间"})
        if print_request.sides not in ["one-sided", "two-sided"]:
            raise HTTPException(status_code=400, detail={"error": "单双面设置必须是 'one-sided' 或 'two-sided'"})
        if print_request.color_mode not in ["monochrome", "color"]:
            raise HTTPException(status_code=400, detail={"error": "色彩模式必须是 'monochrome' 或 'color'"})
        if print_request.print_quality not in ["draft", "normal", "high"]:
            raise HTTPException(status_code=400, detail={"error": "打印质量必须是 'draft', 'normal' 或 'high'"})
        if print_request.orientation not in ["portrait", "landscape"]:
            raise HTTPException(status_code=400, detail={"error": "打印方向必须是 'portrait' 或 'landscape'"})

        # 2. 按需生成合并PDF（仅在打印时合并）
        pdf_bytes = get_merged_pdf(session)

        # 3. 检查PDF数据
        if not pdf_bytes:
            raise HTTPException(status_code=404, detail={"error": "PDF为空，请先上传文件"})

        # 4. 读取并转换打印机 URI (使用统一配置)
        printer_uri = config.PRINTER_IPP_URL or "ipp://localhost:631/printers/default"

        # aiohttp 走的是原生 TCP 请求，需要将 ipp:// 转换为 http://，ipps:// 转换为 https://
        http_url = printer_uri.replace("ipp://", "http://").replace("ipps://", "https://")

        # 5. 映射 IPP 标准属性关键字
        ipp_sides = "two-sided-long-edge" if print_request.sides == "two-sided" else "one-sided"
        ipp_color = "monochrome" if print_request.color_mode == "monochrome" else "color"
        # 使用配置中的默认值或请求中的值
        ipp_media = print_request.media or config.IPP_DEFAULT_MEDIA
        ipp_quality = print_request.print_quality or config.IPP_DEFAULT_QUALITY
        ipp_orientation = print_request.orientation or config.IPP_DEFAULT_ORIENTATION

        # 6. 构建 IPP 协议 Payload
        ipp_payload = build_ipp_print_request(
            printer_uri=printer_uri,
            pdf_bytes=pdf_bytes,
            copies=print_request.copies,
            sides=ipp_sides,
            color_mode=ipp_color,
            media=ipp_media,
            print_quality=ipp_quality,
            orientation=ipp_orientation
        )

        try:
            # 复用全局 HTTP session (连接池)
            http_session = await get_http_session()
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
                    # IPP 规范中 0x0000~0x0007 均代表各类"成功"状态
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
            # 无论成功与否，销毁当前会话状态
            delete_session(session.token)

        return {"status": "success", "message": "打印任务已提交到打印队列"}

    # ============================================================================
    # 取消会话端点
    # ============================================================================

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

    # 挂载静态文件目录（前端）- 必须放在所有路由之后
    app.mount("/", StaticFiles(directory=config.FRONTEND_PATH, html=True), name="frontend")
