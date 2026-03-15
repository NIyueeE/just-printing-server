"""
文件处理模块

提供文件验证、转换和PDF处理功能
"""

import os
import io
import asyncio
import tempfile
import mimetypes
from typing import Tuple

from fastapi import UploadFile

from PIL import Image
import img2pdf
import PyPDF2

from ..config import config
from ..logging_config import get_logger

logger = get_logger(__name__)


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
    """
    验证文件大小、类型和安全性

    Args:
        file: 上传的文件对象

    Returns:
        (是否有效, 错误信息) 元组
    """
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
    """
    将上传文件转换为PDF字节流

    Args:
        file: 上传的文件对象

    Returns:
        PDF 字节数据

    Raises:
        ValueError: 转换失败时抛出
    """
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


def merge_pdfs(pdf_list: list) -> bytes:
    """
    合并多个PDF字节流为一个PDF

    Args:
        pdf_list: PDF 字节数据列表

    Returns:
        合并后的 PDF 字节数据
    """
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


def get_merged_pdf(session) -> bytes:
    """
    从会话中的单个文件动态生成合并PDF

    Args:
        session: 会话对象

    Returns:
        合并后的 PDF 字节数据
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
    """
    计算PDF的页数

    Args:
        pdf_bytes: PDF 字节数据

    Returns:
        页数
    """
    if not pdf_bytes:
        return 0
    try:
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_bytes))
        return len(pdf_reader.pages)
    except Exception:
        return 0
