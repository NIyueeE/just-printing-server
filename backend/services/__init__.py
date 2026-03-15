"""
服务模块 - 导出所有服务

按功能领域组织：
- session: 会话管理
- auth: 认证依赖
- file_handler: 文件处理
- printer: 打印服务
"""

from .session import (
    SessionData,
    sessions,
    create_session,
    get_session,
    delete_session,
    cleanup_sessions,
)

from .auth import (
    verify_token,
    require_session,
    get_or_create_session,
)

from .file_handler import (
    validate_file,
    convert_to_pdf,
    merge_pdfs,
    get_merged_pdf,
    get_pdf_page_count,
    conversion_semaphore,
    ALLOWED_EXTENSIONS,
    ALLOWED_MIME_TYPES,
)

from .printer import (
    build_ipp_print_request,
    get_printer_status,
    get_printer_capabilities,
    DEFAULT_CAPABILITIES,
)

# 重新导出 SessionData 作为主要类型
__all__ = [
    # Session
    "SessionData",
    "sessions",
    "create_session",
    "get_session",
    "delete_session",
    "cleanup_sessions",
    # Auth
    "verify_token",
    "require_session",
    "get_or_create_session",
    # File Handler
    "validate_file",
    "convert_to_pdf",
    "merge_pdfs",
    "get_merged_pdf",
    "get_pdf_page_count",
    "conversion_semaphore",
    "ALLOWED_EXTENSIONS",
    "ALLOWED_MIME_TYPES",
    # Printer
    "build_ipp_print_request",
    "get_printer_status",
    "get_printer_capabilities",
    "DEFAULT_CAPABILITIES",
]
