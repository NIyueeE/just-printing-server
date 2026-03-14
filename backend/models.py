"""
Pydantic 数据模型
"""

from pydantic import BaseModel


class AuthRequest(BaseModel):
    """认证请求模型"""
    token: str


class PrintRequest(BaseModel):
    """打印请求模型"""
    copies: int = 1
    sides: str = "one-sided"  # "one-sided" or "two-sided"
    color_mode: str = "monochrome"  # "monochrome" or "color"
