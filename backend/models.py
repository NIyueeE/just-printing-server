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
    media: str = "iso-a4"  # 纸张尺寸
    print_quality: str = "normal"  # 打印质量: "draft", "normal", "high"
    orientation: str = "portrait"  # 打印方向: "portrait", "landscape"
    resolution: str = ""  # 打印分辨率
