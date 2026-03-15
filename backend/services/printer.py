"""
打印服务模块

提供 IPP 打印机交互功能
"""

import asyncio
import struct
from typing import Dict, Any

from pyipp import IPP
from pyipp.enums import IppOperation

from ..config import config
from ..logging_config import get_logger

logger = get_logger(__name__)


# 默认能力列表（打印机不支持时使用）
DEFAULT_CAPABILITIES = {
    "media": ["iso-a4"],
    "sides": ["one-sided"],
    "color_mode": ["monochrome"],
    "print_quality": ["normal"],
    "orientation": ["portrait"],
    "document_formats": ["application/pdf", "image/jpeg", "image/png"],
    "printer_resolution": [],
    "printer_name": ""
}


def build_ipp_print_request(
    printer_uri: str,
    pdf_bytes: bytes,
    copies: int,
    sides: str,
    color_mode: str,
    media: str = "iso-a4",
    print_quality: str = "normal",
    orientation: str = "portrait"
) -> bytes:
    """
    手动构建标准的 IPP Print-Job 协议二进制 Payload

    参数:
        printer_uri: 打印机 URI
        pdf_bytes: PDF 文档字节数据
        copies: 打印份数
        sides: 单双面设置 (one-sided, two-sided-long-edge, two-sided-short-edge)
        color_mode: 色彩模式 (monochrome, color)
        media: 纸张尺寸 (iso-a4, iso-a3, letter, etc.)
        print_quality: 打印质量 (draft, normal, high)
        orientation: 打印方向 (portrait, landscape)
    """

    # IPP 协议常量定义
    # 参考: RFC 8010 (IPP/1.1) 和 RFC 8011 (IPP Model)

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

    # 构建 IPP 请求
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

    # requesting-user-name = fastapi (使用配置中的用户名)
    user_name = config.IPP_USER_NAME or "fastapi"
    req.extend(b'\x42\x00\x14requesting-user-name')
    req.extend(struct.pack('>H', len(user_name)))
    req.extend(user_name.encode('utf-8'))

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

    # media (keyword) - 纸张尺寸
    req.extend(b'\x44\x00\x05media')
    req.extend(struct.pack('>H', len(media)))
    req.extend(media.encode('utf-8'))

    # print-quality (keyword) - 打印质量
    req.extend(b'\x44\x00\x0dprint-quality')
    req.extend(struct.pack('>H', len(print_quality)))
    req.extend(print_quality.encode('utf-8'))

    # orientation-requested (enum) - 打印方向
    # 3 = portrait, 4 = landscape, 5 = reverse-landscape, 6 = reverse-portrait
    orientation_map = {
        "portrait": 3,
        "landscape": 4,
        "reverse-landscape": 5,
        "reverse-portrait": 6
    }
    orientation_value = orientation_map.get(orientation, 3)
    req.extend(b'\x22\x00\x15orientation-requested\x00\x04')
    req.extend(struct.pack('>I', orientation_value))

    # 3. End of Attributes (0x03)
    req.append(IPP_GROUP_END)

    # 4. Document Content (追加真实的 PDF 二进制数据)
    req.extend(pdf_bytes)

    return bytes(req)


async def get_printer_status(token: str) -> Dict[str, Any]:
    """
    查询IPP打印机状态

    参数:
        token: 认证 token

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


async def get_printer_capabilities(token: str) -> Dict[str, Any]:
    """
    获取IPP打印机能力（支持的打印选项）

    参数:
        token: 认证 token

    返回:
        - 成功: 包含各类能力的字典
        - 离线: {"status": "offline", "error": "错误详情"}
    """
    logger.info(f"获取打印机能力: token {token[:8]}...")

    # 检查打印机URL是否配置
    if not config.PRINTER_IPP_URL:
        logger.error("打印机未配置: PRINTER_IPP_URL 环境变量为空")
        return {"status": "offline", "error": "打印机未配置"}

    try:
        # 使用 pyipp 库获取打印机能力
        CAPABILITY_ATTRIBUTES = [
            "media-supported",
            "sides-supported",
            "color-supported",
            "print-quality-supported",
            "document-format-supported",
            "printer-resolution-supported",
            "printer-name",
        ]

        async with IPP(config.PRINTER_IPP_URL) as ipp:
            # 使用自定义属性列表获取打印机能力
            response_data = await ipp.execute(
                IppOperation.GET_PRINTER_ATTRIBUTES,
                {
                    "operation-attributes-tag": {
                        "requested-attributes": CAPABILITY_ATTRIBUTES,
                    },
                },
            )

            parsed = next(iter(response_data.get("printers", [{}])), {})

            logger.info(f"打印机能力原始数据: {parsed}")

            capabilities = {
                "media": [],
                "sides": [],
                "color_mode": [],
                "print_quality": [],
                "orientation": [],
                "document_formats": [],
                "printer_resolution": [],
                "printer_name": ""
            }

            # 从解析结果中获取属性
            # media-supported
            if "media-supported" in parsed:
                capabilities["media"] = list(parsed["media-supported"])
            else:
                capabilities["media"] = DEFAULT_CAPABILITIES["media"]

            # sides-supported
            if "sides-supported" in parsed:
                capabilities["sides"] = list(parsed["sides-supported"])
            else:
                capabilities["sides"] = DEFAULT_CAPABILITIES["sides"]

            # color-supported (可能是 boolean)
            if "color-supported" in parsed:
                color_val = parsed["color-supported"]
                if isinstance(color_val, bool):
                    capabilities["color_mode"] = ["monochrome", "color"] if color_val else ["monochrome"]
                else:
                    capabilities["color_mode"] = list(color_val)
            else:
                capabilities["color_mode"] = DEFAULT_CAPABILITIES["color_mode"]

            # print-quality-supported
            if "print-quality-supported" in parsed:
                quality_val = parsed["print-quality-supported"]
                # 处理枚举类型或列表
                if hasattr(quality_val, '__iter__') and not isinstance(quality_val, str):
                    # 是枚举列表，使用 .name 获取名称
                    capabilities["print_quality"] = [q.name.lower() if hasattr(q, 'name') else str(q) for q in quality_val]
                else:
                    # 单个枚举值，使用 .name 获取名称
                    if hasattr(quality_val, 'name'):
                        capabilities["print_quality"] = [quality_val.name.lower()]
                    else:
                        capabilities["print_quality"] = [str(quality_val)]
            else:
                capabilities["print_quality"] = DEFAULT_CAPABILITIES["print_quality"]

            # document-format-supported
            if "document-format-supported" in parsed:
                capabilities["document_formats"] = list(parsed["document-format-supported"])
            else:
                capabilities["document_formats"] = DEFAULT_CAPABILITIES["document_formats"]

            # printer-resolution-supported
            if "printer-resolution-supported" in parsed:
                # 转换为字符串列表
                res_list = []
                for res in parsed["printer-resolution-supported"]:
                    if hasattr(res, '__str__'):
                        res_list.append(str(res))
                    else:
                        res_list.append(res)
                capabilities["printer_resolution"] = res_list
            else:
                capabilities["printer_resolution"] = DEFAULT_CAPABILITIES["printer_resolution"]

            # printer-name
            if "printer-name" in parsed:
                capabilities["printer_name"] = parsed["printer-name"]
            else:
                capabilities["printer_name"] = config.PRINTER_NAME or "Unknown Printer"

            # orientation-requested 通常不是打印机属性，使用默认值
            capabilities["orientation"] = DEFAULT_CAPABILITIES["orientation"]

            logger.info(f"成功获取打印机能力: {config.PRINTER_NAME}")
            return capabilities

    except asyncio.TimeoutError:
        logger.warning(f"获取打印机能力超时: {config.PRINTER_NAME} ({config.PRINTER_IPP_URL})")
        return {"status": "offline", "error": "连接打印机超时（5秒）"}
    except Exception as e:
        logger.warning(f"获取打印机能力失败: {config.PRINTER_NAME} ({config.PRINTER_IPP_URL}) - {str(e)}")
        # 失败时返回默认能力
        return DEFAULT_CAPABILITIES
