"""
日志配置模块 - 统一的日志配置
"""

import logging
from .config import config


def setup_logging() -> None:
    """
    配置统一的日志格式和级别
    输出到 stdout 以支持容器日志 (docker compose log / podman compose log)
    """
    # 创建一个 handler 输出到 stdout
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s")
    )

    # 获取根 logger 并配置
    root_logger = logging.getLogger()
    root_logger.setLevel(config.LOG_LEVEL)
    root_logger.handlers.clear()  # 清除默认 handler
    root_logger.addHandler(handler)

    # 设置 uvicorn logger 级别
    logging.getLogger("uvicorn").setLevel(config.LOG_LEVEL)
    logging.getLogger("uvicorn.access").setLevel(config.LOG_LEVEL)


def get_logger(name: str) -> logging.Logger:
    """
    获取指定名称的logger

    Args:
        name: logger 名称

    Returns:
        配置好的 logger 实例
    """
    return logging.getLogger(name)
