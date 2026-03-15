"""
会话管理模块

提供基于内存的会话管理功能
"""

import logging
from typing import Dict, Optional
from datetime import datetime
from dataclasses import dataclass, field

from ..logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class SessionData:
    """会话数据类"""
    token: str = ""
    pdf_bytes: bytes = b""
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_accessed: datetime = field(default_factory=datetime.utcnow)
    files: list = field(default_factory=list)


# 全局会话存储
sessions: Dict[str, SessionData] = {}


def create_session(token: str) -> SessionData:
    """
    创建新会话，如果已存在则覆盖

    Args:
        token: 会话令牌

    Returns:
        创建的会话对象
    """
    session = SessionData(token=token)
    sessions[token] = session
    logger.info(f"创建会话: {token[:8]}...")
    return session


def get_session(token: str, max_age_seconds: int = 1800) -> Optional[SessionData]:
    """
    获取会话并更新最后访问时间，如果会话过期则删除

    Args:
        token: 会话令牌
        max_age_seconds: 会话最大存活时间（秒），默认30分钟

    Returns:
        会话对象，如果不存在或已过期则返回 None
    """
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
    """
    删除会话

    Args:
        token: 会话令牌

    Returns:
        是否成功删除
    """
    if token in sessions:
        del sessions[token]
        logger.info(f"删除会话: {token[:8]}...")
        return True
    return False


def cleanup_sessions(max_age_seconds: int = 1800) -> int:
    """
    清理超过指定时间未活动的会话

    Args:
        max_age_seconds: 会话最大存活时间（秒），默认30分钟

    Returns:
        清理的会话数量
    """
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
