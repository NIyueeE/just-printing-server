"""
认证模块

提供基于 token 的认证功能
"""

import secrets
from typing import Optional

from fastapi import Depends, HTTPException, Query, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from ..config import config
from .session import get_session, create_session
from ..logging_config import get_logger

logger = get_logger(__name__)


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

    Args:
        authorization: Authorization 头信息
        token_query: token 查询参数

    Returns:
        验证通过的token字符串

    Raises:
        HTTPException: 认证失败时抛出 401 异常
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


async def require_session(token: str = Depends(verify_token)):
    """
    依赖函数：验证token并返回会话，如果不存在则返回404

    Args:
        token: 验证通过的token

    Returns:
        会话对象

    Raises:
        HTTPException: 会话不存在时抛出 404 异常
    """
    from .session import SessionData
    session = get_session(token)
    if not session:
        logger.warning(f"会话不存在: {token[:8]}...")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"error": "会话不存在或已过期"}
        )
    return session


async def get_or_create_session(token: str = Depends(verify_token)):
    """
    依赖函数：验证token并返回会话，如果不存在则创建

    Args:
        token: 验证通过的token

    Returns:
        会话对象
    """
    from .session import SessionData
    from datetime import datetime

    session = get_session(token)
    if not session:
        session = create_session(token)
        logger.info(f"为新token创建会话: {token[:8]}...")
    else:
        # 确保会话有token字段（兼容之前创建的会话）
        session.token = token
        session.last_accessed = datetime.utcnow()
    return session
