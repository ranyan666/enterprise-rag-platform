"""接口层依赖：从当前应用实例取配置与仓储，避免误用进程级单例。"""

from __future__ import annotations

from fastapi import Request

from app.core.config import Settings
from app.repositories.documents import DocumentRepository


def get_app_settings(request: Request) -> Settings:
    return request.app.state.settings


def get_repo(request: Request) -> DocumentRepository:
    return request.app.state.documents_repo
