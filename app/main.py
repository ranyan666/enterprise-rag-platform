"""FastAPI 应用入口。"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.documents import router as documents_router
from app.api.health import router as health_router
from app.core.config import Settings, get_settings
from app.core.logging import configure_logging
from app.repositories.documents import DocumentRepository
from app.services.documents import initialize_storage

logger = logging.getLogger(__name__)


def create_app(settings: Settings | None = None) -> FastAPI:
    """按给定配置构建应用；测试可传入独立 Settings。"""

    settings = settings or get_settings()
    configure_logging(settings.log_level)
    repository = DocumentRepository(settings.sqlite_path)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        initialize_storage(settings, repository)
        logger.info("%s v%s 启动", settings.app_name, settings.app_version)
        try:
            yield
        finally:
            logger.info("应用已关闭")

    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        lifespan=lifespan,
    )
    app.state.settings = settings
    app.state.documents_repo = repository
    app.include_router(health_router)
    app.include_router(documents_router)

    @app.get("/", tags=["meta"], summary="服务信息")
    async def root() -> dict[str, str]:
        return {
            "name": settings.app_name,
            "version": settings.app_version,
            "docs": "/docs",
            "health": "/health",
            "documents": "/api/v1/documents",
        }

    return app


app = create_app()
