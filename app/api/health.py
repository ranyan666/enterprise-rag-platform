"""健康检查接口：``GET /health``。"""

from __future__ import annotations

from dataclasses import asdict

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from app.api.deps import get_app_settings
from app.core.config import Settings
from app.services import health as health_service

router = APIRouter(tags=["health"])


class DependencyStatusOut(BaseModel):
    name: str
    ok: bool
    latency_ms: float
    detail: str = ""


class HealthResponse(BaseModel):
    status: str = Field(description="ok=全部依赖正常；degraded=至少一个依赖异常")
    version: str
    dependencies: list[DependencyStatusOut]


@router.get("/health", response_model=HealthResponse, summary="依赖健康检查")
async def health(settings: Settings = Depends(get_app_settings)) -> HealthResponse:
    statuses = await health_service.check_all(settings)
    return HealthResponse(
        status="ok" if all(item.ok for item in statuses) else "degraded",
        version=settings.app_version,
        dependencies=[DependencyStatusOut(**asdict(item)) for item in statuses],
    )
