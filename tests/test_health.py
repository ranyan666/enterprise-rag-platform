"""/health 接口与依赖检查逻辑测试。"""

from __future__ import annotations

import pytest

from app.core.config import Settings
from app.services import health as health_service
from app.services.health import DependencyStatus


def _status(name: str, ok: bool = True, detail: str = "") -> DependencyStatus:
    return DependencyStatus(name=name, ok=ok, latency_ms=1.23, detail=detail)


async def test_health_all_ok(client, monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_check_all(_settings: Settings) -> list[DependencyStatus]:
        return [_status("milvus"), _status("redis"), _status("ollama", detail="模型就绪")]

    monkeypatch.setattr(health_service, "check_all", fake_check_all)

    response = await client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["version"] == "0.1.0"
    assert [item["name"] for item in body["dependencies"]] == ["milvus", "redis", "ollama"]
    assert body["dependencies"][2]["detail"] == "模型就绪"


async def test_health_degraded_when_any_dependency_fails(client, monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_check_all(_settings: Settings) -> list[DependencyStatus]:
        return [_status("milvus", ok=False, detail="ConnectionError"), _status("redis"), _status("ollama")]

    monkeypatch.setattr(health_service, "check_all", fake_check_all)

    response = await client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "degraded"
    assert body["dependencies"][0]["ok"] is False


async def test_check_all_preserves_declared_order(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake(name: str):
        async def _check(_settings: Settings) -> DependencyStatus:
            return _status(name)

        return _check

    monkeypatch.setattr(health_service, "check_milvus", fake("milvus"))
    monkeypatch.setattr(health_service, "check_redis", fake("redis"))
    monkeypatch.setattr(health_service, "check_ollama", fake("ollama"))

    statuses = await health_service.check_all(Settings(_env_file=None))

    assert [item.name for item in statuses] == ["milvus", "redis", "ollama"]


async def test_check_redis_reports_failure_without_raising() -> None:
    settings = Settings(
        _env_file=None,
        redis_url="redis://127.0.0.1:1/0",
        request_timeout_seconds=0.5,
    )

    status = await health_service.check_redis(settings)

    assert status.name == "redis"
    assert status.ok is False
    assert status.latency_ms >= 0
    assert status.detail


def test_describe_truncates_long_errors() -> None:
    detail = health_service._describe(RuntimeError("x" * 1000))

    assert len(detail) <= health_service.DETAIL_MAX_LENGTH
    assert detail.startswith("RuntimeError: ")
