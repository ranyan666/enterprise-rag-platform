"""应用元信息接口测试。"""

from __future__ import annotations


async def test_root_returns_service_metadata(client) -> None:
    response = await client.get("/")

    assert response.status_code == 200
    body = response.json()
    assert body["name"] == "企业级智能文档检索与问答平台"
    assert body["version"] == "0.1.0"
    assert body["health"] == "/health"


async def test_openapi_exposes_health(client) -> None:
    response = await client.get("/openapi.json")

    assert response.status_code == 200
    assert "/health" in response.json()["paths"]
