"""混合检索接口测试。"""

from __future__ import annotations

from httpx import AsyncClient

from app.services.vector_store import VectorStoreError

CONTENT = (
    "# 第一章 权限模型\n\n"
    "权限模型包含 RBAC 角色、用户与资源，支持数据权限与字段权限控制。\n\n"
    "# 第二章 计费系统\n\n"
    "计费系统按月生成账单，支持发票与对账流程。"
).encode("utf-8")


async def _upload(client: AsyncClient, content: bytes = CONTENT, name: str = "design.md"):
    return await client.post(
        "/api/v1/documents",
        files={"file": (name, content, "text/markdown")},
    )


async def test_search_returns_fused_results_with_parent_context(client) -> None:
    await _upload(client)

    response = await client.post("/api/v1/search", json={"query": "权限模型"})

    assert response.status_code == 200
    body = response.json()
    assert body["query"] == "权限模型"
    assert body["results"], "应该有召回结果"
    first = body["results"][0]
    assert first["filename"] == "design.md"
    assert first["parent_text"]
    assert first["child_text"]
    assert first["score"] > 0
    assert set(body["timings"]) == {"bm25_ms", "dense_ms", "rerank_ms", "fusion_ms", "total_ms"}
    assert body["candidates"]["fused"] >= 1


async def test_search_returns_empty_without_documents(client) -> None:
    response = await client.post("/api/v1/search", json={"query": "权限模型"})

    assert response.status_code == 200
    body = response.json()
    assert body["results"] == []
    assert body["candidates"] == {"bm25": 0, "dense": 0, "fused": 0}


async def test_search_rejects_blank_query(client) -> None:
    response = await client.post("/api/v1/search", json={"query": "   "})

    assert response.status_code == 422


async def test_search_returns_503_when_vector_store_unavailable(client, vector_store) -> None:
    await _upload(client)
    vector_store.fail_search_with = VectorStoreError("milvus down")

    response = await client.post("/api/v1/search", json={"query": "权限模型"})

    assert response.status_code == 503
    assert "milvus down" in response.json()["detail"]


async def test_search_respects_limit(client) -> None:
    await _upload(client)

    response = await client.post("/api/v1/search", json={"query": "权限模型", "limit": 1})

    assert response.status_code == 200
    assert len(response.json()["results"]) == 1
