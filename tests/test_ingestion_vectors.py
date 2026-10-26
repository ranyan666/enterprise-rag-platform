"""向量入库链路测试：上传即索引、失败可恢复、删除清向量。"""

from __future__ import annotations

from httpx import AsyncClient

from app.providers.embedding import EmbeddingError
from app.services.vector_store import VectorStoreError

CONTENT = ("# 第一章 概述\n\n" + "这是平台概述内容，用于验证向量入库链路。" * 20).encode("utf-8")


async def _upload(client: AsyncClient, content: bytes = CONTENT, name: str = "guide.md"):
    return await client.post(
        "/api/v1/documents",
        files={"file": (name, content, "text/markdown")},
    )


async def test_upload_indexes_child_chunks(client, vector_store, embedder) -> None:
    response = await _upload(client)
    doc_id = response.json()["id"]

    detail = (await client.get(f"/api/v1/documents/{doc_id}")).json()

    assert detail["status"] == "ready"
    assert detail["vector_count"] == detail["child_count"] > 0
    assert vector_store.count(doc_id) == detail["child_count"]
    records = list(vector_store.records[doc_id].values())
    assert all(record.document_id == doc_id for record in records)
    assert all(record.parent_id for record in records)
    assert all(record.text for record in records)
    assert embedder.batches, "至少调用一次嵌入"


async def test_embedding_failure_marks_failed(client, embedder, vector_store) -> None:
    embedder.fail_with = EmbeddingError("模型 bge-m3 未下载")

    response = await _upload(client)
    doc_id = response.json()["id"]
    detail = (await client.get(f"/api/v1/documents/{doc_id}")).json()

    assert detail["status"] == "failed"
    assert "bge-m3" in detail["error"]
    assert detail["vector_count"] == 0
    assert vector_store.count(doc_id) == 0


async def test_vector_store_failure_marks_failed(client, vector_store) -> None:
    vector_store.fail_with = VectorStoreError("milvus down")

    response = await _upload(client)
    doc_id = response.json()["id"]
    detail = (await client.get(f"/api/v1/documents/{doc_id}")).json()

    assert detail["status"] == "failed"
    assert "milvus down" in detail["error"]
    assert detail["vector_count"] == 0


async def test_reindex_recovers_from_embedding_failure(client, embedder, vector_store) -> None:
    embedder.fail_with = EmbeddingError("temporary failure")
    failed_id = (await _upload(client)).json()["id"]
    assert (await client.get(f"/api/v1/documents/{failed_id}")).json()["status"] == "failed"

    embedder.fail_with = None
    response = await client.post(f"/api/v1/documents/{failed_id}/reindex")

    assert response.status_code == 202
    detail = (await client.get(f"/api/v1/documents/{failed_id}")).json()
    assert detail["status"] == "ready"
    assert detail["vector_count"] == detail["child_count"] > 0
    assert vector_store.count(failed_id) == detail["vector_count"]
    assert failed_id in vector_store.deleted


async def test_reindex_conflicts_with_processing(client, repo) -> None:
    repo.create_document(
        doc_id="busy-doc",
        filename="busy.txt",
        ext="txt",
        size_bytes=1,
        sha256="e" * 64,
    )
    repo.mark_status("busy-doc", "processing")

    response = await client.post("/api/v1/documents/busy-doc/reindex")

    assert response.status_code == 409


async def test_reindex_missing_returns_404(client) -> None:
    response = await client.post("/api/v1/documents/missing/reindex")

    assert response.status_code == 404


async def test_delete_clears_vectors(client, vector_store) -> None:
    doc_id = (await _upload(client)).json()["id"]
    assert vector_store.count(doc_id) > 0

    response = await client.delete(f"/api/v1/documents/{doc_id}")

    assert response.status_code == 204
    assert vector_store.count(doc_id) == 0
    assert doc_id in vector_store.deleted
