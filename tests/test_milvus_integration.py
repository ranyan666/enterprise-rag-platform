"""Milvus 真库集成测试（默认跳过）。

手动运行：``pytest -m integration tests/test_milvus_integration.py``
"""

from __future__ import annotations

import uuid

import pytest

from app.core.config import Settings
from app.services.vector_store import MilvusVectorStore, VectorRecord

pytestmark = pytest.mark.integration


def test_milvus_roundtrip() -> None:
    settings = Settings(_env_file=None)
    collection = f"it_chunks_{uuid.uuid4().hex[:8]}"
    store = MilvusVectorStore(uri=settings.milvus_uri, collection=collection, dim=4)

    try:
        store.ensure_collection()
        record = VectorRecord(
            id="chunk-1",
            document_id="doc-1",
            parent_id="parent-1",
            text="集成测试内容",
            section="第一章",
            page=1,
            embedding=[0.1, 0.2, 0.3, 0.4],
        )

        assert store.upsert_chunks([record]) == 1
        assert store.count("doc-1") == 1

        hits = store.search([0.1, 0.2, 0.3, 0.4], 5)
        assert hits and hits[0].id == "chunk-1"
        assert hits[0].document_id == "doc-1"

        store.delete_document("doc-1")
        assert store.count("doc-1") == 0
    finally:
        try:
            store.client.drop_collection(collection_name=collection)
        except Exception:  # noqa: BLE001 - 清理阶段失败不影响断言
            pass
        store.close()
