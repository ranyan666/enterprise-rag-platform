"""向量库：只索引子块，父块保留在 SQLite 供检索回填。

Milvus collection 结构与 SQLite 的 chunks 保持一致（id 相同），
因此子块 ID 天然幂等，重复处理不会产生重复向量。
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Any, Protocol, Sequence

from pymilvus import DataType, MilvusClient

logger = logging.getLogger(__name__)

PAGE_UNKNOWN = -1
MAX_TEXT_LENGTH = 8192
MAX_SECTION_LENGTH = 1024
ID_MAX_LENGTH = 64
INDEX_TYPE = "HNSW"
METRIC_TYPE = "COSINE"
INDEX_PARAMS = {"M": 16, "efConstruction": 200}


class VectorStoreError(RuntimeError):
    """向量库操作失败。"""


@dataclass(frozen=True, slots=True)
class VectorRecord:
    id: str
    document_id: str
    parent_id: str
    text: str
    section: str | None
    page: int | None
    embedding: list[float]

    def to_milvus(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "document_id": self.document_id,
            "parent_id": self.parent_id,
            "text": self.text[:MAX_TEXT_LENGTH],
            "section": (self.section or "")[:MAX_SECTION_LENGTH],
            "page": PAGE_UNKNOWN if self.page is None else int(self.page),
            "embedding": self.embedding,
        }


@dataclass(frozen=True, slots=True)
class DenseHit:
    id: str
    document_id: str
    parent_id: str
    text: str
    section: str | None
    page: int | None
    score: float


class VectorStore(Protocol):
    def upsert_chunks(self, records: Sequence[VectorRecord]) -> int: ...

    def delete_document(self, doc_id: str) -> None: ...

    def count(self, doc_id: str | None = None) -> int: ...

    def search(self, query_vector: Sequence[float], limit: int) -> list[DenseHit]: ...


class MilvusVectorStore:
    def __init__(
        self,
        *,
        uri: str,
        collection: str,
        dim: int,
        client: MilvusClient | None = None,
    ) -> None:
        self._uri = uri
        self._collection = collection
        self._dim = dim
        self._client = client
        self._lock = threading.Lock()
        self._ready = False

    @property
    def client(self) -> MilvusClient:
        if self._client is None:
            self._client = MilvusClient(uri=self._uri)
        return self._client

    def ensure_collection(self) -> None:
        if self._ready:
            return
        with self._lock:
            if self._ready:
                return
            client = self.client
            if client.has_collection(self._collection):
                self._validate_dimension(client)
            else:
                self._create_collection(client)
            client.load_collection(self._collection)
            self._ready = True

    def _create_collection(self, client: MilvusClient) -> None:
        schema = client.create_schema(auto_id=False, enable_dynamic_field=False)
        schema.add_field("id", DataType.VARCHAR, is_primary=True, max_length=ID_MAX_LENGTH)
        schema.add_field("document_id", DataType.VARCHAR, max_length=ID_MAX_LENGTH)
        schema.add_field("parent_id", DataType.VARCHAR, max_length=ID_MAX_LENGTH)
        schema.add_field("text", DataType.VARCHAR, max_length=MAX_TEXT_LENGTH)
        schema.add_field("section", DataType.VARCHAR, max_length=MAX_SECTION_LENGTH)
        schema.add_field("page", DataType.INT64)
        schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=self._dim)

        index_params = client.prepare_index_params()
        index_params.add_index(
            field_name="embedding",
            index_type=INDEX_TYPE,
            metric_type=METRIC_TYPE,
            params=dict(INDEX_PARAMS),
        )
        client.create_collection(self._collection, schema=schema, index_params=index_params)
        logger.info("已创建 Milvus collection %s（dim=%d）", self._collection, self._dim)

    def _validate_dimension(self, client: MilvusClient) -> None:
        description = client.describe_collection(self._collection)
        for field in description.get("fields", []):
            if field.get("name") != "embedding":
                continue
            params = field.get("params") or {}
            actual = int(params.get("dim", field.get("dim", 0)) or 0)
            if actual != self._dim:
                raise VectorStoreError(
                    f"collection {self._collection} 的向量维度为 {actual}，"
                    f"与配置 embedding_dim={self._dim} 不一致"
                )
            return
        raise VectorStoreError(f"collection {self._collection} 缺少 embedding 字段")

    def upsert_chunks(self, records: Sequence[VectorRecord]) -> int:
        if not records:
            return 0
        self.ensure_collection()
        try:
            self.client.upsert(
                collection_name=self._collection,
                data=[record.to_milvus() for record in records],
            )
        except Exception as exc:  # noqa: BLE001 - 统一转领域异常
            raise VectorStoreError(f"写入 Milvus 失败: {exc}") from exc
        return len(records)

    def delete_document(self, doc_id: str) -> None:
        if not self._ready and not self.client.has_collection(self._collection):
            return
        self.ensure_collection()
        try:
            self.client.delete(
                collection_name=self._collection,
                filter=f'document_id == "{doc_id}"',
            )
        except Exception as exc:  # noqa: BLE001
            raise VectorStoreError(f"删除 Milvus 向量失败: {exc}") from exc

    def count(self, doc_id: str | None = None) -> int:
        if not self._ready and not self.client.has_collection(self._collection):
            return 0
        self.ensure_collection()
        expression = f'document_id == "{doc_id}"' if doc_id else 'document_id != ""'
        try:
            rows = self.client.query(
                collection_name=self._collection,
                filter=expression,
                output_fields=["count(*)"],
                consistency_level="Strong",
            )
        except Exception as exc:  # noqa: BLE001
            raise VectorStoreError(f"统计 Milvus 向量失败: {exc}") from exc
        return int(rows[0]["count(*)"]) if rows else 0

    def search(self, query_vector: Sequence[float], limit: int) -> list[DenseHit]:
        if limit <= 0:
            return []
        if not self._ready and not self.client.has_collection(self._collection):
            return []
        self.ensure_collection()
        try:
            raw_results = self.client.search(
                collection_name=self._collection,
                data=[list(query_vector)],
                anns_field="embedding",
                limit=limit,
                output_fields=["document_id", "parent_id", "text", "section", "page"],
                search_params={"metric_type": METRIC_TYPE, "params": {"ef": max(64, limit)}},
            )
        except Exception as exc:  # noqa: BLE001
            raise VectorStoreError(f"Milvus 向量检索失败: {exc}") from exc

        hits: list[DenseHit] = []
        for item in raw_results[0] if raw_results else []:
            entity = item.get("entity") or {}
            page = entity.get("page")
            hits.append(
                DenseHit(
                    id=str(item.get("id", "")),
                    document_id=str(entity.get("document_id", "")),
                    parent_id=str(entity.get("parent_id", "")),
                    text=str(entity.get("text", "")),
                    section=entity.get("section") or None,
                    page=None if page in (None, PAGE_UNKNOWN) else int(page),
                    score=float(item.get("distance", 0.0)),
                )
            )
        return hits

    def close(self) -> None:
        client, self._client = self._client, None
        self._ready = False
        if client is not None:
            try:
                client.close()
            except Exception:  # noqa: BLE001 - 关闭阶段的异常不影响退出
                logger.warning("关闭 Milvus 客户端失败", exc_info=True)
