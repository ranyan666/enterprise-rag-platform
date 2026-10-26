"""测试用替身：嵌入器与向量库，保证单元测试完全离线。"""

from __future__ import annotations

from collections import defaultdict
from typing import Sequence

from app.services.vector_store import DenseHit, VectorRecord


class FakeEmbedder:
    def __init__(self, dim: int = 8, fail_with: Exception | None = None) -> None:
        self.dim = dim
        self.fail_with = fail_with
        self.batches: list[list[str]] = []

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if self.fail_with is not None:
            raise self.fail_with
        batch = list(texts)
        self.batches.append(batch)
        return [[0.1 * (index + 1)] * self.dim for index, _ in enumerate(batch)]

    def close(self) -> None:
        pass


class FakeVectorStore:
    def __init__(self) -> None:
        self.records: dict[str, dict[str, VectorRecord]] = defaultdict(dict)
        self.deleted: list[str] = []
        self.fail_with: Exception | None = None
        self.fail_search_with: Exception | None = None

    def upsert_chunks(self, records: Sequence[VectorRecord]) -> int:
        if self.fail_with is not None:
            raise self.fail_with
        for record in records:
            self.records[record.document_id][record.id] = record
        return len(records)

    def delete_document(self, doc_id: str) -> None:
        self.deleted.append(doc_id)
        self.records.pop(doc_id, None)

    def count(self, doc_id: str | None = None) -> int:
        if doc_id is None:
            return sum(len(items) for items in self.records.values())
        return len(self.records.get(doc_id, {}))

    def search(self, query_vector: Sequence[float], limit: int) -> list[DenseHit]:
        if self.fail_search_with is not None:
            raise self.fail_search_with
        hits: list[DenseHit] = []
        for items in self.records.values():
            for record in items.values():
                hits.append(
                    DenseHit(
                        id=record.id,
                        document_id=record.document_id,
                        parent_id=record.parent_id,
                        text=record.text,
                        section=record.section,
                        page=record.page,
                        score=0.9,
                    )
                )
        return hits[:limit]

    def close(self) -> None:
        pass
