"""测试用替身：嵌入器、向量库与对话模型，保证单元测试完全离线。"""

from __future__ import annotations

from collections import defaultdict
from typing import AsyncIterator, Sequence

from app.agentic.nodes import CritiqueResult, RouteDecision
from app.providers.chat import ChatDelta
from app.providers.reranker import RerankError
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


class FakeChatModel:
    """按固定 token 序列产出流式响应，支持注入异常。"""

    def __init__(
        self,
        tokens: Sequence[str] | None = None,
        fail_with: Exception | None = None,
    ) -> None:
        self.tokens = list(
            tokens if tokens is not None else ["根据", "资料", "，", "权限模型", "包含", "RBAC", "[1]"]
        )
        self.fail_with = fail_with
        self.calls: list[list] = []

    async def stream(self, messages: Sequence) -> AsyncIterator[ChatDelta]:
        if self.fail_with is not None:
            raise self.fail_with
        self.calls.append(list(messages))
        for token in self.tokens:
            yield ChatDelta(text=token)
        yield ChatDelta(text="", done=True, eval_count=len(self.tokens))

    async def close(self) -> None:
        pass


class FakeReranker:
    """按给定分数或逆序返回，便于验证重排是否生效。"""

    def __init__(
        self,
        scores: Sequence[float] | None = None,
        fail_with: Exception | None = None,
    ) -> None:
        self.scores = list(scores) if scores is not None else None
        self.fail_with = fail_with
        self.calls: list[tuple[str, list[str]]] = []

    def score(self, query: str, documents: Sequence[str]) -> list[float]:
        if self.fail_with is not None:
            raise RerankError(str(self.fail_with))
        passages = list(documents)
        self.calls.append((query, passages))
        if self.scores is not None:
            return self.scores[: len(passages)]
        return [float(len(passages) - index) for index in range(len(passages))]


class FakeAgentLLM:
    """Agent 流程用的可编排 LLM 替身。"""

    def __init__(
        self,
        *,
        route: str = "vector",
        relevant: Sequence[bool] | None = None,
        rewrite_text: str = "改写后的检索词",
        condense_text: str = "消解指代后的问题",
        tokens: Sequence[str] | None = None,
    ) -> None:
        self.route = route
        self.relevant = list(relevant if relevant is not None else [True])
        self.rewrite_text = rewrite_text
        self.condense_text = condense_text
        self.tokens = list(tokens if tokens is not None else ["答案", "[1]"])
        self.text_prompts: list[str] = []
        self.json_prompts: list[str] = []
        self.stream_prompts: list[str] = []

    async def text(self, prompt: str, *, system: str | None = None) -> str:
        self.text_prompts.append(prompt)
        if "改写为独立完整的问题" in prompt:
            return self.condense_text
        if "重写检索词" in prompt:
            return self.rewrite_text
        return "直接回答"

    async def json(self, prompt: str, schema: type) -> object | None:
        self.json_prompts.append(prompt)
        if schema is RouteDecision:
            return RouteDecision(route=self.route, reason="测试路由")
        if schema is CritiqueResult:
            value = self.relevant.pop(0) if self.relevant else True
            return CritiqueResult(relevant=value, reason="测试自检")
        return None

    async def stream(self, prompt: str, *, system: str | None = None) -> AsyncIterator[str]:
        self.stream_prompts.append(prompt)
        for token in self.tokens:
            yield token

