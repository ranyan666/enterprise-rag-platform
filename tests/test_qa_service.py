"""问答服务测试：引用构建、事件流、异常与空结果。"""

from __future__ import annotations

import pytest

from app.providers.chat import ChatError
from app.services.qa import (
    NO_CONTEXT_ANSWER,
    AnswerService,
    build_citations,
    build_messages,
)
from app.services.search import SearchOutcome, SearchResult
from app.services.vector_store import VectorStoreError
from tests.fakes import FakeChatModel


class _StubRetrieval:
    def __init__(self, outcome: SearchOutcome | None = None, error: Exception | None = None) -> None:
        self.outcome = outcome
        self.error = error
        self.queries: list[str] = []

    def search(self, query: str, *, top_k=None, limit=None) -> SearchOutcome:
        self.queries.append(query)
        if self.error is not None:
            raise self.error
        assert self.outcome is not None
        return self.outcome


def _result(
    *,
    parent_text: str = "权限模型包含角色、用户与资源。",
    filename: str = "权限手册.md",
    section: str = "权限模型 > 字段权限",
    page: int | None = 3,
    score: float = 0.03,
) -> SearchResult:
    return SearchResult(
        chunk_id="c1",
        document_id="d1",
        filename=filename,
        parent_id="p1",
        section=section,
        page=page,
        score=score,
        bm25_rank=1,
        dense_rank=1,
        bm25_score=2.5,
        dense_score=0.9,
        child_text="权限模型包含角色、用户与资源。",
        parent_text=parent_text,
    )


def _outcome(results: tuple[SearchResult, ...]) -> SearchOutcome:
    return SearchOutcome(
        results=results,
        bm25_candidates=len(results),
        dense_candidates=len(results),
        fused_candidates=len(results),
        timings={"bm25_ms": 1.0, "dense_ms": 2.0, "fusion_ms": 0.5, "total_ms": 12.3},
    )


def test_build_citations_numbers_headers_and_truncates() -> None:
    long_text = "字" * 2000

    context, citations = build_citations(
        [_result(parent_text=long_text)],
        max_chars=1200,
    )

    assert context.startswith("[1] 来源: 权限手册.md | 章节: 权限模型 > 字段权限 | 页码: 3")
    assert context.endswith("…")
    assert len(context.split("\n", 1)[1]) == 1201
    assert citations[0].index == 1
    assert citations[0].filename == "权限手册.md"
    assert citations[0].section == "权限模型 > 字段权限"
    assert citations[0].page == 3


def test_build_messages_includes_context_and_question() -> None:
    messages = build_messages("怎么配置字段权限？", "[1] 来源: 手册\n内容")

    assert messages[0].role == "system"
    assert "只使用资料中的事实" in messages[0].content
    assert "怎么配置字段权限？" in messages[1].content
    assert "[1] 来源: 手册" in messages[1].content


@pytest.mark.asyncio
async def test_stream_answer_emits_meta_tokens_and_done() -> None:
    chat = FakeChatModel(tokens=["权限", "模型", "[1]"])
    service = AnswerService(
        _StubRetrieval(_outcome((_result(),))),
        chat,
        context_limit=5,
        context_chars=1200,
    )

    events = [event async for event in service.stream_answer("权限模型是什么")]

    assert [event.event for event in events] == ["meta", "token", "token", "token", "done"]
    meta = events[0].data
    assert meta["query"] == "权限模型是什么"
    assert meta["citations"][0]["filename"] == "权限手册.md"
    assert meta["retrieval_ms"] == 12.3

    tokens = "".join(event.data["text"] for event in events if event.event == "token")
    done = events[-1].data
    assert tokens == "权限模型[1]"
    assert done["answer"] == "权限模型[1]"
    assert done["eval_count"] == 3
    assert done["timings"]["ttft_ms"] is not None
    assert done["timings"]["retrieval_ms"] == 12.3
    assert chat.calls, "应调用对话模型"
    prompt = "\n".join(message.content for message in chat.calls[0])
    assert "权限模型是什么" in prompt
    assert "[1]" in prompt


@pytest.mark.asyncio
async def test_stream_answer_without_context_uses_fallback() -> None:
    chat = FakeChatModel()
    service = AnswerService(_StubRetrieval(_outcome(())), chat)

    events = [event async for event in service.stream_answer("完全不相关的问题")]

    assert [event.event for event in events] == ["meta", "token", "done"]
    assert events[0].data["citations"] == []
    assert events[1].data["text"] == NO_CONTEXT_ANSWER
    assert events[2].data["answer"] == NO_CONTEXT_ANSWER
    assert events[2].data["eval_count"] == 0
    assert chat.calls == []


@pytest.mark.asyncio
async def test_stream_answer_reports_chat_error() -> None:
    chat = FakeChatModel(fail_with=ChatError("模型不可用"))
    service = AnswerService(_StubRetrieval(_outcome((_result(),))), chat)

    events = [event async for event in service.stream_answer("权限模型")]

    assert [event.event for event in events] == ["meta", "error"]
    assert "模型不可用" in events[-1].data["message"]


@pytest.mark.asyncio
async def test_stream_answer_reports_retrieval_error() -> None:
    service = AnswerService(
        _StubRetrieval(error=VectorStoreError("milvus down")),
        FakeChatModel(),
    )

    events = [event async for event in service.stream_answer("权限模型")]

    assert [event.event for event in events] == ["error"]
    assert "milvus down" in events[0].data["message"]
