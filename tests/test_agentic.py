"""Agentic RAG 测试：路由、反思循环、多轮消解、SSE 与会话记忆。"""

from __future__ import annotations

import json

import pytest

from app.agentic.graph import AgenticRagService, build_graph
from app.agentic.memory import InMemoryHistoryStore, RedisHistoryStore
from app.agentic.nodes import REFUSE_ANSWER, AgentNodes
from app.services.search import SearchOutcome, SearchResult
from tests.fakes import FakeAgentLLM


class _StubRetrieval:
    def __init__(self, results: list[SearchResult]) -> None:
        self.results = results
        self.calls: list[str] = []

    def search(self, query: str, *, top_k=None, limit=None, rerank=None) -> SearchOutcome:
        self.calls.append(query)
        return SearchOutcome(
            results=tuple(self.results),
            bm25_candidates=len(self.results),
            dense_candidates=len(self.results),
            fused_candidates=len(self.results),
            timings={"total_ms": 1.0},
        )


class _FakeRedis:
    def __init__(self) -> None:
        self.data: dict[str, str] = {}
        self.ttl: int | None = None

    async def get(self, key: str):
        return self.data.get(key)

    async def setex(self, key: str, ttl: int, value: str) -> None:
        self.ttl = ttl
        self.data[key] = value


def _result() -> SearchResult:
    return SearchResult(
        chunk_id="c1",
        document_id="d1",
        filename="权限模型.md",
        parent_id="p1",
        section="权限模型 > 字段权限",
        page=3,
        score=0.03,
        bm25_rank=1,
        dense_rank=1,
        bm25_score=2.0,
        dense_score=0.9,
        child_text="字段权限控制敏感字段可见性。",
        parent_text="字段权限控制敏感字段可见性，手机号默认脱敏。",
    )


def _state(question: str = "权限模型如何控制字段权限", history=None) -> dict:
    return {
        "question": question,
        "search_query": question,
        "history": history or [],
        "run_id": None,
        "session_id": "s1",
        "rewrite_count": 0,
    }


def _graph(llm, retrieval, *, max_rewrites: int = 2):
    nodes = AgentNodes(
        retrieval,
        llm,
        top_k=5,
        result_limit=3,
        context_limit=3,
        context_chars=500,
        max_rewrites=max_rewrites,
    )
    return build_graph(nodes)


async def test_direct_route_skips_retrieval() -> None:
    llm = FakeAgentLLM(route="direct", tokens=["你好"])
    retrieval = _StubRetrieval([_result()])
    result = await _graph(llm, retrieval).ainvoke(_state("你好"))

    assert result["route"] == "direct"
    assert result["answer"] == "你好"
    assert retrieval.calls == []


async def test_refuse_route_returns_refusal() -> None:
    llm = FakeAgentLLM(route="refuse")
    retrieval = _StubRetrieval([])
    result = await _graph(llm, retrieval).ainvoke(_state("我的工资是多少"))

    assert result["route"] == "refuse"
    assert result["answer"] == REFUSE_ANSWER
    assert retrieval.calls == []


async def test_vector_route_generates_with_sources() -> None:
    llm = FakeAgentLLM(route="vector", relevant=[True], tokens=["答案", "[1]"])
    retrieval = _StubRetrieval([_result()])

    result = await _graph(llm, retrieval).ainvoke(_state())

    assert retrieval.calls == ["权限模型如何控制字段权限"]
    assert result["answer"] == "答案[1]"
    assert result["sources"][0]["filename"] == "权限模型.md"
    assert result["sources"][0]["section"] == "权限模型 > 字段权限"


async def test_rewrite_loop_retries_once_then_generates() -> None:
    llm = FakeAgentLLM(route="vector", relevant=[False, True], rewrite_text="改写查询")
    retrieval = _StubRetrieval([_result()])

    result = await _graph(llm, retrieval).ainvoke(_state())

    assert retrieval.calls == ["权限模型如何控制字段权限", "改写查询"]
    assert result["rewrite_count"] == 1
    assert result["answer"]


async def test_rewrite_stops_at_max_and_marks_limited() -> None:
    llm = FakeAgentLLM(route="vector", relevant=[False, False, False])
    retrieval = _StubRetrieval([_result()])

    result = await _graph(llm, retrieval).ainvoke(_state())

    assert len(retrieval.calls) == 3  # 首次 + 2 次改写
    assert result["rewrite_count"] == 2
    assert result["limited"] is True
    assert result["answer"]


async def test_condense_uses_history() -> None:
    llm = FakeAgentLLM(route="vector", condense_text="Milvus 的端口是多少")
    retrieval = _StubRetrieval([_result()])
    history = [{"role": "user", "content": "Milvus 是什么"}, {"role": "assistant", "content": "向量数据库"}]

    await _graph(llm, retrieval).ainvoke(_state("它的端口是多少", history))

    assert retrieval.calls == ["Milvus 的端口是多少"]


async def test_condense_without_history_keeps_question() -> None:
    llm = FakeAgentLLM(route="vector")
    retrieval = _StubRetrieval([_result()])

    await _graph(llm, retrieval).ainvoke(_state("权限模型如何控制字段权限"))

    assert retrieval.calls == ["权限模型如何控制字段权限"]
    assert llm.text_prompts == []


async def test_service_streams_events_and_saves_history() -> None:
    llm = FakeAgentLLM(route="vector", relevant=[True], tokens=["答案", "[1]"])
    retrieval = _StubRetrieval([_result()])
    store = InMemoryHistoryStore()
    service = AgenticRagService(_graph(llm, retrieval), store)

    events = [event async for event in service.stream("权限模型如何控制字段权限")]

    names = [event["type"] for event in events]
    assert names[0] == "start"
    assert "route" in names and "retrieval" in names and "critique" in names
    assert names.count("token") == 2
    assert names[-1] == "done"
    done = events[-1]
    assert done["answer"] == "答案[1]"
    assert done["citations"][0]["filename"] == "权限模型.md"
    assert done["timings"]["ttft_ms"] is not None
    history = await store.get(done["session_id"])
    assert [item["content"] for item in history] == ["权限模型如何控制字段权限", "答案[1]"]


async def test_memory_store_trims_messages() -> None:
    store = InMemoryHistoryStore(max_messages=4)

    for index in range(3):
        await store.append("s1", f"q{index}", f"a{index}")

    messages = await store.get("s1")
    assert len(messages) == 4
    assert messages[0]["content"] == "q1"


async def test_redis_history_store_roundtrip() -> None:
    client = _FakeRedis()
    store = RedisHistoryStore(client, ttl_seconds=600, max_messages=4)

    await store.append("s1", "问题", "答案")

    assert client.ttl == 600
    history = await store.get("s1")
    assert history == [
        {"role": "user", "content": "问题"},
        {"role": "assistant", "content": "答案"},
    ]


async def _read_events(response):
    events: list[tuple[str, dict]] = []
    event_name: str | None = None
    data_lines: list[str] = []

    def flush() -> None:
        nonlocal event_name, data_lines
        if event_name is not None:
            events.append((event_name, json.loads("\n".join(data_lines))))
        event_name = None
        data_lines = []

    async for line in response.aiter_lines():
        if line == "":
            flush()
        elif line.startswith("event:"):
            event_name = line[len("event:") :].strip()
        elif line.startswith("data:"):
            data_lines.append(line[len("data:") :].strip())
    flush()
    return events


async def test_chat_api_streams_agent_events(client, app) -> None:
    llm = FakeAgentLLM(route="vector", relevant=[True], tokens=["答案", "[1]"])
    retrieval = _StubRetrieval([_result()])
    app.state.agentic_service = AgenticRagService(
        _graph(llm, retrieval),
        InMemoryHistoryStore(),
    )

    async with client.stream("POST", "/api/v1/chat", json={"question": "权限模型如何控制字段权限"}) as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        events = await _read_events(response)

    names = [name for name, _ in events]
    assert names[0] == "start"
    assert names[-1] == "done"
    assert "token" in names
    assert events[-1][1]["answer"] == "答案[1]"


async def test_chat_api_rejects_blank_question(client) -> None:
    response = await client.post("/api/v1/chat", json={"question": "   "})

    assert response.status_code == 422
