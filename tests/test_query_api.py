"""SSE 问答接口测试。"""

from __future__ import annotations

import json

from httpx import AsyncClient

from app.providers.chat import ChatError
from app.services.qa import NO_CONTEXT_ANSWER

CONTENT = (
    "# 第一章 权限模型\n\n"
    "权限模型包含 RBAC 角色、用户与资源，支持数据权限与字段权限控制。\n\n"
    "# 第二章 计费系统\n\n"
    "计费系统按月生成账单，支持发票与对账流程。"
).encode("utf-8")


async def _read_events(response) -> list[tuple[str, dict]]:
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


async def _upload(client: AsyncClient, content: bytes = CONTENT, name: str = "design.md"):
    return await client.post(
        "/api/v1/documents",
        files={"file": (name, content, "text/markdown")},
    )


async def test_query_streams_meta_tokens_and_done(client) -> None:
    await _upload(client)

    async with client.stream("POST", "/api/v1/query", json={"question": "权限模型包含什么"}) as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        events = await _read_events(response)

    names = [name for name, _ in events]
    assert names[0] == "meta"
    assert names[-1] == "done"
    assert "token" in names

    meta = events[0][1]
    assert meta["citations"][0]["filename"] == "design.md"
    assert meta["retrieval_ms"] >= 0

    tokens = "".join(data["text"] for name, data in events if name == "token")
    done = events[-1][1]
    assert tokens
    assert done["answer"] == tokens
    assert done["timings"]["ttft_ms"] is not None
    assert done["eval_count"] > 0


async def test_query_returns_422_for_blank_question(client) -> None:
    response = await client.post("/api/v1/query", json={"question": "   "})

    assert response.status_code == 422


async def test_query_without_documents_returns_fallback(client) -> None:
    async with client.stream("POST", "/api/v1/query", json={"question": "权限模型"}) as response:
        events = await _read_events(response)

    assert [name for name, _ in events] == ["meta", "token", "done"]
    assert events[0][1]["citations"] == []
    assert events[2][1]["answer"] == NO_CONTEXT_ANSWER


async def test_query_reports_chat_failure(client, chat_model) -> None:
    await _upload(client)
    chat_model.fail_with = ChatError("模型不可用")

    async with client.stream("POST", "/api/v1/query", json={"question": "权限模型"}) as response:
        events = await _read_events(response)

    assert [name for name, _ in events] == ["meta", "error"]
    assert "模型不可用" in events[-1][1]["message"]
