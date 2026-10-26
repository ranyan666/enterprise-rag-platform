"""Ollama 流式对话解析测试（MockTransport，完全离线）。"""

from __future__ import annotations

import json

import httpx
import pytest

from app.providers.chat import ChatError, ChatMessage, OllamaChatModel


def _model(handler) -> OllamaChatModel:
    return OllamaChatModel(
        base_url="http://ollama.local",
        model="qwen2.5:7b-instruct",
        temperature=0.2,
        max_tokens=128,
        timeout=10,
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )


async def _collect(model: OllamaChatModel) -> list:
    return [delta async for delta in model.stream([ChatMessage(role="user", content="你好")])]


async def test_stream_parses_ndjson_deltas() -> None:
    payloads: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payloads.append(json.loads(request.content))
        lines = [
            {"message": {"role": "assistant", "content": "你"}, "done": False},
            {"message": {"role": "assistant", "content": "好"}, "done": False},
            {"message": {"role": "assistant", "content": ""}, "done": True, "eval_count": 2},
        ]
        body = "\n".join(json.dumps(line, ensure_ascii=False) for line in lines) + "\n"
        return httpx.Response(200, content=body.encode("utf-8"))

    deltas = await _collect(_model(handler))

    assert [delta.text for delta in deltas] == ["你", "好", ""]
    assert [delta.done for delta in deltas] == [False, False, True]
    assert deltas[-1].eval_count == 2
    assert payloads[0]["model"] == "qwen2.5:7b-instruct"
    assert payloads[0]["stream"] is True
    assert payloads[0]["messages"] == [{"role": "user", "content": "你好"}]
    assert payloads[0]["options"]["num_predict"] == 128


async def test_stream_raises_on_http_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="boom")

    with pytest.raises(ChatError, match="HTTP 500"):
        await _collect(_model(handler))


async def test_stream_raises_on_error_payload() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b'{"error":"model not found"}\n')

    with pytest.raises(ChatError, match="model not found"):
        await _collect(_model(handler))


async def test_stream_raises_on_invalid_json_line() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"not-json\n")

    with pytest.raises(ChatError, match="无法解析"):
        await _collect(_model(handler))
