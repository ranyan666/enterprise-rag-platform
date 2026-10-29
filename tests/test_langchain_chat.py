"""LangChain 对话提供者测试（注入 fake ChatModel，离线）。"""

from __future__ import annotations

import pytest

from app.providers.chat import ChatError, ChatMessage
from app.providers.langchain_chat import LangChainChatModel


class _Chunk:
    def __init__(self, content: str | list, metadata: dict | None = None) -> None:
        self.content = content
        self.response_metadata = metadata or {}


class _FakeChatModel:
    def __init__(self, chunks: list[_Chunk], error: Exception | None = None) -> None:
        self.chunks = chunks
        self.error = error
        self.messages: list = []

    async def astream(self, messages):
        self.messages = list(messages)
        if self.error is not None:
            raise self.error
        for chunk in self.chunks:
            yield chunk


async def test_streams_text_chunks_and_eval_count() -> None:
    model = _FakeChatModel([_Chunk("你"), _Chunk("好", {"eval_count": 2})])
    provider = LangChainChatModel(model)  # type: ignore[arg-type]

    deltas = [
        delta
        async for delta in provider.stream(
            [ChatMessage(role="system", content="sys"), ChatMessage(role="user", content="hi")]
        )
    ]

    assert [delta.text for delta in deltas] == ["你", "好", ""]
    assert deltas[-1].done is True
    assert deltas[-1].eval_count == 2
    assert model.messages[0].content == "sys"
    assert model.messages[1].content == "hi"


async def test_supports_structured_content_parts() -> None:
    model = _FakeChatModel([_Chunk([{"type": "text", "text": "结构化"}])])
    provider = LangChainChatModel(model)  # type: ignore[arg-type]

    deltas = [delta async for delta in provider.stream([ChatMessage(role="user", content="hi")])]

    assert deltas[0].text == "结构化"


async def test_wraps_model_errors_as_chat_error() -> None:
    model = _FakeChatModel([], error=RuntimeError("connection reset"))
    provider = LangChainChatModel(model)  # type: ignore[arg-type]

    with pytest.raises(ChatError, match="connection reset"):
        [delta async for delta in provider.stream([ChatMessage(role="user", content="hi")])]
