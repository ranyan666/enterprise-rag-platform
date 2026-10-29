"""LangChain 对话提供者：使用 ChatOllama 做流式生成。

只在这一层引入 LangChain：提示词模板与模型客户端由 LangChain 管理，
检索、父子分块与 Milvus 写入仍保留自研实现，便于精确控制字段与元数据。
"""

from __future__ import annotations

from typing import Any, AsyncIterator, Sequence

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

from app.providers.chat import ChatDelta, ChatError, ChatMessage


def _to_lc_messages(messages: Sequence[ChatMessage]) -> list[Any]:
    converted: list[Any] = []
    for message in messages:
        if message.role == "system":
            converted.append(SystemMessage(content=message.content))
        elif message.role == "assistant":
            converted.append(AIMessage(content=message.content))
        else:
            converted.append(HumanMessage(content=message.content))
    return converted


def _text_of(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict) and part.get("type") == "text":
                parts.append(str(part.get("text", "")))
        return "".join(parts)
    return ""


class LangChainChatModel:
    """把 LangChain ChatModel 适配成项目内的 ``ChatModel`` 协议。"""

    def __init__(self, model: ChatOllama) -> None:
        self._model = model

    async def stream(self, messages: Sequence[ChatMessage]) -> AsyncIterator[ChatDelta]:
        eval_count: int | None = None
        try:
            async for chunk in self._model.astream(_to_lc_messages(messages)):
                metadata = getattr(chunk, "response_metadata", {}) or {}
                if metadata.get("eval_count"):
                    eval_count = int(metadata["eval_count"])
                text = _text_of(getattr(chunk, "content", ""))
                if text:
                    yield ChatDelta(text=text)
        except ChatError:
            raise
        except Exception as exc:  # noqa: BLE001 - 统一转领域异常
            raise ChatError(f"LangChain 对话流失败: {exc}") from exc
        yield ChatDelta(text="", done=True, eval_count=eval_count)

    async def close(self) -> None:
        return None
