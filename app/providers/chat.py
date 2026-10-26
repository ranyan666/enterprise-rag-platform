"""对话模型：通过 Ollama ``/api/chat`` 流式生成回答。"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import AsyncIterator, Protocol, Sequence

import httpx

logger = logging.getLogger(__name__)


class ChatError(RuntimeError):
    """对话模型调用失败。"""


@dataclass(frozen=True, slots=True)
class ChatMessage:
    role: str
    content: str


@dataclass(frozen=True, slots=True)
class ChatDelta:
    text: str
    done: bool = False
    eval_count: int | None = None


class ChatModel(Protocol):
    def stream(self, messages: Sequence[ChatMessage]) -> AsyncIterator[ChatDelta]: ...


class OllamaChatModel:
    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        temperature: float = 0.2,
        max_tokens: int = 512,
        timeout: float = 180.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._client = client or httpx.AsyncClient(timeout=httpx.Timeout(timeout))
        self._owns_client = client is None

    async def stream(self, messages: Sequence[ChatMessage]) -> AsyncIterator[ChatDelta]:
        payload = {
            "model": self._model,
            "messages": [{"role": message.role, "content": message.content} for message in messages],
            "stream": True,
            "options": {
                "temperature": self._temperature,
                "num_predict": self._max_tokens,
            },
        }
        try:
            async with self._client.stream(
                "POST",
                f"{self._base_url}/api/chat",
                json=payload,
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                    except ValueError as exc:
                        raise ChatError(f"Ollama 返回了无法解析的数据: {line[:120]}") from exc
                    if data.get("error"):
                        raise ChatError(str(data["error"]))
                    message = data.get("message") or {}
                    text = str(message.get("content") or "")
                    done = bool(data.get("done"))
                    if text or done:
                        yield ChatDelta(
                            text=text,
                            done=done,
                            eval_count=data.get("eval_count") if done else None,
                        )
        except httpx.HTTPStatusError as exc:
            raise ChatError(f"Ollama 对话请求失败: HTTP {exc.response.status_code}") from exc
        except httpx.HTTPError as exc:
            raise ChatError(f"Ollama 对话请求异常: {exc}") from exc

    async def close(self) -> None:
        if self._owns_client:
            await self._client.aclose()
