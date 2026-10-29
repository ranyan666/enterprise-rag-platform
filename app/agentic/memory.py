"""会话历史存储：Redis 短期记忆（只存最近几轮，TTL 30 分钟）。"""

from __future__ import annotations

import json
from typing import Any, Protocol

from redis.asyncio import Redis


class HistoryStore(Protocol):
    async def get(self, session_id: str) -> list[dict[str, str]]: ...

    async def append(self, session_id: str, question: str, answer: str) -> None: ...


class InMemoryHistoryStore:
    """测试与无 Redis 环境下的替身。"""

    def __init__(self, max_messages: int = 6) -> None:
        self._data: dict[str, list[dict[str, str]]] = {}
        self._max_messages = max_messages

    async def get(self, session_id: str) -> list[dict[str, str]]:
        return list(self._data.get(session_id, []))

    async def append(self, session_id: str, question: str, answer: str) -> None:
        messages = self._data.setdefault(session_id, [])
        messages.extend(
            [
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer},
            ]
        )
        del messages[: max(0, len(messages) - self._max_messages)]


class RedisHistoryStore:
    def __init__(
        self,
        client: Redis,
        *,
        ttl_seconds: int = 1800,
        max_messages: int = 6,
        prefix: str = "rag:history:",
    ) -> None:
        self._client = client
        self._ttl = ttl_seconds
        self._max_messages = max_messages
        self._prefix = prefix

    def _key(self, session_id: str) -> str:
        return f"{self._prefix}{session_id}"

    async def get(self, session_id: str) -> list[dict[str, str]]:
        raw = await self._client.get(self._key(session_id))
        if not raw:
            return []
        try:
            data = json.loads(raw)
        except ValueError:
            return []
        return [item for item in data if isinstance(item, dict)]

    async def append(self, session_id: str, question: str, answer: str) -> None:
        messages = await self.get(session_id)
        messages.extend(
            [
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer},
            ]
        )
        messages = messages[-self._max_messages :]
        await self._client.setex(
            self._key(session_id),
            self._ttl,
            json.dumps(messages, ensure_ascii=False),
        )


def to_serializable(value: Any) -> Any:
    """保证进入 state 的内容可序列化（避免 Redis/图状态报错）。"""

    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, list):
        return [to_serializable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): to_serializable(item) for key, item in value.items()}
    return str(value)
