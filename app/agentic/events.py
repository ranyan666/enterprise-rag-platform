"""Agentic RAG 事件总线：把图节点里的进度/ token 推给 SSE 接口。

用进程内 asyncio.Queue 做桥接，避免把不可序列化的队列放进 LangGraph state，
也避免依赖 LangGraph 的 stream_writer API（版本差异较大）。
"""

from __future__ import annotations

import asyncio
from typing import Any

_QUEUES: dict[str, asyncio.Queue[dict[str, Any]]] = {}
END_EVENT = {"type": "__end__"}


def register(run_id: str) -> asyncio.Queue[dict[str, Any]]:
    queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
    _QUEUES[run_id] = queue
    return queue


def unregister(run_id: str) -> None:
    _QUEUES.pop(run_id, None)


def emit(run_id: str | None, event: dict[str, Any]) -> None:
    if not run_id:
        return
    queue = _QUEUES.get(run_id)
    if queue is not None:
        queue.put_nowait(event)


def emit_end(run_id: str | None) -> None:
    emit(run_id, END_EVENT)
