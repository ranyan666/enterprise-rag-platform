"""Agentic RAG 状态定义：贯穿 LangGraph 的 TypedDict。"""

from __future__ import annotations

from typing import Any, TypedDict


class AgentState(TypedDict, total=False):
    question: str
    search_query: str
    history: list[dict[str, str]]
    session_id: str
    run_id: str

    route: str
    route_reason: str
    documents: list[dict[str, Any]]
    critique_relevant: bool
    critique_reason: str
    rewrite_count: int
    top_k: int | None
    limit: int | None

    answer: str
    sources: list[dict[str, Any]]
    limited: bool
