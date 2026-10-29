"""LangGraph 编排与 Agentic RAG 服务。

图的控制流：
condense → router ─┬─ direct → direct_answer → END
                   ├─ refuse → refuse → END
                   └─ retrieve → critique ─┬─ 相关 → generate → END
                                           ├─ 不相关且 rewrite_count < max → rewrite → retrieve
                                           └─ 达到上限 → generate（带局限提示）→ END
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from typing import Any, AsyncIterator

from langgraph.graph import END, START, StateGraph

from app.agentic.events import emit_end, register, unregister
from app.agentic.memory import HistoryStore
from app.agentic.nodes import AgentNodes
from app.agentic.state import AgentState

logger = logging.getLogger(__name__)


def build_graph(nodes: AgentNodes):
    graph: StateGraph = StateGraph(AgentState)
    graph.add_node("condense", nodes.condense)
    graph.add_node("router", nodes.router)
    graph.add_node("retrieve", nodes.retrieve)
    graph.add_node("critique", nodes.critique)
    graph.add_node("rewrite", nodes.rewrite)
    graph.add_node("generate", nodes.generate)
    graph.add_node("direct_answer", nodes.direct_answer)
    graph.add_node("refuse", nodes.refuse)

    graph.add_edge(START, "condense")
    graph.add_edge("condense", "router")
    graph.add_conditional_edges(
        "router",
        nodes.route_after_router,
        {"retrieve": "retrieve", "direct": "direct_answer", "refuse": "refuse"},
    )
    graph.add_edge("retrieve", "critique")
    graph.add_conditional_edges(
        "critique",
        nodes.route_after_critique,
        {"generate": "generate", "rewrite": "rewrite"},
    )
    graph.add_edge("rewrite", "retrieve")
    graph.add_edge("generate", END)
    graph.add_edge("direct_answer", END)
    graph.add_edge("refuse", END)
    return graph.compile()


class AgenticRagService:
    def __init__(
        self,
        graph,
        history: HistoryStore,
        *,
        max_rewrites: int = 2,
    ) -> None:
        self._graph = graph
        self._history = history
        self._max_rewrites = max_rewrites

    async def stream(
        self,
        question: str,
        *,
        session_id: str | None = None,
        top_k: int | None = None,
        limit: int | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        started = time.perf_counter()
        session_id = session_id or uuid.uuid4().hex
        run_id = uuid.uuid4().hex
        queue = register(run_id)
        history = await self._history.get(session_id)

        state: AgentState = {
            "question": question,
            "search_query": question,
            "history": history,
            "session_id": session_id,
            "run_id": run_id,
            "rewrite_count": 0,
            "top_k": top_k,
            "limit": limit,
        }

        async def runner():
            try:
                return await self._graph.ainvoke(state)
            finally:
                emit_end(run_id)

        task = asyncio.create_task(runner())
        first_token_ms: float | None = None
        try:
            yield {"type": "start", "session_id": session_id}
            while True:
                event = await queue.get()
                if event.get("type") == "__end__":
                    break
                if event.get("type") == "token" and first_token_ms is None:
                    first_token_ms = (time.perf_counter() - started) * 1000
                yield event
            final = await task
        except Exception as exc:  # noqa: BLE001 - 已开始流式输出，只能事件报错
            logger.exception("Agentic RAG 执行失败")
            yield {"type": "error", "message": f"问答执行失败: {exc}"}
            return
        finally:
            unregister(run_id)

        answer = str(final.get("answer") or "")
        try:
            await self._history.append(session_id, question, answer)
        except Exception:  # noqa: BLE001 - 历史写入失败不影响回答
            logger.warning("写入会话历史失败 session=%s", session_id, exc_info=True)

        yield {
            "type": "done",
            "session_id": session_id,
            "answer": answer,
            "route": final.get("route"),
            "route_reason": final.get("route_reason"),
            "rewrite_count": int(final.get("rewrite_count", 0)),
            "citations": final.get("sources", []),
            "limited": bool(final.get("limited", False)),
            "timings": {
                "ttft_ms": round(first_token_ms, 2) if first_token_ms is not None else None,
                "total_ms": round((time.perf_counter() - started) * 1000, 2),
            },
        }
