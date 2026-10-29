"""Agentic RAG 的各个节点：改写、路由、检索、自检、重写、生成。"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import asdict
from typing import Any

from pydantic import BaseModel, Field

from app.agentic.events import emit
from app.agentic.llm import AgentLLM
from app.agentic.state import AgentState
from app.services.search import RetrievalService

logger = logging.getLogger(__name__)

NO_DOCS_ANSWER = "抱歉，知识库中没有检索到与该问题相关的内容。请换一种问法，或先上传相关资料。"
REFUSE_ANSWER = "抱歉，这个问题超出当前知识库的范围，我无法基于已有资料回答。"


class RouteDecision(BaseModel):
    route: str = Field(pattern="^(vector|direct|refuse)$")
    reason: str = ""


class CritiqueResult(BaseModel):
    relevant: bool
    reason: str = ""


def _doc_text(document: dict[str, Any], limit: int) -> str:
    text = document.get("parent_text") or document.get("child_text") or ""
    return text[:limit]


class AgentNodes:
    def __init__(
        self,
        retrieval: RetrievalService,
        llm: AgentLLM,
        *,
        top_k: int = 20,
        result_limit: int = 5,
        context_limit: int = 5,
        context_chars: int = 1200,
        max_rewrites: int = 2,
    ) -> None:
        self._retrieval = retrieval
        self._llm = llm
        self._top_k = top_k
        self._result_limit = result_limit
        self._context_limit = context_limit
        self._context_chars = context_chars
        self._max_rewrites = max_rewrites

    # ---------- 多轮指代消解 ----------

    async def condense(self, state: AgentState) -> dict[str, Any]:
        question = state["question"]
        history = state.get("history") or []
        if not history:
            return {"search_query": question}

        history_text = "\n".join(
            f"{item.get('role', 'user')}: {str(item.get('content', ''))[:200]}"
            for item in history[-6:]
        )
        prompt = (
            "结合对话历史，把用户最新问题改写为独立完整的问题（消解“它/这个/那”等指代）。\n"
            f"历史：\n{history_text}\n\n最新问题：{question}\n\n"
            "只输出改写后的问题；若本身独立完整则原样输出。"
        )
        try:
            condensed = await self._llm.text(prompt)
        except Exception:  # noqa: BLE001 - 失败时退回原问题
            logger.warning("指代消解失败，使用原问题", exc_info=True)
            condensed = question
        search_query = condensed.strip().strip('"').strip() or question
        if search_query != question:
            emit(state.get("run_id"), {"type": "condense", "search_query": search_query})
        return {"search_query": search_query}

    # ---------- 意图路由 ----------

    async def router(self, state: AgentState) -> dict[str, Any]:
        prompt = (
            "你是企业知识库的查询路由器。根据用户问题判断处理方式：\n"
            "- vector：涉及公司知识库可能记载的内容，包括内部系统（Milvus/Redis/接口/部署/备份/错误码）、"
            "制度流程（权限、报销、发票、入职、设备）、业务数据字典等\n"
            "- direct：打招呼、闲聊、通用常识、通用编程/数学问题，且不涉及公司内部系统细节\n"
            "- refuse：询问知识库范围外的私域信息（个人工资、绩效、未公开营收、他人隐私等）\n"
            "示例：\n"
            "“权限模型怎么控制字段权限” → vector\n"
            "“Milvus 的端口是多少” → vector\n"
            "“Python 列表推导怎么写” → direct\n"
            "“你好，你是谁” → direct\n"
            "“我的工资是多少” → refuse\n"
            '只输出 JSON，例如 {"route":"vector","reason":"询问内部流程"}\n'
            f"用户问题：{state['search_query']}"
        )
        try:
            decision = await self._llm.json(prompt, RouteDecision)
        except Exception:  # noqa: BLE001
            logger.warning("路由调用失败，默认走检索", exc_info=True)
            decision = None

        route = decision.route if decision else "vector"
        reason = decision.reason if decision else "路由失败，默认检索"
        emit(state.get("run_id"), {"type": "route", "route": route, "reason": reason})
        return {"route": route, "route_reason": reason}

    # ---------- 混合检索 ----------

    async def retrieve(self, state: AgentState) -> dict[str, Any]:
        top_k = state.get("top_k") or self._top_k
        limit = state.get("limit") or self._result_limit
        outcome = await asyncio.to_thread(
            self._retrieval.search,
            state["search_query"],
            top_k=top_k,
            limit=limit,
            rerank=False,
        )
        documents = [asdict(result) for result in outcome.results]
        emit(
            state.get("run_id"),
            {
                "type": "retrieval",
                "query": state["search_query"],
                "count": len(documents),
                "candidates": {
                    "bm25": outcome.bm25_candidates,
                    "dense": outcome.dense_candidates,
                    "fused": outcome.fused_candidates,
                },
            },
        )
        return {"documents": documents}

    # ---------- 相关性自检 ----------

    async def critique(self, state: AgentState) -> dict[str, Any]:
        documents = state.get("documents") or []
        if not documents:
            emit(state.get("run_id"), {"type": "critique", "relevant": False, "reason": "无检索结果"})
            return {"critique_relevant": False, "critique_reason": "无检索结果"}

        snippets = "\n\n".join(_doc_text(document, 500) for document in documents[:5])
        prompt = (
            "判断以下检索文档是否包含回答问题所需的信息。\n"
            "判定要宽松：只要文档覆盖问题的部分关键信息（关键词、数值、流程片段）就判 relevant=true；"
            "仅当文档与问题完全无关时才判 false。\n"
            '只输出 JSON，例如 {"relevant":true,"reason":"包含答案"}\n'
            f"问题：{state['search_query']}\n\n文档片段：\n{snippets}"
        )
        try:
            result = await self._llm.json(prompt, CritiqueResult)
        except Exception:  # noqa: BLE001
            logger.warning("自检调用失败，默认相关", exc_info=True)
            result = None

        if result is None:
            relevant, reason = True, "自检失败，默认相关"
        else:
            relevant, reason = result.relevant, result.reason
        emit(state.get("run_id"), {"type": "critique", "relevant": relevant, "reason": reason})
        return {"critique_relevant": relevant, "critique_reason": reason}

    # ---------- Query 改写 ----------

    async def rewrite(self, state: AgentState) -> dict[str, Any]:
        rewrite_count = int(state.get("rewrite_count", 0)) + 1
        prompt = (
            "检索结果无法回答该问题。请换一种表述（同义词扩展/拆解子问题/补充上下文）重写检索词。\n"
            f"原检索词：{state['search_query']}\n"
            f"失败原因：{state.get('critique_reason') or '未命中关键信息'}\n"
            "只输出改写后的检索词。"
        )
        try:
            rewritten = await self._llm.text(prompt)
        except Exception:  # noqa: BLE001
            logger.warning("Query 改写失败，追加提示词", exc_info=True)
            rewritten = f"{state['search_query']} 相关信息"
        rewritten = rewritten.strip().strip('"').strip() or state["search_query"]
        emit(
            state.get("run_id"),
            {"type": "rewrite", "query": rewritten, "rewrite_count": rewrite_count},
        )
        return {"search_query": rewritten, "rewrite_count": rewrite_count}

    # ---------- 生成 ----------

    def _build_context(self, documents: list[dict[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
        blocks: list[str] = []
        sources: list[dict[str, Any]] = []
        for index, document in enumerate(documents[: self._context_limit], start=1):
            text = _doc_text(document, self._context_chars)
            header = f"[{index}] 来源: {document.get('filename') or '未知文档'}"
            if document.get("section"):
                header += f" | 章节: {document['section']}"
            if document.get("page") is not None:
                header += f" | 页码: {document['page']}"
            blocks.append(f"{header}\n{text}")
            sources.append(
                {
                    "index": index,
                    "filename": document.get("filename"),
                    "section": document.get("section"),
                    "page": document.get("page"),
                    "chunk_id": document.get("chunk_id"),
                    "document_id": document.get("document_id"),
                    "score": document.get("score"),
                }
            )
        return "\n\n".join(blocks), sources

    async def generate(self, state: AgentState) -> dict[str, Any]:
        documents = state.get("documents") or []
        if not documents:
            emit(state.get("run_id"), {"type": "token", "text": NO_DOCS_ANSWER})
            return {"answer": NO_DOCS_ANSWER, "sources": [], "limited": True}

        context, sources = self._build_context(documents)
        limited = not state.get("critique_relevant", True)
        limitation = (
            "\n注意：现有资料可能不足以完整回答，请在回答中明确说明局限，不要编造。"
            if limited
            else ""
        )
        prompt = (
            "你是企业内部知识库问答助手。请严格依据【资料】回答，不要编造资料以外的信息；"
            "每个关键结论后用 [编号] 标注来源。\n"
            f"【资料】\n{context}\n\n【问题】\n{state['search_query']}{limitation}"
        )
        answer_parts: list[str] = []
        async for token in self._llm.stream(prompt):
            answer_parts.append(token)
            emit(state.get("run_id"), {"type": "token", "text": token})
        answer = "".join(answer_parts)
        if not answer:
            answer = NO_DOCS_ANSWER
            emit(state.get("run_id"), {"type": "token", "text": answer})
        return {"answer": answer, "sources": sources, "limited": limited}

    # ---------- 直接回答 / 拒答 ----------

    async def direct_answer(self, state: AgentState) -> dict[str, Any]:
        prompt = (
            "你是企业助手。当前问题不需要检索公司文档，请直接、简洁地回答。\n"
            f"问题：{state['question']}"
        )
        answer_parts: list[str] = []
        try:
            async for token in self._llm.stream(prompt):
                answer_parts.append(token)
                emit(state.get("run_id"), {"type": "token", "text": token})
        except Exception:  # noqa: BLE001
            logger.warning("直接回答失败", exc_info=True)
        answer = "".join(answer_parts) or "你好，我是企业知识库助手，可以问我文档相关的问题。"
        if not answer_parts:
            emit(state.get("run_id"), {"type": "token", "text": answer})
        return {"answer": answer, "sources": [], "limited": False}

    async def refuse(self, state: AgentState) -> dict[str, Any]:
        emit(state.get("run_id"), {"type": "token", "text": REFUSE_ANSWER})
        return {"answer": REFUSE_ANSWER, "sources": [], "limited": False}

    # ---------- 条件边 ----------

    def route_after_router(self, state: AgentState) -> str:
        route = state.get("route", "vector")
        return route if route in {"direct", "refuse"} else "retrieve"

    def route_after_critique(self, state: AgentState) -> str:
        if state.get("critique_relevant"):
            return "generate"
        if int(state.get("rewrite_count", 0)) < self._max_rewrites:
            return "rewrite"
        return "generate"
