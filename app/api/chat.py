"""Agentic RAG 对话接口：``POST /api/v1/chat``，SSE 流式。"""

from __future__ import annotations

import json
import logging

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.api.deps import get_agent_service
from app.agentic.graph import AgenticRagService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["agentic"])


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=500, description="用户问题")
    session_id: str | None = Field(default=None, max_length=64, description="会话 ID，用于多轮记忆")
    top_k: int | None = Field(default=None, ge=1, le=100)
    limit: int | None = Field(default=None, ge=1, le=20)


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@router.post(
    "/chat",
    summary="Agentic RAG 对话（SSE 流式）",
    description=(
        "LangGraph 编排：多轮指代消解 → 意图路由（检索/直接回答/拒答）→ 混合检索 → "
        "相关性自检 →（必要时）Query 改写重检索（最多 2 次）→ 带引用流式生成。"
        "事件：start/condense/route/retrieval/critique/rewrite/token/done/error。"
    ),
)
async def chat(
    payload: ChatRequest,
    service: AgenticRagService = Depends(get_agent_service),
) -> StreamingResponse:
    question = payload.question.strip()
    if not question:
        raise HTTPException(status_code=422, detail="question 不能为空")

    async def event_stream():
        try:
            async for event in service.stream(
                question,
                session_id=payload.session_id,
                top_k=payload.top_k,
                limit=payload.limit,
            ):
                yield _sse(str(event.get("type", "message")), event)
        except Exception as exc:  # noqa: BLE001 - 流开始后只能以 error 收尾
            logger.exception("Agentic 对话流异常")
            yield _sse("error", {"message": f"服务内部错误: {exc}"})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )
