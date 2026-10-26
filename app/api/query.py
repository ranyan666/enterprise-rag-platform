"""检索问答接口：``POST /api/v1/query``，SSE 流式输出。"""

from __future__ import annotations

import json
import logging

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.api.deps import get_answer_service
from app.services.qa import AnswerService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["qa"])


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=500, description="用户问题")
    top_k: int | None = Field(default=None, ge=1, le=100, description="每路召回数量")
    limit: int | None = Field(default=None, ge=1, le=50, description="参与生成的引用条数上限")


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@router.post(
    "/query",
    summary="检索问答（SSE 流式）",
    description=(
        "先做混合检索，再调用 Qwen 基于资料流式作答。"
        "事件序列：meta（引用与检索耗时）→ token（增量文本，可多次）→ done（完整答案与耗时）；"
        "异常时返回 error 事件。"
    ),
)
async def query(
    payload: QueryRequest,
    service: AnswerService = Depends(get_answer_service),
) -> StreamingResponse:
    question = payload.question.strip()
    if not question:
        raise HTTPException(status_code=422, detail="question 不能为空")

    async def event_stream():
        try:
            async for event in service.stream_answer(
                question,
                top_k=payload.top_k,
                limit=payload.limit,
            ):
                yield _sse(event.event, event.data)
        except Exception as exc:  # noqa: BLE001 - 流已开始，只能以 error 事件收尾
            logger.exception("问答流式输出异常")
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
