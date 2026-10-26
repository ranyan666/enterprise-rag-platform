"""混合检索接口：``POST /api/v1/search``。"""

from __future__ import annotations

import asyncio
from dataclasses import asdict

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from app.api.deps import get_retrieval
from app.providers.embedding import EmbeddingError
from app.services.search import RetrievalService
from app.services.vector_store import VectorStoreError

router = APIRouter(prefix="/api/v1", tags=["search"])


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=500, description="检索问题")
    top_k: int | None = Field(default=None, ge=1, le=100, description="每路召回数量，默认取配置")
    limit: int | None = Field(default=None, ge=1, le=50, description="最终返回条数，默认取配置")


class SearchHit(BaseModel):
    chunk_id: str
    document_id: str
    filename: str | None = None
    parent_id: str | None = None
    section: str | None = None
    page: int | None = None
    score: float
    bm25_rank: int | None = None
    dense_rank: int | None = None
    bm25_score: float | None = None
    dense_score: float | None = None
    child_text: str
    parent_text: str


class SearchTimings(BaseModel):
    bm25_ms: float
    dense_ms: float
    fusion_ms: float
    total_ms: float


class SearchCandidates(BaseModel):
    bm25: int
    dense: int
    fused: int


class SearchResponse(BaseModel):
    query: str
    results: list[SearchHit]
    timings: SearchTimings
    candidates: SearchCandidates


@router.post("/search", response_model=SearchResponse, summary="混合检索（BM25 + 向量 + RRF）")
async def search(
    payload: SearchRequest,
    retrieval: RetrievalService = Depends(get_retrieval),
) -> SearchResponse:
    query = payload.query.strip()
    if not query:
        raise HTTPException(status_code=422, detail="query 不能为空")

    try:
        outcome = await asyncio.to_thread(
            lambda: retrieval.search(query, top_k=payload.top_k, limit=payload.limit)
        )
    except (EmbeddingError, VectorStoreError) as exc:
        raise HTTPException(status_code=503, detail=f"检索依赖不可用: {exc}") from exc

    return SearchResponse(
        query=query,
        results=[SearchHit(**asdict(item)) for item in outcome.results],
        timings=SearchTimings(**outcome.timings),
        candidates=SearchCandidates(
            bm25=outcome.bm25_candidates,
            dense=outcome.dense_candidates,
            fused=outcome.fused_candidates,
        ),
    )
