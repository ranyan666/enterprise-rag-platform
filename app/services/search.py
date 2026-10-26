"""混合检索：BM25 关键词 + Milvus 稠密向量，RRF 融合后回填父块。

设计要点：
- BM25 索引常驻内存，按仓储指纹自动重建，只索引 ready 文档的子块；
- 两路各召回 top_k，用 Reciprocal Rank Fusion 融合，避免不同量纲的分数直接相加；
- 最终只回填父块文本（原文完整段落），子块文本一并返回便于排查召回质量。
"""

from __future__ import annotations

import logging
import re
import threading
import time
from dataclasses import dataclass, replace
from typing import Any, Sequence

import jieba
from rank_bm25 import BM25Plus

from app.providers.embedding import Embedder
from app.repositories.documents import DocumentRepository
from app.services.vector_store import DenseHit, VectorStore

logger = logging.getLogger(__name__)

STOPWORDS = frozenset(
    {
        "的", "了", "在", "是", "和", "与", "及", "或", "对", "中", "上", "下", "为", "以", "等",
        "也", "就", "都", "而", "被", "把", "让", "从", "到", "这", "那", "其", "之", "并", "很",
        "更", "还", "有", "会", "可以", "一个", "我们", "你们", "他们", "什么", "怎么", "如何",
        "the", "a", "an", "is", "are", "was", "were", "of", "and", "or", "to", "in", "for",
        "with", "on", "at", "by", "as", "be", "this", "that", "it", "from", "how", "what",
    }
)
_TOKEN_PATTERN = re.compile(r"[0-9a-zA-Z_\u4e00-\u9fff]+")


def tokenize(text: str) -> list[str]:
    """中英文混合分词：jieba 搜索模式 + 停用词/标点过滤。"""

    tokens: list[str] = []
    for raw in jieba.lcut_for_search((text or "").lower()):
        token = raw.strip()
        if not token or token in STOPWORDS:
            continue
        if _TOKEN_PATTERN.fullmatch(token):
            tokens.append(token)
    return tokens


@dataclass(frozen=True, slots=True)
class BM25Hit:
    id: str
    document_id: str
    parent_id: str
    text: str
    section: str | None
    page: int | None
    score: float


@dataclass(frozen=True, slots=True)
class FusedCandidate:
    chunk_id: str
    document_id: str
    parent_id: str
    section: str | None
    page: int | None
    child_text: str
    score: float
    bm25_rank: int | None
    dense_rank: int | None
    bm25_score: float | None
    dense_score: float | None


@dataclass(frozen=True, slots=True)
class SearchResult:
    chunk_id: str
    document_id: str
    filename: str | None
    parent_id: str | None
    section: str | None
    page: int | None
    score: float
    bm25_rank: int | None
    dense_rank: int | None
    bm25_score: float | None
    dense_score: float | None
    child_text: str
    parent_text: str


@dataclass(frozen=True, slots=True)
class SearchOutcome:
    results: tuple[SearchResult, ...]
    bm25_candidates: int
    dense_candidates: int
    fused_candidates: int
    timings: dict[str, float]


class BM25Index:
    """常驻内存的 BM25 索引，使用仓储指纹做失效判断。"""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._fingerprint: tuple[int, int, str] | None = None
        self._chunks: list[dict[str, Any]] = []
        self._bm25: BM25Plus | None = None

    def search(self, repo: DocumentRepository, query: str, limit: int) -> list[BM25Hit]:
        self._ensure(repo)
        query_tokens = tokenize(query)
        if not query_tokens or self._bm25 is None or not self._chunks:
            return []

        scores = self._bm25.get_scores(query_tokens)
        ranked = sorted(
            range(len(scores)),
            key=lambda index: float(scores[index]),
            reverse=True,
        )[:limit]
        hits: list[BM25Hit] = []
        for index in ranked:
            score = float(scores[index])
            if score <= 0:
                continue
            chunk = self._chunks[index]
            hits.append(
                BM25Hit(
                    id=chunk["id"],
                    document_id=chunk["document_id"],
                    parent_id=chunk["parent_id"] or "",
                    text=chunk["text"],
                    section=chunk["section"],
                    page=chunk["page"],
                    score=score,
                )
            )
        return hits

    def _ensure(self, repo: DocumentRepository) -> None:
        fingerprint = repo.retrieval_fingerprint()
        if fingerprint == self._fingerprint and self._bm25 is not None:
            return
        with self._lock:
            fingerprint = repo.retrieval_fingerprint()
            if fingerprint == self._fingerprint and self._bm25 is not None:
                return
            indexed = [
                (chunk, tokens)
                for chunk in repo.list_ready_child_chunks()
                if (tokens := tokenize(chunk["text"]))
            ]
            self._chunks = [chunk for chunk, _ in indexed]
            self._bm25 = BM25Plus([tokens for _, tokens in indexed]) if indexed else None
            self._fingerprint = fingerprint
            logger.info("BM25 索引重建：%d 个子块", len(self._chunks))


def rrf_fuse(
    bm25_hits: Sequence[BM25Hit],
    dense_hits: Sequence[DenseHit],
    *,
    k: int = 60,
) -> list[FusedCandidate]:
    """Reciprocal Rank Fusion：score = Σ 1 / (k + rank)。"""

    candidates: dict[str, FusedCandidate] = {}
    for rank, hit in enumerate(bm25_hits, start=1):
        candidates[hit.id] = FusedCandidate(
            chunk_id=hit.id,
            document_id=hit.document_id,
            parent_id=hit.parent_id,
            section=hit.section,
            page=hit.page,
            child_text=hit.text,
            score=1.0 / (k + rank),
            bm25_rank=rank,
            dense_rank=None,
            bm25_score=hit.score,
            dense_score=None,
        )

    for rank, hit in enumerate(dense_hits, start=1):
        contribution = 1.0 / (k + rank)
        existing = candidates.get(hit.id)
        if existing is None:
            candidates[hit.id] = FusedCandidate(
                chunk_id=hit.id,
                document_id=hit.document_id,
                parent_id=hit.parent_id,
                section=hit.section,
                page=hit.page,
                child_text=hit.text,
                score=contribution,
                bm25_rank=None,
                dense_rank=rank,
                bm25_score=None,
                dense_score=hit.score,
            )
        else:
            candidates[hit.id] = replace(
                existing,
                score=existing.score + contribution,
                dense_rank=rank,
                dense_score=hit.score,
            )

    return sorted(
        candidates.values(),
        key=lambda item: (
            -item.score,
            item.bm25_rank if item.bm25_rank is not None else 10**6,
            item.dense_rank if item.dense_rank is not None else 10**6,
        ),
    )


class RetrievalService:
    def __init__(
        self,
        repo: DocumentRepository,
        embedder: Embedder,
        vector_store: VectorStore,
        *,
        bm25_index: BM25Index | None = None,
        top_k: int = 20,
        result_limit: int = 8,
        rrf_k: int = 60,
    ) -> None:
        self._repo = repo
        self._embedder = embedder
        self._vector_store = vector_store
        self._bm25 = bm25_index or BM25Index()
        self._top_k = top_k
        self._result_limit = result_limit
        self._rrf_k = rrf_k

    def search(self, query: str, *, top_k: int | None = None, limit: int | None = None) -> SearchOutcome:
        started = time.perf_counter()
        text = (query or "").strip()
        if not text:
            return SearchOutcome((), 0, 0, 0, {"total_ms": 0.0})

        recall_k = top_k or self._top_k
        result_limit = limit or self._result_limit

        bm25_started = time.perf_counter()
        bm25_hits = self._bm25.search(self._repo, text, recall_k)
        bm25_ms = (time.perf_counter() - bm25_started) * 1000

        dense_started = time.perf_counter()
        query_vector = self._embedder.embed([text])[0]
        dense_hits = self._vector_store.search(query_vector, recall_k)
        dense_ms = (time.perf_counter() - dense_started) * 1000

        fusion_started = time.perf_counter()
        fused = rrf_fuse(bm25_hits, dense_hits, k=self._rrf_k)
        doc_ids = list(dict.fromkeys(candidate.document_id for candidate in fused))
        documents = self._repo.get_documents_by_ids(doc_ids)
        ready = [
            candidate
            for candidate in fused
            if documents.get(candidate.document_id, {}).get("status") == "ready"
        ]
        selected = ready[:result_limit]
        parent_ids = list(dict.fromkeys(c.parent_id for c in selected if c.parent_id))
        parents = self._repo.get_chunks_by_ids(parent_ids)

        results: list[SearchResult] = []
        for candidate in selected:
            document = documents.get(candidate.document_id, {})
            parent = parents.get(candidate.parent_id or "", {})
            results.append(
                SearchResult(
                    chunk_id=candidate.chunk_id,
                    document_id=candidate.document_id,
                    filename=document.get("filename"),
                    parent_id=candidate.parent_id or None,
                    section=candidate.section,
                    page=candidate.page,
                    score=round(candidate.score, 6),
                    bm25_rank=candidate.bm25_rank,
                    dense_rank=candidate.dense_rank,
                    bm25_score=round(candidate.bm25_score, 4) if candidate.bm25_score is not None else None,
                    dense_score=round(candidate.dense_score, 4) if candidate.dense_score is not None else None,
                    child_text=candidate.child_text,
                    parent_text=parent.get("text") or candidate.child_text,
                )
            )
        fusion_ms = (time.perf_counter() - fusion_started) * 1000

        return SearchOutcome(
            results=tuple(results),
            bm25_candidates=len(bm25_hits),
            dense_candidates=len(dense_hits),
            fused_candidates=len(ready),
            timings={
                "bm25_ms": round(bm25_ms, 2),
                "dense_ms": round(dense_ms, 2),
                "fusion_ms": round(fusion_ms, 2),
                "total_ms": round((time.perf_counter() - started) * 1000, 2),
            },
        )
