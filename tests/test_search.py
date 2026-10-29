"""混合检索测试：分词、BM25 索引、RRF 融合与父块回填。"""

from __future__ import annotations

import pytest

from app.repositories.documents import DocumentRepository
from app.services.search import BM25Hit, BM25Index, RetrievalService, rrf_fuse, tokenize
from app.services.vector_store import DenseHit
from tests.fakes import FakeEmbedder


class _StubVectorStore:
    def __init__(self, hits: list[DenseHit]) -> None:
        self.hits = hits

    def search(self, query_vector, limit: int) -> list[DenseHit]:
        return self.hits[:limit]


def _repo(tmp_path) -> DocumentRepository:
    repository = DocumentRepository(tmp_path / "app.sqlite3")
    repository.init_schema()
    return repository


def _seed(
    repo: DocumentRepository,
    doc_id: str,
    texts: list[str],
    *,
    filename: str | None = None,
    page: int = 1,
) -> None:
    repo.create_document(
        doc_id=doc_id,
        filename=filename or f"{doc_id}.md",
        ext="md",
        size_bytes=1,
        sha256=f"{doc_id}-sha",
    )
    parent_id = f"{doc_id}-p1"
    parents = [
        {
            "id": parent_id,
            "parent_id": None,
            "level": "parent",
            "ordinal": 1,
            "text": "\n".join(texts),
            "char_count": sum(len(text) for text in texts),
            "page": page,
            "section": "第一章",
        }
    ]
    children = [
        {
            "id": f"{doc_id}-c{index}",
            "parent_id": parent_id,
            "level": "child",
            "ordinal": index,
            "text": text,
            "char_count": len(text),
            "page": page,
            "section": "第一章",
        }
        for index, text in enumerate(texts, start=1)
    ]
    repo.replace_chunks(doc_id, parents=parents, children=children, page_count=1)


def _dense(doc_id: str, text: str = "权限模型") -> DenseHit:
    return DenseHit(
        id=f"{doc_id}-c1",
        document_id=doc_id,
        parent_id=f"{doc_id}-p1",
        text=text,
        section="第一章",
        page=1,
        score=0.88,
    )


def test_tokenize_keeps_terms_and_drops_stopwords() -> None:
    tokens = tokenize("企业级 RAG 平台的权限模型，如何使用？")

    assert "rag" in tokens
    assert "权限" in tokens
    assert "模型" in tokens
    assert "的" not in tokens
    assert "如何" not in tokens


def test_tokenize_ignores_punctuation_only() -> None:
    assert tokenize("！！！。。。") == []


def test_bm25_ranks_relevant_document_first_and_rebuilds(tmp_path) -> None:
    repo = _repo(tmp_path)
    _seed(repo, "auth", ["权限模型包含 RBAC 角色与用户绑定。"])
    _seed(repo, "billing", ["计费系统按月生成账单并支持发票。"])

    index = BM25Index()
    hits = index.search(repo, "权限模型", 10)

    assert hits
    assert hits[0].document_id == "auth"

    _seed(repo, "rbac", ["新的权限模型支持数据权限与字段权限。"])

    hits_after = index.search(repo, "字段权限", 10)
    assert hits_after[0].document_id == "rbac"


def test_bm25_skips_failed_documents(tmp_path) -> None:
    repo = _repo(tmp_path)
    _seed(repo, "doc-1", ["权限模型内容。"])
    repo.mark_status("doc-1", "failed")

    index = BM25Index()

    assert index.search(repo, "权限模型", 10) == []


def test_rrf_fusion_scores_and_order() -> None:
    bm25_hits = [
        BM25Hit("a", "d1", "p1", "A", None, 1, 3.0),
        BM25Hit("b", "d2", "p2", "B", None, 1, 2.0),
    ]
    dense_hits = [
        DenseHit("b", "d2", "p2", "B", None, 1, 0.9),
        DenseHit("c", "d3", "p3", "C", None, 1, 0.8),
    ]

    fused = rrf_fuse(bm25_hits, dense_hits, k=60)

    assert [item.chunk_id for item in fused] == ["b", "a", "c"]
    assert fused[0].bm25_rank == 2
    assert fused[0].dense_rank == 1
    assert fused[0].score == pytest.approx(1 / 62 + 1 / 61)
    assert fused[1].score == pytest.approx(1 / 61)
    assert fused[2].score == pytest.approx(1 / 62)


def test_retrieval_service_backfills_parent_text(tmp_path) -> None:
    repo = _repo(tmp_path)
    _seed(repo, "d1", ["权限模型包含角色与用户绑定。"], filename="安全设计.md")
    service = RetrievalService(
        repo,
        FakeEmbedder(dim=8),
        _StubVectorStore([_dense("d1")]),
        top_k=10,
        result_limit=5,
    )

    outcome = service.search("权限模型")

    assert outcome.results, "应该有召回结果"
    result = outcome.results[0]
    assert result.document_id == "d1"
    assert result.filename == "安全设计.md"
    assert result.parent_id == "d1-p1"
    assert result.parent_text == "权限模型包含角色与用户绑定。"
    assert result.bm25_rank == 1
    assert result.dense_rank == 1
    assert outcome.bm25_candidates == 1
    assert outcome.dense_candidates == 1
    assert set(outcome.timings) == {"bm25_ms", "dense_ms", "rerank_ms", "fusion_ms", "total_ms"}


def test_retrieval_filters_hits_from_failed_documents(tmp_path) -> None:
    repo = _repo(tmp_path)
    _seed(repo, "d1", ["权限模型内容。"])
    repo.mark_status("d1", "failed")
    service = RetrievalService(repo, FakeEmbedder(dim=8), _StubVectorStore([_dense("d1")]))

    outcome = service.search("权限模型")

    assert outcome.results == ()
    assert outcome.fused_candidates == 0


def test_retrieval_empty_query_returns_empty(tmp_path) -> None:
    repo = _repo(tmp_path)
    service = RetrievalService(repo, FakeEmbedder(dim=8), _StubVectorStore([]))

    outcome = service.search("   ")

    assert outcome.results == ()
    assert outcome.timings == {"total_ms": 0.0}
