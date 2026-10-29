"""重排测试：cross-encoder 封装、检索集成与接口开关。"""

from __future__ import annotations

import pytest

from app.providers.reranker import CrossEncoderReranker, EmbeddingReranker, RerankError
from app.repositories.documents import DocumentRepository
from app.services.search import BM25Index, RetrievalService
from app.services.vector_store import DenseHit
from tests.fakes import FakeEmbedder, FakeReranker


class _FakeCrossEncoder:
    def __init__(self, scores: list[float], fail: Exception | None = None) -> None:
        self.scores = scores
        self.fail = fail
        self.pairs: list[tuple[str, str]] = []

    def predict(self, pairs):
        if self.fail is not None:
            raise self.fail
        self.pairs = list(pairs)
        return self.scores[: len(self.pairs)]


class _StubVectorStore:
    def __init__(self, hits: list[DenseHit]) -> None:
        self.hits = hits

    def search(self, query_vector, limit: int) -> list[DenseHit]:
        return self.hits[:limit]


class _KeywordReranker:
    def score(self, query: str, documents):
        return [1.0 if "第二" in document else 0.0 for document in documents]


def _repo(tmp_path) -> DocumentRepository:
    repository = DocumentRepository(tmp_path / "app.sqlite3")
    repository.init_schema()
    return repository


def _seed(repo: DocumentRepository, doc_id: str, text: str) -> None:
    repo.create_document(
        doc_id=doc_id,
        filename=f"{doc_id}.md",
        ext="md",
        size_bytes=1,
        sha256=f"{doc_id}-sha",
    )
    parent_id = f"{doc_id}-p1"
    repo.replace_chunks(
        doc_id,
        parents=[
            {
                "id": parent_id,
                "parent_id": None,
                "level": "parent",
                "ordinal": 1,
                "text": text,
                "char_count": len(text),
                "page": 1,
                "section": None,
            }
        ],
        children=[
            {
                "id": f"{doc_id}-c1",
                "parent_id": parent_id,
                "level": "child",
                "ordinal": 1,
                "text": text,
                "char_count": len(text),
                "page": 1,
                "section": None,
            }
        ],
        page_count=1,
    )


def _dense(doc_id: str, text: str) -> DenseHit:
    return DenseHit(
        id=f"{doc_id}-c1",
        document_id=doc_id,
        parent_id=f"{doc_id}-p1",
        text=text,
        section=None,
        page=1,
        score=0.9,
    )


def test_cross_encoder_reranker_scores_pairs() -> None:
    model = _FakeCrossEncoder([0.2, 0.9])
    reranker = CrossEncoderReranker("fake-model", model=model)

    scores = reranker.score("查询", ["第一段", "第二段"])

    assert scores == [0.2, 0.9]
    assert model.pairs == [("查询", "第一段"), ("查询", "第二段")]


def test_cross_encoder_reranker_empty_input() -> None:
    reranker = CrossEncoderReranker("fake-model", model=_FakeCrossEncoder([]))

    assert reranker.score("查询", []) == []


def test_cross_encoder_reranker_wraps_errors() -> None:
    reranker = CrossEncoderReranker("fake-model", model=_FakeCrossEncoder([], fail=RuntimeError("boom")))

    with pytest.raises(RerankError, match="boom"):
        reranker.score("查询", ["一段"])


def test_embedding_reranker_cosine_scores() -> None:
    class _FixedEmbedder:
        def embed(self, texts):
            mapping = {
                "查询": [1.0, 0.0],
                "相关": [1.0, 0.0],
                "无关": [0.0, 1.0],
            }
            return [mapping[text] for text in texts]

    reranker = EmbeddingReranker(_FixedEmbedder())

    scores = reranker.score("查询", ["相关", "无关"])

    assert scores == pytest.approx([1.0, 0.0])


def test_retrieval_reranks_candidates(tmp_path) -> None:
    repo = _repo(tmp_path)
    _seed(repo, "d1", "第一份文档内容")
    _seed(repo, "d2", "第二份文档内容")
    service = RetrievalService(
        repo,
        FakeEmbedder(dim=8),
        _StubVectorStore([_dense("d1", "第一份文档内容"), _dense("d2", "第二份文档内容")]),
        top_k=10,
        result_limit=5,
        reranker=_KeywordReranker(),  # type: ignore[arg-type]
        rerank_enabled=True,
    )

    outcome = service.search("文档")

    assert outcome.reranked is True
    assert outcome.results[0].document_id == "d2"
    assert outcome.results[0].rerank_score == 1.0
    assert outcome.timings["rerank_ms"] >= 0


def test_retrieval_without_reranker_has_no_score(tmp_path) -> None:
    repo = _repo(tmp_path)
    _seed(repo, "d1", "第一份文档内容")
    service = RetrievalService(
        repo,
        FakeEmbedder(dim=8),
        _StubVectorStore([_dense("d1", "第一份文档内容")]),
    )

    outcome = service.search("文档")

    assert outcome.reranked is False
    assert outcome.results[0].rerank_score is None


async def test_search_api_rerank_flag(client, app) -> None:
    await client.post(
        "/api/v1/documents",
        files={"file": ("a.md", "# 第一章\n\n权限模型内容。".encode(), "text/markdown")},
    )
    service = RetrievalService(
        app.state.documents_repo,
        app.state.embedder,
        app.state.vector_store,
        bm25_index=BM25Index(),
        reranker=FakeReranker(),  # type: ignore[arg-type]
        rerank_enabled=True,
    )
    app.state.retrieval = service

    reranked = await client.post("/api/v1/search", json={"query": "权限模型", "rerank": True})
    plain = await client.post("/api/v1/search", json={"query": "权限模型", "rerank": False})

    assert reranked.status_code == 200
    assert reranked.json()["reranked"] is True
    assert reranked.json()["results"][0]["rerank_score"] is not None
    assert plain.json()["reranked"] is False
    assert plain.json()["results"][0]["rerank_score"] is None
