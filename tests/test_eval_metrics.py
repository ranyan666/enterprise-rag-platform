"""评测指标计算测试（纯函数，不依赖服务）。"""

from __future__ import annotations

import math

import pytest

from scripts.evaluate_retrieval import hit, ndcg, percentile, rank_for, reciprocal_rank, summarize


def _result(filename: str, *, bm25: int | None = None, dense: int | None = None) -> dict:
    return {
        "filename": filename,
        "bm25_rank": bm25,
        "dense_rank": dense,
    }


def test_rank_for_hybrid_uses_result_order() -> None:
    results = [_result("a.md"), _result("b.md")]

    assert rank_for(results, "b.md", "hybrid") == 2
    assert rank_for(results, "missing.md", "hybrid") is None


def test_rank_for_channel_uses_min_rank_of_that_document() -> None:
    results = [
        _result("a.md", bm25=5, dense=9),
        _result("a.md", bm25=2, dense=1),
        _result("b.md", bm25=3, dense=None),
    ]

    assert rank_for(results, "a.md", "bm25_rank") == 2
    assert rank_for(results, "a.md", "dense_rank") == 1
    assert rank_for(results, "b.md", "dense_rank") is None


def test_hit_and_reciprocal_rank() -> None:
    assert hit(1, 5) == 1.0
    assert hit(5, 5) == 1.0
    assert hit(6, 5) == 0.0
    assert hit(None, 5) == 0.0
    assert reciprocal_rank(4, 10) == 0.25
    assert reciprocal_rank(11, 10) == 0.0
    assert reciprocal_rank(None, 10) == 0.0


def test_summarize_averages() -> None:
    metrics = summarize([1, 3, None, 8])

    assert metrics["hit@1"] == 0.25
    assert metrics["hit@3"] == 0.5
    assert metrics["hit@5"] == 0.5
    assert metrics["mrr@10"] == (1 + 1 / 3 + 0 + 1 / 8) / 4


def test_ndcg_uses_log_discount() -> None:
    assert ndcg(1) == 1.0
    assert ndcg(2) == pytest.approx(1 / math.log2(3))
    assert ndcg(10) == pytest.approx(1 / math.log2(11))
    assert ndcg(11) == 0.0
    assert ndcg(None) == 0.0


def test_percentile_handles_boundaries() -> None:
    assert percentile([], 0.95) == 0.0
    assert percentile([10.0], 0.95) == 10.0
    assert percentile([1.0, 2.0, 3.0, 4.0, 5.0], 0.95) == 5.0
