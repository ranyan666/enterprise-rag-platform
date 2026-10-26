"""Ollama 嵌入提供者测试（MockTransport，完全离线）。"""

from __future__ import annotations

import json

import httpx
import pytest

from app.providers.embedding import EmbeddingError, OllamaEmbedder


def _embedder(handler, *, dim: int = 4, batch_size: int = 2) -> OllamaEmbedder:
    return OllamaEmbedder(
        base_url="http://ollama.local",
        model="bge-m3",
        dim=dim,
        batch_size=batch_size,
        timeout=5,
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )


def test_embed_splits_batches_and_sends_model() -> None:
    payloads: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        payloads.append(body)
        return httpx.Response(200, json={"embeddings": [[0.1] * 4 for _ in body["input"]]})

    embedder = _embedder(handler)
    vectors = embedder.embed(["a", "b", "c", "d", "e"])

    assert len(vectors) == 5
    assert all(len(vector) == 4 for vector in vectors)
    assert [len(payload["input"]) for payload in payloads] == [2, 2, 1]
    assert all(payload["model"] == "bge-m3" for payload in payloads)
    assert embedder.dim == 4


def test_empty_input_skips_request() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("空输入不应发起请求")

    assert _embedder(handler).embed([]) == []


def test_dimension_mismatch_raises() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"embeddings": [[0.1] * 3]})

    with pytest.raises(EmbeddingError, match="维度"):
        _embedder(handler).embed(["a"])


def test_count_mismatch_raises() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"embeddings": []})

    with pytest.raises(EmbeddingError, match="条数"):
        _embedder(handler).embed(["a"])


def test_missing_field_raises() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"unexpected": 1})

    with pytest.raises(EmbeddingError, match="embeddings"):
        _embedder(handler).embed(["a"])


def test_http_error_raises() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="boom")

    with pytest.raises(EmbeddingError, match="HTTP 500"):
        _embedder(handler).embed(["a"])


def test_transport_error_is_retried_once() -> None:
    calls = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        if calls["count"] == 1:
            raise httpx.ConnectError("connection refused", request=request)
        return httpx.Response(200, json={"embeddings": [[0.0] * 4]})

    vectors = _embedder(handler).embed(["a"])

    assert len(vectors) == 1
    assert calls["count"] == 2
