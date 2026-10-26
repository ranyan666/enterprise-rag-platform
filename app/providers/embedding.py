"""嵌入提供者：默认通过 Ollama 调用 BGE-M3。

Ollama 的 ``/api/embed`` 返回 L2 归一化向量，天然适配余弦相似度；
同步客户端便于在后台处理线程中直接调用。
"""

from __future__ import annotations

import logging
from typing import Protocol, Sequence

import httpx

logger = logging.getLogger(__name__)

TRANSPORT_RETRIES = 1


class EmbeddingError(RuntimeError):
    """嵌入调用失败或返回结果不合法。"""


class Embedder(Protocol):
    """嵌入协议，便于替换为 sentence-transformers 等实现。"""

    @property
    def dim(self) -> int: ...

    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...


class OllamaEmbedder:
    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        dim: int,
        batch_size: int = 16,
        timeout: float = 120.0,
        client: httpx.Client | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._dim = dim
        self._batch_size = max(1, batch_size)
        self._client = client or httpx.Client(timeout=timeout)
        self._owns_client = client is None

    @property
    def dim(self) -> int:
        return self._dim

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self._batch_size):
            batch = list(texts[start : start + self._batch_size])
            vectors.extend(self._embed_batch(batch))
        return vectors

    def _embed_batch(self, batch: list[str]) -> list[list[float]]:
        payload = {"model": self._model, "input": batch}
        try:
            response = self._post_with_retry(payload)
        except httpx.HTTPStatusError as exc:
            raise EmbeddingError(
                f"Ollama 嵌入请求失败: HTTP {exc.response.status_code} {exc.response.text[:200]}"
            ) from exc
        except httpx.HTTPError as exc:
            raise EmbeddingError(f"Ollama 嵌入请求异常: {exc}") from exc

        try:
            data = response.json()
            vectors = data["embeddings"]
        except (ValueError, KeyError, TypeError) as exc:
            raise EmbeddingError("Ollama 返回结果缺少 embeddings 字段") from exc

        if len(vectors) != len(batch):
            raise EmbeddingError(f"嵌入条数不匹配: 期望 {len(batch)}，实际 {len(vectors)}")
        for vector in vectors:
            if len(vector) != self._dim:
                raise EmbeddingError(f"嵌入维度不匹配: 期望 {self._dim}，实际 {len(vector)}")
        return [[float(value) for value in vector] for vector in vectors]

    def _post_with_retry(self, payload: dict[str, object]) -> httpx.Response:
        last_error: httpx.TransportError | None = None
        for attempt in range(TRANSPORT_RETRIES + 1):
            try:
                response = self._client.post(f"{self._base_url}/api/embed", json=payload)
                response.raise_for_status()
                return response
            except httpx.TransportError as exc:
                last_error = exc
                logger.warning("嵌入请求传输失败（第 %d 次）: %s", attempt + 1, exc)
        raise last_error  # type: ignore[misc]

    def close(self) -> None:
        if self._owns_client:
            self._client.close()
