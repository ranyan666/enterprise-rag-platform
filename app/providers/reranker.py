"""重排器：对候选段落做二阶段精排。

提供两种实现：
- ``EmbeddingReranker``：复用 bge-m3，对父段重新计算与问题的余弦相似度（无需额外模型）；
- ``CrossEncoderReranker``：交给 sentence-transformers CrossEncoder 精排（需额外模型）。
"""

from __future__ import annotations

import math
from typing import Any, Protocol, Sequence


class RerankError(RuntimeError):
    """重排模型调用失败。"""


class Reranker(Protocol):
    def score(self, query: str, documents: Sequence[str]) -> list[float]: ...


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if not left_norm or not right_norm:
        return 0.0
    return dot / (left_norm * right_norm)


class EmbeddingReranker:
    """用嵌入模型对父段重排，复用现有 bge-m3，避免额外模型下载。"""

    def __init__(self, embedder: Any) -> None:
        self._embedder = embedder

    def score(self, query: str, documents: Sequence[str]) -> list[float]:
        if not documents:
            return []
        vectors = self._embedder.embed([query, *documents])
        if len(vectors) != len(documents) + 1:
            raise RerankError(
                f"重排嵌入条数不匹配: 期望 {len(documents) + 1}，实际 {len(vectors)}"
            )
        query_vector, passages = vectors[0], vectors[1:]
        return [_cosine(query_vector, passage) for passage in passages]


class CrossEncoderReranker:
    """懒加载 sentence-transformers CrossEncoder，避免未启用时占用内存。"""

    def __init__(
        self,
        model_name: str,
        *,
        device: str = "cpu",
        model: Any | None = None,
    ) -> None:
        self._model_name = model_name
        self._device = device
        self._model = model

    def _ensure_model(self) -> Any:
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
            except ImportError as exc:  # pragma: no cover - 依赖安装问题
                raise RerankError("未安装 sentence-transformers，无法启用重排") from exc
            self._model = CrossEncoder(self._model_name, device=self._device)
        return self._model

    def score(self, query: str, documents: Sequence[str]) -> list[float]:
        if not documents:
            return []
        try:
            model = self._ensure_model()
            scores = model.predict([(query, document) for document in documents])
        except RerankError:
            raise
        except Exception as exc:  # noqa: BLE001 - 统一转领域异常
            raise RerankError(f"重排模型调用失败: {exc}") from exc
        return [float(score) for score in scores]
