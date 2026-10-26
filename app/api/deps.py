"""接口层依赖：从当前应用实例取配置、仓储、嵌入器与向量库。"""

from __future__ import annotations

from fastapi import Request

from app.core.config import Settings
from app.providers.embedding import Embedder
from app.repositories.documents import DocumentRepository
from app.services.search import RetrievalService
from app.services.vector_store import VectorStore


def get_app_settings(request: Request) -> Settings:
    return request.app.state.settings


def get_repo(request: Request) -> DocumentRepository:
    return request.app.state.documents_repo


def get_embedder(request: Request) -> Embedder:
    return request.app.state.embedder


def get_vector_store(request: Request) -> VectorStore:
    return request.app.state.vector_store


def get_retrieval(request: Request) -> RetrievalService:
    return request.app.state.retrieval
