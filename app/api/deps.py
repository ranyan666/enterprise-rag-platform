"""接口层依赖：从当前应用实例取配置、仓储、模型与检索服务。"""

from __future__ import annotations

from fastapi import Request

from app.core.config import Settings
from app.providers.chat import ChatModel
from app.providers.embedding import Embedder
from app.repositories.documents import DocumentRepository
from app.services.qa import AnswerService
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


def get_chat_model(request: Request) -> ChatModel:
    return request.app.state.chat_model


def get_answer_service(request: Request) -> AnswerService:
    return request.app.state.answer_service
