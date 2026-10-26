"""pytest 公共夹具。"""

from __future__ import annotations

import os

import pytest
from httpx import ASGITransport, AsyncClient

from app.core.config import Settings, get_settings
from app.main import create_app
from app.repositories.documents import DocumentRepository
from app.services.documents import initialize_storage
from app.services.qa import AnswerService
from app.services.search import BM25Index, RetrievalService
from tests.fakes import FakeChatModel, FakeEmbedder, FakeVectorStore

# 宿主环境可能带有无关服务的变量（例如本机其他项目设置的 MILVUS_URI / REDIS_*），
# 测试必须隔离，否则默认值断言会随机器状态漂移。
_ISOLATED_PREFIXES = (
    "MILVUS_",
    "REDIS_",
    "OLLAMA_",
    "EMBEDDING_",
    "SQLITE_",
    "APP_",
    "UPLOAD_",
    "MAX_UPLOAD_",
    "PARENT_CHUNK_",
    "CHILD_CHUNK_",
)
_ISOLATED_KEYS = {"LOG_LEVEL", "REQUEST_TIMEOUT_SECONDS"}


@pytest.fixture(autouse=True)
def isolate_environment(monkeypatch: pytest.MonkeyPatch):
    for name in list(os.environ):
        if name in _ISOLATED_KEYS or name.startswith(_ISOLATED_PREFIXES):
            monkeypatch.delenv(name, raising=False)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def settings(tmp_path) -> Settings:
    """使用临时目录的确定性配置，避免读写真实 data/。"""

    return Settings(
        _env_file=None,
        sqlite_path=str(tmp_path / "app.sqlite3"),
        upload_dir=str(tmp_path / "uploads"),
    )


@pytest.fixture
def app(settings: Settings):
    application = create_app(settings)
    initialize_storage(settings, application.state.documents_repo)
    application.state.embedder = FakeEmbedder(dim=settings.embedding_dim)
    application.state.vector_store = FakeVectorStore()
    application.state.retrieval = RetrievalService(
        application.state.documents_repo,
        application.state.embedder,
        application.state.vector_store,
        bm25_index=BM25Index(),
        top_k=settings.retrieval_top_k,
        result_limit=settings.retrieval_result_limit,
        rrf_k=settings.rrf_k,
    )
    application.state.chat_model = FakeChatModel()
    application.state.answer_service = AnswerService(
        application.state.retrieval,
        application.state.chat_model,
        context_limit=settings.qa_context_limit,
        context_chars=settings.qa_context_chars,
    )
    yield application


@pytest.fixture
def repo(app) -> DocumentRepository:
    return app.state.documents_repo


@pytest.fixture
def embedder(app) -> FakeEmbedder:
    return app.state.embedder


@pytest.fixture
def vector_store(app) -> FakeVectorStore:
    return app.state.vector_store


@pytest.fixture
def retrieval(app) -> RetrievalService:
    return app.state.retrieval


@pytest.fixture
def chat_model(app) -> FakeChatModel:
    return app.state.chat_model


@pytest.fixture
def answer_service(app) -> AnswerService:
    return app.state.answer_service


@pytest.fixture
async def client(app):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as http_client:
        yield http_client
