"""FastAPI 应用入口。"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from langchain_ollama import ChatOllama

from app.api.documents import router as documents_router
from app.api.health import router as health_router
from app.api.query import router as query_router
from app.api.search import router as search_router
from app.core.config import Settings, get_settings
from app.core.logging import configure_logging
from app.providers.chat import OllamaChatModel
from app.providers.embedding import OllamaEmbedder
from app.providers.langchain_chat import LangChainChatModel
from app.repositories.documents import DocumentRepository
from app.services.documents import initialize_storage
from app.services.qa import AnswerService
from app.services.search import BM25Index, RetrievalService
from app.services.vector_store import MilvusVectorStore

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).resolve().parent / "static"


def create_app(settings: Settings | None = None) -> FastAPI:
    """按给定配置构建应用；测试可传入独立 Settings。"""

    settings = settings or get_settings()
    configure_logging(settings.log_level)
    repository = DocumentRepository(settings.sqlite_path)
    embedder = OllamaEmbedder(
        base_url=settings.ollama_base_url,
        model=settings.embedding_model,
        dim=settings.embedding_dim,
        batch_size=settings.embedding_batch_size,
        timeout=settings.embedding_timeout_seconds,
    )
    vector_store = MilvusVectorStore(
        uri=settings.milvus_uri,
        collection=settings.milvus_collection,
        dim=settings.embedding_dim,
    )
    retrieval = RetrievalService(
        repository,
        embedder,
        vector_store,
        bm25_index=BM25Index(),
        top_k=settings.retrieval_top_k,
        result_limit=settings.retrieval_result_limit,
        rrf_k=settings.rrf_k,
    )
    if settings.chat_backend == "langchain":
        chat_model = LangChainChatModel(
            ChatOllama(
                model=settings.ollama_model,
                base_url=settings.ollama_base_url,
                temperature=settings.qa_temperature,
                num_predict=settings.qa_max_tokens,
                keep_alive="10m",
            )
        )
    else:
        chat_model = OllamaChatModel(
            base_url=settings.ollama_base_url,
            model=settings.ollama_model,
            temperature=settings.qa_temperature,
            max_tokens=settings.qa_max_tokens,
            timeout=settings.ollama_chat_timeout_seconds,
        )
    answer_service = AnswerService(
        retrieval,
        chat_model,
        context_limit=settings.qa_context_limit,
        context_chars=settings.qa_context_chars,
    )
    logger.info(
        "对话后端=%s model=%s",
        settings.chat_backend,
        settings.ollama_model,
    )

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        initialize_storage(settings, repository)
        logger.info("%s v%s 启动", settings.app_name, settings.app_version)
        try:
            yield
        finally:
            embedder.close()
            vector_store.close()
            try:
                await chat_model.close()
            except Exception:  # noqa: BLE001 - 关闭阶段异常不影响退出
                logger.warning("关闭对话模型客户端失败", exc_info=True)
            logger.info("应用已关闭")

    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        lifespan=lifespan,
    )
    app.state.settings = settings
    app.state.documents_repo = repository
    app.state.embedder = embedder
    app.state.vector_store = vector_store
    app.state.retrieval = retrieval
    app.state.chat_model = chat_model
    app.state.answer_service = answer_service
    app.include_router(health_router)
    app.include_router(documents_router)
    app.include_router(search_router)
    app.include_router(query_router)
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/demo", include_in_schema=False)
    async def demo() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/", tags=["meta"], summary="服务信息")
    async def root() -> dict[str, str]:
        return {
            "name": settings.app_name,
            "version": settings.app_version,
            "docs": "/docs",
            "health": "/health",
            "documents": "/api/v1/documents",
            "search": "/api/v1/search",
            "query": "/api/v1/query",
        }

    return app


app = create_app()
