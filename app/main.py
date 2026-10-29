"""FastAPI 应用入口。"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from langchain_ollama import ChatOllama
from redis import asyncio as aioredis

from app.agentic.graph import AgenticRagService, build_graph
from app.agentic.llm import AgentLLM
from app.agentic.memory import RedisHistoryStore
from app.agentic.nodes import AgentNodes
from app.api.chat import router as chat_router
from app.api.documents import router as documents_router
from app.api.health import router as health_router
from app.api.query import router as query_router
from app.api.search import router as search_router
from app.core.config import Settings, get_settings
from app.core.logging import configure_logging
from app.providers.chat import OllamaChatModel
from app.providers.embedding import OllamaEmbedder
from app.providers.langchain_chat import LangChainChatModel
from app.providers.reranker import CrossEncoderReranker, EmbeddingReranker
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
    if not settings.reranker_enabled:
        reranker = None
    elif settings.reranker_backend == "cross-encoder":
        reranker = CrossEncoderReranker(
            settings.reranker_model,
            device=settings.reranker_device,
        )
    else:
        reranker = EmbeddingReranker(embedder)
    retrieval = RetrievalService(
        repository,
        embedder,
        vector_store,
        bm25_index=BM25Index(),
        top_k=settings.retrieval_top_k,
        result_limit=settings.retrieval_result_limit,
        rrf_k=settings.rrf_k,
        reranker=reranker,
        rerank_top_n=settings.reranker_top_n,
        rerank_enabled=settings.reranker_enabled,
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

    agent_llm = AgentLLM(
        ChatOllama(
            model=settings.ollama_model,
            base_url=settings.ollama_base_url,
            temperature=0,
            num_predict=settings.qa_max_tokens,
            keep_alive="10m",
        )
    )
    agent_nodes = AgentNodes(
        retrieval,
        agent_llm,
        top_k=settings.retrieval_top_k,
        result_limit=settings.qa_context_limit,
        context_limit=settings.qa_context_limit,
        context_chars=settings.qa_context_chars,
        max_rewrites=settings.agent_max_rewrites,
    )
    agent_graph = build_graph(agent_nodes)
    redis_client = aioredis.from_url(
        settings.redis_url,
        decode_responses=True,
        socket_connect_timeout=settings.request_timeout_seconds,
        socket_timeout=settings.request_timeout_seconds,
    )
    history_store = RedisHistoryStore(
        redis_client,
        ttl_seconds=settings.agent_history_ttl_seconds,
        max_messages=settings.agent_history_turns * 2,
    )
    agentic_service = AgenticRagService(
        agent_graph,
        history_store,
        max_rewrites=settings.agent_max_rewrites,
    )
    logger.info(
        "对话后端=%s model=%s 重排=%s agentic=%s(max_rewrites=%d)",
        settings.chat_backend,
        settings.ollama_model,
        f"{settings.reranker_backend}:{settings.reranker_top_n}" if settings.reranker_enabled else "off",
        settings.agent_enabled,
        settings.agent_max_rewrites,
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
                await redis_client.aclose()
            except Exception:  # noqa: BLE001
                logger.warning("关闭 Redis 客户端失败", exc_info=True)
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
    app.state.agentic_service = agentic_service
    app.state.redis_client = redis_client
    app.include_router(health_router)
    app.include_router(documents_router)
    app.include_router(search_router)
    app.include_router(query_router)
    app.include_router(chat_router)
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
            "chat": "/api/v1/chat",
        }

    return app


app = create_app()
