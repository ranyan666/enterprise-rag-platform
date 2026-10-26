"""文档接口：上传、列表、详情、分块查询与删除。"""

from __future__ import annotations

import asyncio
import logging
import shutil
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, Query, Response, UploadFile
from pydantic import BaseModel, Field

from app.api.deps import get_app_settings, get_embedder, get_repo, get_vector_store
from app.core.config import Settings
from app.providers.embedding import Embedder
from app.repositories.documents import DocumentNotFoundError, DocumentRepository, DuplicateDocumentError
from app.services.documents import (
    EmptyUploadError,
    UploadTooLargeError,
    process_document_async,
    save_upload,
)
from app.services.parsing import SUPPORTED_EXTENSIONS
from app.services.vector_store import VectorStore

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/documents", tags=["documents"])

StatusLiteral = Literal["pending", "processing", "ready", "failed"]
LevelLiteral = Literal["parent", "child"]


class UploadAccepted(BaseModel):
    id: str
    filename: str
    status: str
    location: str = Field(description="文档资源地址，可轮询状态")


class ReindexAccepted(BaseModel):
    id: str
    status: str


class DocumentOut(BaseModel):
    id: str
    filename: str
    ext: str
    size_bytes: int
    status: StatusLiteral
    error: str | None = None
    page_count: int
    parent_count: int
    child_count: int
    vector_count: int
    created_at: str
    updated_at: str


class DocumentList(BaseModel):
    total: int
    limit: int
    offset: int
    items: list[DocumentOut]


class ChunkOut(BaseModel):
    id: str
    parent_id: str | None = None
    level: LevelLiteral
    ordinal: int
    text: str
    char_count: int
    page: int | None = None
    section: str | None = None


class ChunkList(BaseModel):
    total: int
    limit: int
    offset: int
    items: list[ChunkOut]


def _extension(filename: str | None) -> str:
    return Path(filename or "").suffix.lower().lstrip(".")


def _not_found(doc_id: str) -> HTTPException:
    return HTTPException(status_code=404, detail=f"文档不存在: {doc_id}")


@router.post("", response_model=UploadAccepted, status_code=202, summary="上传文档")
async def upload_document(
    response: Response,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(description="PDF / DOCX / MD / TXT 文件"),
    settings: Settings = Depends(get_app_settings),
    repo: DocumentRepository = Depends(get_repo),
    embedder: Embedder = Depends(get_embedder),
    vector_store: VectorStore = Depends(get_vector_store),
) -> UploadAccepted:
    ext = _extension(file.filename)
    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail=f"不支持的文件类型: {ext or '未知'}；支持 {sorted(SUPPORTED_EXTENSIONS)}",
        )

    try:
        saved = await save_upload(
            file,
            ext=ext,
            max_bytes=settings.max_upload_mb * 1024 * 1024,
            upload_dir=settings.upload_dir,
        )
    except UploadTooLargeError as exc:
        raise HTTPException(status_code=413, detail=str(exc)) from exc
    except EmptyUploadError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    try:
        await asyncio.to_thread(
            repo.create_document,
            doc_id=saved.doc_id,
            filename=saved.filename,
            ext=ext,
            size_bytes=saved.size_bytes,
            sha256=saved.sha256,
        )
    except DuplicateDocumentError as exc:
        shutil.rmtree(saved.path.parent, ignore_errors=True)
        raise HTTPException(
            status_code=409,
            detail={"message": "相同内容的文档已存在", "existing_id": exc.existing_id},
        ) from exc

    background_tasks.add_task(
        process_document_async,
        saved.doc_id,
        settings,
        repo,
        embedder,
        vector_store,
    )
    location = f"/api/v1/documents/{saved.doc_id}"
    response.headers["Location"] = location
    return UploadAccepted(
        id=saved.doc_id,
        filename=saved.filename,
        status="pending",
        location=location,
    )


@router.get("", response_model=DocumentList, summary="文档列表")
async def list_documents(
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    status: StatusLiteral | None = None,
    repo: DocumentRepository = Depends(get_repo),
) -> DocumentList:
    items = await asyncio.to_thread(
        repo.list_documents, limit=limit, offset=offset, status=status
    )
    total = await asyncio.to_thread(repo.count_documents, status)
    return DocumentList(
        total=total,
        limit=limit,
        offset=offset,
        items=[DocumentOut(**item) for item in items],
    )


@router.get("/{doc_id}", response_model=DocumentOut, summary="文档详情")
async def get_document(doc_id: str, repo: DocumentRepository = Depends(get_repo)) -> DocumentOut:
    document = await asyncio.to_thread(repo.get_document, doc_id)
    if document is None:
        raise _not_found(doc_id)
    return DocumentOut(**document)


@router.post(
    "/{doc_id}/reindex",
    response_model=ReindexAccepted,
    status_code=202,
    summary="重新解析并写入向量",
)
async def reindex_document(
    doc_id: str,
    background_tasks: BackgroundTasks,
    settings: Settings = Depends(get_app_settings),
    repo: DocumentRepository = Depends(get_repo),
    embedder: Embedder = Depends(get_embedder),
    vector_store: VectorStore = Depends(get_vector_store),
) -> ReindexAccepted:
    document = await asyncio.to_thread(repo.get_document, doc_id)
    if document is None:
        raise _not_found(doc_id)
    if document["status"] == "processing":
        raise HTTPException(status_code=409, detail="文档正在处理中，请稍后再试")

    try:
        await asyncio.to_thread(vector_store.delete_document, doc_id)
    except Exception:  # noqa: BLE001 - 旧向量清理失败不阻塞重索引
        logger.warning("重索引前清理旧向量失败 id=%s", doc_id, exc_info=True)

    await asyncio.to_thread(repo.reset_for_reindex, doc_id)
    background_tasks.add_task(
        process_document_async,
        doc_id,
        settings,
        repo,
        embedder,
        vector_store,
    )
    return ReindexAccepted(id=doc_id, status="pending")


@router.get("/{doc_id}/chunks", response_model=ChunkList, summary="文档分块查询")
async def list_chunks(
    doc_id: str,
    level: LevelLiteral | None = None,
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    repo: DocumentRepository = Depends(get_repo),
) -> ChunkList:
    try:
        items = await asyncio.to_thread(
            repo.get_chunks, doc_id, level=level, limit=limit, offset=offset
        )
        total = await asyncio.to_thread(repo.count_chunks, doc_id, level)
    except DocumentNotFoundError as exc:
        raise _not_found(doc_id) from exc
    return ChunkList(
        total=total,
        limit=limit,
        offset=offset,
        items=[ChunkOut(**item) for item in items],
    )


@router.delete("/{doc_id}", status_code=204, summary="删除文档")
async def delete_document(
    doc_id: str,
    settings: Settings = Depends(get_app_settings),
    repo: DocumentRepository = Depends(get_repo),
    vector_store: VectorStore = Depends(get_vector_store),
) -> Response:
    document = await asyncio.to_thread(repo.get_document, doc_id)
    if document is None:
        raise _not_found(doc_id)
    if document["status"] == "processing":
        raise HTTPException(status_code=409, detail="文档正在处理中，暂不能删除")

    try:
        await asyncio.to_thread(repo.delete_document, doc_id)
    except DocumentNotFoundError as exc:
        raise _not_found(doc_id) from exc
    try:
        await asyncio.to_thread(vector_store.delete_document, doc_id)
    except Exception:  # noqa: BLE001 - 向量清理失败不阻塞删除，孤儿向量由检索侧过滤
        logger.warning("删除文档向量失败 id=%s", doc_id, exc_info=True)
    shutil.rmtree(Path(settings.upload_dir) / document["id"], ignore_errors=True)
    return Response(status_code=204)
