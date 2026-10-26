"""文档上传保存与后台处理编排。"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import re
import shutil
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path

from fastapi import UploadFile

from app.core.config import Settings
from app.repositories.documents import DocumentRepository
from app.services.chunking import build_chunks
from app.services.parsing import parse_document

logger = logging.getLogger(__name__)

READ_CHUNK_BYTES = 1024 * 1024
FILENAME_MAX_LENGTH = 200
ERROR_MAX_LENGTH = 500
_CONTROL_CHARS = re.compile(r"[\x00-\x1f\x7f]")


class UploadTooLargeError(Exception):
    """上传文件超过大小限制。"""


class EmptyUploadError(Exception):
    """上传了空文件。"""


@dataclass(frozen=True, slots=True)
class SavedUpload:
    doc_id: str
    filename: str
    path: Path
    size_bytes: int
    sha256: str


def sanitize_filename(raw: str, ext: str) -> str:
    """只保留文件名本身，去掉路径、控制字符与首尾点号。"""

    name = (raw or "").replace("\\", "/").split("/")[-1]
    name = _CONTROL_CHARS.sub("", name).strip().strip(".")
    if not name:
        name = f"document.{ext}"
    return name[:FILENAME_MAX_LENGTH]


def _short_error(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"[:ERROR_MAX_LENGTH]


async def save_upload(
    file: UploadFile,
    *,
    ext: str,
    max_bytes: int,
    upload_dir: str,
) -> SavedUpload:
    """分片写入磁盘并计算 sha256，超限时清理半成品。"""

    filename = sanitize_filename(file.filename or f"document.{ext}", ext)
    doc_id = uuid.uuid4().hex
    target_dir = Path(upload_dir) / doc_id
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / f"source.{ext}"

    hasher = hashlib.sha256()
    size = 0
    try:
        with target.open("wb") as output:
            while True:
                chunk = await file.read(READ_CHUNK_BYTES)
                if not chunk:
                    break
                size += len(chunk)
                if size > max_bytes:
                    raise UploadTooLargeError(
                        f"文件超过大小限制（{max_bytes // (1024 * 1024)} MB）"
                    )
                hasher.update(chunk)
                output.write(chunk)
        if size == 0:
            raise EmptyUploadError("上传文件为空")
    except Exception:
        shutil.rmtree(target_dir, ignore_errors=True)
        raise
    finally:
        await file.close()

    return SavedUpload(
        doc_id=doc_id,
        filename=filename,
        path=target,
        size_bytes=size,
        sha256=hasher.hexdigest(),
    )


def initialize_storage(settings: Settings, repo: DocumentRepository) -> None:
    """创建目录、建表，并把上次中断的文档标记为失败。"""

    Path(settings.upload_dir).mkdir(parents=True, exist_ok=True)
    Path(settings.sqlite_path).parent.mkdir(parents=True, exist_ok=True)
    repo.init_schema()
    stale = repo.fail_stale_documents("服务重启，处理中断")
    if stale:
        logger.warning("启动时发现 %d 个中断的文档，已标记为 failed", stale)


def _process_document(doc_id: str, settings: Settings, repo: DocumentRepository) -> None:
    started = time.perf_counter()
    try:
        repo.mark_status(doc_id, "processing")
        document = repo.get_document(doc_id)
        if document is None:
            logger.warning("待处理文档不存在，跳过 id=%s", doc_id)
            return

        source = Path(settings.upload_dir) / doc_id / f"source.{document['ext']}"
        parsed = parse_document(source, document["ext"])
        bundle = build_chunks(
            parsed,
            doc_id=doc_id,
            parent_chars=settings.parent_chunk_chars,
            child_chars=settings.child_chunk_chars,
            child_overlap=settings.child_chunk_overlap,
        )
        repo.replace_chunks(
            doc_id,
            parents=[asdict(chunk) for chunk in bundle.parents],
            children=[asdict(chunk) for chunk in bundle.children],
            page_count=parsed.page_count,
        )
        logger.info(
            "文档处理完成 id=%s parents=%d children=%d 耗时=%.2fs",
            doc_id,
            len(bundle.parents),
            len(bundle.children),
            time.perf_counter() - started,
        )
    except Exception as exc:  # noqa: BLE001 - 后台任务必须吞掉异常并记录状态
        logger.exception("文档处理失败 id=%s", doc_id)
        try:
            repo.mark_status(doc_id, "failed", _short_error(exc))
        except Exception:  # noqa: BLE001
            logger.exception("写入失败状态时出错 id=%s", doc_id)


async def process_document_async(doc_id: str, settings: Settings, repo: DocumentRepository) -> None:
    """后台任务入口：CPU 密集的解析与分块放到线程池。"""

    await asyncio.to_thread(_process_document, doc_id, settings, repo)
