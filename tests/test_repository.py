"""SQLite 仓储测试。"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from app.repositories.documents import (
    DocumentNotFoundError,
    DocumentRepository,
    DuplicateDocumentError,
)

SHA = "a" * 64


def _repo(tmp_path: Path) -> DocumentRepository:
    repository = DocumentRepository(tmp_path / "app.sqlite3")
    repository.init_schema()
    return repository


def _create(repository: DocumentRepository, doc_id: str = "doc-1", sha: str = SHA) -> dict:
    return repository.create_document(
        doc_id=doc_id,
        filename="guide.txt",
        ext="txt",
        size_bytes=12,
        sha256=sha,
    )


def _chunk(chunk_id: str, level: str, ordinal: int, parent_id: str | None = None) -> dict:
    return {
        "id": chunk_id,
        "parent_id": parent_id,
        "level": level,
        "ordinal": ordinal,
        "text": f"{level}-{ordinal} 内容",
        "char_count": 10,
        "page": 1,
        "section": "第一章",
    }


def test_init_schema_is_idempotent(tmp_path: Path) -> None:
    repository = _repo(tmp_path)

    repository.init_schema()

    assert repository.count_documents() == 0


def test_create_get_list_and_count(tmp_path: Path) -> None:
    repository = _repo(tmp_path)

    created = _create(repository)

    assert created["status"] == "pending"
    assert repository.get_document("doc-1")["filename"] == "guide.txt"
    assert repository.get_document("missing") is None
    assert repository.count_documents() == 1
    assert [item["id"] for item in repository.list_documents(limit=10)] == ["doc-1"]


def test_duplicate_sha_reports_existing_id(tmp_path: Path) -> None:
    repository = _repo(tmp_path)
    _create(repository)

    with pytest.raises(DuplicateDocumentError) as exc_info:
        _create(repository, doc_id="doc-2")

    assert exc_info.value.existing_id == "doc-1"


def test_mark_status_and_fail_stale(tmp_path: Path) -> None:
    repository = _repo(tmp_path)
    _create(repository)
    repository.mark_status("doc-1", "processing")

    assert repository.get_document("doc-1")["status"] == "processing"
    assert repository.fail_stale_documents("服务重启") == 1
    assert repository.get_document("doc-1")["status"] == "failed"


def test_mark_status_missing_raises(tmp_path: Path) -> None:
    repository = _repo(tmp_path)

    with pytest.raises(DocumentNotFoundError):
        repository.mark_status("missing", "failed")


def test_replace_chunks_then_delete_cascades(tmp_path: Path) -> None:
    repository = _repo(tmp_path)
    _create(repository)

    repository.replace_chunks(
        "doc-1",
        parents=[_chunk("p1", "parent", 1)],
        children=[_chunk("c1", "child", 1, "p1"), _chunk("c2", "child", 2, "p1")],
        page_count=2,
    )

    document = repository.get_document("doc-1")
    assert document["status"] == "ready"
    assert document["parent_count"] == 1
    assert document["child_count"] == 2
    assert document["page_count"] == 2
    assert document["vector_count"] == 0
    assert repository.count_chunks("doc-1") == 3
    assert repository.count_chunks("doc-1", "child") == 2
    assert [item["id"] for item in repository.get_chunks("doc-1", level="child")] == ["c1", "c2"]

    repository.delete_document("doc-1")

    assert repository.get_document("doc-1") is None
    with pytest.raises(DocumentNotFoundError):
        repository.count_chunks("doc-1")


def test_set_vector_count_and_reset_for_reindex(tmp_path: Path) -> None:
    repository = _repo(tmp_path)
    _create(repository)
    repository.replace_chunks(
        "doc-1",
        parents=[_chunk("p1", "parent", 1)],
        children=[_chunk("c1", "child", 1, "p1")],
        page_count=1,
    )
    repository.set_vector_count("doc-1", 1)
    assert repository.get_document("doc-1")["vector_count"] == 1

    repository.reset_for_reindex("doc-1")

    document = repository.get_document("doc-1")
    assert document["status"] == "pending"
    assert document["error"] is None
    assert document["vector_count"] == 0
    assert document["parent_count"] == 0
    assert repository.count_chunks("doc-1") == 0


def test_init_schema_migrates_legacy_table(tmp_path: Path) -> None:
    db_path = tmp_path / "legacy.sqlite3"
    connection = sqlite3.connect(db_path)
    connection.execute(
        """
        CREATE TABLE documents (
          id TEXT PRIMARY KEY, filename TEXT NOT NULL, ext TEXT NOT NULL,
          size_bytes INTEGER NOT NULL, sha256 TEXT NOT NULL UNIQUE,
          status TEXT NOT NULL, error TEXT,
          page_count INTEGER NOT NULL DEFAULT 0,
          parent_count INTEGER NOT NULL DEFAULT 0,
          child_count INTEGER NOT NULL DEFAULT 0,
          created_at TEXT NOT NULL, updated_at TEXT NOT NULL
        )
        """
    )
    connection.execute(
        "INSERT INTO documents VALUES ('old', 'a.txt', 'txt', 1, ?, 'ready', NULL, 0, 0, 0, 't', 't')",
        ("0" * 64,),
    )
    connection.commit()
    connection.close()

    repository = DocumentRepository(db_path)
    repository.init_schema()

    assert repository.get_document("old")["vector_count"] == 0
