"""文档与分块的 SQLite 仓储。

设计取舍：
- 每次操作使用短连接，避免跨事件循环共享连接；
- 写操作在事务内完成，事务内不做 await；
- 唯一约束冲突转换为领域异常，由接口层映射为 409。
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Sequence

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
  id TEXT PRIMARY KEY,
  filename TEXT NOT NULL,
  ext TEXT NOT NULL,
  size_bytes INTEGER NOT NULL,
  sha256 TEXT NOT NULL UNIQUE,
  status TEXT NOT NULL CHECK (status IN ('pending', 'processing', 'ready', 'failed')),
  error TEXT,
  page_count INTEGER NOT NULL DEFAULT 0,
  parent_count INTEGER NOT NULL DEFAULT 0,
  child_count INTEGER NOT NULL DEFAULT 0,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS chunks (
  id TEXT PRIMARY KEY,
  document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
  parent_id TEXT,
  level TEXT NOT NULL CHECK (level IN ('parent', 'child')),
  ordinal INTEGER NOT NULL,
  text TEXT NOT NULL,
  char_count INTEGER NOT NULL,
  page INTEGER,
  section TEXT
);

CREATE INDEX IF NOT EXISTS idx_documents_created ON documents(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_documents_status ON documents(status);
CREATE INDEX IF NOT EXISTS idx_chunks_document ON chunks(document_id, level, ordinal);
CREATE INDEX IF NOT EXISTS idx_chunks_parent ON chunks(parent_id);
"""


class DocumentNotFoundError(LookupError):
    """文档不存在。"""


class DuplicateDocumentError(RuntimeError):
    """相同内容（sha256）的文档已存在。"""

    def __init__(self, existing_id: str) -> None:
        self.existing_id = existing_id
        super().__init__(f"文档已存在: {existing_id}")


def utcnow() -> str:
    """返回秒级 UTC ISO8601 字符串（以 Z 结尾）。"""

    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


class DocumentRepository:
    """文档元数据与分块的读写入口。"""

    def __init__(self, db_path: str | Path) -> None:
        self._db_path = Path(db_path)

    # ---------- 基础设施 ----------

    def _connect(self) -> sqlite3.Connection:
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self._db_path, timeout=5.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA busy_timeout=5000")
        return conn

    @contextmanager
    def _session(self) -> Iterator[sqlite3.Connection]:
        conn = self._connect()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def init_schema(self) -> None:
        with self._session() as conn:
            conn.executescript(SCHEMA)

    # ---------- documents ----------

    def create_document(
        self,
        *,
        doc_id: str,
        filename: str,
        ext: str,
        size_bytes: int,
        sha256: str,
    ) -> dict[str, Any]:
        now = utcnow()
        try:
            with self._session() as conn:
                conn.execute(
                    """
                    INSERT INTO documents
                        (id, filename, ext, size_bytes, sha256, status, page_count,
                         parent_count, child_count, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, 'pending', 0, 0, 0, ?, ?)
                    """,
                    (doc_id, filename, ext, size_bytes, sha256, now, now),
                )
        except sqlite3.IntegrityError as exc:
            existing = self.find_by_sha256(sha256)
            existing_id = existing["id"] if existing else ""
            raise DuplicateDocumentError(existing_id) from exc
        return self.get_document(doc_id)  # type: ignore[return-value]

    def get_document(self, doc_id: str) -> dict[str, Any] | None:
        with self._session() as conn:
            row = conn.execute("SELECT * FROM documents WHERE id = ?", (doc_id,)).fetchone()
        return dict(row) if row else None

    def find_by_sha256(self, sha256: str) -> dict[str, Any] | None:
        with self._session() as conn:
            row = conn.execute("SELECT * FROM documents WHERE sha256 = ?", (sha256,)).fetchone()
        return dict(row) if row else None

    def list_documents(
        self,
        *,
        limit: int = 20,
        offset: int = 0,
        status: str | None = None,
    ) -> list[dict[str, Any]]:
        sql = "SELECT * FROM documents"
        params: list[Any] = []
        if status:
            sql += " WHERE status = ?"
            params.append(status)
        sql += " ORDER BY created_at DESC, id DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])
        with self._session() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [dict(row) for row in rows]

    def count_documents(self, status: str | None = None) -> int:
        sql = "SELECT COUNT(*) AS n FROM documents"
        params: list[Any] = []
        if status:
            sql += " WHERE status = ?"
            params.append(status)
        with self._session() as conn:
            return int(conn.execute(sql, params).fetchone()["n"])

    def mark_status(self, doc_id: str, status: str, error: str | None = None) -> None:
        with self._session() as conn:
            cursor = conn.execute(
                "UPDATE documents SET status = ?, error = ?, updated_at = ? WHERE id = ?",
                (status, error, utcnow(), doc_id),
            )
            if cursor.rowcount == 0:
                raise DocumentNotFoundError(doc_id)

    def fail_stale_documents(self, reason: str) -> int:
        """把遗留的 pending/processing 文档标记为失败（服务重启恢复）。"""

        with self._session() as conn:
            cursor = conn.execute(
                """
                UPDATE documents
                   SET status = 'failed', error = ?, updated_at = ?
                 WHERE status IN ('pending', 'processing')
                """,
                (reason, utcnow()),
            )
            return cursor.rowcount

    def delete_document(self, doc_id: str) -> str:
        """删除文档及其全部分块，返回删除前的状态。"""

        with self._session() as conn:
            row = conn.execute("SELECT status FROM documents WHERE id = ?", (doc_id,)).fetchone()
            if row is None:
                raise DocumentNotFoundError(doc_id)
            conn.execute("DELETE FROM chunks WHERE document_id = ?", (doc_id,))
            conn.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
            return str(row["status"])

    # ---------- chunks ----------

    def replace_chunks(
        self,
        doc_id: str,
        *,
        parents: Sequence[dict[str, Any]],
        children: Sequence[dict[str, Any]],
        page_count: int,
    ) -> None:
        """事务内重建分块并置为 ready，避免出现半成品数据。"""

        with self._session() as conn:
            exists = conn.execute("SELECT 1 FROM documents WHERE id = ?", (doc_id,)).fetchone()
            if exists is None:
                raise DocumentNotFoundError(doc_id)
            conn.execute("DELETE FROM chunks WHERE document_id = ?", (doc_id,))
            conn.executemany(
                """
                INSERT INTO chunks
                    (id, document_id, parent_id, level, ordinal, text, char_count, page, section)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        item["id"],
                        doc_id,
                        item.get("parent_id"),
                        item["level"],
                        item["ordinal"],
                        item["text"],
                        item["char_count"],
                        item.get("page"),
                        item.get("section"),
                    )
                    for item in [*parents, *children]
                ],
            )
            conn.execute(
                """
                UPDATE documents
                   SET status = 'ready', error = NULL, page_count = ?,
                       parent_count = ?, child_count = ?, updated_at = ?
                 WHERE id = ?
                """,
                (page_count, len(parents), len(children), utcnow(), doc_id),
            )

    def get_chunks(
        self,
        doc_id: str,
        *,
        level: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        self._ensure_document(doc_id)
        sql = "SELECT * FROM chunks WHERE document_id = ?"
        params: list[Any] = [doc_id]
        if level:
            sql += " AND level = ?"
            params.append(level)
        sql += " ORDER BY CASE level WHEN 'parent' THEN 0 ELSE 1 END, ordinal LIMIT ? OFFSET ?"
        params.extend([limit, offset])
        with self._session() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [dict(row) for row in rows]

    def count_chunks(self, doc_id: str, level: str | None = None) -> int:
        self._ensure_document(doc_id)
        sql = "SELECT COUNT(*) AS n FROM chunks WHERE document_id = ?"
        params: list[Any] = [doc_id]
        if level:
            sql += " AND level = ?"
            params.append(level)
        with self._session() as conn:
            return int(conn.execute(sql, params).fetchone()["n"])

    def _ensure_document(self, doc_id: str) -> None:
        with self._session() as conn:
            row = conn.execute("SELECT 1 FROM documents WHERE id = ?", (doc_id,)).fetchone()
        if row is None:
            raise DocumentNotFoundError(doc_id)
