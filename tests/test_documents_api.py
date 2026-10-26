"""文档接口测试：上传、状态、分块、删除与异常路径。"""

from __future__ import annotations

from pathlib import Path

from httpx import ASGITransport, AsyncClient

from app.core.config import Settings
from app.main import create_app
from app.services.documents import initialize_storage


async def _upload(client: AsyncClient, *, name: str = "guide.txt", content: bytes = b"hello") -> object:
    return await client.post(
        "/api/v1/documents",
        files={"file": (name, content, "application/octet-stream")},
    )


async def test_upload_text_document_becomes_ready(client, settings: Settings) -> None:
    content = "第一段内容。\n\n第二段内容。".encode("utf-8")

    response = await _upload(client, content=content)

    assert response.status_code == 202
    accepted = response.json()
    assert accepted["status"] == "pending"
    location = f"/api/v1/documents/{accepted['id']}"
    assert response.headers["location"] == location

    detail = await client.get(location)
    assert detail.status_code == 200
    document = detail.json()
    assert document["status"] == "ready"
    assert document["error"] is None
    assert document["parent_count"] >= 1
    assert document["child_count"] >= 1
    assert (Path(settings.upload_dir) / accepted["id"] / "source.txt").exists()

    chunks = await client.get(f"{location}/chunks")
    assert chunks.status_code == 200
    assert chunks.json()["total"] == document["parent_count"] + document["child_count"]


async def test_upload_markdown_chunks_have_parent_links(client) -> None:
    content = ("# 第一章 概述\n\n" + "这是概述内容。" * 40 + "\n\n# 第二章 设计\n\n" + "这是设计内容。" * 40).encode()

    response = await _upload(client, name="design.md", content=content)
    doc_id = response.json()["id"]

    parents = await client.get(f"/api/v1/documents/{doc_id}/chunks", params={"level": "parent"})
    children = await client.get(f"/api/v1/documents/{doc_id}/chunks", params={"level": "child"})

    parent_ids = {item["id"] for item in parents.json()["items"]}
    assert parents.json()["total"] > 0
    assert children.json()["total"] > 0
    assert all(item["level"] == "parent" for item in parents.json()["items"])
    assert all(item["parent_id"] in parent_ids for item in children.json()["items"])
    assert {item["section"] for item in parents.json()["items"]} == {"第一章 概述", "第二章 设计"}


async def test_list_documents_paginates(client) -> None:
    for index in range(3):
        await _upload(client, name=f"doc-{index}.txt", content=f"文档 {index} 的内容。".encode())

    response = await client.get("/api/v1/documents", params={"limit": 2, "offset": 0})

    body = response.json()
    assert response.status_code == 200
    assert body["total"] == 3
    assert len(body["items"]) == 2
    assert body["limit"] == 2


async def test_duplicate_content_returns_409(client) -> None:
    first = await _upload(client, content="重复内容。".encode())
    second = await _upload(client, name="copy.txt", content="重复内容。".encode())

    assert second.status_code == 409
    assert second.json()["detail"]["existing_id"] == first.json()["id"]


async def test_unsupported_extension_returns_415(client) -> None:
    response = await _upload(client, name="virus.exe", content=b"MZ")

    assert response.status_code == 415


async def test_empty_file_returns_400(client) -> None:
    response = await _upload(client, name="empty.txt", content=b"")

    assert response.status_code == 400


async def test_oversize_file_returns_413(tmp_path: Path) -> None:
    settings = Settings(
        _env_file=None,
        sqlite_path=str(tmp_path / "app.sqlite3"),
        upload_dir=str(tmp_path / "uploads"),
        max_upload_mb=1,
    )
    app = create_app(settings)
    initialize_storage(settings, app.state.documents_repo)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await _upload(client, name="big.txt", content=b"a" * (1024 * 1024 + 10))

    assert response.status_code == 413


async def test_detail_missing_returns_404(client) -> None:
    response = await client.get("/api/v1/documents/does-not-exist")

    assert response.status_code == 404


async def test_delete_removes_rows_and_files(client, settings: Settings, repo) -> None:
    response = await _upload(client, content="待删除文档。".encode())
    doc_id = response.json()["id"]

    deleted = await client.delete(f"/api/v1/documents/{doc_id}")

    assert deleted.status_code == 204
    assert (await client.get(f"/api/v1/documents/{doc_id}")).status_code == 404
    assert repo.get_document(doc_id) is None
    assert not (Path(settings.upload_dir) / doc_id).exists()


async def test_delete_processing_document_returns_409(client, repo) -> None:
    repo.create_document(
        doc_id="processing-doc",
        filename="p.txt",
        ext="txt",
        size_bytes=1,
        sha256="d" * 64,
    )
    repo.mark_status("processing-doc", "processing")

    response = await client.delete("/api/v1/documents/processing-doc")

    assert response.status_code == 409


async def test_parse_failure_marks_document_failed(client) -> None:
    response = await _upload(client, name="broken.pdf", content=b"%PDF-1.4 broken")

    doc_id = response.json()["id"]
    detail = await client.get(f"/api/v1/documents/{doc_id}")

    body = detail.json()
    assert body["status"] == "failed"
    assert body["error"]
    assert body["parent_count"] == 0


async def test_filename_is_sanitized(client) -> None:
    response = await _upload(client, name="../../evil.txt", content="安全内容。".encode())

    assert response.json()["filename"] == "evil.txt"
