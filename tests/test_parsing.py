"""文档解析测试：四种格式 + 异常路径。"""

from __future__ import annotations

from pathlib import Path

import pymupdf
import pytest
from docx import Document as DocxDocument

from app.services.parsing import (
    DocumentParseError,
    EmptyDocumentError,
    UnsupportedFormatError,
    decode_text,
    detect_heading_level,
    normalize_text,
    parse_document,
)


def _write_pdf(path: Path, pages: list[str]) -> None:
    document = pymupdf.open()
    for text in pages:
        page = document.new_page()
        page.insert_text((72, 72), text, fontsize=12)
    document.save(str(path))
    document.close()


def _write_docx(path: Path, paragraphs: list[tuple[str, str]]) -> None:
    document = DocxDocument()
    for text, style in paragraphs:
        document.add_paragraph(text, style=style)
    document.save(str(path))


def test_parse_plain_text_splits_paragraphs(tmp_path: Path) -> None:
    path = tmp_path / "guide.txt"
    path.write_text("第一段内容。\n\n第二段内容。", encoding="utf-8")

    parsed = parse_document(path, "txt")

    assert parsed.ext == "txt"
    assert [block.text for block in parsed.blocks] == ["第一段内容。", "第二段内容。"]
    assert parsed.page_count == 0


def test_parse_gb18030_text(tmp_path: Path) -> None:
    path = tmp_path / "gbk.txt"
    path.write_bytes("中文编码测试".encode("gb18030"))

    parsed = parse_document(path, "txt")

    assert parsed.blocks[0].text == "中文编码测试"


def test_parse_markdown_headings(tmp_path: Path) -> None:
    path = tmp_path / "doc.md"
    path.write_text(
        "# 第一章 概述\n\n平台介绍。\n\n## 1.1 背景\n\n背景说明。\n",
        encoding="utf-8",
    )

    parsed = parse_document(path, "md")

    headings = [(block.text, block.heading_level) for block in parsed.blocks if block.heading_level]
    assert headings == [("第一章 概述", 1), ("1.1 背景", 2)]


def test_parse_docx_heading_styles(tmp_path: Path) -> None:
    path = tmp_path / "doc.docx"
    _write_docx(
        path,
        [
            ("第一章 概述", "Heading 1"),
            ("正文内容。", "Normal"),
            ("1.1 背景", "Heading 2"),
            ("背景说明。", "Normal"),
        ],
    )

    parsed = parse_document(path, "docx")

    levels = [block.heading_level for block in parsed.blocks]
    assert levels == [1, 0, 2, 0]
    assert parsed.page_count == 0


def test_parse_pdf_keeps_page_numbers(tmp_path: Path) -> None:
    path = tmp_path / "doc.pdf"
    _write_pdf(path, ["1 Introduction\nPlatform overview.", "2 Design\nFastAPI and Milvus."])

    parsed = parse_document(path, "pdf")

    assert parsed.page_count == 2
    assert {block.page for block in parsed.blocks} == {1, 2}
    assert any(block.heading_level >= 1 for block in parsed.blocks)


def test_parse_rejects_unsupported_extension(tmp_path: Path) -> None:
    path = tmp_path / "run.exe"
    path.write_bytes(b"MZ")

    with pytest.raises(UnsupportedFormatError):
        parse_document(path, "exe")


def test_parse_rejects_fake_pdf(tmp_path: Path) -> None:
    path = tmp_path / "fake.pdf"
    path.write_text("not a real pdf", encoding="utf-8")

    with pytest.raises(DocumentParseError):
        parse_document(path, "pdf")


def test_parse_empty_file_raises(tmp_path: Path) -> None:
    path = tmp_path / "empty.txt"
    path.write_text("   \n\n", encoding="utf-8")

    with pytest.raises(EmptyDocumentError):
        parse_document(path, "txt")


def test_decode_text_falls_back_to_gb18030() -> None:
    assert decode_text("中文".encode("gb18030")) == "中文"


def test_normalize_text_cleans_control_and_blank_lines() -> None:
    raw = "第一行  \r\n\r\n\r\n\x00第二行\u200b"

    assert normalize_text(raw) == "第一行\n\n第二行"


def test_detect_heading_level_patterns() -> None:
    assert detect_heading_level("# 标题") == 1
    assert detect_heading_level("第三章 系统设计") == 1
    assert detect_heading_level("3.2 权限模型") == 2
    assert detect_heading_level("一、背景") == 2
    assert detect_heading_level("这是一句普通正文。") == 0
    assert detect_heading_level("2024 年销售额增长 20% 的主要原因是渠道扩张，详见附件。") == 0
