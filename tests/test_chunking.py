"""父子分块测试。"""

from __future__ import annotations

import pytest

from app.services.chunking import build_chunks
from app.services.parsing import Block, EmptyDocumentError, ParsedDocument


def _parsed(blocks: list[Block]) -> ParsedDocument:
    return ParsedDocument(ext="txt", blocks=tuple(blocks), page_count=0)


def _build(parsed: ParsedDocument, **overrides: int):
    params = {
        "doc_id": "doc-1",
        "parent_chars": 1800,
        "child_chars": 400,
        "child_overlap": 80,
    }
    params.update(overrides)
    return build_chunks(parsed, **params)


def test_short_document_has_single_parent_and_child() -> None:
    bundle = _build(_parsed([Block("这是一段简短的正文内容。")]))

    assert len(bundle.parents) == 1
    assert len(bundle.children) == 1
    parent, child = bundle.parents[0], bundle.children[0]
    assert child.parent_id == parent.id
    assert parent.ordinal == 1 and child.ordinal == 1
    assert child.section is None


def test_section_path_is_inherited_by_chunks() -> None:
    parsed = _parsed(
        [
            Block("第一章 概述", heading_level=1),
            Block("概述正文。"),
            Block("1.1 背景", heading_level=2),
            Block("背景正文。"),
        ]
    )

    bundle = _build(parsed)

    assert [parent.section for parent in bundle.parents] == ["第一章 概述", "第一章 概述 > 1.1 背景"]
    assert [child.section for child in bundle.children] == ["第一章 概述", "第一章 概述 > 1.1 背景"]


def test_parent_chunks_respect_char_limit() -> None:
    parsed = _parsed([Block("字" * 80) for _ in range(10)])

    bundle = _build(parsed, parent_chars=200, child_chars=100, child_overlap=20)

    assert all(parent.char_count <= 200 for parent in bundle.parents)
    assert len(bundle.parents) == 5


def test_child_windows_have_expected_size_and_overlap() -> None:
    text = "0123456789" * 100  # 1000 字符

    bundle = _build(_parsed([Block(text)]), parent_chars=1000, child_chars=300, child_overlap=50)

    assert len(bundle.parents) == 1
    assert len(bundle.children) == 4
    assert all(child.char_count <= 300 for child in bundle.children)
    assert bundle.children[1].text[:10] in bundle.children[0].text[-50:]


def test_long_paragraph_is_hard_split() -> None:
    bundle = _build(_parsed([Block("A" * 700)]), parent_chars=300, child_chars=100, child_overlap=20)

    assert len(bundle.parents) == 3
    assert all(parent.char_count <= 300 for parent in bundle.parents)


def test_chunk_ids_are_deterministic() -> None:
    parsed = _parsed([Block("固定内容。")])

    first = _build(parsed)
    second = _build(parsed)

    assert [chunk.id for chunk in first.parents] == [chunk.id for chunk in second.parents]
    assert [chunk.id for chunk in first.children] == [chunk.id for chunk in second.children]


def test_page_numbers_are_inherited() -> None:
    parsed = ParsedDocument(ext="pdf", blocks=(Block("第三页正文。", page=3),), page_count=3)

    bundle = _build(parsed)

    assert bundle.parents[0].page == 3
    assert bundle.children[0].page == 3


def test_child_ordinals_restart_per_parent() -> None:
    parsed = _parsed(
        [
            Block("第一章 甲", heading_level=1),
            Block("甲" * 180),
            Block("第二章 乙", heading_level=1),
            Block("乙" * 180),
        ]
    )

    bundle = _build(parsed, parent_chars=300, child_chars=100, child_overlap=20)

    assert [parent.ordinal for parent in bundle.parents] == [1, 2]
    assert [child.ordinal for child in bundle.children] == [1, 2, 1, 2]


def test_empty_document_raises() -> None:
    with pytest.raises(EmptyDocumentError):
        _build(_parsed([]))


def test_whitespace_only_blocks_raise() -> None:
    with pytest.raises(EmptyDocumentError):
        _build(_parsed([Block("   ")]))
