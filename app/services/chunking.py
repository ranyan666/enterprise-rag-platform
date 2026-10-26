"""父子分块：父块保留完整段落，子块用于向量召回。

父块按标题层级与段落边界切分，子块在父块内部滑窗切分并带重叠；
子块通过 ``parent_id`` 指回父块，检索命中后可由父块提供完整上下文。
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Iterator

from app.services.parsing import EmptyDocumentError, ParsedDocument

SENTENCE_ENDINGS = "。！？；.!?;\n"
MIN_CHILD_CHARS = 50
BOUNDARY_LOOKBACK = 100
SECTION_SEPARATOR = " > "


@dataclass(frozen=True, slots=True)
class ChunkData:
    id: str
    parent_id: str | None
    level: str
    ordinal: int
    text: str
    char_count: int
    page: int | None
    section: str | None


@dataclass(frozen=True, slots=True)
class ChunkBundle:
    parents: tuple[ChunkData, ...]
    children: tuple[ChunkData, ...]


def _chunk_id(doc_id: str, *parts: object) -> str:
    key = "rag://" + doc_id + "/" + "/".join(str(part) for part in parts)
    return str(uuid.uuid5(uuid.NAMESPACE_URL, key))


def _iter_sections(parsed: ParsedDocument) -> Iterator[tuple[str | None, list[tuple[str, int | None]]]]:
    """把块序列按标题栈切分为 section，段落携带页码。"""

    stack: list[tuple[int, str]] = []
    paragraphs: list[tuple[str, int | None]] = []
    section: str | None = None

    for block in parsed.blocks:
        if block.heading_level:
            if paragraphs:
                yield section, paragraphs
                paragraphs = []
            while stack and stack[-1][0] >= block.heading_level:
                stack.pop()
            stack.append((block.heading_level, block.text.split("\n", 1)[0].strip()))
            section = SECTION_SEPARATOR.join(title for _, title in stack) or None
            continue
        text = block.text.strip()
        if text:
            paragraphs.append((text, block.page))

    if paragraphs:
        yield section, paragraphs


def _split_parents(
    paragraphs: list[tuple[str, int | None]],
    limit: int,
) -> list[tuple[str, int | None]]:
    parents: list[tuple[str, int | None]] = []
    buffer: list[str] = []
    size = 0
    page: int | None = None

    def flush() -> None:
        nonlocal buffer, size, page
        text = "\n".join(buffer).strip()
        if text:
            parents.append((text, page))
        buffer = []
        size = 0
        page = None

    for paragraph, paragraph_page in paragraphs:
        if len(paragraph) > limit:
            flush()
            for start in range(0, len(paragraph), limit):
                piece = paragraph[start : start + limit].strip()
                if piece:
                    parents.append((piece, paragraph_page))
            continue
        addition = len(paragraph) + (1 if buffer else 0)
        if buffer and size + addition > limit:
            flush()
        if page is None:
            page = paragraph_page
        buffer.append(paragraph)
        size += addition

    flush()
    return parents


def _split_children(text: str, size: int, overlap: int) -> list[str]:
    if len(text) <= size:
        return [text]

    pieces: list[str] = []
    start = 0
    length = len(text)
    while start < length:
        end = min(start + size, length)
        if end < length:
            lower_bound = max(start + size - BOUNDARY_LOOKBACK, start + 1)
            for index in range(end, lower_bound - 1, -1):
                if text[index - 1] in SENTENCE_ENDINGS:
                    end = index
                    break
        if end <= start:
            end = min(start + size, length)
        piece = text[start:end].strip()
        if piece:
            pieces.append(piece)
        if end >= length:
            break
        start = max(end - overlap, start + 1)

    merged: list[str] = []
    for piece in pieces:
        if merged and len(piece) < MIN_CHILD_CHARS:
            merged[-1] = f"{merged[-1]}\n{piece}"
        else:
            merged.append(piece)
    return merged


def build_chunks(
    parsed: ParsedDocument,
    *,
    doc_id: str,
    parent_chars: int,
    child_chars: int,
    child_overlap: int,
) -> ChunkBundle:
    """切出父子块；没有任何正文时抛 ``EmptyDocumentError``。"""

    parents: list[ChunkData] = []
    children: list[ChunkData] = []
    parent_ordinal = 0

    for section, paragraphs in _iter_sections(parsed):
        for parent_text, page in _split_parents(paragraphs, parent_chars):
            parent_ordinal += 1
            parent_id = _chunk_id(doc_id, "parent", parent_ordinal)
            parents.append(
                ChunkData(
                    id=parent_id,
                    parent_id=None,
                    level="parent",
                    ordinal=parent_ordinal,
                    text=parent_text,
                    char_count=len(parent_text),
                    page=page,
                    section=section,
                )
            )
            for child_ordinal, child_text in enumerate(
                _split_children(parent_text, child_chars, child_overlap),
                start=1,
            ):
                children.append(
                    ChunkData(
                        id=_chunk_id(doc_id, "child", parent_ordinal, child_ordinal),
                        parent_id=parent_id,
                        level="child",
                        ordinal=child_ordinal,
                        text=child_text,
                        char_count=len(child_text),
                        page=page,
                        section=section,
                    )
                )

    if not parents:
        raise EmptyDocumentError("未切分出有效文本块")
    return ChunkBundle(parents=tuple(parents), children=tuple(children))
