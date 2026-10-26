"""多格式文档解析：PDF / DOCX / Markdown / 纯文本。

解析产物是统一的 ``Block`` 序列，标题层级与页码作为元数据向后传递，
供分块层切出可追溯的父/子块。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pymupdf
from docx import Document as DocxDocument

SUPPORTED_EXTENSIONS = frozenset({"pdf", "docx", "md", "txt"})
MAX_HEADING_LENGTH = 80
_TERMINAL_PUNCTUATION = ("。", "；", "，", "！", "？", ".", ";", ",", ":", "：")
_ZERO_WIDTH = re.compile(r"[\u200b-\u200d\ufeff]")
_MULTI_NEWLINE = re.compile(r"\n{3,}")
_MD_HEADING = re.compile(r"^(#{1,6})\s+(\S.*)$")
_CN_CHAPTER = re.compile(r"^第[一二三四五六七八九十百零〇\d]+[章篇部]\s*")
_CN_SECTION = re.compile(r"^第[一二三四五六七八九十百零〇\d]+节\s*")
_CN_ENUM = re.compile(r"^[一二三四五六七八九十]+[、.．]\s*")
_NUM_HEADING = re.compile(r"^(\d+(?:\.\d+){0,3})[、.．\s]\s*\S")


class DocumentParseError(Exception):
    """文档解析失败的基类。"""


class UnsupportedFormatError(DocumentParseError):
    """扩展名不在支持范围内。"""


class EmptyDocumentError(DocumentParseError):
    """解析后没有可用文本（空文档、扫描件等）。"""


class DocumentDecodeError(DocumentParseError):
    """文本编码无法识别。"""


@dataclass(frozen=True, slots=True)
class Block:
    text: str
    heading_level: int = 0
    page: int | None = None


@dataclass(frozen=True, slots=True)
class ParsedDocument:
    ext: str
    blocks: tuple[Block, ...]
    page_count: int


def normalize_text(text: str) -> str:
    """统一换行、去零宽字符与控制字符、压缩空行。"""

    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\x00", "")
    text = _ZERO_WIDTH.sub("", text)
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    return _MULTI_NEWLINE.sub("\n\n", text).strip()


def decode_text(raw: bytes) -> str:
    for encoding in ("utf-8-sig", "utf-8", "gb18030"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise DocumentDecodeError("无法识别文本编码（已尝试 utf-8 / gb18030）")


def detect_heading_level(text: str) -> int:
    """根据首行文本判断标题层级，0 表示正文。"""

    first_line = text.split("\n", 1)[0].strip()
    if not first_line or len(first_line) > MAX_HEADING_LENGTH:
        return 0
    if first_line.endswith(_TERMINAL_PUNCTUATION):
        return 0

    markdown = _MD_HEADING.match(first_line)
    if markdown:
        return len(markdown.group(1))
    if _CN_CHAPTER.match(first_line):
        return 1
    if _CN_SECTION.match(first_line):
        return 2
    if _CN_ENUM.match(first_line):
        return 2
    numeric = _NUM_HEADING.match(first_line)
    if numeric:
        return min(numeric.group(1).count(".") + 1, 6)
    return 0


def _docx_heading_level(style_name: str) -> int:
    name = (style_name or "").strip().lower()
    if name.startswith("heading"):
        suffix = name.replace("heading", "").strip()
        return int(suffix) if suffix.isdigit() else 1
    if name in {"title", "subtitle"}:
        return 1
    return 0


def _parse_pdf(path: Path) -> ParsedDocument:
    with path.open("rb") as handle:
        if not handle.read(5).startswith(b"%PDF-"):
            raise DocumentParseError("PDF 文件头不合法，可能不是真正的 PDF")

    blocks: list[Block] = []
    with pymupdf.open(str(path)) as doc:
        page_count = doc.page_count
        for page_index, page in enumerate(doc, start=1):
            for raw_block in page.get_text("blocks"):
                if len(raw_block) >= 7 and raw_block[6] != 0:  # 1 = 图片块
                    continue
                text = normalize_text(raw_block[4])
                if not text:
                    continue
                blocks.append(Block(text=text, heading_level=detect_heading_level(text), page=page_index))
    return ParsedDocument(ext="pdf", blocks=tuple(blocks), page_count=page_count)


def _parse_docx(path: Path) -> ParsedDocument:
    with path.open("rb") as handle:
        if not handle.read(4).startswith(b"PK\x03\x04"):
            raise DocumentParseError("DOCX 文件头不合法，可能不是真正的 DOCX")

    document = DocxDocument(str(path))
    blocks: list[Block] = []
    for paragraph in document.paragraphs:
        text = normalize_text(paragraph.text)
        if not text:
            continue
        style_name = paragraph.style.name if paragraph.style is not None else ""
        blocks.append(Block(text=text, heading_level=_docx_heading_level(style_name)))
    for table in document.tables:
        for row in table.rows:
            cells = [normalize_text(cell.text) for cell in row.cells]
            line = " | ".join(cell for cell in cells if cell)
            if line:
                blocks.append(Block(text=line))
    return ParsedDocument(ext="docx", blocks=tuple(blocks), page_count=0)


def _paragraph_blocks(text: str) -> list[Block]:
    blocks: list[Block] = []
    for raw_paragraph in re.split(r"\n\s*\n", text):
        paragraph = normalize_text(raw_paragraph)
        if paragraph:
            blocks.append(Block(text=paragraph, heading_level=0))
    return blocks


def _parse_markdown(text: str) -> ParsedDocument:
    blocks: list[Block] = []
    buffer: list[str] = []

    def flush() -> None:
        paragraph = normalize_text("\n".join(buffer))
        buffer.clear()
        if paragraph:
            blocks.append(Block(text=paragraph))

    for line in text.split("\n"):
        stripped = line.strip()
        heading = _MD_HEADING.match(stripped)
        if heading:
            flush()
            blocks.append(Block(text=heading.group(2).strip(), heading_level=len(heading.group(1))))
        elif not stripped:
            flush()
        else:
            buffer.append(line)
    flush()
    return ParsedDocument(ext="md", blocks=tuple(blocks), page_count=0)


def _parse_plain_text(text: str, ext: str) -> ParsedDocument:
    return ParsedDocument(ext=ext, blocks=tuple(_paragraph_blocks(text)), page_count=0)


def parse_document(path: Path, ext: str) -> ParsedDocument:
    """按扩展名解析文件；失败时抛出 ``DocumentParseError`` 子类。"""

    normalized_ext = (ext or "").lower().lstrip(".")
    if normalized_ext not in SUPPORTED_EXTENSIONS:
        raise UnsupportedFormatError(f"不支持的文件类型: {ext!r}")

    if normalized_ext == "pdf":
        parsed = _parse_pdf(path)
    elif normalized_ext == "docx":
        parsed = _parse_docx(path)
    else:
        text = decode_text(path.read_bytes())
        parsed = _parse_markdown(text) if normalized_ext == "md" else _parse_plain_text(text, normalized_ext)

    if not any(block.text.strip() for block in parsed.blocks):
        raise EmptyDocumentError("未解析到有效文本（可能是空文档或扫描件）")
    return parsed
