"""Chunking strategy.

Goal: chunks large enough to carry a complete idea (a definition plus the
paragraph that explains it), small enough that an embedding stays specific, and
overlapping enough that a concept split across a boundary is still retrievable.

Method - recursive, structure-aware packing:
  1. split a page into paragraphs (blank-line boundaries),
  2. pack paragraphs into a window up to `chunk_size` characters,
  3. if a single paragraph exceeds the window, fall back to sentence packing,
  4. carry `chunk_overlap` characters of the previous window into the next one.

Every chunk records the page range it came from, so retrieved context can be
cited back to "Mitchell, p. 54" in the UI.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.rag.loaders import LoadedDocument

_PARAGRAPH_RE = re.compile(r"\n\s*\n")
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
# Headings in these books look like "3.4 Decision Tree Learning" or ALL CAPS lines.
_HEADING_RE = re.compile(r"^(?:\d+(?:\.\d+)*\s+[A-Z][\w \-,()']{3,80}|[A-Z][A-Z \-]{6,60})$")
_MIN_CHUNK_CHARS = 220


@dataclass
class Chunk:
    text: str
    page_start: int
    page_end: int
    section: str = ""
    ordinal: int = 0
    metadata: dict = field(default_factory=dict)


def _split_paragraphs(text: str) -> list[str]:
    parts = [p.strip() for p in _PARAGRAPH_RE.split(text) if p.strip()]
    return parts or ([text.strip()] if text.strip() else [])


def _split_long(paragraph: str, limit: int) -> list[str]:
    sentences = _SENTENCE_RE.split(paragraph)
    out: list[str] = []
    buf = ""
    for sentence in sentences:
        if len(buf) + len(sentence) + 1 > limit and buf:
            out.append(buf.strip())
            buf = ""
        if len(sentence) > limit:
            for i in range(0, len(sentence), limit):
                out.append(sentence[i : i + limit].strip())
            continue
        buf = f"{buf} {sentence}".strip()
    if buf:
        out.append(buf.strip())
    return [o for o in out if o]


def _detect_heading(paragraph: str, current: str) -> str:
    first_line = paragraph.split("\n", 1)[0].strip()
    if _HEADING_RE.match(first_line):
        return first_line[:120]
    return current


def _tail_overlap(text: str, overlap: int) -> str:
    if overlap <= 0:
        return ""
    if len(text) <= overlap:
        return text
    tail = text[-overlap:]
    # start the overlap at a sentence boundary when there is one nearby
    match = _SENTENCE_RE.search(tail)
    return tail[match.end() :] if match else tail


def chunk_document(
    document: LoadedDocument,
    *,
    chunk_size: int = 1400,
    chunk_overlap: int = 220,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    buf = ""
    buf_start_page = document.pages[0].number if document.pages else 1
    buf_end_page = buf_start_page
    section = ""

    def flush() -> None:
        nonlocal buf
        text = buf.strip()
        if len(text) >= _MIN_CHUNK_CHARS:
            chunks.append(
                Chunk(
                    text=text,
                    page_start=buf_start_page,
                    page_end=buf_end_page,
                    section=section,
                    ordinal=len(chunks),
                )
            )
        elif text and chunks:
            # too small to stand alone: fold it into the previous chunk
            chunks[-1].text = f"{chunks[-1].text}\n{text}"
            chunks[-1].page_end = buf_end_page
        buf = ""

    for page in document.pages:
        for paragraph in _split_paragraphs(page.text):
            section = _detect_heading(paragraph, section)
            pieces = (
                [paragraph]
                if len(paragraph) <= chunk_size
                else _split_long(paragraph, chunk_size)
            )
            for piece in pieces:
                if buf and len(buf) + len(piece) + 1 > chunk_size:
                    carry = _tail_overlap(buf, chunk_overlap)
                    flush()
                    buf = carry
                    buf_start_page = page.number
                if not buf:
                    buf_start_page = page.number
                buf = f"{buf}\n{piece}".strip() if buf else piece
                buf_end_page = page.number
    flush()
    return chunks
