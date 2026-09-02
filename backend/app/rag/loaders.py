"""Document loading.

Everything that enters the pipeline - corpus books and candidate resumes - is
normalised into `LoadedDocument` (per-page text plus provenance) so downstream
stages never care whether the source was a PDF, Markdown or plain text.
"""

from __future__ import annotations

import io
import re
from dataclasses import dataclass, field
from pathlib import Path

from pypdf import PdfReader

from app.core.errors import ValidationError

SUPPORTED_SUFFIXES = {".pdf", ".txt", ".md"}

_LIGATURES = {"ﬀ": "ff", "ﬁ": "fi", "ﬂ": "fl", "ﬃ": "ffi", "ﬄ": "ffl"}


@dataclass
class LoadedPage:
    number: int
    text: str


@dataclass
class LoadedDocument:
    source: str
    pages: list[LoadedPage] = field(default_factory=list)

    @property
    def text(self) -> str:
        return "\n\n".join(p.text for p in self.pages)


def clean_text(text: str) -> str:
    """Undo the usual PDF extraction damage: ligatures, hyphenation, ragged whitespace."""
    for bad, good in _LIGATURES.items():
        text = text.replace(bad, good)
    text = text.replace("­", "")
    # join words split across a line break: "gener-\nalisation" -> "generalisation"
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    text = re.sub(r"[ \t\x0b\f\r]+", " ", text)
    text = re.sub(r" ?\n ?", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def load_pdf_bytes(data: bytes, source: str) -> LoadedDocument:
    try:
        reader = PdfReader(io.BytesIO(data))
    except Exception as exc:  # pragma: no cover - depends on the file
        raise ValidationError(f"Could not read PDF '{source}': {exc}") from exc
    pages: list[LoadedPage] = []
    for index, page in enumerate(reader.pages, start=1):
        try:
            text = clean_text(page.extract_text() or "")
        except Exception:
            text = ""
        if text:
            pages.append(LoadedPage(number=index, text=text))
    if not pages:
        raise ValidationError(
            f"No extractable text in '{source}'. Scanned PDFs need OCR before ingestion."
        )
    return LoadedDocument(source=source, pages=pages)


def load_text_bytes(data: bytes, source: str) -> LoadedDocument:
    text = clean_text(data.decode("utf-8", errors="replace"))
    if not text:
        raise ValidationError(f"'{source}' is empty.")
    return LoadedDocument(source=source, pages=[LoadedPage(number=1, text=text)])


def load_bytes(data: bytes, source: str, content_type: str | None = None) -> LoadedDocument:
    suffix = Path(source).suffix.lower()
    is_pdf = suffix == ".pdf" or (content_type or "").endswith("pdf") or data[:5] == b"%PDF-"
    if is_pdf:
        return load_pdf_bytes(data, source)
    if suffix in {"", ".txt", ".md"} or (content_type or "").startswith("text/"):
        return load_text_bytes(data, source)
    raise ValidationError(
        f"Unsupported file type '{suffix or content_type}'. Upload a PDF, TXT or MD file."
    )


def load_path(path: Path) -> LoadedDocument:
    if path.suffix.lower() not in SUPPORTED_SUFFIXES:
        raise ValidationError(f"Unsupported corpus file: {path.name}")
    return load_bytes(path.read_bytes(), path.name)


def iter_corpus_files(directory: Path) -> list[Path]:
    if not directory.exists():
        return []
    return sorted(
        p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES
    )
