"""Knowledge ingestion pipeline.

    corpus files -> load -> clean -> chunk -> quality filter -> embed -> index

Run through `backend/scripts/ingest_kb.py`. Ingestion is idempotent per role: the
collection is rebuilt from whatever currently sits in
`data/knowledge_base/<role>/`, so re-running after adding a book is safe.

The quality filter matters more than it looks. Textbook PDFs carry indexes,
bibliographies and equation soup that extract as noise; embedding them wastes
API budget and, worse, they surface as "context" and produce nonsense questions.
"""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from app.core.config import Settings, get_settings
from app.db.models import KnowledgeDocument
from app.db.session import session_scope
from app.rag.chunking import Chunk, chunk_document
from app.rag.embeddings import Embedder, get_embedder
from app.rag.loaders import iter_corpus_files, load_path
from app.rag.vector_store import ChunkRecord, NumpyVectorStore, get_vector_store

logger = logging.getLogger(__name__)

_WORD_RE = re.compile(r"[A-Za-z]{2,}")
_MIN_WORDS = 40
_MIN_ALPHA_RATIO = 0.55
# Lines that are mostly dot leaders / page numbers: tables of contents and indexes.
_TOC_RE = re.compile(r"\.{4,}\s*\d+|\b\d+\s*[-–]\s*\d+,\s*\d+")


@dataclass
class IngestReport:
    role: str
    documents: int = 0
    pages: int = 0
    chunks_seen: int = 0
    chunks_indexed: int = 0
    sources: list[str] | None = None

    def as_dict(self) -> dict:
        return {
            "role": self.role,
            "documents": self.documents,
            "pages": self.pages,
            "chunks_seen": self.chunks_seen,
            "chunks_indexed": self.chunks_indexed,
            "sources": self.sources or [],
        }


def is_useful_chunk(text: str) -> bool:
    """Reject front matter, indexes, reference lists and OCR/equation noise."""
    words = _WORD_RE.findall(text)
    if len(words) < _MIN_WORDS:
        return False
    alpha = sum(c.isalpha() or c.isspace() for c in text)
    if alpha / max(len(text), 1) < _MIN_ALPHA_RATIO:
        return False
    if len(_TOC_RE.findall(text)) >= 3:
        return False
    # bibliography blocks: many "Surname, A. (1998)" patterns
    if len(re.findall(r"\(\d{4}\)", text)) >= 4:
        return False
    return True


def _chunk_id(source: str, chunk: Chunk) -> str:
    raw = f"{source}:{chunk.page_start}:{chunk.ordinal}:{chunk.text[:120]}"
    return hashlib.blake2b(raw.encode("utf-8"), digest_size=12).hexdigest()


class IngestionPipeline:
    def __init__(
        self,
        *,
        settings: Settings | None = None,
        embedder: Embedder | None = None,
        store: NumpyVectorStore | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.embedder = embedder or get_embedder(self.settings)
        self.store = store or get_vector_store()

    def ingest_role(
        self,
        role_slug: str,
        *,
        max_pages_per_doc: int | None = None,
        skip_leading_pages: int = 0,
        progress: bool = True,
    ) -> IngestReport:
        directory = self.settings.knowledge_base_dir / role_slug
        files = iter_corpus_files(directory)
        report = IngestReport(role=role_slug, sources=[])
        if not files:
            logger.warning("no corpus files found in %s", directory)
            return report

        records: list[ChunkRecord] = []
        texts: list[str] = []
        doc_rows: list[KnowledgeDocument] = []

        for path in files:
            document = load_path(path)
            pages = document.pages[skip_leading_pages:]
            if max_pages_per_doc:
                pages = pages[:max_pages_per_doc]
            document.pages = pages
            chunks = chunk_document(
                document,
                chunk_size=self.settings.chunk_size_chars,
                chunk_overlap=self.settings.chunk_overlap_chars,
            )
            kept = [c for c in chunks if is_useful_chunk(c.text)]
            for chunk in kept:
                records.append(
                    ChunkRecord(
                        id=_chunk_id(path.name, chunk),
                        text=chunk.text,
                        source=path.name,
                        page_start=chunk.page_start,
                        page_end=chunk.page_end,
                        section=chunk.section,
                        metadata={"role": role_slug},
                    )
                )
                texts.append(chunk.text)
            report.documents += 1
            report.pages += len(pages)
            report.chunks_seen += len(chunks)
            report.chunks_indexed += len(kept)
            report.sources.append(path.name)  # type: ignore[union-attr]
            doc_rows.append(
                KnowledgeDocument(
                    role_slug=role_slug,
                    filename=path.name,
                    content_hash=hashlib.blake2b(
                        path.read_bytes(), digest_size=16
                    ).hexdigest(),
                    pages=len(pages),
                    chunks=len(kept),
                )
            )
            if progress:
                logger.info(
                    "%s: %s pages -> %s chunks (%s kept)",
                    path.name,
                    len(pages),
                    len(chunks),
                    len(kept),
                )

        if not records:
            logger.warning("nothing worth indexing for role %s", role_slug)
            return report

        vectors = self._embed_all(texts, progress=progress)
        self.store.upsert(role_slug, records, vectors)

        with session_scope() as db:
            db.query(KnowledgeDocument).filter(
                KnowledgeDocument.role_slug == role_slug
            ).delete()
            db.add_all(doc_rows)
        return report

    def _embed_all(self, texts: list[str], *, progress: bool) -> np.ndarray:
        window_size = max(self.settings.embedding_batch_size, 1) * 4
        chunks_out: list[np.ndarray] = []
        total = len(texts)
        for start in range(0, total, window_size):
            window = texts[start : start + window_size]
            chunks_out.append(self.embedder.embed(window, task="document"))
            if progress:
                done = min(start + len(window), total)
                logger.info("embedded %s/%s chunks (%.0f%%)", done, total, 100 * done / total)
        return np.vstack(chunks_out)


def corpus_sections(role_slug: str, limit: int = 60) -> list[str]:
    """Distinct section headings in a role's index - used to ground interview planning."""
    store = get_vector_store()
    try:
        records = store.records(role_slug)
    except Exception:
        return []
    seen: dict[str, int] = {}
    for record in records:
        section = (record.section or "").strip()
        if len(section) > 6:
            seen[section] = seen.get(section, 0) + 1
    ranked = sorted(seen.items(), key=lambda kv: -kv[1])
    return [s for s, _ in ranked[:limit]]
