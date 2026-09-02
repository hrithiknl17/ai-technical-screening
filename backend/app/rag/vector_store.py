"""Vector store.

Interface first, implementation second - the retriever only ever sees
`VectorStore`, so swapping in Chroma/Qdrant/pgvector later is a one-class change.

The shipped implementation is a persistent, per-role flat index:

    data/vector_store/<role>/vectors.npy   float32 [n, dim], L2-normalised
    data/vector_store/<role>/records.jsonl chunk text + provenance, aligned by row
    data/vector_store/<role>/meta.json     embedder, dimension, counts, timestamps

Why flat and not an ANN index: the corpora here are books (10^3-10^4 chunks).
A normalised matrix-vector product over 10k x 768 floats is ~7M FLOPs - well
under a millisecond in NumPy, and it is exact. An ANN index would add a
dependency and an approximation for no measurable win at this scale. The
partitioning that does matter (per role) is done explicitly, because a question
for one role must never retrieve another role's corpus.
"""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

import numpy as np

from app.core.errors import KnowledgeBaseError

logger = logging.getLogger(__name__)


@dataclass
class ChunkRecord:
    """One indexed passage plus everything needed to cite it."""

    id: str
    text: str
    source: str
    page_start: int
    page_end: int
    section: str = ""
    metadata: dict = field(default_factory=dict)


@dataclass
class ScoredChunk:
    record: ChunkRecord
    score: float
    dense_score: float = 0.0
    lexical_score: float = 0.0

    def citation(self) -> str:
        pages = (
            f"p. {self.record.page_start}"
            if self.record.page_start == self.record.page_end
            else f"pp. {self.record.page_start}-{self.record.page_end}"
        )
        return f"{self.record.source}, {pages}"


class VectorStore(Protocol):
    def upsert(self, collection: str, records: list[ChunkRecord], vectors: np.ndarray) -> None: ...
    def search(self, collection: str, vector: np.ndarray, top_k: int) -> list[ScoredChunk]: ...
    def records(self, collection: str) -> list[ChunkRecord]: ...
    def count(self, collection: str) -> int: ...
    def collections(self) -> list[str]: ...
    def drop(self, collection: str) -> None: ...


class NumpyVectorStore:
    """Persistent flat-index store, one collection (= role) per directory."""

    def __init__(self, root: Path) -> None:
        self._root = Path(root)
        self._root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._cache: dict[str, tuple[np.ndarray, list[ChunkRecord]]] = {}

    # --- paths -------------------------------------------------------------
    def _dir(self, collection: str) -> Path:
        return self._root / collection

    def _paths(self, collection: str) -> tuple[Path, Path, Path]:
        base = self._dir(collection)
        return base / "vectors.npy", base / "records.jsonl", base / "meta.json"

    # --- io ----------------------------------------------------------------
    def _load(self, collection: str) -> tuple[np.ndarray, list[ChunkRecord]]:
        with self._lock:
            if collection in self._cache:
                return self._cache[collection]
            vectors_path, records_path, _ = self._paths(collection)
            if not vectors_path.exists() or not records_path.exists():
                raise KnowledgeBaseError(
                    f"No knowledge base indexed for '{collection}'. "
                    "Run `python -m scripts.ingest_kb --role " + collection + "` first.",
                    details={"collection": collection},
                )
            vectors = np.load(vectors_path).astype(np.float32)
            records = [
                ChunkRecord(**json.loads(line))
                for line in records_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            if len(records) != vectors.shape[0]:
                raise KnowledgeBaseError(
                    f"Index for '{collection}' is corrupt: "
                    f"{vectors.shape[0]} vectors vs {len(records)} records. Re-ingest."
                )
            self._cache[collection] = (vectors, records)
            return vectors, records

    def upsert(self, collection: str, records: list[ChunkRecord], vectors: np.ndarray) -> None:
        """Replace the collection contents (ingestion is idempotent by design)."""
        if vectors.shape[0] != len(records):
            raise ValueError("vectors and records must be the same length")
        with self._lock:
            base = self._dir(collection)
            base.mkdir(parents=True, exist_ok=True)
            vectors_path, records_path, meta_path = self._paths(collection)
            np.save(vectors_path, vectors.astype(np.float32))
            with records_path.open("w", encoding="utf-8") as handle:
                for record in records:
                    handle.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")
            meta = {
                "collection": collection,
                "count": len(records),
                "dimension": int(vectors.shape[1]) if vectors.size else 0,
                "sources": sorted({r.source for r in records}),
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }
            meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
            self._cache.pop(collection, None)
            logger.info("indexed %s chunks into collection '%s'", len(records), collection)

    def search(self, collection: str, vector: np.ndarray, top_k: int) -> list[ScoredChunk]:
        vectors, records = self._load(collection)
        query = np.asarray(vector, dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(query))
        if norm:
            query = query / norm
        if query.shape[0] != vectors.shape[1]:
            raise KnowledgeBaseError(
                f"Query dimension {query.shape[0]} does not match index dimension "
                f"{vectors.shape[1]} for '{collection}'. Re-ingest with the current embedder."
            )
        scores = vectors @ query
        k = min(top_k, scores.shape[0])
        top = np.argpartition(-scores, k - 1)[:k] if k < scores.shape[0] else np.arange(k)
        top = top[np.argsort(-scores[top])]
        return [
            ScoredChunk(record=records[i], score=float(scores[i]), dense_score=float(scores[i]))
            for i in top
        ]

    def records(self, collection: str) -> list[ChunkRecord]:
        return self._load(collection)[1]

    def count(self, collection: str) -> int:
        try:
            return len(self._load(collection)[1])
        except KnowledgeBaseError:
            return 0

    def meta(self, collection: str) -> dict:
        _, _, meta_path = self._paths(collection)
        if not meta_path.exists():
            return {"collection": collection, "count": 0, "sources": []}
        return json.loads(meta_path.read_text(encoding="utf-8"))

    def collections(self) -> list[str]:
        return sorted(p.name for p in self._root.iterdir() if p.is_dir()) if self._root.exists() else []

    def drop(self, collection: str) -> None:
        with self._lock:
            base = self._dir(collection)
            for path in base.glob("*"):
                path.unlink()
            if base.exists():
                base.rmdir()
            self._cache.pop(collection, None)


_store: NumpyVectorStore | None = None


def get_vector_store() -> NumpyVectorStore:
    global _store
    if _store is None:
        from app.core.config import get_settings

        _store = NumpyVectorStore(get_settings().vector_store_dir)
    return _store


def reset_vector_store() -> None:
    """Test hook."""
    global _store
    _store = None
