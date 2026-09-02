"""Retrieval.

Three ideas, in order of importance:

1. **Hybrid scoring.** Dense embeddings capture paraphrase ("regularisation" ~
   "penalising complexity"); BM25 captures the exact technical token a resume
   mentions ("XGBoost", "EM algorithm"). Textbook retrieval needs both, so the
   two scores are min-max normalised per query and blended with `dense_weight`.

2. **MMR diversity.** Books repeat themselves: a top-k by score alone returns
   five near-copies of the same paragraph, which produces five near-identical
   interview questions. Maximal Marginal Relevance trades a little relevance for
   coverage, which is what a multi-question interview actually needs.

3. **Traceability.** Every result carries its source, page range and both raw
   scores, and `RetrievalResult` is persisted with the question that used it.
"""

from __future__ import annotations

import logging
import math
import re
from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from app.core.config import Settings, get_settings
from app.rag.embeddings import Embedder, get_embedder
from app.rag.vector_store import NumpyVectorStore, ScoredChunk, get_vector_store

logger = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "is", "are", "for", "on", "with",
    "that", "this", "it", "as", "be", "by", "we", "you", "can", "how", "what", "which",
    "when", "from", "at", "if", "then", "than", "so", "such", "these", "those", "its",
}


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOPWORDS and len(t) > 1]


@dataclass
class RetrievalResult:
    """What retrieval returned for one query - persisted as the question's trace."""

    query: str
    chunks: list[ScoredChunk] = field(default_factory=list)

    def context_block(self, max_chars: int = 6000) -> str:
        """Numbered, cited passages, ready to drop into a prompt."""
        parts: list[str] = []
        budget = max_chars
        for index, chunk in enumerate(self.chunks, start=1):
            body = chunk.record.text[: max(budget, 0)]
            if not body:
                break
            header = f"[{index}] {chunk.citation()}"
            if chunk.record.section:
                header += f" - section: {chunk.record.section}"
            block = f"{header}\n{body}"
            parts.append(block)
            budget -= len(block)
            if budget <= 0:
                break
        return "\n\n".join(parts)

    def trace(self) -> dict:
        return {
            "query": self.query,
            "chunks": [
                {
                    "id": c.record.id,
                    "source": c.record.source,
                    "citation": c.citation(),
                    "section": c.record.section,
                    "page_start": c.record.page_start,
                    "page_end": c.record.page_end,
                    "score": round(c.score, 4),
                    "dense_score": round(c.dense_score, 4),
                    "lexical_score": round(c.lexical_score, 4),
                    "excerpt": c.record.text[:400],
                }
                for c in self.chunks
            ],
        }


class BM25Index:
    """Compact BM25 (k1=1.5, b=0.75) over a collection's chunk texts."""

    def __init__(self, texts: list[str], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        self.docs = [Counter(tokenize(t)) for t in texts]
        self.lengths = np.array([sum(d.values()) for d in self.docs], dtype=np.float32)
        self.avg_length = float(self.lengths.mean()) if len(self.lengths) else 0.0
        n = len(self.docs)
        df = Counter()
        for doc in self.docs:
            df.update(doc.keys())
        self.idf = {
            term: math.log(1 + (n - count + 0.5) / (count + 0.5)) for term, count in df.items()
        }
        self.postings: dict[str, list[int]] = {}
        for index, doc in enumerate(self.docs):
            for term in doc:
                self.postings.setdefault(term, []).append(index)

    def scores(self, query: str) -> np.ndarray:
        out = np.zeros(len(self.docs), dtype=np.float32)
        if not self.avg_length:
            return out
        for term in set(tokenize(query)):
            idf = self.idf.get(term)
            if idf is None:
                continue
            for index in self.postings.get(term, ()):
                freq = self.docs[index][term]
                length = self.lengths[index]
                denominator = freq + self.k1 * (1 - self.b + self.b * length / self.avg_length)
                out[index] += idf * (freq * (self.k1 + 1)) / denominator
        return out


def _minmax(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values
    low, high = float(values.min()), float(values.max())
    if high - low < 1e-9:
        return np.zeros_like(values)
    return (values - low) / (high - low)


class HybridRetriever:
    """Dense + BM25 fusion with MMR re-ranking, scoped to one role collection."""

    def __init__(
        self,
        *,
        settings: Settings | None = None,
        store: NumpyVectorStore | None = None,
        embedder: Embedder | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.store = store or get_vector_store()
        self.embedder = embedder or get_embedder(self.settings)
        self._bm25: dict[str, BM25Index] = {}

    def _bm25_index(self, collection: str) -> BM25Index:
        if collection not in self._bm25:
            records = self.store.records(collection)
            self._bm25[collection] = BM25Index([r.text for r in records])
        return self._bm25[collection]

    def retrieve(
        self,
        collection: str,
        query: str,
        *,
        top_k: int | None = None,
        exclude_ids: set[str] | None = None,
    ) -> RetrievalResult:
        top_k = top_k or self.settings.retrieval_top_k
        candidate_k = max(self.settings.retrieval_candidate_k, top_k * 4)
        records = self.store.records(collection)
        if not records:
            return RetrievalResult(query=query, chunks=[])

        vectors = self.store._load(collection)[0]  # noqa: SLF001 - same package, hot path
        query_vector = self.embedder.embed([query], task="query")[0]
        dense = vectors @ query_vector
        lexical = self._bm25_index(collection).scores(query)

        blended = (
            self.settings.dense_weight * _minmax(dense)
            + (1 - self.settings.dense_weight) * _minmax(lexical)
        )
        if exclude_ids:
            for index, record in enumerate(records):
                if record.id in exclude_ids:
                    blended[index] = -1.0

        limit = min(candidate_k, blended.shape[0])
        candidates = np.argpartition(-blended, limit - 1)[:limit]
        candidates = candidates[np.argsort(-blended[candidates])]
        # excluded chunks were pushed to -1; everything else stays eligible, even
        # when min-max normalisation flattens a small candidate pool to zero.
        candidates = [int(i) for i in candidates if blended[i] >= 0]

        selected = self._mmr(candidates, vectors, blended, top_k)
        chunks = [
            ScoredChunk(
                record=records[i],
                score=float(blended[i]),
                dense_score=float(dense[i]),
                lexical_score=float(lexical[i]),
            )
            for i in selected
        ]
        logger.debug("retrieved %s chunks for %r from %s", len(chunks), query[:60], collection)
        return RetrievalResult(query=query, chunks=chunks)

    def _mmr(
        self,
        candidates: list[int],
        vectors: np.ndarray,
        relevance: np.ndarray,
        top_k: int,
    ) -> list[int]:
        """Greedy Maximal Marginal Relevance over the candidate pool."""
        if not candidates:
            return []
        lambda_ = self.settings.retrieval_mmr_lambda
        selected: list[int] = [candidates[0]]
        pool = candidates[1:]
        while pool and len(selected) < top_k:
            selected_matrix = vectors[selected]
            best_index, best_value = None, -math.inf
            for index in pool:
                redundancy = float(np.max(selected_matrix @ vectors[index]))
                value = lambda_ * float(relevance[index]) - (1 - lambda_) * redundancy
                if value > best_value:
                    best_index, best_value = index, value
            selected.append(best_index)  # type: ignore[arg-type]
            pool.remove(best_index)  # type: ignore[arg-type]
        return selected

    def retrieve_many(
        self,
        collection: str,
        queries: list[str],
        *,
        per_query_k: int | None = None,
    ) -> list[RetrievalResult]:
        """Retrieve for several queries, never returning the same chunk twice."""
        seen: set[str] = set()
        results: list[RetrievalResult] = []
        for query in queries:
            result = self.retrieve(collection, query, top_k=per_query_k, exclude_ids=seen)
            seen.update(c.record.id for c in result.chunks)
            results.append(result)
        return results
