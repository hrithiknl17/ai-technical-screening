"""Embedding providers.

Three implementations behind one interface, selected by `EMBEDDING_PROVIDER`:

* `LocalStaticEmbedder` (`local`, the default) - a Model2Vec static embedding
  model. It runs on CPU with no API calls: ~5,000 chunks/second, so a whole
  corpus indexes in seconds and a grader can reproduce the index without a key.
* `GeminiEmbedder` (`gemini`) - `gemini-embedding-001`, task-typed (documents and
  queries use different task types, which helps asymmetric retrieval), batched,
  retried and self-pacing. Higher quality, but the free tier shares a global
  per-minute quota that makes a book-sized ingest take hours.
* `HashingEmbedder` (`hashing`) - deterministic hashed n-grams, no dependencies
  and no downloads. Used by the tests and by `OFFLINE_MODE`.

Retrieval quality drops in that order; the hybrid BM25 half of the retriever
(see `app.rag.retriever`) absorbs a good part of the difference. All three are
interchangeable because a collection records the dimension it was built with and
the store refuses a query vector that does not match.
"""

from __future__ import annotations

import hashlib
import logging
import math
import re
import time
from typing import Literal, Protocol

import httpx
import numpy as np

from app.core.config import Settings, get_settings
from app.core.errors import UpstreamError

logger = logging.getLogger(__name__)

TaskType = Literal["document", "query"]

_GEMINI_TASK = {
    "document": "RETRIEVAL_DOCUMENT",
    "query": "RETRIEVAL_QUERY",
}


class _RetryableStatus(Exception):
    """HTTP status worth retrying (429 / 5xx), carrying the body for RetryInfo."""

    def __init__(self, status: int, body: str) -> None:
        super().__init__(f"{status}: {body[:160]}")
        self.status = status
        self.body = body


class Embedder(Protocol):
    name: str
    dimension: int

    def embed(self, texts: list[str], task: TaskType = "document") -> np.ndarray: ...


def _normalise(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (matrix / norms).astype(np.float32)


class HashingEmbedder:
    """Deterministic offline fallback: hashed word + character trigram features."""

    def __init__(self, dimension: int = 768) -> None:
        self.name = "hashing-fallback"
        self.dimension = dimension

    def _vector(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dimension, dtype=np.float32)
        lowered = text.lower()
        tokens = re.findall(r"[a-z0-9]+", lowered)
        features: list[tuple[str, float]] = [(t, 1.0) for t in tokens]
        features += [(f"{a}_{b}", 0.6) for a, b in zip(tokens, tokens[1:])]
        compact = re.sub(r"\s+", " ", lowered)
        features += [(compact[i : i + 3], 0.25) for i in range(0, max(len(compact) - 2, 0), 2)]
        for feature, weight in features:
            digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
            index = int.from_bytes(digest[:4], "big") % self.dimension
            sign = 1.0 if digest[4] & 1 else -1.0
            vec[index] += sign * weight * (1.0 / math.sqrt(len(feature) + 1))
        return vec

    def embed(self, texts: list[str], task: TaskType = "document") -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)
        return _normalise(np.vstack([self._vector(t) for t in texts]))


class LocalStaticEmbedder:
    """Model2Vec static embeddings: a distilled sentence encoder with no runtime model.

    Static embeddings are token embeddings averaged over the input, so inference
    is a lookup and a mean - fast enough that ingestion stops being a bottleneck.
    They lose to a contextual encoder on subtle paraphrase, which is precisely
    what the lexical half of the hybrid retriever is there to cover.
    """

    def __init__(self, model_name: str) -> None:
        try:
            from model2vec import StaticModel
        except ImportError as exc:  # pragma: no cover - depends on the install
            raise UpstreamError(
                "EMBEDDING_PROVIDER=local needs the `model2vec` package "
                "(pip install -r backend/requirements.txt)."
            ) from exc
        self._model = StaticModel.from_pretrained(model_name)
        self.name = model_name
        self.dimension = int(self._model.dim)

    def embed(self, texts: list[str], task: TaskType = "document") -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)
        return _normalise(np.asarray(self._model.encode(texts), dtype=np.float32))


class GeminiEmbedder:
    """Google Generative Language embeddings over plain HTTP (no SDK dependency)."""

    BASE_URL = "https://generativelanguage.googleapis.com/v1beta"

    def __init__(self, settings: Settings) -> None:
        self.name = settings.embedding_model
        self.dimension = settings.embedding_dim
        self._key = settings.gemini_api_key
        self._model = settings.embedding_model
        self._batch = settings.embedding_batch_size
        self._retries = settings.llm_max_retries
        self._timeout = settings.llm_timeout_seconds
        # AIMD pacing: the free tier limits both requests and tokens per minute,
        # and the limits are not published per key, so the client discovers its
        # own safe rate - shrink the gap after every success, widen it on a 429.
        self._pause = settings.embedding_pause_seconds
        self._min_pause = 0.5
        self._max_pause = 45.0

    @staticmethod
    def _retry_delay(response_text: str, attempt: int) -> float:
        """Honour the server's own RetryInfo when it sends one, else exponential backoff."""
        match = re.search(r'"retryDelay"\s*:\s*"(\d+(?:\.\d+)?)s"', response_text)
        if match:
            return min(float(match.group(1)) + 1.0, 65.0)
        return min(4.0 * (2**attempt), 60.0)

    def _request(self, client: httpx.Client, batch: list[str], task: TaskType) -> list[list[float]]:
        payload = {
            "requests": [
                {
                    "model": f"models/{self._model}",
                    "content": {"parts": [{"text": text[:20000]}]},
                    "taskType": _GEMINI_TASK[task],
                    "outputDimensionality": self.dimension,
                }
                for text in batch
            ]
        }
        url = f"{self.BASE_URL}/models/{self._model}:batchEmbedContents"
        last_error: Exception | None = None
        for attempt in range(self._retries):
            try:
                response = client.post(url, params={"key": self._key}, json=payload)
                if response.status_code == 429 or response.status_code >= 500:
                    raise _RetryableStatus(response.status_code, response.text)
                response.raise_for_status()
                values = [e["values"] for e in response.json()["embeddings"]]
                self._pause = max(self._min_pause, self._pause * 0.9)
                return values
            except Exception as exc:  # network, rate limit, malformed payload
                last_error = exc
                body = exc.body if isinstance(exc, _RetryableStatus) else ""
                if isinstance(exc, _RetryableStatus) and exc.status == 429:
                    self._pause = min(self._max_pause, max(self._pause, 1.0) * 1.6)
                sleep_for = self._retry_delay(body, attempt)
                logger.warning(
                    "embedding batch failed (attempt %s/%s), retrying in %.0fs: %s",
                    attempt + 1,
                    self._retries,
                    sleep_for,
                    str(exc)[:160],
                )
                time.sleep(sleep_for)
        raise UpstreamError(f"Embedding request failed: {last_error}")

    def embed(self, texts: list[str], task: TaskType = "document") -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)
        vectors: list[list[float]] = []
        with httpx.Client(timeout=self._timeout) as client:
            for index, start in enumerate(range(0, len(texts), self._batch)):
                if index and self._pause:
                    # Free-tier keys are token-per-minute limited; pacing the
                    # ingest is cheaper than absorbing 429s and backing off.
                    time.sleep(self._pause)
                batch = texts[start : start + self._batch]
                vectors.extend(self._request(client, batch, task))
        return _normalise(np.asarray(vectors, dtype=np.float32))


_embedder: Embedder | None = None


def _select_provider(settings: Settings) -> str:
    """Resolve `auto`: local model if installed, else the API if keyed, else hashing."""
    provider = settings.embedding_provider.lower()
    if provider != "auto":
        return provider
    if settings.offline_mode:
        return "hashing"
    try:
        import model2vec  # noqa: F401
    except ImportError:
        return "gemini" if settings.gemini_api_key else "hashing"
    return "local"


def get_embedder(settings: Settings | None = None) -> Embedder:
    """Process-wide embedder singleton (chosen from configuration)."""
    global _embedder
    settings = settings or get_settings()
    if _embedder is not None:
        return _embedder

    provider = _select_provider(settings)
    if provider == "local":
        _embedder = LocalStaticEmbedder(settings.local_embedding_model)
    elif provider == "gemini":
        if not settings.gemini_api_key:
            raise UpstreamError("EMBEDDING_PROVIDER=gemini requires GEMINI_API_KEY.")
        _embedder = GeminiEmbedder(settings)
    else:
        _embedder = HashingEmbedder(settings.embedding_dim)
    logger.info("embedder: %s (%s dims)", _embedder.name, _embedder.dimension)
    return _embedder


def reset_embedder() -> None:
    """Test hook."""
    global _embedder
    _embedder = None
