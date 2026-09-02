"""Gemini client over the REST API.

No vendor SDK: one thin `httpx` wrapper keeps the dependency surface small and
makes retry/timeout behaviour explicit. `generate_json` uses Gemini's
`responseSchema` so the model is constrained to the shape the caller asked for,
which removes the usual "parse the JSON out of the prose" failure mode.
"""

from __future__ import annotations

import json
import logging
import random
import re
import threading
import time
from typing import Any

import httpx

from app.core.config import Settings, get_settings
from app.core.errors import UpstreamError

logger = logging.getLogger(__name__)

_JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)
_RETRY_DELAY_RE = re.compile(r'"retryDelay"\s*:\s*"(\d+(?:\.\d+)?)s"')


def _retry_after(body: str, attempt: int) -> float:
    """Free-tier limits are per minute; honour the server's RetryInfo when present."""
    match = _RETRY_DELAY_RE.search(body)
    if match:
        return min(float(match.group(1)) + 1.0, 65.0)
    return min(4.0 * (2**attempt), 60.0) + random.uniform(0, 0.5)


class _RateLimiter:
    """Process-wide minimum spacing between LLM calls.

    The free tier allows a fixed number of requests per minute; an interview
    fires several calls back to back (plan, question, grade, report), so without
    spacing the first burst spends its budget and every later call pays a
    60-second penalty. Spacing them costs less than absorbing the 429s.
    """

    def __init__(self, requests_per_minute: int) -> None:
        self._interval = 60.0 / max(requests_per_minute, 1)
        self._lock = threading.Lock()
        self._next_allowed = 0.0

    def acquire(self) -> None:
        with self._lock:
            now = time.monotonic()
            wait = self._next_allowed - now
            if wait > 0:
                time.sleep(wait)
                now = time.monotonic()
            self._next_allowed = now + self._interval


_limiters: dict[int, _RateLimiter] = {}
_limiters_lock = threading.Lock()


def _limiter_for(requests_per_minute: int) -> _RateLimiter:
    with _limiters_lock:
        if requests_per_minute not in _limiters:
            _limiters[requests_per_minute] = _RateLimiter(requests_per_minute)
        return _limiters[requests_per_minute]


class _RetryableStatus(Exception):
    """A 429/5xx worth retrying, carrying the body so RetryInfo can be read."""

    def __init__(self, status: int, body: str) -> None:
        super().__init__(f"{status}: {body[:160]}")
        self.status = status
        self.body = body


class GeminiClient:
    BASE_URL = "https://generativelanguage.googleapis.com/v1beta"

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.name = self.settings.llm_model
        self._key = self.settings.gemini_api_key
        self._limiter = _limiter_for(self.settings.llm_requests_per_minute)

    # --- transport ---------------------------------------------------------
    def _call(self, payload: dict) -> str:
        url = f"{self.BASE_URL}/models/{self.settings.llm_model}:generateContent"
        last_error: Exception | None = None
        for attempt in range(self.settings.llm_max_retries):
            try:
                self._limiter.acquire()
                with httpx.Client(timeout=self.settings.llm_timeout_seconds) as client:
                    response = client.post(url, params={"key": self._key}, json=payload)
                if response.status_code == 429 or response.status_code >= 500:
                    raise _RetryableStatus(response.status_code, response.text)
                response.raise_for_status()
                data = response.json()
                candidates = data.get("candidates") or []
                if not candidates:
                    raise UpstreamError(
                        "Model returned no candidates "
                        f"({data.get('promptFeedback', {}).get('blockReason', 'unknown reason')})."
                    )
                parts = candidates[0].get("content", {}).get("parts", [])
                text = "".join(part.get("text", "") for part in parts).strip()
                if not text:
                    raise UpstreamError("Model returned an empty response.")
                return text
            except UpstreamError:
                raise
            except Exception as exc:
                last_error = exc
                body = exc.body if isinstance(exc, _RetryableStatus) else ""
                backoff = _retry_after(body, attempt)
                logger.warning(
                    "LLM call failed (attempt %s/%s): %s",
                    attempt + 1,
                    self.settings.llm_max_retries,
                    exc,
                )
                time.sleep(backoff)
        raise UpstreamError(f"LLM request failed after retries: {last_error}")

    @staticmethod
    def _payload(
        prompt: str,
        system: str | None,
        temperature: float,
        schema: dict | None = None,
    ) -> dict:
        generation_config: dict[str, Any] = {"temperature": temperature}
        if schema is not None:
            generation_config["responseMimeType"] = "application/json"
            generation_config["responseSchema"] = schema
        payload: dict[str, Any] = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": generation_config,
        }
        if system:
            payload["systemInstruction"] = {"parts": [{"text": system}]}
        return payload

    # --- api ---------------------------------------------------------------
    def generate_text(
        self, prompt: str, *, system: str | None = None, temperature: float = 0.7
    ) -> str:
        return self._call(self._payload(prompt, system, temperature))

    def generate_json(
        self,
        prompt: str,
        schema: dict,
        *,
        system: str | None = None,
        temperature: float = 0.7,
    ) -> Any:
        raw = self._call(self._payload(prompt, system, temperature, schema))
        return parse_json(raw)


def parse_json(raw: str) -> Any:
    """Tolerant JSON parsing: schema-constrained output, fenced blocks, or prose."""
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    match = _JSON_BLOCK_RE.search(raw)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass
    for opener, closer in (("{", "}"), ("[", "]")):
        start, end = raw.find(opener), raw.rfind(closer)
        if start != -1 and end > start:
            try:
                return json.loads(raw[start : end + 1])
            except json.JSONDecodeError:
                continue
    raise UpstreamError("Model response was not valid JSON.", details={"raw": raw[:500]})
