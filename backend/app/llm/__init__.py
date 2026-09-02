"""LLM package: interface, implementations and the factory used by services."""

from __future__ import annotations

import logging

from app.core.config import Settings, get_settings
from app.llm.base import LLMClient
from app.llm.gemini import GeminiClient
from app.llm.stub import StubLLM

logger = logging.getLogger(__name__)

_client: LLMClient | None = None


def get_llm(settings: Settings | None = None) -> LLMClient:
    global _client
    settings = settings or get_settings()
    if _client is None:
        if settings.llm_enabled:
            _client = GeminiClient(settings)
        else:
            logger.warning("No GEMINI_API_KEY / offline mode: using the stub LLM.")
            _client = StubLLM()
    return _client


def reset_llm() -> None:
    """Test hook."""
    global _client
    _client = None


__all__ = ["LLMClient", "GeminiClient", "StubLLM", "get_llm", "reset_llm"]
