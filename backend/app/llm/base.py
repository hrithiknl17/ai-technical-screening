"""LLM abstraction.

The services depend on `LLMClient`, never on a vendor SDK. Two implementations
exist: `GeminiClient` (real) and `StubLLM` (deterministic, offline). Structured
output is a first-class method because every call in this system - profile
extraction, planning, question generation, grading, reporting - wants JSON.
"""

from __future__ import annotations

from typing import Any, Protocol


class LLMClient(Protocol):
    name: str

    def generate_json(
        self,
        prompt: str,
        schema: dict,
        *,
        system: str | None = None,
        temperature: float = 0.7,
    ) -> Any: ...

    def generate_text(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float = 0.7,
    ) -> str: ...
