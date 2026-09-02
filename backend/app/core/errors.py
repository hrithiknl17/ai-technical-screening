"""Domain error hierarchy.

Services raise these; the API layer translates them into HTTP responses in a
single place (`app.main`), so business logic never imports HTTP concerns.
"""

from __future__ import annotations


class AppError(Exception):
    """Base class for expected, user-visible failures."""

    status_code = 500
    code = "internal_error"

    def __init__(self, message: str, *, details: dict | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}


class ValidationError(AppError):
    status_code = 422
    code = "validation_error"


class NotFoundError(AppError):
    status_code = 404
    code = "not_found"


class ConflictError(AppError):
    """The resource exists but is not in a state that allows this operation."""

    status_code = 409
    code = "conflict"


class UpstreamError(AppError):
    """A dependency (LLM / embedding API) failed."""

    status_code = 502
    code = "upstream_error"


class KnowledgeBaseError(AppError):
    status_code = 503
    code = "knowledge_base_unavailable"
