from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy import text

from app.core.config import Settings, get_settings
from app.db.session import engine
from app.domain.roles import list_roles
from app.rag.embeddings import get_embedder
from app.rag.vector_store import get_vector_store

router = APIRouter(tags=["system"])


@router.get("/health", summary="Liveness and dependency status")
def health(settings: Settings = Depends(get_settings)) -> dict:
    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
        database_ok = True
    except Exception:
        database_ok = False

    store = get_vector_store()
    embedder = get_embedder(settings)
    return {
        "status": "ok" if database_ok else "degraded",
        "environment": settings.app_env,
        "database": "ok" if database_ok else "unavailable",
        "llm": {
            "configured": settings.llm_enabled,
            "model": settings.llm_model if settings.llm_enabled else "offline-stub",
        },
        # Report the embedder actually in use, not the configured preference:
        # `auto` resolves at startup and the index is tied to what it picked.
        "embeddings": {"model": embedder.name, "dimensions": embedder.dimension},
        "knowledge_base": {
            role.slug: store.count(role.slug) for role in list_roles()
        },
    }
