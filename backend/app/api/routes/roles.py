from __future__ import annotations

from fastapi import APIRouter

from app.domain.roles import get_role, list_roles
from app.rag.vector_store import get_vector_store
from app.schemas.dto import KnowledgeStatusOut, RoleOut

router = APIRouter(prefix="/roles", tags=["roles"])


@router.get("", response_model=list[RoleOut], summary="List interviewable roles")
def get_roles() -> list[RoleOut]:
    store = get_vector_store()
    out: list[RoleOut] = []
    for role in list_roles():
        indexed = store.count(role.slug)
        out.append(
            RoleOut(
                slug=role.slug,
                title=role.title,
                description=role.description,
                competencies=role.competencies,
                corpus=role.corpus,
                knowledge_base_ready=indexed > 0,
                indexed_chunks=indexed,
            )
        )
    return out


@router.get("/{slug}/knowledge", response_model=KnowledgeStatusOut, summary="Index status")
def knowledge_status(slug: str) -> KnowledgeStatusOut:
    role = get_role(slug)
    meta = get_vector_store().meta(role.slug)
    return KnowledgeStatusOut(
        role=role.slug,
        indexed_chunks=meta.get("count", 0),
        sources=meta.get("sources", []),
        dimension=meta.get("dimension", 0),
        updated_at=meta.get("updated_at"),
    )
