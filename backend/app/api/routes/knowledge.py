"""Knowledge-base introspection.

Not required by the interview flow, but retrieval you cannot inspect is
retrieval you cannot debug - this endpoint is what the frontend's "Knowledge
base" panel uses to show that the corpus is really being searched.
"""

from __future__ import annotations

from fastapi import APIRouter

from app.domain.roles import get_role
from app.rag.retriever import HybridRetriever
from app.schemas.dto import RetrievedChunkOut, SearchIn, SearchOut

router = APIRouter(prefix="/knowledge", tags=["knowledge"])


@router.post("/search", response_model=SearchOut, summary="Run a retrieval query directly")
def search(payload: SearchIn) -> SearchOut:
    role = get_role(payload.role)
    result = HybridRetriever().retrieve(role.slug, payload.query, top_k=payload.top_k)
    return SearchOut(
        role=role.slug,
        query=payload.query,
        results=[RetrievedChunkOut(**c) for c in result.trace()["chunks"]],
    )
