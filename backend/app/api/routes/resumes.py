from __future__ import annotations

from fastapi import APIRouter, Depends, File, Form, UploadFile
from sqlalchemy.orm import Session

from app.core.config import Settings, get_settings
from app.db.session import get_db
from app.domain.roles import get_role
from app.api.mappers import resume_to_dto
from app.schemas.dto import ResumeOut
from app.services.resume_service import ResumeService

router = APIRouter(prefix="/resumes", tags=["resumes"])


@router.post(
    "",
    response_model=ResumeOut,
    status_code=201,
    summary="Upload a resume (PDF/TXT/MD) and extract a structured profile",
)
async def upload_resume(
    file: UploadFile = File(..., description="PDF, TXT or Markdown resume"),
    role: str = Form(..., description="Target role slug, used to focus the extraction"),
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> ResumeOut:
    role_obj = get_role(role)
    data = await file.read()
    resume = ResumeService(db).create_from_upload(
        data=data,
        filename=file.filename or "resume",
        content_type=file.content_type,
        role_title=role_obj.title,
        max_bytes=settings.max_resume_bytes,
    )
    return resume_to_dto(resume)


@router.post(
    "/text",
    response_model=ResumeOut,
    status_code=201,
    summary="Submit resume text directly (no file upload)",
)
def submit_resume_text(
    payload: dict,
    db: Session = Depends(get_db),
) -> ResumeOut:
    role_obj = get_role(payload.get("role", ""))
    resume = ResumeService(db).create_from_text(
        text=payload.get("text", ""), role_title=role_obj.title
    )
    return resume_to_dto(resume)


@router.get("/{resume_id}", response_model=ResumeOut, summary="Fetch a stored resume profile")
def get_resume(resume_id: str, db: Session = Depends(get_db)) -> ResumeOut:
    return resume_to_dto(ResumeService(db).get(resume_id))
