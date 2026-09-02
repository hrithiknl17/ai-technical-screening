"""Interview session endpoints.

The lifecycle is modelled explicitly rather than hidden behind one chat endpoint:

    POST /sessions                      start (plan is built here)
    GET  /sessions/{id}                 full state, including the transcript
    POST /sessions/{id}/next-question   ask for the next question
    POST /sessions/{id}/answers         submit an answer (graded synchronously)
    POST /sessions/{id}/complete        close the interview and build the report
    GET  /sessions/{id}/report          fetch the report

Handlers are declared `def`, not `async def`: they perform blocking LLM and
database work, so FastAPI runs them in its worker threadpool and one slow
interview never stalls the event loop for everyone else.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Response
from sqlalchemy.orm import Session

from app.api.mappers import question_to_dto, report_to_dto, session_to_dto
from app.core.errors import ConflictError
from app.db.models import Resume
from app.db.session import get_db
from app.schemas.dto import (
    AnswerAcceptedOut,
    AnswerIn,
    EvaluationOut,
    QuestionOut,
    ReportOut,
    SessionCreateIn,
    SessionOut,
)
from app.services.interview_service import InterviewService
from app.services.report_service import ReportService
from app.services.resume_service import ResumeService

router = APIRouter(prefix="/sessions", tags=["interview"])


def _dto(service: InterviewService, session, db: Session) -> SessionOut:
    resume = db.get(Resume, session.resume_id)
    return session_to_dto(session, resume=resume, current=service.current_question(session))


@router.post("", response_model=SessionOut, status_code=201, summary="Start an interview")
def create_session(payload: SessionCreateIn, db: Session = Depends(get_db)) -> SessionOut:
    resume = ResumeService(db).get(payload.resume_id)
    service = InterviewService(db)
    session = service.create_session(
        resume=resume, role_slug=payload.role, question_count=payload.question_count
    )
    service.next_question(session)  # generate the opening question eagerly
    return _dto(service, session, db)


@router.get("", response_model=list[SessionOut], summary="List recent sessions")
def list_sessions(db: Session = Depends(get_db)) -> list[SessionOut]:
    service = InterviewService(db)
    return [_dto(service, s, db) for s in service.list_sessions()]


@router.get("/{session_id}", response_model=SessionOut, summary="Fetch session state")
def get_session(session_id: str, db: Session = Depends(get_db)) -> SessionOut:
    service = InterviewService(db)
    return _dto(service, service.get_session(session_id), db)


@router.post(
    "/{session_id}/next-question",
    response_model=QuestionOut | None,
    summary="Get the pending question, or generate the next one",
)
def next_question(
    session_id: str, response: Response, db: Session = Depends(get_db)
) -> QuestionOut | None:
    service = InterviewService(db)
    session = service.get_session(session_id)
    question = service.next_question(session)
    if question is None:
        service.complete(session)
        response.status_code = 204
        return None
    return question_to_dto(question)


@router.post(
    "/{session_id}/answers",
    response_model=AnswerAcceptedOut,
    summary="Submit an answer; returns the grade and the updated session",
)
def submit_answer(
    session_id: str, payload: AnswerIn, db: Session = Depends(get_db)
) -> AnswerAcceptedOut:
    service = InterviewService(db)
    session = service.get_session(session_id)
    _, evaluation = service.submit_answer(
        session=session,
        question_id=payload.question_id,
        answer_text=payload.answer,
        time_taken_seconds=payload.time_taken_seconds,
    )
    return AnswerAcceptedOut(
        evaluation=EvaluationOut(
            score=evaluation.get("score", 0),
            verdict=evaluation.get("verdict", ""),
            covered_points=evaluation.get("covered_points", []),
            missed_points=evaluation.get("missed_points", []),
            feedback=evaluation.get("feedback", ""),
        ),
        session=_dto(service, session, db),
    )


@router.post(
    "/{session_id}/complete",
    response_model=ReportOut,
    summary="End the interview and generate the report",
)
def complete_session(session_id: str, db: Session = Depends(get_db)) -> ReportOut:
    service = InterviewService(db)
    session = service.get_session(session_id)
    if service.answered_count(session) == 0:
        raise ConflictError("Answer at least one question before ending the interview.")
    service.complete(session)
    return report_to_dto(ReportService(db).generate(session))


@router.get("/{session_id}/report", response_model=ReportOut, summary="Fetch the report")
def get_report(session_id: str, db: Session = Depends(get_db)) -> ReportOut:
    service = InterviewService(db)
    session = service.get_session(session_id)
    report_service = ReportService(db)
    report = report_service.get(session) or report_service.generate(session)
    return report_to_dto(report)
