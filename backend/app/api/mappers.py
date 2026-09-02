"""ORM -> DTO mapping, kept out of both the routes and the services."""

from __future__ import annotations

from app.db.models import InterviewSession, Question, Report, Resume
from app.domain.roles import get_role
from app.schemas.dto import (
    AnswerOut,
    EvaluationOut,
    PlanTopicOut,
    QuestionOut,
    ReportOut,
    ResumeOut,
    ResumeProfileOut,
    RetrievalTraceOut,
    SessionOut,
)


def resume_to_dto(resume: Resume) -> ResumeOut:
    return ResumeOut(
        id=resume.id,
        filename=resume.filename,
        characters=len(resume.raw_text or ""),
        profile=ResumeProfileOut(**(resume.profile or {})),
    )


def question_to_dto(question: Question, *, include_trace: bool = True) -> QuestionOut:
    answer = None
    if question.answer is not None:
        evaluation = question.answer.evaluation or {}
        answer = AnswerOut(
            text=question.answer.text,
            time_taken_seconds=question.answer.time_taken_seconds,
            submitted_at=question.answer.created_at,
            evaluation=EvaluationOut(
                score=evaluation.get("score", 0),
                verdict=evaluation.get("verdict", ""),
                covered_points=evaluation.get("covered_points", []),
                missed_points=evaluation.get("missed_points", []),
                feedback=evaluation.get("feedback", ""),
            ),
        )
    return QuestionOut(
        id=question.id,
        position=question.position,
        text=question.text,
        topic=question.topic,
        difficulty=question.difficulty,
        question_type=question.question_type,
        origin=question.origin,
        rationale=question.rationale,
        expected_points=question.expected_points or [],
        retrieval=(
            RetrievalTraceOut(**(question.retrieval_trace or {}))
            if include_trace
            else RetrievalTraceOut()
        ),
        answer=answer,
    )


def session_to_dto(
    session: InterviewSession,
    *,
    resume: Resume | None = None,
    current: Question | None = None,
) -> SessionOut:
    plan = session.plan or {}
    return SessionOut(
        id=session.id,
        role=session.role_slug,
        role_title=get_role(session.role_slug).title,
        status=session.status,  # type: ignore[arg-type]
        candidate_name=session.candidate_name,
        planned_questions=session.planned_questions,
        answered_questions=sum(1 for q in session.questions if q.answer is not None),
        focus_summary=plan.get("focus_summary", ""),
        topics=[PlanTopicOut(**t) for t in plan.get("topics", [])],
        profile=ResumeProfileOut(**(resume.profile or {})) if resume else None,
        questions=[question_to_dto(q) for q in session.questions],
        current_question=question_to_dto(current) if current is not None else None,
        created_at=session.created_at,
        completed_at=session.completed_at,
    )


def report_to_dto(report: Report) -> ReportOut:
    payload = report.payload or {}
    return ReportOut(
        session_id=report.session_id,
        overall_score=report.overall_score,
        recommendation=report.recommendation,
        summary=report.summary,
        strengths=payload.get("strengths", []),
        gaps=payload.get("gaps", []),
        topic_breakdown=payload.get("topic_breakdown", []),
        next_steps=payload.get("next_steps", []),
        stats=payload.get("stats", {}),
        generated_at=report.created_at,
    )
