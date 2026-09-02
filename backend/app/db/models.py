"""Persistence model.

The schema mirrors the pipeline the assignment asks for:

    Resume -> Session -> Question (with its retrieval trace) -> Answer -> Report

Every question keeps the query and the exact chunks that produced it, so any
generated question can be traced back to the corpus passage it came from.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    JSON,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base


def _uuid() -> str:
    return uuid.uuid4().hex


def _now() -> datetime:
    return datetime.now(timezone.utc)


class TimestampMixin:
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_now, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=_now, onupdate=_now, nullable=False
    )


class Resume(Base, TimestampMixin):
    __tablename__ = "resumes"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    filename: Mapped[str] = mapped_column(String(255))
    content_type: Mapped[str] = mapped_column(String(100), default="text/plain")
    raw_text: Mapped[str] = mapped_column(Text)
    #: Structured profile extracted by the resume parser (skills, seniority, ...).
    profile: Mapped[dict] = mapped_column(JSON, default=dict)

    sessions: Mapped[list["InterviewSession"]] = relationship(back_populates="resume")


class InterviewSession(Base, TimestampMixin):
    __tablename__ = "interview_sessions"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    resume_id: Mapped[str] = mapped_column(ForeignKey("resumes.id"), index=True)
    role_slug: Mapped[str] = mapped_column(String(64), index=True)
    candidate_name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    status: Mapped[str] = mapped_column(String(32), default="in_progress", index=True)
    planned_questions: Mapped[int] = mapped_column(Integer, default=6)
    #: Interview plan produced from the resume + role (topics, difficulty ladder).
    plan: Mapped[dict] = mapped_column(JSON, default=dict)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    resume: Mapped[Resume] = relationship(back_populates="sessions")
    questions: Mapped[list["Question"]] = relationship(
        back_populates="session", order_by="Question.position", cascade="all, delete-orphan"
    )
    report: Mapped["Report | None"] = relationship(
        back_populates="session", uselist=False, cascade="all, delete-orphan"
    )


class Question(Base, TimestampMixin):
    __tablename__ = "questions"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    session_id: Mapped[str] = mapped_column(
        ForeignKey("interview_sessions.id"), index=True
    )
    position: Mapped[int] = mapped_column(Integer)
    text: Mapped[str] = mapped_column(Text)
    topic: Mapped[str] = mapped_column(String(255), default="")
    difficulty: Mapped[str] = mapped_column(String(32), default="medium")
    question_type: Mapped[str] = mapped_column(String(32), default="conceptual")
    #: "planned" for ladder questions, "follow_up" for adaptive probes.
    origin: Mapped[str] = mapped_column(String(32), default="planned")
    #: Why this question was asked, in the generator's own words.
    rationale: Mapped[str] = mapped_column(Text, default="")
    #: What a strong answer should contain — used later for grading.
    expected_points: Mapped[list] = mapped_column(JSON, default=list)
    #: Full retrieval trace: query, chunk ids, scores, source document + page.
    retrieval_trace: Mapped[dict] = mapped_column(JSON, default=dict)

    session: Mapped[InterviewSession] = relationship(back_populates="questions")
    answer: Mapped["Answer | None"] = relationship(
        back_populates="question", uselist=False, cascade="all, delete-orphan"
    )


class Answer(Base, TimestampMixin):
    __tablename__ = "answers"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    question_id: Mapped[str] = mapped_column(ForeignKey("questions.id"), unique=True)
    text: Mapped[str] = mapped_column(Text)
    time_taken_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)
    #: Grader output: score 0-5, verdict, covered/missed points, feedback.
    evaluation: Mapped[dict] = mapped_column(JSON, default=dict)

    question: Mapped[Question] = relationship(back_populates="answer")


class Report(Base, TimestampMixin):
    __tablename__ = "reports"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    session_id: Mapped[str] = mapped_column(
        ForeignKey("interview_sessions.id"), unique=True, index=True
    )
    overall_score: Mapped[float] = mapped_column(Float, default=0.0)
    recommendation: Mapped[str] = mapped_column(String(64), default="")
    summary: Mapped[str] = mapped_column(Text, default="")
    payload: Mapped[dict] = mapped_column(JSON, default=dict)

    session: Mapped[InterviewSession] = relationship(back_populates="report")


class KnowledgeDocument(Base, TimestampMixin):
    """Ingestion bookkeeping — what is indexed, from which file, at which hash."""

    __tablename__ = "knowledge_documents"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    role_slug: Mapped[str] = mapped_column(String(64), index=True)
    filename: Mapped[str] = mapped_column(String(512))
    content_hash: Mapped[str] = mapped_column(String(64), index=True)
    pages: Mapped[int] = mapped_column(Integer, default=0)
    chunks: Mapped[int] = mapped_column(Integer, default=0)
