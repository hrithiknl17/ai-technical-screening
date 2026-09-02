"""Request/response contracts.

These are the only shapes the frontend sees. ORM rows are mapped into them
explicitly (rather than exposing the models) so storage can change without
breaking the API.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


# --- roles -----------------------------------------------------------------
class RoleOut(BaseModel):
    slug: str
    title: str
    description: str
    competencies: list[str]
    corpus: list[str]
    knowledge_base_ready: bool
    indexed_chunks: int


# --- resume ----------------------------------------------------------------
class ProjectOut(BaseModel):
    name: str = ""
    summary: str = ""
    technologies: list[str] = Field(default_factory=list)


class ResumeProfileOut(BaseModel):
    candidate_name: str = ""
    headline: str = ""
    years_experience: float = 0.0
    seniority: str = "junior"
    skills: list[str] = Field(default_factory=list)
    technologies: list[str] = Field(default_factory=list)
    domains: list[str] = Field(default_factory=list)
    projects: list[ProjectOut] = Field(default_factory=list)
    strength_signals: list[str] = Field(default_factory=list)
    gap_signals: list[str] = Field(default_factory=list)


class ResumeOut(BaseModel):
    id: str
    filename: str
    characters: int
    profile: ResumeProfileOut


# --- sessions --------------------------------------------------------------
class SessionCreateIn(BaseModel):
    resume_id: str
    role: str
    question_count: int | None = Field(default=None, ge=3, le=12)


class RetrievedChunkOut(BaseModel):
    id: str
    source: str
    citation: str
    section: str = ""
    page_start: int = 0
    page_end: int = 0
    score: float = 0.0
    dense_score: float = 0.0
    lexical_score: float = 0.0
    excerpt: str = ""


class RetrievalTraceOut(BaseModel):
    query: str = ""
    chunks: list[RetrievedChunkOut] = Field(default_factory=list)


class EvaluationOut(BaseModel):
    score: int = 0
    verdict: str = ""
    covered_points: list[str] = Field(default_factory=list)
    missed_points: list[str] = Field(default_factory=list)
    feedback: str = ""


class AnswerOut(BaseModel):
    text: str
    time_taken_seconds: float | None = None
    evaluation: EvaluationOut | None = None
    submitted_at: datetime | None = None


class QuestionOut(BaseModel):
    id: str
    position: int
    text: str
    topic: str
    difficulty: str
    question_type: str
    origin: str
    rationale: str
    expected_points: list[str] = Field(default_factory=list)
    retrieval: RetrievalTraceOut = Field(default_factory=RetrievalTraceOut)
    answer: AnswerOut | None = None


class PlanTopicOut(BaseModel):
    topic: str
    why: str = ""
    difficulty: str = "medium"
    query: str = ""
    question_type: str = "conceptual"
    status: str = "pending"


class SessionOut(BaseModel):
    id: str
    role: str
    role_title: str
    status: Literal["in_progress", "completed"]
    candidate_name: str | None = None
    planned_questions: int
    answered_questions: int
    focus_summary: str = ""
    topics: list[PlanTopicOut] = Field(default_factory=list)
    profile: ResumeProfileOut | None = None
    questions: list[QuestionOut] = Field(default_factory=list)
    current_question: QuestionOut | None = None
    created_at: datetime | None = None
    completed_at: datetime | None = None


class AnswerIn(BaseModel):
    question_id: str
    answer: str = Field(min_length=1, max_length=20000)
    time_taken_seconds: float | None = Field(default=None, ge=0)


class AnswerAcceptedOut(BaseModel):
    evaluation: EvaluationOut
    session: SessionOut


# --- report ----------------------------------------------------------------
class TopicScoreOut(BaseModel):
    topic: str
    assessment: str = ""
    score: float = 0.0


class ReportOut(BaseModel):
    session_id: str
    overall_score: float
    recommendation: str
    summary: str
    strengths: list[str] = Field(default_factory=list)
    gaps: list[str] = Field(default_factory=list)
    topic_breakdown: list[TopicScoreOut] = Field(default_factory=list)
    next_steps: list[str] = Field(default_factory=list)
    stats: dict[str, Any] = Field(default_factory=dict)
    generated_at: datetime | None = None


# --- knowledge base --------------------------------------------------------
class KnowledgeStatusOut(BaseModel):
    role: str
    indexed_chunks: int
    sources: list[str] = Field(default_factory=list)
    dimension: int = 0
    updated_at: str | None = None


class SearchIn(BaseModel):
    role: str
    query: str = Field(min_length=2, max_length=500)
    top_k: int = Field(default=5, ge=1, le=20)


class SearchOut(BaseModel):
    role: str
    query: str
    results: list[RetrievedChunkOut]


class ErrorOut(BaseModel):
    code: str
    message: str
    details: dict[str, Any] = Field(default_factory=dict)
