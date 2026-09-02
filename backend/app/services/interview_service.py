"""Interview orchestration - the core of the system.

One class owns the interview lifecycle so the API layer stays a thin translation
of HTTP to method calls:

    plan(resume, role) -> question(topic) -> answer(question) -> adapt -> report

Adaptation is where a screening interview differs from a question list. After
every answer the service:

* re-calibrates the difficulty of the remaining topics (a strong answer raises
  the next question's difficulty, a weak one lowers it), and
* may queue one follow-up on the topic just answered - to push a strong answer
  deeper, or to give a vague answer one chance to be specific.

Follow-ups are budgeted (`_FOLLOW_UP_BUDGET`) so an interview cannot run away.
"""

from __future__ import annotations

import copy
import logging
from datetime import datetime, timezone

from sqlalchemy.orm import Session

from app.core.config import Settings, get_settings
from app.core.errors import ConflictError, KnowledgeBaseError, NotFoundError, UpstreamError
from app.db.models import Answer, InterviewSession, Question, Resume
from app.domain.roles import Role, get_role
from app.llm import get_llm
from app.llm.prompts import (
    EVALUATION_SCHEMA,
    INTERVIEW_PLAN_SCHEMA,
    INTERVIEWER_SYSTEM,
    QUESTION_SCHEMA,
    evaluation_prompt,
    interview_plan_prompt,
    question_prompt,
)
from app.rag.ingest import corpus_sections
from app.rag.retriever import HybridRetriever, RetrievalResult
from app.rag.vector_store import get_vector_store

logger = logging.getLogger(__name__)

_DIFFICULTY_LADDER = ["easy", "medium", "hard"]
_FOLLOW_UP_BUDGET = 2


def _copy_plan(plan: dict | None) -> dict:
    return copy.deepcopy(plan) if plan else {}


def _step_difficulty(difficulty: str, delta: int) -> str:
    try:
        index = _DIFFICULTY_LADDER.index(difficulty)
    except ValueError:
        index = 1
    return _DIFFICULTY_LADDER[max(0, min(len(_DIFFICULTY_LADDER) - 1, index + delta))]


class InterviewService:
    def __init__(
        self,
        db: Session,
        *,
        settings: Settings | None = None,
        retriever: HybridRetriever | None = None,
    ) -> None:
        self.db = db
        self.settings = settings or get_settings()
        self.retriever = retriever or HybridRetriever(settings=self.settings)
        self.llm = get_llm(self.settings)

    # --- lookup ------------------------------------------------------------
    def get_session(self, session_id: str) -> InterviewSession:
        session = self.db.get(InterviewSession, session_id)
        if session is None:
            raise NotFoundError(f"Interview session '{session_id}' not found.")
        return session

    def list_sessions(self, limit: int = 50) -> list[InterviewSession]:
        return (
            self.db.query(InterviewSession)
            .order_by(InterviewSession.created_at.desc())
            .limit(limit)
            .all()
        )

    # --- creation ----------------------------------------------------------
    def create_session(
        self, *, resume: Resume, role_slug: str, question_count: int | None = None
    ) -> InterviewSession:
        role = get_role(role_slug)
        indexed = get_vector_store().count(role_slug)
        if indexed == 0:
            raise KnowledgeBaseError(
                f"The knowledge base for '{role.title}' has not been indexed yet. "
                f"Run: python -m scripts.ingest_kb --role {role_slug}",
                details={"role": role_slug},
            )
        count = min(
            question_count or self.settings.default_question_count,
            self.settings.max_question_count,
        )
        plan = self._build_plan(role=role, profile=resume.profile, question_count=count)
        session = InterviewSession(
            resume_id=resume.id,
            role_slug=role_slug,
            candidate_name=(resume.profile or {}).get("candidate_name") or None,
            planned_questions=len(plan["topics"]),
            plan=plan,
            status="in_progress",
        )
        self.db.add(session)
        self.db.flush()
        logger.info(
            "created session %s for role %s with %s topics",
            session.id,
            role_slug,
            len(plan["topics"]),
        )
        return session

    def _build_plan(self, *, role: Role, profile: dict, question_count: int) -> dict:
        """Ask the model for a candidate-specific topic plan; fall back to the role's own."""
        sections = corpus_sections(role.slug)
        try:
            raw = self.llm.generate_json(
                interview_plan_prompt(
                    role_title=role.title,
                    role_description=role.description,
                    competencies=role.competencies,
                    profile=profile,
                    corpus_sections=sections,
                    question_count=question_count,
                ),
                INTERVIEW_PLAN_SCHEMA,
                system=INTERVIEWER_SYSTEM,
                temperature=0.6,
            )
            topics = [t for t in (raw.get("topics") or []) if t.get("topic") and t.get("query")]
            focus = raw.get("focus_summary", "")
        except UpstreamError as exc:
            logger.warning("planning failed, falling back to role defaults: %s", exc)
            topics, focus = [], "Fallback plan built from the role's competency list."

        if not topics:
            topics = [
                {
                    "topic": competency,
                    "why": "Core competency for this role.",
                    "difficulty": "medium",
                    "query": query,
                    "question_type": "conceptual",
                }
                for competency, query in zip(
                    role.competencies,
                    role.fallback_queries * 3,
                )
            ]
        topics = topics[:question_count]
        for topic in topics:
            topic.setdefault("question_type", "conceptual")
            topic.setdefault("difficulty", "medium")
            topic["status"] = "pending"
        return {"focus_summary": focus, "topics": topics, "follow_ups_used": 0, "queue": []}

    # --- questions ---------------------------------------------------------
    def current_question(self, session: InterviewSession) -> Question | None:
        """The question awaiting an answer, if any."""
        for question in session.questions:
            if question.answer is None:
                return question
        return None

    def next_question(self, session: InterviewSession) -> Question | None:
        """Return the pending question, or generate the next one. None = interview over."""
        if session.status == "completed":
            return None
        pending = self.current_question(session)
        if pending is not None:
            return pending

        item = self._pop_next_item(session)
        if item is None:
            return None

        role = get_role(session.role_slug)
        resume = self.db.get(Resume, session.resume_id)
        profile = (resume.profile if resume else {}) or {}

        retrieval = self.retriever.retrieve(
            session.role_slug,
            item["query"],
            exclude_ids=self._already_used_chunk_ids(session),
        )
        if not retrieval.chunks:  # nothing new: retry without the exclusion set
            retrieval = self.retriever.retrieve(session.role_slug, item["query"])

        payload = self._generate_question(
            role=role,
            item=item,
            profile=profile,
            retrieval=retrieval,
            history=self._history_digest(session),
        )
        question = Question(
            session_id=session.id,
            position=len(session.questions) + 1,
            text=payload["question"],
            topic=item["topic"],
            difficulty=item.get("difficulty", "medium"),
            question_type=item.get("question_type", "conceptual"),
            origin=item.get("origin", "planned"),
            rationale=payload.get("rationale", ""),
            expected_points=payload.get("expected_points", []),
            retrieval_trace=retrieval.trace(),
        )
        self.db.add(question)
        self.db.flush()
        self.db.refresh(session)
        return question

    def _generate_question(
        self,
        *,
        role: Role,
        item: dict,
        profile: dict,
        retrieval: RetrievalResult,
        history: str,
    ) -> dict:
        context_block = retrieval.context_block()
        if not context_block:
            raise KnowledgeBaseError(
                "Retrieval returned no usable context for this topic. Re-ingest the corpus."
            )
        try:
            payload = self.llm.generate_json(
                question_prompt(
                    role_title=role.title,
                    topic=item["topic"],
                    why=item.get("why", ""),
                    difficulty=item.get("difficulty", "medium"),
                    question_type=item.get("question_type", "conceptual"),
                    profile=profile,
                    context_block=context_block,
                    history=history,
                    follow_up_focus=item.get("follow_up_focus"),
                ),
                QUESTION_SCHEMA,
                system=INTERVIEWER_SYSTEM,
                temperature=0.8,
            )
        except UpstreamError as exc:
            logger.warning("question generation failed: %s", exc)
            raise
        if not payload.get("question"):
            raise UpstreamError("The model returned an empty question.")
        return payload

    def _pop_next_item(self, session: InterviewSession) -> dict | None:
        """Take the next follow-up if one is queued, otherwise the next planned topic."""
        # Deep copy, never mutate in place: a plain JSON column is compared by
        # value on flush, so mutating the loaded object would make the "new"
        # value equal the "old" one and SQLAlchemy would emit no UPDATE.
        plan = _copy_plan(session.plan)
        queue = list(plan.get("queue") or [])
        if queue:
            item = queue.pop(0)
            plan["queue"] = queue
            session.plan = plan
            return item
        topics = list(plan.get("topics") or [])
        for index, topic in enumerate(topics):
            if topic.get("status") == "pending":
                topic["status"] = "asked"
                topics[index] = topic
                plan["topics"] = topics
                session.plan = plan
                return {**topic, "origin": "planned"}
        return None

    def _already_used_chunk_ids(self, session: InterviewSession) -> set[str]:
        used: set[str] = set()
        for question in session.questions:
            for chunk in (question.retrieval_trace or {}).get("chunks", []):
                used.add(chunk.get("id", ""))
        return used - {""}

    def _history_digest(self, session: InterviewSession, limit: int = 3) -> str:
        lines: list[str] = []
        for question in session.questions[-limit:]:
            lines.append(f"Q{question.position} [{question.topic}]: {question.text}")
            if question.answer:
                evaluation = question.answer.evaluation or {}
                lines.append(
                    f"A{question.position} (scored {evaluation.get('score', '?')}/5): "
                    f"{question.answer.text[:400]}"
                )
        return "\n".join(lines)

    # --- answers -----------------------------------------------------------
    def submit_answer(
        self,
        *,
        session: InterviewSession,
        question_id: str,
        answer_text: str,
        time_taken_seconds: float | None = None,
    ) -> tuple[Question, dict]:
        if session.status == "completed":
            raise ConflictError("This interview is already complete.")
        question = next((q for q in session.questions if q.id == question_id), None)
        if question is None:
            raise NotFoundError(f"Question '{question_id}' does not belong to this session.")
        if question.answer is not None:
            raise ConflictError("This question has already been answered.")

        evaluation = self._evaluate(question, answer_text)
        answer = Answer(
            question_id=question.id,
            text=answer_text.strip(),
            time_taken_seconds=time_taken_seconds,
            evaluation=evaluation,
        )
        self.db.add(answer)
        self.db.flush()

        self._adapt(session, question, evaluation)
        # Flush the adapted plan before refreshing: `refresh` reloads the row and
        # would otherwise discard the queued follow-up and difficulty changes.
        self.db.flush()
        self.db.refresh(session)
        return question, evaluation

    def _evaluate(self, question: Question, answer_text: str) -> dict:
        context_block = "\n\n".join(
            f"[{i}] {c.get('citation', '')}\n{c.get('excerpt', '')}"
            for i, c in enumerate((question.retrieval_trace or {}).get("chunks", []), start=1)
        )
        try:
            evaluation = self.llm.generate_json(
                evaluation_prompt(
                    question=question.text,
                    expected_points=question.expected_points or [],
                    context_block=context_block,
                    answer=answer_text,
                    difficulty=question.difficulty,
                ),
                EVALUATION_SCHEMA,
                system=INTERVIEWER_SYSTEM,
                temperature=0.2,
            )
        except UpstreamError as exc:
            logger.warning("evaluation failed, storing an ungraded answer: %s", exc)
            return {
                "score": 0,
                "verdict": "ungraded",
                "covered_points": [],
                "missed_points": [],
                "feedback": "Automatic grading was unavailable for this answer.",
                "follow_up_needed": False,
            }
        evaluation["score"] = max(0, min(5, int(evaluation.get("score", 0))))
        return evaluation

    def _adapt(self, session: InterviewSession, question: Question, evaluation: dict) -> None:
        """Re-calibrate remaining difficulty and optionally queue a follow-up."""
        plan = _copy_plan(session.plan)
        topics = list(plan.get("topics") or [])
        score = evaluation.get("score", 3)

        delta = 1 if score >= 4 else (-1 if score <= 2 else 0)
        if delta:
            for index, topic in enumerate(topics):
                if topic.get("status") == "pending":
                    topic["difficulty"] = _step_difficulty(
                        topic.get("difficulty", "medium"), delta
                    )
                    topics[index] = topic
                    break
        plan["topics"] = topics

        queue = list(plan.get("queue") or [])
        used = int(plan.get("follow_ups_used") or 0)
        wants_follow_up = bool(evaluation.get("follow_up_needed")) and score not in (0,)
        if wants_follow_up and used < _FOLLOW_UP_BUDGET and question.origin != "follow_up":
            focus = evaluation.get("follow_up_focus") or "the weakest part of the answer"
            queue.append(
                {
                    "topic": question.topic,
                    "why": f"Follow-up on the previous answer ({evaluation.get('verdict')}).",
                    "difficulty": _step_difficulty(question.difficulty, 1 if score >= 4 else -1),
                    "query": f"{question.topic} {focus}",
                    "question_type": "applied" if score >= 4 else "conceptual",
                    "origin": "follow_up",
                    "follow_up_focus": focus,
                }
            )
            plan["follow_ups_used"] = used + 1
            logger.info("queued follow-up on '%s' (score %s)", question.topic, score)
        plan["queue"] = queue
        session.plan = plan

    # --- lifecycle ---------------------------------------------------------
    def is_exhausted(self, session: InterviewSession) -> bool:
        plan = session.plan or {}
        pending_topics = any(t.get("status") == "pending" for t in plan.get("topics", []))
        return not pending_topics and not plan.get("queue") and self.current_question(session) is None

    def complete(self, session: InterviewSession) -> InterviewSession:
        if session.status != "completed":
            # A question generated but never answered would otherwise sit in the
            # transcript as an orphan; drop it so the stored record is exactly
            # what happened.
            dangling = self.current_question(session)
            if dangling is not None:
                session.questions.remove(dangling)
                self.db.delete(dangling)
            session.status = "completed"
            session.completed_at = datetime.now(timezone.utc)
            self.db.flush()
        return session

    def answered_count(self, session: InterviewSession) -> int:
        return sum(1 for q in session.questions if q.answer is not None)
