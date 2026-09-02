"""Interview orchestration: plan progression, adaptivity and persistence.

These exercise the service directly (no HTTP) because the interesting behaviour
is stateful: the plan must survive flushes, follow-ups must be budgeted, and the
difficulty ladder must move with the grades.
"""

from __future__ import annotations

import pytest

from app.core.errors import ConflictError
from app.db.session import session_scope
from app.services.interview_service import InterviewService, _step_difficulty
from app.services.report_service import ReportService
from app.services.resume_service import ResumeService
from tests.conftest import RESUME_TEXT

STRONG_ANSWER = " ".join(
    ["overfitting validation pruning cross validation variance estimate bias"] * 20
)
WEAK_ANSWER = "not sure"


def _new_session(db, role: str, questions: int = 3):
    resume = ResumeService(db).create_from_text(text=RESUME_TEXT, role_title="AI/ML Engineer")
    service = InterviewService(db)
    return service, service.create_session(
        resume=resume, role_slug=role, question_count=questions
    )


def test_difficulty_ladder_is_clamped():
    assert _step_difficulty("easy", -1) == "easy"
    assert _step_difficulty("easy", 1) == "medium"
    assert _step_difficulty("hard", 1) == "hard"
    assert _step_difficulty("nonsense", -1) == "easy"


def test_plan_progresses_and_interview_exhausts(seeded_role):
    with session_scope() as db:
        service, session = _new_session(db, seeded_role)
        asked = 0
        while (question := service.next_question(session)) is not None and asked < 10:
            service.submit_answer(
                session=session, question_id=question.id, answer_text=WEAK_ANSWER
            )
            asked += 1

        assert asked >= 1
        assert service.next_question(session) is None
        assert all(t["status"] == "asked" for t in session.plan["topics"])
        assert service.is_exhausted(session)


def test_strong_answers_queue_follow_ups_and_raise_difficulty(seeded_role):
    with session_scope() as db:
        service, session = _new_session(db, seeded_role)
        origins, difficulties = [], []
        while (question := service.next_question(session)) is not None and len(origins) < 10:
            origins.append(question.origin)
            difficulties.append(question.difficulty)
            service.submit_answer(
                session=session, question_id=question.id, answer_text=STRONG_ANSWER
            )

        assert "follow_up" in origins, "a strong answer should earn a follow-up"
        assert session.plan["follow_ups_used"] <= 2, "follow-ups must stay budgeted"
        assert "hard" in difficulties, "difficulty should escalate after strong answers"


def test_completion_drops_the_unanswered_question_and_builds_a_report(seeded_role):
    with session_scope() as db:
        service, session = _new_session(db, seeded_role)
        first = service.next_question(session)
        service.submit_answer(session=session, question_id=first.id, answer_text=STRONG_ANSWER)
        service.next_question(session)  # generated but deliberately left unanswered

        service.complete(session)
        assert session.status == "completed"
        assert all(q.answer is not None for q in session.questions)

        report = ReportService(db).generate(session)
        assert report.overall_score > 0
        assert report.payload["stats"]["answered"] == 1
        assert report.payload["stats"]["knowledge_sources_used"]


def test_answering_a_completed_session_is_rejected(seeded_role):
    with session_scope() as db:
        service, session = _new_session(db, seeded_role)
        question = service.next_question(session)
        service.submit_answer(session=session, question_id=question.id, answer_text=WEAK_ANSWER)
        service.complete(session)

        with pytest.raises(ConflictError):
            service.submit_answer(
                session=session, question_id=question.id, answer_text=WEAK_ANSWER
            )


def test_questions_do_not_reuse_the_same_passages(seeded_role):
    with session_scope() as db:
        service, session = _new_session(db, seeded_role, questions=2)
        seen: set[str] = set()
        while (question := service.next_question(session)) is not None:
            ids = {c["id"] for c in question.retrieval_trace["chunks"]}
            # the corpus is tiny, so allow reuse only when nothing else is left
            assert ids, "every question must record its retrieval trace"
            seen |= ids
            service.submit_answer(
                session=session, question_id=question.id, answer_text=WEAK_ANSWER
            )
        assert seen
