"""Session reporting.

Two layers again: deterministic statistics computed from stored grades (which
are always correct and always available), and an LLM narrative that interprets
them. The numbers in the report never come from the model - it is asked to
explain the evidence, not to invent the scores.
"""

from __future__ import annotations

import logging
from statistics import mean

from sqlalchemy.orm import Session

from app.core.errors import ConflictError, UpstreamError
from app.db.models import InterviewSession, Report, Resume
from app.domain.roles import get_role
from app.llm import get_llm
from app.llm.prompts import INTERVIEWER_SYSTEM, REPORT_SCHEMA, report_prompt

logger = logging.getLogger(__name__)

_RECOMMENDATION_BY_SCORE = [
    (4.3, "strong_hire"),
    (3.4, "hire"),
    (2.4, "borderline"),
    (0.0, "no_hire"),
]


def _recommendation_for(score: float) -> str:
    for threshold, label in _RECOMMENDATION_BY_SCORE:
        if score >= threshold:
            return label
    return "no_hire"


class ReportService:
    def __init__(self, db: Session) -> None:
        self.db = db
        self.llm = get_llm()

    def get(self, session: InterviewSession) -> Report | None:
        return session.report

    def generate(self, session: InterviewSession, *, force: bool = False) -> Report:
        answered = [q for q in session.questions if q.answer is not None]
        if not answered:
            raise ConflictError("Cannot produce a report before any question is answered.")
        if session.report is not None and not force:
            return session.report

        stats = self._statistics(session, answered)
        role = get_role(session.role_slug)
        resume = self.db.get(Resume, session.resume_id)
        transcript = self._transcript(session)

        try:
            narrative = self.llm.generate_json(
                report_prompt(
                    role_title=role.title,
                    profile=(resume.profile if resume else {}) or {},
                    transcript=transcript,
                    average_score=stats["average_score"],
                    coverage=stats["topics_covered"],
                ),
                REPORT_SCHEMA,
                system=INTERVIEWER_SYSTEM,
                temperature=0.4,
            )
        except UpstreamError as exc:
            logger.warning("report narrative failed, emitting statistics only: %s", exc)
            narrative = {
                "summary": (
                    "Automatic narrative generation was unavailable. "
                    f"The candidate answered {stats['answered']} questions with a mean score of "
                    f"{stats['average_score']:.1f}/5."
                ),
                "recommendation": _recommendation_for(stats["average_score"]),
                "strengths": [],
                "gaps": [],
                "topic_breakdown": [],
                "next_steps": [],
            }

        # Scores are authoritative from the database; the model only narrates.
        breakdown = narrative.get("topic_breakdown") or []
        by_topic = {b.get("topic"): b for b in breakdown}
        merged_breakdown = [
            {
                "topic": topic,
                "assessment": (by_topic.get(topic) or {}).get("assessment", ""),
                "score": round(score, 2),
            }
            for topic, score in stats["topic_scores"].items()
        ]

        report = session.report or Report(session_id=session.id)
        report.overall_score = round(stats["average_score"], 2)
        report.recommendation = narrative.get(
            "recommendation", _recommendation_for(stats["average_score"])
        )
        report.summary = narrative.get("summary", "")
        report.payload = {
            "strengths": narrative.get("strengths", []),
            "gaps": narrative.get("gaps", []),
            "next_steps": narrative.get("next_steps", []),
            "topic_breakdown": merged_breakdown,
            "stats": stats,
        }
        self.db.add(report)
        self.db.flush()
        logger.info("report for session %s: %s", session.id, report.recommendation)
        return report

    # --- deterministic parts ----------------------------------------------
    def _statistics(self, session: InterviewSession, answered: list) -> dict:
        scores = [(q.answer.evaluation or {}).get("score", 0) for q in answered]
        topic_scores: dict[str, list[int]] = {}
        for question in answered:
            topic_scores.setdefault(question.topic, []).append(
                (question.answer.evaluation or {}).get("score", 0)
            )
        durations = [
            q.answer.time_taken_seconds for q in answered if q.answer.time_taken_seconds
        ]
        sources: set[str] = set()
        for question in session.questions:
            for chunk in (question.retrieval_trace or {}).get("chunks", []):
                sources.add(chunk.get("source", ""))
        return {
            "answered": len(answered),
            "planned": session.planned_questions,
            "follow_ups": sum(1 for q in session.questions if q.origin == "follow_up"),
            "average_score": float(mean(scores)) if scores else 0.0,
            "score_by_position": scores,
            "topic_scores": {t: float(mean(s)) for t, s in topic_scores.items()},
            "topics_covered": list(topic_scores),
            "difficulty_mix": {
                level: sum(1 for q in answered if q.difficulty == level)
                for level in ("easy", "medium", "hard")
            },
            "median_seconds_per_answer": (
                round(sorted(durations)[len(durations) // 2], 1) if durations else None
            ),
            "knowledge_sources_used": sorted(s for s in sources if s),
        }

    def _transcript(self, session: InterviewSession) -> str:
        lines: list[str] = []
        for question in session.questions:
            if question.answer is None:
                continue
            evaluation = question.answer.evaluation or {}
            lines.append(
                f"--- Q{question.position} [{question.topic} | {question.difficulty} | "
                f"{question.origin}]\n"
                f"Question: {question.text}\n"
                f"Expected: {'; '.join(question.expected_points or [])}\n"
                f"Answer: {question.answer.text}\n"
                f"Grade: {evaluation.get('score')}/5 ({evaluation.get('verdict')}) - "
                f"{evaluation.get('feedback', '')}"
            )
        return "\n".join(lines)
