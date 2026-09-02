"""Offline stub LLM.

Every prompt in `app.llm.prompts` starts with a `TASK: <name>` line. The stub
dispatches on that tag and returns a deterministic, schema-shaped payload built
from the retrieved context that was passed in. It exists so the full pipeline
(upload -> plan -> retrieve -> question -> answer -> report) can be exercised in
tests and demos with no API key and no network.
"""

from __future__ import annotations

import re
from typing import Any

_TASK_RE = re.compile(r"^TASK:\s*([a-z_]+)", re.MULTILINE)
_CITATION_RE = re.compile(r"^\[\d+\]\s*(.+)$", re.MULTILINE)
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


def _first_context_sentence(prompt: str) -> str:
    body = prompt.split("CONTEXT PASSAGES", 1)[-1]
    sentences = [s.strip() for s in _SENTENCE_RE.split(body) if len(s.strip()) > 60]
    return sentences[0][:240] if sentences else "the retrieved passage"


class StubLLM:
    name = "offline-stub"

    def generate_text(self, prompt: str, *, system: str | None = None, temperature: float = 0.7) -> str:
        return "Offline stub response."

    def generate_json(
        self,
        prompt: str,
        schema: dict,
        *,
        system: str | None = None,
        temperature: float = 0.7,
    ) -> Any:
        match = _TASK_RE.search(prompt)
        task = match.group(1) if match else "unknown"
        handler = getattr(self, f"_task_{task}", None)
        if handler is None:
            return {}
        return handler(prompt)

    # --- task handlers -----------------------------------------------------
    def _task_resume_profile(self, prompt: str) -> dict:
        text = prompt.lower()
        known = [
            "python", "pytorch", "tensorflow", "scikit-learn", "sql", "spark", "docker",
            "kubernetes", "fastapi", "transformers", "nlp", "computer vision", "aws",
        ]
        skills = [s for s in known if s in text] or ["python"]
        return {
            "candidate_name": "",
            "headline": "Offline profile extraction",
            # left at zero so the deterministic date parse wins the merge
            "years_experience": 0.0,
            "seniority": "junior",
            "skills": skills,
            "technologies": skills[:5],
            "domains": ["machine learning"],
            "projects": [],
            "strength_signals": ["Hands-on project work"],
            "gap_signals": ["Depth of theory unclear from resume"],
        }

    def _task_interview_plan(self, prompt: str) -> dict:
        return {
            "focus_summary": "Offline plan: cover core role competencies.",
            "topics": [
                {
                    "topic": "Model generalisation and overfitting",
                    "why": "Central to the role and referenced in the resume.",
                    "difficulty": "medium",
                    "query": "overfitting generalisation bias variance trade-off",
                    "question_type": "conceptual",
                },
                {
                    "topic": "Evaluation methodology",
                    "why": "Candidate reports applied project work.",
                    "difficulty": "medium",
                    "query": "cross validation evaluating hypotheses model comparison",
                    "question_type": "applied",
                },
            ],
        }

    def _task_question(self, prompt: str) -> dict:
        excerpt = _first_context_sentence(prompt)
        citations = _CITATION_RE.findall(prompt)[:2]
        return {
            "question": (
                "Based on the retrieved material - "
                f"\"{excerpt}\" - explain the idea in your own words and describe how you "
                "would apply it in a project you have worked on."
            ),
            "rationale": "Offline stub: derived directly from the top retrieved passage.",
            "expected_points": [
                "Correct definition of the concept",
                "Link to a concrete project decision",
                "Awareness of the trade-off involved",
            ],
            "grounded_in": citations,
        }

    def _task_evaluation(self, prompt: str) -> dict:
        answer = prompt.split("CANDIDATE ANSWER", 1)[-1]
        length = len(answer.split())
        score = 1 if length < 20 else 3 if length < 80 else 4
        return {
            "score": score,
            "verdict": "partial" if score < 4 else "strong",
            "covered_points": ["Answered the question"] if score > 1 else [],
            "missed_points": ["More depth expected"] if score < 4 else [],
            "feedback": "Offline stub evaluation based on answer length only.",
            "follow_up_needed": score >= 4,
            "follow_up_focus": "Push one level deeper on the same concept.",
        }

    def _task_report(self, prompt: str) -> dict:
        return {
            "summary": "Offline stub report. Run with a GEMINI_API_KEY for real analysis.",
            "recommendation": "borderline",
            "strengths": ["Completed the interview"],
            "gaps": ["Depth not assessable offline"],
            "topic_breakdown": [],
            "next_steps": ["Re-run with an API key configured"],
        }
