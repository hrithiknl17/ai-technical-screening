"""Prompt and response-schema definitions.

Kept in one module so the wording of every model interaction is reviewable in a
single place. Each prompt opens with a `TASK:` tag (used by the offline stub to
dispatch) and each has a matching Gemini `responseSchema`, so callers receive a
predictable object instead of free prose.
"""

from __future__ import annotations

INTERVIEWER_SYSTEM = (
    "You are a senior technical interviewer running a screening interview. "
    "You are rigorous, specific and fair. You never ask generic questions such as "
    "'what is machine learning?' or 'tell me about yourself'. Every question you ask "
    "must be answerable from first principles by a competent candidate and must be "
    "grounded in the reference material you are given. You write in plain, direct English."
)

# --------------------------------------------------------------------------
# 1. Resume -> structured profile
# --------------------------------------------------------------------------

RESUME_PROFILE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "candidate_name": {"type": "STRING"},
        "headline": {"type": "STRING"},
        "years_experience": {"type": "NUMBER"},
        "seniority": {"type": "STRING", "enum": ["intern", "junior", "mid", "senior"]},
        "skills": {"type": "ARRAY", "items": {"type": "STRING"}},
        "technologies": {"type": "ARRAY", "items": {"type": "STRING"}},
        "domains": {"type": "ARRAY", "items": {"type": "STRING"}},
        "projects": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "name": {"type": "STRING"},
                    "summary": {"type": "STRING"},
                    "technologies": {"type": "ARRAY", "items": {"type": "STRING"}},
                },
                "required": ["name", "summary"],
            },
        },
        "strength_signals": {"type": "ARRAY", "items": {"type": "STRING"}},
        "gap_signals": {"type": "ARRAY", "items": {"type": "STRING"}},
    },
    "required": ["skills", "technologies", "seniority", "strength_signals", "gap_signals"],
}


def resume_profile_prompt(resume_text: str, role_title: str) -> str:
    return f"""TASK: resume_profile

Extract a structured profile from the resume below. The profile will be used to
target a technical interview for the role: {role_title}.

Rules:
- Only record what the resume actually supports. Do not invent employers, dates or skills.
- `skills` are capabilities (e.g. "model evaluation", "distributed training").
  `technologies` are named tools/libraries (e.g. "PyTorch", "Airflow").
- `seniority` reflects depth of hands-on work, not job titles.
- `strength_signals`: concrete evidence of depth (shipped systems, scale, ownership).
- `gap_signals`: areas a screener should probe because the resume is thin or vague there.
- If years of experience are not stated, estimate from dates; use 0 if impossible.

RESUME TEXT
-----------
{resume_text[:16000]}
"""


# --------------------------------------------------------------------------
# 2. Profile + role -> interview plan (topics and retrieval queries)
# --------------------------------------------------------------------------

INTERVIEW_PLAN_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "focus_summary": {"type": "STRING"},
        "topics": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "topic": {"type": "STRING"},
                    "why": {"type": "STRING"},
                    "difficulty": {
                        "type": "STRING",
                        "enum": ["easy", "medium", "hard"],
                    },
                    "query": {"type": "STRING"},
                    "question_type": {
                        "type": "STRING",
                        "enum": ["conceptual", "applied", "design", "debugging"],
                    },
                },
                "required": ["topic", "why", "difficulty", "query", "question_type"],
            },
        },
    },
    "required": ["focus_summary", "topics"],
}


def interview_plan_prompt(
    *,
    role_title: str,
    role_description: str,
    competencies: list[str],
    profile: dict,
    corpus_sections: list[str],
    question_count: int,
) -> str:
    sections = "\n".join(f"- {s}" for s in corpus_sections[:60]) or "- (not available)"
    competency_list = "\n".join(f"- {c}" for c in competencies)
    return f"""TASK: interview_plan

Plan a {question_count}-question screening interview for the role below, targeted
at this specific candidate.

ROLE: {role_title}
{role_description}

COMPETENCIES TO COVER (priority order)
{competency_list}

CANDIDATE PROFILE (extracted from their resume)
{profile}

THE KNOWLEDGE BASE FOR THIS ROLE CONTAINS THESE SECTIONS
{sections}

Produce exactly {question_count} topics. For each topic:
- `topic`: the specific concept to test, not a broad area.
- `why`: tie it to something concrete in the candidate's profile (a technology they
  claim, a project they list, or a gap that must be probed).
- `difficulty`: calibrate to the candidate's seniority - a candidate who claims
  production experience with a technology should get `hard` on that technology, and
  areas the resume never mentions should start at `easy` or `medium`.
- `query`: a retrieval query in the vocabulary of the textbooks above (technical terms,
  no candidate names, no "interview question" phrasing). This string is embedded and
  matched against the corpus, so it must read like a passage from the book.
- `question_type`: `conceptual` (explain/derive), `applied` (use it on a scenario),
  `design` (build a system), `debugging` (diagnose a failure).

Sequence the topics so the interview opens on the candidate's strongest claimed area
and progressively moves toward the areas that need probing.
"""


# --------------------------------------------------------------------------
# 3. Retrieved context -> interview question
# --------------------------------------------------------------------------

QUESTION_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "question": {"type": "STRING"},
        "rationale": {"type": "STRING"},
        "expected_points": {"type": "ARRAY", "items": {"type": "STRING"}},
        "grounded_in": {"type": "ARRAY", "items": {"type": "STRING"}},
    },
    "required": ["question", "rationale", "expected_points"],
}


def question_prompt(
    *,
    role_title: str,
    topic: str,
    why: str,
    difficulty: str,
    question_type: str,
    profile: dict,
    context_block: str,
    history: str,
    follow_up_focus: str | None = None,
) -> str:
    follow_up = (
        f"\nThis is a FOLLOW-UP to the previous answer. Focus it on: {follow_up_focus}\n"
        if follow_up_focus
        else ""
    )
    return f"""TASK: question

Write ONE interview question for a {role_title} candidate.

TOPIC: {topic}
WHY THIS TOPIC: {why}
DIFFICULTY: {difficulty}
QUESTION TYPE: {question_type}
{follow_up}
CANDIDATE PROFILE (use it to make the question personal and concrete)
{profile}

INTERVIEW SO FAR
{history or "(this is the first question)"}

CONTEXT PASSAGES (retrieved from the role's knowledge base - the question must be
grounded in these, and a correct answer must be checkable against them)
{context_block}

Requirements:
- The question must be answerable using the concepts in the passages above.
- Anchor it in the candidate's own experience where the profile allows: name their
  technology or project, then ask about the concept through that lens.
- `conceptual` -> ask them to explain or justify a mechanism, not to recite a definition.
  `applied`/`design` -> give a short, concrete scenario with real constraints.
  `debugging` -> describe a plausible failure and ask for a diagnosis.
- Never ask two questions in one. Never ask something answerable with a single word.
- Do not mention "the passage", "the textbook", or that context was retrieved.
- `expected_points`: 3-5 things a strong answer contains. These are graded against later.
- `grounded_in`: the citation labels (e.g. "Mitchell, p. 54") of the passages you used.
- Keep the question under 90 words.
"""


# --------------------------------------------------------------------------
# 4. Answer grading
# --------------------------------------------------------------------------

EVALUATION_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "score": {"type": "INTEGER"},
        "verdict": {
            "type": "STRING",
            "enum": ["incorrect", "weak", "partial", "strong", "excellent"],
        },
        "covered_points": {"type": "ARRAY", "items": {"type": "STRING"}},
        "missed_points": {"type": "ARRAY", "items": {"type": "STRING"}},
        "feedback": {"type": "STRING"},
        "follow_up_needed": {"type": "BOOLEAN"},
        "follow_up_focus": {"type": "STRING"},
    },
    "required": ["score", "verdict", "covered_points", "missed_points", "feedback"],
}


def evaluation_prompt(
    *,
    question: str,
    expected_points: list[str],
    context_block: str,
    answer: str,
    difficulty: str,
) -> str:
    expected = "\n".join(f"- {p}" for p in expected_points) or "- (none recorded)"
    return f"""TASK: evaluation

Grade the candidate's answer. Be fair but not generous: this is a screening signal,
not encouragement.

QUESTION ({difficulty})
{question}

WHAT A STRONG ANSWER CONTAINS
{expected}

REFERENCE MATERIAL (the ground truth for this question)
{context_block[:5000]}

CANDIDATE ANSWER
{answer[:6000]}

Scoring, 0-5:
0 no answer or entirely off-topic; 1 fundamental misunderstanding;
2 partially right with a material error; 3 correct but shallow;
4 correct, complete and well-reasoned; 5 that plus insight beyond the expected points.

Rules:
- Judge the substance, not the writing style or length.
- An answer that is correct but phrased differently from the expected points is correct.
- Flag a factual error explicitly in `feedback`, quoting what the candidate claimed.
- `follow_up_needed`: true when one more question would materially sharpen the signal
  (a strong answer worth pushing deeper, or a vague answer worth one clarification).
- `follow_up_focus`: what that follow-up should target. Empty if not needed.
"""


# --------------------------------------------------------------------------
# 5. Session report
# --------------------------------------------------------------------------

REPORT_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "summary": {"type": "STRING"},
        "recommendation": {
            "type": "STRING",
            "enum": ["strong_hire", "hire", "borderline", "no_hire"],
        },
        "strengths": {"type": "ARRAY", "items": {"type": "STRING"}},
        "gaps": {"type": "ARRAY", "items": {"type": "STRING"}},
        "topic_breakdown": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "topic": {"type": "STRING"},
                    "assessment": {"type": "STRING"},
                    "score": {"type": "NUMBER"},
                },
                "required": ["topic", "assessment", "score"],
            },
        },
        "next_steps": {"type": "ARRAY", "items": {"type": "STRING"}},
    },
    "required": ["summary", "recommendation", "strengths", "gaps", "topic_breakdown"],
}


def report_prompt(
    *,
    role_title: str,
    profile: dict,
    transcript: str,
    average_score: float,
    coverage: list[str],
) -> str:
    covered = "\n".join(f"- {c}" for c in coverage) or "- (none)"
    return f"""TASK: report

Write the screening report for this {role_title} interview.

CANDIDATE PROFILE
{profile}

TOPICS COVERED
{covered}

MEAN SCORE ACROSS ANSWERS: {average_score:.2f} / 5

TRANSCRIPT (question, expected points, answer, grade)
{transcript[:20000]}

Rules:
- `summary`: 3-5 sentences a hiring manager can act on. Reference specific answers.
- `recommendation`: consistent with the scores - do not recommend `hire` on a mean below 3.
- `strengths` / `gaps`: evidence-based, each tied to something the candidate actually said.
- `topic_breakdown`: one entry per topic asked, with the score they earned on it.
- `next_steps`: what a follow-up interview should probe, or what to verify.
"""
