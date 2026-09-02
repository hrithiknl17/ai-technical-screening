"""Resume ingestion and profiling.

Parsing is deliberately two-layered:

* a deterministic pass (regex/keyword) that always runs and always produces
  something usable - contact stripping, skill dictionary match, years of
  experience from date ranges;
* an LLM pass that produces the richer structured profile.

The deterministic pass is not a fallback afterthought: its output is merged into
the LLM result, so a skill the model misses but the text plainly states still
reaches the planner, and the system degrades to something sensible when the LLM
is unavailable.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime

from sqlalchemy.orm import Session

from app.core.errors import NotFoundError, UpstreamError, ValidationError
from app.db.models import Resume
from app.llm import get_llm
from app.llm.prompts import RESUME_PROFILE_SCHEMA, resume_profile_prompt
from app.rag.loaders import load_bytes

logger = logging.getLogger(__name__)

_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")
_PHONE_RE = re.compile(r"(?:\+?\d[\d\-\s()]{7,}\d)")
_YEAR_RANGE_RE = re.compile(r"(19|20)\d{2}\s*[-–—to]{1,3}\s*((19|20)\d{2}|present|now)", re.I)

# A compact dictionary is enough to catch what matters for these roles and keeps
# the deterministic pass dependency-free.
_SKILL_DICTIONARY = {
    "python", "java", "c++", "go", "rust", "sql", "r", "scala", "typescript",
    "pytorch", "tensorflow", "keras", "jax", "scikit-learn", "sklearn", "xgboost",
    "lightgbm", "pandas", "numpy", "spark", "hadoop", "airflow", "dbt", "kafka",
    "fastapi", "flask", "django", "node.js", "react", "next.js", "postgresql",
    "mysql", "mongodb", "redis", "elasticsearch", "docker", "kubernetes", "aws",
    "gcp", "azure", "terraform", "mlflow", "weights & biases", "huggingface",
    "transformers", "langchain", "llamaindex", "faiss", "pinecone", "chroma",
    "opencv", "nlp", "computer vision", "reinforcement learning", "recommender",
    "time series", "a/b testing", "feature engineering", "rag", "fine-tuning",
    "distributed training", "model serving", "etl", "data pipeline",
}


def _skill_pattern(skill: str) -> re.Pattern[str]:
    """Whole-token match, so "r" does not match every word and "go" does not match "google"."""
    return re.compile(rf"(?<![a-z0-9+#.]){re.escape(skill)}(?![a-z0-9+#])", re.I)


_SKILL_PATTERNS = {skill: _skill_pattern(skill) for skill in _SKILL_DICTIONARY}


def _extract_deterministic(text: str) -> dict:
    lowered = text.lower()
    skills = sorted({s for s, pattern in _SKILL_PATTERNS.items() if pattern.search(lowered)})

    years: float = 0.0
    spans: list[tuple[int, int]] = []
    current_year = datetime.now().year
    for match in _YEAR_RANGE_RE.finditer(text):
        raw = match.group(0)
        start = int(re.search(r"(19|20)\d{2}", raw).group(0))  # type: ignore[union-attr]
        end_match = re.search(r"((19|20)\d{2})\s*$", raw.strip())
        end = int(end_match.group(1)) if end_match else current_year
        if 1980 <= start <= end <= current_year + 1:
            spans.append((start, end))
    if spans:
        # union of ranges, so overlapping roles are not double counted
        spans.sort()
        merged: list[list[int]] = [list(spans[0])]
        for start, end in spans[1:]:
            if start <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], end)
            else:
                merged.append([start, end])
        years = float(sum(end - start for start, end in merged))

    name = ""
    for line in text.splitlines()[:6]:
        stripped = line.strip()
        if (
            2 <= len(stripped.split()) <= 5
            and not _EMAIL_RE.search(stripped)
            and not _PHONE_RE.search(stripped)
            and stripped.replace(" ", "").replace(".", "").replace("-", "").isalpha()
        ):
            name = stripped.title()
            break

    return {"skills": skills, "years_experience": years, "candidate_name": name}


def _merge_profiles(llm_profile: dict, deterministic: dict) -> dict:
    profile = dict(llm_profile or {})
    profile.setdefault("skills", [])
    profile.setdefault("technologies", [])
    known = {s.lower() for s in profile["skills"]} | {t.lower() for t in profile["technologies"]}
    for skill in deterministic["skills"]:
        if skill.lower() not in known:
            profile["technologies"].append(skill)
    if not profile.get("candidate_name"):
        profile["candidate_name"] = deterministic["candidate_name"]
    if not profile.get("years_experience"):
        profile["years_experience"] = deterministic["years_experience"]
    profile.setdefault("seniority", "junior")
    profile.setdefault("domains", [])
    profile.setdefault("projects", [])
    profile.setdefault("strength_signals", [])
    profile.setdefault("gap_signals", [])
    profile.setdefault("headline", "")
    return profile


class ResumeService:
    def __init__(self, db: Session) -> None:
        self.db = db

    def create_from_upload(
        self,
        *,
        data: bytes,
        filename: str,
        content_type: str | None,
        role_title: str,
        max_bytes: int,
    ) -> Resume:
        if not data:
            raise ValidationError("The uploaded file is empty.")
        if len(data) > max_bytes:
            raise ValidationError(
                f"Resume is too large ({len(data) // 1024} KB). Limit is {max_bytes // 1024} KB."
            )
        document = load_bytes(data, filename, content_type)
        text = document.text
        if len(text) < 120:
            raise ValidationError(
                "Could not read enough text from this resume. "
                "If it is a scanned PDF, upload a text-based version."
            )
        profile = self.build_profile(text, role_title)
        resume = Resume(
            filename=filename,
            content_type=content_type or "application/octet-stream",
            raw_text=text,
            profile=profile,
        )
        self.db.add(resume)
        self.db.flush()
        logger.info("stored resume %s (%s chars)", resume.id, len(text))
        return resume

    def create_from_text(self, *, text: str, role_title: str) -> Resume:
        if len(text.strip()) < 120:
            raise ValidationError("Resume text is too short to profile (minimum 120 characters).")
        profile = self.build_profile(text, role_title)
        resume = Resume(
            filename="pasted-resume.txt",
            content_type="text/plain",
            raw_text=text.strip(),
            profile=profile,
        )
        self.db.add(resume)
        self.db.flush()
        return resume

    def build_profile(self, text: str, role_title: str) -> dict:
        deterministic = _extract_deterministic(text)
        try:
            llm_profile = get_llm().generate_json(
                resume_profile_prompt(text, role_title),
                RESUME_PROFILE_SCHEMA,
                temperature=0.2,
            )
        except UpstreamError as exc:
            logger.warning("LLM profiling failed, using deterministic profile only: %s", exc)
            llm_profile = {
                "headline": "Profile extracted without LLM assistance",
                "skills": deterministic["skills"],
                "technologies": [],
                "strength_signals": [],
                "gap_signals": ["Automated profiling was unavailable for this resume"],
            }
        return _merge_profiles(llm_profile, deterministic)

    def get(self, resume_id: str) -> Resume:
        resume = self.db.get(Resume, resume_id)
        if resume is None:
            raise NotFoundError(f"Resume '{resume_id}' not found.")
        return resume
