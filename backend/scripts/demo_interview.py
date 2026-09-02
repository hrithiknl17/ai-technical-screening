"""Headless end-to-end demo against a running API.

    uvicorn app.main:app --port 8000          # in one terminal
    python -m scripts.demo_interview --role ai_ml_engineer --questions 3

Useful for verifying a deployment (or narrating the pipeline in a demo video)
without clicking through the UI. Answers are canned; the point is to show the
plan, the retrieval traces, the grades and the final report.
"""

from __future__ import annotations

import argparse
import sys
import textwrap
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

DEFAULT_RESUME = Path(__file__).resolve().parents[2] / "data" / "samples" / "sample_resume_ml.md"

# Deliberately generic answers: the point of the smoke run is to exercise the
# pipeline, and a mediocre answer also shows the grader doing its job.
CANNED_ANSWERS = [
    "Overfitting is when the hypothesis fits noise in the training sample: training error "
    "keeps falling while error on held-out data rises. In the churn model I watched the gap "
    "between train and validation AUC, used early stopping on a validation fold, capped tree "
    "depth and used L2 regularisation, and I always compared against a simpler logistic "
    "baseline before shipping.",
    "I use k-fold cross validation rather than a single split because a single estimate has "
    "high variance; averaging over folds gives a tighter estimate of true error. For the "
    "ranking work I used grouped folds so documents from one session never straddled the "
    "split, which would otherwise leak information and flatter the metric.",
    "For imbalanced churn data accuracy is useless, so I looked at precision-recall AUC and "
    "calibration. I chose the operating threshold from the cost of a retention offer versus "
    "the margin on a retained customer, then verified the lift in an A/B test rather than "
    "trusting the offline metric alone.",
]


def wrap(text: str, indent: str = "    ") -> str:
    return textwrap.fill(text, width=92, initial_indent=indent, subsequent_indent=indent)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a full interview against the API.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/api")
    parser.add_argument("--role", default="ai_ml_engineer")
    parser.add_argument("--questions", type=int, default=3)
    parser.add_argument("--resume", type=Path, default=DEFAULT_RESUME)
    args = parser.parse_args()

    client = httpx.Client(base_url=args.base_url, timeout=900)

    health = client.get("/health").json()
    print(
        f"API {health['status']} · llm {health['llm']['model']} · "
        f"embeddings {health['embeddings']['model']} ({health['embeddings']['dimensions']}d)"
    )
    print(f"indexed chunks: {health['knowledge_base']}\n")

    resume_text = args.resume.read_text(encoding="utf-8")
    resume = client.post("/resumes/text", json={"text": resume_text, "role": args.role}).json()
    profile = resume["profile"]
    print(f"PROFILE  {profile['candidate_name'] or 'candidate'} · {profile['seniority']} · "
          f"{profile['years_experience']} yrs")
    print(wrap("technologies: " + ", ".join(profile["technologies"][:12])))
    print(wrap("gaps: " + "; ".join(profile["gap_signals"][:2])) + "\n")

    session = client.post(
        "/sessions",
        json={"resume_id": resume["id"], "role": args.role, "question_count": args.questions},
    ).json()
    print(f"PLAN  {session['focus_summary']}")
    for index, topic in enumerate(session["topics"], start=1):
        print(f"  {index}. {topic['topic']}  [{topic['difficulty']} · {topic['question_type']}]")
        print(wrap(topic["why"], "     "))
    print()

    question = session["current_question"]
    answered = 0
    while question is not None:
        print(f"--- Q{question['position']} [{question['topic']} · {question['difficulty']}"
              f"{' · follow-up' if question['origin'] == 'follow_up' else ''}]")
        print(wrap(question["text"]))
        print(f"    retrieved for: “{question['retrieval']['query']}”")
        for chunk in question["retrieval"]["chunks"][:3]:
            print(f"      · {chunk['citation']}  (hybrid {chunk['score']:.3f})")

        answer = CANNED_ANSWERS[answered % len(CANNED_ANSWERS)]
        print(wrap(f"ANSWER: {answer}", "    > "))
        graded = client.post(
            f"/sessions/{session['id']}/answers",
            json={"question_id": question["id"], "answer": answer, "time_taken_seconds": 90},
        ).json()["evaluation"]
        print(f"    GRADE {graded['score']}/5 ({graded['verdict']})")
        print(wrap(graded["feedback"], "      "))
        answered += 1

        response = client.post(f"/sessions/{session['id']}/next-question")
        question = response.json() if response.status_code == 200 else None
        print()

    report = client.post(f"/sessions/{session['id']}/complete").json()
    print("=== REPORT ===")
    print(f"{report['recommendation'].upper()} · mean {report['overall_score']}/5")
    print(wrap(report["summary"]))
    for topic in report["topic_breakdown"]:
        print(f"  {topic['score']:.1f}  {topic['topic']}")
    print("  sources used: " + ", ".join(report["stats"].get("knowledge_sources_used", [])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
