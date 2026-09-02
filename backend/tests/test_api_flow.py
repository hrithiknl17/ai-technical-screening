"""End-to-end API tests: the whole lifecycle against the offline stubs."""

from __future__ import annotations

from tests.conftest import RESUME_TEXT


def test_health_reports_dependencies(client):
    body = client.get("/api/health").json()
    assert body["status"] == "ok"
    assert body["llm"]["model"] == "offline-stub"


def test_roles_expose_knowledge_base_readiness(client, seeded_role):
    roles = client.get("/api/roles").json()
    by_slug = {r["slug"]: r for r in roles}
    assert by_slug[seeded_role]["knowledge_base_ready"] is True
    assert by_slug[seeded_role]["indexed_chunks"] > 0


def test_resume_profiling_extracts_technologies(client, seeded_role):
    response = client.post(
        "/api/resumes/text", json={"text": RESUME_TEXT, "role": seeded_role}
    )
    assert response.status_code == 201
    profile = response.json()["profile"]
    technologies = {t.lower() for t in profile["technologies"] + profile["skills"]}
    assert "pytorch" in technologies
    assert "fastapi" in technologies
    assert profile["years_experience"] >= 3


def test_full_interview_lifecycle(client, seeded_role):
    resume_id = client.post(
        "/api/resumes/text", json={"text": RESUME_TEXT, "role": seeded_role}
    ).json()["id"]

    session = client.post(
        "/api/sessions",
        json={"resume_id": resume_id, "role": seeded_role, "question_count": 3},
    ).json()
    assert session["status"] == "in_progress"
    assert session["current_question"]["text"]

    # every question must carry the passages it was generated from
    trace = session["current_question"]["retrieval"]
    assert trace["query"]
    assert trace["chunks"][0]["citation"].startswith("notes.md")

    answered = 0
    question = session["current_question"]
    while question and answered < 6:
        accepted = client.post(
            f"/api/sessions/{session['id']}/answers",
            json={
                "question_id": question["id"],
                "answer": (
                    "Overfitting happens when the hypothesis fits noise in the training "
                    "sample; I use a held-out validation split, pruning and cross "
                    "validation to detect it, and I compare against a simpler baseline."
                ),
                "time_taken_seconds": 42.0,
            },
        )
        assert accepted.status_code == 200
        assert accepted.json()["evaluation"]["score"] >= 0
        answered += 1

        response = client.post(f"/api/sessions/{session['id']}/next-question")
        question = response.json() if response.status_code == 200 else None

    report = client.post(f"/api/sessions/{session['id']}/complete").json()
    assert report["session_id"] == session["id"]
    assert report["stats"]["answered"] == answered
    assert len(report["topic_breakdown"]) >= 1

    final = client.get(f"/api/sessions/{session['id']}").json()
    assert final["status"] == "completed"
    assert all(q["answer"] is not None for q in final["questions"])


def test_double_answer_is_rejected(client, seeded_role):
    resume_id = client.post(
        "/api/resumes/text", json={"text": RESUME_TEXT, "role": seeded_role}
    ).json()["id"]
    session = client.post(
        "/api/sessions",
        json={"resume_id": resume_id, "role": seeded_role, "question_count": 3},
    ).json()
    question_id = session["current_question"]["id"]
    payload = {"question_id": question_id, "answer": "A first answer about bias variance."}

    assert client.post(f"/api/sessions/{session['id']}/answers", json=payload).status_code == 200
    conflict = client.post(f"/api/sessions/{session['id']}/answers", json=payload)
    assert conflict.status_code == 409
    assert conflict.json()["code"] == "conflict"


def test_unknown_role_and_session_return_structured_errors(client):
    missing_role = client.post(
        "/api/resumes/text", json={"text": RESUME_TEXT, "role": "astronaut"}
    )
    assert missing_role.status_code == 404
    assert missing_role.json()["code"] == "not_found"

    missing_session = client.get("/api/sessions/does-not-exist")
    assert missing_session.status_code == 404


def test_short_resume_is_rejected(client, seeded_role):
    response = client.post("/api/resumes/text", json={"text": "too short", "role": seeded_role})
    assert response.status_code == 422
    assert response.json()["code"] == "validation_error"


def test_knowledge_search_endpoint(client, seeded_role):
    response = client.post(
        "/api/knowledge/search",
        json={"role": seeded_role, "query": "backpropagation local minimum", "top_k": 2},
    )
    assert response.status_code == 200
    results = response.json()["results"]
    assert results and results[0]["source"] == "notes.md"
