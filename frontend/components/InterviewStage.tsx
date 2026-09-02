"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { ApiError, api } from "@/lib/api";
import type { Evaluation, InterviewSession, Question } from "@/lib/types";
import { RetrievalTracePanel } from "./RetrievalTrace";

const VERDICT_COLOUR: Record<string, string> = {
  excellent: "var(--positive)",
  strong: "var(--positive)",
  partial: "var(--caution)",
  weak: "var(--negative)",
  incorrect: "var(--negative)",
  ungraded: "var(--on-surface-variant)",
};

interface Props {
  sessionId: string;
  onFinished: () => void;
  onReset: () => void;
}

export function InterviewStage({ sessionId, onFinished, onReset }: Props) {
  const [session, setSession] = useState<InterviewSession | null>(null);
  const [question, setQuestion] = useState<Question | null>(null);
  const [draft, setDraft] = useState("");
  const [evaluation, setEvaluation] = useState<Evaluation | null>(null);
  const [busy, setBusy] = useState<"" | "loading" | "grading" | "next" | "finishing">("loading");
  const [error, setError] = useState<string | null>(null);
  const startedAt = useRef<number>(Date.now());

  const load = useCallback(async () => {
    try {
      const data = await api.getSession(sessionId);
      setSession(data);
      setQuestion(data.current_question);
      startedAt.current = Date.now();
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Could not load the session.");
    } finally {
      setBusy("");
    }
  }, [sessionId]);

  useEffect(() => {
    void load();
  }, [load]);

  async function submit() {
    if (!question || draft.trim().length === 0) return;
    setError(null);
    setBusy("grading");
    try {
      const seconds = (Date.now() - startedAt.current) / 1000;
      const result = await api.submitAnswer(sessionId, question.id, draft, seconds);
      setEvaluation(result.evaluation);
      setSession(result.session);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Could not submit that answer.");
    } finally {
      setBusy("");
    }
  }

  async function next() {
    setError(null);
    setBusy("next");
    try {
      const upcoming = await api.nextQuestion(sessionId);
      if (!upcoming) {
        await finish();
        return;
      }
      setQuestion(upcoming);
      setEvaluation(null);
      setDraft("");
      startedAt.current = Date.now();
      setSession(await api.getSession(sessionId));
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Could not fetch the next question.");
    } finally {
      setBusy("");
    }
  }

  async function finish() {
    setBusy("finishing");
    try {
      await api.completeSession(sessionId);
      onFinished();
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Could not build the report.");
      setBusy("");
    }
  }

  const answered = session?.answered_questions ?? 0;
  const planned = session?.planned_questions ?? 0;
  const progress = useMemo(
    () => Math.min(100, planned === 0 ? 0 : (answered / planned) * 100),
    [answered, planned]
  );

  if (busy === "loading" && !session) {
    return (
      <div className="card stack stack-sm">
        <div className="shimmer" style={{ width: "35%" }} />
        <div className="shimmer" style={{ width: "80%", height: 24 }} />
        <div className="shimmer" style={{ width: "60%" }} />
      </div>
    );
  }

  return (
    <div className="interview-grid">
      <div className="stack stack-lg">
        <div className="row-between">
          <div className="row" style={{ gap: 8 }}>
            <span className="eyebrow">
              Question {question?.position ?? answered} of ~{planned}
            </span>
            {question?.origin === "follow_up" && (
              <span className="chip chip-coral">adaptive follow-up</span>
            )}
          </div>
          <button className="btn btn-ghost" onClick={onReset}>
            Abandon
          </button>
        </div>

        <div className="score-bar">
          <span style={{ width: `${progress}%` }} />
        </div>

        {question && (
          <div className="card stack stack-md">
            <div className="chip-row">
              <span className="chip chip-solid">{question.topic}</span>
              <span className="chip">{question.difficulty}</span>
              <span className="chip">{question.question_type}</span>
            </div>
            <p className="question-text">{question.text}</p>
            <RetrievalTracePanel trace={question.retrieval} rationale={question.rationale} />
          </div>
        )}

        {error && <div className="banner banner-error">{error}</div>}

        {!evaluation && question && (
          <div className="stack stack-sm">
            <textarea
              rows={9}
              placeholder="Type your answer. Reasoning counts for more than length."
              value={draft}
              onChange={(e) => setDraft(e.target.value)}
              onKeyDown={(e) => {
                if ((e.metaKey || e.ctrlKey) && e.key === "Enter") void submit();
              }}
            />
            <div className="row-between">
              <span className="muted" style={{ fontSize: 14 }}>
                {draft.trim().split(/\s+/).filter(Boolean).length} words · ⌘/Ctrl + Enter to
                submit
              </span>
              <button
                className="btn btn-primary"
                disabled={draft.trim().length === 0 || busy !== ""}
                onClick={submit}
              >
                {busy === "grading" ? (
                  <>
                    <span className="spinner" /> Grading against the source…
                  </>
                ) : (
                  "Submit answer"
                )}
              </button>
            </div>
          </div>
        )}

        {evaluation && (
          <div className="card stack stack-md">
            <div className="row-between">
              <h3>Grade</h3>
              <span
                className="chip"
                style={{
                  color: VERDICT_COLOUR[evaluation.verdict] ?? "var(--espresso)",
                  borderColor: "currentColor",
                }}
              >
                {evaluation.score}/5 · {evaluation.verdict}
              </span>
            </div>
            <p>{evaluation.feedback}</p>
            <div className="grid-2">
              <div className="stack stack-xs">
                <span className="eyebrow">Covered</span>
                {evaluation.covered_points.length === 0 && (
                  <p className="muted" style={{ fontSize: 14 }}>
                    Nothing from the expected points.
                  </p>
                )}
                {evaluation.covered_points.map((point) => (
                  <p key={point} style={{ fontSize: 14 }}>
                    ✓ {point}
                  </p>
                ))}
              </div>
              <div className="stack stack-xs">
                <span className="eyebrow">Missed</span>
                {evaluation.missed_points.length === 0 && (
                  <p className="muted" style={{ fontSize: 14 }}>
                    Nothing material missing.
                  </p>
                )}
                {evaluation.missed_points.map((point) => (
                  <p key={point} style={{ fontSize: 14 }}>
                    · {point}
                  </p>
                ))}
              </div>
            </div>
            <div className="row">
              <button className="btn btn-primary" disabled={busy !== ""} onClick={next}>
                {busy === "next" || busy === "finishing" ? (
                  <>
                    <span className="spinner" /> Preparing…
                  </>
                ) : (
                  "Next question →"
                )}
              </button>
              <button className="btn btn-secondary" disabled={busy !== ""} onClick={finish}>
                End &amp; see report
              </button>
            </div>
          </div>
        )}
      </div>

      <aside className="stack stack-md">
        <div className="card-soft stack stack-sm">
          <span className="eyebrow">Interview plan</span>
          {session?.focus_summary && (
            <p style={{ fontSize: 14 }}>{session.focus_summary}</p>
          )}
          <div className="stack stack-xs">
            {session?.topics.map((topic, index) => (
              <div key={`${topic.topic}-${index}`} className="row" style={{ gap: 8 }}>
                <span
                  className="step-dot"
                  style={{
                    background:
                      topic.status === "asked"
                        ? "var(--blush)"
                        : "var(--surface-container-lowest)",
                    borderColor:
                      topic.status === "asked"
                        ? "rgba(240, 98, 146, 0.45)"
                        : "var(--hairline-strong)",
                    color: topic.status === "asked" ? "var(--rose-ink)" : undefined,
                  }}
                >
                  {index + 1}
                </span>
                <div>
                  <p style={{ fontSize: 14, color: "var(--espresso)" }}>{topic.topic}</p>
                  <p className="muted" style={{ fontSize: 14 }}>
                    {topic.difficulty} · {topic.question_type}
                  </p>
                </div>
              </div>
            ))}
          </div>
        </div>

        {session?.profile && (
          <div className="card stack stack-sm">
            <span className="eyebrow">Candidate</span>
            <p style={{ fontSize: 15, color: "var(--espresso)" }}>
              {session.profile.candidate_name || "Candidate"} · {session.profile.seniority}
            </p>
            <div className="chip-row">
              {session.profile.technologies.slice(0, 10).map((tech) => (
                <span key={tech} className="chip">
                  {tech}
                </span>
              ))}
            </div>
          </div>
        )}

        {session && session.questions.filter((q) => q.answer).length > 0 && (
          <div className="card stack stack-sm">
            <span className="eyebrow">Answered</span>
            {session.questions
              .filter((q) => q.answer)
              .map((q) => (
                <div key={q.id} className="row-between" style={{ fontSize: 14 }}>
                  <span className="muted" style={{ maxWidth: 190 }}>
                    Q{q.position} · {q.topic}
                  </span>
                  <strong>{q.answer?.evaluation?.score ?? "-"}/5</strong>
                </div>
              ))}
          </div>
        )}
      </aside>
    </div>
  );
}
