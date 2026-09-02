"use client";

import { useEffect, useState } from "react";
import { ApiError, api } from "@/lib/api";
import type { InterviewSession, Report } from "@/lib/types";
import { RetrievalTracePanel } from "./RetrievalTrace";

const RECOMMENDATION_LABEL: Record<string, string> = {
  strong_hire: "Strong hire",
  hire: "Hire",
  borderline: "Borderline",
  no_hire: "No hire",
};

const RECOMMENDATION_CLASS: Record<string, string> = {
  strong_hire: "card-forest",
  hire: "card-forest",
  borderline: "card-dark",
  no_hire: "card-coral",
};

export function ReportStage({
  sessionId,
  onReset,
}: {
  sessionId: string;
  onReset: () => void;
}) {
  const [report, setReport] = useState<Report | null>(null);
  const [session, setSession] = useState<InterviewSession | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    Promise.all([api.getReport(sessionId), api.getSession(sessionId)])
      .then(([r, s]) => {
        setReport(r);
        setSession(s);
      })
      .catch((err: ApiError) => setError(err.message));
  }, [sessionId]);

  if (error) return <div className="banner banner-error">{error}</div>;

  if (!report || !session) {
    return (
      <div className="stack stack-md">
        <div className="shimmer" style={{ width: "30%", height: 20 }} />
        <div className="card stack stack-sm">
          <div className="shimmer" style={{ width: "70%" }} />
          <div className="shimmer" style={{ width: "55%" }} />
        </div>
      </div>
    );
  }

  const stats = report.stats ?? {};

  return (
    <div className="stack stack-xl">
      <section className="stack stack-md">
        <span className="eyebrow">Screening report</span>
        <h1>
          {session.candidate_name || "Candidate"} · {session.role_title}
        </h1>
        <div className="grid-2">
          <div className={`${RECOMMENDATION_CLASS[report.recommendation] ?? "card-dark"} stack stack-sm`}>
            <span className="eyebrow" style={{ color: "rgba(255,255,255,0.7)" }}>
              Recommendation
            </span>
            <h2 style={{ color: "#fff" }}>
              {RECOMMENDATION_LABEL[report.recommendation] ?? report.recommendation}
            </h2>
            <p style={{ color: "rgba(255,255,255,0.8)" }}>
              Mean score {report.overall_score.toFixed(2)} / 5 across {stats.answered ?? 0}{" "}
              answers ({stats.follow_ups ?? 0} adaptive follow-up
              {(stats.follow_ups ?? 0) === 1 ? "" : "s"}).
            </p>
          </div>
          <div className="card stack stack-sm">
            <span className="eyebrow">Summary</span>
            <p>{report.summary}</p>
          </div>
        </div>
      </section>

      <section className="grid-2">
        <div className="card stack stack-sm">
          <span className="eyebrow">Strengths</span>
          {report.strengths.length === 0 && <p className="muted">None recorded.</p>}
          {report.strengths.map((s) => (
            <p key={s} style={{ fontSize: 14 }}>
              ✓ {s}
            </p>
          ))}
        </div>
        <div className="card stack stack-sm">
          <span className="eyebrow">Gaps</span>
          {report.gaps.length === 0 && <p className="muted">None recorded.</p>}
          {report.gaps.map((g) => (
            <p key={g} style={{ fontSize: 14 }}>
              · {g}
            </p>
          ))}
        </div>
      </section>

      <section className="card stack stack-md">
        <span className="eyebrow">Per-topic performance</span>
        {report.topic_breakdown.map((topic) => (
          <div key={topic.topic} className="stack stack-xs">
            <div className="meter-row">
              <span style={{ color: "var(--espresso)" }}>{topic.topic}</span>
              <span className="score-bar">
                <span style={{ width: `${(topic.score / 5) * 100}%` }} />
              </span>
              <strong style={{ textAlign: "right" }}>{topic.score.toFixed(1)}</strong>
            </div>
            {topic.assessment && (
              <p className="muted" style={{ fontSize: 13 }}>
                {topic.assessment}
              </p>
            )}
          </div>
        ))}
      </section>

      {report.next_steps.length > 0 && (
        <section className="card-cream stack stack-sm">
          <span className="eyebrow">Suggested next steps</span>
          <ul style={{ margin: 0, paddingLeft: 18, fontSize: 14 }}>
            {report.next_steps.map((step) => (
              <li key={step}>{step}</li>
            ))}
          </ul>
        </section>
      )}

      <section className="stack stack-md">
        <div className="row-between">
          <h2>Transcript</h2>
          <span className="muted" style={{ fontSize: 13 }}>
            grounded in {(stats.knowledge_sources_used ?? []).join(", ") || "the role corpus"}
          </span>
        </div>
        {session.questions
          .filter((q) => q.answer)
          .map((q) => (
            <div key={q.id} className="card stack stack-sm">
              <div className="row-between">
                <div className="chip-row">
                  <span className="chip chip-solid">Q{q.position}</span>
                  <span className="chip">{q.topic}</span>
                  <span className="chip">{q.difficulty}</span>
                  {q.origin === "follow_up" && (
                    <span className="chip chip-coral">follow-up</span>
                  )}
                </div>
                <strong>{q.answer?.evaluation?.score ?? "-"}/5</strong>
              </div>
              <p style={{ fontSize: 16, color: "var(--espresso)" }}>{q.text}</p>
              <div className="card-soft">
                <p style={{ fontSize: 14, whiteSpace: "pre-wrap" }}>{q.answer?.text}</p>
              </div>
              {q.answer?.evaluation?.feedback && (
                <p className="muted" style={{ fontSize: 15 }}>
                  <strong>Grader.</strong> {q.answer.evaluation.feedback}
                </p>
              )}
              <RetrievalTracePanel trace={q.retrieval} rationale={q.rationale} />
            </div>
          ))}
      </section>

      <div className="row">
        <button className="btn btn-primary" onClick={onReset}>
          Screen another candidate
        </button>
        <a className="btn btn-secondary" href={`/api/sessions/${sessionId}`} onClick={(e) => e.preventDefault()}>
          Session id: {sessionId.slice(0, 8)}
        </a>
      </div>
    </div>
  );
}
