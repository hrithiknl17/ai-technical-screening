"use client";

import type { RetrievalTrace as Trace } from "@/lib/types";

/**
 * The provenance panel: the retrieval query and the exact passages the question
 * was generated from. This is what makes a generated question auditable.
 */
export function RetrievalTracePanel({
  trace,
  rationale,
}: {
  trace: Trace;
  rationale?: string;
}) {
  if (!trace?.chunks?.length) return null;
  return (
    <details className="card-soft">
      <summary>
        Why this question — {trace.chunks.length} retrieved passage
        {trace.chunks.length === 1 ? "" : "s"}
      </summary>
      <div className="stack stack-sm" style={{ marginTop: 14 }}>
        {rationale && (
          <p style={{ fontSize: 14 }}>
            <strong style={{ color: "var(--espresso)" }}>Interviewer&apos;s reasoning. </strong>
            {rationale}
          </p>
        )}
        <p className="mono" style={{ color: "var(--on-surface-variant)" }}>
          retrieval query → “{trace.query}”
        </p>
        <div className="trace">
          {trace.chunks.map((chunk, index) => (
            <div key={chunk.id || index} className="trace-item">
              <div className="row" style={{ flexWrap: "wrap", gap: 8 }}>
                <span className="chip chip-coral">{chunk.citation}</span>
                <span className="mono muted">
                  hybrid {chunk.score.toFixed(3)} · dense {chunk.dense_score.toFixed(3)} ·
                  bm25 {chunk.lexical_score.toFixed(2)}
                </span>
              </div>
              {chunk.section && (
                <p className="muted" style={{ fontSize: 13, marginTop: 4 }}>
                  {chunk.section}
                </p>
              )}
              <p className="excerpt" style={{ marginTop: 6 }}>
                {chunk.excerpt}…
              </p>
            </div>
          ))}
        </div>
      </div>
    </details>
  );
}
