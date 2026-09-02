"use client";

const STEPS = ["Role & resume", "Interview", "Report"] as const;

export function Stepper({ current }: { current: 0 | 1 | 2 }) {
  return (
    <div className="stepper">
      {STEPS.map((label, index) => (
        <div key={label} style={{ display: "contents" }}>
          <div
            className={`step ${index === current ? "active" : ""} ${
              index < current ? "done" : ""
            }`}
          >
            <span className="step-dot">{index < current ? "✓" : index + 1}</span>
            <span>{label}</span>
          </div>
          {index < STEPS.length - 1 && <span className="step-sep" />}
        </div>
      ))}
    </div>
  );
}
