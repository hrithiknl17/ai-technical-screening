"use client";

import { useEffect, useRef, useState } from "react";
import { ApiError, api } from "@/lib/api";
import type { Resume, Role } from "@/lib/types";

interface Props {
  onStarted: (sessionId: string) => void;
}

export function SetupStage({ onStarted }: Props) {
  const [roles, setRoles] = useState<Role[]>([]);
  const [roleSlug, setRoleSlug] = useState<string>("");
  const [mode, setMode] = useState<"upload" | "paste">("upload");
  const [file, setFile] = useState<File | null>(null);
  const [pasted, setPasted] = useState("");
  const [questionCount, setQuestionCount] = useState(6);
  const [resume, setResume] = useState<Resume | null>(null);
  const [dragging, setDragging] = useState(false);
  const [busy, setBusy] = useState<"" | "profiling" | "starting">("");
  const [error, setError] = useState<string | null>(null);
  const fileInput = useRef<HTMLInputElement>(null);

  useEffect(() => {
    api
      .listRoles()
      .then((data) => {
        setRoles(data);
        const ready = data.find((r) => r.knowledge_base_ready);
        if (ready) setRoleSlug(ready.slug);
      })
      .catch((err: ApiError) => setError(err.message));
  }, []);

  const role = roles.find((r) => r.slug === roleSlug);

  async function profile() {
    setError(null);
    setBusy("profiling");
    try {
      const result =
        mode === "upload" && file
          ? await api.uploadResume(file, roleSlug)
          : await api.submitResumeText(pasted, roleSlug);
      setResume(result);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Could not read that resume.");
    } finally {
      setBusy("");
    }
  }

  async function start() {
    if (!resume) return;
    setError(null);
    setBusy("starting");
    try {
      const session = await api.createSession(resume.id, roleSlug, questionCount);
      onStarted(session.id);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Could not start the interview.");
      setBusy("");
    }
  }

  const canProfile =
    roleSlug !== "" && (mode === "upload" ? file !== null : pasted.trim().length > 120);

  return (
    <div className="stack stack-xl">
      <section className="stack stack-md">
        <span className="eyebrow">Step 1 — target role</span>
        <h1>What are we screening for?</h1>
        <p className="lede">
          Pick a role. Its interview is generated from that role&apos;s own textbook corpus —
          nothing is asked that the corpus cannot ground.
        </p>
        <div className="grid-3">
          {roles.length === 0 && (
            <>
              <div className="card">
                <div className="shimmer" style={{ width: "60%" }} />
              </div>
              <div className="card">
                <div className="shimmer" style={{ width: "70%" }} />
              </div>
              <div className="card">
                <div className="shimmer" style={{ width: "50%" }} />
              </div>
            </>
          )}
          {roles.map((r) => (
            <button
              key={r.slug}
              type="button"
              disabled={!r.knowledge_base_ready}
              onClick={() => setRoleSlug(r.slug)}
              className={`role-card ${roleSlug === r.slug ? "selected" : ""} ${
                r.knowledge_base_ready ? "" : "disabled"
              }`}
            >
              <h4>{r.title}</h4>
              <p style={{ fontSize: 15 }}>{r.description}</p>
              <div style={{ marginTop: "auto", paddingTop: 12 }}>
                <span className={`chip ${r.knowledge_base_ready ? "chip-mint" : ""}`}>
                  {r.knowledge_base_ready
                    ? `${r.indexed_chunks.toLocaleString()} chunks indexed`
                    : "knowledge base not indexed"}
                </span>
              </div>
            </button>
          ))}
        </div>
        {role && (
          <div className="card-soft stack stack-sm">
            <span className="eyebrow">Corpus for this role</span>
            <div className="chip-row">
              {role.corpus.map((book) => (
                <span key={book} className="chip">
                  {book}
                </span>
              ))}
            </div>
            <span className="eyebrow" style={{ marginTop: 8 }}>
              Competencies the interview will cover
            </span>
            <ul style={{ margin: 0, paddingLeft: 18, color: "var(--charcoal)", fontSize: 15 }}>
              {role.competencies.map((c) => (
                <li key={c}>{c}</li>
              ))}
            </ul>
          </div>
        )}
      </section>

      <hr className="hairline" />

      <section className="stack stack-md">
        <span className="eyebrow">Step 2 — the candidate</span>
        <h2>Upload a resume</h2>
        <p className="lede">
          It is parsed into a structured profile: skills, technologies, seniority and the
          gaps worth probing. That profile decides which topics get retrieved and how hard
          the questions are.
        </p>

        <div className="row">
          <button
            className={`btn btn-sm ${mode === "upload" ? "btn-primary" : "btn-secondary"}`}
            onClick={() => setMode("upload")}
          >
            Upload a file
          </button>
          <button
            className={`btn btn-sm ${mode === "paste" ? "btn-primary" : "btn-secondary"}`}
            onClick={() => setMode("paste")}
          >
            Paste text
          </button>
        </div>

        {mode === "upload" ? (
          <div
            className={`dropzone ${dragging ? "dragging" : ""}`}
            onClick={() => fileInput.current?.click()}
            onDragOver={(e) => {
              e.preventDefault();
              setDragging(true);
            }}
            onDragLeave={() => setDragging(false)}
            onDrop={(e) => {
              e.preventDefault();
              setDragging(false);
              const dropped = e.dataTransfer.files?.[0];
              if (dropped) setFile(dropped);
            }}
          >
            <input
              ref={fileInput}
              type="file"
              accept=".pdf,.txt,.md"
              hidden
              onChange={(e) => setFile(e.target.files?.[0] ?? null)}
            />
            <p style={{ color: "var(--espresso)", fontSize: 15 }}>
              {file ? file.name : "Drop a PDF, TXT or Markdown resume here"}
            </p>
            <p style={{ fontSize: 14, marginTop: 6 }}>
              {file
                ? `${(file.size / 1024).toFixed(0)} KB — click to replace`
                : "or click to browse"}
            </p>
          </div>
        ) : (
          <textarea
            rows={10}
            placeholder="Paste the resume text here (minimum 120 characters)…"
            value={pasted}
            onChange={(e) => setPasted(e.target.value)}
          />
        )}

        <div className="row-between">
          <label className="row" style={{ fontSize: 15, color: "var(--muted)" }}>
            Questions
            <input
              type="range"
              min={3}
              max={10}
              value={questionCount}
              onChange={(e) => setQuestionCount(Number(e.target.value))}
            />
            <strong style={{ color: "var(--espresso)" }}>{questionCount}</strong>
          </label>
          <button
            className="btn btn-secondary"
            disabled={!canProfile || busy !== ""}
            onClick={profile}
          >
            {busy === "profiling" ? (
              <>
                <span className="spinner spinner-dark" /> Reading resume…
              </>
            ) : (
              "Parse resume"
            )}
          </button>
        </div>
      </section>

      {error && <div className="banner banner-error">{error}</div>}

      {resume && (
        <section className="stack stack-md">
          <hr className="hairline" />
          <span className="eyebrow">Step 3 — extracted profile</span>
          <div className="grid-2">
            <div className="card stack stack-sm">
              <div className="row-between">
                <h3>{resume.profile.candidate_name || "Candidate"}</h3>
                <span className="chip chip-solid">{resume.profile.seniority}</span>
              </div>
              {resume.profile.headline && <p>{resume.profile.headline}</p>}
              <div className="chip-row">
                {[...resume.profile.technologies, ...resume.profile.skills]
                  .slice(0, 16)
                  .map((s) => (
                    <span key={s} className="chip">
                      {s}
                    </span>
                  ))}
              </div>
              <p className="muted" style={{ fontSize: 13 }}>
                {resume.profile.years_experience > 0
                  ? `≈ ${resume.profile.years_experience} year(s) of dated experience · `
                  : ""}
                {resume.filename} · {resume.characters.toLocaleString()} characters parsed
              </p>
            </div>
            <div className="card-cream stack stack-sm">
              <span className="eyebrow">What the interview will probe</span>
              <ul style={{ margin: 0, paddingLeft: 18, fontSize: 15, color: "var(--espresso)" }}>
                {resume.profile.strength_signals.slice(0, 3).map((s) => (
                  <li key={s}>
                    <strong>Strength.</strong> {s}
                  </li>
                ))}
                {resume.profile.gap_signals.slice(0, 3).map((s) => (
                  <li key={s}>
                    <strong>Gap.</strong> {s}
                  </li>
                ))}
              </ul>
            </div>
          </div>
          <div>
            <button className="btn btn-primary" disabled={busy !== ""} onClick={start}>
              {busy === "starting" ? (
                <>
                  <span className="spinner" /> Planning the interview…
                </>
              ) : (
                `Start the ${role?.title ?? ""} interview →`
              )}
            </button>
          </div>
        </section>
      )}
    </div>
  );
}
