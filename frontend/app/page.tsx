"use client";

import { useCallback, useEffect, useState } from "react";
import { InterviewStage } from "@/components/InterviewStage";
import { ReportStage } from "@/components/ReportStage";
import { SetupStage } from "@/components/SetupStage";
import { Stepper } from "@/components/Stepper";

type Stage = "setup" | "interview" | "report";

const STORAGE_KEY = "grounded.session";

/**
 * The stage machine. Session id lives in localStorage so a refresh mid-interview
 * returns the candidate to the question they were on - the backend is the source
 * of truth for everything else.
 */
export default function Home() {
  const [stage, setStage] = useState<Stage>("setup");
  const [sessionId, setSessionId] = useState<string | null>(null);

  useEffect(() => {
    const stored = window.localStorage.getItem(STORAGE_KEY);
    if (stored) {
      const { id, stage: storedStage } = JSON.parse(stored) as { id: string; stage: Stage };
      setSessionId(id);
      setStage(storedStage);
    }
  }, []);

  const persist = useCallback((id: string | null, next: Stage) => {
    if (id) {
      window.localStorage.setItem(STORAGE_KEY, JSON.stringify({ id, stage: next }));
    } else {
      window.localStorage.removeItem(STORAGE_KEY);
    }
  }, []);

  const reset = useCallback(() => {
    persist(null, "setup");
    setSessionId(null);
    setStage("setup");
  }, [persist]);

  return (
    <main className="container container-wide stack stack-xl">
      <Stepper current={stage === "setup" ? 0 : stage === "interview" ? 1 : 2} />

      {stage === "setup" && (
        <SetupStage
          onStarted={(id) => {
            setSessionId(id);
            setStage("interview");
            persist(id, "interview");
          }}
        />
      )}

      {stage === "interview" && sessionId && (
        <InterviewStage
          sessionId={sessionId}
          onFinished={() => {
            setStage("report");
            persist(sessionId, "report");
          }}
          onReset={reset}
        />
      )}

      {stage === "report" && sessionId && <ReportStage sessionId={sessionId} onReset={reset} />}
    </main>
  );
}
