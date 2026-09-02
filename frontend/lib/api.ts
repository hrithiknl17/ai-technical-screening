/**
 * Typed client for the backend API.
 *
 * One place owns the base URL and the error shape, so components handle
 * `ApiError` rather than raw fetch failures.
 */

import type {
  AnswerAccepted,
  InterviewSession,
  Question,
  Report,
  Resume,
  Role,
} from "./types";

// "/api" when the UI is served by the API itself; an absolute URL in dev.
const BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL ?? "/api";

export class ApiError extends Error {
  code: string;
  status: number;
  details: Record<string, unknown>;

  constructor(status: number, code: string, message: string, details: Record<string, unknown> = {}) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.code = code;
    this.details = details;
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`${BASE_URL}${path}`, {
      ...init,
      headers: {
        ...(init?.body instanceof FormData ? {} : { "Content-Type": "application/json" }),
        ...(init?.headers ?? {}),
      },
    });
  } catch {
    throw new ApiError(
      0,
      "network_error",
      `Cannot reach the backend at ${BASE_URL}. Is the API running?`
    );
  }

  if (response.status === 204) {
    return null as T;
  }

  const raw = await response.text();
  const body = raw ? JSON.parse(raw) : null;

  if (!response.ok) {
    throw new ApiError(
      response.status,
      body?.code ?? "error",
      body?.message ?? `Request failed with status ${response.status}`,
      body?.details ?? {}
    );
  }
  return body as T;
}

export const api = {
  listRoles: () => request<Role[]>("/roles"),

  uploadResume: (file: File, role: string) => {
    const form = new FormData();
    form.append("file", file);
    form.append("role", role);
    return request<Resume>("/resumes", { method: "POST", body: form });
  },

  submitResumeText: (text: string, role: string) =>
    request<Resume>("/resumes/text", {
      method: "POST",
      body: JSON.stringify({ text, role }),
    }),

  createSession: (resumeId: string, role: string, questionCount: number) =>
    request<InterviewSession>("/sessions", {
      method: "POST",
      body: JSON.stringify({ resume_id: resumeId, role, question_count: questionCount }),
    }),

  getSession: (sessionId: string) => request<InterviewSession>(`/sessions/${sessionId}`),

  nextQuestion: (sessionId: string) =>
    request<Question | null>(`/sessions/${sessionId}/next-question`, { method: "POST" }),

  submitAnswer: (sessionId: string, questionId: string, answer: string, seconds?: number) =>
    request<AnswerAccepted>(`/sessions/${sessionId}/answers`, {
      method: "POST",
      body: JSON.stringify({
        question_id: questionId,
        answer,
        time_taken_seconds: seconds ?? null,
      }),
    }),

  completeSession: (sessionId: string) =>
    request<Report>(`/sessions/${sessionId}/complete`, { method: "POST" }),

  getReport: (sessionId: string) => request<Report>(`/sessions/${sessionId}/report`),

  searchKnowledge: (role: string, query: string, topK = 5) =>
    request<{ role: string; query: string; results: Question["retrieval"]["chunks"] }>(
      "/knowledge/search",
      { method: "POST", body: JSON.stringify({ role, query, top_k: topK }) }
    ),
};
