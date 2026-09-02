export interface Role {
  slug: string;
  title: string;
  description: string;
  competencies: string[];
  corpus: string[];
  knowledge_base_ready: boolean;
  indexed_chunks: number;
}

export interface Project {
  name: string;
  summary: string;
  technologies: string[];
}

export interface ResumeProfile {
  candidate_name: string;
  headline: string;
  years_experience: number;
  seniority: string;
  skills: string[];
  technologies: string[];
  domains: string[];
  projects: Project[];
  strength_signals: string[];
  gap_signals: string[];
}

export interface Resume {
  id: string;
  filename: string;
  characters: number;
  profile: ResumeProfile;
}

export interface RetrievedChunk {
  id: string;
  source: string;
  citation: string;
  section: string;
  page_start: number;
  page_end: number;
  score: number;
  dense_score: number;
  lexical_score: number;
  excerpt: string;
}

export interface RetrievalTrace {
  query: string;
  chunks: RetrievedChunk[];
}

export interface Evaluation {
  score: number;
  verdict: string;
  covered_points: string[];
  missed_points: string[];
  feedback: string;
}

export interface Answer {
  text: string;
  time_taken_seconds: number | null;
  evaluation: Evaluation | null;
  submitted_at: string | null;
}

export interface Question {
  id: string;
  position: number;
  text: string;
  topic: string;
  difficulty: string;
  question_type: string;
  origin: string;
  rationale: string;
  expected_points: string[];
  retrieval: RetrievalTrace;
  answer: Answer | null;
}

export interface PlanTopic {
  topic: string;
  why: string;
  difficulty: string;
  query: string;
  question_type: string;
  status: string;
}

export interface InterviewSession {
  id: string;
  role: string;
  role_title: string;
  status: "in_progress" | "completed";
  candidate_name: string | null;
  planned_questions: number;
  answered_questions: number;
  focus_summary: string;
  topics: PlanTopic[];
  profile: ResumeProfile | null;
  questions: Question[];
  current_question: Question | null;
  created_at: string | null;
  completed_at: string | null;
}

export interface TopicScore {
  topic: string;
  assessment: string;
  score: number;
}

export interface Report {
  session_id: string;
  overall_score: number;
  recommendation: string;
  summary: string;
  strengths: string[];
  gaps: string[];
  topic_breakdown: TopicScore[];
  next_steps: string[];
  stats: {
    answered?: number;
    planned?: number;
    follow_ups?: number;
    average_score?: number;
    score_by_position?: number[];
    difficulty_mix?: Record<string, number>;
    median_seconds_per_answer?: number | null;
    knowledge_sources_used?: string[];
    topics_covered?: string[];
  };
  generated_at: string | null;
}

export interface AnswerAccepted {
  evaluation: Evaluation;
  session: InterviewSession;
}
