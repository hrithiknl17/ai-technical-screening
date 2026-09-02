# Grounded — RAG-driven technical screening

![Python](https://img.shields.io/badge/python-3.11+-1f2328)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-1f2328)
![Next.js](https://img.shields.io/badge/Next.js-15-1f2328)
![Tests](https://img.shields.io/badge/tests-22%20offline-1f2328)
![License](https://img.shields.io/badge/license-MIT-1f2328)

An AI interviewer that reads a candidate's resume, picks the topics worth testing for a
target role, retrieves the supporting material from that role's **textbook corpus**, and
asks questions that are grounded in what it retrieved — then grades the answers against
the same passages and writes a screening report.

Nothing in the interview is a stored question template. Every question carries the
retrieval query and the exact book passages (with page numbers) it was generated from, and
the UI shows them under **“Why this question”**.

```
resume ──► profile ──► interview plan ──► retrieval ──► question ──► answer ──► grade
                │             │               │            │           │          │
             skills        topics +        hybrid       grounded    rubric     adaptive
          technologies     queries        dense+BM25     + cited    scoring    follow-up
             gaps          per topic      + MMR                                    │
                                                                                   ▼
                                                                          report + transcript
```

**Documents.** [`docs/design-document.pdf`](docs/design-document.pdf) is the full design and
architecture write-up (problem framing, RAG decisions, data model, failure modes, trade-offs).
[`docs/submission-cover-sheet.pdf`](docs/submission-cover-sheet.pdf) is the one-page submission
sheet with the deliverables checklist.

**Contents.** [Quick start](#1-quick-start) · [Layout](#2-repository-layout) ·
[System flow](#3-system-flow) · [API](#4-api) · [Design decisions](#5-design-decisions) ·
[Operating notes](#6-operating-notes) · [Tests](#7-tests) · [Hosting](#8-hosting) ·
[Demo](#9-demo-video-shot-list) · [Next](#10-what-i-would-do-next)

---

## 1. Quick start

Requirements: Python 3.11+ and Node 18+. A `GEMINI_API_KEY` (Google AI Studio, free tier
is enough) is needed for real question generation; without one the system still runs
end-to-end on deterministic offline stubs.

```bash
# 1. backend deps
python -m venv .venv
.venv/Scripts/activate           # Windows;  source .venv/bin/activate on macOS/Linux
pip install -r backend/requirements.txt

# 2. configuration
cp .env.example .env             # then put your GEMINI_API_KEY in it

# 3. corpus: download the assignment's books, then index them
cd backend
python -m scripts.fetch_books            # writes data/knowledge_base/<role>/*.pdf
python -m scripts.ingest_kb --all        # chunk -> embed -> index  (see notes below)
python -m scripts.ingest_kb --status     # what is on disk and what is indexed

# 4. run the API
uvicorn app.main:app --reload --port 8000       # docs at http://127.0.0.1:8000/docs

# 5. run the UI (second terminal)
cd frontend
npm install
npm run dev                                      # http://localhost:3000
```

A sample resume to try is in `data/samples/sample_resume_ml.md`.

Ingestion of all six books (~3,900 indexed chunks) takes **under a minute** with the
default local embedder. To index only part of each book:

```bash
python -m scripts.ingest_kb --all --max-pages 200 --skip-leading 10
```

There is also a headless demo that drives the whole flow through the API and prints the
plan, the retrieval traces, the grades and the report — handy for a screen recording:

```bash
python -m scripts.demo_interview --role ai_ml_engineer --questions 3
```

### Running without an API key

```bash
OFFLINE_MODE=1 uvicorn app.main:app --port 8000
```

The embedder falls back to a deterministic hashing embedder and the LLM to a stub that
returns schema-shaped payloads built from the retrieved passages. Question quality is
obviously not comparable — the point is that the pipeline, storage and UI stay exercisable
(this is also how the test suite runs).

---

## 2. Repository layout

```
backend/
  app/
    main.py               app factory, CORS, request logging, error translation
    core/                 settings (env-driven), error hierarchy, logging
    api/routes/           health, roles, resumes, sessions, knowledge
    api/mappers.py        ORM -> DTO, so routes and services stay free of each other
    schemas/dto.py        the request/response contract
    services/             resume_service, interview_service, report_service
    rag/                  loaders, chunking, embeddings, vector_store, retriever, ingest
    llm/                  client interface, Gemini implementation, offline stub, prompts
    db/                   SQLAlchemy models and session handling
    domain/roles.py       role registry: corpus + competencies + fallback queries
  scripts/                fetch_books.py, ingest_kb.py, demo_interview.py
Dockerfile                one image: builds the UI, serves it from the API
render.yaml               Render Blueprint for the same image
  tests/                  offline end-to-end and unit tests
frontend/
  app/                    Next.js App Router: layout, page (stage machine), design tokens
  components/             SetupStage, InterviewStage, ReportStage, RetrievalTrace, Stepper
  lib/                    typed API client and shared types
data/
  knowledge_base/<role>/  corpus PDFs (not committed)
  vector_store/<role>/    the index: vectors.npy + records.jsonl + meta.json
  samples/                a sample resume
docs/
  design-document.pdf     design & architecture write-up (source: the .html next to it)
  submission-cover-sheet.pdf
```

---

## 3. System flow

1. **Candidate entry.** The UI lists roles and shows, per role, which corpus backs it and
   how many chunks are indexed. A role whose knowledge base is not indexed cannot be
   selected — the system refuses to interview from an empty corpus rather than
   hallucinating.
2. **Resume processing.** `POST /api/resumes` accepts PDF/TXT/MD. Two extraction layers
   run: a deterministic pass (skill dictionary with word-boundary matching, date-range
   union for years of experience, name heuristics) and an LLM pass that produces the
   structured profile. They are merged, the deterministic values filling anything the
   model left empty. The result is a profile with skills, technologies, seniority,
   strength signals and **gap signals**.
3. **Context construction.** `InterviewService._build_plan` asks the model for N topics
   given the profile, the role's competencies and the section headings actually present in
   the index. Each topic carries: why it was chosen (tied to something in the resume), a
   difficulty calibrated to seniority, a question type, and — importantly — a **retrieval
   query written in textbook vocabulary**, not interview phrasing. Query construction is
   therefore resume-driven and role-driven, which is exactly what the retrieval step needs.
4. **Knowledge retrieval.** Hybrid dense + BM25 search over the role's collection, MMR
   re-ranking for diversity, and chunk-level exclusion of everything already used earlier
   in the same interview (so question 5 is not a paraphrase of question 2).
5. **Question generation.** The retrieved passages, the candidate profile and the last few
   turns go into the question prompt. The model returns the question, its rationale, the
   3-5 points a strong answer must contain, and the citations it used.
6. **Interactive interview.** Answers are graded synchronously against the expected points
   *and the retrieved passages* — the grader sees the same ground truth the question came
   from. The candidate sees the score, what was covered and what was missed.
7. **Adaptation.** After each answer: the difficulty of the next planned topic moves up or
   down one step, and the grader may request a follow-up (budgeted at two per interview) —
   pushing deeper after a strong answer, or asking for specifics after a vague one.
8. **Final output.** The report combines statistics computed from stored grades (means,
   per-topic scores, difficulty mix, follow-up count, which books were used) with an LLM
   narrative. **The numbers never come from the model**; it is asked to interpret the
   evidence, not to produce the scores.

---

## 4. API

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/api/health` | liveness plus dependency status (DB, LLM, per-role index size) |
| `GET` | `/api/roles` | roles with corpus, competencies and index readiness |
| `GET` | `/api/roles/{slug}/knowledge` | index metadata for one role |
| `POST` | `/api/resumes` | multipart upload → parsed profile |
| `POST` | `/api/resumes/text` | same, from pasted text |
| `POST` | `/api/sessions` | start an interview (builds the plan, asks Q1) |
| `GET` | `/api/sessions/{id}` | full session state incl. transcript and traces |
| `POST` | `/api/sessions/{id}/next-question` | pending question, or generate the next (204 when exhausted) |
| `POST` | `/api/sessions/{id}/answers` | submit an answer → grade + updated session |
| `POST` | `/api/sessions/{id}/complete` | close the interview, build the report |
| `GET` | `/api/sessions/{id}/report` | fetch the report |
| `POST` | `/api/knowledge/search` | run a retrieval query directly (introspection/debugging) |

Errors are uniform: `{"code": ..., "message": ..., "details": {...}}`, produced by one
exception handler translating the domain error hierarchy (`ValidationError` → 422,
`NotFoundError` → 404, `ConflictError` → 409, `UpstreamError` → 502,
`KnowledgeBaseError` → 503). Interactive docs: `/docs`.

---

## 5. Design decisions

**Why a service layer instead of route logic.** Routes translate HTTP to method calls and
nothing else; `InterviewService` owns the lifecycle. That is what makes the adaptive rules
testable without a web server, and it is why the offline test suite can drive the entire
flow through the API in 1.7 seconds.

**Chunking: structure-aware packing, 1400 chars with 220 overlap.** Paragraph boundaries
first, sentence packing only when a paragraph is oversized, and a character overlap
started at a sentence boundary. 1400 characters is roughly a full textbook argument — a
definition plus the paragraph that motivates it — which is the unit a good interview
question is built from. Much smaller and the retrieved context is a fragment that cannot
support a question; much larger and the embedding averages several ideas and retrieval
gets vague. Chunks below 220 characters are folded into their predecessor instead of being
indexed as noise, and each chunk keeps its page range so questions can be cited.

**A quality filter before embedding.** Textbook PDFs extract tables of contents,
bibliographies, index pages and equation soup. Those embed fine and then surface as
"context", producing nonsense questions. `is_useful_chunk` rejects short blocks, blocks
with a low alphabetic ratio, dot-leader TOC lines and reference-list patterns. On this
corpus it drops ~8% of chunks — cheaper embeddings *and* better retrieval.

**Embeddings: a pluggable provider, defaulting to a local static model.** Three
implementations sit behind one interface (`EMBEDDING_PROVIDER`):

| provider | what it is | indexing 3,900 chunks | why you'd pick it |
| --- | --- | --- | --- |
| `local` *(default)* | Model2Vec static model (`potion-base-8M`, 256-d), CPU only | ~40 s | reproducible, no key, no quota |
| `gemini` | `gemini-embedding-001` at 768-d, task-typed | hours on the free tier | best retrieval quality |
| `hashing` | hashed n-grams, zero dependencies | seconds | tests, CI, air-gapped runs |

The `gemini` path is the better encoder — documents are embedded with
`RETRIEVAL_DOCUMENT` and queries with `RETRIEVAL_QUERY`, and that asymmetry measurably
helps — but its free tier shares a *global* per-minute embed quota, and a book-sized
corpus takes hours behind it. That is a poor default for something a reviewer has to be
able to reproduce, so the local static model is the default and the hosted encoder is one
environment variable away. Static embeddings lose ground on subtle paraphrase, which is
part of why retrieval is hybrid rather than dense-only. A collection records the dimension
it was built with, and the store rejects a query vector that does not match, so switching
provider tells you to re-ingest instead of silently returning nonsense.

**Retrieval: hybrid dense + BM25, then MMR.** Dense embeddings match paraphrase
("regularisation" ≈ "penalising complexity"); BM25 matches the exact token a resume brags
about ("XGBoost", "EM algorithm"). Both are min-max normalised per query and blended
(`DENSE_WEIGHT`, default 0.7). MMR (λ = 0.65) then trades a little relevance for coverage,
because books repeat themselves and a pure top-k returns five paraphrases of one paragraph
— which yields five near-identical questions. Chunks already used in the session are
excluded outright.

**Vector store: a persistent flat index behind an interface.** The corpora are books:
10³–10⁴ chunks. An exact normalised matrix-vector product over 10k × 768 floats is ~7
MFLOPs — well under a millisecond in NumPy — so an ANN index would add a dependency and an
approximation for no measurable gain. The partitioning that *does* matter is per role, and
that is explicit: one collection per role, so a question for one role can never retrieve
another role's corpus. `VectorStore` is a Protocol; swapping in Chroma/Qdrant/pgvector is
a one-class change, and the retriever would not notice.

**Storage: SQLAlchemy + SQLite by default.** `DATABASE_URL` is the only thing standing
between this and Postgres. The schema mirrors the pipeline — resume → session → question →
answer → report — and every question row stores its full `retrieval_trace` (query, chunk
ids, scores, source, pages, excerpt). Traceability is a stored artefact, not a log line.

**Two-layer everything.** Resume parsing, grading and reporting each pair a deterministic
component with an LLM component. The deterministic half is always right and always
available; the LLM half adds judgement. When the LLM is unavailable the system degrades
instead of failing: profiling falls back to the dictionary pass, grading stores an
"ungraded" answer, and the report emits the statistics with a note.

**Prompts live in one module.** `app/llm/prompts.py` holds every prompt and its response
schema. Gemini's `responseSchema` constrains the output shape, which removes the usual
"parse JSON out of prose" failure mode; `parse_json` is still tolerant of fenced blocks as
a backstop.

**Frontend state.** Next.js App Router, one stage machine (`setup → interview → report`).
The backend is the source of truth for the session; the client stores only the session id
in `localStorage`, so a mid-interview refresh returns to the current question. The visual
language follows the design tokens in `DESIGN-airtable.md` (white canvas, dark ink,
signature colour cards, near-black pill CTA).

---

## 6. Operating notes

**Model choice.** `LLM_MODEL` defaults to `gemini-3.1-flash-lite`, not because it is the
strongest model but because free-tier *daily* caps differ enormously per model —
`gemini-2.5-flash` allows about 20 generate-content requests per day, which one interview
exhausts. Point `LLM_MODEL` at `gemini-2.5-flash` (or a paid key) for sharper questions.

**Free-tier rate limits.** Both the LLM and the hosted embedder are rate limited per
minute, and the embedding ceiling is a *global* quota, not a per-key one. Two mechanisms
deal with it: every client honours a server-supplied `retryDelay` over its own backoff,
the LLM client spaces its own calls (`LLM_REQUESTS_PER_MINUTE`), and the hosted embedder
paces itself with AIMD — the inter-batch pause shrinks 10% after
each success and grows 60% after a 429 — so a long ingest survives unattended instead of
dying on the first burst. When the LLM is rate-limited past its retries the interview does
not break: the answer is stored ungraded and the report falls back to statistics, which is
the degradation path described above.

**Ingestion is idempotent.** A role's collection is rebuilt from whatever is in
`data/knowledge_base/<role>/`. Add a book, re-run, done.

**Configuration** is entirely environment-driven (`.env.example` lists every knob):
model names, embedding dimension, chunk size/overlap, `RETRIEVAL_TOP_K`, `DENSE_WEIGHT`,
`RETRIEVAL_MMR_LAMBDA`, question counts, CORS origins, `OFFLINE_MODE`.

**Corpus note.** `fetch_books.py` pulls the books listed in the assignment. The host for
Burkov's *Hundred-Page Machine Learning Book* serves an invalid TLS certificate, so that
one usually fails to download; the AI/ML role is still backed by Mitchell and *ML for
Absolute Beginners*. Drop the PDF into `data/knowledge_base/ai_ml_engineer/` and re-run
ingestion if you have a copy.

---

## 7. Tests

```bash
cd backend
python -m pytest -q          # 22 tests, fully offline, no API key needed
```

They cover the PDF text repair, chunk sizing/overlap, the quality filter, embedder
determinism, BM25 ranking, vector-store round-trip and citation format, hybrid retrieval
grounding and no-repeat behaviour; the orchestration rules (plan statuses advance and
persist, the interview exhausts, strong answers earn a budgeted follow-up and push the
difficulty ladder up, completion drops an unanswered question); and the full API lifecycle
plus the error contracts (409 on double answer, 404 on unknown role/session, 422 on a
too-short resume).

---

## 8. Hosting

The production build is **single-origin**: `next build` with `STATIC_EXPORT=1` emits a static
site into `frontend/out`, and the API mounts it at `/` (after the routers, so `/api/*` always
wins). One process, one port, no CORS configuration, and the client calls a relative `/api`.

```bash
cd frontend && STATIC_EXPORT=1 npm run build      # -> frontend/out
cd ../backend && python -m uvicorn app.main:app --port 8000
# UI on http://127.0.0.1:8000/ , API on /api , docs on /docs
```

**Container.** The `Dockerfile` does both stages and bakes the embedding model into the image
so the first request is not a model download:

```bash
docker build -t grounded .
docker run -p 8000:8000 -e GEMINI_API_KEY=... grounded
```

**Render.** `render.yaml` is a Blueprint for the free tier — Docker runtime, health check on
`/api/health`, `GEMINI_API_KEY` set in the dashboard. Any Docker host works the same way.

The vector index (`data/vector_store/`, ~9 MB) **is committed**, so a clone runs immediately
without downloading 90 MB of textbooks. The PDFs themselves are not; run
`scripts.fetch_books` then `scripts.ingest_kb --all` to rebuild the index from source.

**Temporary public URL.** For a demo or a review link, a Cloudflare quick tunnel needs no
account and no DNS:

```bash
cloudflared tunnel --url http://127.0.0.1:8000
```

It prints a `https://<random>.trycloudflare.com` URL that proxies to the local app. The URL is
random per run and the tunnel dies with the process — fine for a demo, not a deployment. While
it is open the API is reachable by anyone holding the link, and every interview spends your
Gemini quota.

---

## 9. Demo video shot list

The flow that shows the most in the least time:

1. `python -m scripts.ingest_kb --status` — the corpora on disk and the indexed chunk
   counts per role.
2. The role picker: each role shows its books and its index size; an unindexed role cannot
   be started.
3. Paste or upload the sample resume → the extracted profile, with the **gaps** the
   interview will probe.
4. Start the interview → the plan sidebar (topics, difficulty, question type) and the first
   question, which names the candidate's own technology.
5. Expand **“Why this question”** — the retrieval query, the cited pages, and the hybrid /
   dense / BM25 scores behind each passage. This is the part worth lingering on.
6. Answer well → the grade with covered/missed points, and watch the next topic's
   difficulty move up in the sidebar; answer badly → the follow-up probe appears.
7. End the interview → the report: recommendation, per-topic meters, transcript with every
   question's provenance still attached.
8. Optionally `/docs` for the API surface and `pytest -q` for the suite.

---

## 10. What I would do next

- Move ingestion to a background worker with a progress endpoint, so a corpus can be added
  from the UI rather than the CLI.
- Cache question generation per (topic, chunk-set) to make demos cheap and repeatable.
- Add a second grader pass for disagreement detection — one model grading, one auditing.
- Voice answers (the answer path is already a plain string; only the input surface changes).

---

## 11. License

MIT — see [`LICENSE`](LICENSE).
