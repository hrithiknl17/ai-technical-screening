# Single-image deployment: the API serves the built UI from its own origin, so
# there is one process, one port and no CORS to configure.
#
#   docker build -t grounded .
#   docker run -p 8000:8000 -e GEMINI_API_KEY=... grounded
#
# The image expects a prebuilt index in data/vector_store (committed, ~9 MB).
# To rebuild it from the source PDFs instead:
#   docker run --rm -v "$PWD/data:/app/data" grounded \
#     sh -c "cd backend && python -m scripts.fetch_books && python -m scripts.ingest_kb --all"

# --- 1. build the frontend as a static export -------------------------------
FROM node:22-alpine AS ui
WORKDIR /ui
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
ENV STATIC_EXPORT=1
RUN npm run build

# --- 2. runtime -------------------------------------------------------------
FROM python:3.12-slim AS runtime
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/app/.cache/huggingface

WORKDIR /app

COPY backend/requirements.txt backend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt

COPY backend/ backend/
COPY data/vector_store/ data/vector_store/
COPY data/samples/ data/samples/
COPY --from=ui /ui/out/ frontend/out/

# Warm the embedding model into the image so the first request is not a download.
RUN python -c "from model2vec import StaticModel; StaticModel.from_pretrained('minishlab/potion-base-8M')"

EXPOSE 8000
WORKDIR /app/backend
CMD ["sh", "-c", "python -m uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
