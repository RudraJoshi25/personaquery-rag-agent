# syntax=docker/dockerfile:1

# ---------------------------------------------------------------------------
# PersonaQuery backend
#
# The embedding stack runs on FastEmbed (ONNX Runtime) — no torch, no
# sentence-transformers. An earlier version of this file installed CPU-only
# torch and pre-downloaded via SentenceTransformer(); after the move to
# fastembed that pre-download line would have FAILED the build outright
# (sentence-transformers is no longer installed) and torch was ~200MB of
# dead weight in the image.
#
# Models are still baked in at build time: downloading them on first request
# would blow the cold-start budget and can hit HF rate limits. Both models
# are needed — the dense embedder AND the ColBERT reranker.
# ---------------------------------------------------------------------------

FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.hf \
    FASTEMBED_CACHE_PATH=/app/.hf/fastembed \
    TOKENIZERS_PARALLELISM=false

WORKDIR /app

# Minimal build deps; removed in the same layer to keep the image small.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --upgrade pip && pip install -r requirements.txt

# Pre-download both ONNX models into the image. EMBED_MODEL uses the same
# alias mapping as src/rag/embedder.py (all-MiniLM-L6-v2 -> bge-small);
# keep these in sync with _FASTEMBED_ALIASES if the default ever changes.
ARG DENSE_MODEL=BAAI/bge-small-en-v1.5
ARG COLBERT_MODEL=colbert-ir/colbertv2.0
RUN python -c "\
from fastembed import TextEmbedding, LateInteractionTextEmbedding; \
TextEmbedding('${DENSE_MODEL}'); \
LateInteractionTextEmbedding('${COLBERT_MODEL}')"

COPY . .

# Non-root: Render and Fly both run containers unprivileged by preference.
RUN useradd -m -u 10001 appuser && chown -R appuser:appuser /app
USER appuser

ENV PORT=8000
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD curl -fsS "http://localhost:${PORT}/health" || exit 1

# Render injects $PORT; the shell form is required so it expands.
CMD ["sh", "-c", "uvicorn src.main:app --host 0.0.0.0 --port ${PORT} --workers 1"]
