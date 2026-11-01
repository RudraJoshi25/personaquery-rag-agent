# syntax=docker/dockerfile:1

# ---------------------------------------------------------------------------
# PersonaQuery backend (Google Cloud Run)
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

# OMP_NUM_THREADS / MALLOC_ARENA_MAX: ONNX Runtime otherwise spawns a thread
# pool and glibc an arena per core, which costs memory for no gain at
# --workers 1 on a 1-vCPU instance.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.hf \
    FASTEMBED_CACHE_PATH=/app/.hf/fastembed \
    TOKENIZERS_PARALLELISM=false \
    OMP_NUM_THREADS=1 \
    MALLOC_ARENA_MAX=2

WORKDIR /app

# curl for HEALTHCHECK only (local Docker; Cloud Run ignores it). Every
# Python dep ships a wheel, so no compiler.
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --upgrade pip && pip install -r requirements.txt

# Non-root user created BEFORE the model download, so the models are written
# as appuser and no `chown -R` layer duplicates them (that layer used to add
# ~200MB). Only the empty /app dir needs chowning here.
RUN useradd -m -u 10001 appuser && chown appuser:appuser /app
USER appuser

# Pre-download both ONNX models into the image. EMBED_MODEL uses the same
# alias mapping as src/rag/embedder.py (all-MiniLM-L6-v2 -> bge-small);
# keep these in sync with _FASTEMBED_ALIASES if the default ever changes.
ARG DENSE_MODEL=BAAI/bge-small-en-v1.5
ARG COLBERT_MODEL=answerdotai/answerai-colbert-small-v1
RUN python -c "\
from fastembed import TextEmbedding, LateInteractionTextEmbedding; \
TextEmbedding('${DENSE_MODEL}'); \
LateInteractionTextEmbedding('${COLBERT_MODEL}')"

COPY --chown=appuser:appuser . .

# Cloud Run sets $PORT (8080); 8000 is the local default.
ENV PORT=8000
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD curl -fsS "http://localhost:${PORT}/health" || exit 1

# Shell form so $PORT expands. No --forwarded-allow-ips: the app reads
# X-Forwarded-For itself (TRUST_PROXY / TRUSTED_PROXY_HOPS).
CMD ["sh", "-c", "exec uvicorn src.main:app --host 0.0.0.0 --port ${PORT} --workers 1"]
