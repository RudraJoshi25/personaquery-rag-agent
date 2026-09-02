# syntax=docker/dockerfile:1
#
# NOTE: this is an OPTIONAL/secondary deploy path (local reproducibility,
# running the eval harness in a clean env, or future migration groundwork).
# Production today deploys via Render's native Python buildpack directly
# from requirements.txt - this Dockerfile does NOT replace that, and
# switching the live Render service to it is a separate decision.
#
# Build context is this directory (app/backend), matching how Render's
# native buildpack already treats it as the install root:
#   docker build -f app/backend/Dockerfile -t personaquery-backend app/backend
#   docker run -p 8000:8000 -e GROQ_API_KEY=... personaquery-backend

ARG PYTHON_VERSION=3.12-slim
ARG EMBED_MODEL=all-MiniLM-L6-v2

FROM python:${PYTHON_VERSION} AS builder
WORKDIR /build
ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --prefix=/install -r requirements.txt

# Pre-download the sentence-transformers embedding model at build time so
# container startup doesn't depend on network access to huggingface.co and
# the first request isn't slowed by a cold model download.
FROM builder AS model-cache
ARG EMBED_MODEL
ENV PYTHONPATH=/install/lib/python3.12/site-packages
ENV HF_HOME=/model-cache
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('${EMBED_MODEL}')"

FROM python:${PYTHON_VERSION} AS runtime
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/app/.cache/huggingface \
    PORT=8000
RUN addgroup --system app && adduser --system --ingroup app app
COPY --from=builder /install /usr/local
COPY --from=model-cache /model-cache /app/.cache/huggingface
COPY src/ ./src
COPY storage/ ./storage
RUN chown -R app:app /app
USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import os, urllib.request; urllib.request.urlopen('http://127.0.0.1:' + os.environ.get('PORT', '8000') + '/health/live')" || exit 1
CMD ["sh", "-c", "uvicorn src.main:app --host 0.0.0.0 --port ${PORT:-8000} --proxy-headers --forwarded-allow-ips='*'"]
