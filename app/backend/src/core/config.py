# src/core/config.py
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env from backend root if present
BACKEND_ROOT = Path(__file__).resolve().parents[2]  # .../backend
ENV_PATH = BACKEND_ROOT / ".env"
if ENV_PATH.exists():
    load_dotenv(ENV_PATH)

# Data
PRIVATE_DATA_DIR = os.getenv("PRIVATE_DATA_DIR", "../../data/private")
RAG_STORAGE_DIR = os.getenv("RAG_STORAGE_DIR", "storage")
LOG_DIR = Path(os.getenv("RAG_LOG_DIR", "logs"))

TOP_K = int(os.getenv("TOP_K", "8"))

# ONNX model cache. FastEmbed's default is the system temp dir, which on
# Render's native runtime is NOT carried over from the build: every cold
# start re-downloaded both models. Inside the project it survives the build.
MODEL_CACHE_DIR = os.getenv("FASTEMBED_CACHE_PATH", str(BACKEND_ROOT / ".models"))

# Retrieval toggles
RAG_VECTOR_ENABLED = os.getenv("RAG_VECTOR_ENABLED", "true").lower() in {"1", "true", "yes"}

# Safety
INJECTION_GUARD_ENABLED = os.getenv("INJECTION_GUARD_ENABLED", "true").lower() in {"1", "true", "yes"}

# API hardening
MAX_QUESTION_CHARS = int(os.getenv("MAX_QUESTION_CHARS", "300"))
RATE_LIMIT_REQUESTS = int(os.getenv("RATE_LIMIT_REQUESTS", "20"))
RATE_LIMIT_WINDOW_S = float(os.getenv("RATE_LIMIT_WINDOW_S", "60"))
# Only true behind a reverse proxy that appends X-Forwarded-For (Render does).
TRUST_PROXY = os.getenv("TRUST_PROXY", "false").lower() in {"1", "true", "yes"}
# Proxies that append to X-Forwarded-For after the client. Production is
# "client, Cloudflare, Render-internal" -> 2.
TRUSTED_PROXY_HOPS = int(os.getenv("TRUSTED_PROXY_HOPS", "2"))
