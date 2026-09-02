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


def _bool(name: str, default: str) -> bool:
    return os.getenv(name, default).lower() in {"1", "true", "yes"}


ENVIRONMENT = os.getenv("ENVIRONMENT", "production")
STRICT_STARTUP_CHECKS = _bool("STRICT_STARTUP_CHECKS", "true")

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

# Data
PRIVATE_DATA_DIR = os.getenv("PRIVATE_DATA_DIR", "../../data/private")
RAG_STORAGE_DIR = os.getenv("RAG_STORAGE_DIR", "storage")
EMBED_MODEL = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")

TOP_K = int(os.getenv("TOP_K", "8"))

# Retrieval toggles
RAG_VECTOR_ENABLED = _bool("RAG_VECTOR_ENABLED", "true")
RAG_HYBRID_ENABLED = _bool("RAG_HYBRID_ENABLED", "true")

# Safety
INJECTION_GUARD_ENABLED = _bool("INJECTION_GUARD_ENABLED", "true")
DEFENSIVE_PII_REDACTION_ENABLED = _bool("DEFENSIVE_PII_REDACTION_ENABLED", "true")

# PII redaction allowlist: the owner's own already-public contact details.
# Anything else matching these patterns in ingested text gets redacted.
OWNER_EMAIL = os.getenv("OWNER_EMAIL", "rudrajoshi.cs@gmail.com")
OWNER_PHONE = os.getenv("OWNER_PHONE", "+61466988797")

# CORS
CORS_ALLOW_ALL = _bool("CORS_ALLOW_ALL", "false")
CORS_ALLOWED_ORIGINS = [o.strip() for o in os.getenv("CORS_ALLOWED_ORIGINS", "").split(",") if o.strip()]
CORS_ALLOW_ORIGIN_REGEX = os.getenv("CORS_ALLOW_ORIGIN_REGEX", "").strip() or None

# Rate limiting (per client IP; slowapi "limits" syntax e.g. "10/minute;200/day")
RATE_LIMIT_CHAT = os.getenv("RATE_LIMIT_CHAT", "10/minute;200/day")
RATE_LIMIT_INTERVIEW_START = os.getenv("RATE_LIMIT_INTERVIEW_START", "5/minute;50/day")
RATE_LIMIT_INTERVIEW_ANSWER = os.getenv("RATE_LIMIT_INTERVIEW_ANSWER", "10/minute;200/day")

# Input validation
CHAT_MAX_QUESTION_LENGTH = int(os.getenv("CHAT_MAX_QUESTION_LENGTH", "2000"))
INTERVIEW_ANSWER_MAX_LENGTH = int(os.getenv("INTERVIEW_ANSWER_MAX_LENGTH", "4000"))

# Interview session hygiene
INTERVIEW_SESSION_TTL_SECONDS = int(os.getenv("INTERVIEW_SESSION_TTL_SECONDS", "1800"))
INTERVIEW_MAX_SESSIONS = int(os.getenv("INTERVIEW_MAX_SESSIONS", "500"))

# Logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = os.getenv("LOG_FORMAT", "json")
LOG_INCLUDE_QUESTION_PREVIEW = _bool("LOG_INCLUDE_QUESTION_PREVIEW", "true")
LOG_QUESTION_PREVIEW_CHARS = int(os.getenv("LOG_QUESTION_PREVIEW_CHARS", "120"))


def validate_startup_config() -> None:
    """Fail fast on configuration that would make this a broken deployment
    rather than a working one. Called once at process startup, before the
    FastAPI app is constructed."""
    import logging

    logger = logging.getLogger("personaquery.startup")

    if not GROQ_API_KEY:
        raise RuntimeError(
            "GROQ_API_KEY is not set. The service cannot answer any questions without it - "
            "refusing to start rather than deploying a broken instance."
        )

    if STRICT_STARTUP_CHECKS and CORS_ALLOW_ALL and ENVIRONMENT == "production":
        raise RuntimeError(
            "CORS_ALLOW_ALL=true is set together with ENVIRONMENT=production. This is a dev-only "
            "escape hatch and must not be enabled in production. Set STRICT_STARTUP_CHECKS=false "
            "to override in an emergency."
        )

    if not CORS_ALLOWED_ORIGINS and not CORS_ALLOW_ALL and ENVIRONMENT == "production":
        logger.warning(
            "CORS_ALLOWED_ORIGINS is empty in production - the API will reject all browser "
            "cross-origin requests until it is configured. This is fail-safe, not a startup blocker."
        )

    storage_dir = Path(RAG_STORAGE_DIR)
    required = ["chunks.jsonl", "vectors.npy", "bm25.json"]
    if not all((storage_dir / f).exists() for f in required):
        logger.warning(
            "RAG storage files not found under %s - /health/ready will report not-ready until "
            "ingestion has been run.",
            storage_dir,
        )
