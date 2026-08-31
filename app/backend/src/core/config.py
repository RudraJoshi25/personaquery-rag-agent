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

# Generation. Anthropic is the only provider; see src/rag/llm_anthropic.py
# for why the previous Groq/Cerebras/NVIDIA failover chain was removed.
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
ANTHROPIC_CHAT_MODEL = os.getenv("ANTHROPIC_CHAT_MODEL", "claude-sonnet-5")

# Data
PRIVATE_DATA_DIR = os.getenv("PRIVATE_DATA_DIR", "../../data/private")
RAG_STORAGE_DIR = os.getenv("RAG_STORAGE_DIR", "storage")
EMBED_MODEL = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")

TOP_K = int(os.getenv("TOP_K", "8"))

# Retrieval toggles
RAG_VECTOR_ENABLED = os.getenv("RAG_VECTOR_ENABLED", "true").lower() in {"1", "true", "yes"}
RAG_HYBRID_ENABLED = os.getenv("RAG_HYBRID_ENABLED", "true").lower() in {"1", "true", "yes"}

# Vector store backend: "chroma" (persistent, indexed) or "numpy" (legacy in-RAM matrix)
VECTOR_BACKEND = os.getenv("VECTOR_BACKEND", "chroma").lower()
CHROMA_DIR = os.getenv("CHROMA_DIR", os.path.join(RAG_STORAGE_DIR, "chroma"))
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "personaquery")

# Safety
INJECTION_GUARD_ENABLED = os.getenv("INJECTION_GUARD_ENABLED", "true").lower() in {"1", "true", "yes"}

# API hardening
MAX_QUESTION_CHARS = int(os.getenv("MAX_QUESTION_CHARS", "2000"))
RATE_LIMIT_REQUESTS = int(os.getenv("RATE_LIMIT_REQUESTS", "20"))
RATE_LIMIT_WINDOW_S = float(os.getenv("RATE_LIMIT_WINDOW_S", "60"))
