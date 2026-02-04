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

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile") 

# Data
PRIVATE_DATA_DIR = os.getenv("PRIVATE_DATA_DIR", "../../data/private")
RAG_STORAGE_DIR = os.getenv("RAG_STORAGE_DIR", "storage")
EMBED_MODEL = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")

TOP_K = int(os.getenv("TOP_K", "8"))

# Retrieval toggles
RAG_VECTOR_ENABLED = os.getenv("RAG_VECTOR_ENABLED", "true").lower() in {"1", "true", "yes"}
RAG_HYBRID_ENABLED = os.getenv("RAG_HYBRID_ENABLED", "true").lower() in {"1", "true", "yes"}

# Safety
INJECTION_GUARD_ENABLED = os.getenv("INJECTION_GUARD_ENABLED", "true").lower() in {"1", "true", "yes"}
