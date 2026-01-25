# src/core/config.py
from __future__ import annotations

import os
from pathlib import Path
from dotenv import load_dotenv

# backend/ directory (…/app/backend)
BACKEND_DIR = Path(__file__).resolve().parents[2]
ROOT_DIR = BACKEND_DIR.parent

# Load .env from backend root first, then repo root if present.
backend_env = BACKEND_DIR / ".env"
root_env = ROOT_DIR / ".env"
if backend_env.exists():
    load_dotenv(backend_env)
if root_env.exists():
    load_dotenv(root_env)

# Now env vars work everywhere (uvicorn, eval, scripts)
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-70b-versatile")
PRIVATE_DATA_DIR = os.getenv("PRIVATE_DATA_DIR", "../../data/private")
INDEX_PERSIST_DIR = os.getenv("INDEX_PERSIST_DIR", "storage")
EMBED_MODEL = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")
TOP_K = int(os.getenv("TOP_K", "5"))


AUTHOR_LINKS = {
    "HealthEcho": "https://healthecho.vercel.app/",
    "GitHub": "https://github.com/RudraJoshi25",
    "LinkedIn": "https://linkedin.com/in/rudrajoshi25",
    "Research Paper (IEEE Xplore)": "https://ieeexplore.ieee.org/document/10065338",
}
