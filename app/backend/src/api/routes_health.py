# src/api/routes_health.py
from __future__ import annotations

import logging
from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from src.core.config import GROQ_API_KEY, RAG_STORAGE_DIR

logger = logging.getLogger("personaquery.health")

router = APIRouter()

_embedder_loadable_cache: bool | None = None


def _store_loaded() -> bool:
    storage_dir = Path(RAG_STORAGE_DIR)
    return all((storage_dir / f).exists() for f in ["chunks.jsonl", "vectors.npy", "bm25.json"])


def _embedder_loadable() -> bool:
    """Cached after first successful check - never reload the model per
    health check, that would burn real time on every /health/ready hit."""
    global _embedder_loadable_cache
    if _embedder_loadable_cache:
        return True
    try:
        from src.rag.retrieval.search import _get_model  # loads/caches the singleton

        _get_model()
        _embedder_loadable_cache = True
    except Exception:
        logger.exception("embedder_load_check_failed")
        _embedder_loadable_cache = False
    return _embedder_loadable_cache


def _readiness_payload() -> tuple[dict, bool]:
    checks = {
        "store_loaded": _store_loaded(),
        "embedder_loadable": _embedder_loadable(),
        "groq_key_present": bool(GROQ_API_KEY),
    }
    ok = all(checks.values())
    return {"status": "ok" if ok else "degraded", "checks": checks}, ok


@router.get("/health/live")
def health_live():
    # Never fails on a fixable-without-restart condition (missing env var,
    # missing storage) - a Render restart wouldn't fix those anyway.
    return {"status": "ok"}


@router.get("/health/ready")
def health_ready():
    payload, ok = _readiness_payload()
    return JSONResponse(content=payload, status_code=200 if ok else 503)


@router.get("/health")
def health():
    # Alias to /health/ready for backward compatibility with whatever
    # health-check path is already configured outside this repo (e.g. in
    # Render's dashboard).
    return health_ready()
