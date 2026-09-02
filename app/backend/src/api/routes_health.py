# src/api/routes_health.py
from __future__ import annotations

import logging
import os
import time
from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from src.core.config import GROQ_API_KEY, RAG_STORAGE_DIR

logger = logging.getLogger("personaquery.health")

router = APIRouter()

# Success is cached permanently (the model is a process-lifetime singleton).
# Failure is NOT cached permanently - readiness must be able to recover on its
# own from a transient condition, because /health/live always returns 200 so
# nothing will ever restart this process on our behalf. Instead we throttle
# retries: the realistic failure mode is OOM loading a ~90MB model on a small
# instance, and re-attempting that on every probe would compound the memory
# pressure and spam tracebacks.
HEALTH_EMBEDDER_RETRY_SECONDS = float(os.getenv("HEALTH_EMBEDDER_RETRY_SECONDS", "60"))

_embedder_loadable_cache: bool | None = None
_embedder_last_attempt_at: float = 0.0


def _store_loaded() -> bool:
    storage_dir = Path(RAG_STORAGE_DIR)
    return all((storage_dir / f).exists() for f in ["chunks.jsonl", "vectors.npy", "bm25.json"])


def _embedder_loadable() -> bool:
    """True once the embedding model has loaded successfully (cached for the
    life of the process). On failure, reports False but re-attempts at most
    once per HEALTH_EMBEDDER_RETRY_SECONDS so readiness can recover on its own
    without hammering a process that is probably already memory-starved."""
    global _embedder_loadable_cache, _embedder_last_attempt_at

    if _embedder_loadable_cache:
        return True

    now = time.monotonic()
    if _embedder_loadable_cache is False and (now - _embedder_last_attempt_at) < HEALTH_EMBEDDER_RETRY_SECONDS:
        # Still inside the back-off window from a recent failure - report the
        # last known result without retrying the load.
        return False

    _embedder_last_attempt_at = now
    first_attempt = _embedder_loadable_cache is None
    try:
        from src.rag.retrieval.search import _get_model  # loads/caches the singleton

        _get_model()
        _embedder_loadable_cache = True
    except Exception as exc:
        # Full traceback on the first failure only; repeats stay terse so a
        # persistent failure doesn't flood the logs on every probe.
        if first_attempt:
            logger.exception("embedder_load_check_failed")
        else:
            logger.warning(
                "embedder_load_check_failed_again",
                extra={"exc_type": type(exc).__name__, "retry_in_seconds": HEALTH_EMBEDDER_RETRY_SECONDS},
            )
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
