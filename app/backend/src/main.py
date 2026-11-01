# src/main.py
from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.api.routes_chat import router as chat_router
from src.core.config import RAG_STORAGE_DIR, RAG_VECTOR_ENABLED


def warm_up() -> None:
    """Load every model and the index once. Raises if the store is missing."""
    from src.rag.embedder import get_embedder
    from src.rag.rerank import RERANK_ENABLED, _get_colbert
    from src.rag.retrieve_custom import _get_store

    get_embedder()
    _get_store()
    if RERANK_ENABLED:
        _get_colbert()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    storage_dir = Path(RAG_STORAGE_DIR)
    storage_ok = all((storage_dir / f).exists() for f in ["chunks.jsonl", "vectors.npy", "bm25.json"])

    # Load models now, not on the first request: an out-of-memory then fails
    # the deploy instead of a visitor's request, and nobody waits on a load.
    # Tests switch it off (tests/conftest.py).
    if os.getenv("PRELOAD_MODELS", "true").lower() in {"1", "true", "yes"}:
        warm_up()

    # Measured AFTER the loads, so the deploy log shows the real footprint.
    try:
        import psutil
        mem_mb = round(psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024, 1)
    except Exception:
        mem_mb = None
    print(
        f"[startup] vector_enabled={RAG_VECTOR_ENABLED} storage_ok={storage_ok} mem_mb={mem_mb}",
        flush=True,
    )
    yield


app = FastAPI(lifespan=lifespan)

cors_allow_all = os.getenv("CORS_ALLOW_ALL", "").lower() in {"1", "true", "yes"}
cors_origins = [o.strip() for o in os.getenv("CORS_ALLOW_ORIGINS", "").split(",") if o.strip()]
# Exact production frontend, e.g. https://personaquery.vercel.app. A
# *.vercel.app wildcard would let anyone's Vercel deploy call this API.
frontend_origin = os.getenv("FRONTEND_ORIGIN", "").strip().rstrip("/")
if frontend_origin:
    cors_origins.append(frontend_origin)

# With no explicit origins configured (local dev), allow localhost only.
cors_origin_regex = os.getenv(
    "CORS_ALLOW_ORIGIN_REGEX",
    r"^https?://(localhost|127\.0\.0\.1|192\.168\.\d+\.\d+):3000$",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if cors_allow_all else cors_origins,
    allow_origin_regex=None if (cors_allow_all or cors_origins) else cors_origin_regex,
    allow_credentials=not cors_allow_all,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    # Log the real error server-side (with a trace id for correlation), but
    # NEVER leak the exception message, stack trace, secrets, or provider
    # error bodies to the client. Return a clean, generic JSON response.
    from src.core.logging import log_event, new_trace_id
    tid = new_trace_id()
    log_event(
        "ERROR", "unhandled_exception", trace_id=tid,
        path=str(request.url.path), error_type=type(exc).__name__, error=str(exc),
    )
    return JSONResponse(
        status_code=500,
        content={
            "error": "internal_error",
            "message": "Something went wrong. Please try again.",
            "trace_id": tid,
        },
    )

@app.get("/health")
def health():
    return {"status": "ok", "vector_enabled": RAG_VECTOR_ENABLED}

app.include_router(chat_router)

