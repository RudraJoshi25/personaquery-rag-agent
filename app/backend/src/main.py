# src/main.py
from __future__ import annotations

import logging

from src.core.logging import setup_logging

setup_logging()

import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from starlette.exceptions import HTTPException as StarletteHTTPException

from src.api.routes_chat import router as chat_router
from src.api.routes_health import router as health_router
from src.api.routes_interview import router as interview_router
from src.core.config import (
    CORS_ALLOW_ALL,
    CORS_ALLOW_ORIGIN_REGEX,
    CORS_ALLOWED_ORIGINS,
    RAG_VECTOR_ENABLED,
    validate_startup_config,
)
from src.core.errors import (
    http_exception_handler,
    unhandled_exception_handler,
    validation_exception_handler,
)
from src.core.middleware import RequestIDMiddleware, SecurityHeadersMiddleware
from src.core.rate_limit import limiter, rate_limit_exceeded_handler

logger = logging.getLogger("personaquery.main")

validate_startup_config()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Lightweight diagnostics for Render
    try:
        import psutil  # optional

        mem_mb = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
    except Exception:
        mem_mb = None

    storage_dir = Path(os.getenv("RAG_STORAGE_DIR", "storage"))
    storage_ok = all((storage_dir / f).exists() for f in ["chunks.jsonl", "vectors.npy", "bm25.json"])
    logger.info(
        "startup",
        extra={
            "vector_enabled": RAG_VECTOR_ENABLED,
            "storage_ok": storage_ok,
            "mem_mb": mem_mb,
        },
    )
    yield


app = FastAPI(lifespan=lifespan)

app.state.limiter = limiter

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if CORS_ALLOW_ALL else CORS_ALLOWED_ORIGINS,
    allow_origin_regex=None if CORS_ALLOW_ALL else CORS_ALLOW_ORIGIN_REGEX,
    allow_credentials=False if CORS_ALLOW_ALL else True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)
app.add_middleware(SlowAPIMiddleware)
app.add_middleware(SecurityHeadersMiddleware)
# Registered last so it's the OUTERMOST middleware - every response,
# including CORS-rejected and rate-limited ones, gets a request id.
app.add_middleware(RequestIDMiddleware)

app.add_exception_handler(RateLimitExceeded, rate_limit_exceeded_handler)
app.add_exception_handler(StarletteHTTPException, http_exception_handler)
app.add_exception_handler(RequestValidationError, validation_exception_handler)
app.add_exception_handler(Exception, unhandled_exception_handler)

app.include_router(chat_router)
app.include_router(interview_router)
app.include_router(health_router)
