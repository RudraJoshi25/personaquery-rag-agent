# src/main.py
from __future__ import annotations
import os
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes_chat import router as chat_router
from src.core.config import RAG_VECTOR_ENABLED
from pathlib import Path

app = FastAPI()

cors_allow_all = os.getenv("CORS_ALLOW_ALL", "").lower() in {"1", "true", "yes"}
cors_origins = [o.strip() for o in os.getenv("CORS_ALLOW_ORIGINS", "").split(",") if o.strip()]

# Allow localhost + any *.vercel.app by default
cors_origin_regex = os.getenv(
    "CORS_ALLOW_ORIGIN_REGEX",
    r"^https?://(localhost|127\.0\.0\.1|192\.168\.\d+\.\d+):3000$|^https?://.*\.vercel\.app$",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if cors_allow_all else cors_origins,
    allow_origin_regex=None if (cors_allow_all or cors_origins) else cors_origin_regex,
    allow_credentials=False if cors_allow_all else True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    # Ensures frontend always gets JSON (and CORS headers still apply)
    return JSONResponse(
        status_code=500,
        content={"error": "Internal Server Error", "detail": str(exc)},
    )

@app.get("/health")
def health():
    return {"status": "ok", "vector_enabled": RAG_VECTOR_ENABLED}

app.include_router(chat_router)


@app.on_event("startup")
async def startup_log():
    # Lightweight diagnostics for Render
    try:
        import psutil  # optional
        mem_mb = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
    except Exception:
        mem_mb = None

    storage_dir = Path(os.getenv("RAG_STORAGE_DIR", "storage"))
    storage_ok = all((storage_dir / f).exists() for f in ["chunks.jsonl", "vectors.npy", "bm25.json"])
    print(
        f"[startup] vector_enabled={RAG_VECTOR_ENABLED} load_vectors={RAG_VECTOR_ENABLED} storage_ok={storage_ok} mem_mb={mem_mb}",
        flush=True,
    )
