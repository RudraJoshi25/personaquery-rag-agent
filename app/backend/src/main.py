# src/main.py
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import os
from pathlib import Path

from src.api.routes_chat import router as chat_router
from src.api.routes_interview import router as interview_router

app = FastAPI(title="PersonaQuery API")

# CORS: allow explicit origins from env; fallback to localhost + Vercel
cors_allow_all = os.getenv("CORS_ALLOW_ALL", "").lower() in {"1", "true", "yes"}
cors_origins = [o.strip() for o in os.getenv("CORS_ALLOW_ORIGINS", "").split(",") if o.strip()]
cors_origin_regex = os.getenv(
    "CORS_ALLOW_ORIGIN_REGEX",
    r"^https?://(localhost|127\.0\.0\.1|192\.168\.\d+\.\d+):3000$|^https?://(localhost|127\.0\.0\.1|192\.168\.\d+\.\d+):3001$|^https?://.*\.vercel\.app$",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if cors_allow_all else cors_origins,
    allow_origin_regex=None if (cors_allow_all or cors_origins) else cors_origin_regex,
    allow_credentials=False if cors_allow_all else True,
    allow_methods=["*"],   # IMPORTANT: lets OPTIONS pass
    allow_headers=["*"],
)

app.include_router(chat_router)
app.include_router(interview_router)

@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    # Return JSON even on unhandled errors so frontend can read it (CORS headers added by middleware)
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": str(exc)},
    )

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
    vector_enabled = os.getenv("RAG_VECTOR_ENABLED", "1").lower() in {"1", "true", "yes"}

    print(f"[startup] vector_enabled={vector_enabled} storage_ok={storage_ok} mem_mb={mem_mb}", flush=True)

@app.get("/health")
def health():
    return {"ok": True}
