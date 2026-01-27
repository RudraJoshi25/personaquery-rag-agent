# src/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os

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

@app.get("/health")
def health():
    return {"ok": True}
