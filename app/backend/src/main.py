# src/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes_chat import router as chat_router
from src.api.routes_interview import router as interview_router

app = FastAPI(title="PersonaQuery API")

# allow localhost/LAN for dev + Vercel preview/prod
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"^https?://(localhost|127\.0\.0\.1|192\.168\.\d+\.\d+):3000$|^https?://(localhost|127\.0\.0\.1|192\.168\.\d+\.\d+):3001$|^https?://.*\\.vercel\\.app$",
    allow_credentials=True,
    allow_methods=["*"],   # IMPORTANT: lets OPTIONS pass
    allow_headers=["*"],
)

app.include_router(chat_router)
app.include_router(interview_router)

@app.get("/health")
def health():
    return {"ok": True}
