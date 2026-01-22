from fastapi import FastAPI
from src.api.routes_health import router as health_router
from src.api.routes_chat import router as chat_router

app = FastAPI(title="PersonaQuery API", version="0.1.0")

app.include_router(health_router)
app.include_router(chat_router)
