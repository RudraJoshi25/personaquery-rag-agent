from fastapi import APIRouter, Request
from pydantic import BaseModel, Field, field_validator

from src.core.config import CHAT_MAX_QUESTION_LENGTH, RATE_LIMIT_CHAT, TOP_K
from src.core.rate_limit import limiter
from src.rag.chat import run_rag

router = APIRouter()


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=CHAT_MAX_QUESTION_LENGTH)

    @field_validator("question")
    @classmethod
    def _normalize(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("question must not be empty or whitespace-only")
        return v


@router.post("/chat")
@limiter.limit(RATE_LIMIT_CHAT)
def chat(request: Request, req: ChatRequest):
    return run_rag(req.question, top_k=TOP_K, mode="chat")
