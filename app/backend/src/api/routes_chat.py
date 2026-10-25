# src/api/routes_chat.py
from __future__ import annotations

from fastapi import APIRouter, Depends
from pydantic import BaseModel, field_validator

from src.api.rate_limit import rate_limit_dep
from src.core.config import MAX_QUESTION_CHARS
from src.rag.rag import run_rag

router = APIRouter()


class ChatRequest(BaseModel):
    question: str

    @field_validator("question")
    @classmethod
    def _validate_question(cls, v: str) -> str:
        # Type is already enforced by the `str` annotation (422 on mismatch).
        v = v.strip()
        if not v:
            raise ValueError("question must not be empty")
        if len(v) > MAX_QUESTION_CHARS:
            raise ValueError(f"question too long (max {MAX_QUESTION_CHARS} characters)")
        return v


@router.post("/chat")
def chat(req: ChatRequest, client: str = Depends(rate_limit_dep)):
    return run_rag(req.question, top_k=8, mode="chat")
