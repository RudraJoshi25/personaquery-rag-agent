# src/api/routes_chat.py
from __future__ import annotations
from fastapi import APIRouter
from pydantic import BaseModel, Field
from typing import Literal, Optional

from src.rag.rag import run_rag

router = APIRouter()

class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1)
    mode: Literal["chat", "interview"] = "chat"
    top_k: int = 12

@router.post("/chat")
def chat(req: ChatRequest):
    return run_rag(req.question, top_k=req.top_k, mode=req.mode)
