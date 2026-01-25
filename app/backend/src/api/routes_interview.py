# src/api/routes_interview.py
from __future__ import annotations
from fastapi import APIRouter
from pydantic import BaseModel, Field
from typing import Dict, Any, List
import uuid

from src.rag.interview import start_interview, answer_interview

router = APIRouter()

class StartReq(BaseModel):
    n_questions: int = Field(6, ge=3, le=12)

class AnswerReq(BaseModel):
    session_id: str
    answer: str = Field(..., min_length=1)

@router.post("/interview/start")
def interview_start(req: StartReq):
    return start_interview(req.n_questions)

@router.post("/interview/answer")
def interview_answer(req: AnswerReq):
    return answer_interview(req.session_id, req.answer)
