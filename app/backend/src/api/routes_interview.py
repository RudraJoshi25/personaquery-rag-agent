# src/api/routes_interview.py
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from src.core.config import (
    INTERVIEW_ANSWER_MAX_LENGTH,
    RATE_LIMIT_INTERVIEW_ANSWER,
    RATE_LIMIT_INTERVIEW_START,
)
from src.core.rate_limit import limiter
from src.rag.interview import (
    LLMUnavailableError,
    SessionNotFoundError,
    answer_interview,
    start_interview,
)

router = APIRouter()


class StartReq(BaseModel):
    n_questions: int = Field(6, ge=3, le=12)


class AnswerReq(BaseModel):
    session_id: str
    answer: str = Field(..., min_length=1, max_length=INTERVIEW_ANSWER_MAX_LENGTH)


@router.post("/interview/start")
@limiter.limit(RATE_LIMIT_INTERVIEW_START)
def interview_start(request: Request, req: StartReq):
    return start_interview(req.n_questions)


@router.post("/interview/answer")
@limiter.limit(RATE_LIMIT_INTERVIEW_ANSWER)
def interview_answer(request: Request, req: AnswerReq):
    try:
        return answer_interview(req.session_id, req.answer)
    except SessionNotFoundError as e:
        raise HTTPException(
            status_code=404, detail="Invalid or expired session_id. Start a new interview."
        ) from e
    except LLMUnavailableError as e:
        raise HTTPException(status_code=502, detail="Grading service temporarily unavailable. Try again.") from e
