from fastapi import APIRouter
from pydantic import BaseModel
from src.rag.rag import run_rag

router = APIRouter()

class ChatRequest(BaseModel):
    question: str

@router.post("/chat")
def chat(req: ChatRequest):
    return run_rag(req.question, top_k=8, mode="chat")
