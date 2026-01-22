from fastapi import APIRouter
from pydantic import BaseModel

from src.rag.retrieve_custom import retrieve, make_context_pack
from src.rag.llm_groq import answer_with_groq

router = APIRouter()

class ChatRequest(BaseModel):
    question: str

@router.post("/chat")
def chat(req: ChatRequest):
    hits = retrieve(req.question, top_k=5)
    context = make_context_pack(hits)
    answer = answer_with_groq(req.question, context)

    sources = []
    for h in hits:
        m = h["metadata"]
        sources.append({
            "file_name": m.get("file_name", "unknown"),
            "page_label": m.get("page_label"),
            "score": h["score"],
            "snippet": h["text"][:280],
        })

    return {"answer": answer, "sources": sources}
