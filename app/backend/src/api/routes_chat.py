from fastapi import APIRouter
from pydantic import BaseModel
from src.rag.retrieve import get_query_engine


router = APIRouter()

class ChatRequest(BaseModel):
    question: str

@router.post("/chat")
def chat(req: ChatRequest):
    qe = get_query_engine()
    result = qe.query(req.question)

    sources = []
    for node in getattr(result, "source_nodes", []) or []:
        meta = node.node.metadata or {}
        sources.append({
            "file_name": meta.get("file_name") or meta.get("filename") or "unknown",
            "page_label": meta.get("page_label"),
            "score": node.score,
            "snippet": node.node.get_text()[:280],
        })

    return {
        "answer": str(result),
        "sources": sources,
    }
