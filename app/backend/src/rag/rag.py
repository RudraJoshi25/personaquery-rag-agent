# src/rag/rag.py
from __future__ import annotations
from typing import Dict, Any, List
import os

from src.rag.embedder import get_embedder
from src.rag.store import HybridStore
from src.rag.guardrails import check_question
from src.rag.retrieve_custom import retrieve as hybrid_retrieve, make_context_pack
from src.rag.llm_groq import answer_with_groq

_store_singleton: HybridStore | None = None

def get_store() -> HybridStore:
    global _store_singleton
    if _store_singleton is None:
        embedder = get_embedder()
        st = HybridStore(embed_dim=embedder.dim)
        loaded = st.load()
        if not loaded:
            # product-like message
            raise RuntimeError(
                "RAG store not found. Run ingestion to create storage/chunks.jsonl, vectors.npy, bm25.json"
            )
        _store_singleton = st
    return _store_singleton

def run_rag(question: str, top_k: int = 12, mode: str = "chat") -> Dict[str, Any]:
    gr = check_question(question)
    if not gr.allowed:
        return {
            "answer": gr.reason or "Request blocked by guardrails.",
            "sources": [],
        }

    q = gr.sanitized_question or question
    store = get_store()
    # retrieve() loads from the persisted store; no need to pass store instance
    hits = hybrid_retrieve(q, top_k=top_k)
    context = make_context_pack(hits)

    answer = answer_with_groq(q, context, mode=mode)

    # Enforce citations if model skipped them: append a short sources list.
    if "[" not in answer or "]" not in answer:
        anchors = []
        for h in hits:
            m = h["metadata"]
            file_name = m.get("file_name", "unknown")
            page = m.get("page_label", "n/a")
            section = m.get("section", "Document")
            anchors.append(f"[{file_name} | p.{page} | {section}]")
        if anchors:
            uniq = list(dict.fromkeys(anchors))
            answer = answer.rstrip() + "\n\nSources used: " + ", ".join(uniq[:6])

    # sources response: exact doc + page + section + snippet
    sources = []
    for h in hits:
        m = h["metadata"]
        sources.append({
            "file_name": m.get("file_name", "unknown"),
            "page_label": m.get("page_label", "n/a"),
            "section": m.get("section", "Document"),
            "score": h.get("score", 0.0),
            "channel": m.get("channel", "hybrid"),
            "snippet": h["text"][:320],
        })

    return {"answer": answer, "sources": sources}
