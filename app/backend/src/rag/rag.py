# src/rag/rag.py
from __future__ import annotations
from typing import Dict, Any, List
import os

from src.rag.embedder import get_embedder
from src.rag.store import HybridStore
from src.rag.guardrails import check_question
from src.rag.retrieve_custom import retrieve as hybrid_retrieve, make_context_pack
from src.rag.llm_groq import answer_with_groq

import re

SOURCE_ID_RE = re.compile(r"\[\[cite:([0-9,\s]+)\]\]")

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
    # Assign stable source ids (1..N) for answer citations
    sources_with_ids = list(zip(range(1, len(hits) + 1), hits))
    context = make_context_pack([h for _, h in sources_with_ids], source_ids=[sid for sid, _ in sources_with_ids])

    answer = answer_with_groq(q, context, mode=mode)

    # Cite-only-if-used: match [n] markers
    used_ids: set[int] = set()
    for match in SOURCE_ID_RE.findall(answer or ""):
        parts = [p.strip() for p in match.split(",")]
        for p in parts:
            if p.isdigit():
                used_ids.add(int(p))
    used_hits: List[tuple[int, Dict[str, Any]]] = []
    if used_ids:
        for sid, h in sources_with_ids:
            if sid in used_ids:
                used_hits.append((sid, h))
    # fallback: if model didn't cite, return top 3 sources
    sources_with_ids = used_hits if used_hits else sources_with_ids[: min(3, len(sources_with_ids))]

    # sources response: exact doc + page + section + snippet
    sources = []
    for sid, h in sources_with_ids:
        m = h["metadata"]
        sources.append({
            "id": sid,
            "file_name": m.get("file_name", "unknown"),
            "page_label": m.get("page_label", "n/a"),
            "section": m.get("section", "Document"),
            "relevance": h.get("score", 0.0),
            "channel": m.get("channel", "hybrid"),
            "snippet": h["text"][:320],
        })

    return {"answer": answer, "sources": sources}
