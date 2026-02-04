# src/rag/rag.py
from __future__ import annotations

from typing import Dict, Any, List
import re
import os
import time

from src.core.config import TOP_K, INJECTION_GUARD_ENABLED
from src.rag.guardrails import check_question
from src.rag.retrieve_custom import retrieve, make_context_pack
from src.rag.llm_groq import answer_with_groq

SOURCE_ID_RE = re.compile(r"\[\[cite:([0-9,\s]+)\]\]")


def run_rag(question: str, top_k: int = TOP_K, mode: str = "chat") -> Dict[str, Any]:
    debug = os.getenv("DEBUG_RAG", "0").lower() in {"1", "true", "yes"}
    t0 = time.perf_counter()
    if INJECTION_GUARD_ENABLED:
        gr = check_question(question)
        if not gr.allowed:
            return {"answer": gr.reason or "Request blocked by guardrails.", "sources": []}
        question = gr.sanitized_question or question

    hits = retrieve(question, top_k=top_k)
    if debug:
        print(f"[rag] retrieve: {len(hits)} hits in {time.perf_counter() - t0:.2f}s", flush=True)

    # Assign stable source ids (1..N) for answer citations
    sources_with_ids = list(zip(range(1, len(hits) + 1), hits))
    context = make_context_pack(
        [h for _, h in sources_with_ids],
        source_ids=[sid for sid, _ in sources_with_ids],
    )

    answer = answer_with_groq(question, context, mode=mode)
    if debug:
        print(f"[rag] llm done in {time.perf_counter() - t0:.2f}s", flush=True)

    # Cite-only-if-used: match [[cite:1,2]] markers
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
