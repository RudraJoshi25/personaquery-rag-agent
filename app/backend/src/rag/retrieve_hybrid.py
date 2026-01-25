# src/rag/retrieve_hybrid.py
from __future__ import annotations
from typing import List, Dict, Any
from src.rag.store import HybridStore
from src.rag.embedder import get_embedder

def hybrid_retrieve(store: HybridStore, query: str, top_k: int = 12) -> List[Dict[str, Any]]:
    """
    Returns list of hits:
      { "text": ..., "score": ..., "metadata": ..., "channel": "vector|keyword" }
    Merge vector + keyword; fallback if embeddings fail.
    """
    # keyword always available
    kw = store.search_bm25(query, top_k=top_k)

    hits_map: dict[int, Dict[str, Any]] = {}

    # vector
    try:
        embedder = get_embedder()
        qv = embedder.embed([query])[0]
        vec = store.search_vector(qv, top_k=top_k)
    except Exception:
        vec = []

    # merge (normalize-ish)
    for idx, s in vec:
        ch = store.chunks[idx]
        hits_map[idx] = {
            "text": ch.text,
            "score": float(s),
            "metadata": ch.metadata,
            "channel": "vector",
        }

    for idx, s in kw:
        ch = store.chunks[idx]
        if idx in hits_map:
            # combine
            hits_map[idx]["score"] = float(hits_map[idx]["score"]) + float(s) * 0.02
            hits_map[idx]["channel"] = "hybrid"
        else:
            hits_map[idx] = {
                "text": ch.text,
                "score": float(s) * 0.02,  # keep scale somewhat comparable
                "metadata": ch.metadata,
                "channel": "keyword",
            }

    merged = sorted(hits_map.values(), key=lambda x: x["score"], reverse=True)[:top_k]
    return merged

def make_context_pack(hits: List[Dict[str, Any]], max_chars: int = 12000) -> str:
    """
    Context includes citation anchors: [file | page | section]
    """
    parts: List[str] = []
    total = 0

    for h in hits:
        m = h["metadata"]
        file_name = m.get("file_name", "unknown")
        page = m.get("page_label", "n/a")
        section = m.get("section", "Document")
        anchor = f"[{file_name} | p.{page} | {section}]"
        block = f"{anchor}\n{h['text']}\n"
        if total + len(block) > max_chars:
            break
        parts.append(block)
        total += len(block)

    return "\n---\n".join(parts)
