# src/rag/faq_cache.py
"""
Precomputed answers for the questions this bot actually gets asked.

WHY THIS EXISTS
---------------
A portfolio assistant is asked the same narrow set of things by almost
every visitor: contact details, education, skills, projects, awards, the
patent. Those answers are derived from documents that only change when
Rudra re-runs ingestion. Generating them live, per visitor, means paying
an LLM round trip (and burning free-tier quota) to reproduce an answer
that was already correct yesterday.

Precomputing them offline moves that cost to build time, where a rate
limit is harmless — the script can simply wait and retry. At request time
a hit costs a dictionary lookup plus one embedding of the question: no
network, no quota, no provider outage exposure, ~10ms instead of ~2-15s.

MATCHING
--------
Exact string match is far too brittle ("what's your email" vs "what is
your email address?" vs "how do I contact you"). Matching is therefore
semantic: the visitor's question is embedded with the SAME model used for
retrieval, and compared by cosine similarity against precomputed question
embeddings. FAQ_MIN_SIMILARITY is deliberately high (0.86) — a wrong
cached answer served confidently is far worse than a cache miss, since a
miss just falls through to the normal grounded pipeline.

Staleness is handled by binding the cache to the ingest that produced it:
if storage/chunks.jsonl is newer than the cache file, the cache is ignored
rather than serving answers from a superseded corpus.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from src.core.logging import log_event

STORAGE_DIR = os.getenv("RAG_STORAGE_DIR", "storage")
FAQ_PATH = Path(STORAGE_DIR) / "faq_cache.json"
CHUNKS_PATH = Path(STORAGE_DIR) / "chunks.jsonl"

FAQ_ENABLED = os.getenv("FAQ_CACHE_ENABLED", "true").lower() in {"1", "true", "yes"}
FAQ_MIN_SIMILARITY = float(os.getenv("FAQ_MIN_SIMILARITY", "0.86"))

_cache: Optional[Dict[str, Any]] = None
_loaded = False


def _load() -> Optional[Dict[str, Any]]:
    global _cache, _loaded
    if _loaded:
        return _cache
    _loaded = True

    if not (FAQ_ENABLED and FAQ_PATH.exists()):
        return None

    try:
        with open(FAQ_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as exc:
        print(f"[faq] could not read cache ({exc}); disabled", flush=True)
        return None

    # Refuse to serve answers generated from an older corpus.
    if CHUNKS_PATH.exists() and CHUNKS_PATH.stat().st_mtime > FAQ_PATH.stat().st_mtime:
        print("[faq] cache is older than the index; ignoring it. "
              "Re-run scripts/precompute_faq.py to refresh.", flush=True)
        return None

    entries = data.get("entries", [])
    if not entries:
        return None

    data["_matrix"] = np.array([e["embedding"] for e in entries], dtype="float32")
    _cache = data
    print(f"[faq] loaded {len(entries)} precomputed answers", flush=True)
    return _cache


def lookup_faq(question: str, trace_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Returns the cached {answer, sources} for a semantically-equivalent
    question, or None to fall through to the live pipeline."""
    data = _load()
    if not data:
        return None

    try:
        from src.rag.embedder import get_embedder

        q = np.array(get_embedder().embed_query(question), dtype="float32")
        q /= (np.linalg.norm(q) + 1e-12)

        sims = data["_matrix"] @ q
        best = int(np.argmax(sims))
        score = float(sims[best])

        if score < FAQ_MIN_SIMILARITY:
            return None

        entry = data["entries"][best]
        log_event("INFO", "faq_cache_hit", trace_id=trace_id,
                  matched=entry["question"], similarity=round(score, 4))
        return {
            "answer": entry["answer"],
            "sources": entry.get("sources", []),
            "faq_matched": entry["question"],
            "faq_similarity": round(score, 4),
        }
    except Exception as exc:
        # Never let the cache break a request — fall through to live generation.
        log_event("WARN", "faq_cache_error", trace_id=trace_id, error=str(exc))
        return None
