"""
Retrieval pipeline.

    query
      -> conditional expansion into N paraphrases   (query_expansion.py)
      -> per-variant hybrid search: dense + BM25     (run in parallel)
      -> Reciprocal Rank Fusion across every list
      -> source-diversity cap
      -> late-interaction (ColBERT) rerank           (rerank.py)
      -> top_k

NOTE ON THE REMOVED IDENTITY BOOST
----------------------------------
This module used to detect "is this about Rudra?" with a regex and add a
large constant to any chunk from the resume. That was a workaround for an
ingestion bug: one 20-page research paper produced 82% of the index, so
identity questions retrieved co-authors' contact details instead of Rudra's.

The real fix was per-document chunk profiles (see doc_profiles.py), which
rebalanced the index. What remains here is a principled, source-agnostic
diversity cap: no single document may occupy more than a set number of slots
in one result set. That solves the same failure without hard-coding which
document is allowed to win, and without fighting the reranker.
"""
from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from src.core.config import RAG_VECTOR_ENABLED, TOP_K
from src.core.logging import log_event
from src.rag.doc_profiles import MAX_CHUNKS_PER_SOURCE_IN_RESULTS
from src.rag.query_expansion import expand_query
from src.rag.rerank import rerank as rerank_hits
from src.rag.store import HybridStore

RRF_K = int(os.getenv("RRF_K", "60"))
_store: HybridStore | None = None


def _get_store() -> HybridStore:
    """Loads storage/chunks.jsonl + vectors.npy + bm25.json once per process."""
    global _store
    if _store is not None:
        return _store

    from src.rag.embedder import get_embedder

    embed_dim = int(os.getenv("EMBED_DIM", "0")) or get_embedder().dim
    st = HybridStore(embed_dim=embed_dim)
    if not st.load(load_vectors=RAG_VECTOR_ENABLED):
        raise RuntimeError(
            "RAG store not found. Run ingestion to create storage/chunks.jsonl, "
            "vectors.npy, bm25.json"
        )
    _store = st
    return st


def _rrf_fuse(ranked_lists: list[list[int]], k: int = RRF_K) -> dict[int, float]:
    """
    Reciprocal Rank Fusion over any number of ranked id lists.

    RRF is used rather than score averaging because BM25 scores and cosine
    similarities live on incomparable scales, and normalising them requires
    calibration that shifts every time the corpus changes. Rank position is
    stable and needs no tuning. Extending it from 2 lists to N is free, which
    is what makes multi-query expansion cheap to fuse.
    """
    scores: dict[int, float] = {}
    for ranked in ranked_lists:
        for rank, doc_id in enumerate(ranked):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    return scores


def _search_one(
    store: HybridStore,
    query: str,
    cand_k: int,
    vector_enabled: bool,
) -> tuple[list[int], list[int], dict[int, float]]:
    """One query -> (vector_ranked_ids, bm25_ranked_ids, vec_scores)."""
    bm25_ranked = [doc_id for doc_id, _ in store.search_bm25(query, top_k=cand_k)]

    vec_ranked: list[int] = []
    vec_scores: dict[int, float] = {}
    if vector_enabled:
        try:
            from src.rag.cache import cached_embed
            from src.rag.embedder import get_embedder

            embedder = get_embedder()
            q_vec, _ = cached_embed(query, embedder.embed_query)
            hits = store.search_vector(q_vec, top_k=cand_k)
            vec_ranked = [doc_id for doc_id, _ in hits]
            vec_scores = {doc_id: float(s) for doc_id, s in hits}
        except Exception:
            vec_ranked = []  # BM25-only fallback

    return vec_ranked, bm25_ranked, vec_scores


def _apply_diversity_cap(
    ordered_ids: list[int],
    store: HybridStore,
    max_per_source: int = MAX_CHUNKS_PER_SOURCE_IN_RESULTS,
) -> list[int]:
    """
    Keep the fused order, but stop any one file from monopolising the context
    window. Overflow chunks are appended after the diverse set rather than
    discarded, so recall is preserved when there is nothing else to show.
    """
    kept: list[int] = []
    overflow: list[int] = []
    per_source: dict[str, int] = {}

    for doc_id in ordered_ids:
        src = str(store.chunks[int(doc_id)].metadata.get("file_name", "?"))
        if per_source.get(src, 0) < max_per_source:
            kept.append(doc_id)
            per_source[src] = per_source.get(src, 0) + 1
        else:
            overflow.append(doc_id)

    return kept + overflow


def retrieve(
    question: str,
    top_k: int = TOP_K,
    trace_id: str | None = None,
) -> list[dict[str, Any]]:
    store = _get_store()

    queries = expand_query(question, trace_id=trace_id)

    # Over-fetch: the reranker can only reorder what retrieval handed it, so
    # the candidate pool must be wider than the final answer.
    cand_k = max(top_k * 4, 16)

    # Variants are independent, and each is dominated by an embedding call —
    # running them serially would multiply latency by the number of variants.
    if len(queries) > 1:
        with ThreadPoolExecutor(max_workers=min(len(queries), 4)) as pool:
            results = list(pool.map(
                lambda q: _search_one(store, q, cand_k, RAG_VECTOR_ENABLED), queries
            ))
    else:
        results = [_search_one(store, queries[0], cand_k, RAG_VECTOR_ENABLED)]

    ranked_lists: list[list[int]] = []
    vec_scores_all: dict[int, float] = {}
    bm25_seen: set = set()
    vec_seen: set = set()

    for vec_ranked, bm25_ranked, vec_scores in results:
        if vec_ranked:
            ranked_lists.append(vec_ranked)
            vec_seen.update(vec_ranked)
        if bm25_ranked:
            ranked_lists.append(bm25_ranked)
            bm25_seen.update(bm25_ranked)
        for doc_id, s in vec_scores.items():
            vec_scores_all[doc_id] = max(vec_scores_all.get(doc_id, 0.0), s)

    if not ranked_lists:
        return []

    fused = _rrf_fuse(ranked_lists)
    ordered = sorted(fused.keys(), key=lambda d: fused[d], reverse=True)

    # Drop ids the chunk store doesn't have. An external vector backend can
    # legitimately return an id that no longer exists (stale entries after the
    # corpus shrinks, a partially-written index, a backend restored from an
    # older snapshot). Without this, one orphaned id raised IndexError and
    # took down EVERY query — a data-freshness problem became a total outage.
    n_chunks = len(store.chunks)
    valid = [d for d in ordered if 0 <= int(d) < n_chunks]
    if len(valid) != len(ordered):
        log_event("WARN", "stale_vector_ids_dropped", trace_id=trace_id,
                  dropped=len(ordered) - len(valid), n_chunks=n_chunks,
                  hint="vector backend is out of sync with chunks.jsonl; re-run ingestion")
    ordered = _apply_diversity_cap(valid, store)

    # Build hit dicts for the rerank stage.
    candidates: list[dict[str, Any]] = []
    for doc_id in ordered[: cand_k * 2]:
        ch = store.chunks[int(doc_id)]
        in_vec = int(doc_id) in vec_seen
        in_bm25 = int(doc_id) in bm25_seen
        channel = "hybrid" if (in_vec and in_bm25) else ("vector" if in_vec else "keyword")

        colbert_vec = None
        if store.colbert_vectors is not None:
            colbert_vec = store.colbert_vectors[int(doc_id)]

        candidates.append({
            "score": float(fused.get(int(doc_id), 0.0)),
            "text": ch.text,
            "colbert_vec": colbert_vec,
            "metadata": {
                **ch.metadata,
                "doc_id": int(doc_id),
                "vec_sim": float(vec_scores_all.get(int(doc_id), 0.0)),
                "channel": channel,
            },
        })

    # Dedupe identical text, keeping the best-scoring copy. Keyed on content,
    # not (file, page): single-page documents like the resume hold contact,
    # skills and projects on one page, and keying on page collapses them.
    deduped: dict[str, dict[str, Any]] = {}
    for h in candidates:
        key = " ".join((h.get("text") or "").split())
        if key not in deduped or h["score"] > deduped[key]["score"]:
            deduped[key] = h
    candidates = sorted(deduped.values(), key=lambda x: x["score"], reverse=True)

    min_score = float(os.getenv("RAG_MIN_SCORE", "0") or "0")
    if min_score > 0:
        candidates = [h for h in candidates if h["score"] >= min_score]

    log_event(
        "INFO", "retrieval_candidates", trace_id=trace_id,
        n_queries=len(queries), n_candidates=len(candidates),
    )

    return rerank_hits(question, candidates, top_k, trace_id=trace_id)


def make_context_pack(
    hits: list[dict[str, Any]],
    max_chars: int = 12000,
    source_ids: list[int] | None = None,
) -> str:
    """Context block handed to the LLM, with citable source identifiers."""
    blocks = []
    total = 0
    for idx, h in enumerate(hits):
        m = h["metadata"]
        file_name = m.get("file_name", "unknown")
        page = m.get("page_label", "?")
        section = m.get("section") or "Document"

        if source_ids and idx < len(source_ids):
            header = f"[SOURCE {source_ids[idx]}] {file_name} | p.{page} | {section}"
        else:
            header = f"[{file_name} | p.{page} | {section}]"

        block = header + "\n" + h["text"]
        if total + len(block) > max_chars:
            break
        blocks.append(block)
        total += len(block)

    return "\n\n---\n\n".join(blocks)
