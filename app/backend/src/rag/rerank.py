"""
Reranking: late-interaction (ColBERT), run locally.

WHY LATE INTERACTION
--------------------
Single-vector embeddings compress a whole chunk into one point, so a chunk
that answers a question in one clause gets diluted by everything else in it.
Late interaction keeps a vector PER TOKEN and scores a pair with MaxSim: for
each query token, take its best-matching document token, then sum. A chunk
that nails two words of the question is rewarded for exactly those two words
instead of being averaged into mediocrity.

WHY LOCALLY
-----------
Storage is not the issue people expect. This corpus is a few hundred chunks;
at ~200 tokens x 128 dims x 4 bytes that is single-digit megabytes. And
running it through FastEmbed's ONNX runtime costs ~40-80ms with no network
hop, which is FASTER than calling a hosted reranker API (~250-500ms). The
usual reason to avoid ColBERT — heavyweight torch inference — does not apply
when the candidate set is already narrowed to a few dozen chunks.
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

from src.core.logging import log_event

COLBERT_MODEL = os.getenv("COLBERT_MODEL", "colbert-ir/colbertv2.0")
RERANK_ENABLED = os.getenv("RERANK_ENABLED", "true").lower() in {"1", "true", "yes"}

_colbert = None
_colbert_failed = False


def _get_colbert():
    """Lazy singleton; degrades to None so retrieval still works without it."""
    global _colbert, _colbert_failed
    if _colbert is not None or _colbert_failed:
        return _colbert
    try:
        from fastembed import LateInteractionTextEmbedding

        _colbert = LateInteractionTextEmbedding(model_name=COLBERT_MODEL)
        print(f"[rerank] colbert ready: {COLBERT_MODEL}", flush=True)
    except Exception as exc:
        _colbert_failed = True
        print(f"[rerank] colbert unavailable ({exc}); falling back to fusion order", flush=True)
    return _colbert


def _maxsim(q_vecs, d_vecs) -> float:
    """
    ColBERT MaxSim: for every query token take its best match in the document,
    then sum. Vectors arrive L2-normalized, so a dot product is cosine.
    """
    import numpy as np

    if q_vecs is None or d_vecs is None or len(q_vecs) == 0 or len(d_vecs) == 0:
        return 0.0
    sim = np.matmul(q_vecs, d_vecs.T)      # (n_query_tokens, n_doc_tokens)
    return float(sim.max(axis=1).sum())


COLBERT_INGEST_BATCH_SIZE = int(os.getenv("COLBERT_INGEST_BATCH_SIZE", "8"))


def precompute_doc_vectors(texts: List[str]):
    """
    Encode every chunk's ColBERT token vectors ONCE, at ingest time.

    This exists because colbert_rerank() used to call model.embed() on the
    full candidate pool (dozens of chunks) on EVERY query, which meant paying
    a full BERT-base-scale forward pass per candidate per request — measured
    at 60-100s+ per query on CPU. The corpus is static between ingests, so
    there is no reason to re-encode a chunk's text more than once. Only the
    query (a few tokens) needs to be encoded at request time.

    Runs in small batches rather than one call over the whole corpus: fastembed's
    default batch_size (256) tries to encode the entire candidate set in a single
    ONNX Runtime call, and on modest hardware that blew the allocator ("bad
    allocation" from the BiasGelu op) once the corpus grew past ~80 chunks.
    Small batches trade a bit of ingest-time speed for staying within memory.

    Returns None (not a hard failure) if ColBERT can't load, or if encoding
    fails even at a small batch size (e.g. genuinely insufficient RAM) — so
    ingestion never crashes on a reranking model being unavailable or heavy.
    Query time then falls back to per-hit on-the-fly encoding for anything
    missing a cached vector, same as an older store.
    """
    model = _get_colbert()
    if model is None or not texts:
        return None
    import numpy as np

    try:
        out: List[np.ndarray] = []
        for i in range(0, len(texts), COLBERT_INGEST_BATCH_SIZE):
            batch = texts[i : i + COLBERT_INGEST_BATCH_SIZE]
            out.extend(np.array(v, dtype="float32") for v in model.embed(batch, batch_size=len(batch)))
            print(f"[ingest] colbert vectors: {min(i + len(batch), len(texts))}/{len(texts)}", flush=True)
        return out
    except Exception as exc:
        print(f"[ingest] colbert precompute failed ({exc}); queries will encode on the fly", flush=True)
        return None


def colbert_rerank(
    question: str,
    hits: List[Dict[str, Any]],
    top_k: int,
    trace_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    model = _get_colbert()
    if model is None or not hits:
        return hits[:top_k]

    try:
        import numpy as np

        q_vecs = np.array(list(model.query_embed([question]))[0], dtype="float32")

        # Steady state: every hit already carries its vector from the ingest-time
        # cache (see precompute_doc_vectors / store.colbert_vectors). Only hits
        # missing one — an older store, or a chunk added outside ingestion — get
        # encoded here, so a normal query never re-runs ColBERT over document text.
        need_encode = [i for i, h in enumerate(hits) if h.get("colbert_vec") is None]
        if need_encode:
            fresh = list(model.embed([hits[i]["text"] for i in need_encode]))
            for pos, i in enumerate(need_encode):
                hits[i] = {**hits[i], "colbert_vec": np.array(fresh[pos], dtype="float32")}
            log_event("INFO", "colbert_cache_miss", trace_id=trace_id, n=len(need_encode))

        scored: List[Tuple[float, Dict[str, Any]]] = []
        for h in hits:
            score = _maxsim(q_vecs, h["colbert_vec"])
            scored.append((score, {**h, "colbert_score": score}))

        scored.sort(key=lambda x: x[0], reverse=True)
        log_event("INFO", "colbert_rerank", trace_id=trace_id, n_in=len(hits), n_out=min(top_k, len(scored)))
        return [h for _, h in scored[:top_k]]

    except Exception as exc:
        log_event("WARN", "colbert_rerank_failed", trace_id=trace_id, error=str(exc))
        return hits[:top_k]


def rerank(
    question: str,
    hits: List[Dict[str, Any]],
    top_k: int,
    trace_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    if not RERANK_ENABLED or not hits:
        return hits[:top_k]

    return colbert_rerank(question, hits, top_k, trace_id=trace_id)[:top_k]
