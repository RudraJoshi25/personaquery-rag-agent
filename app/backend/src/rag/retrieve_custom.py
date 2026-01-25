from __future__ import annotations
from typing import List, Dict, Any, Tuple, Optional
import os

from sentence_transformers import SentenceTransformer

from src.core.config import EMBED_MODEL, TOP_K
from src.rag.store import HybridStore


# singletons
_model: Optional[SentenceTransformer] = None
_store: Optional[HybridStore] = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBED_MODEL)
    return _model


def _get_store() -> HybridStore:
    """
    Loads persisted hybrid store:
      storage/chunks.jsonl
      storage/vectors.npy
      storage/bm25.json
    """
    global _store
    if _store is not None:
        return _store

    # Default embed dim for all-MiniLM-L6-v2 is 384.
    # If you switch embedding model later, update this or compute it dynamically.
    embed_dim = int(os.getenv("EMBED_DIM", "384"))
    st = HybridStore(embed_dim=embed_dim)
    loaded = st.load()
    if not loaded:
        raise RuntimeError(
            "RAG store not found. Run ingestion to create storage/chunks.jsonl, vectors.npy, bm25.json"
        )
    _store = st
    return st


def _rrf_fuse(
    vec_ranked: List[int],
    bm25_ranked: List[int],
    k: int = 60,
) -> List[Tuple[int, float]]:
    """
    Reciprocal Rank Fusion (RRF) over doc ids.
    Robust because it doesn't require score calibration between BM25 and cosine similarity.
    """
    scores: Dict[int, float] = {}

    for rank, doc_id in enumerate(vec_ranked):
        scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)

    for rank, doc_id in enumerate(bm25_ranked):
        scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)

    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return fused


def retrieve(question: str, top_k: int = TOP_K) -> List[Dict[str, Any]]:
    """
    Hybrid retrieval:
      - Vector search (cosine)
      - BM25 keyword search
      - RRF fusion
    Fallback:
      - If embedding fails, return BM25 only
    """
    model = _get_model()
    store = _get_store()

    # Pull more candidates than final top_k for better fusion
    cand_k = max(top_k * 4, 12)

    # BM25 always available
    bm25_hits = store.search_bm25(question, top_k=cand_k)
    bm25_ranked_ids = [doc_id for doc_id, _ in bm25_hits]

    # Vector retrieval (may fail)
    vec_ranked_ids: List[int] = []
    vec_scores_by_id: Dict[int, float] = {}

    try:
        q_vec = model.encode([question], normalize_embeddings=True)[0].tolist()
        vec_hits = store.search_vector(q_vec, top_k=cand_k)
        vec_ranked_ids = [doc_id for doc_id, _ in vec_hits]
        vec_scores_by_id = {doc_id: float(score) for doc_id, score in vec_hits}
    except Exception:
        # fallback = BM25-only
        vec_ranked_ids = []

    # Fuse
    if vec_ranked_ids:
        fused = _rrf_fuse(vec_ranked_ids, bm25_ranked_ids, k=60)
        fused_ids = [doc_id for doc_id, _ in fused[:top_k]]
        fused_rrf_score = {doc_id: float(score) for doc_id, score in fused}
    else:
        # BM25-only fallback
        fused_ids = bm25_ranked_ids[:top_k]
        fused_rrf_score = {doc_id: 0.0 for doc_id in fused_ids}

    # Build final hits
    hits: List[Dict[str, Any]] = []
    for doc_id in fused_ids:
        ch = store.chunks[int(doc_id)]
        m = ch.metadata
        in_vec = int(doc_id) in vec_scores_by_id
        in_bm25 = int(doc_id) in bm25_ranked_ids
        if in_vec and in_bm25:
            channel = "hybrid"
        elif in_vec:
            channel = "vector"
        else:
            channel = "keyword"

        hits.append(
            {
                "score": float(fused_rrf_score.get(int(doc_id), 0.0)),
                "text": ch.text,
                "metadata": {
                    **m,
                    # helpful debug fields:
                    "doc_id": int(doc_id),
                    "vec_sim": float(vec_scores_by_id.get(int(doc_id), 0.0)),
                    "channel": channel,
                },
            }
        )

    return hits


def make_context_pack(hits: List[Dict[str, Any]], max_chars: int = 12000) -> str:
    """
    Context pack given to the LLM. Includes file + page/section identifiers.
    """
    blocks = []
    total = 0
    for h in hits:
        m = h["metadata"]
        file_name = m.get("file_name", "unknown")
        page = m.get("page_label", "?")
        section = m.get("section", None)

        section_label = section or "Document"
        header = f"[{file_name} | p.{page} | {section_label}]"

        block = header + "\n" + h["text"]
        if total + len(block) > max_chars:
            break
        blocks.append(block)
        total += len(block)

    return "\n\n---\n\n".join(blocks)
