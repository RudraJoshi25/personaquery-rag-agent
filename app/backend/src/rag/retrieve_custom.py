from __future__ import annotations
from typing import List, Dict, Any
import numpy as np
from sentence_transformers import SentenceTransformer

from src.core.config import EMBED_MODEL, TOP_K
from src.rag.index_custom import load_index


_model = None
_index_cache = None


def _get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBED_MODEL)
    return _model


def _get_index():
    global _index_cache
    if _index_cache is None:
        _index_cache = load_index()
    return _index_cache


def retrieve(question: str, top_k: int = TOP_K) -> List[Dict[str, Any]]:
    model = _get_model()
    idx = _get_index()

    q = model.encode([question], normalize_embeddings=True).astype(np.float32)[0]
    emb = idx["embeddings"]  # shape: (N, D)

    # cosine similarity (since normalized embeddings)
    scores = emb @ q
    top_idx = np.argsort(-scores)[:top_k]

    results = []
    for i in top_idx:
        item = idx["meta"][int(i)]
        results.append(
            {
                "score": float(scores[int(i)]),
                "text": item["text"],
                "metadata": item["metadata"],
            }
        )
    return results


def make_context_pack(hits: List[Dict[str, Any]]) -> str:
    blocks = []
    for h in hits:
        m = h["metadata"]
        header = f"[SOURCE: {m.get('file_name','unknown')} | page {m.get('page_label','?')}]"
        blocks.append(header + "\n" + h["text"])
    return "\n\n---\n\n".join(blocks)
