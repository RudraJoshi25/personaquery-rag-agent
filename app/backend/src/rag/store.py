from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

import numpy as np
from rank_bm25 import BM25Okapi

from src.core.config import RAG_STORAGE_DIR as STORAGE_DIR

CHUNKS_PATH = os.path.join(STORAGE_DIR, "chunks.jsonl")
VECTORS_PATH = os.path.join(STORAGE_DIR, "vectors.npy")
BM25_PATH = os.path.join(STORAGE_DIR, "bm25.json")
# Per-chunk ColBERT (late-interaction) token vectors, precomputed at ingest
# time. Without this, every query re-runs the ~110M-param ColBERT model over
# the full candidate pool, which is what pushed per-query rerank latency to
# 60-100s+ on CPU. The corpus is static between ingests, so paying that cost
# once per chunk here (instead of once per chunk per query) is the fix.
COLBERT_VECS_PATH = os.path.join(STORAGE_DIR, "colbert_vecs.npz")


def ensure_storage_dir():
    os.makedirs(STORAGE_DIR, exist_ok=True)


def simple_tokenize(text: str) -> list[str]:
    import re
    return re.findall(r"[a-z0-9]+", (text or "").lower())


@dataclass
class StoredChunk:
    text: str
    metadata: dict[str, Any]


class HybridStore:
    """
    Persisted:
      - chunks.jsonl (text + metadata)
      - vectors.npy  (float32 normalized embeddings)
      - bm25.json    (tokenized corpus)
    """

    def __init__(self, embed_dim: int):
        self.embed_dim = embed_dim
        self.chunks: list[StoredChunk] = []
        self.vectors: np.ndarray | None = None  # (N, D) float32 normalized
        self.bm25: BM25Okapi | None = None
        self._bm25_tokens: list[list[str]] = []
        # Parallel to self.chunks: colbert_vectors[i] is the (n_tokens, dim)
        # late-interaction matrix for chunk i, or None if not precomputed
        # (older store, or ColBERT was unavailable at ingest time).
        self.colbert_vectors: list[np.ndarray | None] | None = None

    def is_built(self) -> bool:
        return (len(self.chunks) > 0) and (self.bm25 is not None)

    def load(self, load_vectors: bool = True) -> bool:
        if not (os.path.exists(CHUNKS_PATH) and os.path.exists(BM25_PATH)):
            return False

        if load_vectors and not os.path.exists(VECTORS_PATH):
            return False

        # chunks
        self.chunks = []
        with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                self.chunks.append(StoredChunk(text=obj["text"], metadata=obj["metadata"]))

        # vectors
        # ponytail: brute-force cosine over an in-RAM matrix; fine to ~100k
        # chunks, swap in an ANN index (faiss/hnswlib) past that.
        if not load_vectors:
            self.vectors = None
        else:
            self.vectors = np.load(VECTORS_PATH).astype("float32")
            if self.vectors.ndim != 2 or self.vectors.shape[1] != self.embed_dim:
                raise RuntimeError(f"vectors.npy shape mismatch. got {self.vectors.shape}, expected (*, {self.embed_dim})")

        # bm25
        with open(BM25_PATH, "r", encoding="utf-8") as f:
            bm = json.load(f)
        self._bm25_tokens = bm["tokens"]
        self.bm25 = BM25Okapi(self._bm25_tokens)

        # colbert vectors (optional: absent on stores built before this cache existed)
        if os.path.exists(COLBERT_VECS_PATH):
            npz = np.load(COLBERT_VECS_PATH, allow_pickle=True)
            arr = npz["vecs"]
            self.colbert_vectors = [
                None if v is None or (hasattr(v, "size") and v.size == 0) else np.asarray(v, dtype="float32")
                for v in arr
            ]
            if len(self.colbert_vectors) != len(self.chunks):
                # Stale/mismatched cache (e.g. re-ingested chunks without regenerating this
                # file) — degrade to on-the-fly encoding rather than serve wrong vectors.
                self.colbert_vectors = None
        else:
            self.colbert_vectors = None

        return True

    def save(self) -> None:
        ensure_storage_dir()

        with open(CHUNKS_PATH, "w", encoding="utf-8") as f:
            f.writelines(json.dumps({"text": ch.text, "metadata": ch.metadata}, ensure_ascii=False) + "\n" for ch in self.chunks)

        if self.vectors is None:
            raise RuntimeError("vectors missing")
        np.save(VECTORS_PATH, self.vectors.astype("float32"))

        with open(BM25_PATH, "w", encoding="utf-8") as f:
            json.dump({"tokens": self._bm25_tokens}, f)

        if self.colbert_vectors is not None:
            arr = np.empty(len(self.colbert_vectors), dtype=object)
            for i, v in enumerate(self.colbert_vectors):
                arr[i] = v if v is not None else np.empty((0, 0), dtype="float32")
            np.savez(COLBERT_VECS_PATH, vecs=arr)

    def set_colbert_vectors(self, vecs: list[np.ndarray | None]) -> None:
        if len(vecs) != len(self.chunks):
            raise ValueError(f"colbert vector/chunk mismatch: {len(vecs)} vs {len(self.chunks)}")
        self.colbert_vectors = vecs

    def build(self, embeddings: list[list[float]], chunks: list[StoredChunk]) -> None:
        if not chunks:
            raise ValueError("No chunks to build index.")
        if not embeddings:
            raise ValueError("No embeddings to build index.")
        if len(embeddings) != len(chunks):
            raise ValueError(f"Embedding/chunk mismatch: {len(embeddings)} vs {len(chunks)}")

        self.chunks = chunks

        X = np.array(embeddings, dtype="float32")
        # ✅ Normalize for cosine similarity
        norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
        X = X / norms
        self.vectors = X

        # BM25
        self._bm25_tokens = [simple_tokenize(c.text) for c in chunks]
        self.bm25 = BM25Okapi(self._bm25_tokens)

    def search_bm25(self, query: str, top_k: int = 10) -> list[tuple[int, float]]:
        if self.bm25 is None:
            return []
        q = simple_tokenize(query)
        scores = self.bm25.get_scores(q)
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]
        return [(int(i), float(s)) for i, s in ranked if s > 0]

    def search_vector(self, query_vec: list[float], top_k: int = 10) -> list[tuple[int, float]]:
        if self.vectors is None:
            return []
        q = np.array(query_vec, dtype="float32")
        q = q / (np.linalg.norm(q) + 1e-12)

        sims = self.vectors @ q  # cosine
        top_idx = np.argsort(-sims)[:top_k]
        return [(int(i), float(sims[int(i)])) for i in top_idx]
