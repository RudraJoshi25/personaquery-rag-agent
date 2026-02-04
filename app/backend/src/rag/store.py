from __future__ import annotations
import json
import os
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
from rank_bm25 import BM25Okapi

STORAGE_DIR = os.getenv("RAG_STORAGE_DIR", "storage")

CHUNKS_PATH = os.path.join(STORAGE_DIR, "chunks.jsonl")
VECTORS_PATH = os.path.join(STORAGE_DIR, "vectors.npy")
BM25_PATH = os.path.join(STORAGE_DIR, "bm25.json")


def ensure_storage_dir():
    os.makedirs(STORAGE_DIR, exist_ok=True)


def simple_tokenize(text: str) -> List[str]:
    import re
    return re.findall(r"[a-z0-9]+", (text or "").lower())


@dataclass
class StoredChunk:
    text: str
    metadata: Dict[str, Any]


class HybridStore:
    """
    Persisted:
      - chunks.jsonl (text + metadata)
      - vectors.npy  (float32 normalized embeddings)
      - bm25.json    (tokenized corpus)
    """

    def __init__(self, embed_dim: int):
        self.embed_dim = embed_dim
        self.chunks: List[StoredChunk] = []
        self.vectors: Optional[np.ndarray] = None  # (N, D) float32 normalized
        self.bm25: Optional[BM25Okapi] = None
        self._bm25_tokens: List[List[str]] = []

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

        # vectors (optional)
        if load_vectors:
            self.vectors = np.load(VECTORS_PATH).astype("float32")
            if self.vectors.ndim != 2 or self.vectors.shape[1] != self.embed_dim:
                raise RuntimeError(f"vectors.npy shape mismatch. got {self.vectors.shape}, expected (*, {self.embed_dim})")
        else:
            self.vectors = None

        # bm25
        with open(BM25_PATH, "r", encoding="utf-8") as f:
            bm = json.load(f)
        self._bm25_tokens = bm["tokens"]
        self.bm25 = BM25Okapi(self._bm25_tokens)

        return True

    def save(self) -> None:
        ensure_storage_dir()

        with open(CHUNKS_PATH, "w", encoding="utf-8") as f:
            for ch in self.chunks:
                f.write(json.dumps({"text": ch.text, "metadata": ch.metadata}, ensure_ascii=False) + "\n")

        if self.vectors is None:
            raise RuntimeError("vectors missing")
        np.save(VECTORS_PATH, self.vectors.astype("float32"))

        with open(BM25_PATH, "w", encoding="utf-8") as f:
            json.dump({"tokens": self._bm25_tokens}, f)

    def build(self, embeddings: List[List[float]], chunks: List[StoredChunk]) -> None:
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

    def search_bm25(self, query: str, top_k: int = 10) -> List[Tuple[int, float]]:
        if self.bm25 is None:
            return []
        q = simple_tokenize(query)
        scores = self.bm25.get_scores(q)
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]
        return [(int(i), float(s)) for i, s in ranked if s > 0]

    def search_vector(self, query_vec: List[float], top_k: int = 10) -> List[Tuple[int, float]]:
        if self.vectors is None:
            return []
        q = np.array(query_vec, dtype="float32")
        q = q / (np.linalg.norm(q) + 1e-12)

        sims = self.vectors @ q  # cosine
        top_idx = np.argsort(-sims)[:top_k]
        return [(int(i), float(sims[int(i)])) for i in top_idx]
