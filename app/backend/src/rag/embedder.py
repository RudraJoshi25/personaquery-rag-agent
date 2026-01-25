from __future__ import annotations
import os
from dataclasses import dataclass
from typing import List

from sentence_transformers import SentenceTransformer

DEFAULT_EMBED_MODEL = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")


@dataclass
class Embedder:
    model_name: str = DEFAULT_EMBED_MODEL

    def __post_init__(self):
        self.model = SentenceTransformer(self.model_name)
        # ✅ dim works for sentence-transformers models
        self.dim = int(self.model.get_sentence_embedding_dimension())

    def embed(self, texts: List[str]) -> List[List[float]]:
        # ✅ Guard: no texts
        if not texts:
            return []

        cleaned = [t.strip() if t is not None else "" for t in texts]
        if any(not t for t in cleaned):
            raise ValueError("Empty text passed to embed(). Filter empties before embedding.")

        vecs = self.model.encode(
            cleaned,
            normalize_embeddings=True,      # ✅ cosine-ready
            show_progress_bar=True,
            batch_size=int(os.getenv("EMBED_BATCH", "16")),
        )
        return vecs.tolist()


def get_embedder() -> Embedder:
    return Embedder()
