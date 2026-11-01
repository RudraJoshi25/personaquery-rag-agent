from __future__ import annotations

import os
from dataclasses import dataclass, field

DEFAULT_EMBED_MODEL = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")

# FastEmbed model ids differ from the sentence-transformers short names.
_FASTEMBED_ALIASES = {
    "all-MiniLM-L6-v2": "BAAI/bge-small-en-v1.5",
    "sentence-transformers/all-MiniLM-L6-v2": "BAAI/bge-small-en-v1.5",
}

# FastEmbed (ONNX Runtime) only: torch would not fit a 512MB free-tier
# container, and in-process embedding avoids the 150-350ms network hop a
# hosted embedding API would add to every query.


@dataclass
class Embedder:
    model_name: str = DEFAULT_EMBED_MODEL
    backend: str = field(default="fastembed", init=False)
    dim: int = field(default=0, init=False)

    def __post_init__(self):
        from fastembed import TextEmbedding

        from src.core.config import MODEL_CACHE_DIR

        fe_name = _FASTEMBED_ALIASES.get(self.model_name, self.model_name)
        self._fe = TextEmbedding(model_name=fe_name, cache_dir=MODEL_CACHE_DIR)
        # Probe once rather than trusting a hardcoded table.
        self.dim = len(next(iter(self._fe.embed(["dimension probe"]))))

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        cleaned = [(t or "").strip() for t in texts]
        if any(not t for t in cleaned):
            raise ValueError("Empty text passed to embed(). Filter empties before embedding.")

        import numpy as np

        # Batch explicitly. fastembed defaults to batch_size=256, which asks
        # ONNX Runtime for one attention tensor covering every chunk at once —
        # on a 120-chunk corpus that was a ~908MB allocation and ingestion
        # died with "Failed to allocate memory".
        batch = int(os.getenv("EMBED_BATCH", "16"))
        out: list[list[float]] = []
        for i in range(0, len(cleaned), batch):
            part = cleaned[i : i + batch]
            vecs = np.array(
                list(self._fe.embed(part, batch_size=len(part))), dtype="float32"
            )
            # fastembed does not normalize; cosine search assumes unit vectors.
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            out.extend((vecs / norms).tolist())
        return out

    def embed_query(self, text: str) -> list[float]:
        return self.embed([text])[0]


_singleton: Embedder | None = None


def get_embedder() -> Embedder:
    """Process-wide singleton — model load is the expensive part."""
    global _singleton
    if _singleton is None:
        _singleton = Embedder()
        print(f"[embedder] backend={_singleton.backend} dim={_singleton.dim}", flush=True)
    return _singleton
