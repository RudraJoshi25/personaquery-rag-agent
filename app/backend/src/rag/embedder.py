from __future__ import annotations
import os
from dataclasses import dataclass, field
from typing import List, Optional

DEFAULT_EMBED_MODEL = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")

# FastEmbed model ids differ from the sentence-transformers short names.
_FASTEMBED_ALIASES = {
    "all-MiniLM-L6-v2": "BAAI/bge-small-en-v1.5",
    "sentence-transformers/all-MiniLM-L6-v2": "BAAI/bge-small-en-v1.5",
}

# Backend selection:
#   "fastembed" — ONNX Runtime, ~50MB of models, no torch.
#   "st"        — sentence-transformers (pulls torch, ~2GB installed).
#   "auto"      — prefer fastembed, fall back to sentence-transformers.
#
# fastembed is the default because torch is what makes this service too heavy
# for a 512MB free-tier container, and because keeping embeddings in-process
# avoids the 150-350ms network hop a hosted embedding API would add to every
# query. At this corpus size that hop costs more than it buys.
EMBED_BACKEND = os.getenv("EMBED_BACKEND", "auto").lower()


@dataclass
class Embedder:
    model_name: str = DEFAULT_EMBED_MODEL
    backend: str = field(default="", init=False)
    dim: int = field(default=0, init=False)

    def __post_init__(self):
        self._st = None
        self._fe = None

        if EMBED_BACKEND in {"auto", "fastembed"}:
            if self._try_fastembed():
                return
            if EMBED_BACKEND == "fastembed":
                raise RuntimeError(
                    "EMBED_BACKEND=fastembed but the fastembed package is not installed."
                )

        self._init_sentence_transformers()

    # -- backends ---------------------------------------------------------

    def _try_fastembed(self) -> bool:
        try:
            from fastembed import TextEmbedding
        except ImportError:
            return False

        fe_name = _FASTEMBED_ALIASES.get(self.model_name, self.model_name)
        self._fe = TextEmbedding(model_name=fe_name)
        self.backend = "fastembed"
        # Probe once rather than trusting a hardcoded table.
        self.dim = len(next(iter(self._fe.embed(["dimension probe"]))))
        return True

    def _init_sentence_transformers(self) -> None:
        from sentence_transformers import SentenceTransformer

        self._st = SentenceTransformer(self.model_name)
        self.backend = "sentence-transformers"
        self.dim = int(self._st.get_sentence_embedding_dimension())

    # -- api --------------------------------------------------------------

    def embed(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []

        cleaned = [(t or "").strip() for t in texts]
        if any(not t for t in cleaned):
            raise ValueError("Empty text passed to embed(). Filter empties before embedding.")

        if self._fe is not None:
            import numpy as np

            # Batch explicitly. fastembed defaults to batch_size=256, which
            # asks ONNX Runtime for one attention tensor covering every chunk
            # at once — on a 120-chunk corpus that was a ~908MB allocation and
            # it died with "Failed to allocate memory", killing ingestion.
            # (The ColBERT precompute already batches for the same reason;
            # this path was simply missed.) Small batches cost a little
            # ingest-time speed and remove the memory cliff entirely.
            batch = int(os.getenv("EMBED_BATCH", "16"))
            out: List[List[float]] = []
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

        vecs = self._st.encode(
            cleaned,
            normalize_embeddings=True,
            show_progress_bar=False,
            batch_size=int(os.getenv("EMBED_BATCH", "16")),
        )
        return vecs.tolist()

    def embed_query(self, text: str) -> List[float]:
        return self.embed([text])[0]


_singleton: Optional[Embedder] = None


def get_embedder() -> Embedder:
    """Process-wide singleton — model load is the expensive part."""
    global _singleton
    if _singleton is None:
        _singleton = Embedder()
        print(f"[embedder] backend={_singleton.backend} dim={_singleton.dim}", flush=True)
    return _singleton
