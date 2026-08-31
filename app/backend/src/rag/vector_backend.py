# src/rag/vector_backend.py
"""
Swappable vector-store backends for the hybrid retriever.

The retriever only needs two things from a backend:
  - upsert(ids, embeddings, documents, metadatas)  : load vectors
  - query(embedding, top_k) -> [(chunk_index, similarity), ...]

Backends return the integer chunk index (matching the order of chunks.jsonl)
so the rest of the pipeline (retrieve_custom.py) is unchanged.

Today: ChromaBackend (persistent, HNSW-indexed, no Docker required).
Phase 7 will add a PgVectorBackend behind this same interface.
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Sequence, Tuple, Protocol


class VectorBackend(Protocol):
    def upsert(
        self,
        ids: Sequence[str],
        embeddings: Sequence[Sequence[float]],
        documents: Sequence[str],
        metadatas: Sequence[Dict[str, Any]],
    ) -> None: ...

    def query(self, embedding: Sequence[float], top_k: int) -> List[Tuple[int, float]]: ...

    def count(self) -> int: ...

    def reset(self) -> None: ...


class ChromaBackend:
    """Persistent ChromaDB collection using cosine space."""

    def __init__(
        self,
        persist_dir: str,
        collection_name: str = "personaquery",
    ) -> None:
        import chromadb  # imported lazily so numpy-only mode needs no chromadb

        self.persist_dir = persist_dir
        self.collection_name = collection_name
        os.makedirs(persist_dir, exist_ok=True)
        self._client = chromadb.PersistentClient(path=persist_dir)
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def upsert(
        self,
        ids: Sequence[str],
        embeddings: Sequence[Sequence[float]],
        documents: Sequence[str],
        metadatas: Sequence[Dict[str, Any]],
    ) -> None:
        # Chroma metadata values must be scalar (str/int/float/bool); coerce.
        clean_meta = [
            {k: v for k, v in m.items() if isinstance(v, (str, int, float, bool))}
            for m in metadatas
        ]
        self._collection.upsert(
            ids=list(ids),
            embeddings=[list(e) for e in embeddings],
            documents=list(documents),
            metadatas=clean_meta,
        )

    def query(self, embedding: Sequence[float], top_k: int) -> List[Tuple[int, float]]:
        if self.count() == 0:
            return []
        res = self._collection.query(
            query_embeddings=[list(embedding)],
            n_results=min(top_k, self.count()),
        )
        ids = res.get("ids", [[]])[0]
        dists = res.get("distances", [[]])[0]
        out: List[Tuple[int, float]] = []
        for i, d in zip(ids, dists):
            try:
                idx = int(i)
            except ValueError:
                continue
            # cosine distance -> cosine similarity
            out.append((idx, 1.0 - float(d)))
        return out

    def count(self) -> int:
        return self._collection.count()

    def reset(self) -> None:
        """Drop and recreate the collection.

        Ingest addresses vectors by their positional chunk index ("0", "1",
        ...). Upsert alone therefore OVERWRITES ids that still exist but
        never REMOVES ids past the end of a corpus that shrank: re-ingesting
        127 chunks down to 120 left ids 120-126 alive in Chroma, so vector
        search returned a doc_id with no corresponding entry in
        chunks.jsonl and retrieval raised IndexError on every query.

        A full reset is the right call rather than a targeted delete of the
        tail: ingest rebuilds the entire index from scratch anyway, so any
        surviving vector is by definition stale.
        """
        try:
            self._client.delete_collection(self.collection_name)
        except Exception:
            # Collection may not exist yet on a first run — that's fine.
            pass
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )


def get_vector_backend(
    backend: str,
    persist_dir: str,
    collection_name: str = "personaquery",
) -> VectorBackend:
    backend = (backend or "").lower()
    if backend == "chroma":
        return ChromaBackend(persist_dir=persist_dir, collection_name=collection_name)
    raise ValueError(f"Unknown VECTOR_BACKEND: {backend!r}")
