# app/backend/scripts/migrate_to_chroma.py
"""
One-time migration: existing storage/chunks.jsonl + storage/vectors.npy
-> persistent ChromaDB collection. No re-embedding (reuses saved vectors).

Run from app/backend/:  python -m scripts.migrate_to_chroma
"""
from __future__ import annotations

import json
import os

import numpy as np

from src.core.config import CHROMA_DIR, CHROMA_COLLECTION, RAG_STORAGE_DIR
from src.rag.vector_backend import ChromaBackend

CHUNKS_PATH = os.path.join(RAG_STORAGE_DIR, "chunks.jsonl")
VECTORS_PATH = os.path.join(RAG_STORAGE_DIR, "vectors.npy")


def main() -> None:
    if not (os.path.exists(CHUNKS_PATH) and os.path.exists(VECTORS_PATH)):
        raise SystemExit(f"Missing {CHUNKS_PATH} or {VECTORS_PATH}. Run ingest first.")

    chunks = []
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))

    vectors = np.load(VECTORS_PATH).astype("float32")
    if len(vectors) != len(chunks):
        raise SystemExit(f"Mismatch: {len(vectors)} vectors vs {len(chunks)} chunks")

    print(f"Migrating {len(chunks)} chunks (dim={vectors.shape[1]}) -> Chroma at {CHROMA_DIR}")

    backend = ChromaBackend(persist_dir=CHROMA_DIR, collection_name=CHROMA_COLLECTION)
    backend.upsert(
        ids=[str(i) for i in range(len(chunks))],
        embeddings=[vectors[i].tolist() for i in range(len(chunks))],
        documents=[c["text"] for c in chunks],
        metadatas=[c["metadata"] for c in chunks],
    )

    print(f"Done. Collection '{CHROMA_COLLECTION}' now has {backend.count()} vectors.")


if __name__ == "__main__":
    main()
