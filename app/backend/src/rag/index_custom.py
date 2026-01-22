from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, List
import json

import numpy as np
from sentence_transformers import SentenceTransformer

from src.core.config import INDEX_PERSIST_DIR, EMBED_MODEL
from src.rag.ingest_pdf import load_pdf_chunks


INDEX_FILE = "index.npz"
META_FILE = "meta.json"

# For MVP speed: limit chunks so indexing doesn't feel "stuck".
# Set to None to index everything.
MAX_CHUNKS: int | None = 300


def build_index(private_data_dir: str) -> Dict[str, Any]:
    print("=== BUILD INDEX START ===", flush=True)
    print(f"private_data_dir={private_data_dir}", flush=True)
    print(f"persist_dir={INDEX_PERSIST_DIR}", flush=True)
    print(f"embed_model={EMBED_MODEL}", flush=True)

    print("📄 Step 1: Loading & chunking PDFs...", flush=True)
    chunks = load_pdf_chunks(private_data_dir)
    print(f"✅ Step 1 done: chunks created = {len(chunks)}", flush=True)

    if MAX_CHUNKS is not None and len(chunks) > MAX_CHUNKS:
        print(f"🚀 Using MAX_CHUNKS={MAX_CHUNKS} for quick build (MVP mode)", flush=True)
        chunks = chunks[:MAX_CHUNKS]
        print(f"✅ Trimmed chunks = {len(chunks)}", flush=True)

    print("🧠 Step 2: Loading embedding model...", flush=True)
    model = SentenceTransformer(EMBED_MODEL)
    print("✅ Step 2 done: model loaded", flush=True)

    texts = [c.text for c in chunks]
    print("⚡ Step 3: Creating embeddings (CPU can take a few minutes)...", flush=True)
    embeddings = model.encode(
        texts,
        batch_size=16,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    embeddings = np.array(embeddings, dtype=np.float32)
    print(f"✅ Step 3 done: embeddings shape = {embeddings.shape}", flush=True)

    print("💾 Step 4: Saving index to disk...", flush=True)
    persist_dir = Path(INDEX_PERSIST_DIR).resolve()
    persist_dir.mkdir(parents=True, exist_ok=True)

    # Save embeddings matrix
    np.savez_compressed(persist_dir / INDEX_FILE, embeddings=embeddings)

    # Save metadata + texts
    meta: List[Dict[str, Any]] = [{"text": c.text, "metadata": c.metadata} for c in chunks]
    (persist_dir / META_FILE).write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("✅ Step 4 done: saved index.npz and meta.json", flush=True)
    print("=== BUILD INDEX COMPLETE ===", flush=True)

    return {
        "chunks": len(chunks),
        "embedding_dim": int(embeddings.shape[1]) if embeddings.ndim == 2 else None,
        "persist_dir": str(persist_dir),
        "files": [INDEX_FILE, META_FILE],
        "max_chunks": MAX_CHUNKS,
    }


def load_index() -> Dict[str, Any]:
    persist_dir = Path(INDEX_PERSIST_DIR).resolve()
    idx_path = persist_dir / INDEX_FILE
    meta_path = persist_dir / META_FILE

    if not idx_path.exists() or not meta_path.exists():
        raise FileNotFoundError(
            f"Index not found in {persist_dir}. Expected {INDEX_FILE} and {META_FILE}. "
            f"Run build_index() first."
        )

    data = np.load(idx_path)
    embeddings = data["embeddings"]
    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    return {"embeddings": embeddings, "meta": meta}
