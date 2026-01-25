# app/backend/scripts/ingest.py
from __future__ import annotations

import os
from pathlib import Path
from typing import List

from src.core.config import PRIVATE_DATA_DIR
from src.rag.embedder import get_embedder
from src.rag.store import HybridStore
from src.rag.ingest_pipeline import ingest_paths


def _collect_paths() -> List[str]:
    """
    Collect PDFs/txt/md from your configured private data directory.
    PRIVATE_DATA_DIR resolves to ../../data/private (repo_root/data/private).
    """
    base = Path(PRIVATE_DATA_DIR).resolve()
    if not base.exists():
        print(f"❌ PRIVATE_DATA_DIR not found: {base}")
        return []

    exts = {".pdf", ".txt", ".md"}
    paths = [str(p) for p in base.rglob("*") if p.suffix.lower() in exts]

    print(f"📁 PRIVATE_DATA_DIR: {base}")
    print(f"📄 Found {len(paths)} file(s):")
    for p in paths:
        print("   -", p)
    return paths


def main() -> None:
    print("🚀 Running ingest:", __file__)
    print("🧭 CWD:", os.getcwd())

    paths = _collect_paths()
    if not paths:
        print("❌ No ingestible files found. Put PDFs in data/private and retry.")
        return

    embedder = get_embedder()
    store = HybridStore(embed_dim=embedder.dim)

    ingest_paths(paths, store)

    # store.save() happens inside ingest_paths() in your pipeline
    print("✅ Ingest complete.")
    print("   Output written to: ./storage (relative to app/backend)")


if __name__ == "__main__":
    main()
