#!/usr/bin/env python
"""
Build step: download the ONNX models into MODEL_CACHE_DIR (app/backend/.models
unless FASTEMBED_CACHE_PATH is set), so nothing downloads at runtime.

Render native build command:

    pip install -r requirements.txt && python -m scripts.download_models
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core.config import MODEL_CACHE_DIR
from src.main import warm_up
from src.rag import rerank


def main() -> int:
    warm_up()
    # At runtime ColBERT degrades to fusion order if it fails to load; at
    # build time that would silently ship without reranking, so fail loudly.
    if rerank.RERANK_ENABLED and rerank._get_colbert() is None:
        print(f"ColBERT {rerank.COLBERT_MODEL} failed to load", file=sys.stderr)
        return 1
    print(f"models ready in {MODEL_CACHE_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
