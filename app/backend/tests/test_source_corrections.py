"""Stale CV claims never reach the index, and every correction still fires.
Delete with src/rag/source_corrections.py once the CV PDF is updated."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.core.config import PRIVATE_DATA_DIR, RAG_STORAGE_DIR
from src.rag.source_corrections import APPLIES_TO, CORRECTIONS, apply_corrections

BANNED = ["patent holder", "ChromaDB", "LangChain", "LlamaIndex", "Sentence-Transformers",
          "Llama-3.3-70B", "all-MiniLM", "29 tests", "20-case"]
CV = Path(PRIVATE_DATA_DIR) / APPLIES_TO


def test_banned_claims_not_in_indexed_cv():
    # CV chunks only: projects.md legitimately names Llama-3.3-70B as a past model.
    lines = (Path(RAG_STORAGE_DIR) / "chunks.jsonl").read_text(encoding="utf-8").splitlines()
    cv = [c["text"] for c in map(json.loads, lines) if c["metadata"].get("file_name") == APPLIES_TO]
    assert cv, "CV not in index"
    assert not [b for b in BANNED if any(b in t for t in cv)]


@pytest.mark.skipif(not CV.exists(), reason="private CV not present (CI)")
def test_cv_corrections_all_fire_and_clean_it():
    from pypdf import PdfReader

    from src.rag.ingest_pipeline import _clean_pdf_text, read_pdf_pages

    raw = "\n".join(_clean_pdf_text(p.extract_text() or "") for p in PdfReader(CV).pages)
    _, counts = apply_corrections(raw)
    assert [p for (p, _), n in zip(CORRECTIONS, counts) if n == 0] == []

    text = "\n".join(p["text"] for p in read_pdf_pages(str(CV)))
    assert not [b for b in BANNED if b in text]
