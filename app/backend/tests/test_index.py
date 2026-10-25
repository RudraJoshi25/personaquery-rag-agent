"""The committed index must be internally consistent and built from the
committed curated docs. CI can't re-ingest (data/private isn't in git), so
this is how a doc edit without `python -m scripts.ingest` gets caught."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pytest

STORAGE = Path(__file__).resolve().parents[1] / "storage"
CURATED = Path(__file__).resolve().parents[3] / "data" / "curated"

pytestmark = pytest.mark.skipif(not (STORAGE / "chunks.jsonl").exists(), reason="index not built")


def test_index_files_agree_on_chunk_count():
    n = sum(1 for line in (STORAGE / "chunks.jsonl").read_text(encoding="utf-8").splitlines() if line.strip())
    assert np.load(STORAGE / "vectors.npy").shape[0] == n
    assert len(json.loads((STORAGE / "bm25.json").read_text(encoding="utf-8"))["tokens"]) == n
    colbert = STORAGE / "colbert_vecs.npz"
    if colbert.exists():
        assert len(np.load(colbert, allow_pickle=True)["vecs"]) == n


def test_curated_docs_match_index():
    recorded = json.loads((STORAGE / "sources.json").read_text(encoding="utf-8"))
    for doc in sorted(CURATED.glob("*.md")):
        # Hash with LF endings: git may check out CRLF on Windows.
        sha = hashlib.sha256(doc.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
        assert recorded.get(doc.name) == sha, f"{doc.name} changed since ingest; run python -m scripts.ingest"


def test_phone_number_never_indexed():
    from src.rag.ingest_pipeline import _clean_pdf_text
    out = _clean_pdf_text("RUDRA \u2642phone+61 400111222 /envelopeme@x.com")
    assert "400111222" not in out
    assert "Phone: available on request" in out and "Email: me@x.com" in out
    with open(STORAGE / "chunks.jsonl", encoding="utf-8") as f:
        assert not re.search(r"\+61\s?4\d{8}", f.read())
