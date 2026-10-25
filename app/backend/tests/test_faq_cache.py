"""FAQ cache must refuse to serve answers built from a different index."""
from __future__ import annotations

import json

from src.rag import faq_cache


def test_stale_index_hash_disables_cache(tmp_path, monkeypatch):
    chunks = tmp_path / "chunks.jsonl"
    chunks.write_text('{"text": "a"}\n', encoding="utf-8")
    faq = tmp_path / "faq_cache.json"
    entry = {"question": "q", "answer": "a", "embedding": [1.0]}
    monkeypatch.setattr(faq_cache, "CHUNKS_PATH", chunks)
    monkeypatch.setattr(faq_cache, "FAQ_PATH", faq)
    monkeypatch.setattr(faq_cache, "FAQ_ENABLED", True)

    for sha, expect_loaded in ((faq_cache.index_sha(chunks), True), ("other", False)):
        faq.write_text(json.dumps({"index_sha": sha, "entries": [entry]}), encoding="utf-8")
        monkeypatch.setattr(faq_cache, "_loaded", False)
        monkeypatch.setattr(faq_cache, "_cache", None)
        assert (faq_cache._load() is not None) is expect_loaded
