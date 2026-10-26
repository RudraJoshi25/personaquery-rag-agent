"""FAQ build script: overrides resolve against the real index, and --only
merges one entry without touching the rest."""
from __future__ import annotations

import json
import re

from scripts import precompute_faq as pf
from src.rag import llm
from src.rag.faq_cache import index_sha


class _FakeEmbedder:
    def embed_query(self, q):
        return [1.0, 0.0]


def test_overrides_resolve_and_cite_every_source():
    for q, o in pf.FAQ_OVERRIDES.items():
        sources = pf._resolve_sources(o["sources"])
        cited = {int(n) for m in re.findall(r"\[\[cite:([\d,\s]+)\]\]", o["answer"])
                 for n in m.split(",")}
        assert cited == {s["id"] for s in sources}, q


def test_only_merges_single_entry(tmp_path, monkeypatch):
    out = tmp_path / "faq_cache.json"
    old = {"index_sha": "stale", "entries": [
        {"question": "Keep me?", "answer": "kept [[cite:1]]", "sources": [], "embedding": [0, 1]},
        {"question": "Redo me?", "answer": "old [[cite:1]]", "sources": [], "embedding": [0, 1]},
    ]}
    out.write_text(json.dumps(old), encoding="utf-8")
    monkeypatch.setattr(pf, "OUT_PATH", out)
    monkeypatch.setattr(pf, "STORAGE_DIR", str(tmp_path))
    monkeypatch.setattr(pf, "get_embedder", lambda: _FakeEmbedder())
    monkeypatch.setattr(pf, "run_rag", lambda q, **k: {"answer": "new [[cite:1]]", "sources": []})
    monkeypatch.setattr(llm, "GROQ_MAX_TOKENS", llm.GROQ_MAX_TOKENS)  # main() raises it; restore

    assert pf.main(["--only", "redo me?"]) == 0

    data = json.loads(out.read_text(encoding="utf-8"))
    by_q = {e["question"].lower(): e["answer"] for e in data["entries"]}
    assert by_q == {"keep me?": "kept [[cite:1]]", "redo me?": "new [[cite:1]]"}
    assert data["index_sha"] == index_sha()


def test_only_keeps_successes_when_a_later_entry_fails(tmp_path, monkeypatch):
    out = tmp_path / "faq_cache.json"
    out.write_text(json.dumps({"index_sha": "x", "entries": []}), encoding="utf-8")
    monkeypatch.setattr(pf, "OUT_PATH", out)
    monkeypatch.setattr(pf, "STORAGE_DIR", str(tmp_path))
    monkeypatch.setattr(pf, "get_embedder", lambda: _FakeEmbedder())
    monkeypatch.setattr(pf.time, "sleep", lambda s: None)
    monkeypatch.setattr(llm, "GROQ_MAX_TOKENS", llm.GROQ_MAX_TOKENS)
    answers = {"good?": "fine [[cite:1]].", "bad?": "I'm having trouble generating a polished answer"}
    monkeypatch.setattr(pf, "run_rag", lambda q, **k: {"answer": answers[q], "sources": []})

    assert pf.main(["--only", "good?", "--only", "bad?"]) == 1

    entries = json.loads(out.read_text(encoding="utf-8"))["entries"]
    assert [e["question"] for e in entries] == ["good?"]
