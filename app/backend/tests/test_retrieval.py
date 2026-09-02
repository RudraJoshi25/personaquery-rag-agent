import pytest

from src.rag.retrieval.search import make_context_pack, retrieve


def test_hybrid_retrieval_returns_hits(patched_retrieval):
    hits = retrieve("Rudra RAG Groq", top_k=3)
    assert len(hits) >= 1
    assert all("text" in h and "metadata" in h for h in hits)


def test_retrieval_channel_tagging(patched_retrieval):
    hits = retrieve("Rudra RAG Groq", top_k=3)
    channels = {h["metadata"]["channel"] for h in hits}
    assert channels.issubset({"hybrid", "vector", "keyword"})


def test_retrieval_top_k_truncation(patched_retrieval):
    hits = retrieve("Rudra RAG Groq patent skills", top_k=1)
    assert len(hits) <= 1


def test_retrieval_dedup_by_file_and_page(patched_retrieval, monkeypatch):
    # All three synthetic chunks live on (file_name, page_label) pairs:
    # (resume.pdf, 1) x2, (resume.pdf, 2) x1 - so a query hitting all
    # three should collapse the two page-1 chunks to one.
    hits = retrieve("Rudra Python distributed systems", top_k=10)
    keys = [(h["metadata"]["file_name"], h["metadata"]["page_label"]) for h in hits]
    assert len(keys) == len(set(keys))


def test_bm25_fallback_when_vector_disabled(patched_retrieval, monkeypatch):
    monkeypatch.setenv("RAG_VECTOR_ENABLED", "false")
    hits = retrieve("anomaly detection patent", top_k=5)
    assert len(hits) >= 1
    assert all(h["metadata"]["channel"] == "keyword" for h in hits)


def test_make_context_pack_header_format():
    hits = [
        {
            "text": "some chunk text",
            "metadata": {"file_name": "resume.pdf", "page_label": "1", "section": "Skills"},
        }
    ]
    pack = make_context_pack(hits, source_ids=[1])
    assert "[SOURCE 1] resume.pdf | p.1 | Skills" in pack
    assert "some chunk text" in pack


def test_make_context_pack_respects_max_chars():
    hits = [
        {"text": "x" * 100, "metadata": {"file_name": "a.pdf", "page_label": "1", "section": "S"}},
        {"text": "y" * 100, "metadata": {"file_name": "a.pdf", "page_label": "2", "section": "S"}},
    ]
    pack = make_context_pack(hits, max_chars=50)
    assert "x" * 100 not in pack
    assert pack == ""


def test_make_context_pack_empty_hits():
    assert make_context_pack([]) == ""


@pytest.mark.slow
def test_real_store_retrieval_smoke(real_store, monkeypatch):
    """Validates the actual shipped committed storage/ index isn't
    corrupted/drifted - the synthetic-store tests above structurally
    cannot catch this."""
    import src.rag.retrieval.search as rc

    monkeypatch.setattr(rc, "_store", real_store)
    hits = retrieve("What projects has Rudra worked on?", top_k=5)
    assert len(hits) >= 1
