import pytest

from src.rag.retrieval.store import HybridStore, StoredChunk


def _chunks():
    return [
        StoredChunk(text="alpha beta gamma", metadata={"file_name": "a.pdf", "page_label": "1"}),
        StoredChunk(text="delta epsilon zeta", metadata={"file_name": "a.pdf", "page_label": "2"}),
    ]


def test_build_save_load_round_trip(patch_storage_paths):
    store = HybridStore(embed_dim=2)
    store.build(embeddings=[[1.0, 0.0], [0.0, 1.0]], chunks=_chunks())
    store.save()

    loaded = HybridStore(embed_dim=2)
    assert loaded.load(load_vectors=True) is True
    assert len(loaded.chunks) == 2
    assert loaded.chunks[0].text == "alpha beta gamma"
    assert loaded.vectors.shape == (2, 2)


def test_load_returns_false_when_missing(patch_storage_paths):
    store = HybridStore(embed_dim=2)
    assert store.load() is False


def test_load_raises_on_embed_dim_mismatch(patch_storage_paths):
    store = HybridStore(embed_dim=2)
    store.build(embeddings=[[1.0, 0.0], [0.0, 1.0]], chunks=_chunks())
    store.save()

    wrong_dim_store = HybridStore(embed_dim=5)
    with pytest.raises(RuntimeError):
        wrong_dim_store.load(load_vectors=True)


def test_build_raises_on_empty_chunks():
    store = HybridStore(embed_dim=2)
    with pytest.raises(ValueError):
        store.build(embeddings=[[1.0, 0.0]], chunks=[])


def test_build_raises_on_mismatched_lengths():
    store = HybridStore(embed_dim=2)
    with pytest.raises(ValueError):
        store.build(embeddings=[[1.0, 0.0]], chunks=_chunks())


def test_search_bm25_only_positive_scores_descending():
    # A 2-document corpus where a term appears in exactly half the docs
    # degenerates rank_bm25's IDF to exactly 0 (log((N-n+0.5)/(n+0.5)) with
    # N=2, n=1 == log(1) == 0), which the store's `s > 0` filter then drops.
    # Use 3 documents so the matched term has genuine positive IDF.
    chunks = _chunks() + [StoredChunk(text="eta theta iota", metadata={"file_name": "b.pdf", "page_label": "1"})]
    store = HybridStore(embed_dim=2)
    store.build(embeddings=[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], chunks=chunks)
    hits = store.search_bm25("alpha", top_k=10)
    assert len(hits) == 1
    assert hits[0][0] == 0
    assert hits[0][1] > 0


def test_search_vector_cosine_ranking():
    store = HybridStore(embed_dim=2)
    store.build(embeddings=[[1.0, 0.0], [0.0, 1.0]], chunks=_chunks())
    hits = store.search_vector([1.0, 0.0], top_k=2)
    assert hits[0][0] == 0
    assert hits[0][1] > hits[1][1]
