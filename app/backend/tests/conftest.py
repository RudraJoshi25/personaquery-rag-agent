import os

# Must happen before any `src...` import anywhere in the session - config.py
# reads GROQ_API_KEY/etc. as module-level constants at import time, so
# monkeypatch.setenv after import is too late for these.
os.environ.setdefault("GROQ_API_KEY", "test-dummy-key")
os.environ.setdefault("RAG_VECTOR_ENABLED", "true")
os.environ.setdefault("INJECTION_GUARD_ENABLED", "true")
os.environ.setdefault("LOG_FORMAT", "text")

import pytest
from fastapi.testclient import TestClient

from src.main import app as fastapi_app


@pytest.fixture()
def client() -> TestClient:
    return TestClient(fastapi_app)


def _mock_response(mocker, status_code, json_data=None, text=""):
    resp = mocker.Mock(status_code=status_code, text=text)
    if json_data is not None:
        resp.json.return_value = json_data
    return resp


@pytest.fixture()
def mock_groq_success(mocker):
    """Returns a factory: mock_groq_success("some answer [[cite:1]]", usage={...})"""

    def _make(content: str, usage: dict | None = None):
        resp = _mock_response(
            mocker,
            200,
            json_data={"choices": [{"message": {"content": content}}], "usage": usage},
        )
        return mocker.patch("src.rag.generation.llm_groq._session.post", return_value=resp)

    return _make


@pytest.fixture()
def mock_groq_error(mocker):
    def _make(status_code: int = 500, text: str = "internal error"):
        resp = _mock_response(mocker, status_code, text=text)
        return mocker.patch("src.rag.generation.llm_groq._session.post", return_value=resp)

    return _make


@pytest.fixture()
def mock_groq_exception(mocker):
    def _make(exc: Exception):
        return mocker.patch("src.rag.generation.llm_groq._session.post", side_effect=exc)

    return _make


@pytest.fixture()
def patch_storage_paths(monkeypatch, tmp_path):
    """retrieval/store.py binds CHUNKS_PATH/VECTORS_PATH/BM25_PATH at import
    time from env; env-var monkeypatching after import has no effect, so
    patch the already-bound module attributes directly."""
    monkeypatch.setattr("src.rag.retrieval.store.CHUNKS_PATH", str(tmp_path / "chunks.jsonl"))
    monkeypatch.setattr("src.rag.retrieval.store.VECTORS_PATH", str(tmp_path / "vectors.npy"))
    monkeypatch.setattr("src.rag.retrieval.store.BM25_PATH", str(tmp_path / "bm25.json"))
    return tmp_path


@pytest.fixture()
def tiny_store(patch_storage_paths):
    """Small synthetic HybridStore for fast, deterministic unit tests - no
    ML model, no real committed storage/ files."""
    from src.rag.retrieval.store import HybridStore, StoredChunk

    chunks = [
        StoredChunk(
            text="Rudra built a RAG chatbot using Groq and hybrid search.",
            metadata={"file_name": "resume.pdf", "page_label": "1", "section": "Projects"},
        ),
        StoredChunk(
            text="Skills include Python, PyTorch, distributed systems.",
            metadata={"file_name": "resume.pdf", "page_label": "1", "section": "Skills"},
        ),
        StoredChunk(
            text="Filed a patent on anomaly detection in telemetry.",
            metadata={"file_name": "resume.pdf", "page_label": "2", "section": "Publications"},
        ),
    ]
    store = HybridStore(embed_dim=2)
    store.build(embeddings=[[1.0, 0.0], [0.0, 1.0], [0.7, 0.7]], chunks=chunks)
    store.save()
    store.load(load_vectors=True)
    return store


class _FakeModel:
    def encode(self, texts, normalize_embeddings=True):
        # retrieval/search.py calls .tolist() on the result, matching real
        # SentenceTransformer.encode()'s numpy-array return type.
        import numpy as np

        return np.array([[1.0, 0.0] for _ in texts], dtype="float32")


@pytest.fixture()
def patched_retrieval(monkeypatch, tiny_store):
    """Point retrieval.search's module-level singletons at the synthetic
    store + a fake embedder, so retrieve()/make_context_pack() run fast and
    deterministically."""
    import src.rag.retrieval.search as rc

    monkeypatch.setattr(rc, "_store", tiny_store)
    monkeypatch.setattr(rc, "_get_model", lambda: _FakeModel())
    return rc


@pytest.fixture(scope="session")
def real_store():
    from src.rag.retrieval.search import _get_store

    return _get_store()


@pytest.fixture(autouse=True)
def _clear_interview_sessions():
    yield
    try:
        from src.rag.interview import _SESSIONS

        _SESSIONS.clear()
    except ImportError:
        pass
