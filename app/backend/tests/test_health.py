import pytest

import src.api.routes_health as health


@pytest.fixture()
def reset_embedder_cache():
    """_embedder_loadable() memoizes across calls in module globals; reset
    both before and after so these tests don't leak state into each other."""
    health._embedder_loadable_cache = None
    health._embedder_last_attempt_at = 0.0
    yield
    health._embedder_loadable_cache = None
    health._embedder_last_attempt_at = 0.0


def test_health_live_always_ok(client):
    r = client.get("/health/live")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_health_ready_reports_checks(client):
    r = client.get("/health/ready")
    assert r.status_code in (200, 503)
    body = r.json()
    assert set(body["checks"].keys()) == {"store_loaded", "embedder_loadable", "groq_key_present"}


def test_health_alias_matches_ready(client):
    ready = client.get("/health/ready")
    alias = client.get("/health")
    assert alias.status_code == ready.status_code
    assert alias.json() == ready.json()


def test_health_ready_degraded_when_storage_missing(client, monkeypatch, tmp_path):
    monkeypatch.setattr("src.api.routes_health.RAG_STORAGE_DIR", str(tmp_path / "missing"))
    r = client.get("/health/ready")
    assert r.status_code == 503
    assert r.json()["checks"]["store_loaded"] is False

    live = client.get("/health/live")
    assert live.status_code == 200


def test_embedder_success_is_cached(mocker, reset_embedder_cache):
    get_model = mocker.patch("src.rag.retrieval.search._get_model", return_value=object())

    assert health._embedder_loadable() is True
    assert health._embedder_loadable() is True
    assert health._embedder_loadable() is True

    # Loaded once, then served from cache for the life of the process.
    assert get_model.call_count == 1


def test_embedder_failure_is_not_retried_inside_backoff_window(mocker, reset_embedder_cache):
    get_model = mocker.patch("src.rag.retrieval.search._get_model", side_effect=MemoryError("OOM"))

    assert health._embedder_loadable() is False
    assert health._embedder_loadable() is False
    assert health._embedder_loadable() is False

    # One real attempt; subsequent probes short-circuit rather than
    # re-attempting an expensive load on an already-struggling process.
    assert get_model.call_count == 1


def test_embedder_failure_recovers_after_backoff_window(mocker, reset_embedder_cache, monkeypatch):
    monkeypatch.setattr(health, "HEALTH_EMBEDDER_RETRY_SECONDS", 30.0)
    get_model = mocker.patch("src.rag.retrieval.search._get_model", side_effect=MemoryError("OOM"))

    fake_now = 1000.0
    monkeypatch.setattr(health.time, "monotonic", lambda: fake_now)
    assert health._embedder_loadable() is False
    assert get_model.call_count == 1

    # Past the back-off window, and the underlying condition has cleared:
    # readiness must be able to recover without a process restart.
    fake_now = 1031.0
    get_model.side_effect = None
    get_model.return_value = object()
    assert health._embedder_loadable() is True
    assert get_model.call_count == 2
