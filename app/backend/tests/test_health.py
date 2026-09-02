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
