# tests/test_api.py
"""Phase 5: API hardening — input validation, rate limiting, no leaks."""
from __future__ import annotations

import pytest
from starlette.requests import Request
from starlette.testclient import TestClient

from src import main
from src.api import rate_limit, routes_chat
from src.main import app


@pytest.fixture
def client(monkeypatch):
    # Fast, deterministic /chat: no model load, no LLM call.
    monkeypatch.setattr(
        routes_chat, "run_rag",
        lambda q, top_k=8, mode="chat": {"answer": "stub [[cite:1]]", "sources": [], "trace_id": "t"},
    )
    # Generous fresh limiter so validation tests don't trip rate limiting.
    rate_limit._limiter = rate_limit.InProcessRateLimiter(limit=1000, window_s=60)
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


def test_valid_request_ok(client):
    r = client.post("/chat", json={"question": "What is my email?"})
    assert r.status_code == 200
    assert "answer" in r.json()


def test_empty_question_rejected(client):
    r = client.post("/chat", json={"question": "   "})
    assert r.status_code == 422
    assert "empty" in r.text.lower()


def test_oversized_question_rejected(client):
    r = client.post("/chat", json={"question": "x" * 5000})
    assert r.status_code == 422
    assert "too long" in r.text.lower()


def test_missing_field_rejected(client):
    r = client.post("/chat", json={})
    assert r.status_code == 422


def test_wrong_type_rejected(client):
    r = client.post("/chat", json={"question": {"nested": "obj"}})
    assert r.status_code == 422


def test_rate_limiting_throttles(monkeypatch):
    monkeypatch.setattr(
        routes_chat, "run_rag",
        lambda q, top_k=8, mode="chat": {"answer": "ok", "sources": [], "trace_id": "t"},
    )
    rate_limit._limiter = rate_limit.InProcessRateLimiter(limit=3, window_s=60)
    with TestClient(app, raise_server_exceptions=False) as c:
        codes = [c.post("/chat", json={"question": "hello"}).status_code for _ in range(4)]
    assert codes[:3] == [200, 200, 200]
    assert codes[3] == 429


def test_rate_limit_response_clean(monkeypatch):
    monkeypatch.setattr(
        routes_chat, "run_rag",
        lambda q, top_k=8, mode="chat": {"answer": "ok", "sources": [], "trace_id": "t"},
    )
    rate_limit._limiter = rate_limit.InProcessRateLimiter(limit=1, window_s=60)
    with TestClient(app, raise_server_exceptions=False) as c:
        c.post("/chat", json={"question": "hi"})
        r = c.post("/chat", json={"question": "hi"})
    assert r.status_code == 429
    assert "Retry-After" in r.headers
    body = r.text
    assert "Traceback" not in body


def test_induced_error_is_clean_no_leak(monkeypatch):
    """An internal exception returns a generic 500 with nothing sensitive leaked."""
    SECRET = "sk-supersecret-APIKEY-12345"

    def boom(*a, **k):
        raise RuntimeError(f"db exploded with creds {SECRET}")

    monkeypatch.setattr(routes_chat, "run_rag", boom)
    rate_limit._limiter = rate_limit.InProcessRateLimiter(limit=1000, window_s=60)
    with TestClient(app, raise_server_exceptions=False) as c:
        r = c.post("/chat", json={"question": "trigger error"})
    assert r.status_code == 500
    body = r.text
    assert SECRET not in body          # no secret leak
    assert "Traceback" not in body     # no stack trace
    assert "db exploded" not in body   # no raw exception message
    j = r.json()
    assert j["error"] == "internal_error"
    assert "trace_id" in j             # correlation id for server-side log


def test_spoofed_forwarded_for_cannot_dodge_limit(monkeypatch):
    """A client-chosen first XFF hop must not become a fresh identity."""
    monkeypatch.setattr(rate_limit, "TRUST_PROXY", True)
    monkeypatch.setattr(
        routes_chat, "run_rag",
        lambda q, top_k=8, mode="chat": {"answer": "ok", "sources": [], "trace_id": "t"},
    )
    rate_limit._limiter = rate_limit.InProcessRateLimiter(limit=1, window_s=60)
    with TestClient(app, raise_server_exceptions=False) as c:
        codes = [
            c.post("/chat", json={"question": "hi"},
                   headers={"X-Forwarded-For": f"10.0.0.{i}, 203.0.113.7"}).status_code
            for i in range(2)
        ]
    assert codes == [200, 429]


def test_rate_limiter_memory_bounded():
    lim = rate_limit.InProcessRateLimiter(limit=5, window_s=60, maxsize=100)
    for i in range(1000):
        lim.check(f"10.0.{i // 256}.{i % 256}")
    assert lim._hits.stats()["size"] == 100


def test_cors_rejects_foreign_vercel_app(client):
    r = client.options("/chat", headers={
        "Origin": "https://evil.vercel.app",
        "Access-Control-Request-Method": "POST",
    })
    assert "access-control-allow-origin" not in r.headers


def _req(xff=None, peer="10.9.9.9"):
    headers = [(b"x-forwarded-for", xff.encode())] if xff else []
    return Request({"type": "http", "headers": headers, "client": (peer, 1234)})


@pytest.fixture
def behind_proxies(monkeypatch):
    monkeypatch.setattr(rate_limit, "TRUST_PROXY", True)
    monkeypatch.setattr(rate_limit, "TRUSTED_PROXY_HOPS", 2)


def test_xff_three_entries_returns_client(behind_proxies):
    assert rate_limit.client_id(_req("198.51.100.4, 172.68.1.1, 10.0.0.2")) == "198.51.100.4"


def test_xff_spoofed_left_entry_ignored(behind_proxies):
    assert rate_limit.client_id(_req("6.6.6.6, 198.51.100.4, 172.68.1.1, 10.0.0.2")) == "198.51.100.4"


def test_xff_too_short_falls_back_to_peer(behind_proxies):
    assert rate_limit.client_id(_req("198.51.100.4, 10.0.0.2")) == "10.9.9.9"
    assert rate_limit.client_id(_req(None)) == "10.9.9.9"


def test_xff_ignored_without_trust_proxy(monkeypatch):
    monkeypatch.setattr(rate_limit, "TRUST_PROXY", False)
    assert rate_limit.client_id(_req("198.51.100.4, 172.68.1.1, 10.0.0.2")) == "10.9.9.9"


def test_startup_preloads_models(monkeypatch):
    calls = []
    monkeypatch.setattr(main, "warm_up", lambda: calls.append(1))
    monkeypatch.setenv("PRELOAD_MODELS", "true")
    with TestClient(app):
        pass
    assert calls == [1]
