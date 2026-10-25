# tests/test_resilience.py
"""Prove the LLM resilience layer retries transient failures then degrades
gracefully rather than raising: retry on transient, don't retry on
permanent, open the breaker after repeated failure, never raise."""
from __future__ import annotations

import json

import pytest

from src.rag import llm


class FakeResp:
    def __init__(self, status_code, json_body=None, headers=None):
        self.status_code = status_code
        self._json = json_body or {}
        self.headers = headers or {}
        self.text = json.dumps(self._json) if json_body else ""

    def json(self):
        return self._json


def _ok_body(text="hello [[cite:1]]"):
    """OpenAI-compatible chat completion shape."""
    return {"choices": [{"message": {"role": "assistant", "content": text}}]}


@pytest.fixture(autouse=True)
def fast_and_clean(monkeypatch):
    # No real token dependency, instant backoff, fresh breaker per test.
    monkeypatch.setattr(llm, "GROQ_API_KEY", "test-token")
    monkeypatch.setenv("LLM_BASE_DELAY", "0")
    monkeypatch.setenv("LLM_MAX_DELAY", "0")
    monkeypatch.setenv("LLM_MAX_RETRIES", "4")
    llm._breaker = None  # rebuilt lazily per test
    yield


def _patch_post(monkeypatch, seq):
    calls = {"n": 0}

    def fake_post(*a, **k):
        r = seq[min(calls["n"], len(seq) - 1)]
        calls["n"] += 1
        return r

    monkeypatch.setattr(llm.requests, "post", fake_post)
    return calls


def test_retries_then_succeeds(monkeypatch):
    """429 twice, then 200 -> success after retries, no crash."""
    calls = _patch_post(monkeypatch, [
        FakeResp(429), FakeResp(429), FakeResp(200, _ok_body("ok [[cite:1]]")),
    ])
    res = llm.answer("q", "ctx")
    assert res.ok is True
    assert res.status == "ok"
    assert res.text == "ok [[cite:1]]"
    assert calls["n"] == 3


def test_server_error_is_retried(monkeypatch):
    calls = _patch_post(monkeypatch, [
        FakeResp(503), FakeResp(200, _ok_body("recovered [[cite:1]]")),
    ])
    res = llm.answer("q", "ctx")
    assert res.ok is True
    assert calls["n"] == 2


def test_rate_limited_exhausts_and_degrades(monkeypatch):
    """Persistent 429 -> clean degraded result, never an exception."""
    _patch_post(monkeypatch, [FakeResp(429)])
    res = llm.answer("q", "ctx")
    assert res.ok is False
    assert res.status == "rate_limited"
    assert res.text == llm.DEGRADED_MESSAGE


def test_bad_request_is_not_retried(monkeypatch):
    """A 400 (bad model id / invalid token) will never succeed on retry."""
    calls = _patch_post(monkeypatch, [FakeResp(400)])
    res = llm.answer("q", "ctx")
    assert res.ok is False
    assert res.text == llm.DEGRADED_MESSAGE
    assert calls["n"] == 1, "non-retryable status must not be retried"


def test_empty_content_is_a_failure(monkeypatch):
    """A 200 with no content must not be served as an empty answer."""
    _patch_post(monkeypatch, [FakeResp(200, {"choices": []})])
    res = llm.answer("q", "ctx")
    assert res.ok is False


def test_payload_shape():
    """System prompt goes in messages; bearer auth; model from config."""
    p = llm._build_payload("q", "ctx", "chat")
    assert p["model"] == llm.GROQ_CHAT_MODEL
    assert [m["role"] for m in p["messages"]] == ["system", "user"]


def test_missing_token_is_misconfig(monkeypatch):
    monkeypatch.setattr(llm, "GROQ_API_KEY", "")
    res = llm.answer("q", "ctx")
    assert res.ok is False
    assert res.status == "misconfig"


def test_circuit_opens_after_repeated_failure(monkeypatch):
    """Once open, the breaker short-circuits without touching the network."""
    monkeypatch.setenv("CB_FAIL_THRESHOLD", "1")
    monkeypatch.setenv("LLM_MAX_RETRIES", "0")
    llm._breaker = None

    calls = _patch_post(monkeypatch, [FakeResp(500)])
    llm.answer("q", "ctx")   # trips the breaker
    before = calls["n"]

    res = llm.answer("q", "ctx")
    assert res.status == "circuit_open"
    assert calls["n"] == before, "open circuit must not issue a request"


def test_daily_quota_hard_stop(monkeypatch):
    """At the daily limit: no provider call, clean degraded result, breaker untouched."""
    calls = _patch_post(monkeypatch, [FakeResp(200, _ok_body())])
    monkeypatch.setattr(llm, "_day", llm.datetime.now(llm.UTC).date().isoformat())
    monkeypatch.setattr(llm, "_day_calls", llm.DAILY_CALL_LIMIT)
    res = llm.answer("q", "ctx")
    assert res.ok is False and res.status == "quota_exhausted"
    assert calls["n"] == 0
    assert llm._get_breaker().allow()


def test_daily_quota_resets_at_utc_midnight(monkeypatch):
    _patch_post(monkeypatch, [FakeResp(200, _ok_body())])
    monkeypatch.setattr(llm, "_day", "2000-01-01")
    monkeypatch.setattr(llm, "_day_calls", llm.DAILY_CALL_LIMIT)
    assert llm.answer("q", "ctx").ok is True
    assert llm.calls_today() == 1


def test_expansion_skipped_above_soft_limit(monkeypatch):
    from src.rag import query_expansion
    calls = _patch_post(monkeypatch, [FakeResp(200, _ok_body("a\nb\nc"))])
    monkeypatch.setattr(llm, "_day", llm.datetime.now(llm.UTC).date().isoformat())
    monkeypatch.setattr(llm, "_day_calls", llm.EXPANSION_CALL_LIMIT + 1)
    q = "which machine learning projects has rudra shipped to production"
    assert query_expansion.expand_query(q) == [q]
    assert calls["n"] == 0


def test_groq_daily_429_is_quota_not_breaker_failure(monkeypatch):
    """Groq's per-day 429: no retry, no breaker strike, rest of day short-circuits."""
    body = {"error": {"message": "Rate limit reached for model `qwen/qwen3.8-27b` on "
                      "requests per day (RPD): Limit 1000, Used 1000", "code": "rate_limit_exceeded"}}
    calls = _patch_post(monkeypatch, [FakeResp(429, body)])
    monkeypatch.setattr(llm, "_day", llm.datetime.now(llm.UTC).date().isoformat())
    monkeypatch.setattr(llm, "_day_calls", 0)
    res = llm.answer("q", "ctx")
    assert res.ok is False and res.status == "quota_exhausted"
    assert calls["n"] == 1                      # not retried
    assert llm._get_breaker().status()["fails"] == 0
    assert llm.answer("q", "ctx").status == "quota_exhausted"
    assert calls["n"] == 1                      # second call never reached Groq


def test_groq_request_too_large_not_retried(monkeypatch):
    """OTPM 'Request too large' can never succeed: fail once, no retries."""
    body = {"error": {"message": "Request too large for model `qwen/qwen3.8-27b` on output "
                      "tokens per minute (OTPM): Limit 1000, Requested 1024."}}
    calls = _patch_post(monkeypatch, [FakeResp(429, body)])
    res = llm.answer("q", "ctx")
    assert res.ok is False and res.status == "request_too_large"
    assert calls["n"] == 1


def test_default_budget_fits_free_tier_otpm():
    from src.rag.query_expansion import EXPANSION_MAX_TOKENS
    assert llm.GROQ_MAX_TOKENS + EXPANSION_MAX_TOKENS <= 1000
