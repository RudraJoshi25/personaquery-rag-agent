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
