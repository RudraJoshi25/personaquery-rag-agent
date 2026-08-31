# tests/test_resilience.py
"""Prove the LLM resilience layer retries transient failures then degrades
gracefully rather than raising.

Rewritten from the Groq version when the pipeline consolidated onto
Anthropic. The assertions are deliberately unchanged in spirit — retry on
transient, don't retry on permanent, open the breaker after repeated
failure, never raise — because those are properties of the pipeline, not of
any one vendor. Only the response shape and the 529 case are new.
"""
from __future__ import annotations

import json

import pytest

from src.rag import llm_anthropic


class FakeResp:
    def __init__(self, status_code, json_body=None, headers=None):
        self.status_code = status_code
        self._json = json_body or {}
        self.headers = headers or {}
        # `.text` must be the serialized body, not a placeholder: the error
        # path reads r.text to surface (and match on) the API's message, so
        # a fake that stubs it out can't exercise that path faithfully.
        self.text = json.dumps(self._json) if json_body else ""

    def json(self):
        return self._json


def _ok_body(text="hello [[cite:1]]"):
    """Anthropic returns typed content blocks, not choices[0].message."""
    return {"content": [{"type": "text", "text": text}]}


@pytest.fixture(autouse=True)
def fast_and_clean(monkeypatch):
    # No real key dependency, instant backoff, fresh breaker per test.
    monkeypatch.setattr(llm_anthropic, "ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv("ANTHROPIC_BASE_DELAY", "0")
    monkeypatch.setenv("ANTHROPIC_MAX_DELAY", "0")
    monkeypatch.setenv("ANTHROPIC_MAX_RETRIES", "4")
    llm_anthropic._breaker = None  # rebuilt lazily per test
    yield


def _patch_post(monkeypatch, seq):
    calls = {"n": 0}

    def fake_post(*a, **k):
        r = seq[min(calls["n"], len(seq) - 1)]
        calls["n"] += 1
        return r

    monkeypatch.setattr(llm_anthropic.requests, "post", fake_post)
    return calls


def test_retries_then_succeeds(monkeypatch):
    """429 twice, then 200 -> success after retries, no crash."""
    calls = _patch_post(monkeypatch, [
        FakeResp(429), FakeResp(429), FakeResp(200, _ok_body("ok [[cite:1]]")),
    ])
    res = llm_anthropic.answer_with_anthropic("q", "ctx")
    assert res.ok is True
    assert res.status == "ok"
    assert res.text == "ok [[cite:1]]"
    assert calls["n"] == 3


def test_overloaded_529_is_retried(monkeypatch):
    """529 is Anthropic-specific 'overloaded' and must be treated as transient."""
    calls = _patch_post(monkeypatch, [
        FakeResp(529), FakeResp(200, _ok_body("recovered [[cite:1]]")),
    ])
    res = llm_anthropic.answer_with_anthropic("q", "ctx")
    assert res.ok is True
    assert calls["n"] == 2


def test_rate_limited_exhausts_and_degrades(monkeypatch):
    """Persistent 429 -> clean degraded result, never an exception."""
    _patch_post(monkeypatch, [FakeResp(429)])
    res = llm_anthropic.answer_with_anthropic("q", "ctx")
    assert res.ok is False
    assert res.status == "rate_limited"
    assert res.text == llm_anthropic.DEGRADED_MESSAGE


def test_bad_request_is_not_retried(monkeypatch):
    """A 400 (bad model id / invalid key) will never succeed on retry."""
    calls = _patch_post(monkeypatch, [FakeResp(400)])
    res = llm_anthropic.answer_with_anthropic("q", "ctx")
    assert res.ok is False
    assert res.text == llm_anthropic.DEGRADED_MESSAGE
    assert calls["n"] == 1, "non-retryable status must not be retried"


def test_empty_content_is_a_failure(monkeypatch):
    """A 200 with no text block must not be served as an empty answer."""
    _patch_post(monkeypatch, [FakeResp(200, {"content": []})])
    res = llm_anthropic.answer_with_anthropic("q", "ctx")
    assert res.ok is False


def test_non_text_blocks_are_ignored(monkeypatch):
    """A thinking/tool block must never be mistaken for the answer."""
    body = {"content": [
        {"type": "thinking", "thinking": "internal reasoning"},
        {"type": "text", "text": "the real answer [[cite:2]]"},
    ]}
    _patch_post(monkeypatch, [FakeResp(200, body)])
    res = llm_anthropic.answer_with_anthropic("q", "ctx")
    assert res.text == "the real answer [[cite:2]]"


def test_temperature_omitted_by_default():
    """claude-sonnet-5 rejects `temperature` outright, so it must not be sent."""
    p = llm_anthropic._build_payload("q", "ctx", "chat")
    assert "temperature" not in p
    assert "max_tokens" in p, "max_tokens is required by the Messages API"


def test_deprecated_temperature_is_dropped_and_retried(monkeypatch):
    """If a model drops a param mid-flight, self-heal instead of 400-ing."""
    monkeypatch.setattr(llm_anthropic, "_TEMPERATURE_RAW", "0.2")
    body = {"type": "error", "error": {
        "type": "invalid_request_error",
        "message": "`temperature` is deprecated for this model."}}

    seen = []

    def fake_post(url, headers=None, json=None, timeout=None):
        seen.append(dict(json))
        if "temperature" in json:
            return FakeResp(400, body)
        return FakeResp(200, _ok_body("recovered [[cite:1]]"))

    monkeypatch.setattr(llm_anthropic.requests, "post", fake_post)
    monkeypatch.setattr(llm_anthropic.requests, "Timeout", Exception, raising=False)

    res = llm_anthropic.answer_with_anthropic("q", "ctx")
    assert res.ok is True, "should recover by dropping the deprecated param"
    assert "temperature" in seen[0], "first attempt sends it"
    assert "temperature" not in seen[1], "retry drops it"


def test_missing_key_is_misconfig(monkeypatch):
    monkeypatch.setattr(llm_anthropic, "ANTHROPIC_API_KEY", "")
    res = llm_anthropic.answer_with_anthropic("q", "ctx")
    assert res.ok is False
    assert res.status == "misconfig"


def test_circuit_opens_after_repeated_failure(monkeypatch):
    """Once open, the breaker short-circuits without touching the network."""
    monkeypatch.setenv("CB_FAIL_THRESHOLD", "1")
    monkeypatch.setenv("ANTHROPIC_MAX_RETRIES", "0")
    llm_anthropic._breaker = None

    calls = _patch_post(monkeypatch, [FakeResp(500)])
    llm_anthropic.answer_with_anthropic("q", "ctx")   # trips the breaker
    before = calls["n"]

    res = llm_anthropic.answer_with_anthropic("q", "ctx")
    assert res.status == "circuit_open"
    assert calls["n"] == before, "open circuit must not issue a request"
