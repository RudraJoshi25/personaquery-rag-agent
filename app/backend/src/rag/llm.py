# src/rag/llm.py
"""
Generation via Groq (free tier), which exposes an OpenAI-compatible
/chat/completions endpoint. Called with plain `requests`,
so no provider SDK is needed in the deploy image.

What protects availability with a single provider:
  1. faq_cache.py    — common questions answered with NO provider call
  2. retry + breaker — transient 429/5xx handled here
  3. extractive fallback in rag.py — if generation is down, the answer
     degrades to cited source text instead of an apology
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Optional

import requests

from src.core.logging import log_event
from src.rag.prompt import SYSTEM_PROMPT, build_user_prompt
from src.rag.resilience import (
    CircuitBreaker,
    LLMError,
    LLMRateLimited,
    LLMServerError,
    LLMTimeout,
    retry_with_backoff,
)

DEGRADED_MESSAGE = "The answer service is temporarily unavailable. Please try again shortly."

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_URL = os.getenv("GROQ_URL", "https://api.groq.com/openai/v1/chat/completions")
# One model for answers and query expansion: the free tier is rate-limited
# per model, and a single id keeps config minimal.
GROQ_CHAT_MODEL = os.getenv("GROQ_CHAT_MODEL", "qwen/qwen3.8-27b")
GROQ_EXPANSION_MODEL = os.getenv("GROQ_EXPANSION_MODEL", GROQ_CHAT_MODEL)
GROQ_MAX_TOKENS = int(os.getenv("GROQ_MAX_TOKENS", "1024"))
# Low temperature: grounded answers should not vary between runs.
GROQ_TEMPERATURE = float(os.getenv("GROQ_TEMPERATURE", "0.2"))

# Qwen3 may emit a <think>...</think> block before the answer; never show it.
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)

# Number of times the API was actually invoked.
CALL_COUNT = 0

_breaker: Optional[CircuitBreaker] = None


@dataclass
class LLMResult:
    text: str
    ok: bool
    status: str          # "ok" | "rate_limited" | "timeout" | "server_error" | "circuit_open" | "error" | "misconfig"
    attempts: int = 0
    provider: str = "groq"


def _get_breaker() -> CircuitBreaker:
    global _breaker
    if _breaker is None:
        _breaker = CircuitBreaker(
            fail_threshold=int(os.getenv("CB_FAIL_THRESHOLD", "5")),
            cooldown_s=float(os.getenv("CB_COOLDOWN_S", "30")),
        )
    return _breaker


def _extract_text(data: dict) -> str:
    try:
        return _THINK_RE.sub("", data["choices"][0]["message"]["content"] or "").strip()
    except (KeyError, IndexError, TypeError):
        return ""


def _post(payload: dict, timeout: float) -> str:
    global CALL_COUNT
    CALL_COUNT += 1
    try:
        r = requests.post(
            GROQ_URL,
            headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
            json=payload,
            timeout=timeout,
        )
    except requests.Timeout as e:
        raise LLMTimeout(str(e))
    except requests.RequestException as e:
        raise LLMServerError(f"transport: {type(e).__name__}")

    if r.status_code == 200:
        text = _extract_text(r.json())
        if not text:
            # A 200 with no content is a real failure, not an empty answer;
            # returning it as success would cache an empty string.
            raise LLMServerError("empty_content")
        return text

    if r.status_code == 429:
        ra = r.headers.get("retry-after")
        try:
            retry_after = float(ra) if ra else None
        except ValueError:
            retry_after = None
        raise LLMRateLimited("429 rate limited", retry_after=retry_after)

    if r.status_code in (500, 502, 503, 504):
        raise LLMServerError(f"http_{r.status_code}", code=f"http_{r.status_code}")

    # Other 4xx (bad model id, invalid token, no credit) won't fix themselves
    # on retry. Log a truncated body — it names the problem and never
    # contains the token.
    detail = (r.text or "")[:300].replace("\n", " ")
    log_event("ERROR", "llm_http_error", status=r.status_code, detail=detail)
    raise LLMError(f"http_{r.status_code}: {detail}", code=f"http_{r.status_code}")


def _build_payload(question: str, context: str, mode: str) -> dict:
    return {
        "model": GROQ_CHAT_MODEL,
        "max_tokens": GROQ_MAX_TOKENS,
        "temperature": GROQ_TEMPERATURE,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(question, context, mode)},
        ],
    }


def answer(question: str, context: str, mode: str = "chat",
           trace_id: Optional[str] = None) -> LLMResult:
    """
    Resilient grounded-answer call. Never raises and never leaks a stack
    trace or provider body to the caller — always returns a clean LLMResult.
    """
    if not GROQ_API_KEY:
        log_event("ERROR", "llm_misconfig", trace_id=trace_id, reason="missing GROQ_API_KEY")
        return LLMResult("Server misconfiguration: GROQ_API_KEY is missing.", False, "misconfig")

    breaker = _get_breaker()
    if not breaker.allow():
        log_event("WARN", "llm_circuit_open", trace_id=trace_id, breaker=breaker.status())
        return LLMResult(DEGRADED_MESSAGE, False, "circuit_open")

    payload = _build_payload(question, context, mode)
    timeout = float(os.getenv("LLM_TIMEOUT", "30"))
    calls = {"n": 0}

    def _call() -> str:
        calls["n"] += 1
        return _post(payload, timeout)

    def _on_retry(attempt: int, exc: Exception, delay: float) -> None:
        log_event("WARN", "llm_retry", trace_id=trace_id,
                  attempt=attempt + 1, error_code=getattr(exc, "code", "llm_error"),
                  delay_s=round(delay, 2))

    try:
        text = retry_with_backoff(
            _call,
            max_retries=int(os.getenv("LLM_MAX_RETRIES", "2")),
            base_delay=float(os.getenv("LLM_BASE_DELAY", "1.0")),
            max_delay=float(os.getenv("LLM_MAX_DELAY", "8")),
            jitter=True,
            on_retry=_on_retry,
        )
        breaker.record_success()
        log_event("INFO", "llm_ok", trace_id=trace_id, attempts=calls["n"])
        return LLMResult(text, True, "ok", calls["n"])
    except LLMError as e:
        breaker.record_failure()
        log_event("ERROR", "llm_failed", trace_id=trace_id,
                  error_code=e.code, attempts=calls["n"], breaker=breaker.status())
        return LLMResult(DEGRADED_MESSAGE, False, e.code, calls["n"])


def raw_completion(prompt: str, model: Optional[str] = None,
                   max_tokens: int = 256, temperature: float = 0.3,
                   timeout: float = 8.0) -> str:
    """
    Bare completion with no RAG system prompt — used for pipeline-internal
    work (query expansion). Deliberately NOT wrapped in retry/breaker:
    callers treat it as best-effort and degrade gracefully.
    """
    if not GROQ_API_KEY:
        raise LLMError("misconfig")
    return _post({
        "model": model or GROQ_EXPANSION_MODEL,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "messages": [{"role": "user", "content": prompt}],
    }, timeout)
