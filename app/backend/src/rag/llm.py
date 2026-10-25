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
import threading
from dataclasses import dataclass
from datetime import UTC, datetime

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
# Free tier caps OUTPUT tokens at 1000/min per org (OTPM), reasoning
# included, and rejects any request whose max_tokens alone exceeds it
# ("Request too large"). 700 here + 250 for expansion keeps one full
# question under the cap.
GROQ_MAX_TOKENS = int(os.getenv("GROQ_MAX_TOKENS", "700"))
# Low temperature: grounded answers should not vary between runs.
GROQ_TEMPERATURE = float(os.getenv("GROQ_TEMPERATURE", "0.2"))

# Qwen3.8 always reasons first, and reasoning tokens count against
# max_tokens: at the default effort a 200-token expansion call spent 198 on
# reasoning and returned empty content. Groq rejects "none"; "low" is the floor.
GROQ_REASONING_EFFORT = os.getenv("GROQ_REASONING_EFFORT", "low")

# Qwen3 may emit a <think>...</think> block before the answer; never show it.
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)

# Number of times the API was actually invoked.
CALL_COUNT = 0

# Global daily budget, shared by every visitor: the Groq free tier caps the
# whole org at 1K req/day, so one abuser could otherwise starve everyone.
# ponytail: in-process, resets on restart and per replica; move to Redis if
# the service ever runs more than one instance.
DAILY_CALL_LIMIT = int(os.getenv("LLM_DAILY_CALL_LIMIT", "800"))
EXPANSION_CALL_LIMIT = int(os.getenv("LLM_EXPANSION_CALL_LIMIT", "600"))
_day = ""
_day_calls = 0
_day_lock = threading.Lock()


def calls_today() -> int:
    global _day, _day_calls
    today = datetime.now(UTC).date().isoformat()
    with _day_lock:
        if today != _day:
            _day, _day_calls = today, 0
        return _day_calls


def _take_call() -> None:
    """Reserve one call from today's budget, or raise (not retryable)."""
    global _day_calls
    calls_today()  # roll the day over if needed
    with _day_lock:
        if _day_calls >= DAILY_CALL_LIMIT:
            raise LLMError("daily quota exhausted", code="quota_exhausted")
        _day_calls += 1

_breaker: CircuitBreaker | None = None


@dataclass
class LLMResult:
    text: str
    ok: bool
    status: str          # "ok" | "rate_limited" | "timeout" | "server_error" | "circuit_open" | "quota_exhausted" | "error" | "misconfig"
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
    global CALL_COUNT, _day_calls
    _take_call()
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
        body = (r.text or "").lower()
        # "Request too large": this request can never fit the per-minute cap,
        # so retrying only burns time. A config error, not load.
        if "request too large" in body:
            log_event("ERROR", "llm_request_too_large", detail=(r.text or "")[:200])
            raise LLMError("request exceeds per-minute token cap", code="request_too_large")
        # Groq says which window was hit: "...on requests per day (RPD)" or
        # "tokens per day (TPD)". A daily cap won't clear on retry, and it is
        # not a provider fault, so treat it like our own budget running out
        # and stop calling for the rest of the UTC day.
        if "per day" in body:
            with _day_lock:
                _day_calls = max(_day_calls, DAILY_CALL_LIMIT)
            log_event("WARN", "llm_provider_daily_quota", detail=(r.text or "")[:200])
            raise LLMError("provider daily quota exhausted", code="quota_exhausted")
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
        "reasoning_effort": GROQ_REASONING_EFFORT,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(question, context, mode)},
        ],
    }


def answer(question: str, context: str, mode: str = "chat",
           trace_id: str | None = None) -> LLMResult:
    """
    Resilient grounded-answer call. Never raises and never leaks a stack
    trace or provider body to the caller — always returns a clean LLMResult.
    """
    if not GROQ_API_KEY:
        log_event("ERROR", "llm_misconfig", trace_id=trace_id, reason="missing GROQ_API_KEY")
        return LLMResult("Server misconfiguration: GROQ_API_KEY is missing.", False, "misconfig")

    if calls_today() >= DAILY_CALL_LIMIT:
        log_event("WARN", "llm_quota_exhausted", trace_id=trace_id, limit=DAILY_CALL_LIMIT)
        return LLMResult(DEGRADED_MESSAGE, False, "quota_exhausted")

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
        # Our own budget running out is not a provider failure.
        if e.code != "quota_exhausted":
            breaker.record_failure()
        log_event("ERROR", "llm_failed", trace_id=trace_id,
                  error_code=e.code, attempts=calls["n"], breaker=breaker.status())
        return LLMResult(DEGRADED_MESSAGE, False, e.code, calls["n"])


def raw_completion(prompt: str, model: str | None = None,
                   max_tokens: int = 256, temperature: float = 0.3,
                   timeout: float = 8.0) -> str:
    """
    Bare completion with no RAG system prompt — used for pipeline-internal
    work (query expansion). Deliberately NOT wrapped in retry/breaker:
    callers treat it as best-effort and degrade gracefully.
    """
    if not GROQ_API_KEY:
        raise LLMError("misconfig")
    # Above the soft limit, save what is left of the budget for answers.
    if calls_today() > EXPANSION_CALL_LIMIT:
        raise LLMError("expansion budget exhausted", code="quota_soft")
    return _post({
        "model": model or GROQ_EXPANSION_MODEL,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "reasoning_effort": GROQ_REASONING_EFFORT,
        "messages": [{"role": "user", "content": prompt}],
    }, timeout)
