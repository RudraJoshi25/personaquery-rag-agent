# src/rag/llm_anthropic.py
"""
Anthropic (Claude) — the sole generation provider.

HISTORY (why this file looks the way it does)
---------------------------------------------
This pipeline previously ran a four-provider failover chain (Cerebras ->
Groq -> NVIDIA), which existed for exactly one reason: free-tier token
caps. Groq's 8K tokens/minute could not sustain a grounded answer costing
3-4K tokens, so a couple of questions in quick succession would 429. The
chain was scaffolding around a quota problem.

With prepaid Anthropic credits that problem is gone, so the scaffolding
went with it: llm_groq / llm_cerebras / llm_nvidia / llm_base / llm_chain
were deleted. One provider, one code path, far less to reason about.

WHAT STILL PROTECTS AVAILABILITY
--------------------------------
Dropping to a single provider removes failover, so the remaining layers
matter more, not less:
  1. faq_cache.py    — common questions answered with NO provider call
  2. retry + breaker — transient 429/500/529 handled here
  3. extractive fallback in rag.py — if generation is fully down, retrieval
     still works, so the answer degrades to cited source text instead of
     an apology

API SHAPE
---------
Anthropic's Messages API is not OpenAI-compatible:
  - auth is `x-api-key`, not `Authorization: Bearer`
  - `anthropic-version` header is required
  - the system prompt is a top-level `system` field, not a role in messages
  - `max_tokens` is required
  - the response is a list of typed content blocks, not `choices[0].message`
"""
from __future__ import annotations

import os
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

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
ANTHROPIC_URL = os.getenv("ANTHROPIC_URL", "https://api.anthropic.com/v1/messages")
ANTHROPIC_VERSION = os.getenv("ANTHROPIC_VERSION", "2023-06-01")

# Sonnet by default: this pipeline leans hard on instruction-following (the
# citation contract, the identity rules, the ASCII rule), and Sonnet handles
# that reliably without Opus pricing on every visitor question. Set
# ANTHROPIC_CHAT_MODEL=claude-opus-5 if answer quality matters more than cost.
ANTHROPIC_CHAT_MODEL = os.getenv("ANTHROPIC_CHAT_MODEL", "claude-sonnet-5")

# Query expansion is a throwaway paraphrase of a short string whose output
# never reaches the user — the cheapest fast model is the right tool.
ANTHROPIC_EXPANSION_MODEL = os.getenv(
    "ANTHROPIC_EXPANSION_MODEL", "claude-haiku-4-5-20251001"
)

ANTHROPIC_MAX_TOKENS = int(os.getenv("ANTHROPIC_MAX_TOKENS", "1024"))

# TEMPERATURE IS OMITTED BY DEFAULT.
#
# claude-sonnet-5 rejects it outright:
#   400 invalid_request_error: "`temperature` is deprecated for this model."
#
# Note "deprecated", not "must be 1" — the field is gone, not constrained,
# so the fix is to not send it rather than to send a different value.
# Verified: the same question returns the identical answer with the field
# omitted and with temperature=1, because sampling is now managed by the
# model rather than by a caller-supplied knob.
#
# Losing an explicit low temperature does NOT loosen grounding. In this
# pipeline factual accuracy comes from what the model is given and what it
# is forbidden to do — retrieved-context-only, the citation contract, and
# the "Not stated in the documents" rule — none of which are sampling
# parameters. What temperature affected was phrasing variation between
# runs, and the FAQ cache pins the common answers to one generated result
# anyway (see faq_cache.py), so repeat visitors get a stable answer.
#
# Set ANTHROPIC_TEMPERATURE explicitly only when pointing at an older model
# that still accepts it; empty/unset means "don't send the field".
_TEMPERATURE_RAW = os.getenv("ANTHROPIC_TEMPERATURE", "").strip()

# Number of times the API was actually invoked (used by cache tests to prove
# a cache hit did not reach the network).
CALL_COUNT = 0

_breaker: Optional[CircuitBreaker] = None


@dataclass
class LLMResult:
    text: str
    ok: bool
    status: str          # "ok" | "rate_limited" | "timeout" | "server_error" | "circuit_open" | "error" | "misconfig"
    attempts: int = 0
    provider: str = "anthropic"


def _get_breaker() -> CircuitBreaker:
    global _breaker
    if _breaker is None:
        _breaker = CircuitBreaker(
            fail_threshold=int(os.getenv("CB_FAIL_THRESHOLD", "5")),
            cooldown_s=float(os.getenv("CB_COOLDOWN_S", "30")),
        )
    return _breaker


def _headers() -> dict:
    return {
        "x-api-key": ANTHROPIC_API_KEY,
        "anthropic-version": ANTHROPIC_VERSION,
        "content-type": "application/json",
    }


def _extract_text(data: dict) -> str:
    """Join the text blocks, ignoring any non-text block types.

    Filtering by type (rather than indexing [0]) means a thinking or tool
    block appearing in future can't silently become the 'answer'.
    """
    blocks = data.get("content") or []
    return "".join(
        b.get("text", "") for b in blocks if isinstance(b, dict) and b.get("type") == "text"
    ).strip()


def _post(payload: dict, timeout: float) -> str:
    global CALL_COUNT
    CALL_COUNT += 1
    try:
        r = requests.post(ANTHROPIC_URL, headers=_headers(), json=payload, timeout=timeout)
    except requests.Timeout as e:
        raise LLMTimeout(str(e))
    except requests.RequestException as e:
        raise LLMServerError(f"transport: {type(e).__name__}")

    if r.status_code == 200:
        text = _extract_text(r.json())
        if not text:
            # A 200 with no text block is a real failure, not an empty answer;
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

    # 529 is Anthropic-specific ("overloaded") and IS retryable — it means the
    # API is busy, not that the request is malformed.
    if r.status_code in (500, 502, 503, 504, 529):
        raise LLMServerError(f"http_{r.status_code}", code=f"http_{r.status_code}")

    # Other 4xx (bad model id, invalid key, exhausted credit, malformed
    # request) won't fix themselves on retry. Anthropic returns a structured
    # {"error": {"type", "message"}} body here that names the offending
    # field — log a truncated form of it, because "http_400" alone is not
    # actionable and the alternative is guessing at the payload. The body
    # never contains the API key.
    detail = (r.text or "")[:300].replace("\n", " ")
    log_event("ERROR", "anthropic_http_error", status=r.status_code, detail=detail)
    raise LLMError(f"http_{r.status_code}: {detail}", code=f"http_{r.status_code}")


def _build_payload(question: str, context: str, mode: str) -> dict:
    payload = {
        "model": ANTHROPIC_CHAT_MODEL,
        "max_tokens": ANTHROPIC_MAX_TOKENS,      # required by this API
        "system": SYSTEM_PROMPT,
        "messages": [{"role": "user", "content": build_user_prompt(question, context, mode)}],
    }
    if _TEMPERATURE_RAW:                          # see the note above
        payload["temperature"] = float(_TEMPERATURE_RAW)
    return payload


def answer_with_anthropic(question: str, context: str, mode: str = "chat",
                          trace_id: Optional[str] = None) -> LLMResult:
    """
    Resilient grounded-answer call. Never raises and never leaks a stack
    trace or provider body to the caller — always returns a clean LLMResult.
    """
    if not ANTHROPIC_API_KEY:
        log_event("ERROR", "llm_misconfig", trace_id=trace_id,
                  reason="missing ANTHROPIC_API_KEY")
        return LLMResult("Server misconfiguration: ANTHROPIC_API_KEY is missing.",
                         False, "misconfig")

    breaker = _get_breaker()
    if not breaker.allow():
        log_event("WARN", "llm_circuit_open", trace_id=trace_id, breaker=breaker.status())
        return LLMResult(DEGRADED_MESSAGE, False, "circuit_open")

    payload = _build_payload(question, context, mode)
    timeout = float(os.getenv("ANTHROPIC_TIMEOUT", "30"))
    calls = {"n": 0}

    def _call() -> str:
        calls["n"] += 1
        try:
            return _post(payload, timeout)
        except LLMError as e:
            # Self-heal a param the target model has dropped. This exact
            # class of break has now happened twice in this project (a Groq
            # model deprecation, then this), and the failure mode is bad:
            # a hard 400 on every request with a generic error. If the API
            # tells us a field is deprecated, drop it and retry once rather
            # than failing the visitor's question over a config detail.
            msg = str(e).lower()
            if "temperature" in msg and "deprecat" in msg and "temperature" in payload:
                payload.pop("temperature")
                log_event("WARN", "anthropic_dropped_deprecated_param",
                          trace_id=trace_id, param="temperature",
                          model=payload.get("model"))
                return _post(payload, timeout)
            raise

    def _on_retry(attempt: int, exc: Exception, delay: float) -> None:
        log_event("WARN", "llm_retry", trace_id=trace_id,
                  attempt=attempt + 1, error_code=getattr(exc, "code", "llm_error"),
                  delay_s=round(delay, 2))

    try:
        text = retry_with_backoff(
            _call,
            max_retries=int(os.getenv("ANTHROPIC_MAX_RETRIES", "2")),
            base_delay=float(os.getenv("ANTHROPIC_BASE_DELAY", "1.0")),
            max_delay=float(os.getenv("ANTHROPIC_MAX_DELAY", "8")),
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


def answer_text(question: str, context: str, mode: str = "chat") -> str:
    """Convenience wrapper returning just the answer text."""
    return answer_with_anthropic(question, context, mode=mode).text


def raw_completion(prompt: str, model: Optional[str] = None,
                   max_tokens: int = 256, temperature: float = 0.3,
                   timeout: float = 8.0) -> str:
    """
    Bare completion with no RAG system prompt and no citation contract —
    used for pipeline-internal work (query expansion), where the grounded
    answer scaffolding would only get in the way.

    Deliberately NOT wrapped in the retry/breaker layer: callers treat it as
    best-effort and degrade gracefully, so a retry loop would cost more
    latency than the feature saves.
    """
    if not ANTHROPIC_API_KEY:
        raise LLMError("misconfig")
    payload = {
        "model": model or ANTHROPIC_EXPANSION_MODEL,
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": prompt}],
    }
    # Same temperature rule as the main call. This path matters because
    # query_expansion swallows exceptions and silently falls back to a
    # single query — a rejected `temperature` here would have quietly
    # disabled multi-query retrieval with only a WARN in the logs.
    if _TEMPERATURE_RAW:
        payload["temperature"] = float(_TEMPERATURE_RAW)
    return _post(payload, timeout)
