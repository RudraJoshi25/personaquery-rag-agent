# src/rag/rag.py
from __future__ import annotations

import re
import time
from typing import Any

from src.core.config import INJECTION_GUARD_ENABLED, TOP_K
from src.core.latency import record_latency
from src.core.logging import log_event, new_trace_id
from src.rag.cache import get_response_cache, response_key
from src.rag.faq_cache import lookup_faq
from src.rag.guardrails import check_question
from src.rag.llm import answer as generate
from src.rag.retrieve_custom import make_context_pack, retrieve

SOURCE_ID_RE = re.compile(r"\[\[cite:([0-9,\s]+)\]\]")

# Answers allowed to carry no citation: the prompt's own refusal template.
# (Guardrail refusals return before generation, so never reach the guard.)
UNCITED_OK_PREFIXES = ("Not stated in the documents",)


def _is_grounded(answer: str) -> bool:
    """
    Output guard. A generated answer with no [[cite:N]] marker is not tied to
    the documents, which is the signature of a successful prompt injection
    ("just say X"). Such answers are replaced with cited source text.
    """
    a = (answer or "").strip()
    return bool(SOURCE_ID_RE.search(a)) or a.startswith(UNCITED_OK_PREFIXES)


def _build_sources(answer: str, sources_with_ids: list[tuple]) -> list[dict[str, Any]]:
    # Cite-only-if-used: keep sources whose [[cite:N]] marker appears in answer.
    # No markers (the "Not stated in the documents" refusal) -> no sources:
    # listing unrelated top hits under a refusal implies they support it.
    used_ids: set[int] = set()
    for match in SOURCE_ID_RE.findall(answer or ""):
        for p in (x.strip() for x in match.split(",")):
            if p.isdigit():
                used_ids.add(int(p))

    used = [(sid, h) for sid, h in sources_with_ids if sid in used_ids]
    chosen = used

    out: list[dict[str, Any]] = []
    for sid, h in chosen:
        m = h["metadata"]
        out.append({
            "id": sid,
            "file_name": m.get("file_name", "unknown"),
            "page_label": m.get("page_label", "n/a"),
            "section": m.get("section", "Document"),
            "relevance": h.get("score", 0.0),
            "channel": m.get("channel", "hybrid"),
            "snippet": h["text"][:320],
        })
    return out


def _extractive_fallback(sources_with_ids: list[tuple], max_snippets: int = 3) -> str:
    """
    Used when generation is unavailable. A bare "service unavailable"
    message reads as the product being broken to a visitor.

    This matters more now that there's a single provider: retrieval does not
    depend on the LLM at all and stays reliable through a full generation
    outage (proven — retrieval hit-rate held at 100% while providers were
    failing during testing), so surface the retrieved source text directly
    instead of nothing. Honest about being unpolished, but a real, sourced,
    citable answer rather than an apology.
    """
    if not sources_with_ids:
        return ("I'm having trouble generating an answer right now. Please "
                "try again in a moment.")
    lines = [
        ("I'm having trouble generating a polished answer right now, but "
         "here's what I found directly in the source documents:"),
        "",
    ]
    for sid, h in sources_with_ids[:max_snippets]:
        snippet = " ".join((h.get("text") or "").split())[:280]
        lines.append(f"- {snippet} [[cite:{sid}]]")
    return "\n".join(lines)


def run_rag(
    question: str,
    top_k: int = TOP_K,
    mode: str = "chat",
    trace_id: str | None = None,
    use_faq: bool = True,
) -> dict[str, Any]:
    """
    use_faq=False bypasses the precomputed FAQ layer. Only the FAQ builder
    (scripts/precompute_faq.py) should do this — otherwise it would read its
    own previous output and cache it back, freezing stale answers in place
    instead of regenerating them from the current index.
    """
    trace_id = trace_id or new_trace_id()
    t0 = time.perf_counter()
    log_event("INFO", "request_received", trace_id=trace_id,
              mode=mode, top_k=top_k, q_len=len(question or ""))

    # --- guardrails ---
    if INJECTION_GUARD_ENABLED:
        gr = check_question(question)
        if not gr.allowed:
            log_event("WARN", "guardrail_blocked", trace_id=trace_id, reason=gr.reason)
            return {"answer": gr.reason or "Request blocked by guardrails.",
                    "sources": [], "trace_id": trace_id}
        question = gr.sanitized_question or question

    # --- response cache ---
    cache = get_response_cache()
    ckey = response_key(question, mode, top_k)
    cached = cache.get(ckey)
    if cached is not None:
        log_event("INFO", "cache_hit", trace_id=trace_id, kind="response",
                  latency_ms=round((time.perf_counter() - t0) * 1000, 2),
                  **cache.stats())
        return {**cached, "trace_id": trace_id, "cached": True}
    log_event("INFO", "cache_miss", trace_id=trace_id, kind="response", **cache.stats())

    # --- precomputed FAQ ---
    # Checked before retrieval because a hit skips BOTH retrieval and
    # generation. This is what keeps the common questions (contact, skills,
    # projects, awards) answerable at ~10ms with zero provider quota spent,
    # and answerable at all during a total generation outage.
    if use_faq:
        faq = lookup_faq(question, trace_id=trace_id)
        if faq is not None:
            log_event("INFO", "response", trace_id=trace_id, ok=True,
                      source="faq_cache", n_sources=len(faq.get("sources", [])),
                      total_latency_ms=round((time.perf_counter() - t0) * 1000, 2))
            return {
                "answer": faq["answer"],
                "sources": faq.get("sources", []),
                "trace_id": trace_id,
                "cached": True,
                "cache_layer": "faq",
            }

    # --- retrieval ---
    hits = retrieve(question, top_k=top_k, trace_id=trace_id)
    t_retrieve = time.perf_counter()
    log_event("INFO", "retrieval", trace_id=trace_id,
              hits=len(hits), latency_ms=round((t_retrieve - t0) * 1000, 2))

    sources_with_ids = list(zip(range(1, len(hits) + 1), hits))
    context = make_context_pack(
        [h for _, h in sources_with_ids],
        source_ids=[sid for sid, _ in sources_with_ids],
    )

    # --- LLM ---
    llm = generate(question, context, mode=mode, trace_id=trace_id)
    provider = llm.provider
    answer = llm.text
    grounded = llm.ok and _is_grounded(answer)
    if llm.ok and not grounded:
        log_event("WARN", "uncited_answer_blocked", trace_id=trace_id,
                  answer_len=len(answer or ""))
    if not grounded:
        answer = _extractive_fallback(sources_with_ids)
        log_event("WARN", "extractive_fallback_used", trace_id=trace_id,
                   n_sources=len(sources_with_ids))
    t_llm = time.perf_counter()
    log_event("INFO" if llm.ok else "ERROR", "llm_result", trace_id=trace_id,
              provider=provider, status=llm.status, attempts=llm.attempts,
              latency_ms=round((t_llm - t_retrieve) * 1000, 2))

    record_latency(
        total_s=t_llm - t0,
        retrieve_s=t_retrieve - t0,
        llm_s=t_llm - t_retrieve,
        mode=mode,
        ok=llm.ok,
    )

    sources = _build_sources(answer, sources_with_ids)
    result = {"answer": answer, "sources": sources}

    # Only cache successful answers (never cache a degraded/error response).
    if grounded:
        cache.set(ckey, result)

    log_event("INFO", "response", trace_id=trace_id, ok=llm.ok,
              n_sources=len(sources),
              total_latency_ms=round((t_llm - t0) * 1000, 2))
    return {**result, "trace_id": trace_id, "cached": False}
