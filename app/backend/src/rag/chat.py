# src/rag/chat.py
from __future__ import annotations

import logging
import re
import time
from typing import Any

from src.core.config import (
    DEFENSIVE_PII_REDACTION_ENABLED,
    INJECTION_GUARD_ENABLED,
    LOG_INCLUDE_QUESTION_PREVIEW,
    LOG_QUESTION_PREVIEW_CHARS,
    OWNER_EMAIL,
    TOP_K,
)
from src.rag.generation.llm_groq import answer_with_groq
from src.rag.retrieval.search import make_context_pack, retrieve
from src.rag.safety.guardrails import check_question

logger = logging.getLogger("personaquery.chat")

SOURCE_ID_RE = re.compile(r"\[\[cite:([0-9,\s]+)\]\]")

# Defensive last-line-of-defense scrub for third-party emails that slip
# through ingestion-time redaction (src.rag.ingestion.pii_redact) or were already
# committed before it existed. Narrow scope on purpose: emails only.
_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")


def _scrub_output(text: str) -> str:
    if not DEFENSIVE_PII_REDACTION_ENABLED or not text:
        return text

    def _sub(m: re.Match) -> str:
        return m.group(0) if m.group(0).lower() == (OWNER_EMAIL or "").lower() else "[redacted email]"

    return _EMAIL_RE.sub(_sub, text)


def _question_preview(question: str) -> str | None:
    if not LOG_INCLUDE_QUESTION_PREVIEW:
        return None
    return (question or "")[:LOG_QUESTION_PREVIEW_CHARS]


def run_rag(question: str, top_k: int = TOP_K, mode: str = "chat") -> dict[str, Any]:
    t0 = time.perf_counter()

    if INJECTION_GUARD_ENABLED:
        gr = check_question(question)
        if not gr.allowed:
            logger.warning(
                "guardrail_blocked",
                extra={"question_preview": _question_preview(question), "reason": gr.reason},
            )
            return {"answer": gr.reason or "Request blocked by guardrails.", "sources": []}
        question = gr.sanitized_question or question

    hits = retrieve(question, top_k=top_k)

    # Assign stable source ids (1..N) for answer citations
    sources_with_ids = list(zip(range(1, len(hits) + 1), hits, strict=False))
    context = make_context_pack(
        [h for _, h in sources_with_ids],
        source_ids=[sid for sid, _ in sources_with_ids],
    )

    result = answer_with_groq(question, context, mode=mode)
    answer = _scrub_output(result.text)

    # Cite-only-if-used: match [[cite:1,2]] markers
    used_ids: set[int] = set()
    for match in SOURCE_ID_RE.findall(answer or ""):
        parts = [p.strip() for p in match.split(",")]
        for p in parts:
            if p.isdigit():
                used_ids.add(int(p))

    used_hits: list[tuple[int, dict[str, Any]]] = []
    if used_ids:
        for sid, h in sources_with_ids:
            if sid in used_ids:
                used_hits.append((sid, h))

    # fallback: if model didn't cite, return top 3 sources
    sources_with_ids = used_hits if used_hits else sources_with_ids[: min(3, len(sources_with_ids))]

    sources = []
    for sid, h in sources_with_ids:
        m = h["metadata"]
        sources.append({
            "id": sid,
            "file_name": m.get("file_name", "unknown"),
            "page_label": m.get("page_label", "n/a"),
            "section": m.get("section", "Document"),
            "relevance": h.get("score", 0.0),
            "channel": m.get("channel", "hybrid"),
            "snippet": _scrub_output(h["text"][:320]),
        })

    logger.info(
        "chat_request_completed",
        extra={
            "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
            "retrieval_hit_count": len(hits),
            "usage": result.usage,
            "status": "ok" if result.error is None else result.error,
            "question_preview": _question_preview(question),
        },
    )

    return {"answer": answer, "sources": sources}
