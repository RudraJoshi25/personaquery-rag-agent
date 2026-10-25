"""
Query rewriting before the index is ever touched.

A visitor's phrasing rarely matches the document's phrasing. "What's he good
at?" shares almost no lexical overlap with a resume line reading "Skills:
Python, PyTorch, FastAPI", and dense retrieval only partly closes that gap.
Generating a few paraphrases and searching with all of them, then fusing the
result lists, recovers documents any single phrasing would have missed.

The expansion is CONDITIONAL. Short factual lookups ("what is my email")
already match well and do not justify an extra LLM round trip, so they skip
straight to retrieval. This keeps the median query inside the ~1-2s budget
while still giving hard questions the full treatment.
"""
from __future__ import annotations

import os
import re
from typing import List

from src.core.logging import log_event

# Expansion model is GROQ_EXPANSION_MODEL (see llm.py); its output never
# reaches the user, so a smaller, faster model is fine here.


def _raw_completion(prompt: str) -> str:
    from src.rag.llm import raw_completion

    return raw_completion(prompt, max_tokens=200, temperature=0.4)

N_VARIANTS = int(os.getenv("QUERY_VARIANTS", "3"))
EXPANSION_ENABLED = os.getenv("QUERY_EXPANSION_ENABLED", "true").lower() in {"1", "true", "yes"}

# Questions that are already a direct lookup of a named field.
_SIMPLE_PATTERNS = [
    r"^\s*(what|what's|whats)\s+(is|are)\s+(my|his|rudra'?s)\s+\w+",
    r"\b(email|phone|number|address|linkedin|github|contact)\b",
    r"^\s*(who|where)\s+(is|am|are)\b",
]
_SIMPLE_RE = re.compile("|".join(f"(?:{p})" for p in _SIMPLE_PATTERNS), re.IGNORECASE)

_PROMPT = """Rewrite the user's question as {n} alternative search queries that \
would retrieve the same information from a personal document collection \
(resume, transcripts, research paper, patent, project docs, LinkedIn profile).

Rules:
- Vary the vocabulary: use synonyms and the formal terms documents actually use.
- Keep every rewrite self-contained and answerable on its own.
- Do not invent facts, names, or numbers that are not in the question.
- Output ONLY the queries, one per line, no numbering, no preamble.

Question: {q}"""


def needs_expansion(question: str) -> bool:
    """False for short, direct field lookups — those retrieve fine as-is."""
    if not EXPANSION_ENABLED:
        return False
    q = (question or "").strip()
    if len(q.split()) <= 4:
        return False
    if _SIMPLE_RE.search(q):
        return False
    return True


def expand_query(question: str, trace_id: str | None = None) -> List[str]:
    """
    Returns [original, *variants]. The original is always first and always
    kept — a bad rewrite must never be able to displace what the user asked.
    """
    if not needs_expansion(question):
        return [question]

    try:
        text = _raw_completion(_PROMPT.format(n=N_VARIANTS, q=question))
    except Exception as exc:
        # Expansion is an optimisation, never a dependency.
        log_event("WARN", "query_expansion_failed", trace_id=trace_id, error=str(exc))
        return [question]

    variants: List[str] = []
    for line in (text or "").splitlines():
        s = line.strip().lstrip("-•*0123456789.) ").strip()
        if not s or len(s) < 5:
            continue
        if s.lower() == question.strip().lower():
            continue
        variants.append(s)
        if len(variants) >= N_VARIANTS:
            break

    log_event("INFO", "query_expanded", trace_id=trace_id, n_variants=len(variants))
    return [question] + variants
