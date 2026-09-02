# src/rag/safety/guardrails.py
#
# This is a cheap, easily-bypassed first filter (paraphrase, translation, or
# splitting a phrase across turns all defeat it trivially). The real security
# boundary is the system prompt's grounding rules in
# src/rag/generation/llm_groq.py plus the
# fact that the LLM only ever sees retrieved document context, not arbitrary
# tool/data access. Don't invest further engineering here (e.g. an ML
# classifier) - diminishing returns for a public single-purpose Q&A bot with
# no sensitive backend actions to protect.
from __future__ import annotations

import logging
import re
import unicodedata
from dataclasses import dataclass

logger = logging.getLogger("personaquery.guardrails")

_ZERO_WIDTH_RE = re.compile("[​‌‍﻿]")

INJECTION_PATTERNS = [
    r"ignore (?:(?:all|any|previous|prior)\s+)+instructions",
    r"disregard (?:(?:all|any|previous|prior)\s+)+instructions",
    r"reveal (the )?(system prompt|developer message|hidden instructions)",
    r"show (me )?(your|the) system prompt",
    r"you are now (dan|developer|system)",
    r"fabricate (sources|citations|references)",
    r"make up (sources|citations|references)",
    r"bypass (safety|policy|guardrails)",
    r"print (the )?prompt",
    r"leak(?:s|ed|ing)? (the )?(system prompt|instructions|api key|credentials|secrets?)",
    r"act as .*(no restrictions|jailbreak|dan mode)",
    r"repeat (the )?(words|text) above",
    r"output (the )?(text|prompt) above",
    r"what (were you|are your) (told|instructions)",
    r"translate (the )?system prompt",
]

INJ_RE = re.compile("|".join(f"(?:{p})" for p in INJECTION_PATTERNS), re.IGNORECASE)


@dataclass
class GuardrailResult:
    allowed: bool
    reason: str | None = None
    sanitized_question: str | None = None


def check_question(q: str) -> GuardrailResult:
    q2 = (q or "").strip()
    if not q2:
        return GuardrailResult(False, "Empty question.")

    # Normalize Unicode (fullwidth/homoglyph variants) and strip zero-width
    # characters before matching, closing a trivial regex-evasion trick.
    normalized = unicodedata.normalize("NFKC", q2)
    normalized = _ZERO_WIDTH_RE.sub("", normalized)

    if INJ_RE.search(normalized):
        logger.warning("guardrail_rejected", extra={"question_preview": normalized[:120]})
        return GuardrailResult(
            allowed=False,
            reason="Prompt injection detected. Please ask a normal question about the documents.",
        )

    # light sanitization: remove excessive control tokens
    q2 = q2.replace("\0", "").strip()
    return GuardrailResult(True, sanitized_question=q2)
