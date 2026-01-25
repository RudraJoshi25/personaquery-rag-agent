# src/rag/guardrails.py
from __future__ import annotations
import re
from dataclasses import dataclass
from typing import Optional

INJECTION_PATTERNS = [
    r"ignore (all|any|previous|prior) instructions",
    r"disregard (all|any|previous|prior) instructions",
    r"reveal (the )?(system prompt|developer message|hidden instructions)",
    r"show (me )?(your|the) system prompt",
    r"you are now (dan|developer|system)",
    r"fabricate (sources|citations|references)",
    r"make up (sources|citations|references)",
    r"bypass (safety|policy|guardrails)",
    r"print (the )?prompt",
    r"leak",
]

INJ_RE = re.compile("|".join(f"(?:{p})" for p in INJECTION_PATTERNS), re.IGNORECASE)

@dataclass
class GuardrailResult:
    allowed: bool
    reason: Optional[str] = None
    sanitized_question: Optional[str] = None

def check_question(q: str) -> GuardrailResult:
    q2 = (q or "").strip()
    if not q2:
        return GuardrailResult(False, "Empty question.")

    if INJ_RE.search(q2):
        return GuardrailResult(
            allowed=False,
            reason="Prompt injection detected. Please ask a normal question about the documents.",
        )

    # light sanitization: remove excessive control tokens
    q2 = q2.replace("\0", "").strip()
    return GuardrailResult(True, sanitized_question=q2)
