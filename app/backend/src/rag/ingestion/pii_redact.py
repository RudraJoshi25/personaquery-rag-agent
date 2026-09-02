# src/rag/pii_redact.py
from __future__ import annotations

import re

from src.core.config import OWNER_EMAIL, OWNER_PHONE

EMAIL_RE = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")

# Conservative phone matcher: requires either a leading "+CC" international
# prefix or a "(area) xxx-xxxx" shape, with grouped-digit separators. This
# deliberately does NOT try to catch every bare local-format number, because
# the corpus includes patent/paper numeric content (DOIs, patent numbers,
# dates, equation/figure numbers) that a loose digit-grouping regex would
# false-positive on heavily. Manual spot-check after re-ingestion is still
# required (see docs/README instructions).
PHONE_RE = re.compile(
    r"(?<!\w)(?:\+\d{1,3}[\s\-.]?)(?:\(?\d{2,4}\)?[\s\-.]?){2,5}\d{2,4}(?!\w)"
    r"|(?<!\w)\(\d{2,4}\)[\s\-.]?\d{3,4}[\s\-.]?\d{3,4}(?!\w)"
)


def redact_text(text: str, owner_email: str = OWNER_EMAIL, owner_phone: str = OWNER_PHONE) -> str:
    """Redact third-party emails/phone numbers from raw ingested text,
    preserving the owner's own contact info verbatim. Call once per
    page/section during ingestion, BEFORE chunking, so redaction never
    straddles a chunk boundary."""
    if not text:
        return text

    def _email_sub(m: re.Match) -> str:
        return m.group(0) if m.group(0).lower() == (owner_email or "").lower() else "[REDACTED_EMAIL]"

    text = EMAIL_RE.sub(_email_sub, text)

    owner_digits = re.sub(r"\D", "", owner_phone or "")

    def _phone_sub(m: re.Match) -> str:
        raw = m.group(0)
        digits = re.sub(r"\D", "", raw)
        if owner_digits and len(owner_digits) >= 9 and digits.endswith(owner_digits[-9:]):
            return raw  # owner's own number, keep verbatim
        if 8 <= len(digits) <= 15:
            return "[REDACTED_PHONE]"
        return raw

    text = PHONE_RE.sub(_phone_sub, text)
    return text
