"""
Per-document chunking profiles.

The original chunker used one setting for everything: 220-word windows.
That is right for a research paper, where an idea spans paragraphs, and badly
wrong for a resume, where a 220-word window swallows "Education", "Experience"
and half the skills list into a single chunk. The result was an index where
one 20-page paper produced 69 of 84 chunks (82%) while the resume — the
document that answers most visitor questions — produced 8.

Fine-grained profiles fix the imbalance at its source, which is why the
IDENTITY_BOOST regex hack in retrieve_custom.py can then be deleted: the
resume wins on merit instead of on a thumb pressed to the scale.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional
import re


@dataclass(frozen=True)
class ChunkProfile:
    """How to split one class of document."""
    name: str
    max_words: int
    overlap_words: int
    # Split on blank lines / bullets before falling back to word windows.
    prefer_atomic_lines: bool = False
    # Minimum words for a chunk to be kept (drops page-number noise).
    min_words: int = 8


# Dense, fact-per-line documents: every bullet is independently answerable.
RESUME = ChunkProfile(
    name="resume",
    max_words=90,
    overlap_words=20,
    prefer_atomic_lines=True,
    min_words=5,
)

# Transcripts are tabular: course rows should not bleed into each other.
TRANSCRIPT = ChunkProfile(
    name="transcript",
    max_words=110,
    overlap_words=15,
    prefer_atomic_lines=True,
    min_words=5,
)

# Hand-written project docs: markdown, section-structured, medium density.
PROJECT_DOC = ChunkProfile(
    name="project_doc",
    max_words=160,
    overlap_words=30,
    prefer_atomic_lines=False,
)

# LinkedIn posts / short-form: each post is its own unit.
SHORTFORM = ChunkProfile(
    name="shortform",
    max_words=140,
    overlap_words=20,
    prefer_atomic_lines=False,
    min_words=5,
)

# Papers and patents: argument spans paragraphs, keep windows wide.
PAPER = ChunkProfile(
    name="paper",
    max_words=220,
    overlap_words=40,
    prefer_atomic_lines=False,
)

DEFAULT = PAPER


# Filename-pattern -> profile. First match wins, so order matters.
_RULES = [
    (re.compile(r"resume|cv\b", re.I), RESUME),
    (re.compile(r"transcript|academic|grade|marksheet", re.I), TRANSCRIPT),
    (re.compile(r"linkedin|post|profile", re.I), SHORTFORM),
    (re.compile(r"patent", re.I), PAPER),
    (re.compile(r"paper|draft|ieee|journal|thesis", re.I), PAPER),
    (re.compile(r"\.md$|project|readme", re.I), PROJECT_DOC),
]


def profile_for(file_name: str, override: Optional[str] = None) -> ChunkProfile:
    """Pick a chunk profile from the filename, or an explicit override."""
    if override:
        got = BY_NAME.get(override.lower())
        if got:
            return got

    for pattern, prof in _RULES:
        if pattern.search(file_name or ""):
            return prof
    return DEFAULT


BY_NAME: Dict[str, ChunkProfile] = {
    p.name: p for p in (RESUME, TRANSCRIPT, PROJECT_DOC, SHORTFORM, PAPER)
}


# ---------------------------------------------------------------------------
# Index balance
# ---------------------------------------------------------------------------

# No single source may exceed this share of the index. A long paper still gets
# indexed fully, but retrieval-time diversity (see retrieve) stops it from
# crowding out everything else.
MAX_SOURCE_SHARE = 0.45

# At most this many chunks from one file may appear in a single result set,
# unless there simply aren't enough other candidates. This is the principled
# replacement for IDENTITY_BOOST: instead of boosting one favoured document,
# we cap any document's ability to monopolise the context window.
MAX_CHUNKS_PER_SOURCE_IN_RESULTS = 4
