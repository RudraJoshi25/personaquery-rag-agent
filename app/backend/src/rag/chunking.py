# src/rag/chunking.py
"""
Section-aware chunking with per-document-type profiles.

See doc_profiles.py for why the profiles exist. The short version: one global
window size produced an index that was 82% research paper, so identity and
skills questions retrieved the wrong document and needed a regex boost to
compensate.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple
import re

from src.rag.doc_profiles import ChunkProfile, profile_for

# Heading detection: works for resumes, papers and patents reasonably well.
HEADING_PATTERNS = [
    re.compile(r"^\s{0,3}#{1,6}\s+\S+.*$"),              # markdown
    re.compile(r"^\s*[A-Z][A-Z0-9\s,&/\-]{3,}\s*$"),     # ALL CAPS
    re.compile(r"^\s*.{2,60}:\s*$"),                     # "EXPERIENCE:"
]

# A line that reads as a self-contained fact: bullet, or "Label: value".
_BULLET_RE = re.compile(r"^\s*(?:[-•*•●‣⁃]|\d+[.)])\s+")
_LABELLED_RE = re.compile(r"^\s*[A-Za-z][\w \-/&()]{1,40}:\s*\S")


def is_heading(line: str) -> bool:
    s = line.strip()
    if not s or len(s) > 120:
        return False
    # A bullet is never a heading even if it ends in a colon.
    if _BULLET_RE.match(s):
        return False
    return any(p.match(s) for p in HEADING_PATTERNS)


def clean_text(t: str) -> str:
    t = (t or "").replace("\r", "\n")
    t = re.sub(r"\n{3,}", "\n\n", t)
    t = re.sub(r"[ \t]{2,}", " ", t)
    return t.strip()


@dataclass
class Chunk:
    text: str
    metadata: Dict[str, Any]


def _split_into_sections(text: str) -> List[Tuple[str, str]]:
    """Returns [(section_title, section_text)]; one 'Document' if no headings."""
    lines = text.split("\n")
    sections: List[Tuple[str, List[str]]] = []
    cur_title = "Document"
    cur_buf: List[str] = []

    for line in lines:
        if is_heading(line):
            if cur_buf:
                sections.append((cur_title, cur_buf))
                cur_buf = []
            cur_title = line.strip().lstrip("#").strip().rstrip(":").strip() or "Document"
            continue
        cur_buf.append(line)

    if cur_buf:
        sections.append((cur_title, cur_buf))

    out: List[Tuple[str, str]] = []
    for title, buf in sections:
        body = clean_text("\n".join(buf))
        if body:
            out.append((title, body))
    return out or [("Document", clean_text(text))]


def _atomic_groups(text: str, prof: ChunkProfile) -> List[str]:
    """
    Group consecutive bullets / 'Label: value' lines into chunks, never letting
    a group exceed max_words. Keeps each resume bullet and each transcript row
    independently retrievable instead of melting them into a 220-word blob.
    """
    lines = [ln for ln in text.split("\n") if ln.strip()]
    groups: List[str] = []
    buf: List[str] = []
    buf_words = 0

    def flush():
        nonlocal buf, buf_words
        if buf:
            groups.append("\n".join(buf).strip())
            buf = []
            buf_words = 0

    for ln in lines:
        w = len(ln.split())
        atomic = bool(_BULLET_RE.match(ln) or _LABELLED_RE.match(ln))

        # A new atomic line starts a new group once we've accumulated enough.
        if atomic and buf_words >= prof.max_words * 0.6:
            flush()

        if buf_words + w > prof.max_words and buf:
            flush()

        buf.append(ln)
        buf_words += w

    flush()
    return groups


def _chunk_by_words(text: str, max_words: int, overlap_words: int) -> List[str]:
    words = text.split()
    if not words:
        return []
    chunks: List[str] = []
    i = 0
    while i < len(words):
        j = min(i + max_words, len(words))
        chunks.append(" ".join(words[i:j]))
        if j == len(words):
            break
        i = max(0, j - overlap_words)
    return chunks


def make_chunks(
    text: str,
    file_name: str,
    page_label: Optional[str],
    doc_id: Optional[str] = None,
    profile_override: Optional[str] = None,
) -> List[Chunk]:
    text = clean_text(text)
    if not text:
        return []

    prof = profile_for(file_name, profile_override)
    out: List[Chunk] = []

    for section_title, section_text in _split_into_sections(text):
        if prof.prefer_atomic_lines:
            parts = _atomic_groups(section_text, prof)
            # Fall back to windows if the section had no line structure at all.
            if not parts:
                parts = _chunk_by_words(section_text, prof.max_words, prof.overlap_words)
        else:
            parts = _chunk_by_words(section_text, prof.max_words, prof.overlap_words)

        for k, part in enumerate(parts):
            if len(part.split()) < prof.min_words:
                continue

            # Prepending the section title gives the embedding local context —
            # a bare bullet like "Python, PyTorch" is ambiguous without it.
            embed_text = f"{section_title}\n{part}" if section_title != "Document" else part

            meta = {
                "file_name": file_name,
                "page_label": page_label or "n/a",
                "section": section_title,
                "chunk_id": k,
                "profile": prof.name,
            }
            if doc_id is not None:
                meta["doc_id"] = doc_id

            out.append(Chunk(text=embed_text, metadata=meta))

    return out


def chunk_text(text: str, file_name: str, page_num: int) -> List[Chunk]:
    """Backwards-compatible wrapper used by older ingestion paths."""
    return make_chunks(text, file_name=file_name, page_label=str(page_num), doc_id=file_name)


def chunk_text_for_file(text: str, file_name: str, page_num: int) -> List[Chunk]:
    return chunk_text(text, file_name=file_name, page_num=page_num)
