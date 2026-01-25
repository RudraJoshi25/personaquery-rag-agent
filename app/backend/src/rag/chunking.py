# src/rag/chunking.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, List, Optional
import re

# Simple heading detection: works for resumes + papers + patents reasonably well
# - Lines in ALL CAPS
# - Lines ending with ":" (like "EXPERIENCE:")
# - Markdown headings (#, ##)
HEADING_PATTERNS = [
    re.compile(r"^\s{0,3}#{1,6}\s+\S+.*$"),              # markdown
    re.compile(r"^\s*[A-Z][A-Z0-9\s,&/\-]{3,}\s*$"),     # ALL CAPS
    re.compile(r"^\s*.+:\s*$"),                          # ends with ":"
]

def is_heading(line: str) -> bool:
    s = line.strip()
    if not s:
        return False
    if len(s) > 120:
        return False
    return any(p.match(s) for p in HEADING_PATTERNS)

def clean_text(t: str) -> str:
    # normalize whitespace but keep paragraphs
    t = t.replace("\r", "\n")
    t = re.sub(r"\n{3,}", "\n\n", t)
    t = re.sub(r"[ \t]{2,}", " ", t)
    return t.strip()

@dataclass
class Chunk:
    text: str
    metadata: Dict[str, Any]

def _split_into_sections(text: str) -> List[tuple[str, str]]:
    """
    Returns list of (section_title, section_text)
    If no headings exist, one section "Document".
    """
    lines = text.split("\n")
    sections: List[tuple[str, List[str]]] = []
    cur_title = "Document"
    cur_buf: List[str] = []

    for line in lines:
        if is_heading(line):
            if cur_buf:
                sections.append((cur_title, cur_buf))
                cur_buf = []
            cur_title = line.strip()
            continue
        cur_buf.append(line)

    if cur_buf:
        sections.append((cur_title, cur_buf))

    out: List[tuple[str, str]] = []
    for title, buf in sections:
        body = clean_text("\n".join(buf))
        if body:
            out.append((title, body))
    return out if out else [("Document", clean_text(text))]

def _chunk_by_words(text: str, max_words: int = 220, overlap_words: int = 40) -> List[str]:
    words = text.split()
    if not words:
        return []
    chunks = []
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
) -> List[Chunk]:
    """
    Section-aware chunking:
    1) split by headings into sections
    2) chunk each section by word windows with overlap
    """
    text = clean_text(text)
    if not text:
        return []

    sections = _split_into_sections(text)
    out: List[Chunk] = []

    for section_title, section_text in sections:
        parts = _chunk_by_words(section_text, max_words=220, overlap_words=40)
        for k, part in enumerate(parts):
            meta = {
                "file_name": file_name,
                "page_label": page_label or "n/a",
                "section": section_title,
                "chunk_id": k,
            }
            if doc_id is not None:
                meta["doc_id"] = doc_id

            out.append(Chunk(text=part, metadata=meta))

    return out


def chunk_text(text: str, file_name: str, page_num: int) -> List[Chunk]:
    """
    Backwards-compatible wrapper used by older ingestion paths.
    """
    return make_chunks(text, file_name=file_name, page_label=str(page_num), doc_id=file_name)


def chunk_text_for_file(text: str, file_name: str, page_num: int) -> List[Chunk]:
    """
    Alias for chunk_text for older scripts.
    """
    return chunk_text(text, file_name=file_name, page_num=page_num)
