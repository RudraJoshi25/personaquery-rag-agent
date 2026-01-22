from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict
import re


@dataclass
class Chunk:
    text: str
    metadata: Dict


# -------------------------
# Helpers
# -------------------------
def clean_text(t: str) -> str:
    t = t.replace("\x00", " ")
    # normalize whitespace
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


def _truncate_if_huge(text: str, max_chars: int) -> str:
    if len(text) > max_chars:
        return text[:max_chars]
    return text


def _make_chunks_from_blocks(
    blocks: List[str],
    file_name: str,
    page_num: int,
    chunk_size: int,
    overlap: int,
    max_chunks_per_page: int,
) -> List[Chunk]:
    """
    Pack blocks (paragraphs or section text) into ~chunk_size chunks.
    Add a light overlap (last N chars) to reduce boundary loss.
    """
    chunks: List[Chunk] = []
    buf = ""

    def flush_buf():
        nonlocal buf
        if not buf.strip():
            buf = ""
            return
        chunks.append(
            Chunk(
                text=buf.strip(),
                metadata={
                    "file_name": file_name,
                    "page_label": str(page_num),
                },
            )
        )
        # overlap
        if overlap > 0 and len(buf) > overlap:
            buf = buf[-overlap:]
        else:
            buf = ""

    created = 0
    for b in blocks:
        b = b.strip()
        if not b:
            continue

        # If single block is gigantic, split it deterministically
        while len(b) > chunk_size:
            piece = b[:chunk_size]
            b = b[chunk_size - overlap :] if overlap > 0 else b[chunk_size:]
            if buf:
                flush_buf()
                created += 1
                if created >= max_chunks_per_page:
                    return chunks
            chunks.append(
                Chunk(
                    text=piece.strip(),
                    metadata={"file_name": file_name, "page_label": str(page_num)},
                )
            )
            created += 1
            if created >= max_chunks_per_page:
                return chunks

        # Normal packing
        if len(buf) + len(b) + 1 <= chunk_size:
            buf = (buf + "\n" + b).strip() if buf else b
        else:
            flush_buf()
            created += 1
            if created >= max_chunks_per_page:
                return chunks
            buf = b

    flush_buf()
    return chunks


# -------------------------
# Option A: Paragraph-first (best for resumes/patents)
# -------------------------
def chunk_by_paragraphs(
    text: str,
    file_name: str,
    page_num: int,
    chunk_size: int = 1100,
    overlap: int = 200,
    max_page_chars: int = 200_000,
    max_chunks_per_page: int = 120,
) -> List[Chunk]:
    text = clean_text(text)
    if not text:
        return []

    text = _truncate_if_huge(text, max_page_chars)

    # Split into paragraphs: blank line OR bullet-heavy lines
    # Keep bullets together by splitting on double newlines.
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]

    return _make_chunks_from_blocks(
        blocks=paras,
        file_name=file_name,
        page_num=page_num,
        chunk_size=chunk_size,
        overlap=overlap,
        max_chunks_per_page=max_chunks_per_page,
    )


# -------------------------
# Option B: Heading-aware (best for papers)
# -------------------------
_HEADING_RE = re.compile(
    r"^(abstract|introduction|related work|methodology|methods|results|discussion|conclusion|references)\b",
    re.IGNORECASE,
)

_NUMBERED_HEADING_RE = re.compile(r"^\s*(\d+(\.\d+)*)\s+.+")  # e.g., 1. / 2.1 /
_ALLCAPS_RE = re.compile(r"^[A-Z][A-Z0-9 \-,:]{6,}$")        # e.g., "EXPERIMENTAL RESULTS"


def chunk_heading_aware(
    text: str,
    file_name: str,
    page_num: int,
    chunk_size: int = 1200,
    overlap: int = 200,
    max_page_chars: int = 250_000,
    max_chunks_per_page: int = 140,
) -> List[Chunk]:
    text = clean_text(text)
    if not text:
        return []

    text = _truncate_if_huge(text, max_page_chars)

    lines = [ln.strip() for ln in text.splitlines()]
    blocks: List[str] = []
    current = ""

    def is_heading(line: str) -> bool:
        if not line:
            return False
        if _HEADING_RE.match(line):
            return True
        if _NUMBERED_HEADING_RE.match(line):
            return True
        if _ALLCAPS_RE.match(line):
            return True
        # short line ending with ":" is often a heading
        if len(line) <= 60 and line.endswith(":"):
            return True
        return False

    for ln in lines:
        if is_heading(ln):
            # flush previous section
            if current.strip():
                blocks.append(current.strip())
            # start new block with heading emphasized
            current = f"{ln}\n"
        else:
            # keep adding lines; insert spacing around paragraphs lightly
            if ln:
                current += ln + " "
            else:
                current += "\n\n"

    if current.strip():
        blocks.append(current.strip())

    return _make_chunks_from_blocks(
        blocks=blocks,
        file_name=file_name,
        page_num=page_num,
        chunk_size=chunk_size,
        overlap=overlap,
        max_chunks_per_page=max_chunks_per_page,
    )


# -------------------------
# Router: choose strategy per file
# -------------------------
def chunk_text_for_file(text: str, file_name: str, page_num: int) -> List[Chunk]:
    """
    Assign strategies:
    - Resume/Patent: paragraph-first
    - Research paper: heading-aware
    Uses filename heuristics. Adjust as you like.
    """
    name = file_name.lower()

    # Heuristics: treat IEEE/research paper filenames as "paper"
    is_paper = any(k in name for k in ["cognitive", "paper", "ieee", "draft", "final"])

    if is_paper:
        return chunk_heading_aware(text, file_name=file_name, page_num=page_num)

    # Default for resume/patent/etc.
    return chunk_by_paragraphs(text, file_name=file_name, page_num=page_num)
