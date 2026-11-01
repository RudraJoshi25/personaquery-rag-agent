from __future__ import annotations

import os
import re

from pypdf import PdfReader

# HTML comments in the curated markdown are editorial notes addressed to
# Rudra (privacy-exclusion rationale, DISCREPANCY flags about his LinkedIn,
# CHECK items on illegible certificates) — they are commentary ABOUT the
# corpus, not part of it. Indexing them turned out to be actively harmful,
# not just noisy: the note explaining that the phone number was deliberately
# excluded from linkedin-profile.md was the TOP BM25 hit for "what is my
# phone number", outranking the resume chunk that actually contains the
# number — so the model read "deliberately excluded" and answered "not
# stated". They also expose internal editorial commentary to visitors.
_HTML_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)


def _strip_html_comments(text: str) -> str:
    return _HTML_COMMENT_RE.sub("", text)


# ---------------------------------------------------------------------------
# PDF icon-glyph cleanup
#
# LaTeX resume templates (Awesome-CV and friends) render contact details with
# FontAwesome icons. pypdf extracts the icon as a stray glyph AND the icon's
# text label, both fused to the value:
#
#     "/envel<GLYPH>perudrajoshi.cs@gmail.com"   <- "envelope" + the email
#     "♂phone+61 4XXXXXXXX"                  <- icon + "phone" + number
#     "/linkedinlinkedin.com/in/rudrajoshi-ai/"   <- label + the URL
#
# This is not cosmetic. The tokenizer split the address into
# ['envel', 'perudrajoshi', ...] — so the résumé chunk did not contain the
# token "rudrajoshi" at all, and BM25 could never match a contact question
# against the one document that actually holds the answer. Every contact
# query lost to the curated markdown instead.
#
# Labels are listed rather than pattern-guessed because the fused form has no
# word boundary to detect: only knowing "envelope" lets you split
# "envelperudrajoshi" correctly. Extend this list if a new icon appears.
# ---------------------------------------------------------------------------
# DO NOT simply delete the labels. The first version of this cleanup did, and
# it made retrieval WORSE (hit-rate 90% -> 85%, and "What is my phone number?"
# regressed from pass to fail). Reason: the raw text "♂phone+61 4XXXXXXXX"
# tokenizes to ['phone', '61', '4XXXXXXXX'] — the icon's label was the ONLY
# occurrence of the word "phone" in the chunk, and it was what BM25 matched.
# Deleting it removed the search term along with the noise.
#
# So labels are REWRITTEN into clean field labels instead. That fixes the
# fused value AND keeps (indeed strengthens) the searchable term, and reads
# better in the context window:
#
#     "♂phone+61 4XXXXXXXX"              -> "Phone: +61 4XXXXXXXX"
#     "/envel<GLYPH>perudrajoshi.cs@..." -> "Email: rudrajoshi.cs@..."
#     "/linkedinlinkedin.com/in/x"       -> "LinkedIn: linkedin.com/in/x"
_EMAIL_FIELD_RE = re.compile(
    r"[/\s]*(?:envel\w*?pe|envelope|envel|e-?mail)\s*([\w.+-]+@[\w.-]+\.\w+)",
    re.IGNORECASE,
)
_PHONE_FIELD_RE = re.compile(
    r"[/\s]*(?:telephone|phone|mobile|tel|cell)\s*(\+?\d[\d\s()-]{5,})",
    re.IGNORECASE,
)
_LINKEDIN_FIELD_RE = re.compile(r"[/\s]*linkedin\s*(linkedin\.com\S*)", re.IGNORECASE)
_GITHUB_FIELD_RE = re.compile(r"[/\s]*github\s*(github\.com\S*)", re.IGNORECASE)
# Bare "/github", "/linkedin" markers whose value sits on the next line.
_BARE_ICON_RE = re.compile(
    r"/(?:github|linkedin|globe|home|mapmarker|website)(?![\w.])", re.IGNORECASE
)

# Symbol / private-use glyphs that carry no textual meaning (the icons
# themselves). Letters, digits, punctuation and whitespace are preserved.
_GLYPH_CATEGORIES = {"So", "Co", "Cn", "Sk"}


def _clean_pdf_text(text: str) -> str:
    """Normalize icon-glyph contact markup into clean, searchable fields."""
    import unicodedata

    text = unicodedata.normalize("NFKC", text)
    text = "".join(
        ch for ch in text
        if ch in "\n\t" or unicodedata.category(ch) not in _GLYPH_CATEGORIES
    )
    text = _EMAIL_FIELD_RE.sub(r" Email: \1", text)
    # The number itself is never indexed (owner's choice: email only). The
    # "Phone:" label stays so phone questions still retrieve this chunk and
    # get an honest "available on request" instead of a co-author's number.
    text = _PHONE_FIELD_RE.sub(" Phone: available on request ", text)
    text = _LINKEDIN_FIELD_RE.sub(r" LinkedIn: \1", text)
    text = _GITHUB_FIELD_RE.sub(r" GitHub: \1", text)
    text = _BARE_ICON_RE.sub(" ", text)
    return text

from src.rag.chunking import make_chunks
from src.rag.embedder import get_embedder
from src.rag.rerank import precompute_doc_vectors
from src.rag.source_corrections import APPLIES_TO, CORRECTIONS, apply_corrections
from src.rag.store import HybridStore, StoredChunk


def read_pdf_pages(path: str) -> list[dict]:
    reader = PdfReader(path)
    correct = os.path.basename(path) == APPLIES_TO
    totals = [0] * len(CORRECTIONS)
    pages = []
    for i, page in enumerate(reader.pages):
        txt = _clean_pdf_text(page.extract_text() or "")
        if correct:
            txt, counts = apply_corrections(txt)
            totals = [a + b for a, b in zip(totals, counts)]
        pages.append({"page_label": str(i + 1), "text": txt})
    if correct:
        for (phrase, _), n in zip(CORRECTIONS, totals):
            if n == 0:
                print(f"[ingest] WARNING: CV correction matched nothing: {phrase!r}")
    return pages


def ingest_paths(paths: list[str], store: HybridStore) -> None:
    embedder = get_embedder()

    max_chunks = int(os.getenv("MAX_CHUNKS", "600"))  # ✅ cap to avoid RAM blowups
    all_chunks: list[StoredChunk] = []

    for p in paths:
        file_name = os.path.basename(p)
        ext = os.path.splitext(p)[1].lower()

        if ext == ".pdf":
            for pg in read_pdf_pages(p):
                raw = (pg["text"] or "").strip()
                if len(raw) < 20:
                    continue  # skip empty pages

                chunks = make_chunks(
                    raw,
                    file_name=file_name,
                    page_label=pg["page_label"],
                    doc_id=file_name,
                )
                for c in chunks:
                    txt = (c.text or "").strip()
                    if len(txt) < 20:
                        continue
                    all_chunks.append(StoredChunk(text=txt, metadata=c.metadata))
                    if len(all_chunks) >= max_chunks:
                        break
                if len(all_chunks) >= max_chunks:
                    break

        else:
            with open(p, "r", encoding="utf-8", errors="ignore") as f:
                txt = (f.read() or "").strip()

            if ext in (".md", ".markdown"):
                txt = _strip_html_comments(txt).strip()

            if len(txt) < 20:
                continue

            chunks = make_chunks(txt, file_name=file_name, page_label=None, doc_id=file_name)
            for c in chunks:
                t = (c.text or "").strip()
                if len(t) < 20:
                    continue
                all_chunks.append(StoredChunk(text=t, metadata=c.metadata))
                if len(all_chunks) >= max_chunks:
                    break

        if len(all_chunks) >= max_chunks:
            break

    if not all_chunks:
        raise RuntimeError("No chunks created. PDF extraction might be empty or chunking is too strict.")

    # ✅ embeddings (must match chunk count)
    texts = [c.text for c in all_chunks]
    vectors = embedder.embed(texts)

    if not vectors:
        raise RuntimeError("Embedding returned 0 vectors. Likely empty/short chunks. Check PDF extraction.")
    if len(vectors) != len(all_chunks):
        raise RuntimeError(
            f"Vector/chunk mismatch: vectors={len(vectors)} chunks={len(all_chunks)}. "
            "Ensure you filter empty/short chunks only in ingest_pipeline."
        )

    store.build(vectors, all_chunks)

    # Precompute ColBERT late-interaction vectors once per chunk, so query time
    # only ever has to encode the (short) question. Degrades to None (i.e. the
    # old on-the-fly-per-query behavior) if ColBERT is unavailable in this
    # environment — ingestion must never hard-fail on a reranking model.
    doc_vecs = precompute_doc_vectors(texts)
    if doc_vecs is not None:
        store.set_colbert_vectors(doc_vecs)
        print(f"[ingest] precomputed colbert vectors for {len(doc_vecs)} chunks", flush=True)
    else:
        print("[ingest] colbert unavailable at ingest time; queries will encode on the fly", flush=True)

    store.save()
