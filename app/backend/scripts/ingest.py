# app/backend/scripts/ingest.py
from __future__ import annotations

import os
from pathlib import Path
from typing import List

from src.core.config import PRIVATE_DATA_DIR
from src.rag.embedder import get_embedder
from src.rag.store import HybridStore
from src.rag.ingest_pipeline import ingest_paths


def _collect_paths() -> List[str]:
    """
    Collect ingestible files from two directories:

      data/private/  — original source documents (PDFs)
      data/curated/  — hand-written / verified markdown

    Curated markdown exists because several sources do not survive automated
    extraction. Transcripts in particular are PDF tables that pypdf flattens
    into unbound column blocks (all subject codes, then all grades, then all
    subject names), so indexing them raw invites the model to pair the wrong
    grade with the wrong subject. education.md carries the verified version
    instead, and the raw transcript PDFs are deliberately NOT ingested.

    The IEEE paper and the patent publication are excluded for the same
    reason, plus two more:

      1. Junk-to-signal ratio: at 220-word windows the paper alone produced
         58 of 120 chunks (48% of the whole index) for a portfolio bot a
         recruiter uses to ask "what has Rudra worked on" — and a third of
         those chunks were the references list (raw DOIs / journal citation
         strings), not anything answerable.
      2. PII exposure: the raw PDFs contain co-authors' personal emails
         (e.g. kshitijjoshi017@gmail.com) and, in the patent publication, a
         co-applicant's home mailing address. None of that belongs in a
         public-facing chatbot's index, identity guardrail or not — the
         guardrail stops it from being ATTRIBUTED to Rudra, but the raw text
         is still sitting there retrievable.

    projects.md already carries a clean, accurate summary of both (award,
    IEEE Xplore link, patent number/title/filing date) written for exactly
    the questions a recruiter would actually ask. That's kept; the raw PDFs
    are not ingested.
    """
    exts = {".pdf", ".txt", ".md"}
    excluded_pdfs = {
        "Copy of Cognitive-Chair (Final Draft).pdf",  # see docstring: junk ratio + co-author PII
        "Patent PDF.pdf",                              # see docstring: co-applicant home address
    }
    paths: List[str] = []

    private = Path(PRIVATE_DATA_DIR).resolve()
    curated = (private.parent / "curated").resolve()

    for base, label in ((private, "private"), (curated, "curated")):
        if not base.exists():
            print(f"⚠️  {label} dir not found: {base}")
            continue
        found = sorted(
            str(p) for p in base.rglob("*")
            if p.suffix.lower() in exts and p.name not in excluded_pdfs
        )
        skipped = sorted(
            p.name for p in base.rglob("*")
            if p.suffix.lower() in exts and p.name in excluded_pdfs
        )
        print(f"📁 {label}: {base}  ({len(found)} file(s))")
        for p in found:
            print("   -", Path(p).name)
        for name in skipped:
            print(f"   - {name}  [SKIPPED: see _collect_paths docstring]")
        paths.extend(found)

    return paths


def main() -> None:
    print("🚀 Running ingest:", __file__)
    print("🧭 CWD:", os.getcwd())

    paths = _collect_paths()
    if not paths:
        print("❌ No ingestible files found. Put PDFs in data/private and retry.")
        return

    embedder = get_embedder()
    store = HybridStore(embed_dim=embedder.dim)

    ingest_paths(paths, store)

    # store.save() happens inside ingest_paths() in your pipeline
    print("✅ Ingest complete.")
    print("   Output written to: ./storage (relative to app/backend)")


if __name__ == "__main__":
    main()
