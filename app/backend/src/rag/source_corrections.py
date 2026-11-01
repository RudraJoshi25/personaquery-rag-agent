"""
Ingest-time corrections for stale claims in the CV PDF.

TEMPORARY. This exists only because data/private/Rudra_CV.pdf still carries
claims the code no longer backs (ChromaDB, LangChain/LlamaIndex, Llama-3.3-70B,
old test counts and eval figures, "patent holder"), and the PDF can't be
edited right now. The file on disk is never modified; only the indexed text
is. Delete this module, its call in ingest_pipeline.py and its test once the
CV itself is updated.

Each phrase is matched with every space meaning "any whitespace run", so a
phrase the PDF extractor split across a line break still matches while the
line structure (which section detection needs) is left alone.
"""
from __future__ import annotations

import re

APPLIES_TO = "Rudra_CV.pdf"

# Ordered: the specific ChromaDB sentence must go before the bare "ChromaDB, ".
CORRECTIONS: list[tuple[str, str]] = [
    ("Indian Government patent holder", "Indian patent applicant (application published 2022)"),
    ("migrated the vector store to ChromaDB behind a swappable backend abstraction",
     "built an in-process vector store (NumPy + JSONL, no external database)"),
    ("vector databases (ChromaDB), ", ""),
    ("ChromaDB, ", ""),
    ("LangChain, LlamaIndex, ", ""),
    ("LangChain/LlamaIndex, ", ""),
    (", Sentence-Transformers", ""),
    ("Groq (Llama-3.3-70B)", "Groq (Qwen3.8-27B, GPT-OSS-20B)"),
    ("all-MiniLM-L6-v2, 384-dim", "bge-small-en-v1.5 via FastEmbed, 384-dim"),
    ("a 20-case golden-set", "an 18-case golden-set"),
    (": 100% recall@8, 65% source-precision@8", ""),
    (", 29 tests", ""),
]

_COMPILED = [(re.compile(re.escape(p).replace(r"\ ", r"\s+")), r) for p, r in CORRECTIONS]


def apply_corrections(text: str) -> tuple[str, list[int]]:
    """Return corrected text and the match count of each correction, in order."""
    counts = []
    for rx, repl in _COMPILED:
        text, n = rx.subn(repl, text)
        counts.append(n)
    return text, counts
