from __future__ import annotations
import os
from typing import List
from pypdf import PdfReader

from src.rag.chunking import make_chunks
from src.rag.store import HybridStore, StoredChunk
from src.rag.embedder import get_embedder


def read_pdf_pages(path: str) -> List[dict]:
    reader = PdfReader(path)
    pages = []
    for i, page in enumerate(reader.pages):
        txt = page.extract_text() or ""
        pages.append({"page_label": str(i + 1), "text": txt})
    return pages


def ingest_paths(paths: List[str], store: HybridStore) -> None:
    embedder = get_embedder()

    max_chunks = int(os.getenv("MAX_CHUNKS", "600"))  # ✅ cap to avoid RAM blowups
    all_chunks: List[StoredChunk] = []

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
    store.save()
