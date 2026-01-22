from __future__ import annotations
from pathlib import Path
from typing import List
from pypdf import PdfReader

from src.rag.chunking import Chunk, chunk_text_for_file


def load_pdf_chunks(private_data_dir: str) -> List[Chunk]:
    base = Path(private_data_dir).resolve()
    if not base.exists():
        raise FileNotFoundError(f"PRIVATE_DATA_DIR not found: {base}")

    pdfs = sorted(base.rglob("*.pdf"))
    if not pdfs:
        raise FileNotFoundError(f"No PDFs found under: {base}")

    all_chunks: List[Chunk] = []

    for pdf_path in pdfs:
        print(f"Reading PDF: {pdf_path.name}", flush=True)
        reader = PdfReader(str(pdf_path))
        total_pages = len(reader.pages)

        for i, page in enumerate(reader.pages, start=1):
            print(f"  page {i}/{total_pages}", flush=True)
            text = page.extract_text() or ""
            all_chunks.extend(chunk_text_for_file(text, file_name=pdf_path.name, page_num=i))

    return all_chunks
