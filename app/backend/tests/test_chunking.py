from src.rag.ingestion.chunking import clean_text, is_heading, make_chunks


def test_is_heading_markdown():
    assert is_heading("## Experience") is True
    assert is_heading("# Title") is True


def test_is_heading_all_caps():
    assert is_heading("EXPERIENCE") is True
    assert is_heading("WORK EXPERIENCE & SKILLS") is True


def test_is_heading_trailing_colon():
    assert is_heading("Skills:") is True


def test_is_heading_rejects_prose():
    assert is_heading("This is a normal sentence about my work.") is False


def test_is_heading_rejects_long_lines():
    assert is_heading("A" * 130 + ":") is False


def test_is_heading_rejects_empty():
    assert is_heading("") is False
    assert is_heading("   ") is False


def test_clean_text_collapses_blank_lines():
    assert clean_text("a\n\n\n\nb") == "a\n\nb"


def test_clean_text_collapses_spaces():
    assert clean_text("a    b\t\tc") == "a b c"


def test_make_chunks_basic_metadata():
    text = "EXPERIENCE\nBuilt a RAG chatbot using Groq and hybrid retrieval for a persona bot."
    chunks = make_chunks(text, file_name="resume.pdf", page_label="1", doc_id="resume.pdf")
    assert len(chunks) >= 1
    c = chunks[0]
    assert c.metadata["file_name"] == "resume.pdf"
    assert c.metadata["page_label"] == "1"
    assert c.metadata["doc_id"] == "resume.pdf"
    assert c.metadata["section"] == "EXPERIENCE"
    assert c.metadata["chunk_id"] == 0


def test_make_chunks_defaults_page_label():
    chunks = make_chunks("Some plain text without headings.", file_name="notes.md", page_label=None)
    assert chunks[0].metadata["page_label"] == "n/a"
    assert "doc_id" not in chunks[0].metadata


def test_make_chunks_empty_text_returns_empty():
    assert make_chunks("", file_name="x.pdf", page_label="1") == []


def test_make_chunks_word_window_overlap():
    words = [f"word{i}" for i in range(300)]
    text = " ".join(words)
    chunks = make_chunks(text, file_name="big.pdf", page_label="1")
    assert len(chunks) >= 2
    # Overlap: the tail of chunk 0 should reappear near the start of chunk 1
    tail_of_first = chunks[0].text.split()[-5:]
    head_of_second = chunks[1].text.split()[:40]
    assert any(w in head_of_second for w in tail_of_first)
