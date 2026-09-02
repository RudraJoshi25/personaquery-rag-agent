from src.rag.ingestion.pii_redact import redact_text

OWNER_EMAIL = "owner@example.com"
OWNER_PHONE = "+15551234567"


def test_owner_email_preserved():
    text = f"Contact {OWNER_EMAIL} for details."
    out = redact_text(text, owner_email=OWNER_EMAIL, owner_phone=OWNER_PHONE)
    assert OWNER_EMAIL in out


def test_owner_email_case_insensitive_preserved():
    text = f"Contact {OWNER_EMAIL.upper()} for details."
    out = redact_text(text, owner_email=OWNER_EMAIL, owner_phone=OWNER_PHONE)
    assert "[REDACTED_EMAIL]" not in out


def test_third_party_email_redacted():
    text = "Coauthor: someone.else@gmail.com"
    out = redact_text(text, owner_email=OWNER_EMAIL, owner_phone=OWNER_PHONE)
    assert "someone.else@gmail.com" not in out
    assert "[REDACTED_EMAIL]" in out


def test_multiple_third_party_emails_all_redacted():
    text = "a@x.com b@y.com c@z.com"
    out = redact_text(text, owner_email=OWNER_EMAIL, owner_phone=OWNER_PHONE)
    assert out.count("[REDACTED_EMAIL]") == 3


def test_owner_phone_preserved():
    text = f"Call me at {OWNER_PHONE}."
    out = redact_text(text, owner_email=OWNER_EMAIL, owner_phone=OWNER_PHONE)
    assert OWNER_PHONE in out


def test_third_party_phone_redacted():
    text = "Call the office at +91 98765 43210 for support."
    out = redact_text(text, owner_email=OWNER_EMAIL, owner_phone=OWNER_PHONE)
    assert "98765 43210" not in out
    assert "[REDACTED_PHONE]" in out


def test_patent_and_doi_numbers_not_mangled():
    # These are the kind of numeric strings that appear throughout the
    # actual corpus (patent filings, DOIs) - the phone regex is
    # deliberately conservative to avoid false-positiving on them.
    text = "Patent US10015009B2 references DOI 10.1109/ACCESS.2020.1234567."
    out = redact_text(text, owner_email=OWNER_EMAIL, owner_phone=OWNER_PHONE)
    assert "[REDACTED_PHONE]" not in out
    assert "US10015009B2" in out


def test_empty_text_returns_empty():
    assert redact_text("", owner_email=OWNER_EMAIL, owner_phone=OWNER_PHONE) == ""


def test_text_with_no_pii_unchanged():
    text = "This resume describes several projects and skills."
    assert redact_text(text, owner_email=OWNER_EMAIL, owner_phone=OWNER_PHONE) == text
