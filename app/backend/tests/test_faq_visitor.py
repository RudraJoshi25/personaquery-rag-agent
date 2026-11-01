"""Visitors are third parties: the FAQ cache must never answer them in the
second person, and paraphrases must hit the right entry or miss (a miss
falls through to the live pipeline; a wrong hit is served confidently)."""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from scripts import precompute_faq as pf
from src.rag import faq_cache

CACHE = Path(__file__).resolve().parents[1] / "storage" / "faq_cache.json"
pytestmark = pytest.mark.skipif(not CACHE.exists(), reason="FAQ cache not built")

_QUOTED = re.compile(r'"[^"\n]*"|“[^”\n]*”')
_YOU = re.compile(r"\b(you|your|yours|you're|you've)\b", re.IGNORECASE)


def _entries():
    return json.loads(CACHE.read_text(encoding="utf-8"))["entries"]


def test_no_cached_answer_addresses_the_reader():
    bad = [(e["question"], m.group(0)) for e in _entries()
           for m in _YOU.finditer(_QUOTED.sub(" ", e["answer"]))]
    assert not bad, bad


def test_first_person_golden_questions_are_not_cached():
    golden = [json.loads(line)["question"] for line in pf.QA_PATH.read_text(encoding="utf-8").splitlines()
              if line.strip() and not line.startswith("#")]
    first_person = {q for q in golden if pf._FIRST_PERSON.search(q)}
    assert first_person, "golden set should still have its first-person identity cases"
    assert not first_person & set(pf._load_eval_questions())
    assert not first_person & {e["question"] for e in _entries()}


PHONE = {"What is his phone number?", "What is Rudra's phone number?"}
CONTACT = {"How can I contact Rudra?", "What are his contact details?"}
EMAIL = {"What is Rudra's email?"} | CONTACT
LOCATION = {"Where is Rudra based?", "Where does Rudra live?", "Which city is he in?"}
SKILLS = {"What are Rudra's technical skills?", "What programming languages does Rudra know?",
          "Which stack is he strongest in?", "What are his skills?", "What tech does he know?"}
PROJECTS = {"What projects has Rudra built?", "What projects has he worked on?"}
EDUCATION = {"What is Rudra's educational background?", "What did Rudra study?",
             "What did he study?", "What degree does he have?"}
PROFILE = {"Summarise Rudra's professional profile.", "Summarise his experience for a recruiter.",
           "Who is Rudra, in one paragraph?"}
NONE: set[str] = set()

# (visitor phrasing, acceptable matches, must it hit at the current threshold?)
PARAPHRASES = [
    ("what is his phone number", PHONE, True),
    ("what's Rudra's phone number?", PHONE, True),
    ("can I get his mobile number", PHONE, True),
    ("what is my phone number", PHONE, True),
    ("what is your phone number", PHONE, True),
    ("how can I call him", PHONE | CONTACT, False),
    ("does he have a number I can ring", PHONE | CONTACT, False),
    ("phone number?", PHONE, False),
    ("his contact details", CONTACT, True),
    ("how do I get in touch with Rudra", CONTACT, True),
    ("how can I reach him", CONTACT | PHONE, False),
    ("what's his email", EMAIL, False),
    ("what is Rudra's email address", EMAIL, True),
    ("email address?", EMAIL, False),
    ("where does he live", LOCATION, False),
    ("where is he based", LOCATION, False),
    ("which city is Rudra in", LOCATION, True),
    ("is he willing to relocate", LOCATION, False),
    ("what are his skills", SKILLS, True),
    ("what technologies does he know", SKILLS, True),
    ("what languages does he code in", SKILLS, False),
    ("what projects has he worked on", PROJECTS, True),
    ("show me his projects", PROJECTS, False),
    ("what did he study at university", EDUCATION, True),
    ("what degree does he have", EDUCATION, True),
    ("where did he go to uni", EDUCATION, False),
    ("what ML frameworks does he use", {"Which AI/ML frameworks has Rudra used?"}, False),
    ("what internships has he done", {"What internships has Rudra done?"}, False),
    ("give me a summary of his profile", PROFILE, False),
    ("tell me about Rudra", PROFILE, True),
    # FAQ_ALIASES wordings, as a visitor would type them.
    ("Where does Rudra live?", LOCATION, True),
    ("Which city is he in?", LOCATION, True),
    ("What tech does he know?", SKILLS, True),
    ("What did he study?", EDUCATION, True),
    ("does he have a degree", EDUCATION, False),
    ("what has he built", PROJECTS, False),
    ("how old is he", NONE, False),
    ("what's the weather today", NONE, False),
]


@pytest.fixture(scope="module")
def lookup():
    faq_cache._loaded, faq_cache._cache = False, None
    if faq_cache._load() is None:
        pytest.skip("FAQ cache doesn't match the index")
    return faq_cache.lookup_faq


@pytest.mark.parametrize("phrase,ok,must_hit", PARAPHRASES)
def test_paraphrase_hits_right_entry_or_misses(lookup, phrase, ok, must_hit):
    hit = lookup(phrase)
    if hit is None:
        assert not must_hit, f"{phrase!r} missed"
    else:
        assert hit["faq_matched"] in ok, f"{phrase!r} -> {hit['faq_matched']} ({hit['faq_similarity']})"


def test_aliases_copy_their_target():
    by_q = {e["question"]: e for e in _entries()}
    for alias, target in pf.FAQ_ALIASES.items():
        assert target in by_q and target not in pf.FAQ_ALIASES, alias
        assert by_q[alias]["answer"] == by_q[target]["answer"], alias
        assert by_q[alias]["sources"] == by_q[target]["sources"], alias
