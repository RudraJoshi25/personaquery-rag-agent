# src/rag/interview.py
from __future__ import annotations

import json
import random
import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from src.core.config import INTERVIEW_MAX_SESSIONS, INTERVIEW_SESSION_TTL_SECONDS
from src.rag.generation.llm_groq import answer_with_groq
from src.rag.retrieval.search import get_store, make_context_pack


class SessionNotFoundError(Exception):
    """Raised when a session_id is unknown or has expired."""


class LLMUnavailableError(Exception):
    """Raised when the LLM grading call fails upstream."""


@dataclass
class InterviewQuestion:
    q: str
    expected_points: list[str] = field(default_factory=list)
    anchors: list[str] = field(default_factory=list)


@dataclass
class InterviewSession:
    session_id: str
    questions: list[InterviewQuestion]
    idx: int = 0
    history: list[dict[str, Any]] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    last_active_at: float = field(default_factory=time.time)


# In-memory sessions: fine for a single-process Render instance. If this
# ever moves to multiple instances/workers, this needs to become a shared
# store (e.g. Redis) since sessions are not sticky across processes.
_SESSIONS: dict[str, InterviewSession] = {}


def _evict_if_needed() -> None:
    now = time.time()
    stale = [sid for sid, s in _SESSIONS.items() if now - s.last_active_at > INTERVIEW_SESSION_TTL_SECONDS]
    for sid in stale:
        _SESSIONS.pop(sid, None)

    overflow = len(_SESSIONS) - INTERVIEW_MAX_SESSIONS + 1
    if overflow > 0:
        oldest = sorted(_SESSIONS.values(), key=lambda s: s.last_active_at)[:overflow]
        for s in oldest:
            _SESSIONS.pop(s.session_id, None)


def _pick_seed_chunks(k: int = 12) -> list[dict[str, Any]]:
    store = get_store()
    ids = list(range(len(store.chunks)))
    random.shuffle(ids)
    hits = []
    for i in ids[:k]:
        ch = store.chunks[i]
        hits.append({"text": ch.text, "score": 1.0, "metadata": ch.metadata, "channel": "seed"})
    return hits


def _strip_code_fences(raw: str) -> str:
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    return m.group(0) if m else raw


def _fallback_questions(seed_hits: list[dict[str, Any]], n_questions: int) -> list[InterviewQuestion]:
    """Grounded fallback used when the LLM fails or returns unparsable JSON -
    builds real questions from seed-chunk metadata rather than a single
    generic canned question."""
    qs: list[InterviewQuestion] = []
    seen = set()
    for h in seed_hits:
        m = h["metadata"]
        section = m.get("section", "Document")
        file_name = m.get("file_name", "the documents")
        key = (file_name, section)
        if key in seen:
            continue
        seen.add(key)
        qs.append(
            InterviewQuestion(
                q=f"Tell me about your experience related to '{section}' as described in {file_name}.",
                anchors=[f"[{file_name} | p.{m.get('page_label', '?')} | {section}]"],
            )
        )
        if len(qs) >= n_questions:
            break
    if not qs:
        qs = [
            InterviewQuestion(
                q="Tell me about your most relevant project experience and the technologies used.",
                expected_points=["Project name(s) from docs", "Stack/tools from docs", "Impact/outcome if present"],
            )
        ]
    return qs


def _parse_questions(
    raw: str, n_questions: int, seed_hits: list[dict[str, Any]], had_error: bool
) -> list[InterviewQuestion]:
    if had_error:
        return _fallback_questions(seed_hits, n_questions)
    try:
        data = json.loads(_strip_code_fences(raw))
        items = data["questions"]
        out = [
            InterviewQuestion(
                q=it["q"],
                expected_points=it.get("expected_points", []),
                anchors=it.get("anchors", []),
            )
            for it in items
            if isinstance(it, dict) and it.get("q")
        ]
        return out or _fallback_questions(seed_hits, n_questions)
    except Exception:
        return _fallback_questions(seed_hits, n_questions)


def start_interview(n_questions: int = 6) -> dict[str, Any]:
    _evict_if_needed()

    seed_hits = _pick_seed_chunks(k=12)
    context = make_context_pack(seed_hits, max_chars=9000)

    prompt = f"""
Create {n_questions} interview questions STRICTLY based on the CONTEXT.
For each question, also produce an "expected_points" list (3-6 bullets) that must be mentioned to be fully correct.
Output JSON only in this schema:
{{
  "questions":[
    {{"q":"...", "expected_points":["...","..."], "anchors":["[file | p.X | section]", "..."]}}
  ]
}}
"""
    result = answer_with_groq(prompt, context, mode="interview_generate", json_mode=True)
    questions = _parse_questions(result.text, n_questions, seed_hits, had_error=result.error is not None)

    session_id = str(uuid.uuid4())
    _SESSIONS[session_id] = InterviewSession(session_id=session_id, questions=questions)

    return {
        "session_id": session_id,
        "question": questions[0].q,
        "question_number": 1,
        "total": len(questions),
    }


def answer_interview(session_id: str, user_answer: str) -> dict[str, Any]:
    s = _SESSIONS.get(session_id)
    if s is None:
        raise SessionNotFoundError(session_id)
    if time.time() - s.last_active_at > INTERVIEW_SESSION_TTL_SECONDS:
        _SESSIONS.pop(session_id, None)
        raise SessionNotFoundError(session_id)

    s.last_active_at = time.time()
    qobj = s.questions[s.idx]
    context = "\n".join(qobj.anchors)

    grade_prompt = f"""
You are an interview grader.
Question: {qobj.q}
Candidate answer: {user_answer}

Expected points (from documents):
{qobj.expected_points}

Return:
1) score out of 10
2) What they did well
3) What they missed (explicitly list missing expected points)
4) A corrected "ideal answer" (short), grounded (no invention)
Output in plain text with bullet points.
"""
    result = answer_with_groq(grade_prompt, context, mode="interview_grade")
    if result.error is not None:
        raise LLMUnavailableError(result.error)

    grading = result.text
    s.history.append({"q": qobj.q, "a": user_answer, "grading": grading})
    s.idx += 1

    if s.idx >= len(s.questions):
        return {
            "done": True,
            "grading": grading,
            "summary": "Interview complete.",
            "history": s.history,
        }

    next_q = s.questions[s.idx].q
    return {
        "done": False,
        "grading": grading,
        "next_question": next_q,
        "question_number": s.idx + 1,
        "total": len(s.questions),
    }
