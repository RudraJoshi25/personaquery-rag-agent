# src/rag/interview.py
from __future__ import annotations
from typing import Dict, Any, List
import uuid
import random

from src.rag.rag import get_store
from src.rag.retrieve_custom import make_context_pack
from src.rag.llm_groq import answer_with_groq

# in-memory sessions (fine for dev; swap to Redis for prod)
_SESSIONS: Dict[str, Dict[str, Any]] = {}

def _pick_seed_chunks(k: int = 10) -> List[Dict[str, Any]]:
    store = get_store()
    # pick random chunks from store to generate doc-grounded questions
    ids = list(range(len(store.chunks)))
    random.shuffle(ids)
    picked = ids[:k]
    hits = []
    for i in picked:
        ch = store.chunks[i]
        hits.append({
            "text": ch.text,
            "score": 1.0,
            "metadata": ch.metadata,
            "channel": "seed",
        })
    return hits

def start_interview(n_questions: int = 6) -> Dict[str, Any]:
    hits = _pick_seed_chunks(k=12)
    context = make_context_pack(hits, max_chars=9000)

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
    raw = answer_with_groq(prompt, context, mode="interview")

    # best effort parse JSON (keep simple)
    import json
    try:
        data = json.loads(raw)
        questions = data["questions"]
    except Exception:
        # fallback: create simple questions
        questions = [{
            "q": "Tell me about your most relevant project experience and the technologies used.",
            "expected_points": ["Project name(s) from docs", "Stack/tools from docs", "Impact/outcome if present"],
            "anchors": [],
        }]

    session_id = str(uuid.uuid4())
    _SESSIONS[session_id] = {
        "idx": 0,
        "questions": questions,
        "history": [],
    }

    return {
        "session_id": session_id,
        "question": questions[0]["q"],
        "question_number": 1,
        "total": len(questions),
    }

def answer_interview(session_id: str, user_answer: str) -> Dict[str, Any]:
    s = _SESSIONS.get(session_id)
    if not s:
        return {"error": "Invalid session_id. Start again."}

    idx = s["idx"]
    qobj = s["questions"][idx]
    expected = qobj.get("expected_points", [])
    anchors = qobj.get("anchors", [])

    # grading prompt: compare answer vs expected points; do not invent
    context = "\n".join(anchors) if anchors else ""
    grade_prompt = f"""
You are an interview grader.
Question: {qobj["q"]}
Candidate answer: {user_answer}

Expected points (from documents):
{expected}

Return:
1) score out of 10
2) What they did well
3) What they missed (explicitly list missing expected points)
4) A corrected "ideal answer" (short), grounded (no invention)
Output in plain text with bullet points.
"""
    grading = answer_with_groq(grade_prompt, context, mode="interview")

    s["history"].append({"q": qobj["q"], "a": user_answer, "grading": grading})

    # next question
    s["idx"] += 1
    if s["idx"] >= len(s["questions"]):
        return {
            "done": True,
            "grading": grading,
            "summary": "Interview complete.",
            "history": s["history"],
        }

    next_q = s["questions"][s["idx"]]["q"]
    return {
        "done": False,
        "grading": grading,
        "next_question": next_q,
        "question_number": s["idx"] + 1,
        "total": len(s["questions"]),
    }
