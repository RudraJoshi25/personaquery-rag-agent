import pytest

from src.main import app


def _interview_mounted() -> bool:
    return any(getattr(r, "path", "") == "/interview/start" for r in app.routes)


pytestmark = pytest.mark.skipif(
    not _interview_mounted(),
    reason="interview router not mounted",
)


QUESTIONS_JSON = (
    '{"questions":[{"q":"Tell me about your RAG project.",'
    '"expected_points":["Groq","hybrid retrieval"],"anchors":[]},'
    '{"q":"What patent did you file?","expected_points":["anomaly detection"],"anchors":[]},'
    '{"q":"q3","expected_points":[],"anchors":[]}]}'
)


def test_start_interview_happy_path(client, patched_retrieval, mock_groq_success):
    mock_groq_success(QUESTIONS_JSON)
    r = client.post("/interview/start", json={"n_questions": 3})
    assert r.status_code == 200
    body = r.json()
    assert body["session_id"]
    assert body["question"] == "Tell me about your RAG project."
    assert body["total"] == 3


def test_n_questions_out_of_range(client):
    r = client.post("/interview/start", json={"n_questions": 20})
    assert r.status_code == 422
    r = client.post("/interview/start", json={"n_questions": 1})
    assert r.status_code == 422


def test_full_flow_to_done(client, patched_retrieval, mock_groq_success):
    mock_groq_success(QUESTIONS_JSON)
    session_id = client.post("/interview/start", json={"n_questions": 3}).json()["session_id"]

    mock_groq_success("Score: 7/10")
    r1 = client.post("/interview/answer", json={"session_id": session_id, "answer": "I used Groq."})
    assert r1.status_code == 200
    assert r1.json()["done"] is False

    r2 = client.post("/interview/answer", json={"session_id": session_id, "answer": "Anomaly detection."})
    assert r2.json()["done"] is False

    r3 = client.post("/interview/answer", json={"session_id": session_id, "answer": "Final answer."})
    body = r3.json()
    assert body["done"] is True
    assert len(body["history"]) == 3


def test_invalid_session_id_returns_404(client):
    r = client.post("/interview/answer", json={"session_id": "not-a-real-session", "answer": "x"})
    assert r.status_code == 404


def test_grading_llm_failure_returns_502(client, patched_retrieval, mock_groq_success, mock_groq_error):
    mock_groq_success(QUESTIONS_JSON)
    session_id = client.post("/interview/start", json={"n_questions": 3}).json()["session_id"]

    mock_groq_error(status_code=500)
    r = client.post("/interview/answer", json={"session_id": session_id, "answer": "answer"})
    assert r.status_code == 502


def test_start_falls_back_to_grounded_questions_on_llm_failure(client, patched_retrieval, mock_groq_error):
    mock_groq_error(status_code=500)
    r = client.post("/interview/start", json={"n_questions": 3})
    assert r.status_code == 200
    assert r.json()["question"]  # non-empty, grounded fallback question
