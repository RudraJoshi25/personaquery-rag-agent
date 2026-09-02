import requests

from src.rag.generation.llm_groq import answer_with_groq


def test_missing_api_key_short_circuits(mocker, monkeypatch):
    monkeypatch.setattr("src.rag.generation.llm_groq.GROQ_API_KEY", "")
    post = mocker.patch("src.rag.generation.llm_groq._session.post")
    result = answer_with_groq("question", "context")
    assert result.error == "misconfigured"
    post.assert_not_called()


def test_success_returns_text_and_usage(mock_groq_success):
    mock_groq_success("The answer. [[cite:1]]", usage={"total_tokens": 10})
    result = answer_with_groq("question", "context")
    assert result.text == "The answer. [[cite:1]]"
    assert result.error is None
    assert result.usage == {"total_tokens": 10}


def test_non_200_returns_generic_message(mock_groq_error):
    mock_groq_error(status_code=500, text="internal error")
    result = answer_with_groq("question", "context")
    assert result.error == "upstream_error"
    assert "trouble generating" in result.text.lower()


def test_timeout_returns_timeout_specific_message(mock_groq_exception):
    mock_groq_exception(requests.Timeout("timed out"))
    result = answer_with_groq("question", "context")
    assert result.error == "timeout"
    assert "too long" in result.text.lower()


def test_other_request_exception(mock_groq_exception):
    mock_groq_exception(requests.ConnectionError("boom"))
    result = answer_with_groq("question", "context")
    assert result.error == "request_failed"


def test_payload_shape(mock_groq_success):
    post = mock_groq_success("ok")
    answer_with_groq("What is X?", "some context", mode="chat")
    _, kwargs = post.call_args
    assert kwargs["headers"]["Authorization"].startswith("Bearer ")
    assert isinstance(kwargs["timeout"], tuple)
    payload = kwargs["json"]
    assert payload["messages"][0]["role"] == "system"
    assert "some context" in payload["messages"][1]["content"]
    assert "response_format" not in payload


def test_json_mode_sets_response_format(mock_groq_success):
    post = mock_groq_success('{"questions": []}')
    answer_with_groq("prompt", "context", mode="interview_generate", json_mode=True)
    _, kwargs = post.call_args
    assert kwargs["json"]["response_format"] == {"type": "json_object"}


def test_advisor_mode_injects_guidance(mock_groq_success):
    post = mock_groq_success("ok")
    answer_with_groq("question", "context", mode="advisor")
    _, kwargs = post.call_args
    assert "Advisor mode" in kwargs["json"]["messages"][1]["content"]
