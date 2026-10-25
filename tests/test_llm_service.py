import socket

import pytest
from llm_stub import StubOllama, ok_reply

from core.chat.config import create_llm_from_env, load_qa_config
from services.llm_service import (
    LLMConnectionError,
    LLMModelNotFoundError,
    LLMResponseError,
    LLMServerError,
    LLMTimeoutError,
    MockLLM,
    OllamaLLM,
)

MODEL = "qwen2.5-coder:7b"
ENV_NAMES = (
    "LLM_PROVIDER", "OLLAMA_BASE_URL", "OLLAMA_MODEL", "LLM_TIMEOUT_SECONDS",
    "LLM_MAX_RETRIES", "LLM_NUM_CTX", "QA_MIN_COSINE", "QA_MIN_KEYWORD_SCORE", "QA_MAX_EVIDENCE",
)


def make_llm(url, **kwargs):
    sleeps: list[float] = []
    return OllamaLLM(base_url=url, model=MODEL, sleep=sleeps.append, **kwargs), sleeps


def closed_port_url():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return f"http://127.0.0.1:{sock.getsockname()[1]}"


def test_mock_provider_is_labeled():
    llm = MockLLM()
    assert llm.is_mock and llm.provider == "mock"
    assert "MOCK" in llm.check_ready().message


def test_success_sends_expected_request():
    with StubOllama([ok_reply("hello")]) as stub:
        llm, _ = make_llm(stub.url)
        response = llm.generate("question", system="rules")
    assert (response.text, response.provider, response.is_mock) == ("hello", "ollama", False)
    sent = stub.chat_requests[0]
    assert sent["model"] == MODEL
    assert sent["stream"] is False
    assert sent["options"]["num_ctx"] == 8192
    assert [m["role"] for m in sent["messages"]] == ["system", "user"]


def test_model_not_pulled_gives_pull_instruction_and_no_retry():
    body = {"error": f"model '{MODEL}' not found"}
    with StubOllama([(404, body, 0)]) as stub:
        llm, sleeps = make_llm(stub.url)
        with pytest.raises(LLMModelNotFoundError, match=f"ollama pull {MODEL}"):
            llm.generate("q")
    assert len(stub.chat_requests) == 1
    assert sleeps == []


def test_server_error_is_retried_then_succeeds():
    with StubOllama([(500, {"error": "boom"}, 0), ok_reply("recovered")]) as stub:
        llm, sleeps = make_llm(stub.url)
        assert llm.generate("q").text == "recovered"
    assert len(stub.chat_requests) == 2
    assert sleeps == [1.0]


def test_server_error_gives_up_after_backoff():
    with StubOllama([(500, {"error": "boom"}, 0)]) as stub:
        llm, sleeps = make_llm(stub.url)
        with pytest.raises(LLMServerError):
            llm.generate("q")
    assert len(stub.chat_requests) == 3
    assert sleeps == [1.0, 2.0]  # exponential backoff


def test_timeout_is_reported_clearly():
    with StubOllama([(200, {"message": {"content": "late"}}, 1.0)]) as stub:
        llm, _ = make_llm(stub.url, timeout=0.3, max_retries=0)
        with pytest.raises(LLMTimeoutError, match="LLM_TIMEOUT_SECONDS"):
            llm.generate("q")


def test_ollama_not_running_gives_clear_message_after_retries():
    llm, sleeps = make_llm(closed_port_url())
    with pytest.raises(LLMConnectionError, match="Is Ollama running"):
        llm.generate("q")
    assert sleeps == [1.0, 2.0]


def test_empty_answer_is_an_error():
    with StubOllama([ok_reply("   ")]) as stub:
        llm, _ = make_llm(stub.url)
        with pytest.raises(LLMResponseError, match="empty"):
            llm.generate("q")


def test_length_cutoff_is_flagged():
    with StubOllama([ok_reply("partial text", done_reason="length")]) as stub:
        llm, _ = make_llm(stub.url)
        assert llm.generate("q").truncated is True


def test_check_ready_reports_each_situation():
    with StubOllama(tags=[MODEL]) as stub:
        assert make_llm(stub.url)[0].check_ready().ok
    with StubOllama(tags=["llama3.2:3b"]) as stub:
        status = make_llm(stub.url)[0].check_ready()
        assert not status.ok
        assert f"ollama pull {MODEL}" in status.message
    status = make_llm(closed_port_url())[0].check_ready()
    assert not status.ok
    assert "Ollama" in status.message


@pytest.mark.parametrize(
    "kwargs",
    [{"base_url": "localhost:11434"}, {"model": " "}, {"max_retries": -1}, {"timeout": 0}],
)
def test_invalid_settings_are_rejected(kwargs):
    with pytest.raises(ValueError):
        OllamaLLM(**kwargs)


def test_is_local_detection():
    assert OllamaLLM(base_url="http://localhost:11434").is_local
    assert OllamaLLM(base_url="http://127.0.0.1:11434").is_local
    assert not OllamaLLM(base_url="http://192.168.1.20:11434").is_local


def test_factory_defaults_and_overrides(monkeypatch):
    for name in ENV_NAMES:
        monkeypatch.delenv(name, raising=False)
    llm = create_llm_from_env(env_file=None)
    assert (llm.provider, llm.model_name) == ("ollama", MODEL)
    monkeypatch.setenv("OLLAMA_MODEL", "qwen2.5-coder:3b")
    assert create_llm_from_env(env_file=None).model_name == "qwen2.5-coder:3b"
    assert create_llm_from_env(provider_override="mock", env_file=None).is_mock


@pytest.mark.parametrize(
    ("name", "value"),
    [("LLM_PROVIDER", "openai"), ("LLM_TIMEOUT_SECONDS", "soon"), ("LLM_MAX_RETRIES", "-1")],
)
def test_factory_rejects_bad_values(monkeypatch, name, value):
    for env_name in ENV_NAMES:
        monkeypatch.delenv(env_name, raising=False)
    monkeypatch.setenv(name, value)
    with pytest.raises(ValueError, match=name):
        create_llm_from_env(env_file=None)


def test_qa_config_reads_environment(monkeypatch):
    for name in ENV_NAMES:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("QA_MIN_COSINE", "0.4")
    assert load_qa_config(env_file=None).min_cosine == 0.4