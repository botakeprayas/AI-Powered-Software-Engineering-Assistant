"""LLM service: one interface, two providers.

OllamaLLM talks to a local Ollama server (free, offline, no API key) using only the Python
standard library. MockLLM is a clearly labeled FAKE for tests and demos.
"""

from __future__ import annotations

import http.client
import json
import logging
import re
import time
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_OLLAMA_MODEL = "qwen2.5-coder:7b"
LOCAL_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})


class LLMError(RuntimeError):
    """Base class. str(error) is a message that is safe to show to the user."""

    retryable = False


class LLMConnectionError(LLMError):
    """The server cannot be reached (usually: Ollama is not running)."""

    retryable = True


class LLMTimeoutError(LLMError):
    retryable = True


class LLMServerError(LLMError):
    """Temporary server-side problem (HTTP 5xx / 429)."""

    retryable = True


class LLMModelNotFoundError(LLMError):
    """The model name is not installed (it was never pulled)."""


class LLMResponseError(LLMError):
    """The server answered, but not with a usable response."""


@dataclass(frozen=True)
class LLMResponse:
    text: str
    model: str
    provider: str
    is_mock: bool = False
    truncated: bool = False  # the model hit its token limit


@dataclass(frozen=True)
class LLMStatus:
    ok: bool
    message: str


class LLMService(ABC):
    """Everything the app needs from a language model. Swap providers without other changes."""

    @property
    @abstractmethod
    def provider(self) -> str:
        """Short provider name, for example 'ollama' or 'mock'."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """The model this service uses."""

    @property
    def is_mock(self) -> bool:
        return False

    @property
    def is_local(self) -> bool:
        """False when prompts (which contain code) leave this computer."""
        return True

    @property
    def endpoint(self) -> str:
        return ""

    @abstractmethod
    def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float = 0.1,
        max_tokens: int = 800,
    ) -> LLMResponse:
        """Return the model's answer. Raises LLMError subclasses on failure."""

    @abstractmethod
    def check_ready(self) -> LLMStatus:
        """Never raises: reports whether the service can answer right now."""


class OllamaLLM(LLMService):
    def __init__(
        self,
        base_url: str = DEFAULT_OLLAMA_URL,
        model: str = DEFAULT_OLLAMA_MODEL,
        timeout: float = 120.0,
        max_retries: int = 2,
        num_ctx: int = 8192,
        backoff_base: float = 1.0,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        parsed = urlparse(base_url)
        if parsed.scheme not in ("http", "https") or not parsed.hostname:
            raise ValueError(
                f"OLLAMA_BASE_URL must look like {DEFAULT_OLLAMA_URL}, got {base_url!r}"
            )
        if not model.strip():
            raise ValueError("OLLAMA_MODEL must not be empty.")
        if max_retries < 0:
            raise ValueError("LLM_MAX_RETRIES must be 0 or more.")
        if timeout <= 0:
            raise ValueError("LLM_TIMEOUT_SECONDS must be greater than 0.")
        self._base = base_url.rstrip("/")
        self._host = parsed.hostname
        self._model = model.strip()
        self._timeout = float(timeout)
        self._max_retries = max_retries
        self._num_ctx = num_ctx
        self._backoff_base = backoff_base
        self._sleep = sleep
        # A system-wide Windows proxy would otherwise intercept calls to localhost.
        handlers = [urllib.request.ProxyHandler({})] if self.is_local else []
        self._opener = urllib.request.build_opener(*handlers)

    @property
    def provider(self) -> str:
        return "ollama"

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def is_local(self) -> bool:
        return self._host in LOCAL_HOSTS

    @property
    def endpoint(self) -> str:
        return self._base

    # ---- messages shown to the user -------------------------------------------------
    def _connection_message(self) -> str:
        return (
            f"Cannot reach Ollama at {self._base}. Is Ollama running? Open the Ollama app "
            "from the Start menu (llama icon in the system tray) and try again. If it is "
            "running, try OLLAMA_BASE_URL=http://127.0.0.1:11434 in .env."
        )

    def _timeout_message(self, timeout: float) -> str:
        return (
            f"Ollama did not answer within {timeout:g} seconds. The model may still be "
            "loading, or this computer is too slow for it. Try again, use a smaller model "
            "(qwen2.5-coder:3b), or raise LLM_TIMEOUT_SECONDS in .env."
        )

    def _missing_model_message(self) -> str:
        return (
            f"Model '{self._model}' is not installed in Ollama. "
            f"Run: ollama pull {self._model}"
        )

    # ---- HTTP -----------------------------------------------------------------------
    def _http_error(self, exc: urllib.error.HTTPError) -> LLMError:
        try:
            detail = exc.read().decode("utf-8", errors="replace")[:300]
        except OSError:
            detail = ""
        logger.warning("Ollama answered HTTP %s: %s", exc.code, detail)
        if exc.code == 404 and "model" in detail.lower():
            return LLMModelNotFoundError(self._missing_model_message())
        if exc.code == 404:
            return LLMResponseError(
                f"The server at {self._base} answered HTTP 404. Is it really Ollama?"
            )
        if exc.code == 429 or exc.code >= 500:
            return LLMServerError(f"Ollama had a temporary problem (HTTP {exc.code}).")
        return LLMResponseError(f"Ollama rejected the request (HTTP {exc.code}): {detail}")

    def _call(self, path: str, payload: dict | None, timeout: float) -> dict:
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            self._base + path,
            data=data,
            headers={"Content-Type": "application/json"},
            method="GET" if payload is None else "POST",
        )
        try:
            with self._opener.open(request, timeout=timeout) as response:
                body = response.read()
        except urllib.error.HTTPError as exc:
            raise self._http_error(exc) from exc
        except urllib.error.URLError as exc:
            if isinstance(exc.reason, TimeoutError):
                raise LLMTimeoutError(self._timeout_message(timeout)) from exc
            logger.warning("Cannot reach Ollama at %s: %s", self._base, exc.reason)
            raise LLMConnectionError(self._connection_message()) from exc
        except TimeoutError as exc:
            raise LLMTimeoutError(self._timeout_message(timeout)) from exc
        except (OSError, http.client.HTTPException) as exc:
            logger.warning("Connection to Ollama failed: %s", exc)
            raise LLMConnectionError(self._connection_message()) from exc
        try:
            parsed = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise LLMResponseError("Ollama returned a response that is not valid JSON.") from exc
        if not isinstance(parsed, dict):
            raise LLMResponseError("Ollama returned an unexpected response.")
        return parsed

    def _parse_chat(self, data: dict) -> LLMResponse:
        if "error" in data:
            error = str(data["error"])
            if "not found" in error.lower():
                raise LLMModelNotFoundError(self._missing_model_message())
            raise LLMResponseError(f"Ollama reported an error: {error[:200]}")
        message = data.get("message")
        text = message.get("content") if isinstance(message, dict) else None
        if not isinstance(text, str) or not text.strip():
            raise LLMResponseError(
                "The model returned an empty answer. Try again, or try a different model."
            )
        return LLMResponse(
            text=text.strip(),
            model=self._model,
            provider="ollama",
            truncated=data.get("done_reason") == "length",
        )

    # ---- public API -----------------------------------------------------------------
    def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float = 0.1,
        max_tokens: int = 800,
    ) -> LLMResponse:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        payload = {
            "model": self._model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_ctx": self._num_ctx,
                "num_predict": max_tokens,
            },
        }
        attempts = self._max_retries + 1
        for attempt in range(attempts):
            try:
                return self._parse_chat(self._call("/api/chat", payload, self._timeout))
            except LLMError as exc:
                if not exc.retryable or attempt == attempts - 1:
                    raise
                delay = self._backoff_base * (2**attempt)
                logger.warning(
                    "LLM attempt %d/%d failed (%s). Retrying in %.1fs",
                    attempt + 1, attempts, type(exc).__name__, delay,
                )
                self._sleep(delay)
        raise LLMResponseError("No attempt was made.")  # only reachable if attempts == 0

    def check_ready(self) -> LLMStatus:
        try:
            data = self._call("/api/tags", None, min(self._timeout, 5.0))
        except LLMError as exc:
            return LLMStatus(False, str(exc))
        models = data.get("models", [])
        names = {m.get("name") for m in models if isinstance(m, dict)}
        wanted = self._model if ":" in self._model else f"{self._model}:latest"
        if wanted in names:
            return LLMStatus(True, f"Ollama is running and model '{self._model}' is installed.")
        installed = ", ".join(sorted(str(n) for n in names if n)[:5]) or "none"
        return LLMStatus(
            False,
            f"Ollama is running, but model '{self._model}' is not installed "
            f"(installed: {installed}). Run: ollama pull {self._model}",
        )


_EVIDENCE_HEADER = re.compile(r"^\[(E\d+)\] ([^|]+?) \| ([^|]+?) \| ([^|]+?) \|", re.MULTILINE)


class MockLLM(LLMService):
    """MOCK PROVIDER: not a real model. Builds canned answers from the prompt it receives.

    behavior: grounded | abstain | uncited | bad_citation | unstructured.
    Pass error=<LLMError> to make generate() fail. `calls` and `prompts` record usage.
    """

    BEHAVIORS = ("grounded", "abstain", "uncited", "bad_citation", "unstructured")

    def __init__(self, behavior: str = "grounded", error: LLMError | None = None) -> None:
        if behavior not in self.BEHAVIORS:
            raise ValueError(f"behavior must be one of {self.BEHAVIORS}")
        self._behavior = behavior
        self._error = error
        self.calls = 0
        self.prompts: list[str] = []

    @property
    def provider(self) -> str:
        return "mock"

    @property
    def model_name(self) -> str:
        return "mock-model"

    @property
    def is_mock(self) -> bool:
        return True

    def check_ready(self) -> LLMStatus:
        return LLMStatus(True, "MOCK provider: no model is used.")

    def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float = 0.1,
        max_tokens: int = 800,
    ) -> LLMResponse:
        self.calls += 1
        self.prompts.append(prompt)
        if self._error is not None:
            raise self._error
        return LLMResponse(text=self._canned(prompt), model=self.model_name,
                           provider="mock", is_mock=True)

    def _canned(self, prompt: str) -> str:
        items = _EVIDENCE_HEADER.findall(prompt)  # (id, location, kind, qualname)
        if self._behavior == "abstain" or not items:
            return (
                "STATUS: INSUFFICIENT_EVIDENCE\nFACTS:\n- none\nINFERENCES:\n- none\n"
                "UNKNOWNS:\n- MOCK ANSWER: the excerpts do not cover this question."
            )
        if self._behavior == "unstructured":
            return "MOCK ANSWER: this is free text that ignores the required format."
        if self._behavior == "uncited":
            return (
                "STATUS: ANSWERED\nFACTS:\n- MOCK ANSWER: this statement has no citation.\n"
                "INFERENCES:\n- none\nUNKNOWNS:\n- none"
            )
        if self._behavior == "bad_citation":
            return (
                "STATUS: ANSWERED\nFACTS:\n- MOCK ANSWER: this cites an excerpt that does "
                "not exist. [E99]\nINFERENCES:\n- none\nUNKNOWNS:\n- none"
            )
        first_id, first_loc, first_kind, first_name = items[0]
        lines = [
            "STATUS: ANSWERED",
            "FACTS:",
            f"- MOCK ANSWER: the most relevant excerpt is {first_name} ({first_kind}) "
            f"in {first_loc}. [{first_id}]",
        ]
        if len(items) > 1:
            second_id, _, _, second_name = items[1]
            lines.append(
                f"- MOCK ANSWER: a second relevant excerpt is {second_name}. [{second_id}]"
            )
        lines += [
            "INFERENCES:",
            f"- MOCK ANSWER: {first_name} is probably what the question is about. [{first_id}]",
            "UNKNOWNS:",
            "- MOCK ANSWER: callers of these functions are not shown in the excerpts.",
        ]
        return "\n".join(lines)