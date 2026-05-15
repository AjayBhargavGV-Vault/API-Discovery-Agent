from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

_log = logging.getLogger(__name__)


@dataclass
class ChatMessage:
    role: str
    content: str


@dataclass
class LLMConfig:
    model: str
    base_url: str
    temperature: float = 0.1
    max_tokens: int = 2048
    timeout: float = 60.0
    max_retries: int = 3
    retry_delay: float = 1.0
    # Set for openai_compatible backend; leave None for vLLM/Ollama
    api_key: str | None = None
    extra_headers: dict[str, str] = field(default_factory=dict)


class BaseLLMClient(ABC):
    def __init__(self, config: LLMConfig) -> None:
        self.config = config

    @abstractmethod
    def _raw_completion(
        self,
        messages: list[ChatMessage],
        temperature: float,
        max_tokens: int,
        timeout: float,
    ) -> str:
        """Return the assistant reply text from the model."""

    # ------------------------------------------------------------------
    # Public instance methods
    # ------------------------------------------------------------------

    def chat_completion(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        timeout: float | None = None,
    ) -> str:
        t = temperature if temperature is not None else self.config.temperature
        m = max_tokens if max_tokens is not None else self.config.max_tokens
        to = timeout if timeout is not None else self.config.timeout

        last_exc: Exception | None = None
        for attempt in range(1, self.config.max_retries + 1):
            try:
                return self._raw_completion(messages, t, m, to)
            except Exception as exc:
                last_exc = exc
                _log.warning(
                    "LLM call failed (attempt %d/%d): %s",
                    attempt,
                    self.config.max_retries,
                    exc,
                )
                if attempt < self.config.max_retries:
                    time.sleep(self.config.retry_delay * attempt)
        raise RuntimeError(
            f"LLM call failed after {self.config.max_retries} attempts"
        ) from last_exc

    def json_completion(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        timeout: float | None = None,
        retries: int | None = None,
    ) -> Any:
        """Chat completion that retries until valid JSON is returned.

        If *retries* is given it overrides config.max_retries for this call.
        On each failed parse attempt a correction message is appended so the
        model can self-correct.
        """
        from llm.json_utils import extract_json

        effective_retries = (
            retries if retries is not None else self.config.max_retries
        )
        working = list(messages)
        last_raw = ""

        for attempt in range(1, effective_retries + 1):
            last_raw = self.chat_completion(
                working,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout,
            )
            try:
                return extract_json(last_raw)
            except ValueError:
                _log.warning(
                    "JSON parse failed (attempt %d/%d). Raw: %.200s",
                    attempt,
                    effective_retries,
                    last_raw,
                )
                if attempt < effective_retries:
                    working = working + [
                        ChatMessage(role="assistant", content=last_raw),
                        ChatMessage(
                            role="user",
                            content=(
                                "Your previous response was not valid JSON. "
                                "Reply with ONLY a valid JSON object, "
                                "no markdown fences, no explanation."
                            ),
                        ),
                    ]

        raise ValueError(
            f"Could not obtain valid JSON after {effective_retries} "
            f"attempt(s). Last response: {last_raw[:300]!r}"
        )
