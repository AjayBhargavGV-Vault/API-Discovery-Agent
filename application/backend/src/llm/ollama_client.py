from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any

from llm.base_client import BaseLLMClient, ChatMessage


def _build_payload(
    model: str,
    messages: list[ChatMessage],
    temperature: float,
    max_tokens: int,
) -> bytes:
    body: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": m.role, "content": m.content} for m in messages
        ],
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
        },
    }
    return json.dumps(body).encode()


class OllamaClient(BaseLLMClient):
    """HTTP client for the Ollama /api/chat endpoint."""

    def _raw_completion(
        self,
        messages: list[ChatMessage],
        temperature: float,
        max_tokens: int,
        timeout: float,
    ) -> str:
        url = self.config.base_url.rstrip("/") + "/api/chat"
        payload = _build_payload(
            self.config.model, messages, temperature, max_tokens
        )
        headers: dict[str, str] = {"Content-Type": "application/json"}
        headers.update(self.config.extra_headers)

        req = urllib.request.Request(url, data=payload, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read())
        except urllib.error.HTTPError as exc:
            raise RuntimeError(
                f"HTTP {exc.code} from {url}: {exc.read().decode()}"
            ) from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"Connection error to {url}: {exc.reason}"
            ) from exc

        try:
            return data["message"]["content"]
        except (KeyError, TypeError) as exc:
            raise RuntimeError(
                f"Unexpected Ollama response from {url}: {data}"
            ) from exc
