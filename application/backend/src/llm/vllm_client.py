from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any

from llm.base_client import BaseLLMClient, ChatMessage


def _chat_endpoint(base_url: str) -> str:
    """Return the /chat/completions URL regardless of whether base_url
    already includes /v1 or not."""
    url = base_url.rstrip("/")
    if url.endswith("/v1"):
        return url + "/chat/completions"
    return url + "/v1/chat/completions"


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
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    return json.dumps(body).encode()


class VllmClient(BaseLLMClient):
    """HTTP client for vLLM and any OpenAI-compatible /v1/chat/completions
    endpoint (LM Studio, LocalAI, Ollama in OpenAI-compat mode, etc.).

    Set config.api_key to include an Authorization: Bearer header.
    Leave it None for unauthenticated local servers.
    """

    def _raw_completion(
        self,
        messages: list[ChatMessage],
        temperature: float,
        max_tokens: int,
        timeout: float,
    ) -> str:
        url = _chat_endpoint(self.config.base_url)
        payload = _build_payload(
            self.config.model, messages, temperature, max_tokens
        )
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
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
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as exc:
            raise RuntimeError(
                f"Unexpected response shape from {url}: {data}"
            ) from exc
