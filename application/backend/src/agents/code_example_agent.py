from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from agent_tools.codegen_tools import (
    CODE_EXAMPLE_SYSTEM,
    build_code_generation_prompt,
    extract_code_block,
    generate_curl,
    generate_node_example,
    generate_python_example,
    generate_sample_payload,
)
from chat.schemas import AgentResult, APIOperationDetails
from llm.base_client import ChatMessage
from llm.model_registry import get_model_client

_log = logging.getLogger(__name__)

_AGENT_NAME = "code_example"

_PROMPT_BUILDERS = {
    "curl": generate_curl,
    "python": generate_python_example,
    "node": generate_node_example,
    "nodejs": generate_node_example,
    "javascript": generate_node_example,
    "payload": generate_sample_payload,
    "json": generate_sample_payload,
}


class CodeExampleAgent:
    """Generates grounded API code examples using DeepSeek-Coder-V2-Lite.

    Falls back to qwen2.5-coder-14b-instruct then to the global fallback model
    if the primary model is unavailable at inference time.
    """

    def __init__(self, env_path: Path | None = None) -> None:
        self._client = get_model_client("code_example", env_path=env_path)
        self._fallback_client = get_model_client(
            "code_fallback", env_path=env_path
        )

    # ------------------------------------------------------------------
    # Primary public API
    # ------------------------------------------------------------------

    def run(
        self,
        operation_details: APIOperationDetails,
        language: str = "python",
    ) -> AgentResult:
        """Generate a grounded code example for *operation_details*.

        Returns AgentResult with data["code"] on success, or success=False
        with an error string when the language is unsupported or all
        model calls fail.
        """
        lang_key = language.lower().strip()
        prompt_builder = _PROMPT_BUILDERS.get(lang_key)

        if prompt_builder is None:
            return AgentResult(
                agent_name=_AGENT_NAME,
                model_name=None,
                success=False,
                data={},
                error=(
                    f"Unsupported language '{language}'. "
                    f"Supported: {sorted(_PROMPT_BUILDERS)}."
                ),
            )

        messages = [
            ChatMessage(role="system", content=CODE_EXAMPLE_SYSTEM),
            ChatMessage(
                role="user", content=prompt_builder(operation_details)
            ),
        ]
        raw, model_used = self._call_with_fallback(messages)

        if raw is None:
            return AgentResult(
                agent_name=_AGENT_NAME,
                model_name=model_used,
                success=False,
                data={},
                error="Code generation failed after all fallback attempts.",
            )

        return AgentResult(
            agent_name=_AGENT_NAME,
            model_name=model_used,
            success=True,
            data={
                "operation_id": operation_details.operation_id,
                "language": lang_key,
                "code": extract_code_block(raw),
            },
        )

    # ------------------------------------------------------------------
    # Legacy API (kept for chat_engine.py compatibility)
    # ------------------------------------------------------------------

    def generate(
        self,
        operations: list[dict[str, Any]],
        query: str,
        language: str = "python",
        base_url: str = "https://api.example.com",
    ) -> list[dict[str, Any]]:
        """Return a list of {operation_id, language, code} dicts."""
        results: list[dict[str, Any]] = []
        for op in operations:
            op_id = op.get("operation_id") or op.get("id") or ""
            prompt = build_code_generation_prompt(op, language, base_url)
            messages = [ChatMessage(role="user", content=prompt)]
            raw, _ = self._call_with_fallback(messages)
            if raw is None:
                _log.warning("Code generation failed for %s", op_id)
                continue
            results.append(
                {
                    "operation_id": op_id,
                    "language": language,
                    "code": extract_code_block(raw),
                }
            )
        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _call_with_fallback(
        self,
        messages: list[ChatMessage],
    ) -> tuple[str | None, str | None]:
        """Try primary then fallback client. Returns (response, model_name)."""
        for client in (self._client, self._fallback_client):
            model_name = client.config.model
            try:
                return (
                    client.chat_completion(
                        messages, temperature=0.1, max_tokens=512
                    ),
                    model_name,
                )
            except Exception as exc:
                _log.warning(
                    "Code generation failed with model '%s': %s",
                    model_name,
                    exc,
                )
        return None, None
