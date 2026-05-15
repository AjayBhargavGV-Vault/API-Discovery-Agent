from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from chat.schemas import AgentResult
from chat.schemas import ValidationResult  # canonical type; re-exported below
from llm.base_client import ChatMessage
from llm.json_utils import safe_extract_json
from llm.model_registry import get_model_client
from prompts.validation_prompts import (
    VALIDATION_SYSTEM,
    VALIDATION_USER_TEMPLATE,
)

_log = logging.getLogger(__name__)

_SOURCES_CHAR_LIMIT = 4_000
_CONTEXT_CHAR_LIMIT = 4_000

_INSUFFICIENT = (
    "The retrieved API catalog does not contain enough information "
    "to answer this question."
)


# ---------------------------------------------------------------------------
# Internal legacy result (field names expected by chat_engine.py)
# ---------------------------------------------------------------------------

@dataclass
class _LegacyResult:
    valid: bool
    issues: list[str]
    confidence: float


# ---------------------------------------------------------------------------
# Source / context formatters
# ---------------------------------------------------------------------------

def _format_source(src: Any, index: int) -> str:
    """Return a one-or-two-line summary of a single source entry."""
    if isinstance(src, dict):
        method = (src.get("method") or "").upper()
        endpoint = src.get("endpoint") or src.get("path") or ""
        summary = (
            src.get("summary")
            or src.get("title")
            or src.get("description")
            or ""
        )
        op_id = src.get("operation_id") or src.get("id") or ""
        auth = src.get("auth_scheme") or src.get("auth") or ""
        schema = src.get("schema_name") or src.get("schema") or ""
    else:
        method = (getattr(src, "method", "") or "").upper()
        endpoint = getattr(src, "endpoint", "") or ""
        summary = (
            getattr(src, "summary", "")
            or getattr(src, "description", "")
            or ""
        )
        op_id = getattr(src, "operation_id", "") or ""
        auth = getattr(src, "auth_scheme", "") or ""
        schema = getattr(src, "schema_name", "") or ""

    label = f"{method} {endpoint}".strip() or "(unnamed)"
    line = f"{index}. {label}"
    if op_id:
        line += f"  [operationId: {op_id}]"
    if summary:
        line += f"\n   {str(summary)[:150]}"
    if auth:
        line += f"\n   auth: {str(auth)[:80]}"
    if schema:
        line += f"\n   schema: {str(schema)[:80]}"
    return line


def _format_sources(sources: list) -> str:
    if not sources:
        return "(no sources provided)"
    parts = [_format_source(s, i) for i, s in enumerate(sources[:20], 1)]
    text = "\n".join(parts)
    if len(text) > _SOURCES_CHAR_LIMIT:
        text = text[:_SOURCES_CHAR_LIMIT] + "\n[sources truncated]"
    return text


def _format_agent_results(agent_results: list) -> str:
    if not agent_results:
        return "(no agent results)"
    parts: list[str] = []
    for r in agent_results:
        if isinstance(r, AgentResult):
            if r.success:
                try:
                    data_str = json.dumps(r.data, indent=2)
                except (TypeError, ValueError):
                    data_str = str(r.data)
                if len(data_str) > 2_000:
                    data_str = data_str[:2_000] + "\n... [truncated]"
                parts.append(f"[{r.agent_name}]\n{data_str}")
            else:
                parts.append(
                    f"[{r.agent_name}] ERROR: {r.error or 'no details'}"
                )
        elif isinstance(r, dict):
            try:
                data_str = json.dumps(r, indent=2)
            except (TypeError, ValueError):
                data_str = str(r)
            if len(data_str) > 2_000:
                data_str = data_str[:2_000] + "\n... [truncated]"
            parts.append(f"[context]\n{data_str}")

    text = "\n\n".join(parts)
    if len(text) > _CONTEXT_CHAR_LIMIT:
        text = text[:_CONTEXT_CHAR_LIMIT] + "\n[context truncated]"
    return text or "(no agent results)"


# ---------------------------------------------------------------------------
# ValidationAgent
# ---------------------------------------------------------------------------

class ValidationAgent:
    """Validates a draft answer against retrieved API sources.

    Primary method : validate_response(answer, sources, agent_results)
    Legacy method  : validate(answer, context_dict)  — kept for chat_engine.py
    """

    def __init__(self, env_path: Path | None = None) -> None:
        self._client = get_model_client("validation", env_path=env_path)
        self._fallback_client = get_model_client(
            "validation_fallback", env_path=env_path
        )

    # ------------------------------------------------------------------
    # Primary public API
    # ------------------------------------------------------------------

    def validate_response(
        self,
        answer: str,
        sources: list,
        agent_results: list,
    ) -> ValidationResult:
        """Run all 7 grounding checks and return a ValidationResult.

        Check 7 (non-empty answer) is enforced before the LLM call.
        Checks 1-6 are performed by the LLM at temperature 0.0.
        Falls back to a pass-through result when all model calls fail.
        """
        if not answer.strip():
            return ValidationResult(
                is_grounded=False,
                unsupported_claims=["Answer is empty."],
                confidence_score=0.0,
                final_answer=_INSUFFICIENT,
                model_name=None,
            )

        sources_text = _format_sources(sources)
        context_text = _format_agent_results(agent_results)
        user_content = VALIDATION_USER_TEMPLATE.format(
            answer=answer,
            sources_text=sources_text,
            context_text=context_text,
        )
        messages = [
            ChatMessage(role="system", content=VALIDATION_SYSTEM),
            ChatMessage(role="user", content=user_content),
        ]
        raw, model_used = self._call_with_fallback(messages)

        if raw is None:
            _log.warning(
                "Validation LLM unavailable; passing answer through"
            )
            return ValidationResult(
                is_grounded=True,
                unsupported_claims=[],
                confidence_score=0.5,
                final_answer=answer,
                model_name=None,
            )

        parsed = safe_extract_json(raw)
        if isinstance(parsed, dict):
            return self._build_result(parsed, answer, model_used)

        _log.warning(
            "Validation returned non-dict JSON; passing through. "
            "Raw: %.200s",
            raw,
        )
        return ValidationResult(
            is_grounded=True,
            unsupported_claims=[],
            confidence_score=0.5,
            final_answer=answer,
            model_name=model_used,
        )

    # ------------------------------------------------------------------
    # Legacy method — kept for chat_engine.py compatibility
    # ------------------------------------------------------------------

    def validate(
        self,
        answer: str,
        context: dict[str, Any],
    ) -> _LegacyResult:
        """Thin wrapper over validate_response for the old call-site."""
        ops = context.get("recommended_operations") or []
        sources: list[Any] = [
            {
                "method": op.get("method"),
                "endpoint": op.get("path") or op.get("endpoint"),
                "summary": op.get("title") or op.get("summary"),
                "operation_id": op.get("operation_id"),
            }
            for op in ops
            if op.get("method") or op.get("path")
        ]
        result = self.validate_response(
            answer, sources=sources, agent_results=[]
        )
        return _LegacyResult(
            valid=result.is_grounded,
            issues=result.unsupported_claims,
            confidence=result.confidence_score,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _call_with_fallback(
        self,
        messages: list[ChatMessage],
        max_tokens: int = 512,
    ) -> tuple[str | None, str | None]:
        """Try primary then fallback client. Returns (text, model_name)."""
        for client in (self._client, self._fallback_client):
            model_name = client.config.model
            try:
                return (
                    client.chat_completion(
                        messages,
                        temperature=0.0,
                        max_tokens=max_tokens,
                    ),
                    model_name,
                )
            except Exception as exc:
                _log.warning(
                    "Validation LLM failed with model '%s': %s",
                    model_name,
                    exc,
                )
        return None, None

    def _build_result(
        self,
        data: dict[str, Any],
        original_answer: str,
        model_used: str | None,
    ) -> ValidationResult:
        """Construct a ValidationResult from validated LLM JSON output."""
        try:
            is_grounded = bool(data.get("is_grounded", True))
            claims = list(data.get("unsupported_claims") or [])
            score = float(data.get("confidence_score", 0.5))
            final = str(data.get("final_answer", original_answer) or "").strip()

            # Enforce non-empty final_answer (check 7 guard)
            if not final:
                final = _INSUFFICIENT if not is_grounded else original_answer

            # If hallucination is detected, ensure the answer is the safer one
            if not is_grounded and final == original_answer:
                _log.warning(
                    "is_grounded=False but final_answer unchanged; "
                    "replacing with insufficient-context message"
                )
                final = _INSUFFICIENT

            return ValidationResult(
                is_grounded=is_grounded,
                unsupported_claims=claims,
                confidence_score=max(0.0, min(1.0, score)),
                final_answer=final,
                model_name=model_used,
            )
        except Exception as exc:
            _log.warning(
                "Could not parse validation result: %s. "
                "Passing answer through.",
                exc,
            )
            return ValidationResult(
                is_grounded=True,
                unsupported_claims=[],
                confidence_score=0.5,
                final_answer=original_answer,
                model_name=model_used,
            )
