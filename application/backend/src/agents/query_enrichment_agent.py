from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from llm.base_client import ChatMessage
from llm.json_utils import safe_extract_json
from llm.model_registry import get_model_client
from prompts.query_enrichment_prompts import (
    QUERY_ENRICHMENT_SYSTEM,
    QUERY_ENRICHMENT_USER_TEMPLATE,
)

_log = logging.getLogger(__name__)


@dataclass
class QueryEnrichmentResult:
    original_query: str
    enriched_query: str
    detected_intent: str
    entities: list[str]
    suggested_agents: list[str]
    confidence_score: float

    @classmethod
    def passthrough(cls, query: str) -> "QueryEnrichmentResult":
        """Safe fallback when enrichment is unavailable."""
        return cls(
            original_query=query,
            enriched_query=query,
            detected_intent="unknown",
            entities=[],
            suggested_agents=["semantic_retriever"],
            confidence_score=0.0,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any], original: str) -> "QueryEnrichmentResult":
        enriched = str(data.get("enriched_query", original)).strip()
        return cls(
            original_query=str(data.get("original_query", original)),
            enriched_query=enriched if enriched else original,
            detected_intent=str(data.get("detected_intent", "unknown")),
            entities=list(data.get("entities") or []),
            suggested_agents=list(data.get("suggested_agents") or ["semantic_retriever"]),
            confidence_score=float(data.get("confidence_score", 0.5)),
        )


class QueryEnrichmentAgent:
    """Rewrites a vague developer question into a retrieval-optimized API discovery query.

    Returns a QueryEnrichmentResult with structured metadata for downstream agents.
    Falls back to a passthrough result when the LLM call fails.
    """

    def __init__(self, env_path: Path | None = None) -> None:
        self._client = get_model_client("query_enrichment", env_path=env_path)

    def enrich(
        self,
        user_message: str,
        chat_history: list,
        last_operations: list,
    ) -> QueryEnrichmentResult:
        history_text = (
            "\n".join(str(t) for t in chat_history[-6:]) if chat_history else "(none)"
        )
        ops_text = (
            "\n".join(f"- {op}" for op in last_operations[:10])
            if last_operations
            else "(none)"
        )
        user_content = QUERY_ENRICHMENT_USER_TEMPLATE.format(
            user_message=user_message,
            chat_history=history_text,
            last_operations=ops_text,
        )
        messages = [
            ChatMessage(role="system", content=QUERY_ENRICHMENT_SYSTEM),
            ChatMessage(role="user", content=user_content),
        ]
        try:
            raw = self._client.chat_completion(
                messages, temperature=0.2, max_tokens=512
            )
            parsed = safe_extract_json(raw)
            if isinstance(parsed, dict):
                return QueryEnrichmentResult.from_dict(parsed, user_message)
            _log.warning(
                "Query enrichment returned non-dict JSON; using passthrough. Raw: %.200s",
                raw,
            )
            return QueryEnrichmentResult.passthrough(user_message)
        except Exception as exc:
            _log.warning("Query enrichment failed; using passthrough: %s", exc)
            return QueryEnrichmentResult.passthrough(user_message)
