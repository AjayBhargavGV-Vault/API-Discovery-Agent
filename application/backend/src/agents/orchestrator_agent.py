from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from agents.query_enrichment_agent import QueryEnrichmentResult
from chat.schemas import AgentResult, AgentTask, ChatResponse
from llm.base_client import ChatMessage
from llm.json_utils import safe_extract_json
from llm.model_registry import get_model_client
from prompts.orchestrator_prompts import (
    ORCHESTRATOR_PLANNING_SYSTEM,
    ORCHESTRATOR_PLANNING_USER_TEMPLATE,
    ORCHESTRATOR_SYNTHESIS_SYSTEM,
    ORCHESTRATOR_SYNTHESIS_USER_TEMPLATE,
    ORCHESTRATOR_SYSTEM,
    ORCHESTRATOR_USER_TEMPLATE,
)

_log = logging.getLogger(__name__)

_CONTEXT_CHAR_LIMIT = 8_000
_RESULTS_CHAR_LIMIT = 8_000

# ---------------------------------------------------------------------------
# Deterministic routing fallback
# ---------------------------------------------------------------------------

# Each entry: (keyword_tuple, agent_name_list)
_INTENT_ROUTING: list[tuple[tuple[str, ...], list[str]]] = [
    (
        ("code", "curl", "example", "snippet", "python", "node", "javascript"),
        ["api_spec", "code_example"],
    ),
    (
        ("lifecycle", "relationship", "related", "depend", "flow", "link"),
        ["graph_retriever", "api_spec"],
    ),
    (
        ("endpoint", "schema", "auth", "authentication", "field", "param"),
        ["api_spec"],
    ),
]
_DEFAULT_ROUTE: list[str] = [
    "semantic_retriever", "graph_retriever", "api_spec"
]

_AGENT_MODELS: dict[str, str | None] = {
    "semantic_retriever": None,
    "graph_retriever": None,
    "api_spec": None,
    "code_example": "deepseek-coder-v2-lite-instruct",
    "query_enrichment": "qwen2.5-14b-instruct",
}
_AGENT_TASK_TYPES: dict[str, str] = {
    "semantic_retriever": "semantic_search",
    "graph_retriever": "graph_traversal",
    "api_spec": "spec_lookup",
    "code_example": "code_generation",
    "query_enrichment": "query_rewrite",
}


# ---------------------------------------------------------------------------
# Context helpers (shared)
# ---------------------------------------------------------------------------

def _summarise_context(context: dict[str, Any]) -> str:
    """Flatten a retrieval context dict into a readable text block."""
    ops = context.get("recommended_operations") or []
    schemas = context.get("schemas") or []
    notes = context.get("confidence_notes") or []

    lines: list[str] = []
    if ops:
        lines.append("=== Recommended Operations ===")
        for op in ops[:10]:
            method = op.get("method") or ""
            path = op.get("path") or ""
            title = op.get("title") or op.get("summary") or ""
            desc = op.get("description") or ""
            pk = op.get("product_key") or ""
            lines.append(
                f"- {method} {path}  [{title}]"
                + (f"  product={pk}" if pk else "")
                + (f"\n  {desc[:200]}" if desc else "")
            )

    if schemas:
        lines.append("\n=== Schemas ===")
        for s in schemas[:6]:
            name = s.get("name") or s.get("title") or s.get("id") or ""
            desc = s.get("description") or ""
            lines.append(
                f"- {name}" + (f": {desc[:150]}" if desc else "")
            )

    if notes:
        lines.append("\n=== Retrieval Notes ===")
        lines.extend(f"- {n}" for n in notes)

    text = "\n".join(lines)
    if len(text) > _CONTEXT_CHAR_LIMIT:
        text = text[:_CONTEXT_CHAR_LIMIT] + "\n[context truncated]"
    return text


# ---------------------------------------------------------------------------
# OrchestratorAgent
# ---------------------------------------------------------------------------

class OrchestratorAgent:
    """Main planning and synthesis agent.

    Public surface:
      create_agent_plan     — produce an ordered list of AgentTask objects
      synthesize_response   — merge agent results into a grounded ChatResponse
      synthesize            — legacy single-call synthesis (chat_engine.py)
    """

    def __init__(self, env_path: Path | None = None) -> None:
        self._client = get_model_client("orchestrator", env_path=env_path)
        self._fallback_client = get_model_client(
            "orchestrator_fallback", env_path=env_path
        )

    # ------------------------------------------------------------------
    # create_agent_plan
    # ------------------------------------------------------------------

    def create_agent_plan(
        self,
        enrichment_result: QueryEnrichmentResult,
        user_message: str,
    ) -> list[AgentTask]:
        """Return an ordered list of AgentTask objects for this query.

        Calls the LLM for a structured JSON plan; falls back to
        deterministic routing if the LLM call fails or returns no tasks.
        """
        user_content = ORCHESTRATOR_PLANNING_USER_TEMPLATE.format(
            user_message=user_message,
            enriched_query=enrichment_result.enriched_query,
            detected_intent=enrichment_result.detected_intent,
            entities=(
                ", ".join(enrichment_result.entities) or "(none)"
            ),
            suggested_agents=(
                ", ".join(enrichment_result.suggested_agents) or "(none)"
            ),
            confidence_score=enrichment_result.confidence_score,
        )
        messages = [
            ChatMessage(role="system", content=ORCHESTRATOR_PLANNING_SYSTEM),
            ChatMessage(role="user", content=user_content),
        ]
        raw, _ = self._call_with_fallback(messages, max_tokens=512)
        if raw is not None:
            tasks = self._parse_plan(
                raw, enrichment_result.enriched_query or user_message
            )
            if tasks:
                return tasks

        _log.warning(
            "LLM planning failed or returned no tasks; "
            "using deterministic routing"
        )
        return self._deterministic_plan(enrichment_result, user_message)

    # ------------------------------------------------------------------
    # synthesize_response
    # ------------------------------------------------------------------

    def synthesize_response(
        self,
        user_message: str,
        enrichment_result: QueryEnrichmentResult,
        agent_results: list[AgentResult],
        history: str = "",
    ) -> ChatResponse:
        """Merge agent results into a grounded ChatResponse.

        Falls back to a minimal text answer derived from agent_results
        when the LLM is unavailable.
        """
        deterministic_answer = self._product_catalog_answer(
            user_message, agent_results
        )
        if deterministic_answer:
            sources = [
                src
                for r in agent_results
                for src in (r.sources or [])
            ]
            return ChatResponse(
                answer=deterministic_answer,
                conversation_id="",
                sources=sources,
                agent_trace=[
                    {
                        "agent_name": r.agent_name,
                        "success": r.success,
                        "model_name": r.model_name,
                    }
                    for r in agent_results
                ],
                guardrail_status="ok",
                confidence_score=enrichment_result.confidence_score,
            )

        results_text = self._format_agent_results(agent_results)
        user_content = ORCHESTRATOR_SYNTHESIS_USER_TEMPLATE.format(
            user_message=user_message,
            detected_intent=enrichment_result.detected_intent,
            agent_results_text=results_text,
            history=history or "None",
        )
        messages = [
            ChatMessage(
                role="system", content=ORCHESTRATOR_SYNTHESIS_SYSTEM
            ),
            ChatMessage(role="user", content=user_content),
        ]
        raw, model_used = self._call_with_fallback(
            messages, max_tokens=1024
        )

        answer = raw.strip() if raw else self._fallback_answer(agent_results)
        if not raw:
            _log.error(
                "Synthesis failed for all models; using fallback answer"
            )

        sources = [
            src
            for r in agent_results
            for src in (r.sources or [])
        ]
        agent_trace: list[Any] = [
            {
                "agent_name": r.agent_name,
                "success": r.success,
                "model_name": r.model_name,
            }
            for r in agent_results
        ]

        return ChatResponse(
            answer=answer,
            conversation_id="",
            sources=sources,
            agent_trace=agent_trace,
            guardrail_status="ok",
            confidence_score=enrichment_result.confidence_score,
        )

    # ------------------------------------------------------------------
    # Legacy method — kept for chat_engine.py compatibility
    # ------------------------------------------------------------------

    def synthesize(
        self,
        query: str,
        context: dict[str, Any],
        history: str = "",
    ) -> str:
        context_text = _summarise_context(context)
        user_content = ORCHESTRATOR_USER_TEMPLATE.format(
            query=query,
            context=context_text,
            history=history or "None",
        )
        messages = [
            ChatMessage(role="system", content=ORCHESTRATOR_SYSTEM),
            ChatMessage(role="user", content=user_content),
        ]
        raw, _ = self._call_with_fallback(messages, max_tokens=1024)
        if raw:
            return raw

        _log.error("Orchestrator synthesize failed for all models")
        ops = context.get("recommended_operations") or []
        if ops:
            top = ops[0]
            return (
                f"Based on your query, consider "
                f"{top.get('method', '')} {top.get('path', '')} "
                f"({top.get('title', '')})."
            )
        return (
            "No matching APIs were found in the catalog for your query. "
            "Try rephrasing or narrowing your search."
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _call_with_fallback(
        self,
        messages: list[ChatMessage],
        temperature: float = 0.1,
        max_tokens: int = 1024,
    ) -> tuple[str | None, str | None]:
        """Try primary then fallback client. Returns (text, model_name)."""
        for client in (self._client, self._fallback_client):
            model_name = client.config.model
            try:
                return (
                    client.chat_completion(
                        messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                    ),
                    model_name,
                )
            except Exception as exc:
                _log.warning(
                    "LLM call failed with model '%s': %s", model_name, exc
                )
        return None, None

    def _parse_plan(
        self,
        raw: str,
        fallback_query: str,
    ) -> list[AgentTask]:
        """Parse LLM JSON output into a list of validated AgentTask objects."""
        parsed = safe_extract_json(raw)
        if not isinstance(parsed, list):
            _log.warning(
                "Planning output is not a JSON array. Raw: %.200s", raw
            )
            return []

        tasks: list[AgentTask] = []
        for item in parsed:
            if not isinstance(item, dict):
                continue
            try:
                task = AgentTask(
                    task_id=str(
                        item.get("task_id") or f"task_{len(tasks) + 1}"
                    ),
                    agent_name=str(item.get("agent_name", "")),
                    task_type=str(item.get("task_type", "search")),
                    input=dict(
                        item.get("input") or {"query": fallback_query}
                    ),
                    reason=str(item.get("reason", "")),
                    model_name=item.get("model_name"),
                )
                if task.agent_name:
                    tasks.append(task)
            except Exception as exc:
                _log.warning(
                    "Skipping malformed plan item %s: %s", item, exc
                )

        return tasks

    def _deterministic_plan(
        self,
        enrichment_result: QueryEnrichmentResult,
        user_message: str,
    ) -> list[AgentTask]:
        """Rule-based routing used when LLM planning is unavailable."""
        combined = (
            f"{enrichment_result.detected_intent} {user_message}".lower()
        )

        agent_names: list[str] = []
        for keywords, agents in _INTENT_ROUTING:
            if any(kw in combined for kw in keywords):
                agent_names = agents
                break

        if not agent_names and any(
            kw in combined
            for kw in ("api", "apis", "endpoint", "endpoints", "product")
        ):
            agent_names = ["semantic_retriever", "api_spec"]

        if not agent_names:
            agent_names = (
                list(enrichment_result.suggested_agents) or _DEFAULT_ROUTE
            )

        query = enrichment_result.enriched_query or user_message
        reason = (
            f"Deterministic routing for intent "
            f"'{enrichment_result.detected_intent}'"
        )
        return [
            AgentTask(
                task_id=f"task_{i}",
                agent_name=name,
                task_type=_AGENT_TASK_TYPES.get(name, "search"),
                input={"query": query},
                reason=reason,
                model_name=_AGENT_MODELS.get(name),
            )
            for i, name in enumerate(agent_names, 1)
        ]

    def _format_agent_results(self, results: list[AgentResult]) -> str:
        """Flatten agent results into a text block for synthesis."""
        parts: list[str] = []
        for r in results:
            if not r.success:
                parts.append(
                    f"[{r.agent_name}] ERROR: {r.error or 'no details'}"
                )
                continue
            try:
                data_str = json.dumps(r.data, indent=2)
            except (TypeError, ValueError):
                data_str = str(r.data)
            limit = (
                4_000
                if r.agent_name in {"api_spec", "semantic_retriever"}
                else 2_000
            )
            if len(data_str) > limit:
                data_str = data_str[:limit] + "\n... [truncated]"
            parts.append(f"[{r.agent_name}]\n{data_str}")

        text = "\n\n".join(parts)
        if len(text) > _RESULTS_CHAR_LIMIT:
            text = text[:_RESULTS_CHAR_LIMIT] + "\n[context truncated]"
        return text or "(no agent results available)"

    def _product_catalog_answer(
        self,
        user_message: str,
        results: list[AgentResult],
    ) -> str | None:
        """Deterministically answer product endpoint-list questions."""
        lower = user_message.lower()
        wants_catalog = (
            (
                "product" in lower
                or "apis" in lower
                or "endpoints" in lower
            )
            and any(
                word in lower
                for word in ("available", "list", "show", "what", "which")
            )
        )
        if not wants_catalog:
            return None

        operations: list[dict[str, Any]] = []
        for result in results:
            if not result.success:
                continue
            data = result.data
            for item in data.get("results") or []:
                operations.append(
                    {
                        "method": item.get("method"),
                        "path": item.get("endpoint") or item.get("path"),
                        "summary": item.get("summary"),
                        "product_key": item.get("product_key"),
                    }
                )
            for item in data.get("recommended_operations") or []:
                operations.append(
                    {
                        "method": item.get("method"),
                        "path": item.get("path") or item.get("endpoint"),
                        "summary": item.get("summary") or item.get("title"),
                        "product_key": item.get("product_key"),
                    }
                )
            for item in data.get("matches") or []:
                operations.append(
                    {
                        "method": item.get("method"),
                        "path": item.get("endpoint") or item.get("path"),
                        "summary": item.get("description"),
                        "product_key": item.get("product"),
                    }
                )

        deduped: list[dict[str, Any]] = []
        seen: set[tuple[str, str]] = set()
        for op in operations:
            method = str(op.get("method") or "").upper()
            path = str(op.get("path") or "")
            if not method or not path:
                continue
            key = (method, path)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(op)

        product_counts: dict[str, int] = {}
        for op in deduped:
            product = str(op.get("product_key") or "")
            if product:
                product_counts[product] = product_counts.get(product, 0) + 1

        dominant_product = None
        if product_counts:
            dominant_product, dominant_count = max(
                product_counts.items(), key=lambda item: item[1]
            )
            if dominant_count >= 3:
                deduped = [
                    op
                    for op in deduped
                    if op.get("product_key") == dominant_product
                ]

        if len(deduped) < 3:
            return None

        product = dominant_product or next(
            (
                op.get("product_key")
                for op in deduped
                if op.get("product_key")
            ),
            "matching",
        )
        lines = [
            f"I found these {product} API operations in the catalog:",
            "",
        ]
        for index, op in enumerate(deduped[:12], start=1):
            summary = op.get("summary") or "No summary available"
            lines.append(
                f"{index}. {str(op.get('method') or '').upper()} "
                f"{op.get('path')} - {summary}"
            )

        lines.append("")
        if len(deduped) > 12:
            lines.append(
                f"Showing 12 of {len(deduped)} matched operations. "
                "Ask for a narrower workflow for more detail."
            )
        else:
            lines.append(
                f"Showing {len(deduped)} matched operations from the "
                "retrieved catalog context."
            )
        return "\n".join(lines)

    def _fallback_answer(self, results: list[AgentResult]) -> str:
        """Extract a minimal answer from agent results without LLM."""
        for r in results:
            if not r.success:
                continue
            ops = (
                r.data.get("operations")
                or r.data.get("recommended_operations")
                or []
            )
            if ops and isinstance(ops[0], dict):
                top = ops[0]
                method = top.get("method") or ""
                path = top.get("path") or top.get("endpoint") or ""
                title = top.get("title") or top.get("summary") or ""
                if method or path:
                    return (
                        f"Based on the available context, consider: "
                        f"{method} {path}"
                        + (f" — {title}" if title else "")
                        + "."
                    )
        return (
            "No matching APIs were found in the catalog for your query. "
            "Try rephrasing or narrowing your search."
        )
