"""Full agentic chat pipeline.

Entry point: ChatEngine.run_chat_engine(request) -> ChatResponse.

Pipeline steps
--------------
1.  Receive ChatRequest
2.  Input guardrail      (Llama-Guard-3)
3.  Load conversation memory
4.  Query enrichment     (Qwen2.5-14B)
5.  Orchestrator plan    (Qwen2.5-32B)
6.  Execute agents       (semantic, graph, api_spec, code)
7.  Synthesize draft     (Qwen2.5-32B)
8.  Validate response    (Qwen2.5-7B)
9.  Output guardrail     (Llama-Guard-3)
10. Save memory
11. Return ChatResponse
"""
from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import Any

from agents.api_spec_agent import ApiSpecAgent
from agents.code_example_agent import CodeExampleAgent
from agents.graph_retriever_agent import GraphRetrieverAgent
from agents.guardrail_agent import GuardrailAgent
from agents.orchestrator_agent import OrchestratorAgent
from agents.query_enrichment_agent import (
    QueryEnrichmentAgent,
    QueryEnrichmentResult,
)
from agents.semantic_retriever_agent import SemanticRetrieverAgent
from agents.validation_agent import ValidationAgent
from chat.memory import ConversationMemory
from chat.schemas import (
    AgentResult,
    AgentTask,
    APIOperationDetails,
    ChatRequest,
    ChatResponse,
    GuardrailResult,
    ValidationResult,
)

_log = logging.getLogger(__name__)

_LANG_KEYWORDS: dict[str, list[str]] = {
    "curl": ["curl", "command line", "cli", "terminal", "shell"],
    "node": [
        "node", "nodejs", "javascript", "js", "fetch", "axios",
    ],
    "python": ["python", "requests", " py "],
    "payload": [
        "payload", "json body", "request body", "sample body",
    ],
}
_METHOD_INTENT_KEYWORDS: dict[str, list[str]] = {
    "POST": ["create", "new", "make", "submit", "send", "post"],
    "GET": ["list", "retrieve", "get", "fetch", "read", "show"],
    "DELETE": ["delete", "remove", "cancel"],
    "PUT": ["replace", "put"],
    "PATCH": ["patch"],
}

_FALLBACK_ANSWER = (
    "The API catalog does not contain enough information "
    "to answer this question."
)
_BLOCKED_ANSWER = (
    "I'm unable to process that request. "
    "Please ask something related to API discovery "
    "or documentation."
)
_OUTPUT_BLOCKED = (
    "The generated response was flagged by the safety filter "
    "and cannot be displayed."
)


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _detect_language(message: str) -> str:
    """Return the code-example language implied by *message* keywords."""
    lower = message.lower()
    for lang, keywords in _LANG_KEYWORDS.items():
        if any(kw in lower for kw in keywords):
            return lang
    return "python"


def _collect_sources(results: list[AgentResult]) -> list[Any]:
    """Deduplicated ordered sources from all agent results."""
    seen: set[str] = set()
    sources: list[Any] = []
    for r in results:
        for s in r.sources:
            key = str(s)
            if key not in seen:
                seen.add(key)
                sources.append(s)
    return sources


def _merge_sources(
    base: list[Any],
    extra: list[Any],
) -> list[Any]:
    """Append *extra* items to *base* without duplicates."""
    seen = {str(s) for s in base}
    merged = list(base)
    for s in extra:
        key = str(s)
        if key not in seen:
            seen.add(key)
            merged.append(s)
    return merged


def _collect_operation_ids(
    results: list[AgentResult],
) -> list[str]:
    """Gather operation_id strings from successful agent results."""
    ids: list[str] = []
    seen: set[str] = set()

    def _add(oid: Any) -> None:
        if oid and isinstance(oid, str) and oid not in seen:
            seen.add(oid)
            ids.append(oid)

    for r in results:
        if not r.success:
            continue
        if op := r.data.get("operation"):
            if isinstance(op, dict):
                _add(op.get("operation_id"))
        for op in r.data.get("results") or []:
            if isinstance(op, dict):
                _add(op.get("operation_id"))
        for match in r.data.get("matches") or []:
            if isinstance(match, dict):
                _add(match.get("operation_id"))
    return ids


def _preferred_methods(message: str) -> list[str]:
    """Return HTTP methods implied by action words in *message*."""
    lower = message.lower()
    methods: list[str] = []
    for method, keywords in _METHOD_INTENT_KEYWORDS.items():
        if any(kw in lower for kw in keywords):
            methods.append(method)
    if "update" in lower:
        methods.extend(["POST", "PATCH", "PUT"])
    return list(dict.fromkeys(methods))


def _operation_match_score(
    operation: APIOperationDetails,
    message: str,
) -> int:
    """Score an operation against the user's action-oriented request."""
    lower = message.lower()
    haystack = " ".join(
        part for part in (
            operation.operation_id,
            operation.endpoint,
            operation.method,
            operation.product_key,
            operation.summary,
            operation.description,
        )
        if part
    ).lower()

    score = 0
    if operation.method in _preferred_methods(message):
        score += 10

    for token in (
        "checkout", "session", "billing", "customer", "subscription",
        "payment", "invoice", "meter", "alert",
    ):
        if token in lower and token in haystack:
            score += 2

    for action in (
        "create", "list", "retrieve", "update", "expire", "delete",
        "cancel",
    ):
        if action in lower and action in haystack:
            score += 4

    return score


def _extract_op_details(
    results: list[AgentResult],
    user_message: str = "",
) -> APIOperationDetails | None:
    """Return the best usable APIOperationDetails found in results."""
    candidates: list[APIOperationDetails] = []
    for r in results:
        if r.agent_name != "api_spec" or not r.success:
            continue
        if op := r.data.get("operation"):
            if isinstance(op, dict):
                try:
                    candidates.append(APIOperationDetails(**op))
                except Exception:
                    pass
        for item in r.data.get("results") or []:
            if isinstance(item, dict):
                try:
                    candidates.append(APIOperationDetails(**item))
                except Exception:
                    pass
    if not candidates:
        return None
    if user_message:
        return max(
            candidates,
            key=lambda op: _operation_match_score(op, user_message),
        )
    return candidates[0]
    return None


def _has_results(results: list[AgentResult]) -> bool:
    """True if any agent returned useful data."""
    for r in results:
        if not r.success:
            continue
        if (
            r.data.get("matches")
            or r.data.get("recommended_operations")
            or r.data.get("results")
            or r.data.get("operation")
        ):
            return True
    return False


def _build_trace(
    plan: list[AgentTask],
    results: list[AgentResult],
    validation: ValidationResult,
    guard_in: GuardrailResult,
    guard_out: GuardrailResult,
) -> list[dict[str, Any]]:
    """Build the agent_trace list for ChatResponse."""
    trace: list[dict[str, Any]] = []
    by_name: dict[str, AgentResult] = {}
    for r in results:
        by_name.setdefault(r.agent_name, r)

    for task in plan:
        r = by_name.get(task.agent_name)
        trace.append({
            "task_id":    task.task_id,
            "agent_name": task.agent_name,
            "task_type":  task.task_type,
            "reason":     task.reason,
            "success":    r.success if r else False,
            "error":      r.error if r else "not executed",
        })

    trace.append({
        "agent_name":        "validation",
        "is_grounded":       validation.is_grounded,
        "confidence_score":  validation.confidence_score,
        "unsupported_claims": validation.unsupported_claims,
        "model_name":        validation.model_name,
    })
    trace.append({
        "agent_name": "guardrail_input",
        "risk_level": guard_in.risk_level,
        "reason":     guard_in.reason,
        "model_name": guard_in.model_name,
    })
    trace.append({
        "agent_name": "guardrail_output",
        "risk_level": guard_out.risk_level,
        "reason":     guard_out.reason,
        "model_name": guard_out.model_name,
    })
    return trace


# ---------------------------------------------------------------------------
# ChatEngine
# ---------------------------------------------------------------------------

class ChatEngine:
    """Orchestrates the full 11-step agentic chat pipeline.

    Instantiate once and call run_chat_engine(request) per turn.
    All agents are constructed at init time; the semantic retriever
    must be fully initialised before passing it here.
    """

    def __init__(
        self,
        semantic_retriever: Any,
        env_path: Path | None = None,
        spec_data_dir: Path | None = None,
    ) -> None:
        self._env_path = env_path
        self.memory = ConversationMemory()
        self._guardrail = GuardrailAgent(env_path=env_path)
        self._enricher = QueryEnrichmentAgent(env_path=env_path)
        self._orchestrator = OrchestratorAgent(env_path=env_path)
        self._semantic = SemanticRetrieverAgent(semantic_retriever)
        self._graph = GraphRetrieverAgent(
            env_path=env_path,
            semantic_retriever=semantic_retriever,
        )
        self._api_spec = ApiSpecAgent(data_dir=spec_data_dir)
        self._code = CodeExampleAgent(env_path=env_path)
        self._validator = ValidationAgent(env_path=env_path)

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run_chat_engine(
        self,
        request: ChatRequest,
    ) -> ChatResponse:
        """Execute the full 11-step pipeline and return a ChatResponse."""
        cid = request.conversation_id or str(uuid.uuid4())

        # Step 2 — Input guardrail
        guard_in = self._guardrail.check_input_guardrails(
            request.user_message
        )
        if guard_in.risk_level == "BLOCK":
            _log.warning(
                "Input BLOCK cid=%s reason=%s",
                cid,
                guard_in.reason,
            )
            return self._blocked_response(
                cid,
                guard_in.safe_response or _BLOCKED_ANSWER,
                guard_in,
            )

        # Step 3 — Load memory
        history = self.memory.get_messages(cid)
        last_ops = self.memory.get_last_referenced_operations(cid)

        # Step 4 — Query enrichment
        enrichment = self._safe_enrich(
            request.user_message, history, last_ops
        )
        _log.info(
            "cid=%s intent=%r enriched=%r",
            cid,
            enrichment.detected_intent,
            enrichment.enriched_query,
        )

        # Step 5 — Orchestrator plan
        plan = self._safe_plan(enrichment, request.user_message)

        # Step 6 — Execute plan
        agent_results = self._execute_plan(
            plan, enrichment, request
        )
        sources = _collect_sources(agent_results)

        # Step 7 — Synthesize draft response
        history_text = self.memory.format_for_prompt(cid)
        draft = self._orchestrator.synthesize_response(
            user_message=request.user_message,
            enrichment_result=enrichment,
            agent_results=agent_results,
            history=history_text,
        )
        draft_answer = draft.answer
        sources = _merge_sources(sources, list(draft.sources))

        # Step 8 — Validate response
        validation = self._validator.validate_response(
            answer=draft_answer,
            sources=sources,
            agent_results=agent_results,
        )
        final_answer = validation.final_answer
        confidence = validation.confidence_score

        # Step 9 — Output guardrail
        guard_out = self._guardrail.check_output_guardrails(
            final_answer, sources
        )
        if guard_out.risk_level == "BLOCK":
            _log.warning(
                "Output BLOCK cid=%s reason=%s",
                cid,
                guard_out.reason,
            )
            final_answer = guard_out.safe_response or _OUTPUT_BLOCKED
        guardrail_status = (
            "BLOCK"
            if guard_out.risk_level == "BLOCK"
            else guard_in.risk_level
        )

        # Step 10 — Save memory
        op_ids = _collect_operation_ids(agent_results)
        self.memory.save_turn(
            cid,
            request.user_message,
            final_answer,
            metadata={
                "enriched_query": enrichment.enriched_query,
                "selected_apis": [
                    r.agent_name
                    for r in agent_results
                    if r.success
                ],
                "operation_ids": op_ids,
                "detected_intent": enrichment.detected_intent,
            },
        )

        # Step 11 — Build and return ChatResponse
        trace = (
            _build_trace(
                plan,
                agent_results,
                validation,
                guard_in,
                guard_out,
            )
            if request.include_agent_trace
            else []
        )
        return ChatResponse(
            answer=final_answer,
            conversation_id=cid,
            sources=sources[:30],
            agent_trace=trace,
            guardrail_status=guardrail_status,
            confidence_score=confidence,
        )

    # ------------------------------------------------------------------
    # Step helpers
    # ------------------------------------------------------------------

    def _safe_enrich(
        self,
        user_message: str,
        history: list[dict[str, str]],
        last_ops: list[str],
    ) -> QueryEnrichmentResult:
        """Run query enrichment; fall back to passthrough on error."""
        try:
            return self._enricher.enrich(
                user_message=user_message,
                chat_history=history,
                last_operations=last_ops,
            )
        except Exception as exc:
            _log.warning("Enrichment failed: %s — passthrough", exc)
            return QueryEnrichmentResult.passthrough(user_message)

    def _safe_plan(
        self,
        enrichment: QueryEnrichmentResult,
        user_message: str,
    ) -> list[AgentTask]:
        """Create orchestrator plan; return empty list on hard failure."""
        try:
            return self._orchestrator.create_agent_plan(
                enrichment_result=enrichment,
                user_message=user_message,
            )
        except Exception as exc:
            _log.warning(
                "Planning failed: %s — falling back to default tasks",
                exc,
            )
            return []

    def _execute_plan(
        self,
        plan: list[AgentTask],
        enrichment: QueryEnrichmentResult,
        request: ChatRequest,
    ) -> list[AgentResult]:
        """Execute all tasks; defer code tasks until spec results exist."""
        code_tasks: list[AgentTask] = []
        non_code: list[AgentResult] = []

        for task in plan:
            if task.agent_name == "code_example":
                code_tasks.append(task)
            else:
                non_code.append(
                    self._execute_task(task, request)
                )

        # If plan is empty, fall back to semantic + graph search
        if not plan:
            _log.info(
                "Empty plan — running default semantic+graph retrieval"
            )
            non_code.append(
                self._semantic.run(
                    enrichment.enriched_query,
                    top_k=request.top_k,
                )
            )
            non_code.append(
                self._graph.run_graph_search(
                    enrichment.enriched_query
                )
            )

        code_results = [
            self._execute_code_task(t, non_code, request)
            for t in code_tasks
        ]
        return non_code + code_results

    def _execute_task(
        self,
        task: AgentTask,
        request: ChatRequest,
    ) -> AgentResult:
        """Route one AgentTask to the correct agent method."""
        try:
            name = task.agent_name
            inp = task.input

            if name == "semantic_retriever":
                query = inp.get("query") or request.user_message
                top_k = int(inp.get("top_k") or request.top_k)
                return self._semantic.run(query, top_k=top_k)

            if name == "graph_retriever":
                query = inp.get("query") or request.user_message
                expansions = int(
                    inp.get("max_graph_expansions") or 3
                )
                return self._graph.run_graph_search(
                    query,
                    max_graph_expansions=expansions,
                )

            if name == "api_spec":
                if "operation_id" in inp:
                    return (
                        self._api_spec
                        .run_get_operation_details(
                            inp["operation_id"]
                        )
                    )
                if "schema_name" in inp:
                    return self._api_spec.run_get_schema_details(
                        inp["schema_name"]
                    )
                query = inp.get("query") or request.user_message
                return self._api_spec.run_search_operations(query)

            if name == "related_operations":
                op_id = str(inp.get("operation_id") or "")
                return self._graph.run_get_related_operations(
                    op_id
                )

            if name == "schema_context":
                op_id = str(inp.get("operation_id") or "")
                return (
                    self._graph
                    .run_get_operation_schema_context(op_id)
                )

            if name == "resource_lifecycle":
                resource = str(inp.get("resource_name") or "")
                return self._graph.run_get_resource_lifecycle(
                    resource
                )

            _log.warning(
                "Unknown agent in plan: %s (task_id=%s)",
                name,
                task.task_id,
            )
            return AgentResult(
                agent_name=name,
                success=False,
                data={},
                error=f"Unknown agent: {name}",
            )

        except Exception as exc:
            _log.error(
                "Task %s (%s) failed: %s",
                task.task_id,
                task.agent_name,
                exc,
                exc_info=True,
            )
            return AgentResult(
                agent_name=task.agent_name,
                success=False,
                data={},
                error=str(exc),
            )

    def _execute_code_task(
        self,
        task: AgentTask,
        prior_results: list[AgentResult],
        request: ChatRequest,
    ) -> AgentResult:
        """Generate a code example using op details from prior results."""
        language = (
            task.input.get("language")
            or _detect_language(request.user_message)
        )
        op_details = _extract_op_details(
            prior_results,
            request.user_message,
        )
        if op_details is None:
            _log.warning(
                "Code task %s: no operation details available",
                task.task_id,
            )
            return AgentResult(
                agent_name="code_example",
                success=False,
                data={},
                error=(
                    "No operation details available "
                    "for code generation."
                ),
            )
        try:
            return self._code.run(op_details, language=language)
        except Exception as exc:
            _log.error(
                "Code generation failed: %s", exc, exc_info=True
            )
            return AgentResult(
                agent_name="code_example",
                success=False,
                data={},
                error=str(exc),
            )

    def _blocked_response(
        self,
        conversation_id: str,
        message: str,
        guard_result: GuardrailResult,
    ) -> ChatResponse:
        """Return a ChatResponse for a guardrail-blocked request."""
        trace: list[dict[str, Any]] = [{
            "agent_name": "guardrail_input",
            "risk_level": guard_result.risk_level,
            "reason":     guard_result.reason,
            "model_name": guard_result.model_name,
        }]
        return ChatResponse(
            answer=message,
            conversation_id=conversation_id,
            sources=[],
            agent_trace=trace,
            guardrail_status=guard_result.risk_level,
            confidence_score=None,
        )
