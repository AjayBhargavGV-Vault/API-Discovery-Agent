from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agent_tools.api_spec_tools import (
    get_operation_details as _get_op_details,
    get_schema_details as _get_schema_details,
)
from agent_tools.milvus_tools import semantic_api_search as _milvus_search
from agent_tools.neo4j_tools import (
    get_operation_schema_context as _neo4j_op_schema,
    get_related_operations as _neo4j_related,
    get_resource_lifecycle as _neo4j_lifecycle,
    graph_search_by_query as _neo4j_graph_search,
)
from agents.code_example_agent import CodeExampleAgent
from agents.guardrail_agent import GuardrailAgent
from agents.query_enrichment_agent import QueryEnrichmentAgent
from agents.validation_agent import ValidationAgent
from chat.schemas import AgentResult, APIOperationDetails

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Schema helpers
# ---------------------------------------------------------------------------

def _input_schema(
    *params: tuple[str, str, str, bool],
) -> dict[str, Any]:
    """Build a JSON Schema-compatible input description.

    Each param: (name, json_type, description, is_required).
    """
    props = {
        p[0]: {"type": p[1], "description": p[2]} for p in params
    }
    required = [p[0] for p in params if p[3]]
    schema: dict[str, Any] = {"type": "object", "properties": props}
    if required:
        schema["required"] = required
    return schema


def _output_schema(
    model: str,
    description: str = "",
) -> dict[str, Any]:
    """Reference an output type by name."""
    return {"$model": model, "description": description}


# ---------------------------------------------------------------------------
# Wrapper factories
# ---------------------------------------------------------------------------

def _neo4j_wrapper(
    fn: Callable[..., Any],
    env_path: Path | None,
) -> Callable[..., Any]:
    """Return a callable that opens/closes a Neo4j expander per call.

    Neo4jGraphExpander is imported lazily to avoid driver errors at
    module import time when the Neo4j driver is not installed.
    """
    def _call(**kwargs: Any) -> Any:
        from data.ingestion.graph_expansion import Neo4jGraphExpander
        expander = Neo4jGraphExpander.from_env(env_path=env_path)
        try:
            return fn(expander, **kwargs)
        finally:
            expander.close()
    return _call


def _neo4j_graph_wrapper(
    env_path: Path | None,
    semantic_retriever: Any,
) -> Callable[..., Any]:
    """Wrapper for graph_search_by_query; injects the semantic retriever."""
    def _call(
        query: str,
        max_graph_expansions: int = 3,
    ) -> Any:
        from data.ingestion.graph_expansion import Neo4jGraphExpander
        expander = Neo4jGraphExpander.from_env(env_path=env_path)
        try:
            return _neo4j_graph_search(
                expander,
                query=query,
                semantic_retriever=semantic_retriever,
                max_graph_expansions=max_graph_expansions,
            )
        finally:
            expander.close()
    return _call


def _code_wrapper(
    agent: CodeExampleAgent,
    language: str,
) -> Callable[[APIOperationDetails], AgentResult]:
    """Bind CodeExampleAgent.run to a fixed output language."""
    def _call(operation_details: APIOperationDetails) -> AgentResult:
        return agent.run(operation_details, language)
    return _call


# ---------------------------------------------------------------------------
# Tool
# ---------------------------------------------------------------------------

@dataclass
class Tool:
    """A registered runtime tool with metadata and a typed callable.

    Attributes:
        name:          Unique tool identifier used for lookup and dispatch.
        description:   Human-readable explanation of the tool's purpose.
        callable:      The function or bound method to invoke.
        owning_agent:  Agent class or module that owns this tool.
        model_used:    LLM model ID, or None for non-LLM tools.
        input_schema:  JSON Schema-compatible dict describing inputs.
        output_schema: Dict describing the return type.
    """

    name: str
    description: str
    callable: Callable[..., Any]
    owning_agent: str
    model_used: str | None
    input_schema: dict[str, Any]
    output_schema: dict[str, Any]

    def __call__(self, **kwargs: Any) -> Any:
        """Invoke the tool with keyword arguments."""
        return self.callable(**kwargs)


# ---------------------------------------------------------------------------
# ToolRegistry
# ---------------------------------------------------------------------------

class ToolRegistry:
    """Registry of all runtime tools available to the agentic pipeline.

    FastAPI routes interact only with this registry — never with raw
    LLM clients or model-level APIs directly.

    Usage::

        registry = ToolRegistry.build(env_path=..., semantic_retriever=...)
        result = registry.call("semantic_api_search", query="payment intent")
    """

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(self, tool: Tool) -> None:
        """Add a Tool to the registry, warning on name collision."""
        if tool.name in self._tools:
            _log.warning(
                "Tool '%s' already registered; overwriting.", tool.name
            )
        self._tools[tool.name] = tool

    # ------------------------------------------------------------------
    # Lookup & invocation
    # ------------------------------------------------------------------

    def get(self, name: str) -> Tool | None:
        """Return the named Tool, or None if not registered."""
        return self._tools.get(name)

    def call(self, name: str, **kwargs: Any) -> Any:
        """Invoke a tool by name with keyword arguments.

        Raises:
            KeyError: if *name* is not registered.
            Any exception raised by the tool callable itself.
        """
        tool = self._tools.get(name)
        if tool is None:
            raise KeyError(
                f"Tool '{name}' is not registered. "
                f"Available: {list(self._tools)}"
            )
        return tool.callable(**kwargs)

    def list_tools(self) -> list[Tool]:
        """Return all registered Tool objects."""
        return list(self._tools.values())

    def names(self) -> list[str]:
        """Return all registered tool names."""
        return list(self._tools.keys())

    def describe_all(self) -> str:
        """Return a human-readable summary of every registered tool."""
        return "\n".join(
            f"- {t.name} [{t.owning_agent}]: {t.description}"
            for t in self._tools.values()
        )

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def build(
        cls,
        env_path: Path | None = None,
        semantic_retriever: Any = None,
    ) -> "ToolRegistry":
        """Instantiate all agents and register the 15 runtime tools.

        Args:
            env_path:           Path to the .env file; forwarded to agents.
            semantic_retriever: Pre-built Milvus retriever. When provided
                                it is registered globally via init_retriever
                                so that semantic_api_search works immediately.
        """
        if semantic_retriever is not None:
            from agent_tools.milvus_tools import init_retriever
            init_retriever(semantic_retriever)

        guardrail = GuardrailAgent(env_path=env_path)
        enricher = QueryEnrichmentAgent(env_path=env_path)
        code_agent = CodeExampleAgent(env_path=env_path)
        validator = ValidationAgent(env_path=env_path)

        registry = cls()

        # ----------------------------------------------------------------
        # 1  check_input_guardrails
        # ----------------------------------------------------------------
        registry.register(Tool(
            name="check_input_guardrails",
            description=(
                "Screen user input for safety violations — prompt injection, "
                "jailbreak, credential extraction, and PII leakage. "
                "Returns SAFE, REVIEW, or BLOCK with an explanatory reason."
            ),
            callable=guardrail.check_input_guardrails,
            owning_agent="guardrail_agent",
            model_used="llama-guard-3-8b",
            input_schema=_input_schema(
                ("user_message", "string",
                 "Raw user query to screen for safety violations", True),
            ),
            output_schema=_output_schema(
                "GuardrailResult",
                "allowed, risk_level (SAFE|REVIEW|BLOCK), reason, "
                "safe_response, model_name.",
            ),
        ))

        # ----------------------------------------------------------------
        # 2  check_output_guardrails
        # ----------------------------------------------------------------
        registry.register(Tool(
            name="check_output_guardrails",
            description=(
                "Screen the agent's draft answer for secret leakage and "
                "internal prompt exposure before returning it to the user."
            ),
            callable=guardrail.check_output_guardrails,
            owning_agent="guardrail_agent",
            model_used="llama-guard-3-8b",
            input_schema=_input_schema(
                ("answer", "string",
                 "Draft answer text to screen", True),
                ("sources", "array",
                 "Retrieved source objects (accepted for API parity)",
                 False),
            ),
            output_schema=_output_schema(
                "GuardrailResult",
                "allowed, risk_level (SAFE|REVIEW|BLOCK), reason, "
                "safe_response, model_name.",
            ),
        ))

        # ----------------------------------------------------------------
        # 3  enrich_user_query
        # ----------------------------------------------------------------
        registry.register(Tool(
            name="enrich_user_query",
            description=(
                "Rewrite a vague developer question into a rich, "
                "retrieval-optimised API discovery query. "
                "Detects intent and entities; recommends downstream agents."
            ),
            callable=enricher.enrich,
            owning_agent="query_enrichment_agent",
            model_used="qwen2.5-14b-instruct",
            input_schema=_input_schema(
                ("user_message", "string",
                 "Original raw developer question", True),
                ("chat_history", "array",
                 "Recent conversation turns, oldest first", False),
                ("last_operations", "array",
                 "Recently surfaced API operation IDs or summaries",
                 False),
            ),
            output_schema=_output_schema(
                "QueryEnrichmentResult",
                "enriched_query, detected_intent, entities, "
                "suggested_agents, confidence_score.",
            ),
        ))

        # ----------------------------------------------------------------
        # 4  semantic_api_search
        # ----------------------------------------------------------------
        registry.register(Tool(
            name="semantic_api_search",
            description=(
                "Vector-similarity search over the API catalog in Milvus. "
                "Returns the top-K most semantically relevant API operations. "
                "Requires a retriever to be registered via init_retriever."
            ),
            callable=_milvus_search,
            owning_agent="milvus_tools",
            model_used=None,
            input_schema=_input_schema(
                ("query", "string",
                 "Search query string (preferably enriched)", True),
                ("top_k", "integer",
                 "Maximum number of results to return (default: 10)", False),
            ),
            output_schema=_output_schema(
                "list[RetrievedAPIMatch]",
                "Ranked list of matching API operations with scores.",
            ),
        ))

        # ----------------------------------------------------------------
        # 5  graph_search_by_query
        # ----------------------------------------------------------------
        registry.register(Tool(
            name="graph_search_by_query",
            description=(
                "Hybrid semantic + Neo4j graph search. Expands retrieved "
                "operations along relationship edges (CALLS, REQUIRES, "
                "RETURNS_SCHEMA) to surface related endpoints and schemas."
            ),
            callable=_neo4j_graph_wrapper(env_path, semantic_retriever),
            owning_agent="neo4j_tools",
            model_used=None,
            input_schema=_input_schema(
                ("query", "string",
                 "Search query for graph traversal", True),
                ("max_graph_expansions", "integer",
                 "Max relationship hops to follow (default: 3)", False),
            ),
            output_schema=_output_schema(
                "dict",
                "recommended_operations, schemas, confidence_notes.",
            ),
        ))

        # ----------------------------------------------------------------
        # 6  get_related_operations
        # ----------------------------------------------------------------
        registry.register(Tool(
            name="get_related_operations",
            description=(
                "Fetch all API operations connected to a given operationId "
                "in the Neo4j knowledge graph via CALLS, REQUIRES, or "
                "RETURNS_SCHEMA relationships."
            ),
            callable=_neo4j_wrapper(_neo4j_related, env_path),
            owning_agent="neo4j_tools",
            model_used=None,
            input_schema=_input_schema(
                ("operation_id", "string",
                 "operationId to find related operations for", True),
            ),
            output_schema=_output_schema(
                "list[dict]",
                "Related operation dicts with operationId, method, path.",
            ),
        ))

        # ----------------------------------------------------------------
        # 7  get_operation_schema_context
        # ----------------------------------------------------------------
        registry.register(Tool(
            name="get_operation_schema_context",
            description=(
                "Retrieve request/response schema nodes linked to an "
                "operation in Neo4j, including field names, types, "
                "and required flags."
            ),
            callable=_neo4j_wrapper(_neo4j_op_schema, env_path),
            owning_agent="neo4j_tools",
            model_used=None,
            input_schema=_input_schema(
                ("operation_id", "string",
                 "operationId whose schema context to retrieve", True),
            ),
            output_schema=_output_schema(
                "dict",
                "request_fields and response_fields lists with type info.",
            ),
        ))

        # ----------------------------------------------------------------
        # 8  get_resource_lifecycle
        # ----------------------------------------------------------------
        registry.register(Tool(
            name="get_resource_lifecycle",
            description=(
                "Retrieve the full CRUD lifecycle of a named resource from "
                "Neo4j: all operations that create, read, update, or "
                "delete it, in dependency order."
            ),
            callable=_neo4j_wrapper(_neo4j_lifecycle, env_path),
            owning_agent="neo4j_tools",
            model_used=None,
            input_schema=_input_schema(
                ("resource_name", "string",
                 "Resource entity name, e.g. 'PaymentIntent'", True),
            ),
            output_schema=_output_schema(
                "list[dict]",
                "Ordered lifecycle operations for the named resource.",
            ),
        ))

        # ----------------------------------------------------------------
        # 9  get_operation_details
        # ----------------------------------------------------------------
        registry.register(Tool(
            name="get_operation_details",
            description=(
                "Look up a single API operation from the OpenAPI spec by "
                "its operationId. Returns endpoint, method, parameters, "
                "request/response schema, and auth scheme."
            ),
            callable=_get_op_details,
            owning_agent="api_spec_tools",
            model_used=None,
            input_schema=_input_schema(
                ("operation_id", "string",
                 "Exact operationId from the OpenAPI spec", True),
            ),
            output_schema=_output_schema(
                "APIOperationDetails | None",
                "Full operation details, or None if operationId not found.",
            ),
        ))

        # ----------------------------------------------------------------
        # 10  get_schema_details
        # ----------------------------------------------------------------
        registry.register(Tool(
            name="get_schema_details",
            description=(
                "Look up a schema component by name from the OpenAPI spec "
                "and ingested field data. Returns field names, types, "
                "descriptions, and required flags."
            ),
            callable=_get_schema_details,
            owning_agent="api_spec_tools",
            model_used=None,
            input_schema=_input_schema(
                ("schema_name", "string",
                 "Schema component name, e.g. 'PaymentIntent'", True),
            ),
            output_schema=_output_schema(
                "dict",
                "Schema dict with fields list and property metadata.",
            ),
        ))

        # ----------------------------------------------------------------
        # 11  generate_curl
        # ----------------------------------------------------------------
        registry.register(Tool(
            name="generate_curl",
            description=(
                "Generate a grounded curl command for an API operation. "
                "Uses only the endpoint, method, and fields from the spec; "
                "never invents endpoints or request fields."
            ),
            callable=_code_wrapper(code_agent, "curl"),
            owning_agent="code_example_agent",
            model_used="deepseek-coder-v2-lite-instruct",
            input_schema=_input_schema(
                ("operation_details", "object",
                 "APIOperationDetails for the target operation", True),
            ),
            output_schema=_output_schema(
                "AgentResult",
                "data['code'] contains the curl command string.",
            ),
        ))

        # ----------------------------------------------------------------
        # 12  generate_python_example
        # ----------------------------------------------------------------
        registry.register(Tool(
            name="generate_python_example",
            description=(
                "Generate a grounded Python `requests` example for an API "
                "operation. Never invents endpoints or request fields."
            ),
            callable=_code_wrapper(code_agent, "python"),
            owning_agent="code_example_agent",
            model_used="deepseek-coder-v2-lite-instruct",
            input_schema=_input_schema(
                ("operation_details", "object",
                 "APIOperationDetails for the target operation", True),
            ),
            output_schema=_output_schema(
                "AgentResult",
                "data['code'] contains the Python snippet.",
            ),
        ))

        # ----------------------------------------------------------------
        # 13  generate_node_example
        # ----------------------------------------------------------------
        registry.register(Tool(
            name="generate_node_example",
            description=(
                "Generate a grounded Node.js fetch example (Node 18+, "
                "built-in fetch, no external dependencies). "
                "Never invents endpoints or request fields."
            ),
            callable=_code_wrapper(code_agent, "node"),
            owning_agent="code_example_agent",
            model_used="deepseek-coder-v2-lite-instruct",
            input_schema=_input_schema(
                ("operation_details", "object",
                 "APIOperationDetails for the target operation", True),
            ),
            output_schema=_output_schema(
                "AgentResult",
                "data['code'] contains the Node.js snippet.",
            ),
        ))

        # ----------------------------------------------------------------
        # 14  generate_sample_payload
        # ----------------------------------------------------------------
        registry.register(Tool(
            name="generate_sample_payload",
            description=(
                "Generate a grounded sample JSON request body for an API "
                "operation. Derives all fields from the spec; uses typed "
                "placeholders for unknown runtime values."
            ),
            callable=_code_wrapper(code_agent, "payload"),
            owning_agent="code_example_agent",
            model_used="deepseek-coder-v2-lite-instruct",
            input_schema=_input_schema(
                ("operation_details", "object",
                 "APIOperationDetails for the target operation", True),
            ),
            output_schema=_output_schema(
                "AgentResult",
                "data['code'] contains the sample JSON payload.",
            ),
        ))

        # ----------------------------------------------------------------
        # 15  validate_response
        # ----------------------------------------------------------------
        registry.register(Tool(
            name="validate_response",
            description=(
                "Validate a draft answer against retrieved API sources. "
                "Checks endpoint existence, method correctness, schema "
                "names, auth scheme presence, unsupported claims, code "
                "example grounding, and non-empty output. "
                "Rewrites final_answer to remove detected hallucinations."
            ),
            callable=validator.validate_response,
            owning_agent="validation_agent",
            model_used="qwen2.5-7b-instruct",
            input_schema=_input_schema(
                ("answer", "string",
                 "Draft answer text to validate", True),
                ("sources", "array",
                 "Retrieved source dicts or APIOperationDetails objects",
                 True),
                ("agent_results", "array",
                 "AgentResult objects from the upstream pipeline", True),
            ),
            output_schema=_output_schema(
                "ValidationResult",
                "is_grounded, unsupported_claims, confidence_score, "
                "final_answer, model_name.",
            ),
        ))

        _log.info(
            "ToolRegistry built with %d tools: %s",
            len(registry._tools),
            list(registry._tools),
        )
        return registry
