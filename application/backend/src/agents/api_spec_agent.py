from __future__ import annotations

import logging
from pathlib import Path

from agent_tools.api_spec_tools import (
    get_operation_details,
    get_schema_details,
    search_operations_by_name_or_path,
)
from chat.schemas import AgentResult

_log = logging.getLogger(__name__)


class ApiSpecAgent:
    """Returns exact API operation details from the OpenAPI spec and indexed
    artifacts.

    No LLM — every result is deterministic.  The three public ``run_*``
    methods each return an AgentResult so the orchestrator can consume them
    uniformly.  Caches for the JSON artifacts and the parsed YAML are
    populated on first access and held for the process lifetime.
    """

    AGENT_NAME = "api_spec"

    # ------------------------------------------------------------------
    # Init
    # ------------------------------------------------------------------

    def __init__(self, data_dir: Path | None = None) -> None:
        # data_dir is accepted for testability (override default paths).
        # api_spec_tools uses module-level defaults; we cannot redirect
        # them per-instance without threading the path through every call,
        # so this is documented as a future extension point.
        if data_dir is not None:
            _log.debug(
                "%s: custom data_dir=%s (not yet wired to tool functions)",
                self.AGENT_NAME,
                data_dir,
            )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _not_found(self, lookup: str, value: str) -> AgentResult:
        msg = (
            f"No {lookup} found for {value!r} in the indexed artifacts or "
            "OpenAPI spec.  The identifier may be incorrect or the spec may "
            "not have been ingested."
        )
        _log.warning("%s: %s", self.AGENT_NAME, msg)
        return AgentResult(
            agent_name=self.AGENT_NAME,
            model_name=None,
            success=False,
            data={lookup: value},
            error=msg,
            sources=[],
        )

    def _error(self, method: str, exc: Exception) -> AgentResult:
        msg = f"{method} raised an unexpected error: {exc}"
        _log.error("%s.%s: %s", self.AGENT_NAME, method, msg, exc_info=True)
        return AgentResult(
            agent_name=self.AGENT_NAME,
            model_name=None,
            success=False,
            data={},
            error=msg,
            sources=[],
        )

    # ------------------------------------------------------------------
    # Public run methods
    # ------------------------------------------------------------------

    def run_get_operation_details(
        self, operation_id: str
    ) -> AgentResult:
        """Return full operation details for a single *operation_id*.

        Accepts operationId strings ("GetAccount"), internal doc IDs
        ("api_operation:get:…"), or neo4j_node_id values.

        On success, data["operation"] holds the serialized
        APIOperationDetails dict.
        """
        _log.info(
            "%s.run_get_operation_details operation_id=%r",
            self.AGENT_NAME,
            operation_id,
        )
        try:
            details = get_operation_details(operation_id)
        except Exception as exc:
            return self._error("run_get_operation_details", exc)

        if details is None:
            return self._not_found("operation_id", operation_id)

        serialized = details.model_dump()
        sources = []
        if details.method and details.endpoint:
            sources.append(f"spec:{details.method}:{details.endpoint}")

        _log.info(
            "%s.run_get_operation_details: found %s %s",
            self.AGENT_NAME,
            details.method,
            details.endpoint,
        )
        return AgentResult(
            agent_name=self.AGENT_NAME,
            model_name=None,
            success=True,
            data={"operation": serialized},
            sources=sources,
        )

    def run_search_operations(
        self, query: str, max_results: int = 20
    ) -> AgentResult:
        """Case-insensitive substring search across operation names, paths,
        summaries, and product keys.

        On success, data["results"] is a list of serialized
        APIOperationDetails dicts ranked by match quality.
        """
        _log.info(
            "%s.run_search_operations query=%r max_results=%d",
            self.AGENT_NAME,
            query,
            max_results,
        )
        try:
            matches = search_operations_by_name_or_path(
                query, max_results=max_results
            )
        except Exception as exc:
            return self._error("run_search_operations", exc)

        results = [m.model_dump() for m in matches]
        _log.info(
            "%s.run_search_operations: %d result(s) for query=%r",
            self.AGENT_NAME,
            len(results),
            query,
        )
        return AgentResult(
            agent_name=self.AGENT_NAME,
            model_name=None,
            success=True,
            data={
                "query":        query,
                "result_count": len(results),
                "results":      results,
            },
            sources=["spec:search"],
        )

    def run_get_schema_details(self, schema_name: str) -> AgentResult:
        """Return field-level schema details from the OpenAPI spec.

        Accepts plain schema names ("account") and $ref strings
        ("#/components/schemas/account").

        On success, data["schema"] holds the full schema dict including
        properties from the YAML and fields from api_schema_fields.json.
        """
        _log.info(
            "%s.run_get_schema_details schema_name=%r",
            self.AGENT_NAME,
            schema_name,
        )
        try:
            schema = get_schema_details(schema_name)
        except Exception as exc:
            return self._error("run_get_schema_details", exc)

        if not schema:
            return self._not_found("schema_name", schema_name)

        resolved_name = schema.get("schema_name") or schema_name
        _log.info(
            "%s.run_get_schema_details: found %r (%d properties, %d fields)",
            self.AGENT_NAME,
            resolved_name,
            len(schema.get("properties") or {}),
            len(schema.get("fields") or []),
        )
        return AgentResult(
            agent_name=self.AGENT_NAME,
            model_name=None,
            success=True,
            data={"schema": schema},
            sources=[f"spec:schema:{resolved_name}"],
        )
