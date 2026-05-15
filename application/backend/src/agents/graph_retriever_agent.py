from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from agent_tools.neo4j_tools import (
    get_operation_schema_context,
    get_related_operations,
    get_resource_lifecycle,
    graph_search_by_query,
)
from chat.schemas import AgentResult
from data.ingestion.graph_expansion import Neo4jGraphExpander

_log = logging.getLogger(__name__)


class GraphRetrieverAgent:
    """Exposes Neo4j graph queries to the orchestrator.

    No LLM inference — all results come from deterministic Cypher.

    Each public ``run_*`` method opens a single Neo4jGraphExpander, runs
    its query, then closes the driver in a ``finally`` block.  Credentials
    are read from .env via Neo4jGraphExpander.from_env(); nothing is
    hardcoded here.
    """

    AGENT_NAME = "graph_retriever"

    def __init__(
        self,
        env_path: Path | None = None,
        semantic_retriever: Any = None,
    ) -> None:
        self._env_path = env_path
        # Optional: share the warmed-up Milvus retriever for graph_search.
        # If None, graph_search_by_query will lazy-load from milvus_tools.
        self._semantic_retriever = semantic_retriever

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _open_expander(self) -> Neo4jGraphExpander:
        if self._env_path:
            return Neo4jGraphExpander.from_env(env_path=self._env_path)
        return Neo4jGraphExpander.from_env()

    def _connection_error(self, method: str, exc: Exception) -> AgentResult:
        msg = (
            f"Neo4j connection failed in {method}: {exc}. "
            "Verify NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD in .env."
        )
        _log.error("%s.%s: %s", self.AGENT_NAME, method, msg)
        return AgentResult(
            agent_name=self.AGENT_NAME,
            model_name=None,
            success=False,
            data={},
            error=msg,
            sources=[],
        )

    def _not_found_error(
        self, method: str, lookup_key: str, lookup_value: str
    ) -> AgentResult:
        msg = (
            f"No Neo4j node found for {lookup_key}={lookup_value!r}. "
            "The ID may be incorrect or the node may not have been ingested."
        )
        _log.warning("%s.%s: %s", self.AGENT_NAME, method, msg)
        return AgentResult(
            agent_name=self.AGENT_NAME,
            model_name=None,
            success=False,
            data={lookup_key: lookup_value},
            error=msg,
            sources=[],
        )

    # ------------------------------------------------------------------
    # Public run methods
    # ------------------------------------------------------------------

    def run_graph_search(
        self,
        query: str,
        max_graph_expansions: int = 3,
    ) -> AgentResult:
        """GraphRAG: semantic search via Milvus + graph expansion via Neo4j.

        Delegates to GraphRagRetriever.retrieve() so retrieval logic stays
        in one place.  Returns recommended operations, schemas, and
        confidence notes from the combined pipeline.
        """
        _log.info(
            "%s.run_graph_search query=%r expansions=%d",
            self.AGENT_NAME,
            query,
            max_graph_expansions,
        )
        expander = None
        try:
            expander = self._open_expander()
            context = graph_search_by_query(
                expander,
                query=query,
                semantic_retriever=self._semantic_retriever,
                max_graph_expansions=max_graph_expansions,
            )
            ops = context.get("recommended_operations") or []
            schemas = context.get("schemas") or []
            sources = list({
                op.get("source") for op in ops if op.get("source")
            })
            _log.info(
                "%s.run_graph_search: %d op(s), %d schema(s) | query=%r",
                self.AGENT_NAME,
                len(ops),
                len(schemas),
                query,
            )
            return AgentResult(
                agent_name=self.AGENT_NAME,
                model_name=None,
                success=True,
                data={
                    "query":                  query,
                    "recommended_operations": ops,
                    "schemas":                schemas,
                    "fields":                 context.get("fields") or [],
                    "confidence_notes":       context.get("confidence_notes") or [],
                    "semantic_hit_count":     len(context.get("semantic_hits") or []),
                    "graph_context_count":    len(context.get("graph_context") or []),
                },
                sources=sources,
            )
        except Exception as exc:
            return self._connection_error("run_graph_search", exc)
        finally:
            if expander is not None:
                expander.close()

    def run_get_related_operations(
        self,
        operation_id: str,
    ) -> AgentResult:
        """Return operations that share a resource lifecycle with *operation_id*."""
        _log.info(
            "%s.run_get_related_operations id=%r",
            self.AGENT_NAME,
            operation_id,
        )
        expander = None
        try:
            expander = self._open_expander()
            ops = get_related_operations(expander, operation_id)
            _log.info(
                "%s.run_get_related_operations: %d related op(s) for id=%r",
                self.AGENT_NAME,
                len(ops),
                operation_id,
            )
            return AgentResult(
                agent_name=self.AGENT_NAME,
                model_name=None,
                success=True,
                data={
                    "operation_id":       operation_id,
                    "related_operations": ops,
                    "operation_count":    len(ops),
                },
                sources=["neo4j:related_lifecycle_operation"],
            )
        except ValueError as exc:
            return self._not_found_error(
                "run_get_related_operations", "operation_id", operation_id
            )
        except Exception as exc:
            return self._connection_error(
                "run_get_related_operations", exc
            )
        finally:
            if expander is not None:
                expander.close()

    def run_get_operation_schema_context(
        self,
        operation_id: str,
    ) -> AgentResult:
        """Return schemas, parameters, and auth context for *operation_id*."""
        _log.info(
            "%s.run_get_operation_schema_context id=%r",
            self.AGENT_NAME,
            operation_id,
        )
        expander = None
        try:
            expander = self._open_expander()
            schema_ctx = get_operation_schema_context(expander, operation_id)
            req = schema_ctx.get("request_schemas") or []
            res = schema_ctx.get("response_schemas") or []
            _log.info(
                "%s.run_get_operation_schema_context: "
                "%d req schema(s), %d res schema(s) for id=%r",
                self.AGENT_NAME,
                len(req),
                len(res),
                operation_id,
            )
            return AgentResult(
                agent_name=self.AGENT_NAME,
                model_name=None,
                success=True,
                data={
                    "operation_id": operation_id,
                    **schema_ctx,
                },
                sources=["neo4j:operation_schema"],
            )
        except ValueError as exc:
            return self._not_found_error(
                "run_get_operation_schema_context", "operation_id", operation_id
            )
        except Exception as exc:
            return self._connection_error(
                "run_get_operation_schema_context", exc
            )
        finally:
            if expander is not None:
                expander.close()

    def run_get_resource_lifecycle(
        self,
        resource_name: str,
    ) -> AgentResult:
        """Return all operations under resources matching *resource_name*.

        Uses a parameterized Cypher query (case-insensitive substring or
        exact id match) against Resource nodes in the knowledge graph.
        Returns success=False with a descriptive error when nothing matches.
        """
        _log.info(
            "%s.run_get_resource_lifecycle name=%r",
            self.AGENT_NAME,
            resource_name,
        )
        expander = None
        try:
            expander = self._open_expander()
            rows = get_resource_lifecycle(expander, resource_name)

            if not rows:
                msg = (
                    f"No Resource node found matching name={resource_name!r}. "
                    "Try a broader name or check the ingested spec."
                )
                _log.warning(
                    "%s.run_get_resource_lifecycle: %s",
                    self.AGENT_NAME,
                    msg,
                )
                return AgentResult(
                    agent_name=self.AGENT_NAME,
                    model_name=None,
                    success=False,
                    data={"resource_name": resource_name, "resources": []},
                    error=msg,
                    sources=[],
                )

            total_ops = sum(
                len(row.get("operations") or []) for row in rows
            )
            _log.info(
                "%s.run_get_resource_lifecycle: "
                "%d resource(s), %d total op(s) for name=%r",
                self.AGENT_NAME,
                len(rows),
                total_ops,
                resource_name,
            )
            return AgentResult(
                agent_name=self.AGENT_NAME,
                model_name=None,
                success=True,
                data={
                    "resource_name":   resource_name,
                    "resources":       rows,
                    "total_resources": len(rows),
                    "total_operations": total_ops,
                },
                sources=["neo4j:resource_lifecycle"],
            )
        except Exception as exc:
            return self._connection_error(
                "run_get_resource_lifecycle", exc
            )
        finally:
            if expander is not None:
                expander.close()
