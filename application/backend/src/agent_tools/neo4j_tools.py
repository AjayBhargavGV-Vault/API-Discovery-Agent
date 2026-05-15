from __future__ import annotations

import logging
from typing import Any

_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Parameterized Cypher — resource lifecycle
# ---------------------------------------------------------------------------
# Finds resources by case-insensitive name substring or exact id, then
# returns their full operation list and owning products.
_RESOURCE_LIFECYCLE_QUERY = """
MATCH (resource:Resource)
WHERE toLower(resource.name) CONTAINS toLower($name)
   OR resource.id = $name
OPTIONAL MATCH (resource)-[:HAS_OPERATION]->(op:Operation)
OPTIONAL MATCH (product:Product)-[:HAS_RESOURCE]->(resource)
WITH resource,
     collect(DISTINCT op {
         .id, .operation_id, .method, .path,
         .summary, .description
     }) AS operations,
     collect(DISTINCT product {
         .id, .name, .business_domain
     }) AS products
RETURN resource {.*}   AS resource,
       operations,
       products
ORDER BY resource.name
LIMIT 20
"""


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def _run_query(
    expander: Any,
    cypher: str,
    parameters: dict[str, Any],
) -> list[dict[str, Any]]:
    """Execute a parameterized Cypher query and return all rows as dicts.

    Uses the driver and database already held by *expander* — no new
    connection is opened.
    """
    with expander.driver.session(
        database=expander.config.database
    ) as session:
        result = session.run(cypher, parameters)
        return [dict(record) for record in result]


def _clean(value: Any) -> Any:
    """Return None for falsy non-zero values so callers get clean dicts."""
    if value is None:
        return None
    if isinstance(value, (list, dict)) and not value:
        return value
    return value


# ---------------------------------------------------------------------------
# Public tool functions  (each accepts a Neo4jGraphExpander instance)
# ---------------------------------------------------------------------------

def graph_search_by_query(
    expander: Any,
    query: str,
    semantic_retriever: Any = None,
    max_graph_expansions: int = 3,
) -> dict[str, Any]:
    """Run a full GraphRAG search: Milvus semantic retrieval + Neo4j expansion.

    If *semantic_retriever* is None the module-level shared retriever from
    milvus_tools is used (registered at app startup).

    Reuses GraphRagRetriever unchanged so retrieval logic stays in one place.
    """
    if semantic_retriever is None:
        from agent_tools.milvus_tools import _get_retriever
        try:
            semantic_retriever = _get_retriever()
        except Exception:
            raise RuntimeError(
                "Milvus retriever is unavailable; "
                "graph search requires semantic retrieval."
            )

    from data.ingestion.graph_expansion import GraphRagRetriever
    rag = GraphRagRetriever(
        semantic_retriever=semantic_retriever,
        graph_expander=expander,
    )
    return rag.retrieve(
        query=query,
        max_graph_expansions=max_graph_expansions,
    )


def get_related_operations(
    expander: Any,
    operation_id: str,
) -> list[dict[str, Any]]:
    """Return operations that share a resource lifecycle with *operation_id*.

    Reuses Neo4jGraphExpander.expand_operation() which already traverses
    the ``(Resource)-[:HAS_OPERATION]->`` relationship.

    Raises ValueError (propagated) if the node does not exist in Neo4j so
    the caller can surface a clear error rather than returning empty data.
    """
    context = expander.expand_operation(operation_id)
    ops: list[Any] = context.get("related_lifecycle_operations") or []
    _log.debug(
        "get_related_operations: %d related ops for %s",
        len(ops),
        operation_id,
    )
    return [op for op in ops if isinstance(op, dict)]


def get_operation_schema_context(
    expander: Any,
    operation_id: str,
) -> dict[str, Any]:
    """Return schema, parameter, and auth context for an operation node.

    Reuses Neo4jGraphExpander.expand_operation() — the Cypher already
    fetches request/response schemas, parameters, auth schemes, and
    schema fields in a single query.

    Raises ValueError if the node does not exist.
    """
    context = expander.expand_operation(operation_id)
    return {
        "operation":        context.get("node") or {},
        "request_schemas":  context.get("request_schemas") or [],
        "response_schemas": context.get("response_schemas") or [],
        "parameters":       context.get("parameters") or [],
        "auth_schemes":     context.get("auth_schemes") or [],
        "schema_fields":    context.get("schema_fields") or [],
    }


def get_resource_lifecycle(
    expander: Any,
    resource_name: str,
) -> list[dict[str, Any]]:
    """Return resources matching *resource_name* with their full operation set.

    Uses a parameterized Cypher query (case-insensitive partial match on
    resource name, or exact match on resource id).  Returns a list of dicts,
    each with keys: resource, operations, products.
    """
    rows = _run_query(
        expander,
        _RESOURCE_LIFECYCLE_QUERY,
        {"name": resource_name},
    )
    _log.debug(
        "get_resource_lifecycle: %d resource row(s) for name=%r",
        len(rows),
        resource_name,
    )
    return rows
