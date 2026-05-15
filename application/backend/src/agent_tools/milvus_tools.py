from __future__ import annotations

import logging
from typing import Any

from chat.schemas import RetrievedAPIMatch

_log = logging.getLogger(__name__)

# Module-level retriever injected at app startup.
# Call init_retriever() once from FastAPI lifespan or agent __init__
# before the first semantic_api_search() call.
_retriever: Any = None
_retriever_unavailable: bool = False  # set after a failed lazy-load attempt


def init_retriever(retriever: Any) -> None:
    """Share the app-level MilvusRetrievalTest instance with this module.

    Should be called once at startup so the expensive SentenceTransformer
    model is only loaded once (the same instance used by /api/search).
    """
    global _retriever
    _retriever = retriever
    _log.debug(
        "milvus_tools: retriever registered (%s)", type(retriever).__name__
    )


def _get_retriever() -> Any:
    """Return the shared retriever, lazy-initializing from .env if needed."""
    global _retriever, _retriever_unavailable
    if _retriever_unavailable:
        raise RuntimeError("Milvus retriever is unavailable (connection failed at startup).")
    if _retriever is None:
        _log.warning(
            "No shared retriever registered; lazy-loading from .env. "
            "This loads SentenceTransformer on first call — prefer calling "
            "init_retriever() at application startup."
        )
        try:
            from data.ingestion.milvus_retrieval_test import MilvusRetrievalTest
            _retriever = MilvusRetrievalTest.from_env()
        except Exception as exc:
            _retriever_unavailable = True
            raise RuntimeError(f"Milvus lazy-load failed: {exc}") from exc
    return _retriever


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

def _extract_entity(hit: dict[str, Any]) -> dict[str, Any]:
    """Return the entity sub-dict from a Milvus hit, falling back to the
    hit itself for flat result shapes."""
    entity = hit.get("entity")
    return entity if isinstance(entity, dict) else hit


def _safe_str(value: Any) -> str | None:
    """Return a non-empty stripped string or None."""
    if value is None:
        return None
    s = str(value).strip()
    return s if s else None


def _hit_to_match(hit: dict[str, Any]) -> RetrievedAPIMatch | None:
    """Normalize one raw Milvus hit into a RetrievedAPIMatch.

    Returns None for hits with no score so callers can filter safely.
    Missing optional fields are silently coerced to None.
    """
    try:
        entity = _extract_entity(hit)

        # Score is required — drop the hit without it
        raw_score = hit.get("distance") if "distance" in hit else hit.get("score")
        if raw_score is None:
            _log.debug(
                "Milvus hit dropped: missing score. "
                "neo4j_node_id=%s id=%s",
                entity.get("neo4j_node_id"),
                entity.get("id"),
            )
            return None

        # Stable semantic ID: prefer the Neo4j graph node ID; fall back to
        # the Milvus document ID (chunk_id) so no provenance is lost.
        operation_id = (
            _safe_str(entity.get("neo4j_node_id"))
            or _safe_str(entity.get("id"))
        )

        # Human-readable description: title first, truncated body as fallback
        title = _safe_str(entity.get("title"))
        body = entity.get("body") or ""
        description: str | None
        if title:
            description = title
        elif body:
            description = (body[:300] + "…") if len(body) > 300 else body
        else:
            description = None

        # Source encodes entity classification and spec provenance
        entity_type = _safe_str(entity.get("entity_type")) or "unknown"
        spec_id = _safe_str(entity.get("spec_id"))
        source = entity_type + (f":{spec_id}" if spec_id else "")

        method_raw = _safe_str(entity.get("method"))
        method = method_raw.upper() if method_raw else None

        return RetrievedAPIMatch(
            operation_id=operation_id,
            endpoint=_safe_str(entity.get("path")),
            method=method,
            product=_safe_str(entity.get("product_key")),
            description=description,
            score=float(raw_score),
            source=source,
        )
    except Exception as exc:
        _log.warning(
            "Failed to normalize Milvus hit — skipping: %s", exc, exc_info=False
        )
        return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def semantic_api_search(
    query: str,
    top_k: int = 10,
) -> list[RetrievedAPIMatch]:
    """Embed *query* and return the top-*top_k* matching API operations.

    Calls ``retriever.client.search()`` directly (not ``retriever.search()``)
    so that *top_k* is honoured per call rather than using the retriever's
    fixed ``config.top_k`` value.  All other search parameters (collection,
    output fields, metric) are taken from the shared retriever instance to
    avoid duplicating connection logic.
    """
    retriever = _get_retriever()
    _log.info("semantic_api_search query=%r top_k=%d", query, top_k)

    embedding = retriever.embed_query(query)

    try:
        raw_results = retriever.client.search(
            collection_name=retriever.config.collection_name,
            data=[embedding],
            limit=top_k,
            output_fields=retriever.OUTPUT_FIELDS,
            search_params={"metric_type": "COSINE"},
        )
    except Exception as exc:
        _log.error("Milvus search failed for query=%r: %s", query, exc)
        return []

    hits: list[dict[str, Any]] = raw_results[0] if raw_results else []
    matches: list[RetrievedAPIMatch] = []
    skipped = 0

    for hit in hits:
        match = _hit_to_match(hit)
        if match is not None:
            matches.append(match)
        else:
            skipped += 1

    _log.info(
        "semantic_api_search: %d matches, %d skipped | query=%r",
        len(matches),
        skipped,
        query,
    )
    return matches


def get_operation_by_id(operation_id: str) -> dict[str, Any] | None:
    """Retrieve a single Milvus document by its string ID.

    Returns the raw entity dict or None if not found.
    """
    retriever = _get_retriever()
    try:
        results = retriever.client.query(
            collection_name=retriever.config.collection_name,
            filter=f'id == "{operation_id}"',
            output_fields=["*"],
        )
        return results[0] if results else None
    except Exception as exc:
        _log.warning(
            "get_operation_by_id failed for id=%r: %s", operation_id, exc
        )
        return None
