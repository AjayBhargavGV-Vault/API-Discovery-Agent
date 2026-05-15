from __future__ import annotations

import logging
from typing import Any

from agent_tools.milvus_tools import init_retriever, semantic_api_search
from chat.schemas import AgentResult

_log = logging.getLogger(__name__)


class SemanticRetrieverAgent:
    """Retrieves relevant API operations from Milvus via semantic search.

    No LLM inference.  Uses the embedding model configured at startup
    (EMBEDDING_MODEL in .env, typically BAAI/bge-large-en-v1.5).

    The shared MilvusRetrievalTest instance from FastAPI startup is
    registered with milvus_tools at construction time so that
    SentenceTransformer is only loaded once across the whole process.
    """

    AGENT_NAME = "semantic_retriever"

    def __init__(self, semantic_retriever: Any) -> None:
        self._retriever = semantic_retriever
        if semantic_retriever is not None:
            init_retriever(semantic_retriever)
        _log.debug(
            "%s: registered retriever (%s)",
            self.AGENT_NAME,
            type(semantic_retriever).__name__,
        )

    def run(self, query: str, top_k: int = 10) -> AgentResult:
        """Search Milvus for API operations matching *query*.

        Returns an AgentResult where:
          data["matches"]     — list of RetrievedAPIMatch dicts
          data["match_count"] — number of results returned
          sources             — deduplicated source labels (entity_type:spec_id)
        """
        _log.info("%s.run query=%r top_k=%d", self.AGENT_NAME, query, top_k)

        if self._retriever is None:
            _log.warning("%s: no Milvus retriever — skipping search", self.AGENT_NAME)
            return AgentResult(
                agent_name=self.AGENT_NAME,
                model_name=None,
                success=False,
                data={"query": query, "top_k": top_k, "match_count": 0, "matches": []},
                error="Milvus retriever not available.",
                sources=[],
            )

        try:
            matches = semantic_api_search(query, top_k=top_k)

            # Preserve unique, non-empty source labels for the agent trace
            sources: list[str] = list(
                dict.fromkeys(m.source for m in matches if m.source)
            )

            _log.info(
                "%s.run: %d match(es) | sources=%s | query=%r",
                self.AGENT_NAME,
                len(matches),
                sources,
                query,
            )

            return AgentResult(
                agent_name=self.AGENT_NAME,
                model_name=None,
                success=True,
                data={
                    "query": query,
                    "top_k": top_k,
                    "match_count": len(matches),
                    "matches": [m.model_dump() for m in matches],
                },
                sources=sources,
            )

        except Exception as exc:
            _log.error(
                "%s.run failed | query=%r: %s",
                self.AGENT_NAME,
                query,
                exc,
                exc_info=True,
            )
            return AgentResult(
                agent_name=self.AGENT_NAME,
                model_name=None,
                success=False,
                data={
                    "query": query,
                    "top_k": top_k,
                    "match_count": 0,
                    "matches": [],
                },
                error=str(exc),
                sources=[],
            )
