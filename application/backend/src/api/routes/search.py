from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from data.ingestion.graph_expansion import (
    GraphRagRetriever,
    Neo4jGraphExpander,
)
from main import ApiDiscoveryAnswerFormatter

router = APIRouter()


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=500)
    filter: str | None = Field(default=None)
    max_graph_expansions: int = Field(default=3, ge=0, le=10)


class OperationResult(BaseModel):
    id: str | None = None
    operation_id: str | None = None
    title: str | None = None
    method: str | None = None
    path: str | None = None
    summary: str | None = None
    description: str | None = None
    score: float | None = None
    source: str = "unknown"
    product_key: str | None = None
    business_domain: str | None = None


class SchemaResult(BaseModel):
    id: str | None = None
    name: str | None = None
    title: str | None = None
    type: str | None = None
    description: str | None = None
    source: str = "unknown"
    status_code: str | None = None


class SearchResponse(BaseModel):
    query: str
    answer: str
    recommended_operations: list[OperationResult]
    schemas: list[SchemaResult]
    confidence_notes: list[str]


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@router.post("/search", response_model=SearchResponse)
def search(body: SearchRequest, request: Request) -> SearchResponse:
    semantic_retriever = request.app.state.semantic_retriever
    env_path = request.app.state.env_path

    if semantic_retriever is None:
        raise HTTPException(
            status_code=503,
            detail="Semantic search unavailable: Milvus not connected.",
        )

    graph_expander = Neo4jGraphExpander.from_env(env_path=env_path)
    try:
        rag = GraphRagRetriever(
            semantic_retriever=semantic_retriever,
            graph_expander=graph_expander,
        )
        context = rag.retrieve(
            query=body.query,
            filter_expression=body.filter,
            max_graph_expansions=body.max_graph_expansions,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        graph_expander.close()

    answer = ApiDiscoveryAnswerFormatter().format_answer(context)

    operations = [
        OperationResult(**op)
        for op in context.get("recommended_operations", [])
    ]
    schemas = [
        SchemaResult(**s)
        for s in context.get("schemas", [])
    ]

    return SearchResponse(
        query=context["query"],
        answer=answer,
        recommended_operations=operations,
        schemas=schemas,
        confidence_notes=context.get("confidence_notes", []),
    )
