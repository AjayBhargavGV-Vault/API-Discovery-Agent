from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    user_message: str = Field(..., min_length=1, max_length=4000)
    conversation_id: str | None = Field(default=None)
    chat_history: list[dict[str, Any]] = Field(default_factory=list)
    top_k: int = Field(default=10, ge=1, le=50)
    include_agent_trace: bool = Field(default=True)


class ChatResponse(BaseModel):
    answer: str
    conversation_id: str
    sources: list[Any] = Field(default_factory=list)
    agent_trace: list[Any] = Field(default_factory=list)
    guardrail_status: str
    confidence_score: float | None = Field(default=None)


class QueryEnrichmentResult(BaseModel):
    original_query: str
    enriched_query: str
    detected_intent: str
    entities: list[str] = Field(default_factory=list)
    suggested_agents: list[str] = Field(default_factory=list)
    confidence_score: float


class AgentTask(BaseModel):
    task_id: str
    agent_name: str
    task_type: str
    input: dict[str, Any]
    reason: str
    model_name: str | None = Field(default=None)


class AgentResult(BaseModel):
    agent_name: str
    model_name: str | None = Field(default=None)
    success: bool
    data: dict[str, Any]
    error: str | None = Field(default=None)
    sources: list[Any] = Field(default_factory=list)


class RetrievedAPIMatch(BaseModel):
    operation_id: str | None = Field(default=None)
    endpoint: str | None = Field(default=None)
    method: str | None = Field(default=None)
    product: str | None = Field(default=None)
    description: str | None = Field(default=None)
    score: float
    source: str | None = Field(default=None)


class APIOperationDetails(BaseModel):
    operation_id: str | None = Field(default=None)
    endpoint: str | None = Field(default=None)
    method: str | None = Field(default=None)
    product_key: str | None = Field(default=None)
    business_domain: str | None = Field(default=None)
    summary: str | None = Field(default=None)
    description: str | None = Field(default=None)
    parameters: list[dict[str, Any]] = Field(default_factory=list)
    request_schema: dict[str, Any] = Field(default_factory=dict)
    response_schema: dict[str, Any] = Field(default_factory=dict)
    auth_scheme: dict[str, Any] = Field(default_factory=dict)
    error_codes: list[Any] = Field(default_factory=list)
    related_operations: list[Any] = Field(default_factory=list)


class GuardrailResult(BaseModel):
    allowed: bool
    reason: str
    safe_response: str | None = Field(default=None)
    risk_level: str
    model_name: str | None = Field(default=None)


class ValidationResult(BaseModel):
    is_grounded: bool
    unsupported_claims: list[str] = Field(default_factory=list)
    confidence_score: float
    final_answer: str
    model_name: str | None = Field(default=None)
