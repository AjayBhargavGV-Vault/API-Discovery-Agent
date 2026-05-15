from __future__ import annotations

import logging
import os

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

from chat.chat_engine import ChatEngine
from chat.schemas import ChatRequest, ChatResponse

_log = logging.getLogger(__name__)

router = APIRouter()

# env-key → (display_name, default)
_MODEL_KEYS: list[tuple[str, str, str]] = [
    ("ORCHESTRATOR_MODEL",     "orchestrator",     "qwen2.5-32b-instruct"),
    ("QUERY_ENRICHMENT_MODEL", "query_enrichment", "qwen2.5-14b-instruct"),
    ("GUARDRAIL_MODEL",        "guardrail",        "llama-guard-3-8b"),
    ("VALIDATION_MODEL",       "validation",       "qwen2.5-7b-instruct"),
    (
        "CODE_MODEL", "code", "deepseek-coder-v2-lite-instruct",
    ),
]


def _get_engine(request: Request) -> ChatEngine:
    engine: ChatEngine | None = getattr(
        request.app.state, "chat_engine", None
    )
    if engine is None:
        raise HTTPException(
            status_code=503,
            detail="Chat engine is not initialised.",
        )
    return engine


def _model_names() -> dict[str, str]:
    return {
        display: os.getenv(env_key, default)
        for env_key, display, default in _MODEL_KEYS
    }


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.get("/chat/health")
def chat_health(request: Request) -> JSONResponse:
    """Return chat-engine readiness and configured model names."""
    engine_status = (
        "ready"
        if getattr(request.app.state, "chat_engine", None) is not None
        else "not initialised"
    )
    return JSONResponse(
        content={
            "status": "ok",
            "chat_engine": engine_status,
            "models": _model_names(),
        }
    )


@router.post("/chat", response_model=ChatResponse)
def chat(body: ChatRequest, request: Request) -> ChatResponse:
    """Run the full agentic chat pipeline and return a response."""
    engine = _get_engine(request)
    try:
        return engine.run_chat_engine(body)
    except Exception as exc:
        _log.error(
            "chat endpoint error cid=%s: %s",
            body.conversation_id,
            exc,
            exc_info=True,
        )
        raise HTTPException(
            status_code=500,
            detail="An internal error occurred. Please try again.",
        )
