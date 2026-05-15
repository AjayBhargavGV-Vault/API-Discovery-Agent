from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Literal

from dotenv import load_dotenv

from llm.base_client import BaseLLMClient, ChatMessage, LLMConfig

_log = logging.getLogger(__name__)

# Default: application/backend/.env (two levels above src/llm/)
DEFAULT_ENV_PATH = Path(__file__).resolve().parents[2] / ".env"

BackendType = Literal["vllm", "ollama", "openai_compatible"]

# Temperature per agent role
_TEMPERATURES: dict[str, float] = {
    "orchestrator": 0.1,
    "orchestrator_fallback": 0.1,
    "query_enrichment": 0.2,
    "guardrail": 0.0,
    "validation": 0.0,
    "validation_fallback": 0.0,
    "code_example": 0.1,
    "code_fallback": 0.1,
    "fallback": 0.1,
}

# .env key that holds each agent's model name
_MODEL_ENV_KEYS: dict[str, str] = {
    "orchestrator": "ORCHESTRATOR_MODEL",
    "orchestrator_fallback": "ORCHESTRATOR_FALLBACK_MODEL",
    "query_enrichment": "QUERY_ENRICHMENT_MODEL",
    "guardrail": "GUARDRAIL_MODEL",
    "validation": "VALIDATION_MODEL",
    "validation_fallback": "VALIDATION_FALLBACK_MODEL",
    "code_example": "CODE_MODEL",
    "code_fallback": "CODE_FALLBACK_MODEL",
    "fallback": "FALLBACK_MODEL",
}

# Defaults used when the env key is absent
_MODEL_DEFAULTS: dict[str, str] = {
    "orchestrator": "qwen2.5-32b-instruct",
    "orchestrator_fallback": "qwen2.5-14b-instruct",
    "query_enrichment": "qwen2.5-14b-instruct",
    "guardrail": "llama-guard-3-8b",
    "validation": "qwen2.5-7b-instruct",
    "validation_fallback": "qwen2.5-14b-instruct",
    "code_example": "deepseek-coder-v2-lite-instruct",
    "code_fallback": "qwen2.5-coder-14b-instruct",
    "fallback": "qwen2.5-7b-instruct",
}

# Base URL env key per backend
_BACKEND_URL_ENV: dict[str, str] = {
    "vllm": "VLLM_BASE_URL",
    "ollama": "OLLAMA_BASE_URL",
    "openai_compatible": "OPENAI_BASE_URL",
}

# Default base URLs per backend
_BACKEND_URL_DEFAULTS: dict[str, str] = {
    "vllm": "http://localhost:8000/v1",
    "ollama": "http://localhost:11434",
    "openai_compatible": "http://localhost:1234/v1",
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_env(env_path: Path) -> None:
    if env_path.exists():
        load_dotenv(env_path, override=False)
    else:
        load_dotenv(override=False)


def _backend() -> BackendType:
    raw = os.environ.get("LLM_BACKEND", "vllm").lower().strip()
    if raw in ("vllm", "ollama", "openai_compatible"):
        return raw  # type: ignore[return-value]
    _log.warning(
        "Unknown LLM_BACKEND=%r; falling back to 'vllm'.", raw
    )
    return "vllm"


def _base_url(backend: BackendType, agent_name: str) -> str:
    # Agent-specific override takes precedence
    agent_key = agent_name.upper().replace("-", "_")
    if agent_name and (specific := os.environ.get(f"{agent_key}_BASE_URL")):
        return specific
    env_key = _BACKEND_URL_ENV[backend]
    return os.environ.get(env_key, _BACKEND_URL_DEFAULTS[backend])


def _api_key(backend: BackendType) -> str | None:
    if backend == "openai_compatible":
        return os.environ.get("OPENAI_API_KEY") or None
    return None


def _build_config(agent_name: str, backend: BackendType) -> LLMConfig:
    model_key = _MODEL_ENV_KEYS.get(agent_name, "FALLBACK_MODEL")
    model = os.environ.get(model_key, _MODEL_DEFAULTS.get(agent_name, ""))
    agent_key = agent_name.upper().replace("-", "_") if agent_name else ""
    return LLMConfig(
        model=model,
        base_url=_base_url(backend, agent_name),
        temperature=_TEMPERATURES.get(agent_name, 0.1),
        max_tokens=int(os.environ.get(f"{agent_key}_MAX_TOKENS", "2048")),
        timeout=float(os.environ.get(f"{agent_key}_TIMEOUT", "60.0")),
        max_retries=int(os.environ.get("LLM_MAX_RETRIES", "3")),
        retry_delay=float(os.environ.get("LLM_RETRY_DELAY", "1.0")),
        api_key=_api_key(backend),
    )


def _make_client(config: LLMConfig) -> BaseLLMClient:
    backend = _backend()
    if backend == "ollama":
        from llm.ollama_client import OllamaClient
        return OllamaClient(config)
    # vllm and openai_compatible both speak the OpenAI-compatible protocol
    from llm.vllm_client import VllmClient
    return VllmClient(config)


# ---------------------------------------------------------------------------
# Primary public API
# ---------------------------------------------------------------------------

def get_model_client(
    agent_name: str,
    *,
    env_path: Path | None = None,
) -> BaseLLMClient:
    """Return a fully configured LLM client for *agent_name*.

    Reads backend and model config from the .env file.  Falls back to
    FALLBACK_MODEL if the agent's model string resolves to empty, or if
    client construction raises.
    """
    _load_env(env_path or DEFAULT_ENV_PATH)
    backend = _backend()
    config = _build_config(agent_name, backend)

    if not config.model:
        _log.warning(
            "No model configured for agent '%s'; using fallback.",
            agent_name,
        )
        config = _build_config("fallback", backend)

    try:
        client = _make_client(config)
        _log.debug(
            "Built %s for agent='%s' model='%s' url='%s'",
            type(client).__name__,
            agent_name,
            config.model,
            config.base_url,
        )
        return client
    except Exception as exc:
        _log.error(
            "Failed to create client for '%s' (model=%s): %s — "
            "retrying with fallback model.",
            agent_name,
            config.model,
            exc,
        )
        return _make_client(_build_config("fallback", backend))


def chat_completion(
    messages: list[ChatMessage],
    model: str,
    temperature: float = 0.1,
    max_tokens: int = 2048,
    *,
    env_path: Path | None = None,
) -> str:
    """Module-level one-off chat completion for an explicit *model* string.

    Uses the backend and base URL from .env.  No agent name resolution —
    pass the model ID directly (e.g. "qwen2.5-14b-instruct").
    """
    _load_env(env_path or DEFAULT_ENV_PATH)
    backend = _backend()
    config = LLMConfig(
        model=model,
        base_url=_base_url(backend, ""),
        temperature=temperature,
        max_tokens=max_tokens,
        api_key=_api_key(backend),
    )
    return _make_client(config).chat_completion(
        messages, temperature=temperature, max_tokens=max_tokens
    )


def json_completion(
    messages: list[ChatMessage],
    model: str,
    temperature: float = 0.1,
    max_tokens: int = 2048,
    retries: int = 2,
    *,
    env_path: Path | None = None,
) -> Any:
    """Module-level one-off JSON completion for an explicit *model* string.

    Retries *retries* times on malformed JSON, appending a self-correction
    prompt each time.
    """
    _load_env(env_path or DEFAULT_ENV_PATH)
    backend = _backend()
    config = LLMConfig(
        model=model,
        base_url=_base_url(backend, ""),
        temperature=temperature,
        max_tokens=max_tokens,
        max_retries=retries,
        api_key=_api_key(backend),
    )
    return _make_client(config).json_completion(
        messages,
        temperature=temperature,
        max_tokens=max_tokens,
        retries=retries,
    )
