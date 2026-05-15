from __future__ import annotations

import json
import re
from typing import Any

from chat.schemas import APIOperationDetails

_FENCE_RE = re.compile(r"```(?:\w+)?\s*(.*?)\s*```", re.DOTALL)

_PLACEHOLDERS_NOTE = (
    "Use ONLY these placeholders for unknown runtime values: "
    "<API_KEY>, <RESOURCE_ID>, <CUSTOMER_ID>, <PAYMENT_INTENT_ID>."
)

CODE_EXAMPLE_SYSTEM = """\
You are a precise API documentation generator.

Rules (non-negotiable):
1. Use ONLY the endpoint path, HTTP method, and fields provided in the spec.
2. Never invent, guess, or add any endpoint, path segment, query parameter, \
or request field that is not present in the spec.
3. Never modify or guess the HTTP method.
4. For unknown runtime values use ONLY these placeholders: \
<API_KEY>, <RESOURCE_ID>, <CUSTOMER_ID>, <PAYMENT_INTENT_ID>.
5. If the spec is missing an endpoint or HTTP method, output exactly: \
The spec does not provide enough detail to generate this example.
6. Return ONLY the requested code or JSON — no prose, no explanation."""


def _format_spec(op: APIOperationDetails) -> str:
    method = (op.method or "").upper()
    endpoint = op.endpoint or ""
    summary = op.summary or op.description or ""

    req_schema = (
        json.dumps(op.request_schema, indent=2)
        if op.request_schema
        else "(none provided)"
    )
    auth = (
        json.dumps(op.auth_scheme, indent=2)
        if op.auth_scheme
        else "(none provided)"
    )
    params = (
        json.dumps(op.parameters, indent=2)
        if op.parameters
        else "(none provided)"
    )

    return (
        f"Operation: {op.operation_id or '(unknown)'}\n"
        f"Method:    {method or '(unknown)'}\n"
        f"Endpoint:  {endpoint or '(unknown)'}\n"
        f"Summary:   {summary}\n"
        f"\nURL parameters:\n{params}\n"
        f"\nRequest body schema:\n{req_schema}\n"
        f"\nAuthentication:\n{auth}"
    )


def generate_curl(op: APIOperationDetails) -> str:
    """Build the user-message prompt for curl command generation."""
    return (
        f"Spec:\n{_format_spec(op)}\n\n"
        f"{_PLACEHOLDERS_NOTE}\n\n"
        f"Generate a single curl command for this operation. "
        f"Include Content-Type and Authorization headers. "
        f"Return it inside a ```sh``` code fence."
    )


def generate_python_example(op: APIOperationDetails) -> str:
    """Build the user-message prompt for a Python requests example."""
    return (
        f"Spec:\n{_format_spec(op)}\n\n"
        f"{_PLACEHOLDERS_NOTE}\n\n"
        f"Generate a minimal Python example using the `requests` library. "
        f"Print the response status code and JSON body. "
        f"Keep it under 30 lines. "
        f"Return it inside a ```python``` code fence."
    )


def generate_node_example(op: APIOperationDetails) -> str:
    """Build the user-message prompt for a Node.js fetch example."""
    return (
        f"Spec:\n{_format_spec(op)}\n\n"
        f"{_PLACEHOLDERS_NOTE}\n\n"
        f"Generate a minimal Node.js example using the built-in fetch API "
        f"(Node 18+, no external dependencies). "
        f"Use async/await. Log the response status and body. "
        f"Keep it under 30 lines. "
        f"Return it inside a ```javascript``` code fence."
    )


def generate_sample_payload(op: APIOperationDetails) -> str:
    """Build the user-message prompt for a sample JSON request payload."""
    return (
        f"Spec:\n{_format_spec(op)}\n\n"
        f"{_PLACEHOLDERS_NOTE}\n\n"
        f"Generate a sample JSON request body for this operation. "
        f"Use the placeholder strings above for unknown values. "
        f"Return ONLY valid JSON inside a ```json``` code fence."
    )


def extract_code_block(response: str) -> str:
    """Return the first fenced code block, or the full response if none."""
    m = _FENCE_RE.search(response)
    return m.group(1).strip() if m else response.strip()


# ---------------------------------------------------------------------------
# Legacy helper kept for backward compatibility with chat_engine.py
# ---------------------------------------------------------------------------

def build_code_generation_prompt(
    operation: dict[str, Any],
    language: str = "python",
    base_url: str = "https://api.example.com",
) -> str:
    method = (operation.get("method") or "GET").upper()
    path = operation.get("path") or "/"
    summary = operation.get("summary") or operation.get("description") or ""
    op_id = operation.get("operation_id") or operation.get("id") or ""

    return (
        f"Write a minimal {language} code example that calls:\n"
        f"  {method} {base_url}{path}\n"
        f"Operation ID: {op_id}\n"
        f"Description: {summary}\n\n"
        f"Requirements:\n"
        f"- Use the standard HTTP library for {language}\n"
        f"- Include a placeholder for the authorization token\n"
        f"- Print the HTTP response status code\n"
        f"- Keep the example under 25 lines\n\n"
        f"Return ONLY the code block, no explanation."
    )
