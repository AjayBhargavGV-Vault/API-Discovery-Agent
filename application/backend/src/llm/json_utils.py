from __future__ import annotations

import json
import re
from typing import Any

# Matches a fenced code block containing a JSON object or array
_FENCE_RE = re.compile(
    r"```(?:json)?\s*([\[{].*?[}\]])\s*```",
    re.DOTALL,
)
_OBJ_RE = re.compile(r"(\{.*\})", re.DOTALL)
_ARR_RE = re.compile(r"(\[.*\])", re.DOTALL)


def extract_json(text: str) -> Any:
    """Extract and parse a JSON value from raw LLM output.

    Attempts (in order):
    1. Direct parse of the full text.
    2. First ```json … ``` or ``` … ``` fence block.
    3. First bare {...} or [...] span.

    Raises ValueError when none succeed.
    """
    text = text.strip()

    # 1 — try direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 2 — fenced block
    m = _FENCE_RE.search(text)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass

    # 3 — bare object / array
    for pattern in (_OBJ_RE, _ARR_RE):
        m = pattern.search(text)
        if m:
            try:
                return json.loads(m.group(1))
            except json.JSONDecodeError:
                pass

    raise ValueError(
        f"No valid JSON found in LLM output: {text[:300]!r}"
    )


def safe_extract_json(text: str, default: Any = None) -> Any:
    """Like extract_json but returns *default* instead of raising."""
    try:
        return extract_json(text)
    except ValueError:
        return default
