from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# Compiled regex patterns for regex-fallback guardrail checks
# ---------------------------------------------------------------------------

# --- Input patterns (user message) ---

# Prompt injection: attempts to override the system / ignore instructions
_RE_PROMPT_INJECTION = re.compile(
    r"\b("
    r"ignore\s+(previous|all|above|prior)\s+(instructions?|prompts?|rules?)|"
    r"disregard\s+(your|all|the)\s+(instructions?|rules?|training)|"
    r"forget\s+(everything|all)\s+(you|above)|"
    r"you\s+are\s+now\s+(a\s+)?(?:different|evil|unrestricted|jailbroken)|"
    r"override\s+(your\s+)?(safety|guidelines?|restrictions?)|"
    r"act\s+as\s+if\s+you\s+(have|had)\s+no\s+(restrictions?|rules?|guidelines?)"
    r")\b",
    re.IGNORECASE,
)

# Jailbreak: DAN, roleplay-as, and similar bypass attempts
_RE_JAILBREAK = re.compile(
    r"\b("
    r"do\s+anything\s+now|"
    r"\bDAN\b|"
    r"jailbreak|"
    r"pretend\s+you\s+(are|have\s+no)|"
    r"roleplay\s+as\s+(an?\s+)?unrestricted|"
    r"in\s+(developer|god|admin|root)\s+mode|"
    r"bypass\s+(your\s+)?(safety|filter|guardrail|restriction)"
    r")\b",
    re.IGNORECASE,
)

# System prompt extraction: fishing for hidden instructions
_RE_SYSTEM_PROMPT_EXTRACTION = re.compile(
    r"\b("
    r"(show|reveal|print|display|output|tell\s+me|what\s+is)\s+"
    r"(your\s+)?(system\s+prompt|initial\s+prompt|original\s+instructions?|"
    r"base\s+prompt|hidden\s+instructions?|prompt\s+template)|"
    r"repeat\s+(the\s+)?instructions?\s+you\s+(were\s+)?given|"
    r"what\s+(are|were)\s+your\s+(exact\s+)?instructions?"
    r")\b",
    re.IGNORECASE,
)

# Credential / API key extraction
_RE_CREDENTIAL_EXTRACTION = re.compile(
    r"\b("
    r"(give|show|print|dump|output|return|list)\s+(me\s+)?(all\s+)?"
    r"(api\s+keys?|tokens?|secrets?|credentials?|passwords?|auth\s+keys?)|"
    r"extract\s+(api\s+key|token|secret|credential|password)|"
    r"(leak|expose)\s+(the\s+)?(api\s+key|secret|credential|token)"
    r")\b",
    re.IGNORECASE,
)

# Malicious API misuse: mass enumeration, scraping, DoS framing
_RE_MALICIOUS_API_MISUSE = re.compile(
    r"\b("
    r"scrape\s+(all|every|the\s+entire)|"
    r"enumerate\s+all\s+(endpoints?|apis?|resources?)|"
    r"flood\s+(the\s+)?(api|endpoint|server)|"
    r"(ddos|dos|denial.of.service)\s+(the\s+)?api|"
    r"brute.?force\s+(the\s+)?(api|endpoint|auth)|"
    r"bypass\s+(rate\s+limit|auth|authentication)|"
    r"mass\s+(download|export|harvest)\s+(all\s+)?(user\s+)?(data|records?)"
    r")\b",
    re.IGNORECASE,
)

# PII extraction
_RE_PII_EXTRACTION = re.compile(
    r"\b("
    r"(give|show|list|dump|export|return)\s+(me\s+)?(all\s+)?"
    r"(user\s+)?(emails?|phone\s+numbers?|ssn|social\s+security|"
    r"credit\s+card\s+numbers?|home\s+address(es)?|date\s+of\s+birth)|"
    r"extract\s+(pii|personal\s+(information|data))|"
    r"(list|dump)\s+all\s+users?"
    r")\b",
    re.IGNORECASE,
)

# --- Output patterns (agent response) ---

# Leaked secrets in output
_RE_OUTPUT_SECRET_LEAK = re.compile(
    r"("
    r"(?:sk|rk|pk)_(?:live|test)_[A-Za-z0-9]{10,}|"   # Stripe-style keys
    r"(?:Bearer\s+)[A-Za-z0-9\-._~+/]{20,}|"
    r"(?:api[_\-]?key|secret|token)\s*[:=]\s*['\"]?[A-Za-z0-9\-._~+/]{16,}|"
    r"-----BEGIN\s+(RSA\s+)?PRIVATE\s+KEY-----|"
    r"(?:password|passwd|pwd)\s*[:=]\s*['\"]?\S{8,}"
    r")",
    re.IGNORECASE,
)

# Output exposing internal prompts / system instructions
_RE_OUTPUT_PROMPT_EXPOSURE = re.compile(
    r"\b("
    r"(my\s+)?(system\s+prompt|initial\s+instructions?|hidden\s+instructions?)"
    r"\s+(is|are|reads?|says?)|"
    r"(the\s+)?instructions?\s+i\s+(was\s+)?given\s+(are|say|include)|"
    r"i\s+(was\s+)?instructed\s+to\s+(never|always|keep|hide)"
    r")\b",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Risk-classification helpers
# ---------------------------------------------------------------------------

# Maps Llama-Guard S-codes to a risk level.
# S1-S4, S9-S11 are hard-block; others default REVIEW.
_BLOCK_SCODES = frozenset({"S1", "S2", "S3", "S4", "S9", "S10", "S11"})


def parse_llama_guard_response(
    response: str,
) -> tuple[str, str | None]:
    """Parse a Llama-Guard 3 reply into (risk_level, category_string | None).

    Expected format::

        safe                  → ("SAFE", None)
        unsafe\\nS1, S4       → ("BLOCK", "S1, S4")   if any code in _BLOCK_SCODES
        unsafe\\nS5, S6       → ("REVIEW", "S5, S6")  softer categories

    Unknown / empty format → fail-open ("SAFE", None).
    """
    text = response.strip().lower()
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return "SAFE", None

    first = lines[0]
    if first == "safe":
        return "SAFE", None

    if first == "unsafe":
        category_str: str | None = lines[1].upper() if len(lines) > 1 else None
        if category_str:
            codes = {c.strip() for c in category_str.split(",")}
            risk = "BLOCK" if codes & _BLOCK_SCODES else "REVIEW"
        else:
            risk = "REVIEW"
        return risk, category_str

    # Unrecognised format → fail open
    return "SAFE", None


def format_conversation_for_guardrail(role: str, message: str) -> str:
    """Format a single turn for the Llama-Guard conversation block."""
    tag = "User" if role == "user" else "Agent"
    return f"{tag}: {message}"


# ---------------------------------------------------------------------------
# Regex-based fallback
# ---------------------------------------------------------------------------

_INPUT_CHECKS: list[tuple[re.Pattern[str], str, str]] = [
    # (pattern, risk_level, reason)
    (_RE_PROMPT_INJECTION,       "BLOCK",  "Prompt injection attempt detected."),
    (_RE_JAILBREAK,              "BLOCK",  "Jailbreak attempt detected."),
    (_RE_SYSTEM_PROMPT_EXTRACTION, "BLOCK", "System prompt extraction attempt."),
    (_RE_CREDENTIAL_EXTRACTION,  "BLOCK",  "Credential / API key extraction attempt."),
    (_RE_MALICIOUS_API_MISUSE,   "REVIEW", "Potentially malicious API misuse pattern."),
    (_RE_PII_EXTRACTION,         "REVIEW", "PII extraction request detected."),
]

_OUTPUT_CHECKS: list[tuple[re.Pattern[str], str, str]] = [
    (_RE_OUTPUT_SECRET_LEAK,     "BLOCK",  "Output contains a secret / API key."),
    (_RE_OUTPUT_PROMPT_EXPOSURE, "REVIEW", "Output may expose internal instructions."),
]


def check_with_regex(text: str, mode: str) -> tuple[str, str]:
    """Regex-only fallback guardrail.

    *mode* must be ``"input"`` or ``"output"``.

    Returns ``(risk_level, reason)`` — one of SAFE/REVIEW/BLOCK.
    The first matching pattern wins (patterns are ordered most-severe first).
    """
    checks = _INPUT_CHECKS if mode == "input" else _OUTPUT_CHECKS
    for pattern, risk, reason in checks:
        if pattern.search(text):
            return risk, reason
    return "SAFE", "No unsafe patterns detected."
