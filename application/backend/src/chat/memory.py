"""Lightweight in-memory conversation store.

Replacement guide (Redis / Postgres)
-------------------------------------
The public surface that callers depend on is:

    save_turn(cid, user_msg, asst_msg, metadata)
    get_chat_history(cid)          -> list[dict]
    get_last_referenced_operations(cid) -> list[str]

    # legacy (chat_engine.py)
    append(sid, role, content)
    format_for_prompt(sid)
    get_text_lines(sid)

To swap the backend:

Redis:
  Replace self._sessions with a Redis client.
  Serialize via json.dumps([t.to_dict() for t in session.turns]).
  Use per-key TTL for automatic expiry.

Postgres:
  One row per turn in a ``turns`` table
  (conversation_id, seq, user_message, assistant_response, ...).
  Query the last MAX_TURNS rows ordered by seq DESC.
"""
from __future__ import annotations

import logging
import re
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

_log = logging.getLogger(__name__)

MAX_TURNS: int = 10

# ---------------------------------------------------------------------------
# Secret masking
# ---------------------------------------------------------------------------

# Each entry: (compiled pattern, replacement string).
# Patterns use \g<1> to back-reference the first capture group.
_SECRET_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    # HTTP Authorization header value
    (
        re.compile(r"(Authorization\s*:\s*)\S+", re.IGNORECASE),
        r"\g<1><REDACTED>",
    ),
    # Bearer token in free text
    (
        re.compile(r"Bearer\s+\S+", re.IGNORECASE),
        "Bearer <REDACTED>",
    ),
    # JWT: three base64url segments separated by dots
    (
        re.compile(
            r"\beyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\b"
        ),
        "<JWT_REDACTED>",
    ),
    # Stripe / common SDK key prefixes: sk-, pk-, rk-, ak-
    (
        re.compile(r"\b(?:sk|pk|rk|ak)-[A-Za-z0-9_]{16,}"),
        "<KEY_REDACTED>",
    ),
    # key= or api_key= assignment
    (
        re.compile(r"((?:api[-_]?)?key\s*[=:]\s*)\S+", re.IGNORECASE),
        r"\g<1><REDACTED>",
    ),
    # token= assignment (auth_token=, access_token=, etc.)
    (
        re.compile(r"((?:\w+[-_])?token\s*[=:]\s*)\S+", re.IGNORECASE),
        r"\g<1><REDACTED>",
    ),
    # password= assignment
    (
        re.compile(r"(password\s*[=:]\s*)\S+", re.IGNORECASE),
        r"\g<1><REDACTED>",
    ),
    # secret= assignment
    (
        re.compile(r"(secret\s*[=:]\s*)\S+", re.IGNORECASE),
        r"\g<1><REDACTED>",
    ),
]


def _mask_secrets(text: str) -> str:
    """Replace recognised secret patterns in *text* with safe placeholders."""
    for pattern, replacement in _SECRET_PATTERNS:
        text = pattern.sub(replacement, text)
    return text


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class Turn:
    """One round-trip between user and assistant, plus pipeline metadata.

    Serialisable via to_dict() for storage backends that need JSON.
    """

    user_message: str
    assistant_response: str
    enriched_query: str | None = None
    selected_apis: list[str] = field(default_factory=list)
    operation_ids: list[str] = field(default_factory=list)
    detected_intent: str | None = None
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation of this turn."""
        return {
            "user_message": self.user_message,
            "assistant_response": self.assistant_response,
            "enriched_query": self.enriched_query,
            "selected_apis": self.selected_apis,
            "operation_ids": self.operation_ids,
            "detected_intent": self.detected_intent,
            "timestamp": self.timestamp,
        }


class ConversationSession:
    """Runtime state for a single conversation.

    Holds at most *max_turns* Turn objects in a bounded deque.
    The *pending_user* attribute buffers a user message received via
    the legacy append() interface until the matching assistant message
    arrives.
    """

    def __init__(self, max_turns: int = MAX_TURNS) -> None:
        self.turns: deque[Turn] = deque(maxlen=max_turns)
        self.pending_user: str = ""

    @property
    def last_intent(self) -> str | None:
        """Detected intent from the most recent turn, or None."""
        return self.turns[-1].detected_intent if self.turns else None

    @property
    def last_operation_ids(self) -> list[str]:
        """Operation IDs referenced in the most recent turn."""
        return list(self.turns[-1].operation_ids) if self.turns else []

    def messages(self) -> list[dict[str, str]]:
        """Flat [{role, content}] pairs derived from stored turns."""
        result: list[dict[str, str]] = []
        for turn in self.turns:
            result.append(
                {"role": "user", "content": turn.user_message}
            )
            result.append(
                {"role": "assistant", "content": turn.assistant_response}
            )
        return result


# ---------------------------------------------------------------------------
# ConversationMemory
# ---------------------------------------------------------------------------

class ConversationMemory:
    """Lightweight in-memory conversation store keyed by conversation_id.

    Each conversation keeps up to MAX_TURNS (10) full turns. Secrets in
    user messages and assistant responses are masked before storage.

    See the module docstring for Redis / Postgres replacement notes.
    """

    def __init__(self, max_turns: int = MAX_TURNS) -> None:
        self.max_turns = max_turns
        self._sessions: dict[str, ConversationSession] = {}

    def _session(self, conversation_id: str) -> ConversationSession:
        """Return (or lazily create) the session for *conversation_id*."""
        if conversation_id not in self._sessions:
            self._sessions[conversation_id] = ConversationSession(
                max_turns=self.max_turns
            )
        return self._sessions[conversation_id]

    # ------------------------------------------------------------------
    # Primary interface
    # ------------------------------------------------------------------

    def save_turn(
        self,
        conversation_id: str,
        user_message: str,
        assistant_response: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Persist one complete conversation exchange.

        Args:
            conversation_id:    Session key.
            user_message:       Raw user query; secrets are masked.
            assistant_response: Agent answer; secrets are masked.
            metadata:           Optional dict with any subset of:
                                  ``enriched_query``  – str
                                  ``selected_apis``   – list[str]
                                  ``operation_ids``   – list[str]
                                  ``detected_intent`` – str
        """
        meta = metadata or {}
        turn = Turn(
            user_message=_mask_secrets(user_message),
            assistant_response=_mask_secrets(assistant_response),
            enriched_query=meta.get("enriched_query"),
            selected_apis=list(meta.get("selected_apis") or []),
            operation_ids=list(meta.get("operation_ids") or []),
            detected_intent=meta.get("detected_intent"),
        )
        session = self._session(conversation_id)
        session.turns.append(turn)
        _log.debug(
            "memory.save_turn cid=%s total_turns=%d",
            conversation_id,
            len(session.turns),
        )

    def get_chat_history(
        self,
        conversation_id: str,
    ) -> list[dict[str, Any]]:
        """Return all stored turns as a list of dicts.

        Each dict contains: user_message, assistant_response,
        enriched_query, selected_apis, operation_ids,
        detected_intent, timestamp.

        Returns an empty list for unknown conversation IDs.
        """
        session = self._sessions.get(conversation_id)
        if session is None:
            return []
        return [t.to_dict() for t in session.turns]

    def get_last_referenced_operations(
        self,
        conversation_id: str,
    ) -> list[str]:
        """Return operation_ids from the most recent turn.

        Returns an empty list when there are no turns or the last
        turn carried no operation IDs.
        """
        session = self._sessions.get(conversation_id)
        if session is None:
            return []
        return session.last_operation_ids

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------

    def get_last_intent(self, conversation_id: str) -> str | None:
        """Return the detected_intent from the most recent turn."""
        session = self._sessions.get(conversation_id)
        return session.last_intent if session else None

    def get_messages(
        self,
        conversation_id: str,
    ) -> list[dict[str, str]]:
        """Return flat [{role, content}] pairs for LLM prompt injection."""
        session = self._sessions.get(conversation_id)
        return session.messages() if session else []

    # ------------------------------------------------------------------
    # Legacy interface — chat_engine.py compatibility
    # ------------------------------------------------------------------

    def append(
        self,
        session_id: str,
        role: str,
        content: str,
    ) -> None:
        """Buffer a single message; pair user+assistant into a Turn.

        chat_engine.py calls ``append(sid, "user", ...)`` then
        ``append(sid, "assistant", ...)`` on each request.  A Turn is
        only committed when the assistant message arrives.
        """
        session = self._session(session_id)
        masked = _mask_secrets(content)
        if role == "user":
            session.pending_user = masked
        elif role == "assistant":
            session.turns.append(
                Turn(
                    user_message=session.pending_user,
                    assistant_response=masked,
                )
            )
            session.pending_user = ""
        else:
            _log.warning(
                "memory.append: unexpected role=%r for sid=%s",
                role,
                session_id,
            )

    def get_turns(
        self,
        session_id: str,
        last_n: int | None = None,
    ) -> list[dict[str, str]]:
        """Legacy: flat [{role, content}] messages, optionally sliced."""
        messages = self.get_messages(session_id)
        return messages[-last_n:] if last_n else messages

    def get_text_lines(
        self,
        session_id: str,
        last_n: int = 6,
    ) -> list[str]:
        """Legacy: ``Role: content`` strings for history injection."""
        return [
            f"{t['role'].capitalize()}: {t['content']}"
            for t in self.get_turns(session_id, last_n)
        ]

    def format_for_prompt(
        self,
        session_id: str,
        last_n: int = 6,
    ) -> str:
        """Legacy: newline-joined history string for LLM prompts."""
        lines = self.get_text_lines(session_id, last_n)
        return "\n".join(lines) if lines else "None"

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def clear(self, session_id: str) -> None:
        """Evict all state for a session."""
        self._sessions.pop(session_id, None)

    def session_count(self) -> int:
        """Return the number of active sessions."""
        return len(self._sessions)
