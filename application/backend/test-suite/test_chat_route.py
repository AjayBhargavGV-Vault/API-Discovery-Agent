"""Tests for api/routes/chat.py.

Test case 7 — Chat route returns a ChatResponse.

Strategy: call the route functions directly as plain Python functions,
passing a minimal ``FakeRequest`` that exposes ``request.app.state``.
This avoids any HTTP stack and does not require ``httpx`` or a running
server.  ``FastAPI.HTTPException`` is caught just like FastAPI would
catch it for a real client.
"""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

PROJECT_SRC = Path(__file__).resolve().parents[1] / "src"
if str(PROJECT_SRC) not in sys.path:
    sys.path.append(str(PROJECT_SRC))

from fastapi import HTTPException
from fastapi.responses import JSONResponse

from api.routes.chat import chat, chat_health
from chat.schemas import ChatRequest, ChatResponse


# ---------------------------------------------------------------------------
# Fake FastAPI request infrastructure
# ---------------------------------------------------------------------------

class _State:
    def __init__(self, chat_engine=None, env_path=None) -> None:
        self.chat_engine = chat_engine
        self.env_path = env_path


class _FakeApp:
    def __init__(self, chat_engine=None, env_path=None) -> None:
        self.state = _State(chat_engine, env_path)


class _FakeRequest:
    def __init__(self, chat_engine=None, env_path=None) -> None:
        self.app = _FakeApp(chat_engine, env_path)


# ---------------------------------------------------------------------------
# Fake ChatEngine
# ---------------------------------------------------------------------------

class _FakeChatEngine:
    """Returns a canned ChatResponse for any request."""

    def run_chat_engine(self, request: ChatRequest) -> ChatResponse:
        return ChatResponse(
            answer="Use POST /v1/payment_intents to create a payment.",
            conversation_id=request.conversation_id or "test-cid",
            sources=["spec:POST:/v1/payment_intents"],
            agent_trace=[{"agent_name": "semantic_retriever"}],
            guardrail_status="SAFE",
            confidence_score=0.91,
        )


class _BrokenChatEngine:
    def run_chat_engine(self, request: ChatRequest) -> ChatResponse:
        raise RuntimeError("unexpected internal error")


# ---------------------------------------------------------------------------
# Test case 7 — POST /api/chat returns ChatResponse
# ---------------------------------------------------------------------------

class ChatEndpointTests(unittest.TestCase):

    def _req(self, engine=None):
        return _FakeRequest(chat_engine=engine or _FakeChatEngine())

    def test_returns_chat_response_instance(self) -> None:
        body = ChatRequest(user_message="How do I create a payment?")
        result = chat(body=body, request=self._req())
        self.assertIsInstance(result, ChatResponse)

    def test_answer_field_populated(self) -> None:
        body = ChatRequest(user_message="How do I create a payment?")
        result = chat(body=body, request=self._req())
        self.assertTrue(result.answer)

    def test_conversation_id_echoed(self) -> None:
        body = ChatRequest(
            user_message="test", conversation_id="my-cid"
        )
        result = chat(body=body, request=self._req())
        self.assertEqual(result.conversation_id, "my-cid")

    def test_new_conversation_id_generated_when_none(self) -> None:
        body = ChatRequest(user_message="test")
        result = chat(body=body, request=self._req())
        self.assertTrue(result.conversation_id)

    def test_sources_field_is_list(self) -> None:
        body = ChatRequest(user_message="test")
        result = chat(body=body, request=self._req())
        self.assertIsInstance(result.sources, list)

    def test_guardrail_status_present(self) -> None:
        body = ChatRequest(user_message="test")
        result = chat(body=body, request=self._req())
        self.assertIsNotNone(result.guardrail_status)

    def test_confidence_score_present(self) -> None:
        body = ChatRequest(user_message="test")
        result = chat(body=body, request=self._req())
        self.assertIsNotNone(result.confidence_score)


class ChatEndpointErrorTests(unittest.TestCase):

    def test_returns_503_when_engine_not_initialised(self) -> None:
        body = ChatRequest(user_message="test")
        req = _FakeRequest(chat_engine=None)
        with self.assertRaises(HTTPException) as ctx:
            chat(body=body, request=req)
        self.assertEqual(ctx.exception.status_code, 503)

    def test_returns_500_on_engine_exception(self) -> None:
        body = ChatRequest(user_message="test")
        req = _FakeRequest(chat_engine=_BrokenChatEngine())
        with self.assertRaises(HTTPException) as ctx:
            chat(body=body, request=req)
        self.assertEqual(ctx.exception.status_code, 500)

    def test_500_detail_does_not_leak_internals(self) -> None:
        body = ChatRequest(user_message="test")
        req = _FakeRequest(chat_engine=_BrokenChatEngine())
        with self.assertRaises(HTTPException) as ctx:
            chat(body=body, request=req)
        # Must not expose the raw exception message
        self.assertNotIn(
            "unexpected internal error", ctx.exception.detail
        )


# ---------------------------------------------------------------------------
# GET /api/chat/health
# ---------------------------------------------------------------------------

_UNSET = object()


class HealthEndpointTests(unittest.TestCase):

    def _body(self, engine=_UNSET) -> dict:
        if engine is _UNSET:
            engine = _FakeChatEngine()
        req = _FakeRequest(chat_engine=engine)
        resp = chat_health(request=req)
        self.assertIsInstance(resp, JSONResponse)
        return json.loads(resp.body)

    def test_status_is_ok(self) -> None:
        body = self._body()
        self.assertEqual(body["status"], "ok")

    def test_chat_engine_ready_when_initialised(self) -> None:
        body = self._body(engine=_FakeChatEngine())
        self.assertEqual(body["chat_engine"], "ready")

    def test_chat_engine_not_initialised_when_none(self) -> None:
        body = self._body(engine=None)
        self.assertIn("not", body["chat_engine"])

    def test_models_key_present(self) -> None:
        body = self._body()
        self.assertIn("models", body)

    def test_models_has_orchestrator(self) -> None:
        body = self._body()
        self.assertIn("orchestrator", body["models"])

    def test_models_has_guardrail(self) -> None:
        body = self._body()
        self.assertIn("guardrail", body["models"])

    def test_models_has_validation(self) -> None:
        body = self._body()
        self.assertIn("validation", body["models"])

    def test_models_has_code(self) -> None:
        body = self._body()
        self.assertIn("code", body["models"])

    def test_models_has_query_enrichment(self) -> None:
        body = self._body()
        self.assertIn("query_enrichment", body["models"])

    def test_model_values_are_non_empty_strings(self) -> None:
        body = self._body()
        for key, value in body["models"].items():
            with self.subTest(model=key):
                self.assertIsInstance(value, str)
                self.assertTrue(value)

    def test_health_always_returns_200_even_without_engine(
        self,
    ) -> None:
        # Health must never raise; it just reports status.
        req = _FakeRequest(chat_engine=None)
        resp = chat_health(request=req)
        self.assertIsInstance(resp, JSONResponse)
        body = json.loads(resp.body)
        self.assertEqual(body["status"], "ok")


if __name__ == "__main__":
    unittest.main()
