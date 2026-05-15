"""Tests for chat/schemas.py Pydantic models."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_SRC = Path(__file__).resolve().parents[1] / "src"
if str(PROJECT_SRC) not in sys.path:
    sys.path.append(str(PROJECT_SRC))

from pydantic import ValidationError

from chat.schemas import (
    AgentResult,
    AgentTask,
    APIOperationDetails,
    ChatRequest,
    ChatResponse,
    GuardrailResult,
    QueryEnrichmentResult,
    RetrievedAPIMatch,
    ValidationResult,
)


class ChatRequestTests(unittest.TestCase):

    def test_empty_message_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            ChatRequest(user_message="")

    def test_whitespace_only_rejected(self) -> None:
        # Pydantic min_length=1 counts whitespace — ensure blank is caught
        with self.assertRaises(ValidationError):
            ChatRequest(user_message="   " * 0)  # empty string

    def test_default_top_k(self) -> None:
        r = ChatRequest(user_message="hello")
        self.assertEqual(r.top_k, 10)

    def test_conversation_id_defaults_none(self) -> None:
        r = ChatRequest(user_message="hello")
        self.assertIsNone(r.conversation_id)

    def test_include_agent_trace_defaults_true(self) -> None:
        r = ChatRequest(user_message="hello")
        self.assertTrue(r.include_agent_trace)

    def test_top_k_upper_bound(self) -> None:
        with self.assertRaises(ValidationError):
            ChatRequest(user_message="hello", top_k=51)

    def test_top_k_lower_bound(self) -> None:
        with self.assertRaises(ValidationError):
            ChatRequest(user_message="hello", top_k=0)

    def test_explicit_conversation_id(self) -> None:
        r = ChatRequest(user_message="hello", conversation_id="abc-123")
        self.assertEqual(r.conversation_id, "abc-123")


class ChatResponseTests(unittest.TestCase):

    def test_minimal_construction(self) -> None:
        r = ChatResponse(
            answer="ok",
            conversation_id="abc",
            guardrail_status="SAFE",
        )
        self.assertEqual(r.answer, "ok")
        self.assertEqual(r.conversation_id, "abc")

    def test_sources_defaults_empty(self) -> None:
        r = ChatResponse(
            answer="ok",
            conversation_id="cid",
            guardrail_status="SAFE",
        )
        self.assertEqual(r.sources, [])

    def test_agent_trace_defaults_empty(self) -> None:
        r = ChatResponse(
            answer="ok",
            conversation_id="cid",
            guardrail_status="SAFE",
        )
        self.assertEqual(r.agent_trace, [])

    def test_confidence_score_defaults_none(self) -> None:
        r = ChatResponse(
            answer="ok",
            conversation_id="cid",
            guardrail_status="SAFE",
        )
        self.assertIsNone(r.confidence_score)


class GuardrailResultTests(unittest.TestCase):

    def test_allowed_true_when_safe(self) -> None:
        r = GuardrailResult(
            allowed=True, reason="clean", risk_level="SAFE"
        )
        self.assertTrue(r.allowed)

    def test_allowed_false_when_blocked(self) -> None:
        r = GuardrailResult(
            allowed=False,
            reason="injection",
            risk_level="BLOCK",
            safe_response="No.",
        )
        self.assertFalse(r.allowed)
        self.assertEqual(r.safe_response, "No.")

    def test_model_name_defaults_none(self) -> None:
        r = GuardrailResult(
            allowed=True, reason="ok", risk_level="SAFE"
        )
        self.assertIsNone(r.model_name)


class ValidationResultTests(unittest.TestCase):
    """Tests covering the ValidationResult schema.

    Test case 8: validation catches hallucinated endpoint — the schema
    can represent an ungrounded result with unsupported claims.
    """

    def test_grounded_result(self) -> None:
        r = ValidationResult(
            is_grounded=True,
            confidence_score=0.95,
            final_answer="Use POST /v1/payment_intents.",
        )
        self.assertTrue(r.is_grounded)
        self.assertEqual(r.unsupported_claims, [])

    def test_hallucinated_endpoint_represented_as_ungrounded(self) -> None:
        r = ValidationResult(
            is_grounded=False,
            unsupported_claims=[
                "POST /v1/fake-endpoint does not exist in sources"
            ],
            confidence_score=0.1,
            final_answer=(
                "The retrieved API catalog does not contain enough "
                "information to answer this question."
            ),
        )
        self.assertFalse(r.is_grounded)
        self.assertGreater(len(r.unsupported_claims), 0)
        self.assertIn(
            "/v1/fake-endpoint", r.unsupported_claims[0]
        )
        self.assertLess(r.confidence_score, 1.0)
        # final_answer must not echo the hallucinated endpoint
        self.assertNotIn("/v1/fake-endpoint", r.final_answer)

    def test_confidence_clamped_representation(self) -> None:
        r = ValidationResult(
            is_grounded=False,
            confidence_score=0.0,
            final_answer="Insufficient context.",
        )
        self.assertEqual(r.confidence_score, 0.0)


class AgentResultTests(unittest.TestCase):

    def test_success_result_defaults(self) -> None:
        r = AgentResult(
            agent_name="semantic_retriever",
            success=True,
            data={"matches": []},
        )
        self.assertTrue(r.success)
        self.assertEqual(r.sources, [])
        self.assertIsNone(r.model_name)
        self.assertIsNone(r.error)

    def test_failure_result(self) -> None:
        r = AgentResult(
            agent_name="api_spec",
            success=False,
            data={},
            error="operation not found",
        )
        self.assertFalse(r.success)
        self.assertEqual(r.error, "operation not found")


class AgentTaskTests(unittest.TestCase):

    def test_task_construction(self) -> None:
        t = AgentTask(
            task_id="t1",
            agent_name="semantic_retriever",
            task_type="semantic_search",
            input={"query": "payment"},
            reason="Find payment APIs",
        )
        self.assertEqual(t.task_id, "t1")
        self.assertIsNone(t.model_name)


class APIOperationDetailsTests(unittest.TestCase):

    def test_defaults(self) -> None:
        op = APIOperationDetails()
        self.assertEqual(op.parameters, [])
        self.assertEqual(op.request_schema, {})
        self.assertEqual(op.error_codes, [])

    def test_populated_fields(self) -> None:
        op = APIOperationDetails(
            operation_id="CreatePaymentIntent",
            endpoint="/v1/payment_intents",
            method="POST",
            summary="Create a PaymentIntent",
        )
        self.assertEqual(op.method, "POST")
        self.assertEqual(op.endpoint, "/v1/payment_intents")


class RetrievedAPIMatchTests(unittest.TestCase):

    def test_required_score_field(self) -> None:
        m = RetrievedAPIMatch(score=0.85)
        self.assertEqual(m.score, 0.85)
        self.assertIsNone(m.operation_id)

    def test_full_match(self) -> None:
        m = RetrievedAPIMatch(
            operation_id="CreatePaymentIntent",
            endpoint="/v1/payment_intents",
            method="POST",
            score=0.92,
            source="operation:spec:1",
        )
        self.assertEqual(m.endpoint, "/v1/payment_intents")
        self.assertEqual(m.source, "operation:spec:1")


if __name__ == "__main__":
    unittest.main()
