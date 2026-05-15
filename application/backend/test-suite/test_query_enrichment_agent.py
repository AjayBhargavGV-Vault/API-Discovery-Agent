"""Tests for agents/query_enrichment_agent.py.

Test case 3 — Query enrichment returns a valid QueryEnrichmentResult.

Mocking strategy: patch ``agents.query_enrichment_agent.get_model_client``
so no live model is required.  A ``_FakeLLMClient`` is injected that
returns a pre-built JSON string.  When the client raises, the agent
must fall back to a passthrough result.
"""
from __future__ import annotations

import json
import sys
import unittest
import unittest.mock
from pathlib import Path

PROJECT_SRC = Path(__file__).resolve().parents[1] / "src"
if str(PROJECT_SRC) not in sys.path:
    sys.path.append(str(PROJECT_SRC))

from agents.query_enrichment_agent import QueryEnrichmentAgent, QueryEnrichmentResult

# ---------------------------------------------------------------------------
# Fake LLM client
# ---------------------------------------------------------------------------

class _FakeConfig:
    model = "fake-qe-model"


class _FakeLLMClient:
    config = _FakeConfig()

    def __init__(self, response: str) -> None:
        self._response = response

    def chat_completion(self, messages, **kwargs) -> str:
        return self._response


class _FailingLLMClient(_FakeLLMClient):
    def chat_completion(self, messages, **kwargs) -> str:
        raise RuntimeError("LLM unreachable in tests")


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

_VALID_JSON_RESPONSE = json.dumps({
    "original_query": "How do I charge a customer?",
    "enriched_query": (
        "Find APIs for creating a PaymentIntent or charge "
        "for a customer via Stripe"
    ),
    "detected_intent": "api_discovery",
    "entities": ["payment", "customer", "charge"],
    "suggested_agents": ["semantic_retriever", "graph_retriever"],
    "confidence_score": 0.88,
})

_PATCH = "agents.query_enrichment_agent.get_model_client"


def _make_agent(response: str) -> QueryEnrichmentAgent:
    with unittest.mock.patch(_PATCH, return_value=_FakeLLMClient(response)):
        return QueryEnrichmentAgent()


def _make_failing_agent() -> QueryEnrichmentAgent:
    with unittest.mock.patch(_PATCH, return_value=_FailingLLMClient("")):
        return QueryEnrichmentAgent()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class QueryEnrichmentResultTests(unittest.TestCase):
    """Test case 3 — valid LLM response → populated QueryEnrichmentResult."""

    def test_returns_query_enrichment_result_instance(self) -> None:
        agent = _make_agent(_VALID_JSON_RESPONSE)
        result = agent.enrich("How do I charge a customer?", [], [])
        self.assertIsInstance(result, QueryEnrichmentResult)

    def test_detected_intent_populated(self) -> None:
        agent = _make_agent(_VALID_JSON_RESPONSE)
        result = agent.enrich("How do I charge a customer?", [], [])
        self.assertEqual(result.detected_intent, "api_discovery")

    def test_enriched_query_is_not_empty(self) -> None:
        agent = _make_agent(_VALID_JSON_RESPONSE)
        result = agent.enrich("How do I charge a customer?", [], [])
        self.assertTrue(result.enriched_query)

    def test_entities_populated(self) -> None:
        agent = _make_agent(_VALID_JSON_RESPONSE)
        result = agent.enrich("How do I charge a customer?", [], [])
        self.assertIn("payment", result.entities)

    def test_suggested_agents_populated(self) -> None:
        agent = _make_agent(_VALID_JSON_RESPONSE)
        result = agent.enrich("How do I charge a customer?", [], [])
        self.assertIn("semantic_retriever", result.suggested_agents)

    def test_confidence_score_positive(self) -> None:
        agent = _make_agent(_VALID_JSON_RESPONSE)
        result = agent.enrich("How do I charge a customer?", [], [])
        self.assertGreater(result.confidence_score, 0.0)

    def test_original_query_preserved(self) -> None:
        agent = _make_agent(_VALID_JSON_RESPONSE)
        result = agent.enrich("How do I charge a customer?", [], [])
        self.assertEqual(
            result.original_query, "How do I charge a customer?"
        )


class FallbackTests(unittest.TestCase):
    """LLM failure and malformed JSON must yield a passthrough result."""

    def test_llm_failure_returns_passthrough(self) -> None:
        agent = _make_failing_agent()
        result = agent.enrich("How do I charge a customer?", [], [])
        self.assertIsInstance(result, QueryEnrichmentResult)
        # Passthrough: enriched_query == original
        self.assertEqual(
            result.enriched_query, "How do I charge a customer?"
        )

    def test_llm_failure_detected_intent_is_unknown(self) -> None:
        agent = _make_failing_agent()
        result = agent.enrich("test query", [], [])
        self.assertEqual(result.detected_intent, "unknown")

    def test_non_dict_json_returns_passthrough(self) -> None:
        # LLM returns a JSON array instead of an object
        agent = _make_agent('["not", "an", "object"]')
        result = agent.enrich("test query", [], [])
        self.assertEqual(result.enriched_query, "test query")

    def test_plain_text_response_returns_passthrough(self) -> None:
        agent = _make_agent("Sure! I can help with that.")
        result = agent.enrich("test query", [], [])
        self.assertEqual(result.enriched_query, "test query")

    def test_passthrough_confidence_is_zero(self) -> None:
        agent = _make_failing_agent()
        result = agent.enrich("test query", [], [])
        self.assertEqual(result.confidence_score, 0.0)

    def test_passthrough_suggests_semantic_retriever(self) -> None:
        agent = _make_failing_agent()
        result = agent.enrich("test query", [], [])
        self.assertIn("semantic_retriever", result.suggested_agents)


class PassthroughClassMethodTests(unittest.TestCase):
    """QueryEnrichmentResult.passthrough() must return a safe default."""

    def test_passthrough_preserves_original_query(self) -> None:
        r = QueryEnrichmentResult.passthrough("find payment API")
        self.assertEqual(r.original_query, "find payment API")
        self.assertEqual(r.enriched_query, "find payment API")

    def test_passthrough_detected_intent_unknown(self) -> None:
        r = QueryEnrichmentResult.passthrough("anything")
        self.assertEqual(r.detected_intent, "unknown")

    def test_passthrough_entities_empty(self) -> None:
        r = QueryEnrichmentResult.passthrough("anything")
        self.assertEqual(r.entities, [])

    def test_passthrough_confidence_zero(self) -> None:
        r = QueryEnrichmentResult.passthrough("anything")
        self.assertEqual(r.confidence_score, 0.0)


class EnrichWithHistoryTests(unittest.TestCase):
    """Enrich must accept chat history and last_operations without error."""

    def test_chat_history_accepted(self) -> None:
        agent = _make_agent(_VALID_JSON_RESPONSE)
        history = [
            {"role": "user", "content": "What is a PaymentIntent?"},
            {"role": "assistant", "content": "It represents a payment."},
        ]
        result = agent.enrich(
            "How do I confirm it?", chat_history=history, last_operations=[]
        )
        self.assertIsInstance(result, QueryEnrichmentResult)

    def test_last_operations_accepted(self) -> None:
        agent = _make_agent(_VALID_JSON_RESPONSE)
        result = agent.enrich(
            "How do I confirm it?",
            chat_history=[],
            last_operations=["CreatePaymentIntent", "ConfirmPaymentIntent"],
        )
        self.assertIsInstance(result, QueryEnrichmentResult)


if __name__ == "__main__":
    unittest.main()
