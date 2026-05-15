"""Tests for agents/orchestrator_agent.py and agents/validation_agent.py.

Test cases covered:
  4. Orchestrator selects semantic retriever for API discovery.
  5. Orchestrator selects graph retriever for lifecycle queries.
  8. Validation catches a hallucinated endpoint.

Mocking strategy: patch ``agents.orchestrator_agent.get_model_client``
(and ``agents.validation_agent.get_model_client``) so no live model is
needed.  The same fake client is returned for both primary and fallback
slots.  A ``_FailingLLMClient`` is used to exercise fallback paths.
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

from agents.query_enrichment_agent import QueryEnrichmentResult
from chat.schemas import AgentResult, AgentTask, ChatResponse

# ---------------------------------------------------------------------------
# Fake LLM clients
# ---------------------------------------------------------------------------

class _FakeConfig:
    model = "fake-orchestrator-model"


class _FakeLLMClient:
    config = _FakeConfig()

    def __init__(self, response: str = "") -> None:
        self._response = response

    def chat_completion(self, messages, **kwargs) -> str:
        if not self._response:
            raise RuntimeError("no LLM response in tests")
        return self._response


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------

_ORCH_PATCH = "agents.orchestrator_agent.get_model_client"


def _make_orchestrator(response: str = ""):
    """OrchestratorAgent with both slots set to the same fake client."""
    client = _FakeLLMClient(response)
    with unittest.mock.patch(_ORCH_PATCH, return_value=client):
        from agents.orchestrator_agent import OrchestratorAgent
        return OrchestratorAgent()


def _enrichment(
    intent: str = "api_discovery",
    query: str = "payment API",
    suggested: list | None = None,
) -> QueryEnrichmentResult:
    return QueryEnrichmentResult(
        original_query=query,
        enriched_query=query,
        detected_intent=intent,
        entities=["payment"],
        suggested_agents=suggested or ["semantic_retriever"],
        confidence_score=0.85,
    )


# ---------------------------------------------------------------------------
# Test case 4 — semantic retriever selected for API discovery
# ---------------------------------------------------------------------------

class DeterministicPlanAPIDiscoveryTests(unittest.TestCase):

    def test_vague_api_query_includes_semantic_retriever(self) -> None:
        agent = _make_orchestrator()
        tasks = agent._deterministic_plan(
            _enrichment("api_discovery", "How do I charge a customer?"),
            "How do I charge a customer?",
        )
        agent_names = [t.agent_name for t in tasks]
        self.assertIn("semantic_retriever", agent_names)

    def test_payment_query_includes_semantic_retriever(self) -> None:
        agent = _make_orchestrator()
        tasks = agent._deterministic_plan(
            _enrichment("api_discovery", "Find payment APIs"),
            "Find payment APIs",
        )
        self.assertIn(
            "semantic_retriever", [t.agent_name for t in tasks]
        )

    def test_default_route_when_intent_unknown(self) -> None:
        agent = _make_orchestrator()
        tasks = agent._deterministic_plan(
            _enrichment("unknown", "something vague"),
            "something vague",
        )
        # Default route always includes semantic_retriever
        self.assertIn(
            "semantic_retriever", [t.agent_name for t in tasks]
        )

    def test_tasks_have_required_fields(self) -> None:
        agent = _make_orchestrator()
        tasks = agent._deterministic_plan(
            _enrichment("api_discovery", "find payment APIs"),
            "find payment APIs",
        )
        for task in tasks:
            self.assertIsInstance(task, AgentTask)
            self.assertTrue(task.task_id)
            self.assertTrue(task.agent_name)
            self.assertTrue(task.task_type)


# ---------------------------------------------------------------------------
# Test case 5 — graph retriever selected for lifecycle queries
# ---------------------------------------------------------------------------

class DeterministicPlanLifecycleTests(unittest.TestCase):

    def test_lifecycle_keyword_selects_graph_retriever(self) -> None:
        agent = _make_orchestrator()
        query = "What is the lifecycle of a payment?"
        tasks = agent._deterministic_plan(
            _enrichment("lifecycle_query", query), query
        )
        self.assertIn(
            "graph_retriever", [t.agent_name for t in tasks]
        )

    def test_relationship_keyword_selects_graph_retriever(self) -> None:
        agent = _make_orchestrator()
        query = "Show me the relationship between charges and intents"
        tasks = agent._deterministic_plan(
            _enrichment("lifecycle_query", query), query
        )
        self.assertIn(
            "graph_retriever", [t.agent_name for t in tasks]
        )

    def test_code_request_selects_api_spec(self) -> None:
        agent = _make_orchestrator()
        query = "Give me a curl example for creating a payment"
        tasks = agent._deterministic_plan(
            _enrichment("code_generation", query), query
        )
        self.assertIn("api_spec", [t.agent_name for t in tasks])

    def test_non_empty_plan_always_returned(self) -> None:
        agent = _make_orchestrator()
        tasks = agent._deterministic_plan(
            _enrichment("unknown", "???"), "???"
        )
        self.assertGreater(len(tasks), 0)


# ---------------------------------------------------------------------------
# LLM planning path
# ---------------------------------------------------------------------------

_PLAN_JSON = json.dumps([
    {
        "task_id": "t1",
        "agent_name": "semantic_retriever",
        "task_type": "semantic_search",
        "input": {"query": "payment API"},
        "reason": "Retrieve relevant operations",
        "model_name": None,
    }
])


class LLMPlanningTests(unittest.TestCase):

    def test_valid_llm_plan_returns_agent_tasks(self) -> None:
        agent = _make_orchestrator(_PLAN_JSON)
        tasks = agent.create_agent_plan(
            _enrichment(), "payment API"
        )
        self.assertGreater(len(tasks), 0)
        self.assertIsInstance(tasks[0], AgentTask)

    def test_llm_plan_task_has_correct_agent_name(self) -> None:
        agent = _make_orchestrator(_PLAN_JSON)
        tasks = agent.create_agent_plan(
            _enrichment(), "payment API"
        )
        self.assertEqual(tasks[0].agent_name, "semantic_retriever")

    def test_llm_failure_falls_back_to_deterministic(self) -> None:
        # Both clients raise — must still return a non-empty plan
        agent = _make_orchestrator("")  # empty → raises
        tasks = agent.create_agent_plan(
            _enrichment(), "payment API"
        )
        self.assertGreater(len(tasks), 0)

    def test_empty_llm_plan_falls_back(self) -> None:
        # LLM returns valid JSON but with an empty list
        agent = _make_orchestrator("[]")
        tasks = agent.create_agent_plan(
            _enrichment(), "payment API"
        )
        self.assertGreater(len(tasks), 0)


# ---------------------------------------------------------------------------
# Synthesis
# ---------------------------------------------------------------------------

class SynthesisTests(unittest.TestCase):

    def test_synthesis_returns_chat_response(self) -> None:
        answer = "Use POST /v1/payment_intents to create a payment."
        agent = _make_orchestrator(answer)
        results = [
            AgentResult(
                agent_name="semantic_retriever",
                success=True,
                data={"matches": []},
            )
        ]
        response = agent.synthesize_response(
            user_message="How do I create a payment?",
            enrichment_result=_enrichment(),
            agent_results=results,
        )
        self.assertIsInstance(response, ChatResponse)
        self.assertEqual(response.answer, answer)

    def test_synthesis_llm_failure_returns_fallback_chat_response(
        self,
    ) -> None:
        agent = _make_orchestrator("")  # always raises
        results = [
            AgentResult(
                agent_name="api_spec",
                success=True,
                data={
                    "operation": {
                        "method": "POST",
                        "endpoint": "/v1/charges",
                    }
                },
                sources=["spec:POST:/v1/charges"],
            )
        ]
        response = agent.synthesize_response(
            user_message="charge API",
            enrichment_result=_enrichment(),
            agent_results=results,
        )
        self.assertIsInstance(response, ChatResponse)
        self.assertTrue(response.answer)

    def test_synthesis_collects_sources(self) -> None:
        agent = _make_orchestrator("The answer is here.")
        results = [
            AgentResult(
                agent_name="api_spec",
                success=True,
                data={},
                sources=["spec:POST:/v1/charges"],
            )
        ]
        response = agent.synthesize_response(
            user_message="test",
            enrichment_result=_enrichment(),
            agent_results=results,
        )
        self.assertIn("spec:POST:/v1/charges", response.sources)

    def test_product_catalog_answer_filters_to_dominant_product(self) -> None:
        agent = _make_orchestrator("unused")
        results = [
            AgentResult(
                agent_name="api_spec",
                success=True,
                data={
                    "results": [
                        {
                            "method": "GET",
                            "endpoint": "/v1/billing/alerts",
                            "summary": "List billing alerts",
                            "product_key": "billing",
                        },
                        {
                            "method": "POST",
                            "endpoint": "/v1/billing/alerts",
                            "summary": "Create a billing alert",
                            "product_key": "billing",
                        },
                        {
                            "method": "GET",
                            "endpoint": "/v1/billing/meters",
                            "summary": "List billing meters",
                            "product_key": "billing",
                        },
                        {
                            "method": "GET",
                            "endpoint": "/v1/customers",
                            "summary": "List customers",
                            "product_key": "customers",
                        },
                    ]
                },
            )
        ]
        response = agent.synthesize_response(
            user_message="What APIs are available in the Billing product?",
            enrichment_result=_enrichment(),
            agent_results=results,
        )
        self.assertIn("/v1/billing/alerts", response.answer)
        self.assertNotIn("/v1/customers", response.answer)


# ---------------------------------------------------------------------------
# Test case 8 — validation catches hallucinated endpoint
# ---------------------------------------------------------------------------

_VAL_PATCH = "agents.validation_agent.get_model_client"


class _FakeValConfig:
    model = "fake-validation-model"


class _FakeValClient:
    config = _FakeValConfig()

    def __init__(self, response: str) -> None:
        self._response = response

    def chat_completion(self, messages, **kwargs) -> str:
        return self._response


def _make_validation_agent(llm_response: str):
    with unittest.mock.patch(
        _VAL_PATCH, return_value=_FakeValClient(llm_response)
    ):
        from agents.validation_agent import ValidationAgent
        return ValidationAgent()


class ValidationHallucinationTests(unittest.TestCase):
    """Test case 8 — validation agent rejects hallucinated endpoints."""

    _UNGROUNDED_JSON = json.dumps({
        "is_grounded": False,
        "unsupported_claims": [
            "POST /v1/fake-endpoint does not exist in sources"
        ],
        "confidence_score": 0.05,
        "final_answer": (
            "The retrieved API catalog does not contain enough "
            "information to answer this question."
        ),
    })

    _GROUNDED_JSON = json.dumps({
        "is_grounded": True,
        "unsupported_claims": [],
        "confidence_score": 0.95,
        "final_answer": (
            "Use POST /v1/payment_intents to create a payment."
        ),
    })

    _SOURCES = [
        {
            "method": "POST",
            "endpoint": "/v1/payment_intents",
            "summary": "Create a PaymentIntent",
            "operation_id": "CreatePaymentIntent",
        }
    ]

    def test_hallucinated_endpoint_is_ungrounded(self) -> None:
        agent = _make_validation_agent(self._UNGROUNDED_JSON)
        result = agent.validate_response(
            answer="Use POST /v1/fake-endpoint to process payments.",
            sources=self._SOURCES,
            agent_results=[],
        )
        self.assertFalse(result.is_grounded)

    def test_hallucinated_endpoint_appears_in_unsupported_claims(
        self,
    ) -> None:
        agent = _make_validation_agent(self._UNGROUNDED_JSON)
        result = agent.validate_response(
            answer="Use POST /v1/fake-endpoint to process payments.",
            sources=self._SOURCES,
            agent_results=[],
        )
        self.assertGreater(len(result.unsupported_claims), 0)
        self.assertTrue(
            any(
                "/v1/fake-endpoint" in c
                for c in result.unsupported_claims
            )
        )

    def test_hallucinated_endpoint_not_in_final_answer(self) -> None:
        agent = _make_validation_agent(self._UNGROUNDED_JSON)
        result = agent.validate_response(
            answer="Use POST /v1/fake-endpoint to process payments.",
            sources=self._SOURCES,
            agent_results=[],
        )
        self.assertNotIn("/v1/fake-endpoint", result.final_answer)

    def test_confidence_low_when_ungrounded(self) -> None:
        agent = _make_validation_agent(self._UNGROUNDED_JSON)
        result = agent.validate_response(
            answer="POST /v1/fake-endpoint",
            sources=self._SOURCES,
            agent_results=[],
        )
        self.assertLess(result.confidence_score, 1.0)

    def test_grounded_answer_passes_validation(self) -> None:
        agent = _make_validation_agent(self._GROUNDED_JSON)
        result = agent.validate_response(
            answer="Use POST /v1/payment_intents to create a payment.",
            sources=self._SOURCES,
            agent_results=[],
        )
        self.assertTrue(result.is_grounded)
        self.assertEqual(result.unsupported_claims, [])

    def test_empty_answer_is_ungrounded_without_llm(self) -> None:
        # Empty answer is caught before the LLM call
        agent = _make_validation_agent(self._GROUNDED_JSON)
        result = agent.validate_response(
            answer="", sources=self._SOURCES, agent_results=[]
        )
        self.assertFalse(result.is_grounded)
        self.assertEqual(result.confidence_score, 0.0)


if __name__ == "__main__":
    unittest.main()
