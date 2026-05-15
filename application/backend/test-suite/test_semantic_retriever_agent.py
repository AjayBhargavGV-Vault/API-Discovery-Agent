"""Tests for agents/semantic_retriever_agent.py.

Mocking strategy:
  - Patch ``agents.semantic_retriever_agent.init_retriever`` to suppress
    the Milvus registration call in __init__.
  - Patch ``agents.semantic_retriever_agent.semantic_api_search`` to
    control what Milvus returns without a live collection.

Both patches must be active at the point of agent construction AND at
the point of agent.run(), because the names are bound via
``from agent_tools.milvus_tools import ...`` at import time.
"""
from __future__ import annotations

import sys
import unittest
import unittest.mock
from pathlib import Path
from typing import Any

PROJECT_SRC = Path(__file__).resolve().parents[1] / "src"
if str(PROJECT_SRC) not in sys.path:
    sys.path.append(str(PROJECT_SRC))

from chat.schemas import AgentResult, RetrievedAPIMatch

# Patch targets (names as imported inside the agent module)
_PATCH_INIT = "agents.semantic_retriever_agent.init_retriever"
_PATCH_SEARCH = "agents.semantic_retriever_agent.semantic_api_search"


# ---------------------------------------------------------------------------
# Fake retriever (stand-in for MilvusRetrievalTest)
# ---------------------------------------------------------------------------

class _FakeRetriever:
    """Minimal object accepted by SemanticRetrieverAgent.__init__."""


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def _match(**kwargs) -> RetrievedAPIMatch:
    defaults = dict(
        operation_id="CreatePaymentIntent",
        endpoint="/v1/payment_intents",
        method="POST",
        score=0.91,
        source="operation:spec:stripe-v1",
    )
    defaults.update(kwargs)
    return RetrievedAPIMatch(**defaults)


def _make_agent_with_results(
    results: list[RetrievedAPIMatch],
):
    """Build SemanticRetrieverAgent, running __init__ and run() inside patches."""
    from agents.semantic_retriever_agent import SemanticRetrieverAgent

    with (
        unittest.mock.patch(_PATCH_INIT),
        unittest.mock.patch(_PATCH_SEARCH, return_value=results),
    ):
        agent = SemanticRetrieverAgent(_FakeRetriever())
        result = agent.run("payment API", top_k=5)
    return result


def _make_agent_raising(exc: Exception):
    from agents.semantic_retriever_agent import SemanticRetrieverAgent

    with (
        unittest.mock.patch(_PATCH_INIT),
        unittest.mock.patch(_PATCH_SEARCH, side_effect=exc),
    ):
        agent = SemanticRetrieverAgent(_FakeRetriever())
        result = agent.run("payment API")
    return result


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class SemanticRetrieverSuccessTests(unittest.TestCase):

    def test_run_returns_agent_result(self) -> None:
        result = _make_agent_with_results([_match()])
        self.assertIsInstance(result, AgentResult)

    def test_run_success_true(self) -> None:
        result = _make_agent_with_results([_match()])
        self.assertTrue(result.success)

    def test_agent_name_is_semantic_retriever(self) -> None:
        result = _make_agent_with_results([_match()])
        self.assertEqual(result.agent_name, "semantic_retriever")

    def test_match_count_reflects_results(self) -> None:
        result = _make_agent_with_results([_match(), _match()])
        self.assertEqual(result.data["match_count"], 2)

    def test_matches_list_serialised_to_dicts(self) -> None:
        result = _make_agent_with_results([_match()])
        matches = result.data["matches"]
        self.assertEqual(len(matches), 1)
        self.assertIsInstance(matches[0], dict)

    def test_match_contains_expected_fields(self) -> None:
        result = _make_agent_with_results([_match()])
        m = result.data["matches"][0]
        self.assertEqual(m["endpoint"], "/v1/payment_intents")
        self.assertEqual(m["method"], "POST")

    def test_sources_populated_from_match_source(self) -> None:
        result = _make_agent_with_results(
            [_match(source="operation:spec:stripe-v1")]
        )
        self.assertIn("operation:spec:stripe-v1", result.sources)

    def test_sources_deduplicated(self) -> None:
        result = _make_agent_with_results([
            _match(source="operation:spec:stripe-v1"),
            _match(
                operation_id="ListPayments",
                source="operation:spec:stripe-v1",
            ),
        ])
        self.assertEqual(result.sources.count("operation:spec:stripe-v1"), 1)

    def test_none_source_excluded(self) -> None:
        result = _make_agent_with_results([_match(source=None)])
        self.assertEqual(result.sources, [])


class SemanticRetrieverEmptyTests(unittest.TestCase):

    def test_empty_results_success_true(self) -> None:
        result = _make_agent_with_results([])
        self.assertTrue(result.success)

    def test_empty_results_match_count_zero(self) -> None:
        result = _make_agent_with_results([])
        self.assertEqual(result.data["match_count"], 0)

    def test_empty_results_sources_empty(self) -> None:
        result = _make_agent_with_results([])
        self.assertEqual(result.sources, [])


class SemanticRetrieverErrorTests(unittest.TestCase):

    def test_milvus_error_returns_failure_result(self) -> None:
        result = _make_agent_raising(
            ConnectionError("Milvus collection not found")
        )
        self.assertFalse(result.success)

    def test_error_message_captured(self) -> None:
        result = _make_agent_raising(
            RuntimeError("embedding model not loaded")
        )
        self.assertIsNotNone(result.error)
        self.assertIn("embedding model", result.error)

    def test_failure_match_count_zero(self) -> None:
        result = _make_agent_raising(IOError("disk error"))
        self.assertEqual(result.data["match_count"], 0)

    def test_failure_matches_empty_list(self) -> None:
        result = _make_agent_raising(IOError("disk error"))
        self.assertEqual(result.data["matches"], [])

    def test_failure_sources_empty(self) -> None:
        result = _make_agent_raising(IOError("disk error"))
        self.assertEqual(result.sources, [])


class SemanticRetrieverInitTests(unittest.TestCase):
    """init_retriever must be called exactly once at construction."""

    def test_init_retriever_called_once(self) -> None:
        from agents.semantic_retriever_agent import SemanticRetrieverAgent

        fake_retriever = _FakeRetriever()
        mock_init = unittest.mock.Mock()
        with (
            unittest.mock.patch(_PATCH_INIT, mock_init),
            unittest.mock.patch(_PATCH_SEARCH, return_value=[]),
        ):
            SemanticRetrieverAgent(fake_retriever)

        mock_init.assert_called_once_with(fake_retriever)


if __name__ == "__main__":
    unittest.main()
