"""Tests for chat_engine helper selection logic."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_SRC = Path(__file__).resolve().parents[1] / "src"
if str(PROJECT_SRC) not in sys.path:
    sys.path.append(str(PROJECT_SRC))

from chat.chat_engine import _extract_op_details
from chat.schemas import AgentResult


class OperationSelectionTests(unittest.TestCase):

    def test_create_request_prefers_post_operation(self) -> None:
        result = AgentResult(
            agent_name="api_spec",
            success=True,
            data={
                "results": [
                    {
                        "operation_id": "list_checkout_sessions",
                        "endpoint": "/v1/checkout/sessions",
                        "method": "GET",
                        "summary": "List all Checkout Sessions",
                    },
                    {
                        "operation_id": "create_checkout_session",
                        "endpoint": "/v1/checkout/sessions",
                        "method": "POST",
                        "summary": "Create a Checkout Session",
                    },
                ],
            },
        )

        op = _extract_op_details(
            [result],
            "Show Python code to create a Checkout Session.",
        )

        self.assertIsNotNone(op)
        self.assertEqual(op.method, "POST")
        self.assertEqual(op.operation_id, "create_checkout_session")


if __name__ == "__main__":
    unittest.main()
