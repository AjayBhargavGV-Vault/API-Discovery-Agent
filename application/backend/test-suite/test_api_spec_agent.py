"""Tests for agents/api_spec_agent.py.

Test case 6 — API spec agent can read openapi.spec3.yaml.

Two test classes:
  ApiSpecYamlTests   — direct file-system checks: the YAML is present,
                       parseable, and has the expected OpenAPI structure.
  ApiSpecAgentTests  — agent behaviour with mocked tool functions, so
                       no external data files are required for unit tests.

Mocking strategy: patch the tool functions as imported inside the agent
(``agents.api_spec_agent.get_operation_details`` etc.) rather than the
originals in ``agent_tools.api_spec_tools``.
"""
from __future__ import annotations

import sys
import unittest
import unittest.mock
from pathlib import Path

PROJECT_SRC = Path(__file__).resolve().parents[1] / "src"
if str(PROJECT_SRC) not in sys.path:
    sys.path.append(str(PROJECT_SRC))

from agents.api_spec_agent import ApiSpecAgent
from chat.schemas import AgentResult, APIOperationDetails

# Patch targets
_PATCH_GET_OP = "agents.api_spec_agent.get_operation_details"
_PATCH_SEARCH = "agents.api_spec_agent.search_operations_by_name_or_path"
_PATCH_SCHEMA = "agents.api_spec_agent.get_schema_details"

# Shared fixtures
_OP = APIOperationDetails(
    operation_id="CreatePaymentIntent",
    endpoint="/v1/payment_intents",
    method="POST",
    summary="Create a PaymentIntent",
)

_SCHEMA = {
    "schema_name": "PaymentIntent",
    "properties": {"id": {"type": "string"}, "amount": {"type": "integer"}},
    "fields": [],
}


# ---------------------------------------------------------------------------
# Test case 6 — OpenAPI YAML is present and parseable
# ---------------------------------------------------------------------------

class ApiSpecYamlTests(unittest.TestCase):

    _SPEC = (
        Path(__file__).resolve().parents[1]
        / "src" / "data" / "raw_specs" / "openapi.spec3.yaml"
    )

    def test_spec_file_exists(self) -> None:
        self.assertTrue(
            self._SPEC.exists(),
            f"OpenAPI spec not found: {self._SPEC}",
        )

    def test_spec_is_valid_yaml(self) -> None:
        import yaml
        with self._SPEC.open(encoding="utf-8") as fh:
            doc = yaml.safe_load(fh)
        self.assertIsInstance(doc, dict)

    def test_spec_has_openapi_key(self) -> None:
        import yaml
        with self._SPEC.open(encoding="utf-8") as fh:
            doc = yaml.safe_load(fh)
        self.assertIn("openapi", doc)

    def test_spec_has_paths(self) -> None:
        import yaml
        with self._SPEC.open(encoding="utf-8") as fh:
            doc = yaml.safe_load(fh)
        self.assertIn("paths", doc)
        self.assertGreater(len(doc["paths"]), 0)

    def test_spec_has_info(self) -> None:
        import yaml
        with self._SPEC.open(encoding="utf-8") as fh:
            doc = yaml.safe_load(fh)
        self.assertIn("info", doc)


# ---------------------------------------------------------------------------
# Agent behaviour tests (mocked tools)
# ---------------------------------------------------------------------------

class ApiSpecAgentOperationTests(unittest.TestCase):

    def test_run_get_operation_details_success(self) -> None:
        with unittest.mock.patch(_PATCH_GET_OP, return_value=_OP):
            result = ApiSpecAgent().run_get_operation_details(
                "CreatePaymentIntent"
            )
        self.assertTrue(result.success)
        self.assertIsInstance(result, AgentResult)

    def test_run_get_operation_details_data_contains_operation(
        self,
    ) -> None:
        with unittest.mock.patch(_PATCH_GET_OP, return_value=_OP):
            result = ApiSpecAgent().run_get_operation_details(
                "CreatePaymentIntent"
            )
        op = result.data.get("operation")
        self.assertIsNotNone(op)
        self.assertEqual(op["operation_id"], "CreatePaymentIntent")

    def test_run_get_operation_details_source_label(self) -> None:
        with unittest.mock.patch(_PATCH_GET_OP, return_value=_OP):
            result = ApiSpecAgent().run_get_operation_details(
                "CreatePaymentIntent"
            )
        self.assertIn("spec:POST:/v1/payment_intents", result.sources)

    def test_run_get_operation_details_not_found(self) -> None:
        with unittest.mock.patch(_PATCH_GET_OP, return_value=None):
            result = ApiSpecAgent().run_get_operation_details("Ghost")
        self.assertFalse(result.success)
        self.assertIsNotNone(result.error)

    def test_run_get_operation_details_exception_wrapped(self) -> None:
        with unittest.mock.patch(
            _PATCH_GET_OP, side_effect=IOError("disk error")
        ):
            result = ApiSpecAgent().run_get_operation_details("any")
        self.assertFalse(result.success)
        self.assertIn("disk error", result.error)


class ApiSpecAgentSearchTests(unittest.TestCase):

    def test_run_search_operations_success(self) -> None:
        with unittest.mock.patch(_PATCH_SEARCH, return_value=[_OP]):
            result = ApiSpecAgent().run_search_operations("payment")
        self.assertTrue(result.success)
        self.assertEqual(result.data["result_count"], 1)

    def test_run_search_operations_results_serialised(self) -> None:
        with unittest.mock.patch(_PATCH_SEARCH, return_value=[_OP]):
            result = ApiSpecAgent().run_search_operations("payment")
        first = result.data["results"][0]
        self.assertIsInstance(first, dict)
        self.assertEqual(first["endpoint"], "/v1/payment_intents")

    def test_run_search_operations_empty(self) -> None:
        with unittest.mock.patch(_PATCH_SEARCH, return_value=[]):
            result = ApiSpecAgent().run_search_operations("xyz")
        self.assertTrue(result.success)
        self.assertEqual(result.data["result_count"], 0)

    def test_run_search_operations_exception_wrapped(self) -> None:
        with unittest.mock.patch(
            _PATCH_SEARCH, side_effect=RuntimeError("index error")
        ):
            result = ApiSpecAgent().run_search_operations("payment")
        self.assertFalse(result.success)
        self.assertIn("index error", result.error)


class ApiSpecAgentSchemaTests(unittest.TestCase):

    def test_run_get_schema_details_success(self) -> None:
        with unittest.mock.patch(_PATCH_SCHEMA, return_value=_SCHEMA):
            result = ApiSpecAgent().run_get_schema_details(
                "PaymentIntent"
            )
        self.assertTrue(result.success)

    def test_run_get_schema_details_source_label(self) -> None:
        with unittest.mock.patch(_PATCH_SCHEMA, return_value=_SCHEMA):
            result = ApiSpecAgent().run_get_schema_details(
                "PaymentIntent"
            )
        self.assertIn("spec:schema:PaymentIntent", result.sources)

    def test_run_get_schema_details_data_contains_schema(self) -> None:
        with unittest.mock.patch(_PATCH_SCHEMA, return_value=_SCHEMA):
            result = ApiSpecAgent().run_get_schema_details(
                "PaymentIntent"
            )
        self.assertIn("schema", result.data)
        self.assertEqual(
            result.data["schema"]["schema_name"], "PaymentIntent"
        )

    def test_run_get_schema_details_not_found(self) -> None:
        with unittest.mock.patch(_PATCH_SCHEMA, return_value=None):
            result = ApiSpecAgent().run_get_schema_details("Ghost")
        self.assertFalse(result.success)

    def test_run_get_schema_details_empty_dict_treated_as_not_found(
        self,
    ) -> None:
        with unittest.mock.patch(_PATCH_SCHEMA, return_value={}):
            result = ApiSpecAgent().run_get_schema_details("Empty")
        self.assertFalse(result.success)

    def test_run_get_schema_details_exception_wrapped(self) -> None:
        with unittest.mock.patch(
            _PATCH_SCHEMA, side_effect=KeyError("bad key")
        ):
            result = ApiSpecAgent().run_get_schema_details("any")
        self.assertFalse(result.success)


if __name__ == "__main__":
    unittest.main()
