from __future__ import annotations

import sys
import unittest
from pathlib import Path
from typing import Any


PROJECT_SRC = Path(__file__).resolve().parents[1] / "src"
if str(PROJECT_SRC) not in sys.path:
    sys.path.append(str(PROJECT_SRC))

from data.ingestion.graph_expansion import GraphExpandedResult
from data.ingestion.graph_expansion import GraphRagRetriever
from data.ingestion.graph_expansion import Neo4jGraphExpander
from data.ingestion.graph_expansion import Neo4jGraphExpansionConfig


class FakeNeo4jResult:
    def __init__(self, row: dict[str, Any] | None) -> None:
        self.row = row

    def single(self) -> dict[str, Any] | None:
        return self.row


class FakeNeo4jSession:
    def __init__(self, rows_by_query_key: dict[str, dict[str, Any] | None]) -> None:
        self.rows_by_query_key = rows_by_query_key
        self.last_parameters: dict[str, Any] | None = None

    def __enter__(self) -> FakeNeo4jSession:
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        return None

    def run(self, query: str, parameters: dict[str, Any]) -> FakeNeo4jResult:
        self.last_parameters = parameters
        if "MATCH (operation:Operation" in query:
            return FakeNeo4jResult(self.rows_by_query_key.get("operation"))
        if "MATCH (schema:Schema" in query:
            return FakeNeo4jResult(self.rows_by_query_key.get("schema"))
        if "MATCH (product:Product" in query:
            return FakeNeo4jResult(self.rows_by_query_key.get("product"))
        if "RETURN labels(node)" in query:
            return FakeNeo4jResult(self.rows_by_query_key.get("labels"))
        return FakeNeo4jResult(None)


class FakeNeo4jDriver:
    def __init__(self, rows_by_query_key: dict[str, dict[str, Any] | None]) -> None:
        self.session_instance = FakeNeo4jSession(rows_by_query_key)

    def session(self, database: str | None = None) -> FakeNeo4jSession:
        return self.session_instance

    def close(self) -> None:
        return None


class FakeSemanticRetriever:
    def __init__(self) -> None:
        self.last_query: str | None = None
        self.last_filter: str | None = None

    def embed_query(self, query: str) -> list[float]:
        self.last_query = query
        return [0.1, 0.2, 0.3]

    def search(
        self,
        query_embedding: list[float],
        filter_expression: str | None = None,
    ) -> list[dict[str, Any]]:
        self.last_filter = filter_expression
        return [
            {
                "distance": 0.75,
                "entity": {
                    "id": "api_search_document:1",
                    "entity_type": "operation",
                    "entity_id": "operation-doc:1",
                    "title": "Confirm a PaymentIntent",
                    "body": "Confirm a Stripe PaymentIntent.",
                    "provider": "stripe",
                    "spec_id": "spec:1",
                    "spec_version": "2026-04-22.dahlia",
                    "business_domain": "payments",
                    "product_key": "payment_intents",
                    "method": "POST",
                    "path": "/v1/payment_intents/{intent}/confirm",
                    "neo4j_node_id": "operation:1",
                },
            }
        ]


class Neo4jGraphExpanderTests(unittest.TestCase):
    def test_expand_operation_returns_connected_context(self) -> None:
        expander = self.build_expander(
            {
                "operation": {
                    "node": {
                        "id": "operation:1",
                        "method": "POST",
                        "path": "/v1/payment_intents/{intent}/confirm",
                    },
                    "products": [{"id": "product:1", "name": "Payment Intents"}],
                    "resources": [{"id": "resource:1", "name": "payment_intents"}],
                    "request_schemas": [{"id": "schema:req", "name": "PaymentIntentConfirmParams"}],
                    "response_schemas": [
                        {"id": "schema:res", "name": "PaymentIntent", "status_code": "200"}
                    ],
                    "parameters": [{"id": "parameter:1", "name": "intent", "location": "path"}],
                    "auth_schemes": [{"id": "auth_scheme:1", "name": "BasicAuth"}],
                    "schema_fields": [
                        {
                            "schema_id": "schema:req",
                            "schema_name": "PaymentIntentConfirmParams",
                            "field": {"id": "field:1", "name": "payment_method"},
                            "referenced_schema": None,
                        }
                    ],
                    "related_lifecycle_operations": [
                        {"id": "operation:2", "summary": "Cancel a PaymentIntent"}
                    ],
                }
            }
        )

        context = expander.expand_operation("operation:1")

        self.assertEqual(context["product"]["name"], "Payment Intents")
        self.assertEqual(context["request_schemas"][0]["name"], "PaymentIntentConfirmParams")
        self.assertEqual(context["response_schemas"][0]["name"], "PaymentIntent")
        self.assertEqual(context["parameters"][0]["name"], "intent")
        self.assertEqual(context["auth_schemes"][0]["name"], "BasicAuth")
        self.assertEqual(context["schema_fields"][0]["field"]["name"], "payment_method")
        self.assertEqual(context["related_lifecycle_operations"][0]["summary"], "Cancel a PaymentIntent")

    def test_expand_schema_returns_fields_references_and_operations(self) -> None:
        expander = self.build_expander(
            {
                "schema": {
                    "node": {"id": "schema:1", "name": "PaymentIntent"},
                    "fields": [{"id": "field:1", "name": "currency"}],
                    "referenced_schemas": [{"id": "schema:2", "name": "Customer"}],
                    "operations_that_use_schema": [{"id": "operation:1", "summary": "Create"}],
                    "operations_that_return_schema": [{"id": "operation:2", "summary": "Retrieve"}],
                }
            }
        )

        context = expander.expand_schema("schema:1")

        self.assertEqual(context["node"]["name"], "PaymentIntent")
        self.assertEqual(context["fields"][0]["name"], "currency")
        self.assertEqual(context["referenced_schemas"][0]["name"], "Customer")
        self.assertEqual(context["operations_that_use_schema"][0]["summary"], "Create")
        self.assertEqual(context["operations_that_return_schema"][0]["summary"], "Retrieve")

    def test_expand_product_returns_operations_schemas_and_resources(self) -> None:
        expander = self.build_expander(
            {
                "product": {
                    "node": {"id": "product:1", "name": "Payment Intents"},
                    "business_domain": "payments",
                    "top_operations": [{"id": "operation:1", "summary": "Create a PaymentIntent"}],
                    "schemas": [{"id": "schema:1", "name": "PaymentIntent"}],
                    "related_resources": [{"id": "resource:1", "name": "payment_intents"}],
                }
            }
        )

        context = expander.expand_product("product:1")

        self.assertEqual(context["business_domain"], "payments")
        self.assertEqual(context["top_operations"][0]["summary"], "Create a PaymentIntent")
        self.assertEqual(context["schemas"][0]["name"], "PaymentIntent")
        self.assertEqual(context["related_resources"][0]["name"], "payment_intents")

    def test_expand_top_milvus_result_uses_neo4j_node_id(self) -> None:
        expander = self.build_expander(
            {
                "operation": {
                    "node": {
                        "id": "operation:1",
                        "operation_id": "PostPaymentIntentsIntentConfirm",
                        "method": "POST",
                        "path": "/v1/payment_intents/{intent}/confirm",
                        "summary": "Confirm a PaymentIntent",
                    },
                    "products": [{"id": "product:1", "name": "Payment Intents"}],
                    "resources": [],
                    "request_schemas": [
                        {"id": "schema:req", "name": "PaymentIntentConfirmParams"}
                    ],
                    "response_schemas": [{"id": "schema:res", "name": "PaymentIntent"}],
                    "parameters": [],
                    "auth_schemes": [],
                    "schema_fields": [
                        {
                            "schema_id": "schema:req",
                            "schema_name": "PaymentIntentConfirmParams",
                            "field": {"id": "field:1", "name": "payment_method"},
                            "referenced_schema": None,
                        }
                    ],
                    "related_lifecycle_operations": [],
                }
            }
        )
        milvus_results = [
            {
                "distance": 0.91,
                "entity": {
                    "entity_type": "operation",
                    "title": "Confirm a PaymentIntent",
                    "neo4j_node_id": "operation:1",
                },
            }
        ]

        result = expander.expand_top_milvus_result(
            milvus_results=milvus_results,
            query="How do I confirm a Stripe payment?",
        )

        self.assertIsInstance(result, GraphExpandedResult)
        self.assertEqual(result.score, 0.91)
        self.assertEqual(result.entity_type, "operation")
        self.assertEqual(result.neo4j_node_id, "operation:1")
        self.assertEqual(result.graph_context["product"]["name"], "Payment Intents")

    def test_entity_type_can_be_inferred_from_neo4j_labels(self) -> None:
        expander = self.build_expander({"labels": {"labels": ["Schema"]}})

        self.assertEqual(expander.entity_type_for_node("schema:1"), "schema")

    def test_graph_rag_retriever_searches_milvus_then_expands_top_result(self) -> None:
        semantic_retriever = FakeSemanticRetriever()
        expander = self.build_expander(
            {
                "operation": {
                    "node": {
                        "id": "operation:1",
                        "operation_id": "PostPaymentIntentsIntentConfirm",
                        "method": "POST",
                        "path": "/v1/payment_intents/{intent}/confirm",
                        "summary": "Confirm a PaymentIntent",
                    },
                    "products": [{"id": "product:1", "name": "Payment Intents"}],
                    "resources": [],
                    "request_schemas": [
                        {"id": "schema:req", "name": "PaymentIntentConfirmParams"}
                    ],
                    "response_schemas": [{"id": "schema:res", "name": "PaymentIntent"}],
                    "parameters": [],
                    "auth_schemes": [],
                    "schema_fields": [
                        {
                            "schema_id": "schema:req",
                            "schema_name": "PaymentIntentConfirmParams",
                            "field": {"id": "field:1", "name": "payment_method"},
                            "referenced_schema": None,
                        }
                    ],
                    "related_lifecycle_operations": [],
                }
            }
        )
        retriever = GraphRagRetriever(
            semantic_retriever=semantic_retriever,
            graph_expander=expander,
        )

        result = retriever.retrieve(
            query="How do I confirm a Stripe payment?",
            filter_expression='entity_type == "operation"',
        )

        self.assertEqual(semantic_retriever.last_query, "How do I confirm a Stripe payment?")
        self.assertEqual(semantic_retriever.last_filter, 'entity_type == "operation"')
        self.assertEqual(result["query"], "How do I confirm a Stripe payment?")
        self.assertEqual(result["semantic_hits"][0]["neo4j_node_id"], "operation:1")
        self.assertEqual(result["graph_context"][0]["neo4j_node_id"], "operation:1")
        self.assertEqual(
            result["recommended_operations"][0]["path"],
            "/v1/payment_intents/{intent}/confirm",
        )
        self.assertEqual(result["schemas"][0]["name"], "PaymentIntentConfirmParams")
        self.assertEqual(result["fields"][0]["name"], "payment_method")
        self.assertTrue(result["confidence_notes"])

    def build_expander(self, rows: dict[str, dict[str, Any] | None]) -> Neo4jGraphExpander:
        config = Neo4jGraphExpansionConfig(
            uri="neo4j://example",
            username="neo4j",
            password="password",
        )
        return Neo4jGraphExpander(config=config, driver=FakeNeo4jDriver(rows))


if __name__ == "__main__":
    unittest.main()
