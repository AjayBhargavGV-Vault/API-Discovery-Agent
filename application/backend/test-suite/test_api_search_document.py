from __future__ import annotations

import hashlib
import sys
import unittest
from datetime import datetime
from datetime import timezone
from pathlib import Path


PROJECT_SRC = Path(__file__).resolve().parents[1] / "src"
if str(PROJECT_SRC) not in sys.path:
    sys.path.append(str(PROJECT_SRC))

from models.api_search_document import ApiSearchDocument


class ApiSearchDocumentTests(unittest.TestCase):
    def test_create_generates_stable_document_id(self) -> None:
        document = self.build_document()
        expected_digest = hashlib.sha256(
            "operation:api_operation:post:abc".encode("utf-8")
        ).hexdigest()[:20]

        self.assertEqual(document.id, f"api_search_document:{expected_digest}")

    def test_neo4j_operation_id_is_stable(self) -> None:
        operation_key = "PostPaymentIntentsIntentConfirm"
        expected_digest = hashlib.sha256(f"stripe:{operation_key}".encode("utf-8")).hexdigest()[
            :20
        ]

        self.assertEqual(
            ApiSearchDocument.neo4j_operation_id("stripe", operation_key),
            f"operation:{expected_digest}",
        )

    def test_body_cannot_be_empty(self) -> None:
        with self.assertRaises(ValueError):
            self.build_document(body=" ")

    def test_entity_type_must_be_supported(self) -> None:
        with self.assertRaises(ValueError):
            ApiSearchDocument(
                id="doc:1",
                spec_id="spec:1",
                entity_type="unknown",
                entity_id="entity:1",
                title="Title",
                body="Body",
                keywords=[],
                provider="stripe",
                spec_version="2026-04-22.dahlia",
                business_domain=None,
                product_key=None,
                method=None,
                path=None,
                neo4j_node_id="node:1",
                embedding_model=None,
                embedding_vector=None,
                indexed_at=datetime.now(timezone.utc),
            )

    def test_to_dict_serializes_indexed_at(self) -> None:
        indexed_at = datetime(2026, 5, 15, tzinfo=timezone.utc)
        document = self.build_document(indexed_at=indexed_at)

        self.assertEqual(document.to_dict()["indexed_at"], "2026-05-15T00:00:00+00:00")

    def build_document(
        self,
        body: str = "Confirm a Stripe payment.",
        indexed_at: datetime | None = None,
    ) -> ApiSearchDocument:
        return ApiSearchDocument.create(
            spec_id="stripe:2026-04-22.dahlia:hash",
            entity_type="operation",
            entity_id="api_operation:post:abc",
            title="Confirm PaymentIntent",
            body=body,
            keywords=["payment_intents", "confirm"],
            provider="stripe",
            spec_version="2026-04-22.dahlia",
            business_domain="payments",
            product_key="payment_intents",
            method="POST",
            path="/v1/payment_intents/{intent}/confirm",
            neo4j_node_id=ApiSearchDocument.neo4j_operation_id(
                "stripe",
                "PostPaymentIntentsIntentConfirm",
            ),
            indexed_at=indexed_at,
        )


if __name__ == "__main__":
    unittest.main()
