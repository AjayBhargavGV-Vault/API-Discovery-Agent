from __future__ import annotations

import json
import logging
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any


PROJECT_SRC = Path(__file__).resolve().parents[1] / "src"
if str(PROJECT_SRC) not in sys.path:
    sys.path.append(str(PROJECT_SRC))

from data.ingestion.milvus_collection_setup import MilvusCollectionConfig  # noqa: E402
from data.ingestion.milvus_document_ingestion import MilvusDocumentIngestion  # noqa: E402


class FakeMilvusClient:
    def __init__(self) -> None:
        self.deleted_filters: list[str] = []
        self.inserted_rows: list[dict[str, Any]] = []
        self.existing_ids: set[str] = set()
        self.flushed = False
        self.loaded = False

    def delete(self, collection_name: str, filter: str) -> None:
        self.deleted_filters.append(filter)

    def insert(
        self, collection_name: str, data: list[dict[str, Any]]
    ) -> None:
        self.inserted_rows.extend(data)

    def query(
        self,
        collection_name: str,
        filter: str,
        output_fields: list[str],
    ) -> list[dict[str, Any]]:
        return [
            {"id": eid}
            for eid in self.existing_ids
            if json.dumps(eid) in filter
        ]

    def flush(self, collection_name: str) -> None:
        self.flushed = True

    def load_collection(self, collection_name: str) -> None:
        self.loaded = True

    def get_collection_stats(self, collection_name: str) -> dict[str, int]:
        return {"row_count": len(self.inserted_rows)}


class FakeCollectionSetup:
    VARCHAR_MAX_LENGTHS = {
        "id": 512,
        "entity_type": 64,
        "entity_id": 512,
        "neo4j_node_id": 512,
        "title": 4096,
        "body": 65535,
        "provider": 128,
        "spec_id": 512,
        "spec_version": 128,
        "business_domain": 128,
        "product_key": 256,
        "method": 32,
        "path": 4096,
        "embedding_model": 256,
        "indexed_at": 128,
    }

    def __init__(self) -> None:
        self.config = MilvusCollectionConfig(
            uri="http://localhost:19530",
            vector_dimension=3,
            collection_name="api_search_documents",
        )
        self.client = FakeMilvusClient()
        self.recreate_requested: bool | None = None

    def setup_collection(self, recreate: bool = False) -> None:
        self.recreate_requested = recreate
        return None


class MilvusDocumentIngestionTests(unittest.TestCase):
    def test_replace_collection_skips_duplicate_input_records(self) -> None:
        setup = FakeCollectionSetup()
        with tempfile.TemporaryDirectory() as temp_dir:
            documents_path = (
                Path(temp_dir) / "api_search_documents_embedded.jsonl"
            )
            self.write_jsonl(
                documents_path,
                [
                    self.document("doc:1"),
                    self.document("doc:1"),
                    self.document("doc:2"),
                ],
            )
            ingestion = MilvusDocumentIngestion(
                collection_setup=setup,  # type: ignore[arg-type]
                documents_path=documents_path,
                batch_size=100,
                replace_collection=True,
                logger=logging.getLogger("test_milvus_document_ingestion"),
            )
            ingestion.logger.disabled = True

            summary = ingestion.ingest().to_dict()

        self.assertEqual(summary["total_records"], 3)
        self.assertEqual(summary["inserted_records"], 2)
        self.assertEqual(summary["skipped_records"], 1)
        self.assertEqual(summary["duplicate_input_records"], 1)
        self.assertEqual(summary["already_existing_records"], 0)
        self.assertEqual(summary["collection_row_count"], 2)
        self.assertTrue(setup.recreate_requested)
        self.assertEqual(len(setup.client.inserted_rows), 2)
        expected_filter = 'id in ["doc:1","doc:2"]'
        self.assertEqual(setup.client.deleted_filters, [expected_filter])

    def test_preserve_collection_skips_already_ingested_records(self) -> None:
        setup = FakeCollectionSetup()
        setup.client.existing_ids = {"doc:1"}
        with tempfile.TemporaryDirectory() as temp_dir:
            documents_path = (
                Path(temp_dir) / "api_search_documents_embedded.jsonl"
            )
            self.write_jsonl(
                documents_path,
                [
                    self.document("doc:1"),
                    self.document("doc:2"),
                ],
            )
            ingestion = MilvusDocumentIngestion(
                collection_setup=setup,  # type: ignore[arg-type]
                documents_path=documents_path,
                batch_size=100,
                replace_collection=False,
                logger=logging.getLogger("test_milvus_document_ingestion"),
            )
            ingestion.logger.disabled = True

            summary = ingestion.ingest().to_dict()

        self.assertEqual(summary["total_records"], 2)
        self.assertEqual(summary["inserted_records"], 1)
        self.assertEqual(summary["already_existing_records"], 1)
        self.assertEqual(summary["duplicate_input_records"], 0)
        self.assertEqual(summary["collection_row_count"], 1)
        self.assertFalse(setup.recreate_requested)
        self.assertEqual(len(setup.client.inserted_rows), 1)
        self.assertEqual(setup.client.inserted_rows[0]["id"], "doc:2")

    def document(self, document_id: str) -> dict[str, Any]:
        return {
            "id": document_id,
            "entity_type": "operation",
            "entity_id": f"entity:{document_id}",
            "neo4j_node_id": f"operation:{document_id}",
            "title": "Title",
            "body": "Body",
            "provider": "stripe",
            "spec_id": "spec:1",
            "spec_version": "2026-04-22.dahlia",
            "business_domain": "payments",
            "product_key": "payment_intents",
            "method": "POST",
            "path": "/v1/payment_intents",
            "embedding_model": "test-model",
            "embedding_vector": [0.1, 0.2, 0.3],
            "indexed_at": "2026-05-15T00:00:00+00:00",
        }

    def write_jsonl(self, path: Path, records: list[dict[str, Any]]) -> None:
        with path.open("w", encoding="utf-8") as target:
            for record in records:
                target.write(json.dumps(record))
                target.write("\n")


if __name__ == "__main__":
    unittest.main()
