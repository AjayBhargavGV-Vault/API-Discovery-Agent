from __future__ import annotations

import json
import os
from datetime import datetime
from datetime import timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from neo4j import GraphDatabase
from neo4j import Session


PROJECT_SRC = Path(__file__).resolve().parents[1]
BACKEND_DIR = PROJECT_SRC.parent
DATA_DIR = PROJECT_SRC / "data"
DEFAULT_ENV_PATH = BACKEND_DIR / ".env"
DEFAULT_DOCUMENTS_JSONL_PATH = DATA_DIR / "api_search_documents.jsonl"
DEFAULT_REPORT_PATH = DATA_DIR / "semantic_neo4j_validation_report.json"


class SemanticNeo4jLinkValidator:
    ENTITY_LABELS = {
        "product": "Product",
        "operation": "Operation",
        "schema": "Schema",
    }

    def __init__(
        self,
        documents_jsonl_path: Path = DEFAULT_DOCUMENTS_JSONL_PATH,
        report_path: Path = DEFAULT_REPORT_PATH,
        env_path: Path = DEFAULT_ENV_PATH,
    ) -> None:
        load_dotenv(env_path)
        self.documents_jsonl_path = documents_jsonl_path.resolve()
        self.report_path = report_path.resolve()
        self.uri = self.neo4j_uri_from_env()
        self.username = self.required_env("NEO4J_USERNAME")
        self.password = self.required_env("NEO4J_PASSWORD")
        self.database = os.getenv("NEO4J_DATABASE") or None
        self.driver = GraphDatabase.driver(self.uri, auth=(self.username, self.password))

    def close(self) -> None:
        self.driver.close()

    def required_env(self, key: str) -> str:
        value = os.getenv(key)
        if not value:
            raise ValueError(f"Missing required environment variable: {key}")
        return value

    def neo4j_uri_from_env(self) -> str:
        uri = self.required_env("NEO4J_URI")
        verify_certificates = os.getenv("NEO4J_VERIFY_CERTIFICATES", "").lower()
        if uri.startswith("neo4j+s://") and verify_certificates not in {"1", "true", "yes"}:
            return uri.replace("neo4j+s://", "neo4j+ssc://", 1)
        return uri

    def validate(self) -> dict[str, Any]:
        documents = self.read_jsonl(self.documents_jsonl_path)
        invalid_records: list[dict[str, Any]] = []

        with self.driver.session(database=self.database) as session:
            existing_node_ids = self.fetch_existing_node_ids_by_label(session, documents)
            for line_number, document in enumerate(documents, start=1):
                reasons = self.local_validation_reasons(document)
                if not reasons:
                    label = self.ENTITY_LABELS[document["entity_type"]]
                    if document["neo4j_node_id"] not in existing_node_ids.get(label, set()):
                        reasons.append("neo4j_node_id does not exist with expected label")

                if reasons:
                    invalid_records.append(
                        {
                            "line_number": line_number,
                            "id": document.get("id"),
                            "entity_type": document.get("entity_type"),
                            "entity_id": document.get("entity_id"),
                            "neo4j_node_id": document.get("neo4j_node_id"),
                            "reasons": reasons,
                        }
                    )

        total_documents = len(documents)
        invalid_count = len(invalid_records)
        report = {
            "validated_at": datetime.now(timezone.utc).isoformat(),
            "documents_path": str(self.documents_jsonl_path),
            "total_documents_checked": total_documents,
            "valid_count": total_documents - invalid_count,
            "invalid_count": invalid_count,
            "invalid_records": invalid_records,
        }
        self.write_report(report)
        return report

    def fetch_existing_node_ids_by_label(
        self,
        session: Session,
        documents: list[dict[str, Any]],
    ) -> dict[str, set[str]]:
        ids_by_label: dict[str, set[str]] = {
            label: set() for label in self.ENTITY_LABELS.values()
        }
        requested_ids_by_label: dict[str, set[str]] = {
            label: set() for label in self.ENTITY_LABELS.values()
        }

        for document in documents:
            entity_type = document.get("entity_type")
            node_id = document.get("neo4j_node_id")
            if entity_type in self.ENTITY_LABELS and node_id:
                requested_ids_by_label[self.ENTITY_LABELS[entity_type]].add(node_id)

        for label, requested_ids in requested_ids_by_label.items():
            for batch in self.batches(sorted(requested_ids)):
                query = f"MATCH (node:{label}) WHERE node.id IN $node_ids RETURN node.id AS id"
                for record in session.run(query, {"node_ids": batch}):
                    ids_by_label[label].add(record["id"])

        return ids_by_label

    def read_jsonl(self, path: Path) -> list[dict[str, Any]]:
        documents = []
        with path.open("r", encoding="utf-8") as source:
            for line_number, line in enumerate(source, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    document = json.loads(stripped)
                except json.JSONDecodeError as exc:
                    documents.append(
                        {
                            "id": None,
                            "entity_type": None,
                            "json_error": f"line {line_number}: {exc}",
                        }
                    )
                    continue
                documents.append(document)
        return documents

    def local_validation_reasons(self, document: dict[str, Any]) -> list[str]:
        reasons = []
        entity_type = document.get("entity_type")

        if not document.get("neo4j_node_id"):
            reasons.append("missing neo4j_node_id")

        if entity_type not in self.ENTITY_LABELS:
            reasons.append("entity_type is not product, operation, or schema")
            return reasons

        if entity_type == "operation":
            if not document.get("method"):
                reasons.append("operation document missing method")
            if not document.get("path"):
                reasons.append("operation document missing path")

        if entity_type == "schema":
            metadata = document.get("metadata", {})
            has_schema_name = isinstance(metadata, dict) and bool(metadata.get("schema_name"))
            if not has_schema_name and not document.get("title"):
                reasons.append("schema document missing schema name or title")

        if entity_type == "product" and not document.get("product_key"):
            reasons.append("product document missing product_key")

        return reasons

    def batches(self, values: list[str], batch_size: int = 500) -> list[list[str]]:
        return [values[index : index + batch_size] for index in range(0, len(values), batch_size)]

    def write_report(self, report: dict[str, Any]) -> None:
        self.report_path.parent.mkdir(parents=True, exist_ok=True)
        with self.report_path.open("w", encoding="utf-8") as target:
            json.dump(report, target, indent=2)
            target.write("\n")
