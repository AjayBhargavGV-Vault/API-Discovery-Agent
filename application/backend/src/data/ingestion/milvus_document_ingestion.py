from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pymilvus.exceptions import MilvusException

from data.ingestion.milvus_collection_setup import MilvusCollectionSetup


DATA_DIR = Path(__file__).resolve().parents[1]
DEFAULT_EMBEDDED_DOCUMENTS_PATH = (
    DATA_DIR / "api_search_documents_embedded.jsonl"
)


@dataclass
class MilvusIngestionSummary:
    total_records: int = 0
    inserted_records: int = 0
    skipped_records: int = 0
    failed_records: int = 0
    duplicate_input_records: int = 0
    already_existing_records: int = 0
    collection_row_count: int | None = None

    def to_dict(self) -> dict[str, int | None]:
        return {
            "total_records": self.total_records,
            "inserted_records": self.inserted_records,
            "skipped_records": self.skipped_records,
            "failed_records": self.failed_records,
            "duplicate_input_records": self.duplicate_input_records,
            "already_existing_records": self.already_existing_records,
            "collection_row_count": self.collection_row_count,
        }


class MilvusDocumentIngestion:
    FIELD_NAMES = (
        "id",
        "entity_type",
        "entity_id",
        "neo4j_node_id",
        "title",
        "body",
        "provider",
        "spec_id",
        "spec_version",
        "business_domain",
        "product_key",
        "method",
        "path",
        "embedding_model",
        "indexed_at",
    )

    def __init__(
        self,
        collection_setup: MilvusCollectionSetup,
        documents_path: Path = DEFAULT_EMBEDDED_DOCUMENTS_PATH,
        batch_size: int = 250,
        replace_collection: bool = False,
        logger: logging.Logger | None = None,
    ) -> None:
        if batch_size < 100 or batch_size > 500:
            raise ValueError("batch_size must be between 100 and 500.")
        self.collection_setup = collection_setup
        self.client = collection_setup.client
        self.config = collection_setup.config
        self.documents_path = documents_path.resolve()
        self.batch_size = batch_size
        self.replace_collection = replace_collection
        self.logger = logger or logging.getLogger(self.__class__.__name__)

    def ingest(self) -> MilvusIngestionSummary:
        self.collection_setup.setup_collection(
            recreate=self.replace_collection
        )
        summary = MilvusIngestionSummary()
        batch: list[dict[str, Any]] = []
        seen_ids: set[str] = set()

        for line_number, record in self.read_jsonl_records():
            summary.total_records += 1
            validation_errors = self.validate_record(record)
            if validation_errors:
                summary.skipped_records += 1
                self.logger.warning(
                    "Skipping line %s (%s): %s",
                    line_number,
                    record.get("id"),
                    "; ".join(validation_errors),
                )
                continue

            record_id = str(record["id"])
            if record_id in seen_ids:
                summary.skipped_records += 1
                summary.duplicate_input_records += 1
                self.logger.warning(
                    "Skipping duplicate input record id on line %s: %s",
                    line_number,
                    record_id,
                )
                continue
            seen_ids.add(record_id)

            batch.append(self.to_milvus_row(record))
            if len(batch) >= self.batch_size:
                self.write_batch(batch=batch, summary=summary)
                batch = []

        if batch:
            self.write_batch(batch=batch, summary=summary)

        self.flush_collection()
        self.load_collection()
        summary.collection_row_count = self.collection_row_count()
        self.logger.info("Milvus ingestion summary: %s", summary.to_dict())
        return summary

    def read_jsonl_records(self):
        with self.documents_path.open("r", encoding="utf-8") as source:
            for line_number, line in enumerate(source, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    yield line_number, json.loads(stripped)
                except json.JSONDecodeError as exc:
                    self.logger.warning(
                        "Skipping invalid JSON on line %s: %s",
                        line_number,
                        exc,
                    )
                    yield line_number, {"_json_error": str(exc)}

    def validate_record(self, record: dict[str, Any]) -> list[str]:
        errors = []
        if record.get("_json_error"):
            errors.append(record["_json_error"])
            return errors

        if not record.get("id"):
            errors.append("missing id")
        if not record.get("neo4j_node_id"):
            errors.append("missing neo4j_node_id")

        vector = record.get("embedding_vector")
        if not isinstance(vector, list) or not vector:
            errors.append("missing embedding_vector")
        elif len(vector) != self.config.vector_dimension:
            errors.append(
                f"embedding_vector dimension {len(vector)} does not match "
                f"configured dimension {self.config.vector_dimension}"
            )
        elif not all(isinstance(value, int | float) for value in vector):
            errors.append("embedding_vector must contain only numeric values")

        return errors

    def to_milvus_row(self, record: dict[str, Any]) -> dict[str, Any]:
        row = {
            "vector": [float(value) for value in record["embedding_vector"]],
        }
        for field_name in self.FIELD_NAMES:
            row[field_name] = self.truncate_varchar(
                field_name=field_name,
                value=self.stringify(record.get(field_name)),
            )
        return row

    def stringify(self, value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        return json.dumps(value, separators=(",", ":"))

    def truncate_varchar(self, field_name: str, value: str) -> str:
        max_length = self.collection_setup.VARCHAR_MAX_LENGTHS.get(field_name)
        if max_length is None:
            return value
        encoded = value.encode("utf-8")
        if len(encoded) <= max_length:
            return value
        truncated = encoded[: max_length - 3].decode("utf-8", errors="ignore")
        return f"{truncated}..."

    def write_batch(
        self,
        batch: list[dict[str, Any]],
        summary: MilvusIngestionSummary,
    ) -> None:
        if not self.replace_collection:
            existing_ids = self.fetch_existing_ids(
                [row["id"] for row in batch]
            )
            if existing_ids:
                new_batch = [
                    row for row in batch
                    if row["id"] not in existing_ids
                ]
                count_existing = len(batch) - len(new_batch)
                summary.already_existing_records += count_existing
                self.logger.info(
                    "Skipping %s already-ingested records;"
                    " inserting %s new records.",
                    count_existing,
                    len(new_batch),
                )
                batch = new_batch

        if not batch:
            return

        try:
            self.replace_batch(batch)
            summary.inserted_records += len(batch)
            self.logger.info("Replaced %s records in Milvus.", len(batch))
        except (AttributeError, NotImplementedError, MilvusException) as exc:
            self.logger.warning(
                "Delete-before-insert failed; trying upsert. Error: %s", exc
            )
            try:
                self.upsert_batch(batch)
                summary.inserted_records += len(batch)
                self.logger.info(
                    "Upserted %s records into Milvus.", len(batch)
                )
            except Exception as fallback_exc:
                summary.failed_records += len(batch)
                self.logger.exception(
                    "Failed to ingest batch: %s", fallback_exc
                )
        except Exception as exc:
            summary.failed_records += len(batch)
            self.logger.exception("Failed to upsert batch: %s", exc)

    def fetch_existing_ids(self, ids: list[str]) -> set[str]:
        if not ids:
            return set()
        try:
            quoted_ids = ",".join(json.dumps(item) for item in ids)
            results = self.client.query(
                collection_name=self.config.collection_name,
                filter=f"id in [{quoted_ids}]",
                output_fields=["id"],
            )
            return {row["id"] for row in results}
        except Exception as exc:
            self.logger.warning(
                "Could not query existing IDs;"
                " proceeding without duplicate check: %s",
                exc,
            )
            return set()

    def replace_batch(self, batch: list[dict[str, Any]]) -> None:
        self.delete_existing_ids([row["id"] for row in batch])
        self.insert_batch(batch)

    def upsert_batch(self, batch: list[dict[str, Any]]) -> None:
        self.client.upsert(
            collection_name=self.config.collection_name,
            data=batch,
        )

    def insert_batch(self, batch: list[dict[str, Any]]) -> None:
        self.client.insert(
            collection_name=self.config.collection_name,
            data=batch,
        )

    def delete_existing_ids(self, ids: list[str]) -> None:
        if not ids:
            return
        quoted_ids = ",".join(json.dumps(item) for item in ids)
        self.client.delete(
            collection_name=self.config.collection_name,
            filter=f"id in [{quoted_ids}]",
        )

    def flush_collection(self) -> None:
        self.logger.info(
            "Flushing Milvus collection %s.", self.config.collection_name
        )
        self.client.flush(collection_name=self.config.collection_name)

    def load_collection(self) -> None:
        self.logger.info(
            "Loading Milvus collection %s.", self.config.collection_name
        )
        self.client.load_collection(
            collection_name=self.config.collection_name
        )

    def collection_row_count(self) -> int | None:
        try:
            stats = self.client.get_collection_stats(
                collection_name=self.config.collection_name
            )
        except Exception as exc:
            self.logger.warning("Unable to read collection stats: %s", exc)
            return None

        row_count = stats.get("row_count")
        return int(row_count) if row_count is not None else None
