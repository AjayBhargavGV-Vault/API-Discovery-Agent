from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from pymilvus import MilvusClient
from sentence_transformers import SentenceTransformer


DATA_DIR = Path(__file__).resolve().parents[1]
PROJECT_SRC = DATA_DIR.parent
BACKEND_DIR = PROJECT_SRC.parent
DEFAULT_ENV_PATH = BACKEND_DIR / ".env"
DEFAULT_OUTPUT_PATH = DATA_DIR / "retrieval_test_results.json"
DEFAULT_COLLECTION_NAME = "api_search_documents"


DEFAULT_TEST_QUERIES = [
    "How do I confirm a Stripe payment?",
    "Which API creates a customer?",
    "How do I refund a payment?",
    "Which schema contains currency?",
    "How do I create a subscription?",
]


@dataclass(frozen=True)
class RetrievalTestConfig:
    milvus_uri: str
    collection_name: str
    embedding_model: str
    top_k: int = 5
    milvus_token: str | None = None
    milvus_username: str | None = None
    milvus_password: str | None = None
    milvus_database: str | None = None


class MilvusRetrievalTest:
    OUTPUT_FIELDS = [
        "id",
        "entity_type",
        "title",
        "body",
        "provider",
        "spec_id",
        "spec_version",
        "business_domain",
        "product_key",
        "method",
        "path",
        "neo4j_node_id",
        "embedding_model",
        "indexed_at",
    ]

    def __init__(
        self,
        config: RetrievalTestConfig,
        output_path: Path = DEFAULT_OUTPUT_PATH,
        logger: logging.Logger | None = None,
    ) -> None:
        self.config = config
        self.output_path = output_path.resolve()
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        self.client = self.create_client()
        self.model = SentenceTransformer(self.config.embedding_model)

    @classmethod
    def from_env(
        cls,
        env_path: Path = DEFAULT_ENV_PATH,
        output_path: Path = DEFAULT_OUTPUT_PATH,
        top_k: int = 5,
        logger: logging.Logger | None = None,
    ) -> MilvusRetrievalTest:
        load_dotenv(env_path)
        config = RetrievalTestConfig(
            milvus_uri=cls.required_env("MILVUS_URI"),
            collection_name=os.getenv("MILVUS_COLLECTION_NAME") or DEFAULT_COLLECTION_NAME,
            embedding_model=cls.required_env("EMBEDDING_MODEL"),
            top_k=top_k,
            milvus_token=os.getenv("MILVUS_TOKEN") or None,
            milvus_username=os.getenv("MILVUS_USERNAME") or os.getenv("MILVUS_USER") or None,
            milvus_password=os.getenv("MILVUS_PASSWORD") or None,
            milvus_database=os.getenv("MILVUS_DATABASE") or os.getenv("MILVUS_DB_NAME") or None,
        )
        return cls(config=config, output_path=output_path, logger=logger)

    @staticmethod
    def required_env(key: str) -> str:
        value = os.getenv(key)
        if not value:
            raise ValueError(f"Missing required environment variable: {key}")
        return value

    def create_client(self) -> MilvusClient:
        kwargs = {"uri": self.config.milvus_uri}
        if self.config.milvus_token:
            kwargs["token"] = self.config.milvus_token
        if self.config.milvus_username and self.config.milvus_password:
            kwargs["user"] = self.config.milvus_username
            kwargs["password"] = self.config.milvus_password
        if self.config.milvus_database:
            kwargs["db_name"] = self.config.milvus_database
        return MilvusClient(**kwargs)

    def run(
        self,
        queries: list[str] = DEFAULT_TEST_QUERIES,
        filters: list[str | None] | None = None,
    ) -> dict[str, Any]:
        filters = filters or [None]
        self.client.load_collection(collection_name=self.config.collection_name)
        report = {
            "collection_name": self.config.collection_name,
            "embedding_model": self.config.embedding_model,
            "top_k": self.config.top_k,
            "queries": [],
        }

        for query in queries:
            query_embedding = self.embed_query(query)
            query_result = {"query": query, "filter_runs": []}
            for filter_expression in filters:
                self.logger.info("Searching query=%r filter=%r", query, filter_expression)
                hits = self.search(query_embedding=query_embedding, filter_expression=filter_expression)
                query_result["filter_runs"].append(
                    {
                        "filter": filter_expression,
                        "results": [self.format_hit(hit) for hit in hits],
                    }
                )
            report["queries"].append(query_result)

        self.write_report(report)
        return report

    def embed_query(self, query: str) -> list[float]:
        vector = self.model.encode(
            [query],
            normalize_embeddings=True,
            show_progress_bar=False,
        )[0]
        return [float(value) for value in vector.tolist()]

    def search(
        self,
        query_embedding: list[float],
        filter_expression: str | None = None,
    ) -> list[dict[str, Any]]:
        kwargs = {
            "collection_name": self.config.collection_name,
            "data": [query_embedding],
            "limit": self.config.top_k,
            "output_fields": self.OUTPUT_FIELDS,
            "search_params": {"metric_type": "COSINE"},
        }
        if filter_expression:
            kwargs["filter"] = self.normalize_filter_expression(filter_expression)
        results = self.client.search(**kwargs)
        return results[0] if results else []

    def normalize_filter_expression(self, filter_expression: str) -> str:
        varchar_fields = {
            "entity_type",
            "business_domain",
            "method",
            "provider",
            "product_key",
            "spec_id",
            "spec_version",
            "path",
            "neo4j_node_id",
            "embedding_model",
        }

        def quote_unquoted_value(match: re.Match[str]) -> str:
            field = match.group("field")
            operator = match.group("operator")
            value = match.group("value")
            if value.startswith(('"', "'")):
                return match.group(0)
            if field not in varchar_fields:
                return match.group(0)
            return f'{field} {operator} "{value}"'

        return re.sub(
            r"(?P<field>\b[a-zA-Z_][a-zA-Z0-9_]*\b)\s*"
            r"(?P<operator>==|!=)\s*"
            r"(?P<value>[A-Za-z0-9_./:{}/-]+)",
            quote_unquoted_value,
            filter_expression,
        )

    def format_hit(self, hit: dict[str, Any]) -> dict[str, Any]:
        entity = hit.get("entity", {})
        return {
            "score": hit.get("distance"),
            "entity_type": entity.get("entity_type"),
            "title": entity.get("title"),
            "method": entity.get("method"),
            "path": entity.get("path"),
            "product_key": entity.get("product_key"),
            "business_domain": entity.get("business_domain"),
            "neo4j_node_id": entity.get("neo4j_node_id"),
            "body_preview": self.preview(entity.get("body")),
        }

    def preview(self, value: Any, limit: int = 500) -> str:
        text = "" if value is None else str(value)
        return text if len(text) <= limit else f"{text[:limit]}..."

    def write_report(self, report: dict[str, Any]) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with self.output_path.open("w", encoding="utf-8") as target:
            json.dump(report, target, indent=2)
            target.write("\n")


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )
