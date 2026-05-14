from __future__ import annotations

import hashlib
from dataclasses import asdict
from dataclasses import dataclass
from datetime import datetime
from datetime import timezone
from typing import Literal


EntityType = Literal["product", "operation", "schema"]


@dataclass
class ApiSearchDocument:
    id: str
    spec_id: str
    entity_type: EntityType
    entity_id: str
    title: str
    body: str
    keywords: list[str]
    provider: str
    spec_version: str
    business_domain: str | None
    product_key: str | None
    method: str | None
    path: str | None
    neo4j_node_id: str
    embedding_model: str | None
    embedding_vector: list[float] | None
    indexed_at: datetime

    def __post_init__(self) -> None:
        if self.entity_type not in {"product", "operation", "schema"}:
            raise ValueError(f"Unsupported entity_type: {self.entity_type}")
        if not self.body or not self.body.strip():
            raise ValueError("ApiSearchDocument.body cannot be empty.")
        if not self.neo4j_node_id or not self.neo4j_node_id.strip():
            raise ValueError("ApiSearchDocument.neo4j_node_id cannot be empty.")
        if not self.title or not self.title.strip():
            raise ValueError("ApiSearchDocument.title cannot be empty.")

    @classmethod
    def create(
        cls,
        spec_id: str,
        entity_type: EntityType,
        entity_id: str,
        title: str,
        body: str,
        keywords: list[str],
        provider: str,
        spec_version: str,
        neo4j_node_id: str,
        business_domain: str | None = None,
        product_key: str | None = None,
        method: str | None = None,
        path: str | None = None,
        embedding_model: str | None = None,
        embedding_vector: list[float] | None = None,
        indexed_at: datetime | None = None,
    ) -> ApiSearchDocument:
        return cls(
            id=cls.document_id(entity_type=entity_type, entity_id=entity_id),
            spec_id=spec_id,
            entity_type=entity_type,
            entity_id=entity_id,
            title=title,
            body=body,
            keywords=keywords,
            provider=provider,
            spec_version=spec_version,
            business_domain=business_domain,
            product_key=product_key,
            method=method,
            path=path,
            neo4j_node_id=neo4j_node_id,
            embedding_model=embedding_model,
            embedding_vector=embedding_vector,
            indexed_at=indexed_at or datetime.now(timezone.utc),
        )

    @staticmethod
    def document_id(entity_type: str, entity_id: str) -> str:
        raw_id = f"{entity_type}:{entity_id}"
        digest = hashlib.sha256(raw_id.encode("utf-8")).hexdigest()[:20]
        return f"api_search_document:{digest}"

    @staticmethod
    def neo4j_product_id(provider: str, product_name: str) -> str:
        return ApiSearchDocument.stable_id("product", provider, product_name)

    @staticmethod
    def neo4j_operation_id(provider: str, operation_key: str) -> str:
        return ApiSearchDocument.stable_id("operation", provider, operation_key)

    @staticmethod
    def neo4j_schema_id(provider: str, schema_name: str) -> str:
        return ApiSearchDocument.stable_id("schema", provider, schema_name)

    @staticmethod
    def stable_id(prefix: str, *parts: str) -> str:
        raw_value = ":".join(parts)
        digest = hashlib.sha256(raw_value.encode("utf-8")).hexdigest()[:20]
        return f"{prefix}:{digest}"

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["indexed_at"] = self.indexed_at.isoformat()
        return payload
