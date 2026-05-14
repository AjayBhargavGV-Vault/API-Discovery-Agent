from __future__ import annotations

import logging
import os
from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from typing import Literal

from dotenv import load_dotenv
from neo4j import GraphDatabase
from neo4j import Driver


DATA_DIR = Path(__file__).resolve().parents[1]
PROJECT_SRC = DATA_DIR.parent
BACKEND_DIR = PROJECT_SRC.parent
DEFAULT_ENV_PATH = BACKEND_DIR / ".env"

GraphEntityType = Literal["product", "operation", "schema"]


@dataclass(frozen=True)
class GraphExpandedResult:
    query: str | None
    entity_type: GraphEntityType
    neo4j_node_id: str
    score: float | None
    semantic_result: dict[str, Any]
    graph_context: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Neo4jGraphExpansionConfig:
    uri: str
    username: str
    password: str
    database: str | None = None


class Neo4jGraphExpander:
    ENTITY_LABELS: dict[str, str] = {
        "product": "Product",
        "operation": "Operation",
        "schema": "Schema",
    }

    OPERATION_CONTEXT_QUERY = """
    MATCH (operation:Operation {id: $node_id})
    OPTIONAL MATCH (product:Product)-[:HAS_OPERATION]->(operation)
    OPTIONAL MATCH (resource:Resource)-[:HAS_OPERATION]->(operation)
    OPTIONAL MATCH (operation)-[:USES_REQUEST_SCHEMA]->(request_schema:Schema)
    OPTIONAL MATCH (operation)-[response_rel:RETURNS_RESPONSE_SCHEMA]->(response_schema:Schema)
    OPTIONAL MATCH (operation)-[:HAS_PARAMETER]->(parameter:Parameter)
    OPTIONAL MATCH (operation)-[:REQUIRES_AUTH]->(auth:AuthScheme)
    WITH operation,
         collect(DISTINCT product {.*}) AS products,
         collect(DISTINCT resource {.*}) AS resources,
         collect(DISTINCT request_schema {.*}) AS request_schemas,
         collect(DISTINCT response_schema {.*, status_code: response_rel.status_code}) AS response_schemas,
         collect(DISTINCT parameter {.*}) AS parameters,
         collect(DISTINCT auth {.*}) AS auth_schemes
    OPTIONAL MATCH (operation)-[:USES_REQUEST_SCHEMA|RETURNS_RESPONSE_SCHEMA]->(schema:Schema)
    OPTIONAL MATCH (schema)-[:HAS_FIELD]->(field:Field)
    OPTIONAL MATCH (field)-[:REFERENCES_SCHEMA]->(referenced_schema:Schema)
    WITH operation, products, resources, request_schemas, response_schemas, parameters, auth_schemes,
         collect(DISTINCT {
            schema_id: schema.id,
            schema_name: schema.name,
            field: field {.*},
            referenced_schema: referenced_schema {.*}
         }) AS schema_fields
    OPTIONAL MATCH (resource:Resource)-[:HAS_OPERATION]->(operation)
    OPTIONAL MATCH (resource)-[:HAS_OPERATION]->(related_operation:Operation)
    WHERE related_operation.id <> operation.id
    RETURN operation {.*} AS node,
           products,
           resources,
           request_schemas,
           response_schemas,
           parameters,
           auth_schemes,
           schema_fields,
           collect(DISTINCT related_operation {.*})[..25] AS related_lifecycle_operations
    """

    SCHEMA_CONTEXT_QUERY = """
    MATCH (schema:Schema {id: $node_id})
    OPTIONAL MATCH (schema)-[:HAS_FIELD]->(field:Field)
    OPTIONAL MATCH (field)-[:REFERENCES_SCHEMA]->(referenced_schema:Schema)
    OPTIONAL MATCH (using_operation:Operation)-[:USES_REQUEST_SCHEMA]->(schema)
    OPTIONAL MATCH (returning_operation:Operation)-[:RETURNS_RESPONSE_SCHEMA]->(schema)
    RETURN schema {.*} AS node,
           collect(DISTINCT field {.*}) AS fields,
           collect(DISTINCT referenced_schema {.*}) AS referenced_schemas,
           collect(DISTINCT using_operation {.*})[..25] AS operations_that_use_schema,
           collect(DISTINCT returning_operation {.*})[..25] AS operations_that_return_schema
    """

    PRODUCT_CONTEXT_QUERY = """
    MATCH (product:Product {id: $node_id})
    OPTIONAL MATCH (product)-[:HAS_RESOURCE]->(resource:Resource)
    OPTIONAL MATCH (product)-[:HAS_OPERATION]->(operation:Operation)
    OPTIONAL MATCH (operation)-[:USES_REQUEST_SCHEMA|RETURNS_RESPONSE_SCHEMA]->(schema:Schema)
    RETURN product {.*} AS node,
           product.business_domain AS business_domain,
           collect(DISTINCT resource {.*}) AS related_resources,
           collect(DISTINCT operation {.*})[..25] AS top_operations,
           collect(DISTINCT schema {.*})[..50] AS schemas
    """

    LABEL_QUERY = """
    MATCH (node {id: $node_id})
    RETURN labels(node) AS labels
    LIMIT 1
    """

    def __init__(
        self,
        config: Neo4jGraphExpansionConfig,
        driver: Driver | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self.config = config
        self.driver = driver or GraphDatabase.driver(
            self.config.uri,
            auth=(self.config.username, self.config.password),
        )
        self.logger = logger or logging.getLogger(self.__class__.__name__)

    @classmethod
    def from_env(
        cls,
        env_path: Path = DEFAULT_ENV_PATH,
        logger: logging.Logger | None = None,
    ) -> Neo4jGraphExpander:
        load_dotenv(env_path)
        config = Neo4jGraphExpansionConfig(
            uri=cls.neo4j_uri_from_env(),
            username=cls.required_env("NEO4J_USERNAME"),
            password=cls.required_env("NEO4J_PASSWORD"),
            database=os.getenv("NEO4J_DATABASE") or None,
        )
        return cls(config=config, logger=logger)

    @staticmethod
    def required_env(key: str) -> str:
        value = os.getenv(key)
        if not value:
            raise ValueError(f"Missing required environment variable: {key}")
        return value

    @classmethod
    def neo4j_uri_from_env(cls) -> str:
        uri = cls.required_env("NEO4J_URI")
        verify_certificates = os.getenv("NEO4J_VERIFY_CERTIFICATES", "").lower()
        if uri.startswith("neo4j+s://") and verify_certificates not in {"1", "true", "yes"}:
            return uri.replace("neo4j+s://", "neo4j+ssc://", 1)
        return uri

    def close(self) -> None:
        self.driver.close()

    def expand_top_milvus_result(
        self,
        milvus_results: list[dict[str, Any]],
        query: str | None = None,
    ) -> GraphExpandedResult:
        if not milvus_results:
            raise ValueError("Cannot expand graph context because Milvus returned no results.")

        top_result = milvus_results[0]
        return self.expand_milvus_result(top_result, query=query)

    def expand_milvus_result(
        self,
        milvus_result: dict[str, Any],
        query: str | None = None,
    ) -> GraphExpandedResult:
        entity = self.extract_milvus_entity(milvus_result)
        neo4j_node_id = entity.get("neo4j_node_id")
        if not neo4j_node_id:
            raise ValueError("Milvus result is missing neo4j_node_id.")

        entity_type = self.normalize_entity_type(entity.get("entity_type"))
        if not entity_type:
            entity_type = self.entity_type_for_node(neo4j_node_id)

        graph_context = self.expand_node(neo4j_node_id=neo4j_node_id, entity_type=entity_type)
        return GraphExpandedResult(
            query=query,
            entity_type=entity_type,
            neo4j_node_id=neo4j_node_id,
            score=self.extract_score(milvus_result),
            semantic_result=entity,
            graph_context=graph_context,
        )

    def expand_node(
        self,
        neo4j_node_id: str,
        entity_type: GraphEntityType | str | None = None,
    ) -> dict[str, Any]:
        normalized_type = self.normalize_entity_type(entity_type) if entity_type else None
        if not normalized_type:
            normalized_type = self.entity_type_for_node(neo4j_node_id)

        self.logger.info("Expanding Neo4j node %s as %s", neo4j_node_id, normalized_type)
        if normalized_type == "operation":
            return self.expand_operation(neo4j_node_id)
        if normalized_type == "schema":
            return self.expand_schema(neo4j_node_id)
        if normalized_type == "product":
            return self.expand_product(neo4j_node_id)
        raise ValueError(f"Unsupported entity_type: {entity_type}")

    def expand_operation(self, neo4j_node_id: str) -> dict[str, Any]:
        row = self.run_single(self.OPERATION_CONTEXT_QUERY, {"node_id": neo4j_node_id})
        if not row:
            raise ValueError(f"Operation node does not exist in Neo4j: {neo4j_node_id}")
        return {
            "node": row.get("node") or {},
            "product": self.first_non_empty(row.get("products")),
            "resources": self.clean_list(row.get("resources")),
            "request_schemas": self.clean_list(row.get("request_schemas")),
            "response_schemas": self.clean_list(row.get("response_schemas")),
            "parameters": self.clean_list(row.get("parameters")),
            "auth_schemes": self.clean_list(row.get("auth_schemes")),
            "related_lifecycle_operations": self.clean_list(
                row.get("related_lifecycle_operations")
            ),
            "schema_fields": self.clean_schema_fields(row.get("schema_fields")),
        }

    def expand_schema(self, neo4j_node_id: str) -> dict[str, Any]:
        row = self.run_single(self.SCHEMA_CONTEXT_QUERY, {"node_id": neo4j_node_id})
        if not row:
            raise ValueError(f"Schema node does not exist in Neo4j: {neo4j_node_id}")
        return {
            "node": row.get("node") or {},
            "fields": self.clean_list(row.get("fields")),
            "referenced_schemas": self.clean_list(row.get("referenced_schemas")),
            "operations_that_use_schema": self.clean_list(row.get("operations_that_use_schema")),
            "operations_that_return_schema": self.clean_list(
                row.get("operations_that_return_schema")
            ),
        }

    def expand_product(self, neo4j_node_id: str) -> dict[str, Any]:
        row = self.run_single(self.PRODUCT_CONTEXT_QUERY, {"node_id": neo4j_node_id})
        if not row:
            raise ValueError(f"Product node does not exist in Neo4j: {neo4j_node_id}")
        return {
            "node": row.get("node") or {},
            "business_domain": row.get("business_domain"),
            "top_operations": self.clean_list(row.get("top_operations")),
            "schemas": self.clean_list(row.get("schemas")),
            "related_resources": self.clean_list(row.get("related_resources")),
        }

    def entity_type_for_node(self, neo4j_node_id: str) -> GraphEntityType:
        row = self.run_single(self.LABEL_QUERY, {"node_id": neo4j_node_id})
        labels = set(row.get("labels") or []) if row else set()
        for entity_type, label in self.ENTITY_LABELS.items():
            if label in labels:
                return entity_type  # type: ignore[return-value]
        raise ValueError(f"Neo4j node is not a supported retrieval entity: {neo4j_node_id}")

    def run_single(self, query: str, parameters: dict[str, Any]) -> dict[str, Any] | None:
        with self.driver.session(database=self.config.database) as session:
            result = session.run(query, parameters)
            record = result.single() if hasattr(result, "single") else None
            if record is None:
                return None
            return dict(record)

    def extract_milvus_entity(self, milvus_result: dict[str, Any]) -> dict[str, Any]:
        entity = milvus_result.get("entity")
        return entity if isinstance(entity, dict) else milvus_result

    def extract_score(self, milvus_result: dict[str, Any]) -> float | None:
        score = milvus_result.get("distance", milvus_result.get("score"))
        return float(score) if score is not None else None

    def normalize_entity_type(self, value: Any) -> GraphEntityType | None:
        if not isinstance(value, str):
            return None
        normalized = value.lower()
        return normalized if normalized in self.ENTITY_LABELS else None  # type: ignore[return-value]

    def clean_schema_fields(self, values: Any) -> list[dict[str, Any]]:
        fields: list[dict[str, Any]] = []
        for value in self.clean_list(values):
            field = value.get("field")
            if not field:
                continue
            fields.append(
                {
                    "schema_id": value.get("schema_id"),
                    "schema_name": value.get("schema_name"),
                    "field": field,
                    "referenced_schema": value.get("referenced_schema") or None,
                }
            )
        return fields

    def clean_list(self, values: Any) -> list[Any]:
        if not isinstance(values, list):
            return []
        return [value for value in values if value]

    def first_non_empty(self, values: Any) -> dict[str, Any] | None:
        for value in self.clean_list(values):
            if isinstance(value, dict):
                return value
        return None


class GraphRagRetriever:
    def __init__(
        self,
        semantic_retriever: Any,
        graph_expander: Neo4jGraphExpander,
        logger: logging.Logger | None = None,
    ) -> None:
        self.semantic_retriever = semantic_retriever
        self.graph_expander = graph_expander
        self.logger = logger or logging.getLogger(self.__class__.__name__)

    def retrieve(
        self,
        query: str,
        filter_expression: str | None = None,
        max_graph_expansions: int | None = None,
    ) -> dict[str, Any]:
        self.logger.info("Running Milvus retrieval for query=%r", query)
        query_embedding = self.semantic_retriever.embed_query(query)
        milvus_results = self.semantic_retriever.search(
            query_embedding=query_embedding,
            filter_expression=filter_expression,
        )
        semantic_hits = [self.format_semantic_hit(hit) for hit in milvus_results]
        graph_context = self.expand_graph_context(
            milvus_results=milvus_results,
            query=query,
            max_graph_expansions=max_graph_expansions,
        )

        return {
            "query": query,
            "semantic_hits": semantic_hits,
            "graph_context": [expanded.to_dict() for expanded in graph_context],
            "recommended_operations": self.collect_recommended_operations(
                semantic_hits=semantic_hits,
                graph_context=graph_context,
            ),
            "schemas": self.collect_schemas(graph_context),
            "fields": self.collect_fields(graph_context),
            "confidence_notes": self.build_confidence_notes(
                semantic_hits=semantic_hits,
                graph_context=graph_context,
                filter_expression=filter_expression,
            ),
        }

    def expand_graph_context(
        self,
        milvus_results: list[dict[str, Any]],
        query: str,
        max_graph_expansions: int | None = None,
    ) -> list[GraphExpandedResult]:
        expanded_results: list[GraphExpandedResult] = []
        results_to_expand = milvus_results[:max_graph_expansions] if max_graph_expansions else milvus_results
        for result in results_to_expand:
            try:
                expanded_results.append(
                    self.graph_expander.expand_milvus_result(
                        milvus_result=result,
                        query=query,
                    )
                )
            except ValueError as error:
                self.logger.warning("Skipping graph expansion for a Milvus hit: %s", error)
        return expanded_results

    def format_semantic_hit(self, hit: dict[str, Any]) -> dict[str, Any]:
        entity = self.graph_expander.extract_milvus_entity(hit)
        return {
            "score": self.graph_expander.extract_score(hit),
            "id": entity.get("id"),
            "entity_type": entity.get("entity_type"),
            "entity_id": entity.get("entity_id"),
            "title": entity.get("title"),
            "provider": entity.get("provider"),
            "spec_id": entity.get("spec_id"),
            "spec_version": entity.get("spec_version"),
            "business_domain": entity.get("business_domain"),
            "product_key": entity.get("product_key"),
            "method": entity.get("method"),
            "path": entity.get("path"),
            "neo4j_node_id": entity.get("neo4j_node_id"),
            "body_preview": self.preview(entity.get("body")),
        }

    def collect_recommended_operations(
        self,
        semantic_hits: list[dict[str, Any]],
        graph_context: list[GraphExpandedResult],
    ) -> list[dict[str, Any]]:
        operations: list[dict[str, Any]] = []
        for hit in semantic_hits:
            if hit.get("entity_type") == "operation":
                operations.append(
                    {
                        "id": hit.get("neo4j_node_id"),
                        "title": hit.get("title"),
                        "method": hit.get("method"),
                        "path": hit.get("path"),
                        "product_key": hit.get("product_key"),
                        "business_domain": hit.get("business_domain"),
                        "score": hit.get("score"),
                        "source": "semantic_hit",
                    }
                )

        for expanded in graph_context:
            context = expanded.graph_context
            entity_type = expanded.entity_type
            if entity_type == "operation":
                operations.append(
                    {
                        **self.operation_summary(context.get("node")),
                        "score": expanded.score,
                        "source": "graph_primary_operation",
                    }
                )
                for operation in context.get("related_lifecycle_operations", []):
                    operations.append(
                        {
                            **self.operation_summary(operation),
                            "score": None,
                            "source": "related_lifecycle_operation",
                        }
                    )
            elif entity_type == "schema":
                for operation in context.get("operations_that_use_schema", []):
                    operations.append(
                        {
                            **self.operation_summary(operation),
                            "score": expanded.score,
                            "source": "schema_used_by_operation",
                        }
                    )
                for operation in context.get("operations_that_return_schema", []):
                    operations.append(
                        {
                            **self.operation_summary(operation),
                            "score": expanded.score,
                            "source": "schema_returned_by_operation",
                        }
                    )
            elif entity_type == "product":
                for operation in context.get("top_operations", []):
                    operations.append(
                        {
                            **self.operation_summary(operation),
                            "score": expanded.score,
                            "source": "product_operation",
                        }
                    )

        return self.dedupe_by_id(operations)

    def collect_schemas(self, graph_context: list[GraphExpandedResult]) -> list[dict[str, Any]]:
        schemas: list[dict[str, Any]] = []
        for expanded in graph_context:
            context = expanded.graph_context
            if expanded.entity_type == "schema":
                schemas.append({**self.schema_summary(context.get("node")), "source": "graph_primary_schema"})
                for schema in context.get("referenced_schemas", []):
                    schemas.append({**self.schema_summary(schema), "source": "referenced_schema"})
            elif expanded.entity_type == "operation":
                for schema in context.get("request_schemas", []):
                    schemas.append({**self.schema_summary(schema), "source": "request_schema"})
                for schema in context.get("response_schemas", []):
                    schemas.append({**self.schema_summary(schema), "source": "response_schema"})
            elif expanded.entity_type == "product":
                for schema in context.get("schemas", []):
                    schemas.append({**self.schema_summary(schema), "source": "product_schema"})
        return self.dedupe_by_id(schemas)

    def collect_fields(self, graph_context: list[GraphExpandedResult]) -> list[dict[str, Any]]:
        fields: list[dict[str, Any]] = []
        for expanded in graph_context:
            context = expanded.graph_context
            if expanded.entity_type == "schema":
                for field in context.get("fields", []):
                    fields.append({**self.field_summary(field), "schema_id": context.get("node", {}).get("id")})
            elif expanded.entity_type == "operation":
                for field_context in context.get("schema_fields", []):
                    field = field_context.get("field") or {}
                    fields.append(
                        {
                            **self.field_summary(field),
                            "schema_id": field_context.get("schema_id"),
                            "schema_name": field_context.get("schema_name"),
                            "referenced_schema": field_context.get("referenced_schema"),
                        }
                    )
        return self.dedupe_by_id(fields)

    def build_confidence_notes(
        self,
        semantic_hits: list[dict[str, Any]],
        graph_context: list[GraphExpandedResult],
        filter_expression: str | None,
    ) -> list[str]:
        notes: list[str] = []
        if not semantic_hits:
            return ["No semantic hits were returned from Milvus."]

        top_hit = semantic_hits[0]
        top_score = top_hit.get("score")
        notes.append(
            "Top semantic match is "
            f"{top_hit.get('title') or top_hit.get('neo4j_node_id')} "
            f"with score {top_score}."
        )
        if filter_expression:
            notes.append(f"Milvus metadata filter applied: {filter_expression}.")
        if not graph_context:
            notes.append("No Neo4j graph context could be expanded for the returned hits.")
        elif len(graph_context) < len(semantic_hits):
            notes.append(
                f"Expanded {len(graph_context)} of {len(semantic_hits)} semantic hits through Neo4j."
            )
        else:
            notes.append("Every semantic hit was expanded through Neo4j.")
        if top_hit.get("entity_type") != "operation":
            notes.append(
                "Top semantic hit is not an operation; recommended operations were inferred from graph neighbors."
            )
        return notes

    def operation_summary(self, operation: Any) -> dict[str, Any]:
        operation = operation if isinstance(operation, dict) else {}
        return {
            "id": operation.get("id"),
            "operation_id": operation.get("operation_id"),
            "title": operation.get("summary") or operation.get("operation_id"),
            "method": operation.get("method"),
            "path": operation.get("path"),
            "summary": operation.get("summary"),
            "description": operation.get("description"),
        }

    def schema_summary(self, schema: Any) -> dict[str, Any]:
        schema = schema if isinstance(schema, dict) else {}
        return {
            "id": schema.get("id"),
            "name": schema.get("name"),
            "title": schema.get("title") or schema.get("name"),
            "type": schema.get("type"),
            "description": schema.get("description"),
            "status_code": schema.get("status_code"),
        }

    def field_summary(self, field: Any) -> dict[str, Any]:
        field = field if isinstance(field, dict) else {}
        return {
            "id": field.get("id"),
            "name": field.get("name"),
            "path": field.get("path"),
            "type": field.get("type"),
            "description": field.get("description"),
            "required": field.get("required"),
            "enum_values": field.get("enum_values"),
        }

    def dedupe_by_id(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        seen: set[str] = set()
        deduped: list[dict[str, Any]] = []
        for row in rows:
            row_id = row.get("id")
            if not row_id:
                deduped.append(row)
                continue
            if row_id in seen:
                continue
            seen.add(row_id)
            deduped.append(row)
        return deduped

    def preview(self, value: Any, limit: int = 500) -> str:
        text = "" if value is None else str(value)
        return text if len(text) <= limit else f"{text[:limit]}..."


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )
