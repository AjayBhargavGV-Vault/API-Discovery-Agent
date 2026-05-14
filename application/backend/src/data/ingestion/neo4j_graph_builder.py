from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from neo4j import GraphDatabase
from neo4j import Session


DATA_DIR = Path(__file__).resolve().parents[1]
PROJECT_SRC = DATA_DIR.parent
BACKEND_DIR = PROJECT_SRC.parent
DEFAULT_ENV_PATH = BACKEND_DIR / ".env"


class Neo4jGraphBuilder:
    NODE_LABELS = (
        "Provider",
        "Product",
        "Resource",
        "Operation",
        "Schema",
        "Field",
        "AuthScheme",
        "Parameter",
    )

    HTTP_METHODS = {"get", "post", "put", "patch", "delete", "options", "head", "trace"}

    def __init__(
        self,
        provider: str,
        source_url: str,
        env_path: Path = DEFAULT_ENV_PATH,
    ) -> None:
        load_dotenv(env_path)
        self.provider = provider
        self.source_url = source_url
        self.uri = self.neo4j_uri_from_env()
        self.username = self.required_env("NEO4J_USERNAME")
        self.password = self.required_env("NEO4J_PASSWORD")
        self.database = os.getenv("NEO4J_DATABASE") or None
        self.driver = GraphDatabase.driver(
            self.uri,
            auth=(self.username, self.password),
        )

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

    def build(self, spec: dict[str, Any]) -> None:
        graph = self.collect_graph(spec)
        with self.driver.session(database=self.database) as session:
            self.create_constraints(session)
            self.bulk_merge_nodes(session, graph["nodes"])
            self.bulk_merge_relationships(session, graph["relationships"])

    def create_constraints(self, session: Session) -> None:
        for label in self.NODE_LABELS:
            session.run(
                f"CREATE CONSTRAINT {label.lower()}_id_unique IF NOT EXISTS "
                f"FOR (n:{label}) REQUIRE n.id IS UNIQUE"
            )

    def collect_graph(self, spec: dict[str, Any]) -> dict[str, Any]:
        nodes: dict[str, dict[str, dict[str, Any]]] = {
            label: {} for label in self.NODE_LABELS
        }
        relationships: list[dict[str, Any]] = []
        version = self.api_version(spec)
        provider_id = self.provider_id()
        info = spec.get("info", {})

        self.add_node(
            nodes,
            "Provider",
            provider_id,
            {
                "name": self.provider,
                "title": info.get("title") if isinstance(info, dict) else None,
                "version": version,
                "source_url": self.source_url,
                "openapi_version": spec.get("openapi"),
            },
        )

        self.collect_schema_nodes(nodes=nodes, relationships=relationships, spec=spec, version=version)
        self.collect_operation_nodes(
            nodes=nodes,
            relationships=relationships,
            spec=spec,
            version=version,
            provider_id=provider_id,
        )

        return {"nodes": nodes, "relationships": relationships}

    def collect_schema_nodes(
        self,
        nodes: dict[str, dict[str, dict[str, Any]]],
        relationships: list[dict[str, Any]],
        spec: dict[str, Any],
        version: str | None,
    ) -> None:
        schemas = spec.get("components", {}).get("schemas", {})
        if not isinstance(schemas, dict):
            return

        for schema_name, schema in schemas.items():
            if not isinstance(schema, dict):
                continue

            schema_id = self.schema_id(schema_name)
            self.add_node(
                nodes,
                "Schema",
                schema_id,
                {
                    "name": schema_name,
                    "type": schema.get("type"),
                    "description": self.clean_text(schema.get("description")),
                    "provider": self.provider,
                    "version": version,
                    "source_url": self.source_url,
                },
            )

            properties = schema.get("properties", {})
            required_fields = set(self.as_string_list(schema.get("required")))
            if not isinstance(properties, dict):
                continue

            for field_name, field_schema in properties.items():
                if not isinstance(field_schema, dict):
                    continue

                field_id = self.field_id(schema_name=schema_name, field_path=field_name)
                self.add_node(
                    nodes,
                    "Field",
                    field_id,
                    {
                        "name": field_name,
                        "path": field_name,
                        "type": self.infer_schema_type(field_schema),
                        "description": self.clean_text(field_schema.get("description")),
                        "required": field_name in required_fields,
                        "enum_values": field_schema.get("enum"),
                        "provider": self.provider,
                        "source_url": self.source_url,
                    },
                )
                self.add_relationship(
                    relationships,
                    source_label="Schema",
                    source_id=schema_id,
                    relationship_type="HAS_FIELD",
                    target_label="Field",
                    target_id=field_id,
                )

                for schema_ref in self.extract_schema_refs(field_schema):
                    referenced_schema_name = self.schema_name_from_ref(schema_ref)
                    if not referenced_schema_name:
                        continue

                    referenced_schema_id = self.schema_id(referenced_schema_name)
                    self.add_node(
                        nodes,
                        "Schema",
                        referenced_schema_id,
                        {
                            "name": referenced_schema_name,
                            "provider": self.provider,
                            "source_url": self.source_url,
                        },
                    )
                    self.add_relationship(
                        relationships,
                        source_label="Field",
                        source_id=field_id,
                        relationship_type="REFERENCES_SCHEMA",
                        target_label="Schema",
                        target_id=referenced_schema_id,
                    )

    def collect_operation_nodes(
        self,
        nodes: dict[str, dict[str, dict[str, Any]]],
        relationships: list[dict[str, Any]],
        spec: dict[str, Any],
        version: str | None,
        provider_id: str,
    ) -> None:
        paths = spec.get("paths", {})
        if not isinstance(paths, dict):
            return

        for path, path_item in paths.items():
            if not isinstance(path_item, dict):
                continue

            path_level_parameters = self.as_list(path_item.get("parameters"))
            for method, operation in path_item.items():
                method = method.lower()
                if method not in self.HTTP_METHODS or not isinstance(operation, dict):
                    continue

                operation_key = operation.get("operationId") or f"{method.upper()}:{path}"
                operation_id = self.operation_id(operation_key)
                product_name = self.infer_product(operation=operation, path=path)
                resource_name = self.infer_resource(path)
                product_id = self.product_id(product_name)
                resource_id = self.resource_id(resource_name)

                self.add_node(
                    nodes,
                    "Product",
                    product_id,
                    {
                        "name": product_name,
                        "provider": self.provider,
                        "version": version,
                        "source_url": self.source_url,
                    },
                )
                self.add_node(
                    nodes,
                    "Resource",
                    resource_id,
                    {
                        "name": resource_name,
                        "path_prefix": self.resource_path_prefix(path),
                        "provider": self.provider,
                        "source_url": self.source_url,
                    },
                )
                self.add_node(
                    nodes,
                    "Operation",
                    operation_id,
                    {
                        "operation_id": operation_key,
                        "method": method.upper(),
                        "path": path,
                        "summary": operation.get("summary"),
                        "description": self.clean_text(operation.get("description")),
                        "provider": self.provider,
                        "version": version,
                        "source_url": self.source_url,
                    },
                )

                self.add_relationship(
                    relationships,
                    source_label="Provider",
                    source_id=provider_id,
                    relationship_type="HAS_PRODUCT",
                    target_label="Product",
                    target_id=product_id,
                )
                self.add_relationship(
                    relationships,
                    source_label="Product",
                    source_id=product_id,
                    relationship_type="HAS_RESOURCE",
                    target_label="Resource",
                    target_id=resource_id,
                )
                self.add_relationship(
                    relationships,
                    source_label="Product",
                    source_id=product_id,
                    relationship_type="HAS_OPERATION",
                    target_label="Operation",
                    target_id=operation_id,
                )
                self.add_relationship(
                    relationships,
                    source_label="Resource",
                    source_id=resource_id,
                    relationship_type="HAS_OPERATION",
                    target_label="Operation",
                    target_id=operation_id,
                )

                self.collect_operation_parameters(
                    nodes=nodes,
                    relationships=relationships,
                    operation_id=operation_id,
                    parameters=[*path_level_parameters, *self.as_list(operation.get("parameters"))],
                )
                self.collect_operation_request_schema(
                    nodes=nodes,
                    relationships=relationships,
                    operation_id=operation_id,
                    operation=operation,
                )
                self.collect_operation_response_schemas(
                    nodes=nodes,
                    relationships=relationships,
                    operation_id=operation_id,
                    operation=operation,
                )
                self.collect_operation_auth_schemes(
                    nodes=nodes,
                    relationships=relationships,
                    operation_id=operation_id,
                    operation=operation,
                    global_security=spec.get("security"),
                )

    def collect_operation_parameters(
        self,
        nodes: dict[str, dict[str, dict[str, Any]]],
        relationships: list[dict[str, Any]],
        operation_id: str,
        parameters: list[Any],
    ) -> None:
        for parameter in parameters:
            if not isinstance(parameter, dict):
                continue

            parameter_name = parameter.get("name")
            parameter_location = parameter.get("in")
            if not parameter_name or not parameter_location:
                continue

            parameter_id = self.parameter_id(
                operation_id=operation_id,
                name=parameter_name,
                location=parameter_location,
            )
            self.add_node(
                nodes,
                "Parameter",
                parameter_id,
                {
                    "name": parameter_name,
                    "location": parameter_location,
                    "required": bool(parameter.get("required", False)),
                    "type": self.infer_schema_type(parameter.get("schema", {})),
                    "description": self.clean_text(parameter.get("description")),
                    "provider": self.provider,
                    "source_url": self.source_url,
                },
            )
            self.add_relationship(
                relationships,
                source_label="Operation",
                source_id=operation_id,
                relationship_type="HAS_PARAMETER",
                target_label="Parameter",
                target_id=parameter_id,
            )

    def collect_operation_request_schema(
        self,
        nodes: dict[str, dict[str, dict[str, Any]]],
        relationships: list[dict[str, Any]],
        operation_id: str,
        operation: dict[str, Any],
    ) -> None:
        schema_name = self.schema_name_from_schema(
            self.application_json_schema(operation.get("requestBody"))
        )
        if not schema_name:
            return

        schema_id = self.schema_id(schema_name)
        self.add_node(
            nodes,
            "Schema",
            schema_id,
            {"name": schema_name, "provider": self.provider, "source_url": self.source_url},
        )
        self.add_relationship(
            relationships,
            source_label="Operation",
            source_id=operation_id,
            relationship_type="USES_REQUEST_SCHEMA",
            target_label="Schema",
            target_id=schema_id,
        )

    def collect_operation_response_schemas(
        self,
        nodes: dict[str, dict[str, dict[str, Any]]],
        relationships: list[dict[str, Any]],
        operation_id: str,
        operation: dict[str, Any],
    ) -> None:
        responses = operation.get("responses", {})
        if not isinstance(responses, dict):
            return

        for status_code, response in responses.items():
            schema_name = self.schema_name_from_schema(self.application_json_schema(response))
            if not schema_name:
                continue

            schema_id = self.schema_id(schema_name)
            self.add_node(
                nodes,
                "Schema",
                schema_id,
                {"name": schema_name, "provider": self.provider, "source_url": self.source_url},
            )
            self.add_relationship(
                relationships,
                source_label="Operation",
                source_id=operation_id,
                relationship_type="RETURNS_RESPONSE_SCHEMA",
                target_label="Schema",
                target_id=schema_id,
                identity_props={"status_code": str(status_code)},
            )

    def collect_operation_auth_schemes(
        self,
        nodes: dict[str, dict[str, dict[str, Any]]],
        relationships: list[dict[str, Any]],
        operation_id: str,
        operation: dict[str, Any],
        global_security: Any,
    ) -> None:
        security = operation["security"] if "security" in operation else global_security
        for requirement in self.as_list(security):
            if not isinstance(requirement, dict):
                continue

            for scheme_name in requirement.keys():
                auth_id = self.auth_scheme_id(scheme_name)
                self.add_node(
                    nodes,
                    "AuthScheme",
                    auth_id,
                    {
                        "name": scheme_name,
                        "provider": self.provider,
                        "source_url": self.source_url,
                    },
                )
                self.add_relationship(
                    relationships,
                    source_label="Operation",
                    source_id=operation_id,
                    relationship_type="REQUIRES_AUTH",
                    target_label="AuthScheme",
                    target_id=auth_id,
                )

    def add_node(
        self,
        nodes: dict[str, dict[str, dict[str, Any]]],
        label: str,
        node_id: str,
        props: dict[str, Any],
    ) -> None:
        existing = nodes[label].get(node_id, {})
        merged_props = {**existing, **self.clean_props(props), "id": node_id}
        nodes[label][node_id] = merged_props

    def add_relationship(
        self,
        relationships: list[dict[str, Any]],
        source_label: str,
        source_id: str,
        relationship_type: str,
        target_label: str,
        target_id: str,
        identity_props: dict[str, Any] | None = None,
        props: dict[str, Any] | None = None,
    ) -> None:
        relationships.append(
            {
                "source_label": source_label,
                "source_id": source_id,
                "relationship_type": relationship_type,
                "target_label": target_label,
                "target_id": target_id,
                "identity_props": self.clean_props(identity_props or {}),
                "props": self.clean_props(props or {}),
            }
        )

    def bulk_merge_nodes(
        self,
        session: Session,
        nodes: dict[str, dict[str, dict[str, Any]]],
    ) -> None:
        for label, nodes_by_id in nodes.items():
            rows = list(nodes_by_id.values())
            for batch in self.batches(rows):
                session.run(
                    f"""
                    UNWIND $rows AS row
                    MERGE (node:{label} {{id: row.id}})
                    SET node += row
                    """,
                    {"rows": batch},
                )

    def bulk_merge_relationships(
        self,
        session: Session,
        relationships: list[dict[str, Any]],
    ) -> None:
        deduped = self.dedupe_relationships(relationships)
        grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
        for relationship in deduped:
            key = (
                relationship["source_label"],
                relationship["relationship_type"],
                relationship["target_label"],
            )
            grouped.setdefault(key, []).append(relationship)

        for (source_label, relationship_type, target_label), rows in grouped.items():
            for batch in self.batches(rows):
                if relationship_type == "RETURNS_RESPONSE_SCHEMA":
                    query = f"""
                    UNWIND $rows AS row
                    MATCH (source:{source_label} {{id: row.source_id}})
                    MATCH (target:{target_label} {{id: row.target_id}})
                    MERGE (source)-[relationship:{relationship_type} {{
                        status_code: row.identity_props.status_code
                    }}]->(target)
                    SET relationship += row.props
                    """
                else:
                    query = f"""
                    UNWIND $rows AS row
                    MATCH (source:{source_label} {{id: row.source_id}})
                    MATCH (target:{target_label} {{id: row.target_id}})
                    MERGE (source)-[relationship:{relationship_type}]->(target)
                    SET relationship += row.props
                    """
                session.run(query, {"rows": batch})

    def dedupe_relationships(
        self,
        relationships: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        deduped: dict[tuple[Any, ...], dict[str, Any]] = {}
        for relationship in relationships:
            identity_tuple = tuple(sorted(relationship["identity_props"].items()))
            key = (
                relationship["source_label"],
                relationship["source_id"],
                relationship["relationship_type"],
                relationship["target_label"],
                relationship["target_id"],
                identity_tuple,
            )
            deduped[key] = relationship
        return list(deduped.values())

    def batches(self, rows: list[dict[str, Any]], batch_size: int = 1000) -> list[list[dict[str, Any]]]:
        return [rows[index : index + batch_size] for index in range(0, len(rows), batch_size)]

    def clean_props(self, props: dict[str, Any]) -> dict[str, Any]:
        return {key: value for key, value in props.items() if value is not None}

    def application_json_schema(self, value: Any) -> dict[str, Any] | None:
        if not isinstance(value, dict):
            return None

        content = value.get("content", {})
        if not isinstance(content, dict):
            return None

        media_type = content.get("application/json")
        if not isinstance(media_type, dict):
            return None

        schema = media_type.get("schema")
        return schema if isinstance(schema, dict) else None

    def schema_name_from_schema(self, schema: dict[str, Any] | None) -> str | None:
        if not schema:
            return None

        ref_name = self.schema_name_from_ref(schema.get("$ref"))
        if ref_name:
            return ref_name

        items = schema.get("items")
        if isinstance(items, dict):
            return self.schema_name_from_ref(items.get("$ref"))

        return None

    def schema_name_from_ref(self, schema_ref: Any) -> str | None:
        if not isinstance(schema_ref, str):
            return None
        prefix = "#/components/schemas/"
        if not schema_ref.startswith(prefix):
            return None
        return schema_ref.removeprefix(prefix)

    def extract_schema_refs(self, value: Any) -> list[str]:
        refs: set[str] = set()
        self.collect_schema_refs(value, refs)
        return sorted(refs)

    def collect_schema_refs(self, value: Any, refs: set[str]) -> None:
        if isinstance(value, dict):
            ref = value.get("$ref")
            if isinstance(ref, str):
                refs.add(ref)
            for child in value.values():
                self.collect_schema_refs(child, refs)
        elif isinstance(value, list):
            for child in value:
                self.collect_schema_refs(child, refs)

    def infer_product(self, operation: dict[str, Any], path: str) -> str:
        tags = self.as_string_list(operation.get("tags"))
        if tags:
            return tags[0]
        return self.first_meaningful_path_segment(path) or "unknown"

    def infer_resource(self, path: str) -> str:
        return self.first_meaningful_path_segment(path) or "unknown"

    def first_meaningful_path_segment(self, path: str) -> str | None:
        for segment in self.path_segments(path):
            if segment not in {"v1", "v2"} and not self.is_path_param(segment):
                return segment
        return None

    def resource_path_prefix(self, path: str) -> str:
        segments = self.path_segments(path)
        prefix_segments = []
        for segment in segments:
            if self.is_path_param(segment):
                break
            prefix_segments.append(segment)
        return "/" + "/".join(prefix_segments)

    def path_segments(self, path: str) -> list[str]:
        return [segment for segment in path.strip("/").split("/") if segment]

    def is_path_param(self, segment: str) -> bool:
        return segment.startswith("{") and segment.endswith("}")

    def infer_schema_type(self, schema: Any) -> str | None:
        if not isinstance(schema, dict):
            return None
        if "$ref" in schema:
            return "ref"
        if "type" in schema:
            return schema["type"]
        if "oneOf" in schema:
            return "oneOf"
        if "anyOf" in schema:
            return "anyOf"
        if "allOf" in schema:
            return "allOf"
        return None

    def api_version(self, spec: dict[str, Any]) -> str | None:
        info = spec.get("info", {})
        return info.get("version") if isinstance(info, dict) else None

    def provider_id(self) -> str:
        return f"provider:{self.provider}"

    def product_id(self, product_name: str) -> str:
        return self.stable_id("product", self.provider, product_name)

    def resource_id(self, resource_name: str) -> str:
        return self.stable_id("resource", self.provider, resource_name)

    def operation_id(self, operation_key: str) -> str:
        return self.stable_id("operation", self.provider, operation_key)

    def schema_id(self, schema_name: str) -> str:
        return self.stable_id("schema", self.provider, schema_name)

    def field_id(self, schema_name: str, field_path: str) -> str:
        return self.stable_id("field", self.provider, schema_name, field_path)

    def auth_scheme_id(self, scheme_name: str) -> str:
        return self.stable_id("auth_scheme", self.provider, scheme_name)

    def parameter_id(self, operation_id: str, name: str, location: str) -> str:
        return self.stable_id("parameter", operation_id, location, name)

    def stable_id(self, prefix: str, *parts: str) -> str:
        raw_value = ":".join(parts)
        digest = hashlib.sha256(raw_value.encode("utf-8")).hexdigest()[:20]
        return f"{prefix}:{digest}"

    def clean_text(self, value: Any) -> str | None:
        if value is None:
            return None
        return " ".join(str(value).replace("\n", " ").split())

    def as_list(self, value: Any) -> list[Any]:
        return value if isinstance(value, list) else []

    def as_string_list(self, value: Any) -> list[str]:
        return [item for item in self.as_list(value) if isinstance(item, str)]


def build_api_knowledge_graph(spec: dict[str, Any], provider: str, source_url: str) -> None:
    builder = Neo4jGraphBuilder(provider=provider, source_url=source_url)
    try:
        builder.build(spec)
    finally:
        builder.close()
