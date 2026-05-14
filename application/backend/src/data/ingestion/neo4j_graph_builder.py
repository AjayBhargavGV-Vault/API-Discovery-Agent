from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from neo4j import GraphDatabase
from neo4j import Session


DATA_DIR = Path(__file__).resolve().parents[1]
PROJECT_SRC = DATA_DIR.parent
BACKEND_DIR = PROJECT_SRC.parent
DEFAULT_SPEC_PATH = DATA_DIR / "raw_specs" / "openapi.spec3.yaml"
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
        self.uri = self.required_env("NEO4J_URI")
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

    def build(self, spec: dict[str, Any]) -> None:
        with self.driver.session(database=self.database) as session:
            self.create_constraints(session)
            provider_id = self.provider_id()
            self.merge_provider(session, spec=spec, provider_id=provider_id)
            self.merge_schemas_and_fields(session, spec=spec)
            self.merge_operations(session, spec=spec, provider_id=provider_id)

    def create_constraints(self, session: Session) -> None:
        for label in self.NODE_LABELS:
            session.run(
                f"CREATE CONSTRAINT {label.lower()}_id_unique IF NOT EXISTS "
                f"FOR (n:{label}) REQUIRE n.id IS UNIQUE"
            )

    def merge_provider(self, session: Session, spec: dict[str, Any], provider_id: str) -> None:
        info = spec.get("info", {})
        session.run(
            """
            MERGE (provider:Provider {id: $id})
            SET provider.name = $name,
                provider.title = $title,
                provider.version = $version,
                provider.source_url = $source_url,
                provider.openapi_version = $openapi_version
            """,
            {
                "id": provider_id,
                "name": self.provider,
                "title": info.get("title"),
                "version": info.get("version"),
                "source_url": self.source_url,
                "openapi_version": spec.get("openapi"),
            },
        )

    def merge_schemas_and_fields(self, session: Session, spec: dict[str, Any]) -> None:
        schemas = spec.get("components", {}).get("schemas", {})
        if not isinstance(schemas, dict):
            return

        for schema_name, schema in schemas.items():
            if not isinstance(schema, dict):
                continue

            schema_id = self.schema_id(schema_name)
            session.run(
                """
                MERGE (schema:Schema {id: $id})
                SET schema.name = $name,
                    schema.type = $type,
                    schema.description = $description,
                    schema.provider = $provider,
                    schema.version = $version,
                    schema.source_url = $source_url
                """,
                {
                    "id": schema_id,
                    "name": schema_name,
                    "type": schema.get("type"),
                    "description": self.clean_text(schema.get("description")),
                    "provider": self.provider,
                    "version": self.api_version(spec),
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
                self.merge_schema_field(
                    session=session,
                    spec=spec,
                    schema_id=schema_id,
                    schema_name=schema_name,
                    field_name=field_name,
                    field_path=field_name,
                    field_schema=field_schema,
                    required=field_name in required_fields,
                )

    def merge_schema_field(
        self,
        session: Session,
        spec: dict[str, Any],
        schema_id: str,
        schema_name: str,
        field_name: str,
        field_path: str,
        field_schema: dict[str, Any],
        required: bool,
    ) -> None:
        field_id = self.field_id(schema_name=schema_name, field_path=field_path)
        schema_refs = self.extract_schema_refs(field_schema)
        session.run(
            """
            MATCH (schema:Schema {id: $schema_id})
            MERGE (field:Field {id: $field_id})
            SET field.name = $field_name,
                field.path = $field_path,
                field.type = $field_type,
                field.description = $description,
                field.required = $required,
                field.enum_values = $enum_values,
                field.provider = $provider,
                field.source_url = $source_url
            MERGE (schema)-[:HAS_FIELD]->(field)
            """,
            {
                "schema_id": schema_id,
                "field_id": field_id,
                "field_name": field_name,
                "field_path": field_path,
                "field_type": self.infer_schema_type(field_schema),
                "description": self.clean_text(field_schema.get("description")),
                "required": required,
                "enum_values": field_schema.get("enum"),
                "provider": self.provider,
                "source_url": self.source_url,
            },
        )

        for schema_ref in schema_refs:
            referenced_schema_name = self.schema_name_from_ref(schema_ref)
            if not referenced_schema_name:
                continue

            session.run(
                """
                MATCH (field:Field {id: $field_id})
                MERGE (referenced:Schema {id: $referenced_schema_id})
                SET referenced.name = $referenced_schema_name,
                    referenced.provider = $provider,
                    referenced.source_url = $source_url
                MERGE (field)-[:REFERENCES_SCHEMA]->(referenced)
                """,
                {
                    "field_id": field_id,
                    "referenced_schema_id": self.schema_id(referenced_schema_name),
                    "referenced_schema_name": referenced_schema_name,
                    "provider": self.provider,
                    "source_url": self.source_url,
                },
            )

    def merge_operations(self, session: Session, spec: dict[str, Any], provider_id: str) -> None:
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

                self.merge_operation(
                    session=session,
                    spec=spec,
                    provider_id=provider_id,
                    path=path,
                    method=method,
                    operation=operation,
                    path_level_parameters=path_level_parameters,
                )

    def merge_operation(
        self,
        session: Session,
        spec: dict[str, Any],
        provider_id: str,
        path: str,
        method: str,
        operation: dict[str, Any],
        path_level_parameters: list[Any],
    ) -> None:
        operation_key = operation.get("operationId") or f"{method.upper()}:{path}"
        operation_id = self.operation_id(operation_key)
        product_name = self.infer_product(operation=operation, path=path)
        resource_name = self.infer_resource(path)
        product_id = self.product_id(product_name)
        resource_id = self.resource_id(resource_name)

        session.run(
            """
            MATCH (provider:Provider {id: $provider_id})
            MERGE (product:Product {id: $product_id})
            SET product.name = $product_name,
                product.provider = $provider,
                product.version = $version,
                product.source_url = $source_url
            MERGE (resource:Resource {id: $resource_id})
            SET resource.name = $resource_name,
                resource.path_prefix = $resource_path_prefix,
                resource.provider = $provider,
                resource.source_url = $source_url
            MERGE (operation:Operation {id: $operation_id})
            SET operation.operation_id = $operation_key,
                operation.method = $method,
                operation.path = $path,
                operation.summary = $summary,
                operation.description = $description,
                operation.provider = $provider,
                operation.version = $version,
                operation.source_url = $source_url
            MERGE (provider)-[:HAS_PRODUCT]->(product)
            MERGE (product)-[:HAS_RESOURCE]->(resource)
            MERGE (product)-[:HAS_OPERATION]->(operation)
            MERGE (resource)-[:HAS_OPERATION]->(operation)
            """,
            {
                "provider_id": provider_id,
                "product_id": product_id,
                "product_name": product_name,
                "resource_id": resource_id,
                "resource_name": resource_name,
                "resource_path_prefix": self.resource_path_prefix(path),
                "operation_id": operation_id,
                "operation_key": operation_key,
                "method": method.upper(),
                "path": path,
                "summary": operation.get("summary"),
                "description": self.clean_text(operation.get("description")),
                "provider": self.provider,
                "version": self.api_version(spec),
                "source_url": self.source_url,
            },
        )

        self.merge_operation_parameters(
            session=session,
            operation_id=operation_id,
            parameters=[*path_level_parameters, *self.as_list(operation.get("parameters"))],
        )
        self.merge_operation_request_schema(session=session, operation_id=operation_id, operation=operation)
        self.merge_operation_response_schemas(session=session, operation_id=operation_id, operation=operation)
        self.merge_operation_auth_schemes(
            session=session,
            operation_id=operation_id,
            operation=operation,
            global_security=spec.get("security"),
        )

    def merge_operation_parameters(
        self,
        session: Session,
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
            schema = parameter.get("schema", {})
            session.run(
                """
                MATCH (operation:Operation {id: $operation_id})
                MERGE (parameter:Parameter {id: $parameter_id})
                SET parameter.name = $name,
                    parameter.location = $location,
                    parameter.required = $required,
                    parameter.type = $type,
                    parameter.description = $description,
                    parameter.provider = $provider,
                    parameter.source_url = $source_url
                MERGE (operation)-[:HAS_PARAMETER]->(parameter)
                """,
                {
                    "operation_id": operation_id,
                    "parameter_id": parameter_id,
                    "name": parameter_name,
                    "location": parameter_location,
                    "required": bool(parameter.get("required", False)),
                    "type": self.infer_schema_type(schema),
                    "description": self.clean_text(parameter.get("description")),
                    "provider": self.provider,
                    "source_url": self.source_url,
                },
            )

    def merge_operation_request_schema(
        self,
        session: Session,
        operation_id: str,
        operation: dict[str, Any],
    ) -> None:
        schema = self.application_json_schema(operation.get("requestBody"))
        schema_name = self.schema_name_from_schema(schema)
        if not schema_name:
            return

        session.run(
            """
            MATCH (operation:Operation {id: $operation_id})
            MERGE (schema:Schema {id: $schema_id})
            SET schema.name = $schema_name,
                schema.provider = $provider,
                schema.source_url = $source_url
            MERGE (operation)-[:USES_REQUEST_SCHEMA]->(schema)
            """,
            {
                "operation_id": operation_id,
                "schema_id": self.schema_id(schema_name),
                "schema_name": schema_name,
                "provider": self.provider,
                "source_url": self.source_url,
            },
        )

    def merge_operation_response_schemas(
        self,
        session: Session,
        operation_id: str,
        operation: dict[str, Any],
    ) -> None:
        responses = operation.get("responses", {})
        if not isinstance(responses, dict):
            return

        for status_code, response in responses.items():
            schema = self.application_json_schema(response)
            schema_name = self.schema_name_from_schema(schema)
            if not schema_name:
                continue

            session.run(
                """
                MATCH (operation:Operation {id: $operation_id})
                MERGE (schema:Schema {id: $schema_id})
                SET schema.name = $schema_name,
                    schema.provider = $provider,
                    schema.source_url = $source_url
                MERGE (operation)-[relationship:RETURNS_RESPONSE_SCHEMA {status_code: $status_code}]->(schema)
                """,
                {
                    "operation_id": operation_id,
                    "schema_id": self.schema_id(schema_name),
                    "schema_name": schema_name,
                    "status_code": str(status_code),
                    "provider": self.provider,
                    "source_url": self.source_url,
                },
            )

    def merge_operation_auth_schemes(
        self,
        session: Session,
        operation_id: str,
        operation: dict[str, Any],
        global_security: Any,
    ) -> None:
        security = operation["security"] if "security" in operation else global_security
        for requirement in self.as_list(security):
            if not isinstance(requirement, dict):
                continue

            for scheme_name in requirement.keys():
                session.run(
                    """
                    MATCH (operation:Operation {id: $operation_id})
                    MERGE (auth:AuthScheme {id: $auth_id})
                    SET auth.name = $scheme_name,
                        auth.provider = $provider,
                        auth.source_url = $source_url
                    MERGE (operation)-[:REQUIRES_AUTH]->(auth)
                    """,
                    {
                        "operation_id": operation_id,
                        "auth_id": self.auth_scheme_id(scheme_name),
                        "scheme_name": scheme_name,
                        "provider": self.provider,
                        "source_url": self.source_url,
                    },
                )

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a Neo4j API knowledge graph from OpenAPI.")
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC_PATH)
    parser.add_argument("--provider", default="stripe")
    parser.add_argument("--source-url", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with args.spec.open("r", encoding="utf-8") as source:
        spec = yaml.safe_load(source)

    raw_spec_url = args.source_url or str(args.spec.resolve())
    build_api_knowledge_graph(spec, provider=args.provider, source_url=raw_spec_url)


if __name__ == "__main__":
    main()
