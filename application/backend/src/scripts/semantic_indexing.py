from __future__ import annotations

import html
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_SRC = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_SRC / "data"
if str(PROJECT_SRC) not in sys.path:
    sys.path.append(str(PROJECT_SRC))

from models.api_search_document import ApiSearchDocument

DEFAULT_API_PRODUCTS_PATH = DATA_DIR / "api_products.json"
DEFAULT_API_OPERATIONS_PATH = DATA_DIR / "api_operations.json"
DEFAULT_API_SCHEMAS_PATH = DATA_DIR / "api_schemas.json"
DEFAULT_API_SCHEMA_FIELDS_PATH = DATA_DIR / "api_schema_fields.json"
DEFAULT_API_SEARCH_DOCUMENTS_PATH = DATA_DIR / "api_search_documents.json"
DEFAULT_API_SEARCH_DOCUMENTS_JSONL_PATH = DATA_DIR / "api_search_documents.jsonl"


class SemanticIndexing:
    def __init__(
        self,
        api_products_path: Path = DEFAULT_API_PRODUCTS_PATH,
        api_operations_path: Path = DEFAULT_API_OPERATIONS_PATH,
        api_schemas_path: Path = DEFAULT_API_SCHEMAS_PATH,
        api_schema_fields_path: Path = DEFAULT_API_SCHEMA_FIELDS_PATH,
        api_search_documents_path: Path = DEFAULT_API_SEARCH_DOCUMENTS_PATH,
        api_search_documents_jsonl_path: Path = DEFAULT_API_SEARCH_DOCUMENTS_JSONL_PATH,
        embedding_model: str | None = None,
    ) -> None:
        self.api_products_path = api_products_path.resolve()
        self.api_operations_path = api_operations_path.resolve()
        self.api_schemas_path = api_schemas_path.resolve()
        self.api_schema_fields_path = api_schema_fields_path.resolve()
        self.api_search_documents_path = api_search_documents_path.resolve()
        self.api_search_documents_jsonl_path = api_search_documents_jsonl_path.resolve()
        self.embedding_model = embedding_model

    def build(self) -> list[dict[str, Any]]:
        products = self.read_json_array(self.api_products_path)
        operations = self.read_json_array(self.api_operations_path)
        schemas = self.read_json_array(self.api_schemas_path)
        fields = self.read_json_array(self.api_schema_fields_path)
        fields_by_schema = self.group_fields_by_schema(fields)
        fields_by_schema_name = self.group_fields_by_schema_name(fields)

        documents = [
            *self.build_product_documents(products),
            *self.build_operation_documents(operations, fields_by_schema_name),
            *self.build_schema_documents(schemas, fields_by_schema),
        ]
        documents.sort(key=lambda item: (item["entity_type"], item["title"], item["id"]))
        self.write_json_array(self.api_search_documents_path, documents)
        self.write_jsonl(self.api_search_documents_jsonl_path, documents)
        return documents

    def utc_now_iso(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def read_json_array(self, path: Path) -> list[dict[str, Any]]:
        if not path.exists():
            return []

        with path.open("r", encoding="utf-8") as source:
            records = json.load(source)

        if not isinstance(records, list):
            raise ValueError(f"{path} must contain a JSON array.")

        return records

    def build_product_documents(self, products: list[dict[str, Any]]) -> list[dict[str, Any]]:
        documents = []
        for product in products:
            summary = self.product_summary(product)
            body_lines = [
                f"{product.get('display_name')} API product.",
                f"Product key: {product.get('product_key')}",
                f"Display name: {product.get('display_name')}",
                f"Business domain: {product.get('business_domain')}",
                f"API version: {self.join_values(product.get('api_versions'))}",
                f"Path prefixes: {self.join_values(product.get('path_prefixes'))}",
                f"Operation count: {product.get('operation_count')}",
                f"Endpoint count: {product.get('operation_count')}",
                f"Schema count: {product.get('schema_count')}",
                f"Path count: {product.get('path_count')}",
                f"Summary: {summary}",
            ]
            if product.get("is_test_product"):
                body_lines.append("Test-only API product.")

            documents.append(
                self.build_document(
                    spec_id=product.get("spec_id"),
                    provider=product.get("provider"),
                    spec_version=self.spec_version_from_spec_id(product.get("spec_id")),
                    entity_type="product",
                    entity_id=product["id"],
                    title=product.get("display_name") or product.get("product_key"),
                    body="\n".join(self.non_empty(body_lines)),
                    keywords=self.unique_values(
                        [
                            product.get("product_key"),
                            product.get("display_name"),
                            product.get("business_domain"),
                            *self.as_list(product.get("tags")),
                            *self.as_list(product.get("path_prefixes")),
                        ]
                    ),
                    metadata={
                        "business_domain": product.get("business_domain"),
                        "product_key": product.get("product_key"),
                        "api_versions": product.get("api_versions"),
                        "path_prefixes": product.get("path_prefixes"),
                        "operation_count": product.get("operation_count"),
                        "schema_count": product.get("schema_count"),
                    },
                    business_domain=product.get("business_domain"),
                    product_key=product.get("product_key"),
                    method=None,
                    path=None,
                    neo4j_node_id=ApiSearchDocument.neo4j_product_id(
                        product.get("provider"),
                        product.get("product_key"),
                    ),
                )
            )

        return documents

    def build_operation_documents(
        self,
        operations: list[dict[str, Any]],
        fields_by_schema_name: dict[str, list[dict[str, Any]]],
    ) -> list[dict[str, Any]]:
        documents = []
        for operation in operations:
            request_fields = self.extract_request_fields(operation)
            request_schemas = self.schema_names_from_refs(operation.get("request_schema_refs", []))
            response_schemas = self.schema_names_from_refs(operation.get("response_schema_refs", []))
            response_fields = self.response_fields(response_schemas, fields_by_schema_name)
            parameter_names = [
                parameter.get("name")
                for parameter in self.as_list(operation.get("parameters"))
                if isinstance(parameter, dict)
            ]
            parameter_summaries = self.parameter_summaries(operation)
            auth_requirements = self.auth_requirements(operation)
            lifecycle_labels = self.lifecycle_labels(operation)
            title = self.operation_title(operation)
            body_lines = [
                f"{self.display_name(operation.get('product_key'))} API.",
                f"Product name: {self.display_name(operation.get('product_key'))}",
                f"Business domain: {operation.get('business_domain')}",
                f"{operation.get('method')} {operation.get('path')}",
                f"HTTP method: {operation.get('method')}",
                f"Path: {operation.get('path')}",
                f"Operation ID: {operation.get('operation_id')}",
                f"Summary: {operation.get('summary')}",
                f"Description: {operation.get('description_text')}",
                f"Operation type: {operation.get('operation_type')}",
                f"Parameters: {self.join_values(parameter_summaries)}",
                f"Request schema: {self.join_values(request_schemas)}",
                f"Request fields: {self.join_values(request_fields)}",
                f"Response schema: {self.join_values(response_schemas)}",
                f"Response fields: {self.join_values(response_fields[:80])}",
                f"Auth requirements: {self.join_values(auth_requirements)}",
                f"Lifecycle/action labels: {self.join_values(lifecycle_labels)}",
            ]

            documents.append(
                self.build_document(
                    spec_id=operation.get("spec_id"),
                    provider=operation.get("provider"),
                    spec_version=self.spec_version_from_spec_id(operation.get("spec_id")),
                    entity_type="operation",
                    entity_id=operation["id"],
                    title=title,
                    body="\n".join(self.non_empty(body_lines)),
                    keywords=self.unique_values(
                        [
                            operation.get("operation_id"),
                            operation.get("method"),
                            operation.get("path"),
                            operation.get("product_key"),
                            operation.get("business_domain"),
                            operation.get("operation_type"),
                            operation.get("action_name"),
                            *request_fields,
                            *parameter_names,
                            *parameter_summaries,
                            *request_schemas,
                            *response_schemas,
                            *response_fields,
                            *auth_requirements,
                            *lifecycle_labels,
                        ]
                    ),
                    metadata={
                        "business_domain": operation.get("business_domain"),
                        "product_key": operation.get("product_key"),
                        "method": operation.get("method"),
                        "path": operation.get("path"),
                        "operation_id": operation.get("operation_id"),
                        "operation_type": operation.get("operation_type"),
                        "resource_name": operation.get("resource_name"),
                        "action_name": operation.get("action_name"),
                        "request_schema_names": request_schemas,
                        "response_schema_names": response_schemas,
                        "auth_requirements": auth_requirements,
                    },
                    business_domain=operation.get("business_domain"),
                    product_key=operation.get("product_key"),
                    method=operation.get("method"),
                    path=operation.get("path"),
                    neo4j_node_id=ApiSearchDocument.neo4j_operation_id(
                        operation.get("provider"),
                        operation.get("operation_id")
                        or f"{operation.get('method')}:{operation.get('path')}",
                    ),
                )
            )

        return documents

    def build_schema_documents(
        self,
        schemas: list[dict[str, Any]],
        fields_by_schema: dict[str, list[dict[str, Any]]],
    ) -> list[dict[str, Any]]:
        documents = []
        for schema in schemas:
            fields = fields_by_schema.get(schema["id"], [])
            field_names = [field.get("field_path") for field in fields]
            field_summaries = self.field_summaries(fields)
            enum_values = self.collect_schema_enum_values(schema, fields)
            sensitive_hints = self.sensitive_field_hints(fields)
            referenced_schemas = self.schema_names_from_refs(schema.get("referenced_schema_refs", []))
            body_lines = [
                f"{schema.get('title') or self.display_name(schema.get('schema_name'))} schema.",
                f"Schema name: {schema.get('schema_name')}",
                f"Schema title: {schema.get('title')}",
                f"Schema description: {schema.get('description_text')}",
                f"Schema type: {schema.get('schema_type')}",
                f"Required fields: {self.join_values(schema.get('required_fields'))}",
                f"Properties: {self.join_values(field_summaries[:120])}",
                f"Fields: {self.join_values(field_names[:120])}",
                f"Enum values: {self.join_values(enum_values)}",
                f"Referenced schemas: {self.join_values(referenced_schemas)}",
                f"Sensitive field hints: {self.join_values(sensitive_hints)}",
            ]

            documents.append(
                self.build_document(
                    spec_id=schema.get("spec_id"),
                    provider=schema.get("provider"),
                    spec_version=self.spec_version_from_spec_id(schema.get("spec_id")),
                    entity_type="schema",
                    entity_id=schema["id"],
                    title=schema.get("title") or schema.get("schema_name"),
                    body="\n".join(self.non_empty(body_lines)),
                    keywords=self.unique_values(
                        [
                            schema.get("schema_name"),
                            schema.get("title"),
                            schema.get("schema_type"),
                            *self.as_list(schema.get("required_fields")),
                            *field_names,
                            *field_summaries,
                            *enum_values,
                            *referenced_schemas,
                            *sensitive_hints,
                        ]
                    ),
                    metadata={
                        "schema_name": schema.get("schema_name"),
                        "schema_type": schema.get("schema_type"),
                        "property_count": schema.get("property_count"),
                        "is_error_schema": schema.get("is_error_schema"),
                        "referenced_schema_names": referenced_schemas,
                        "sensitive_field_hints": sensitive_hints,
                    },
                    business_domain=None,
                    product_key=None,
                    method=None,
                    path=None,
                    neo4j_node_id=ApiSearchDocument.neo4j_schema_id(
                        schema.get("provider"),
                        schema.get("schema_name"),
                    ),
                )
            )

        return documents

    def build_document(
        self,
        spec_id: str | None,
        provider: str | None,
        spec_version: str | None,
        entity_type: str,
        entity_id: str,
        title: str | None,
        body: str,
        keywords: list[str],
        metadata: dict[str, Any],
        business_domain: str | None,
        product_key: str | None,
        method: str | None,
        path: str | None,
        neo4j_node_id: str,
    ) -> dict[str, Any]:
        clean_body = self.clean_text(body) or ""
        document = ApiSearchDocument.create(
            spec_id=spec_id or "",
            entity_type=entity_type,
            entity_id=entity_id,
            title=title or entity_id,
            body=clean_body,
            keywords=keywords,
            provider=provider or "",
            spec_version=spec_version or "",
            business_domain=business_domain,
            product_key=product_key,
            method=method,
            path=path,
            neo4j_node_id=neo4j_node_id,
            embedding_model=self.embedding_model,
            embedding_vector=None,
        ).to_dict()
        document["metadata"] = metadata
        document["embedding_status"] = "pending"
        return document

    def group_fields_by_schema(
        self,
        fields: list[dict[str, Any]],
    ) -> dict[str, list[dict[str, Any]]]:
        grouped: dict[str, list[dict[str, Any]]] = {}
        for field in fields:
            schema_id = field.get("schema_id")
            if schema_id:
                grouped.setdefault(schema_id, []).append(field)
        return grouped

    def group_fields_by_schema_name(
        self,
        fields: list[dict[str, Any]],
    ) -> dict[str, list[dict[str, Any]]]:
        grouped: dict[str, list[dict[str, Any]]] = {}
        for field in fields:
            schema_name = field.get("schema_name")
            if schema_name:
                grouped.setdefault(schema_name, []).append(field)
        return grouped

    def extract_request_fields(self, operation: dict[str, Any]) -> list[str]:
        request_body = operation.get("request_body", {})
        required_fields = []
        if isinstance(request_body, dict):
            required_fields = self.as_list(request_body.get("required_fields"))

        parameter_fields = [
            parameter.get("name")
            for parameter in self.as_list(operation.get("parameters"))
            if isinstance(parameter, dict)
            and parameter.get("location") in {"query", "path"}
            and parameter.get("name") != "expand"
        ]

        return self.unique_values([*required_fields, *parameter_fields])

    def parameter_summaries(self, operation: dict[str, Any]) -> list[str]:
        summaries = []
        for parameter in self.as_list(operation.get("parameters")):
            if not isinstance(parameter, dict):
                continue
            name = parameter.get("name")
            location = parameter.get("location")
            required = "required" if parameter.get("required") else "optional"
            schema_type = parameter.get("schema_type")
            if name:
                summaries.append(
                    " ".join(
                        self.non_empty(
                            [
                                name,
                                f"({location})" if location else None,
                                required,
                                schema_type,
                            ]
                        )
                    )
                )
        return self.unique_values(summaries)

    def auth_requirements(self, operation: dict[str, Any]) -> list[str]:
        if not operation.get("auth_required"):
            return ["none"]
        return self.unique_values(self.as_list(operation.get("security_scheme_names")))

    def lifecycle_labels(self, operation: dict[str, Any]) -> list[str]:
        return self.unique_values(
            [
                operation.get("operation_type"),
                operation.get("resource_name"),
                operation.get("action_name"),
            ]
        )

    def response_fields(
        self,
        response_schemas: list[str],
        fields_by_schema_name: dict[str, list[dict[str, Any]]],
    ) -> list[str]:
        fields = []
        for schema_name in response_schemas:
            for field in fields_by_schema_name.get(schema_name, []):
                field_path = field.get("field_path")
                if field_path:
                    fields.append(f"{schema_name}.{field_path}")
        return self.unique_values(fields)

    def field_summaries(self, fields: list[dict[str, Any]]) -> list[str]:
        summaries = []
        for field in fields:
            field_path = field.get("field_path")
            field_type = field.get("field_type")
            required = "required" if field.get("required") else "optional"
            if field_path:
                summaries.append(
                    " ".join(self.non_empty([field_path, field_type, required]))
                )
        return self.unique_values(summaries)

    def collect_schema_enum_values(
        self,
        schema: dict[str, Any],
        fields: list[dict[str, Any]],
    ) -> list[str]:
        enum_values = [*self.as_list(schema.get("enum_values"))]
        for field in fields:
            field_path = field.get("field_path")
            for enum_value in self.as_list(field.get("enum_values")):
                enum_values.append(f"{field_path}={enum_value}")
        return self.unique_values(enum_values)

    def sensitive_field_hints(self, fields: list[dict[str, Any]]) -> list[str]:
        hints = []
        for field in fields:
            classification = field.get("sensitive_classification")
            field_path = field.get("field_path")
            if classification and field_path:
                hints.append(f"{field_path}:{classification}")
        return self.unique_values(hints)

    def product_summary(self, product: dict[str, Any]) -> str:
        display_name = product.get("display_name") or product.get("product_key")
        business_domain = product.get("business_domain")
        operation_count = product.get("operation_count")
        prefixes = self.join_values(product.get("path_prefixes"))
        return (
            f"{display_name} belongs to the {business_domain} domain and contains "
            f"{operation_count} operations under {prefixes}."
        )

    def operation_title(self, operation: dict[str, Any]) -> str:
        summary = operation.get("summary")
        if summary:
            return summary
        return f"{operation.get('method')} {operation.get('path')}"

    def schema_names_from_refs(self, refs: list[Any]) -> list[str]:
        names = []
        prefix = "#/components/schemas/"
        for ref in refs:
            if isinstance(ref, str) and ref.startswith(prefix):
                names.append(ref.removeprefix(prefix))
        return self.unique_values(names)

    def display_name(self, value: Any) -> str | None:
        if not isinstance(value, str):
            return None
        return value.replace("_", " ").title()

    def document_id(self, entity_type: str, entity_id: str) -> str:
        return ApiSearchDocument.document_id(entity_type=entity_type, entity_id=entity_id)

    def spec_version_from_spec_id(self, spec_id: Any) -> str | None:
        if not isinstance(spec_id, str):
            return None
        parts = spec_id.split(":")
        return parts[1] if len(parts) >= 3 else None

    def clean_text(self, value: Any) -> str | None:
        if value is None:
            return None
        text = html.unescape(str(value))
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def join_values(self, values: Any) -> str:
        cleaned_values = self.unique_values(self.as_list(values))
        return ", ".join(cleaned_values) if cleaned_values else "none"

    def non_empty(self, values: list[Any]) -> list[str]:
        return [str(value) for value in values if value not in (None, "", [], {})]

    def as_list(self, value: Any) -> list[Any]:
        return value if isinstance(value, list) else []

    def unique_values(self, values: list[Any]) -> list[str]:
        seen = set()
        unique = []
        for value in values:
            if value in (None, "", [], {}):
                continue
            text = str(value)
            if text not in seen:
                seen.add(text)
                unique.append(text)
        return unique

    def write_json_array(self, path: Path, records: list[dict[str, Any]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as target:
            json.dump(records, target, indent=2)
            target.write("\n")

    def write_jsonl(self, path: Path, records: list[dict[str, Any]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as target:
            for record in records:
                target.write(json.dumps(record, separators=(",", ":")))
                target.write("\n")
