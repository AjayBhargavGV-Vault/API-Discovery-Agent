from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_SRC = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_SRC / "data"
DEFAULT_API_SPECS_PATH = DATA_DIR / "api_specs.json"
DEFAULT_API_PRODUCTS_PATH = DATA_DIR / "api_products.json"
DEFAULT_API_OPERATIONS_PATH = DATA_DIR / "api_operations.json"
DEFAULT_API_SCHEMAS_PATH = DATA_DIR / "api_schemas.json"
DEFAULT_API_SCHEMA_FIELDS_PATH = DATA_DIR / "api_schema_fields.json"
DEFAULT_API_RELATIONSHIPS_PATH = DATA_DIR / "api_relationships.json"


class RelationshipMining:
    def __init__(
        self,
        api_specs_path: Path = DEFAULT_API_SPECS_PATH,
        api_products_path: Path = DEFAULT_API_PRODUCTS_PATH,
        api_operations_path: Path = DEFAULT_API_OPERATIONS_PATH,
        api_schemas_path: Path = DEFAULT_API_SCHEMAS_PATH,
        api_schema_fields_path: Path = DEFAULT_API_SCHEMA_FIELDS_PATH,
        api_relationships_path: Path = DEFAULT_API_RELATIONSHIPS_PATH,
    ) -> None:
        self.api_specs_path = api_specs_path.resolve()
        self.api_products_path = api_products_path.resolve()
        self.api_operations_path = api_operations_path.resolve()
        self.api_schemas_path = api_schemas_path.resolve()
        self.api_schema_fields_path = api_schema_fields_path.resolve()
        self.api_relationships_path = api_relationships_path.resolve()

    def mine(self) -> list[dict[str, Any]]:
        api_spec = self.get_current_api_spec()
        products = self.read_json_array(self.api_products_path)
        operations = self.read_json_array(self.api_operations_path)
        schemas = self.read_json_array(self.api_schemas_path)
        fields = self.read_json_array(self.api_schema_fields_path)

        schema_by_ref = self.build_schema_ref_index(schemas)
        relationships: list[dict[str, Any]] = []

        relationships.extend(self.mine_product_operation_relationships(api_spec, products, operations))
        relationships.extend(self.mine_operation_schema_relationships(api_spec, operations, schema_by_ref))
        relationships.extend(self.mine_operation_inline_request_schema_relationships(api_spec, operations))
        relationships.extend(self.mine_schema_field_relationships(api_spec, fields))
        relationships.extend(self.mine_field_schema_relationships(api_spec, fields, schema_by_ref))
        relationships.extend(self.mine_operation_auth_relationships(api_spec, operations))
        relationships.extend(self.mine_resource_lifecycle_relationships(api_spec, operations))

        deduped_relationships = self.dedupe_relationships(relationships)
        deduped_relationships.sort(
            key=lambda item: (
                item["source_type"],
                item["source_id"],
                item["relationship_type"],
                item["target_type"],
                item["target_id"],
            )
        )
        self.write_json_array(self.api_relationships_path, deduped_relationships)
        return deduped_relationships

    def utc_now_iso(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def get_current_api_spec(self) -> dict[str, Any]:
        records = self.read_json_array(self.api_specs_path)
        if not records:
            raise ValueError(
                f"No api_specs record found in {self.api_specs_path}. Run raw_spec_intake.py first."
            )
        return records[-1]

    def read_json_array(self, path: Path) -> list[dict[str, Any]]:
        if not path.exists():
            return []

        with path.open("r", encoding="utf-8") as source:
            records = json.load(source)

        if not isinstance(records, list):
            raise ValueError(f"{path} must contain a JSON array.")

        return records

    def build_schema_ref_index(self, schemas: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
        return {
            f"#/components/schemas/{schema['schema_name']}": schema
            for schema in schemas
            if schema.get("schema_name") and schema.get("id")
        }

    def mine_product_operation_relationships(
        self,
        api_spec: dict[str, Any],
        products: list[dict[str, Any]],
        operations: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        product_ids = {product["id"] for product in products if product.get("id")}
        relationships = []

        for operation in operations:
            product_id = operation.get("product_id")
            if product_id not in product_ids:
                continue

            relationships.append(
                self.build_relationship(
                    spec_id=api_spec["id"],
                    source_type="product",
                    source_id=product_id,
                    relationship_type="HAS_OPERATION",
                    target_type="operation",
                    target_id=operation["id"],
                    confidence=1.0,
                    evidence={
                        "product_key": operation.get("product_key"),
                        "method": operation.get("method"),
                        "path": operation.get("path"),
                    },
                )
            )

        return relationships

    def mine_operation_schema_relationships(
        self,
        api_spec: dict[str, Any],
        operations: list[dict[str, Any]],
        schema_by_ref: dict[str, dict[str, Any]],
    ) -> list[dict[str, Any]]:
        relationships = []

        for operation in operations:
            for schema_ref in operation.get("request_schema_refs", []):
                schema = schema_by_ref.get(schema_ref)
                if schema:
                    relationships.append(
                        self.build_relationship(
                            spec_id=api_spec["id"],
                            source_type="operation",
                            source_id=operation["id"],
                            relationship_type="USES_REQUEST_SCHEMA",
                            target_type="schema",
                            target_id=schema["id"],
                            confidence=1.0,
                            evidence=self.operation_schema_evidence(operation, schema_ref),
                        )
                    )

            for schema_ref in operation.get("response_schema_refs", []):
                schema = schema_by_ref.get(schema_ref)
                if schema:
                    relationships.append(
                        self.build_relationship(
                            spec_id=api_spec["id"],
                            source_type="operation",
                            source_id=operation["id"],
                            relationship_type="RETURNS_RESPONSE_SCHEMA",
                            target_type="schema",
                            target_id=schema["id"],
                            confidence=1.0,
                            evidence=self.operation_schema_evidence(operation, schema_ref),
                        )
                    )

        return relationships

    def mine_operation_inline_request_schema_relationships(
        self,
        api_spec: dict[str, Any],
        operations: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        relationships = []

        for operation in operations:
            request_body = operation.get("request_body")
            if not isinstance(request_body, dict):
                continue
            if request_body.get("schema_refs"):
                continue

            content_types = request_body.get("content_types", [])
            if not content_types:
                continue

            relationships.append(
                self.build_relationship(
                    spec_id=api_spec["id"],
                    source_type="operation",
                    source_id=operation["id"],
                    relationship_type="USES_REQUEST_SCHEMA",
                    target_type="inline_request_schema",
                    target_id=f"inline_request_schema:{operation['id']}",
                    confidence=0.8,
                    evidence={
                        "operation_id": operation.get("operation_id"),
                        "method": operation.get("method"),
                        "path": operation.get("path"),
                        "content_types": content_types,
                        "required_fields": request_body.get("required_fields", []),
                    },
                )
            )

        return relationships

    def mine_schema_field_relationships(
        self,
        api_spec: dict[str, Any],
        fields: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        return [
            self.build_relationship(
                spec_id=api_spec["id"],
                source_type="schema",
                source_id=field["schema_id"],
                relationship_type="HAS_FIELD",
                target_type="field",
                target_id=field["id"],
                confidence=1.0,
                evidence={
                    "schema_name": field.get("schema_name"),
                    "field_path": field.get("field_path"),
                    "field_type": field.get("field_type"),
                },
            )
            for field in fields
            if field.get("schema_id") and field.get("id")
        ]

    def mine_field_schema_relationships(
        self,
        api_spec: dict[str, Any],
        fields: list[dict[str, Any]],
        schema_by_ref: dict[str, dict[str, Any]],
    ) -> list[dict[str, Any]]:
        relationships = []

        for field in fields:
            for schema_ref in field.get("schema_refs", []):
                schema = schema_by_ref.get(schema_ref)
                if not schema:
                    continue

                relationships.append(
                    self.build_relationship(
                        spec_id=api_spec["id"],
                        source_type="field",
                        source_id=field["id"],
                        relationship_type="REFERENCES_SCHEMA",
                        target_type="schema",
                        target_id=schema["id"],
                        confidence=1.0,
                        evidence={
                            "schema_name": field.get("schema_name"),
                            "field_path": field.get("field_path"),
                            "schema_ref": schema_ref,
                        },
                    )
                )

        return relationships

    def mine_operation_auth_relationships(
        self,
        api_spec: dict[str, Any],
        operations: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        relationships = []

        for operation in operations:
            for scheme_name in operation.get("security_scheme_names", []):
                relationships.append(
                    self.build_relationship(
                        spec_id=api_spec["id"],
                        source_type="operation",
                        source_id=operation["id"],
                        relationship_type="REQUIRES_AUTH",
                        target_type="auth_scheme",
                        target_id=f"auth_scheme:{scheme_name}",
                        confidence=1.0,
                        evidence={
                            "scheme_name": scheme_name,
                            "method": operation.get("method"),
                            "path": operation.get("path"),
                        },
                    )
                )

        return relationships

    def mine_resource_lifecycle_relationships(
        self,
        api_spec: dict[str, Any],
        operations: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        relationships = []
        lifecycle_types = {"list", "retrieve", "create", "update", "delete", "action", "search"}

        for operation in operations:
            resource_name = operation.get("resource_name")
            operation_type = operation.get("operation_type")
            if not resource_name or operation_type not in lifecycle_types:
                continue

            relationships.append(
                self.build_relationship(
                    spec_id=api_spec["id"],
                    source_type="resource",
                    source_id=self.build_resource_id(api_spec["id"], resource_name),
                    relationship_type="HAS_LIFECYCLE_OPERATION",
                    target_type="operation",
                    target_id=operation["id"],
                    confidence=0.9,
                    evidence={
                        "resource_name": resource_name,
                        "operation_type": operation_type,
                        "action_name": operation.get("action_name"),
                        "method": operation.get("method"),
                        "path": operation.get("path"),
                    },
                )
            )

        return relationships

    def build_resource_id(self, spec_id: str, resource_name: str) -> str:
        raw_id = f"{spec_id}:resource:{resource_name}"
        digest = hashlib.sha256(raw_id.encode("utf-8")).hexdigest()[:16]
        return f"resource:{resource_name}:{digest}"

    def operation_schema_evidence(self, operation: dict[str, Any], schema_ref: str) -> dict[str, Any]:
        return {
            "operation_id": operation.get("operation_id"),
            "method": operation.get("method"),
            "path": operation.get("path"),
            "schema_ref": schema_ref,
        }

    def build_relationship(
        self,
        spec_id: str,
        source_type: str,
        source_id: str,
        relationship_type: str,
        target_type: str,
        target_id: str,
        confidence: float,
        evidence: dict[str, Any],
    ) -> dict[str, Any]:
        relationship_id = self.build_relationship_id(
            spec_id=spec_id,
            source_type=source_type,
            source_id=source_id,
            relationship_type=relationship_type,
            target_type=target_type,
            target_id=target_id,
        )

        return {
            "id": relationship_id,
            "spec_id": spec_id,
            "source_type": source_type,
            "source_id": source_id,
            "relationship_type": relationship_type,
            "target_type": target_type,
            "target_id": target_id,
            "confidence": confidence,
            "evidence": evidence,
            "mined_at": self.utc_now_iso(),
        }

    def build_relationship_id(
        self,
        spec_id: str,
        source_type: str,
        source_id: str,
        relationship_type: str,
        target_type: str,
        target_id: str,
    ) -> str:
        raw_id = f"{spec_id}:{source_type}:{source_id}:{relationship_type}:{target_type}:{target_id}"
        digest = hashlib.sha256(raw_id.encode("utf-8")).hexdigest()[:20]
        return f"api_relationship:{digest}"

    def dedupe_relationships(
        self,
        relationships: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        deduped: dict[str, dict[str, Any]] = {}
        for relationship in relationships:
            deduped[relationship["id"]] = relationship
        return list(deduped.values())

    def write_json_array(self, path: Path, records: list[dict[str, Any]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as target:
            json.dump(records, target, indent=2)
            target.write("\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mine graph relationships from API catalog artifacts.")
    parser.add_argument("--api-specs", type=Path, default=DEFAULT_API_SPECS_PATH)
    parser.add_argument("--api-products", type=Path, default=DEFAULT_API_PRODUCTS_PATH)
    parser.add_argument("--api-operations", type=Path, default=DEFAULT_API_OPERATIONS_PATH)
    parser.add_argument("--api-schemas", type=Path, default=DEFAULT_API_SCHEMAS_PATH)
    parser.add_argument("--api-schema-fields", type=Path, default=DEFAULT_API_SCHEMA_FIELDS_PATH)
    parser.add_argument("--api-relationships", type=Path, default=DEFAULT_API_RELATIONSHIPS_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    mining = RelationshipMining(
        api_specs_path=args.api_specs,
        api_products_path=args.api_products,
        api_operations_path=args.api_operations,
        api_schemas_path=args.api_schemas,
        api_schema_fields_path=args.api_schema_fields,
        api_relationships_path=args.api_relationships,
    )
    relationships = mining.mine()
    print(json.dumps({"relationship_count": len(relationships)}, indent=2))


if __name__ == "__main__":
    main()
