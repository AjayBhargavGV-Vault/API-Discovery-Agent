from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


PROJECT_SRC = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_SRC / "data"
DEFAULT_RAW_SPECS_DIR = DATA_DIR / "raw_specs"
DEFAULT_SPEC_PATH = DEFAULT_RAW_SPECS_DIR / "openapi.spec3.yaml"
DEFAULT_API_SPECS_PATH = DATA_DIR / "api_specs.json"
DEFAULT_API_SCHEMAS_PATH = DATA_DIR / "api_schemas.json"
DEFAULT_API_SCHEMA_FIELDS_PATH = DATA_DIR / "api_schema_fields.json"


class SchemaExtraction:
    COMPOSITION_KEYS = ("oneOf", "anyOf", "allOf")

    def __init__(
        self,
        spec_path: Path = DEFAULT_SPEC_PATH,
        api_specs_path: Path = DEFAULT_API_SPECS_PATH,
        api_schemas_path: Path = DEFAULT_API_SCHEMAS_PATH,
        api_schema_fields_path: Path = DEFAULT_API_SCHEMA_FIELDS_PATH,
    ) -> None:
        self.spec_path = spec_path.resolve()
        self.api_specs_path = api_specs_path.resolve()
        self.api_schemas_path = api_schemas_path.resolve()
        self.api_schema_fields_path = api_schema_fields_path.resolve()

    def extract(self) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        spec = self.load_openapi_spec()
        api_spec = self.get_current_api_spec()
        component_schemas = spec["components"]["schemas"]

        schema_rows: list[dict[str, Any]] = []
        field_rows: list[dict[str, Any]] = []

        for schema_name, schema in sorted(component_schemas.items()):
            if not isinstance(schema, dict):
                continue

            schema_row = self.build_schema_record(
                api_spec=api_spec,
                schema_name=schema_name,
                schema=schema,
            )
            schema_rows.append(schema_row)
            field_rows.extend(
                self.build_field_records(
                    api_spec=api_spec,
                    schema_id=schema_row["id"],
                    schema_name=schema_name,
                    schema=schema,
                )
            )

        self.write_json_array(self.api_schemas_path, schema_rows)
        self.write_json_array(self.api_schema_fields_path, field_rows)
        return schema_rows, field_rows

    def utc_now_iso(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def load_openapi_spec(self) -> dict[str, Any]:
        with self.spec_path.open("r", encoding="utf-8") as source:
            spec = yaml.safe_load(source)

        if not isinstance(spec, dict):
            raise ValueError(f"{self.spec_path} did not parse as an OpenAPI object.")

        schemas = spec.get("components", {}).get("schemas")
        if not isinstance(schemas, dict):
            raise ValueError(f"{self.spec_path} is missing components.schemas.")

        return spec

    def get_current_api_spec(self) -> dict[str, Any]:
        records = self.read_json_array(self.api_specs_path)
        matching_records = [
            record
            for record in records
            if Path(record.get("raw_spec_path", "")).resolve() == self.spec_path
        ]

        if matching_records:
            return matching_records[-1]

        if records:
            return records[-1]

        raise ValueError(
            f"No api_specs record found in {self.api_specs_path}. Run raw_spec_intake.py first."
        )

    def read_json_array(self, path: Path) -> list[dict[str, Any]]:
        if not path.exists():
            return []

        with path.open("r", encoding="utf-8") as source:
            records = json.load(source)

        if not isinstance(records, list):
            raise ValueError(f"{path} must contain a JSON array.")

        return records

    def build_schema_record(
        self,
        api_spec: dict[str, Any],
        schema_name: str,
        schema: dict[str, Any],
    ) -> dict[str, Any]:
        composition = self.extract_composition(schema)
        schema_refs = self.extract_schema_refs(schema)
        enum_values = self.extract_enum_values(schema)
        properties = schema.get("properties", {})

        return {
            "id": self.build_schema_id(api_spec["id"], schema_name),
            "spec_id": api_spec["id"],
            "provider": api_spec.get("provider"),
            "schema_name": schema_name,
            "title": schema.get("title"),
            "description_text": self.clean_text(schema.get("description")),
            "schema_type": schema.get("type"),
            "format": schema.get("format"),
            "required_fields": self.as_string_list(schema.get("required")),
            "enum_values": enum_values,
            "composition_type": composition["composition_type"],
            "composition_refs": composition["composition_refs"],
            "referenced_schema_refs": schema_refs,
            "property_count": len(properties) if isinstance(properties, dict) else 0,
            "is_error_schema": schema_name == "error" or schema_name.startswith("error_"),
            "nullable": bool(schema.get("nullable", False)),
            "raw_schema_hash": self.hash_json(schema),
            "extracted_at": self.utc_now_iso(),
        }

    def build_field_records(
        self,
        api_spec: dict[str, Any],
        schema_id: str,
        schema_name: str,
        schema: dict[str, Any],
    ) -> list[dict[str, Any]]:
        fields: list[dict[str, Any]] = []
        root_required = set(self.as_string_list(schema.get("required")))
        properties = schema.get("properties", {})

        if isinstance(properties, dict):
            for field_name, field_schema in sorted(properties.items()):
                self.collect_field_records(
                    fields=fields,
                    api_spec=api_spec,
                    schema_id=schema_id,
                    schema_name=schema_name,
                    field_name=field_name,
                    field_path=field_name,
                    field_schema=field_schema,
                    required=field_name in root_required,
                )

        for composition_key in self.COMPOSITION_KEYS:
            for index, child_schema in enumerate(self.as_list(schema.get(composition_key))):
                self.collect_field_records(
                    fields=fields,
                    api_spec=api_spec,
                    schema_id=schema_id,
                    schema_name=schema_name,
                    field_name=f"{composition_key}[{index}]",
                    field_path=f"{composition_key}[{index}]",
                    field_schema=child_schema,
                    required=False,
                )

        return fields

    def collect_field_records(
        self,
        fields: list[dict[str, Any]],
        api_spec: dict[str, Any],
        schema_id: str,
        schema_name: str,
        field_name: str,
        field_path: str,
        field_schema: Any,
        required: bool,
    ) -> None:
        if not isinstance(field_schema, dict):
            return

        field_type = self.infer_field_type(field_schema)
        item_schema = field_schema.get("items", {})
        schema_refs = self.extract_schema_refs(field_schema)
        enum_values = self.extract_enum_values(field_schema)

        fields.append(
            {
                "id": self.build_field_id(api_spec["id"], schema_name, field_path),
                "spec_id": api_spec["id"],
                "schema_id": schema_id,
                "schema_name": schema_name,
                "field_name": field_name,
                "field_path": field_path,
                "field_type": field_type,
                "description_text": self.clean_text(field_schema.get("description")),
                "required": required,
                "nullable": bool(field_schema.get("nullable", False)),
                "enum_values": enum_values,
                "array_item_type": self.infer_field_type(item_schema)
                if isinstance(item_schema, dict)
                else None,
                "schema_ref": field_schema.get("$ref"),
                "schema_refs": schema_refs,
                "format": field_schema.get("format"),
                "max_length": field_schema.get("maxLength"),
                "read_only": bool(field_schema.get("readOnly", False)),
                "write_only": bool(field_schema.get("writeOnly", False)),
                "composition_type": self.extract_composition(field_schema)["composition_type"],
                "sensitive_classification": self.classify_sensitive_field(field_path),
                "extracted_at": self.utc_now_iso(),
            }
        )

        self.collect_nested_property_fields(
            fields=fields,
            api_spec=api_spec,
            schema_id=schema_id,
            schema_name=schema_name,
            parent_path=field_path,
            field_schema=field_schema,
        )

    def collect_nested_property_fields(
        self,
        fields: list[dict[str, Any]],
        api_spec: dict[str, Any],
        schema_id: str,
        schema_name: str,
        parent_path: str,
        field_schema: dict[str, Any],
    ) -> None:
        nested_sources = []
        if isinstance(field_schema.get("properties"), dict):
            nested_sources.append(field_schema)
        if isinstance(field_schema.get("items"), dict) and isinstance(
            field_schema["items"].get("properties"), dict
        ):
            nested_sources.append(field_schema["items"])

        for source_schema in nested_sources:
            required_fields = set(self.as_string_list(source_schema.get("required")))
            for nested_name, nested_schema in sorted(source_schema["properties"].items()):
                self.collect_field_records(
                    fields=fields,
                    api_spec=api_spec,
                    schema_id=schema_id,
                    schema_name=schema_name,
                    field_name=nested_name,
                    field_path=f"{parent_path}.{nested_name}",
                    field_schema=nested_schema,
                    required=nested_name in required_fields,
                )

        for composition_key in self.COMPOSITION_KEYS:
            for index, child_schema in enumerate(self.as_list(field_schema.get(composition_key))):
                child_path = f"{parent_path}.{composition_key}[{index}]"
                self.collect_field_records(
                    fields=fields,
                    api_spec=api_spec,
                    schema_id=schema_id,
                    schema_name=schema_name,
                    field_name=f"{composition_key}[{index}]",
                    field_path=child_path,
                    field_schema=child_schema,
                    required=False,
                )

    def build_schema_id(self, spec_id: str, schema_name: str) -> str:
        raw_id = f"{spec_id}:{schema_name}"
        digest = hashlib.sha256(raw_id.encode("utf-8")).hexdigest()[:16]
        return f"api_schema:{schema_name}:{digest}"

    def build_field_id(self, spec_id: str, schema_name: str, field_path: str) -> str:
        raw_id = f"{spec_id}:{schema_name}:{field_path}"
        digest = hashlib.sha256(raw_id.encode("utf-8")).hexdigest()[:16]
        return f"api_schema_field:{digest}"

    def infer_field_type(self, schema: Any) -> str | None:
        if not isinstance(schema, dict):
            return None
        if "$ref" in schema:
            return "ref"
        if "type" in schema:
            return schema["type"]
        for composition_key in self.COMPOSITION_KEYS:
            if composition_key in schema:
                return composition_key
        if "properties" in schema:
            return "object"
        if "items" in schema:
            return "array"
        return None

    def extract_composition(self, schema: dict[str, Any]) -> dict[str, Any]:
        composition_types = [key for key in self.COMPOSITION_KEYS if key in schema]
        composition_refs = {
            key: self.extract_schema_refs(schema.get(key))
            for key in composition_types
        }
        return {
            "composition_type": "+".join(composition_types) if composition_types else None,
            "composition_refs": composition_refs,
        }

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

    def extract_enum_values(self, schema: Any) -> list[Any] | None:
        if not isinstance(schema, dict):
            return None
        if "enum" in schema and isinstance(schema["enum"], list):
            return schema["enum"]
        if isinstance(schema.get("items"), dict):
            return self.extract_enum_values(schema["items"])
        return None

    def classify_sensitive_field(self, field_path: str) -> str | None:
        lowered = field_path.lower()
        sensitive_terms = {
            "bank": "financial_account",
            "card": "card_data",
            "ssn": "government_id",
            "tax_id": "tax_identifier",
            "email": "contact",
            "phone": "contact",
            "address": "address",
            "dob": "date_of_birth",
        }
        for term, classification in sensitive_terms.items():
            if term in lowered:
                return classification
        return None

    def clean_text(self, value: Any) -> str | None:
        if value is None:
            return None
        text = html.unescape(str(value))
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def as_list(self, value: Any) -> list[Any]:
        return value if isinstance(value, list) else []

    def as_string_list(self, value: Any) -> list[str]:
        return [item for item in self.as_list(value) if isinstance(item, str)]

    def hash_json(self, value: Any) -> str:
        payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def write_json_array(self, path: Path, records: list[dict[str, Any]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as target:
            json.dump(records, target, indent=2)
            target.write("\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract reusable schemas from OpenAPI components.")
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC_PATH)
    parser.add_argument("--api-specs", type=Path, default=DEFAULT_API_SPECS_PATH)
    parser.add_argument("--api-schemas", type=Path, default=DEFAULT_API_SCHEMAS_PATH)
    parser.add_argument("--api-schema-fields", type=Path, default=DEFAULT_API_SCHEMA_FIELDS_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    extraction = SchemaExtraction(
        spec_path=args.spec,
        api_specs_path=args.api_specs,
        api_schemas_path=args.api_schemas,
        api_schema_fields_path=args.api_schema_fields,
    )
    schemas, fields = extraction.extract()
    print(json.dumps({"schema_count": len(schemas), "field_count": len(fields)}, indent=2))


if __name__ == "__main__":
    main()
