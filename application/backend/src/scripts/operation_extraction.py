from __future__ import annotations

import hashlib
import html
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from scripts.product_area_discovery import ProductAreaDiscovery


PROJECT_SRC = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_SRC / "data"
DEFAULT_RAW_SPECS_DIR = DATA_DIR / "raw_specs"
DEFAULT_SPEC_PATH = DEFAULT_RAW_SPECS_DIR / "openapi.spec3.yaml"
DEFAULT_API_SPECS_PATH = DATA_DIR / "api_specs.json"
DEFAULT_API_PRODUCTS_PATH = DATA_DIR / "api_products.json"
DEFAULT_API_OPERATIONS_PATH = DATA_DIR / "api_operations.json"


class OperationExtraction:
    HTTP_METHODS = {"get", "post", "put", "patch", "delete", "options", "head", "trace"}

    def __init__(
        self,
        spec_path: Path = DEFAULT_SPEC_PATH,
        api_specs_path: Path = DEFAULT_API_SPECS_PATH,
        api_products_path: Path = DEFAULT_API_PRODUCTS_PATH,
        api_operations_path: Path = DEFAULT_API_OPERATIONS_PATH,
    ) -> None:
        self.spec_path = spec_path.resolve()
        self.api_specs_path = api_specs_path.resolve()
        self.api_products_path = api_products_path.resolve()
        self.api_operations_path = api_operations_path.resolve()
        self.product_discovery = ProductAreaDiscovery(
            spec_path=self.spec_path,
            api_specs_path=self.api_specs_path,
            api_products_path=self.api_products_path,
        )

    def extract(self) -> list[dict[str, Any]]:
        spec = self.load_openapi_spec()
        api_spec = self.get_current_api_spec()
        products_by_key = self.get_products_by_key()
        operations = self.build_operation_records(
            spec=spec,
            api_spec=api_spec,
            products_by_key=products_by_key,
        )
        self.write_api_operations(operations)
        return operations

    def utc_now_iso(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def load_openapi_spec(self) -> dict[str, Any]:
        with self.spec_path.open("r", encoding="utf-8") as source:
            spec = yaml.safe_load(source)

        if not isinstance(spec, dict):
            raise ValueError(f"{self.spec_path} did not parse as an OpenAPI object.")

        paths = spec.get("paths")
        if not isinstance(paths, dict):
            raise ValueError(f"{self.spec_path} is missing a valid paths object.")

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

    def get_products_by_key(self) -> dict[str, dict[str, Any]]:
        products = self.read_json_array(self.api_products_path)
        if not products:
            raise ValueError(
                f"No api_products rows found in {self.api_products_path}. "
                "Run product_area_discovery.py first."
            )
        return {product["product_key"]: product for product in products}

    def read_json_array(self, path: Path) -> list[dict[str, Any]]:
        if not path.exists():
            return []

        with path.open("r", encoding="utf-8") as source:
            records = json.load(source)

        if not isinstance(records, list):
            raise ValueError(f"{path} must contain a JSON array.")

        return records

    def build_operation_records(
        self,
        spec: dict[str, Any],
        api_spec: dict[str, Any],
        products_by_key: dict[str, dict[str, Any]],
    ) -> list[dict[str, Any]]:
        operations: list[dict[str, Any]] = []
        global_security = spec.get("security")

        for path, path_item in spec["paths"].items():
            if not isinstance(path_item, dict):
                continue

            path_level_parameters = path_item.get("parameters", [])
            product_key = self.product_discovery.extract_product_key(path)
            api_version = self.product_discovery.extract_api_version(path)
            product = products_by_key.get(product_key or "")

            for method, operation in path_item.items():
                method = method.lower()
                if method not in self.HTTP_METHODS or not isinstance(operation, dict):
                    continue

                operations.append(
                    self.build_operation_record(
                        api_spec=api_spec,
                        product=product,
                        api_version=api_version,
                        product_key=product_key,
                        path=path,
                        method=method,
                        operation=operation,
                        path_level_parameters=path_level_parameters,
                        global_security=global_security,
                    )
                )

        operations.sort(key=lambda item: (item["path"], item["method"]))
        return operations

    def build_operation_record(
        self,
        api_spec: dict[str, Any],
        product: dict[str, Any] | None,
        api_version: str | None,
        product_key: str | None,
        path: str,
        method: str,
        operation: dict[str, Any],
        path_level_parameters: list[dict[str, Any]],
        global_security: Any,
    ) -> dict[str, Any]:
        merged_parameters = self.merge_parameters(
            path_level_parameters=path_level_parameters,
            operation_parameters=operation.get("parameters", []),
        )
        request_body = self.extract_request_body(operation.get("requestBody"))
        responses = self.extract_responses(operation.get("responses", {}))
        security = self.extract_security(operation=operation, global_security=global_security)
        request_schema_refs = request_body.get("schema_refs", []) if request_body else []
        response_schema_refs = self.unique_sorted(
            ref for response in responses for ref in response.get("schema_refs", [])
        )
        parameter_schema_refs = self.unique_sorted(
            ref for parameter in merged_parameters for ref in parameter.get("schema_refs", [])
        )
        schema_refs = self.unique_sorted(
            [*request_schema_refs, *response_schema_refs, *parameter_schema_refs]
        )
        classification = self.classify_operation(method=method, path=path)

        return {
            "id": self.build_operation_id(
                spec_id=api_spec["id"],
                method=method,
                path=path,
            ),
            "spec_id": api_spec["id"],
            "product_id": product.get("id") if product else None,
            "provider": api_spec.get("provider"),
            "product_key": product_key,
            "business_domain": product.get("business_domain") if product else None,
            "api_version": api_version,
            "method": method.upper(),
            "path": path,
            "normalized_path": self.normalize_path(path),
            "operation_id": operation.get("operationId"),
            "summary": operation.get("summary"),
            "description_text": self.clean_text(operation.get("description")),
            "parameters": merged_parameters,
            "request_body": request_body,
            "responses": responses,
            "auth_required": security["auth_required"],
            "security_scheme_names": security["security_scheme_names"],
            "request_schema_refs": request_schema_refs,
            "response_schema_refs": response_schema_refs,
            "parameter_schema_refs": parameter_schema_refs,
            "schema_refs": schema_refs,
            "operation_type": classification["operation_type"],
            "resource_name": classification["resource_name"],
            "action_name": classification["action_name"],
            "is_deprecated": bool(operation.get("deprecated", False)),
            "is_test_helper": product_key == "test_helpers",
            "extracted_at": self.utc_now_iso(),
        }

    def build_operation_id(self, spec_id: str, method: str, path: str) -> str:
        raw_id = f"{spec_id}:{method.upper()}:{path}"
        digest = hashlib.sha256(raw_id.encode("utf-8")).hexdigest()[:16]
        return f"api_operation:{method}:{digest}"

    def normalize_path(self, path: str) -> str:
        return re.sub(r"\{[^}/]+\}", "{}", path)

    def merge_parameters(
        self,
        path_level_parameters: Any,
        operation_parameters: Any,
    ) -> list[dict[str, Any]]:
        merged: dict[tuple[str, str], dict[str, Any]] = {}

        for parameter in [*self.as_list(path_level_parameters), *self.as_list(operation_parameters)]:
            if not isinstance(parameter, dict):
                continue

            normalized = self.extract_parameter(parameter)
            key = (normalized.get("name", ""), normalized.get("location", ""))
            merged[key] = normalized

        return list(merged.values())

    def extract_parameter(self, parameter: dict[str, Any]) -> dict[str, Any]:
        schema = parameter.get("schema", {})
        schema_refs = self.extract_schema_refs(parameter)

        return {
            "name": parameter.get("name"),
            "location": parameter.get("in"),
            "required": bool(parameter.get("required", False)),
            "style": parameter.get("style"),
            "explode": parameter.get("explode"),
            "description_text": self.clean_text(parameter.get("description")),
            "schema_type": schema.get("type") if isinstance(schema, dict) else None,
            "schema_format": schema.get("format") if isinstance(schema, dict) else None,
            "schema_ref": schema.get("$ref") if isinstance(schema, dict) else None,
            "schema_refs": schema_refs,
            "enum_values": schema.get("enum") if isinstance(schema, dict) else None,
            "default_value": schema.get("default") if isinstance(schema, dict) else None,
            "max_length": schema.get("maxLength") if isinstance(schema, dict) else None,
            "is_expand_parameter": parameter.get("name") == "expand",
        }

    def extract_request_body(self, request_body: Any) -> dict[str, Any] | None:
        if not isinstance(request_body, dict):
            return None

        content = request_body.get("content", {})
        content_types = sorted(content.keys()) if isinstance(content, dict) else []
        schema_refs = self.extract_schema_refs(request_body)
        required_fields = self.extract_required_fields_from_content(content)

        return {
            "required": bool(request_body.get("required", False)),
            "content_types": content_types,
            "schema_refs": schema_refs,
            "required_fields": required_fields,
            "encoding": self.extract_encoding(content),
        }

    def extract_required_fields_from_content(self, content: Any) -> list[str]:
        required_fields: set[str] = set()
        if not isinstance(content, dict):
            return []

        for media_type in content.values():
            if not isinstance(media_type, dict):
                continue

            schema = media_type.get("schema", {})
            if isinstance(schema, dict):
                required_fields.update(
                    field for field in schema.get("required", []) if isinstance(field, str)
                )

        return sorted(required_fields)

    def extract_encoding(self, content: Any) -> dict[str, Any]:
        encodings: dict[str, Any] = {}
        if not isinstance(content, dict):
            return encodings

        for content_type, media_type in content.items():
            if isinstance(media_type, dict) and "encoding" in media_type:
                encodings[content_type] = media_type["encoding"]

        return encodings

    def extract_responses(self, responses: Any) -> list[dict[str, Any]]:
        if not isinstance(responses, dict):
            return []

        extracted_responses = []
        for status_code, response in responses.items():
            if not isinstance(response, dict):
                continue

            content = response.get("content", {})
            content_types = sorted(content.keys()) if isinstance(content, dict) else []
            extracted_responses.append(
                {
                    "status_code": str(status_code),
                    "description": self.clean_text(response.get("description")),
                    "content_types": content_types,
                    "schema_refs": self.extract_schema_refs(response),
                    "is_success": self.is_success_status(status_code),
                    "is_error": not self.is_success_status(status_code),
                }
            )

        return extracted_responses

    def extract_security(self, operation: dict[str, Any], global_security: Any) -> dict[str, Any]:
        security = operation["security"] if "security" in operation else global_security
        if security == [] or security is None:
            return {"auth_required": False, "security_scheme_names": []}

        scheme_names: set[str] = set()
        for requirement in self.as_list(security):
            if isinstance(requirement, dict):
                scheme_names.update(requirement.keys())

        return {
            "auth_required": bool(scheme_names),
            "security_scheme_names": sorted(scheme_names),
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

    def classify_operation(self, method: str, path: str) -> dict[str, str | None]:
        segments = self.product_discovery.path_segments(path)
        meaningful_segments = [segment for segment in segments if segment not in {"v1", "v2"}]
        product_key = self.product_discovery.extract_product_key(path)
        last_segment = meaningful_segments[-1] if meaningful_segments else None
        previous_segment = meaningful_segments[-2] if len(meaningful_segments) > 1 else None
        last_is_param = self.is_path_param(last_segment)
        previous_is_param = self.is_path_param(previous_segment)
        action_name = None

        if method == "get" and last_segment == "search":
            operation_type = "search"
        elif method == "get" and last_is_param:
            operation_type = "retrieve"
        elif method == "get":
            operation_type = "list"
        elif method == "delete":
            operation_type = "delete"
        elif method == "post" and last_is_param:
            operation_type = "update"
        elif method == "post" and previous_is_param and last_segment is not None:
            operation_type = "action"
            action_name = last_segment
        elif method == "post":
            operation_type = "create"
        else:
            operation_type = "other"

        return {
            "operation_type": operation_type,
            "resource_name": product_key,
            "action_name": action_name,
        }

    def is_path_param(self, segment: str | None) -> bool:
        return bool(segment and segment.startswith("{") and segment.endswith("}"))

    def is_success_status(self, status_code: Any) -> bool:
        status = str(status_code)
        return status.startswith("2")

    def clean_text(self, value: Any) -> str | None:
        if value is None:
            return None

        text = html.unescape(str(value))
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def as_list(self, value: Any) -> list[Any]:
        return value if isinstance(value, list) else []

    def unique_sorted(self, values: Any) -> list[str]:
        return sorted({value for value in values if isinstance(value, str)})

    def write_api_operations(self, operations: list[dict[str, Any]]) -> None:
        self.api_operations_path.parent.mkdir(parents=True, exist_ok=True)
        with self.api_operations_path.open("w", encoding="utf-8") as target:
            json.dump(operations, target, indent=2)
            target.write("\n")
