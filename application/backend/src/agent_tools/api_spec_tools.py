from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

import yaml

from chat.schemas import APIOperationDetails

_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_SRC_DIR = Path(__file__).resolve().parents[1]
_DATA_DIR = _SRC_DIR / "data"
_SPEC_PATH = _DATA_DIR / "raw_specs" / "openapi.spec3.yaml"

# ---------------------------------------------------------------------------
# Module-level caches  (populated on first access, never re-read)
# ---------------------------------------------------------------------------
_OPS_BY_OPERATION_ID: dict[str, Any] | None = None   # "GetAccount" → op dict
_OPS_BY_DOC_ID: dict[str, Any] | None = None         # "api_operation:get:…" → op
_SCHEMAS_BY_NAME: dict[str, Any] | None = None        # "account" → schema dict
_FIELDS_BY_SCHEMA: dict[str, list[Any]] | None = None # "account" → [field, …]
_SPEC: dict[str, Any] | None = None                   # parsed YAML
_PRODUCTS: list[dict[str, Any]] | None = None


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def _load_operations() -> tuple[dict[str, Any], dict[str, Any]]:
    global _OPS_BY_OPERATION_ID, _OPS_BY_DOC_ID
    if _OPS_BY_OPERATION_ID is not None:
        return _OPS_BY_OPERATION_ID, _OPS_BY_DOC_ID  # type: ignore[return-value]

    ops_path = _DATA_DIR / "api_operations.json"
    if not ops_path.exists():
        _log.warning("api_operations.json not found at %s", ops_path)
        _OPS_BY_OPERATION_ID = {}
        _OPS_BY_DOC_ID = {}
        return _OPS_BY_OPERATION_ID, _OPS_BY_DOC_ID

    with ops_path.open(encoding="utf-8") as fh:
        ops_list: list[dict[str, Any]] = json.load(fh)

    _OPS_BY_OPERATION_ID = {}
    _OPS_BY_DOC_ID = {}
    for op in ops_list:
        op_id = op.get("operation_id")
        doc_id = op.get("id")
        if op_id:
            _OPS_BY_OPERATION_ID[op_id] = op
        if doc_id:
            _OPS_BY_DOC_ID[doc_id] = op

    _log.info(
        "Loaded %d operations from api_operations.json", len(_OPS_BY_OPERATION_ID)
    )
    return _OPS_BY_OPERATION_ID, _OPS_BY_DOC_ID


def _load_schemas() -> dict[str, Any]:
    global _SCHEMAS_BY_NAME
    if _SCHEMAS_BY_NAME is not None:
        return _SCHEMAS_BY_NAME

    schemas_path = _DATA_DIR / "api_schemas.json"
    if not schemas_path.exists():
        _log.warning("api_schemas.json not found at %s", schemas_path)
        _SCHEMAS_BY_NAME = {}
        return _SCHEMAS_BY_NAME

    with schemas_path.open(encoding="utf-8") as fh:
        schemas_list: list[dict[str, Any]] = json.load(fh)

    _SCHEMAS_BY_NAME = {
        s["schema_name"]: s for s in schemas_list if s.get("schema_name")
    }
    _log.info("Loaded %d schemas from api_schemas.json", len(_SCHEMAS_BY_NAME))
    return _SCHEMAS_BY_NAME


def _load_products() -> list[dict[str, Any]]:
    global _PRODUCTS
    if _PRODUCTS is not None:
        return _PRODUCTS

    products_path = _DATA_DIR / "api_products.json"
    if not products_path.exists():
        _log.warning("api_products.json not found at %s", products_path)
        _PRODUCTS = []
        return _PRODUCTS

    with products_path.open(encoding="utf-8") as fh:
        _PRODUCTS = json.load(fh)
    return _PRODUCTS


def _load_schema_fields() -> dict[str, list[Any]]:
    global _FIELDS_BY_SCHEMA
    if _FIELDS_BY_SCHEMA is not None:
        return _FIELDS_BY_SCHEMA

    fields_path = _DATA_DIR / "api_schema_fields.json"
    if not fields_path.exists():
        _log.warning("api_schema_fields.json not found at %s", fields_path)
        _FIELDS_BY_SCHEMA = {}
        return _FIELDS_BY_SCHEMA

    with fields_path.open(encoding="utf-8") as fh:
        fields_list: list[dict[str, Any]] = json.load(fh)

    _FIELDS_BY_SCHEMA = {}
    for field in fields_list:
        name = field.get("schema_name")
        if name:
            _FIELDS_BY_SCHEMA.setdefault(name, []).append(field)

    _log.info(
        "Loaded fields for %d schemas from api_schema_fields.json",
        len(_FIELDS_BY_SCHEMA),
    )
    return _FIELDS_BY_SCHEMA


def _load_spec() -> dict[str, Any]:
    global _SPEC
    if _SPEC is not None:
        return _SPEC

    if not _SPEC_PATH.exists():
        _log.warning("OpenAPI spec not found at %s", _SPEC_PATH)
        _SPEC = {}
        return _SPEC

    _log.info("Parsing OpenAPI spec from %s (one-time load)…", _SPEC_PATH)
    with _SPEC_PATH.open(encoding="utf-8") as fh:
        _SPEC = yaml.safe_load(fh) or {}
    _log.info("OpenAPI spec loaded (%d top-level keys)", len(_SPEC))
    return _SPEC


# ---------------------------------------------------------------------------
# Field normalisation helpers
# ---------------------------------------------------------------------------

def _clean_html(text: str | None) -> str | None:
    """Strip simple HTML tags that appear in Stripe descriptions."""
    if not text:
        return None
    return re.sub(r"<[^>]+>", "", text).strip() or None


def _normalise_text(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


def _query_terms(query: str) -> list[str]:
    normalised = _normalise_text(query)
    terms = [query.lower().strip(), normalised]

    filler = {
        "api", "apis", "available", "show", "list", "all", "the",
        "product", "products", "endpoint", "endpoints", "what", "which",
        "are", "is", "in", "for", "of", "me",
    }
    terms.extend(
        token for token in normalised.split()
        if token and token not in filler and len(token) > 1
    )

    for product in _load_products():
        key = str(product.get("product_key") or "")
        display = str(product.get("display_name") or "")
        candidates = {
            key,
            key.replace("_", " "),
            _normalise_text(display),
        }
        if any(candidate and candidate in normalised for candidate in candidates):
            terms.insert(0, key.lower())

    return list(dict.fromkeys(term for term in terms if term))


def _normalize_parameters(raw: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert api_operations.json parameter dicts to OpenAPI-ish shape."""
    result = []
    for p in raw:
        schema_block: dict[str, Any] = {}
        if p.get("schema_type"):
            schema_block["type"] = p["schema_type"]
        if p.get("schema_format"):
            schema_block["format"] = p["schema_format"]
        if p.get("enum_values"):
            schema_block["enum"] = p["enum_values"]
        if p.get("max_length"):
            schema_block["maxLength"] = p["max_length"]

        result.append({
            "name":        p.get("name") or "",
            "in":          p.get("location") or "query",
            "required":    bool(p.get("required", False)),
            "description": _clean_html(p.get("description_text")) or "",
            "schema":      schema_block,
        })
    return result


def _extract_yaml_request_properties(
    spec: dict[str, Any],
    path: str,
    method: str,
) -> dict[str, Any]:
    """Extract inline request body properties from the YAML paths section.

    Stripe puts all request fields as inline schema properties under
    requestBody.content.{contentType}.schema.properties — they are NOT
    stored in api_operations.json, so we go to the YAML directly.
    """
    try:
        op_yaml: dict = (
            spec.get("paths", {})
            .get(path, {})
            .get(method.lower(), {})
        )
        content: dict = (
            op_yaml.get("requestBody", {}).get("content", {})
        )
        for content_type in (
            "application/x-www-form-urlencoded",
            "application/json",
            "multipart/form-data",
        ):
            props: dict = (
                content.get(content_type, {})
                .get("schema", {})
                .get("properties", {})
            )
            if props:
                return {
                    name: {k: v for k, v in {
                        "type":        prop.get("type"),
                        "description": _clean_html(prop.get("description")),
                        "enum":        prop.get("enum"),
                        "format":      prop.get("format"),
                        "$ref":        prop.get("$ref"),
                        "nullable":    prop.get("nullable"),
                    }.items() if v is not None}
                    for name, prop in props.items()
                }
    except Exception as exc:
        _log.debug("YAML request properties extraction failed: %s", exc)
    return {}


def _extract_yaml_schema_properties(
    spec: dict[str, Any],
    schema_name: str,
) -> dict[str, Any]:
    """Return top-level properties for a named component schema from YAML."""
    try:
        schema_yaml: dict = (
            spec.get("components", {})
            .get("schemas", {})
            .get(schema_name, {})
        )
        props = schema_yaml.get("properties", {})
        return {
            name: {k: v for k, v in {
                "type":        prop.get("type"),
                "description": _clean_html(prop.get("description")),
                "enum":        prop.get("enum"),
                "format":      prop.get("format"),
                "$ref":        prop.get("$ref"),
                "nullable":    prop.get("nullable"),
                "readOnly":    prop.get("readOnly"),
            }.items() if v is not None}
            for name, prop in props.items()
        }
    except Exception as exc:
        _log.debug(
            "YAML schema properties extraction failed for %s: %s",
            schema_name, exc,
        )
    return {}


def _build_request_schema(
    op: dict[str, Any],
    spec: dict[str, Any] | None,
) -> dict[str, Any]:
    rb = op.get("request_body") or {}
    if not rb and not op.get("request_schema_refs"):
        return {}

    schema: dict[str, Any] = {
        "required":       bool(rb.get("required", False)),
        "content_types":  rb.get("content_types") or [],
        "required_fields": rb.get("required_fields") or [],
        "schema_refs":    rb.get("schema_refs") or [],
    }

    if spec:
        path   = op.get("path", "")
        method = (op.get("method") or "GET").lower()
        inline_props = _extract_yaml_request_properties(spec, path, method)
        if inline_props:
            schema["properties"] = inline_props

    return schema


def _build_response_schemas(
    op: dict[str, Any],
) -> tuple[dict[str, Any], list[Any]]:
    responses = op.get("responses") or []
    response_schema: dict[str, Any] = {}
    error_codes: list[Any] = []

    for resp in responses:
        status = resp.get("status_code") or ""
        entry = {
            "description": resp.get("description") or "",
            "schema_refs":  resp.get("schema_refs") or [],
        }
        if resp.get("is_success"):
            response_schema[status] = entry
        elif resp.get("is_error"):
            error_codes.append({"status_code": status, **entry})

    return response_schema, error_codes


def _build_auth_scheme(
    op: dict[str, Any],
    spec: dict[str, Any] | None,
) -> dict[str, Any]:
    auth_required    = bool(op.get("auth_required", False))
    scheme_names: list[str] = op.get("security_scheme_names") or []

    scheme: dict[str, Any] = {
        "required":     auth_required,
        "scheme_names": scheme_names,
        "definitions":  {},
    }

    if spec and scheme_names:
        security_schemes: dict = (
            spec.get("components", {}).get("securitySchemes", {})
        )
        for name in scheme_names:
            if name in security_schemes:
                scheme["definitions"][name] = dict(security_schemes[name])

    return scheme


def _op_to_details(
    op: dict[str, Any],
    spec: dict[str, Any] | None = None,
) -> APIOperationDetails:
    """Map one api_operations.json entry to APIOperationDetails.

    *spec* is the parsed YAML dict; pass None to skip YAML enrichment
    (used during bulk search to keep it fast).
    """
    response_schema, error_codes = _build_response_schemas(op)
    return APIOperationDetails(
        operation_id  = op.get("operation_id"),
        endpoint      = op.get("path"),
        method        = op.get("method"),
        product_key   = op.get("product_key"),
        business_domain = op.get("business_domain"),
        summary       = op.get("summary"),
        description   = _clean_html(op.get("description_text")),
        parameters    = _normalize_parameters(op.get("parameters") or []),
        request_schema  = _build_request_schema(op, spec),
        response_schema = response_schema,
        auth_scheme     = _build_auth_scheme(op, spec),
        error_codes     = error_codes,
        related_operations = [],   # deterministic: not in spec artifacts
    )


# ---------------------------------------------------------------------------
# Ref resolver
# ---------------------------------------------------------------------------

def _ref_to_name(ref: str) -> str:
    """Extract the schema name from a $ref string.

    '#/components/schemas/account' → 'account'
    """
    return ref.split("/")[-1] if "/" in ref else ref


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_operation_details(operation_id: str) -> APIOperationDetails | None:
    """Return full details for a single operation.

    Accepts:
    - operationId ("GetAccount")
    - Milvus/Neo4j doc id ("api_operation:get:2ebbb181cef1900f")
    - neo4j_node_id if it happens to match an operationId

    Returns None if the operation is not found in the indexed artifacts.
    """
    ops_by_id, ops_by_doc = _load_operations()

    op = ops_by_id.get(operation_id) or ops_by_doc.get(operation_id)
    if op is None:
        _log.debug("Operation not found for id=%r", operation_id)
        return None

    spec = _load_spec()
    _log.debug(
        "get_operation_details: building details for %r", operation_id
    )
    return _op_to_details(op, spec=spec)


def search_operations_by_name_or_path(
    query: str,
    max_results: int = 20,
) -> list[APIOperationDetails]:
    """Case-insensitive substring search across operation_id, path, summary,
    description, product_key, and resource_name.

    Returns up to *max_results* results ranked by match quality.
    No YAML enrichment — uses JSON data only for speed.
    """
    ops_by_id, _ = _load_operations()
    terms = _query_terms(query)
    if not terms:
        return []

    scored: list[tuple[int, dict[str, Any]]] = []
    for op in ops_by_id.values():
        score = 0
        op_id = (op.get("operation_id") or "").lower()
        path = (op.get("path") or "").lower()
        summary = (op.get("summary") or "").lower()
        product_key = (op.get("product_key") or "").lower()
        resource = (op.get("resource_name") or "").lower()
        domain = (op.get("business_domain") or "").lower()
        description = (op.get("description_text") or "").lower()

        for term in terms:
            term_score = 0
            if term == product_key:
                term_score += 8
            if term == domain:
                term_score += 4
            if term in op_id:
                term_score += 5
            if term in path:
                term_score += 4
            if term in summary:
                term_score += 3
            if term in product_key:
                term_score += 3
            if term in resource:
                term_score += 2
            if term in description:
                term_score += 1
            score = max(score, term_score)
        if score:
            scored.append((score, op))

    scored.sort(key=lambda x: x[0], reverse=True)
    _log.debug(
        "search_operations_by_name_or_path: %d match(es) for query=%r",
        len(scored),
        query,
    )
    return [_op_to_details(op) for _, op in scored[:max_results]]


def get_schema_details(schema_name: str) -> dict[str, Any]:
    """Return metadata and field-level details for a named component schema.

    Accepts both plain name ("account") and $ref format
    ("#/components/schemas/account").

    Returns an empty dict if the schema is not found — never raises.
    """
    name = _ref_to_name(schema_name)
    schemas = _load_schemas()
    meta = schemas.get(name)
    if meta is None:
        _log.debug("Schema not found for name=%r", name)
        return {}

    fields = _load_schema_fields().get(name) or []
    spec   = _load_spec()
    props  = _extract_yaml_schema_properties(spec, name) if spec else {}

    return {
        "schema_name":       meta.get("schema_name"),
        "title":             meta.get("title"),
        "description":       _clean_html(meta.get("description_text")),
        "type":              meta.get("schema_type"),
        "format":            meta.get("format"),
        "required_fields":   meta.get("required_fields") or [],
        "enum_values":       meta.get("enum_values"),
        "nullable":          meta.get("nullable", False),
        "is_error_schema":   meta.get("is_error_schema", False),
        "property_count":    meta.get("property_count", 0),
        "referenced_schemas": [
            _ref_to_name(r)
            for r in (meta.get("referenced_schema_refs") or [])
        ],
        "properties":        props,       # from YAML; empty dict if not found
        "fields":            [           # from api_schema_fields.json
            {
                "name":        f.get("field_name"),
                "path":        f.get("field_path"),
                "type":        f.get("field_type"),
                "description": _clean_html(f.get("description_text")),
                "required":    f.get("required", False),
                "nullable":    f.get("nullable", False),
                "enum_values": f.get("enum_values"),
                "schema_ref":  f.get("schema_ref"),
                "read_only":   f.get("read_only", False),
                "write_only":  f.get("write_only", False),
            }
            for f in fields
        ],
    }
