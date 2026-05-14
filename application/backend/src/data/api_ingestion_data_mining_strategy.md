# API Product Ingestion And Data Mining Strategy

Source spec: `openapi.spec3.yaml`

API provider: Stripe API  
OpenAPI version: `3.0.0`  
Stripe spec version: `2026-04-22.dahlia`  
Audience: public GA spec  
Base server: `https://api.stripe.com/`

## Spec Inventory

The OpenAPI file contains:

- 437 API paths
- 619 operations
- 274 `GET` operations
- 311 `POST` operations
- 34 `DELETE` operations
- 1,311 component schemas
- Security schemes: Basic auth and Bearer auth

Primary API product areas discovered from path prefixes:

- Connect and Core Accounts: `account`, `accounts`, `account_links`, `account_sessions`, `external_accounts`, `linked_accounts`, `v2/core`
- Payments: `payment_intents`, `setup_intents`, `payment_methods`, `charges`, `refunds`, `tokens`, `sources`, `mandates`
- Billing and Revenue: `billing`, `billing_portal`, `invoices`, `invoiceitems`, `invoice_payments`, `credit_notes`, `subscriptions`, `subscription_items`, `subscription_schedules`, `prices`, `products`, `plans`, `coupons`, `promotion_codes`, `quotes`
- Checkout: `checkout/sessions`, `payment_links`
- Treasury and Money Movement: `treasury`, `balance`, `balance_transactions`, `payouts`, `topups`, `transfers`, `application_fees`
- Issuing: `issuing/authorizations`, `issuing/cardholders`, `issuing/cards`, `issuing/disputes`, `issuing/tokens`, `issuing/transactions`
- Terminal: `terminal/configurations`, `terminal/connection_tokens`, `terminal/locations`, `terminal/readers`, `terminal/refunds`
- Risk, Compliance, and Identity: `radar`, `reviews`, `disputes`, `identity`, `tax`, `tax_ids`, `tax_rates`, `tax_codes`
- Data, Reporting, and Events: `events`, `webhook_endpoints`, `reporting`, `sigma`, `files`, `file_links`
- Financial Connections: `financial_connections`
- Specialized products: `climate`, `forwarding`, `apple_pay`, `apps`, `entitlements`
- Test support: `test_helpers`
- v2 APIs: `v2/billing`, `v2/commerce`, `v2/core`

## Ingestion Goals

The ingestion pipeline should convert the OpenAPI spec into a searchable API product catalog, an operation-level retrieval index, and a normalized metadata graph that can power API discovery, endpoint recommendation, dependency analysis, and change detection.

Core outputs:

- API product catalog grouped by business capability.
- Endpoint catalog with request, response, auth, pagination, and lifecycle metadata.
- Schema catalog with object models, field metadata, enum values, nested references, and reusable components.
- Relationship graph connecting products, endpoints, schemas, parameters, operations, and resources.
- Text and vector indexes for semantic API discovery.
- Versioned snapshots for diffing future spec updates.

## Ingestion Pipeline

### 1. Raw Spec Intake

Store the original file exactly as received.

Recommended records:

- `source_file_path`
- `source_sha256`
- `openapi_version`
- `provider_name`
- `api_title`
- `api_version`
- `release_phase`
- `audience`
- `server_urls`
- `ingested_at`

Validation:

- Confirm valid OpenAPI structure: `info`, `servers`, `paths`, `components`.
- Validate path and operation uniqueness.
- Detect unsupported OpenAPI features before extraction.

### 2. Product Area Discovery

Derive product group from path segments:

- `/v1/billing/meters` -> product `billing`
- `/v1/payment_intents/{intent}/confirm` -> product `payment_intents`
- `/v2/core/accounts` -> product `core`
- `/v1/test_helpers/...` -> product `test_helpers`

Then map raw prefixes to business domains:

- `payments`: `payment_intents`, `payment_methods`, `charges`, `refunds`, `tokens`, `sources`
- `billing`: `billing`, `invoices`, `subscriptions`, `prices`, `products`, `coupons`, `quotes`
- `connect`: `account`, `accounts`, `account_links`, `account_sessions`, `external_accounts`
- `money_movement`: `balance`, `payouts`, `topups`, `transfers`, `treasury`
- `risk_compliance`: `radar`, `reviews`, `disputes`, `identity`, `tax`
- `platform_data`: `events`, `webhook_endpoints`, `reporting`, `sigma`, `files`

Keep both the raw product key and curated domain. The raw key preserves exact source truth; the curated domain improves discovery.

### 3. Operation Extraction

For every path and HTTP method, extract:

- `operationId`
- `summary`
- `description`
- `method`
- `path`
- `path_template`
- `api_version`
- `product_key`
- `business_domain`
- path parameters
- query parameters
- request content types
- request body schema refs
- request required fields
- response status codes
- response content types
- response schema refs
- auth requirements
- OpenAPI extensions such as `x-stripeBypassValidation`

Classify operation intent:

- `list`: `GET /resources`
- `retrieve`: `GET /resources/{id}`
- `create`: `POST /resources`
- `update`: `POST /resources/{id}`
- `delete`: `DELETE /resources/{id}`
- `action`: `POST /resources/{id}/cancel`, `/confirm`, `/capture`, `/expire`
- `search`: paths ending in `/search`
- `preview`: paths containing `/preview`
- `test_helper`: paths under `/test_helpers`

### 4. Schema Extraction

For each `components.schemas` entry, extract:

- schema name
- type
- title
- description
- required fields
- properties
- enum values
- nullable state
- array item schemas
- object nested schemas
- `$ref` dependencies
- `oneOf`, `anyOf`, `allOf` composition
- read-only/write-only flags when present
- provider extensions

Create field-level records so data mining can answer questions such as:

- Which APIs accept `customer`?
- Which responses expose `currency`?
- Which endpoints return `payment_intent`?
- Which fields are enums and what are the allowed values?

### 5. Relationship Mining

Build a graph with these edges:

- Product `HAS_OPERATION` Operation
- Operation `USES_REQUEST_SCHEMA` Schema
- Operation `RETURNS_RESPONSE_SCHEMA` Schema
- Operation `HAS_PARAMETER` Parameter
- Schema `HAS_FIELD` Field
- Field `REFERENCES_SCHEMA` Schema
- Operation `ACTION_ON_RESOURCE` Resource
- Operation `REQUIRES_AUTH` SecurityScheme
- Operation `HAS_ERROR_RESPONSE` ErrorSchema

Useful derived relationships:

- `create -> retrieve -> update -> delete` lifecycle chains by resource.
- Parent-child nested resource chains, such as customer -> subscriptions.
- Action endpoints attached to a resource, such as payment_intent -> confirm/cancel/capture.
- Event/webhook-related endpoints connected to response object types.

### 6. Semantic Indexing

Create searchable documents at three granularities:

- Product document: one per product area.
- Operation document: one per endpoint operation.
- Schema document: one per schema/object model.

Operation document text should combine:

- product name
- method and path
- operation ID
- summary
- cleaned HTML description
- parameter names and descriptions
- request field names
- response schema names
- enum values
- lifecycle labels

Store both keyword-search fields and vector embeddings. Use chunking only for very large schema descriptions; endpoint records should usually stay as one document.

## Metadata Data Model

The model below can be implemented in SQL tables, document storage, or Pydantic models. Prefer stable IDs generated from source facts so repeated ingestion is idempotent.

### api_specs

Stores one ingested OpenAPI snapshot.

Fields:

- `id`: stable hash of provider, version, source hash
- `provider`: `stripe`
- `title`
- `description`
- `openapi_version`
- `api_version`
- `release_phase`
- `audience`
- `terms_of_service_url`
- `contact_name`
- `contact_url`
- `contact_email`
- `source_path`
- `source_sha256`
- `raw_spec_uri`
- `server_urls`
- `ingested_at`

### api_products

Stores discovered product areas.

Fields:

- `id`: hash of spec ID and product key
- `spec_id`
- `product_key`: raw prefix, for example `payment_intents`
- `display_name`: for example `Payment Intents`
- `business_domain`: for example `payments`
- `api_version`: `v1` or `v2`
- `path_prefixes`
- `operation_count`
- `schema_count`
- `description`
- `is_test_product`
- `tags`

### api_operations

Stores one row per method and path.

Fields:

- `id`: hash of spec ID, method, path
- `spec_id`
- `product_id`
- `operation_id`
- `method`
- `path`
- `normalized_path`
- `api_version`
- `summary`
- `description_text`
- `operation_type`: `list`, `retrieve`, `create`, `update`, `delete`, `action`, `search`, `preview`
- `resource_name`
- `action_name`
- `auth_required`
- `security_scheme_ids`
- `request_content_types`
- `response_content_types`
- `success_status_codes`
- `error_status_codes`
- `request_schema_id`
- `primary_response_schema_id`
- `is_deprecated`
- `is_test_helper`
- `source_line`
- `source_hash`

### api_parameters

Stores path, query, header, and cookie parameters.

Fields:

- `id`
- `operation_id`
- `name`
- `location`: `path`, `query`, `header`, `cookie`
- `required`
- `style`
- `explode`
- `schema_type`
- `schema_ref`
- `description_text`
- `enum_values`
- `default_value`
- `max_length`
- `is_expand_parameter`

### api_request_bodies

Stores operation request body metadata.

Fields:

- `id`
- `operation_id`
- `required`
- `content_type`
- `schema_id`
- `schema_ref`
- `encoding`
- `required_fields`
- `field_count`
- `supports_deep_object_encoding`

### api_responses

Stores operation responses.

Fields:

- `id`
- `operation_id`
- `status_code`
- `description`
- `content_type`
- `schema_id`
- `schema_ref`
- `is_success`
- `is_error`

### api_schemas

Stores reusable object schemas.

Fields:

- `id`: hash of spec ID and schema name
- `spec_id`
- `schema_name`
- `title`
- `description_text`
- `schema_type`
- `required_fields`
- `enum_values`
- `composition_type`
- `referenced_schema_ids`
- `property_count`
- `is_error_schema`
- `raw_schema_hash`

### api_schema_fields

Stores individual fields inside schemas and request objects.

Fields:

- `id`
- `schema_id`
- `field_path`: for example `payment_method_options.card.request_three_d_secure`
- `field_name`
- `field_type`
- `description_text`
- `required`
- `nullable`
- `enum_values`
- `array_item_type`
- `schema_ref`
- `format`
- `max_length`
- `read_only`
- `write_only`
- `sensitive_classification`

### api_relationships

Stores mined graph edges.

Fields:

- `id`
- `spec_id`
- `source_type`
- `source_id`
- `relationship_type`
- `target_type`
- `target_id`
- `confidence`
- `evidence`

Relationship types:

- `HAS_OPERATION`
- `USES_REQUEST_SCHEMA`
- `RETURNS_RESPONSE_SCHEMA`
- `HAS_PARAMETER`
- `HAS_FIELD`
- `REFERENCES_SCHEMA`
- `ACTION_ON_RESOURCE`
- `PARENT_RESOURCE_OF`
- `LIFECYCLE_RELATED_TO`
- `REQUIRES_AUTH`

### api_search_documents

Stores keyword and vector-ready search documents.

Fields:

- `id`
- `spec_id`
- `entity_type`: `product`, `operation`, `schema`
- `entity_id`
- `title`
- `body`
- `keywords`
- `business_domain`
- `product_key`
- `method`
- `path`
- `embedding_model`
- `embedding_vector`
- `indexed_at`

### api_ingestion_runs

Tracks each ingestion execution.

Fields:

- `id`
- `spec_id`
- `started_at`
- `completed_at`
- `status`
- `source_path`
- `source_sha256`
- `paths_seen`
- `operations_seen`
- `schemas_seen`
- `products_created`
- `operations_created`
- `schemas_created`
- `warnings`
- `errors`

## Data Mining Features

High-value mined attributes:

- Product/domain classification from path prefixes.
- CRUD/action classification from method and path shape.
- Resource identity from path templates.
- Parent-child resource nesting.
- Request/response schema reuse.
- Required input fields.
- Enum vocabulary.
- Expandable response support through `expand` parameters.
- Pagination support through list response schemas and common list parameters.
- Search support through `/search` paths.
- Test-only endpoint detection through `/test_helpers`.
- Risk/compliance/sensitive category hints from field names such as account, person, tax, identity, card, bank, ssn, email, phone, address.

## Example Operation Record

```json
{
  "provider": "stripe",
  "api_version": "v1",
  "product_key": "payment_intents",
  "business_domain": "payments",
  "method": "POST",
  "path": "/v1/payment_intents/{intent}/confirm",
  "operation_type": "action",
  "resource_name": "payment_intents",
  "action_name": "confirm",
  "auth_required": true,
  "search_text": "Payment Intents POST /v1/payment_intents/{intent}/confirm confirm payment intent request fields response schema"
}
```

## Recommended Implementation Order

1. Build raw spec loader and snapshot hashing.
2. Extract products, operations, parameters, request bodies, responses, and schemas.
3. Persist normalized metadata records.
4. Generate relationships and lifecycle/action classifications.
5. Build keyword search documents.
6. Add embeddings for semantic discovery.
7. Add version diffing between spec snapshots.
8. Add quality checks and ingestion run reporting.

## Quality Checks

Run these checks after each ingestion:

- Every path operation has a stable operation ID or generated fallback ID.
- Every `$ref` resolves to a component schema.
- Every operation maps to exactly one product.
- Every operation has at least one response.
- Every request/response schema reference is represented in `api_schemas`.
- No duplicate operation IDs after normalization.
- Test helper endpoints are flagged and excluded from production recommendations by default.
- HTML descriptions are cleaned before indexing.

