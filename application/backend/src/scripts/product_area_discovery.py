from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


PROJECT_SRC = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_SRC / "data"
DEFAULT_RAW_SPECS_DIR = DATA_DIR / "raw_specs"
DEFAULT_SPEC_PATH = DEFAULT_RAW_SPECS_DIR / "openapi.spec3.yaml"
DEFAULT_API_SPECS_PATH = DATA_DIR / "api_specs.json"
DEFAULT_API_PRODUCTS_PATH = DATA_DIR / "api_products.json"


class ProductAreaDiscovery:
    DOMAIN_MAP = {
        "payments": {
            "charges",
            "confirmation_tokens",
            "mandates",
            "payment_attempt_records",
            "payment_intents",
            "payment_method_configurations",
            "payment_method_domains",
            "payment_methods",
            "payment_records",
            "refunds",
            "setup_attempts",
            "setup_intents",
            "sources",
            "tokens",
        },
        "billing": {
            "billing",
            "billing_portal",
            "checkout",
            "coupons",
            "credit_notes",
            "customer_sessions",
            "customers",
            "entitlements",
            "invoice_payments",
            "invoice_rendering_templates",
            "invoiceitems",
            "invoices",
            "payment_links",
            "plans",
            "prices",
            "products",
            "promotion_codes",
            "quotes",
            "shipping_rates",
            "subscription_items",
            "subscription_schedules",
            "subscriptions",
        },
        "connect": {
            "account",
            "account_links",
            "account_sessions",
            "accounts",
            "application_fees",
            "external_accounts",
            "linked_accounts",
            "link_account_sessions",
        },
        "money_movement": {
            "balance",
            "balance_settings",
            "balance_transactions",
            "financial_connections",
            "payouts",
            "topups",
            "transfers",
            "treasury",
        },
        "risk_compliance": {
            "disputes",
            "identity",
            "radar",
            "reviews",
            "tax",
            "tax_codes",
            "tax_ids",
            "tax_rates",
        },
        "platform_data": {
            "apps",
            "ephemeral_keys",
            "events",
            "file_links",
            "files",
            "forwarding",
            "reporting",
            "sigma",
            "webhook_endpoints",
        },
        "issuing": {"issuing"},
        "terminal": {"terminal"},
        "specialized": {"apple_pay", "climate", "country_specs", "exchange_rates"},
        "test_support": {"test_helpers"},
        "commerce": {"commerce"},
        "core": {"core"},
    }

    HTTP_METHODS = {"get", "post", "put", "patch", "delete", "options", "head", "trace"}

    def __init__(
        self,
        spec_path: Path = DEFAULT_SPEC_PATH,
        api_specs_path: Path = DEFAULT_API_SPECS_PATH,
        api_products_path: Path = DEFAULT_API_PRODUCTS_PATH,
    ) -> None:
        self.spec_path = spec_path.resolve()
        self.api_specs_path = api_specs_path.resolve()
        self.api_products_path = api_products_path.resolve()

    def discover(self) -> list[dict[str, Any]]:
        spec = self.load_openapi_spec()
        api_spec = self.get_current_api_spec()
        product_stats = self.collect_product_stats(spec)
        products = [
            self.build_api_product_record(api_spec=api_spec, product_key=product_key, stats=stats)
            for product_key, stats in sorted(product_stats.items())
        ]
        self.write_api_products(products)
        return products

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

    def read_json_array(self, path: Path) -> list[dict[str, Any]]:
        if not path.exists():
            return []

        with path.open("r", encoding="utf-8") as source:
            records = json.load(source)

        if not isinstance(records, list):
            raise ValueError(f"{path} must contain a JSON array.")

        return records

    def collect_product_stats(self, spec: dict[str, Any]) -> dict[str, dict[str, Any]]:
        product_stats: dict[str, dict[str, Any]] = {}

        for path, path_item in spec["paths"].items():
            product_key = self.extract_product_key(path)
            api_version = self.extract_api_version(path)
            if product_key is None or api_version is None:
                continue

            stats = product_stats.setdefault(
                product_key,
                {
                    "api_versions": set(),
                    "paths": set(),
                    "path_prefixes": set(),
                    "operation_count": 0,
                },
            )
            stats["api_versions"].add(api_version)
            stats["paths"].add(path)
            stats["path_prefixes"].add(self.build_path_prefix(api_version, product_key))

            if isinstance(path_item, dict):
                stats["operation_count"] += sum(
                    1 for method in path_item if method.lower() in self.HTTP_METHODS
                )

        return product_stats

    def extract_api_version(self, path: str) -> str | None:
        segments = self.path_segments(path)
        if segments and segments[0] in {"v1", "v2"}:
            return segments[0]
        return None

    def extract_product_key(self, path: str) -> str | None:
        segments = self.path_segments(path)
        meaningful_segments = [
            segment for segment in segments if segment not in {"v1", "v2"} and not self.is_path_param(segment)
        ]
        if not meaningful_segments:
            return None
        return meaningful_segments[0]

    def path_segments(self, path: str) -> list[str]:
        return [segment for segment in path.strip("/").split("/") if segment]

    def is_path_param(self, segment: str) -> bool:
        return segment.startswith("{") and segment.endswith("}")

    def build_path_prefix(self, api_version: str, product_key: str) -> str:
        return f"/{api_version}/{product_key}"

    def build_api_product_record(
        self,
        api_spec: dict[str, Any],
        product_key: str,
        stats: dict[str, Any],
    ) -> dict[str, Any]:
        api_versions = sorted(stats["api_versions"])
        business_domain = self.map_business_domain(product_key)
        product_id = self.build_product_id(
            spec_id=api_spec["id"],
            product_key=product_key,
            api_versions=api_versions,
        )

        return {
            "id": product_id,
            "spec_id": api_spec["id"],
            "provider": api_spec.get("provider"),
            "product_key": product_key,
            "display_name": self.display_name(product_key),
            "business_domain": business_domain,
            "api_versions": api_versions,
            "path_prefixes": sorted(stats["path_prefixes"]),
            "path_count": len(stats["paths"]),
            "operation_count": stats["operation_count"],
            "schema_count": 0,
            "description": None,
            "is_test_product": product_key == "test_helpers",
            "tags": self.build_tags(product_key=product_key, business_domain=business_domain),
            "discovered_at": self.utc_now_iso(),
        }

    def build_product_id(self, spec_id: str, product_key: str, api_versions: list[str]) -> str:
        raw_id = f"{spec_id}:{product_key}:{','.join(api_versions)}"
        digest = hashlib.sha256(raw_id.encode("utf-8")).hexdigest()[:16]
        return f"api_product:{product_key}:{digest}"

    def map_business_domain(self, product_key: str) -> str:
        for domain, product_keys in self.DOMAIN_MAP.items():
            if product_key in product_keys:
                return domain
        return "other"

    def display_name(self, product_key: str) -> str:
        return product_key.replace("_", " ").title()

    def build_tags(self, product_key: str, business_domain: str) -> list[str]:
        tags = [business_domain]
        if product_key == "test_helpers":
            tags.append("test_only")
        return tags

    def write_api_products(self, products: list[dict[str, Any]]) -> None:
        self.api_products_path.parent.mkdir(parents=True, exist_ok=True)
        with self.api_products_path.open("w", encoding="utf-8") as target:
            json.dump(products, target, indent=2)
            target.write("\n")
