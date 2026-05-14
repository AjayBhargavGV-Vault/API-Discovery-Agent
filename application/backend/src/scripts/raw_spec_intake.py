from __future__ import annotations

import argparse
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


class RawSpecIntake:
    def __init__(
        self,
        spec_path: Path = DEFAULT_SPEC_PATH,
        api_specs_path: Path = DEFAULT_API_SPECS_PATH,
        provider: str = "stripe",
    ) -> None:
        self.spec_path = spec_path.resolve()
        self.api_specs_path = api_specs_path.resolve()
        self.provider = provider

    def ingest(self) -> dict[str, Any]:
        source_sha256 = self.sha256_file(self.spec_path)
        spec = self.load_openapi_spec(self.spec_path)
        record = self.build_api_spec_record(spec=spec, source_sha256=source_sha256)
        self.upsert_api_spec_record(record)
        return record

    def utc_now_iso(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def sha256_file(self, path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def load_openapi_spec(self, path: Path) -> dict[str, Any]:
        with path.open("r", encoding="utf-8") as source:
            spec = yaml.safe_load(source)

        if not isinstance(spec, dict):
            raise ValueError(f"{path} did not parse as an OpenAPI object.")

        for field in ("info", "servers", "paths", "components"):
            if field not in spec:
                raise ValueError(f"{path} is missing required top-level field: {field}")

        return spec

    def build_api_spec_record(
        self,
        spec: dict[str, Any],
        source_sha256: str,
    ) -> dict[str, Any]:
        info = spec.get("info", {})
        contact = info.get("contact", {})
        servers = spec.get("servers", [])
        paths = spec.get("paths", {})
        components = spec.get("components", {})
        schemas = components.get("schemas", {})
        security_schemes = components.get("securitySchemes", {})

        api_version = info.get("version", "unknown")

        return {
            "id": f"{self.provider}:{api_version}:{source_sha256[:16]}",
            "provider": self.provider,
            "title": info.get("title"),
            "description": info.get("description"),
            "openapi_version": spec.get("openapi"),
            "api_version": api_version,
            "release_phase": info.get("x-stripeReleasePhase"),
            "audience": info.get("x-stripeSpecAudience"),
            "spec_filename": info.get("x-stripeSpecFilename"),
            "terms_of_service_url": info.get("termsOfService"),
            "contact_name": contact.get("name"),
            "contact_url": contact.get("url"),
            "contact_email": contact.get("email"),
            "server_urls": [
                server.get("url") for server in servers if isinstance(server, dict)
            ],
            "source_path": str(self.spec_path),
            "source_sha256": source_sha256,
            "raw_spec_path": str(self.spec_path),
            "path_count": len(paths) if isinstance(paths, dict) else 0,
            "component_schema_count": len(schemas) if isinstance(schemas, dict) else 0,
            "security_scheme_names": list(security_schemes.keys())
            if isinstance(security_schemes, dict)
            else [],
            "top_level_fields": sorted(spec.keys()),
            "ingested_at": self.utc_now_iso(),
        }

    def read_api_specs(self) -> list[dict[str, Any]]:
        if not self.api_specs_path.exists():
            return []

        with self.api_specs_path.open("r", encoding="utf-8") as source:
            records = json.load(source)

        if not isinstance(records, list):
            raise ValueError(f"{self.api_specs_path} must contain a JSON array.")

        return records

    def upsert_api_spec_record(self, record: dict[str, Any]) -> None:
        records = self.read_api_specs()
        records = [item for item in records if item.get("id") != record["id"]]
        records.append(record)
        records.sort(key=lambda item: (item.get("provider", ""), item.get("api_version", "")))

        self.api_specs_path.parent.mkdir(parents=True, exist_ok=True)
        with self.api_specs_path.open("w", encoding="utf-8") as target:
            json.dump(records, target, indent=2)
            target.write("\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create an api_specs record from a raw OpenAPI spec.")
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC_PATH)
    parser.add_argument("--api-specs", type=Path, default=DEFAULT_API_SPECS_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    intake = RawSpecIntake(
        spec_path=args.spec,
        api_specs_path=args.api_specs,
    )
    record = intake.ingest()
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
