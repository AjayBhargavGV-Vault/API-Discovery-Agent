from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import yaml

from pathlib import Path
from typing import Any

from data.ingestion.embedding_generation import SearchDocumentEmbeddingGenerator
from data.ingestion.graph_expansion import GraphRagRetriever
from data.ingestion.graph_expansion import Neo4jGraphExpander
from data.ingestion.milvus_collection_setup import MilvusCollectionSetup
from data.ingestion.milvus_document_ingestion import MilvusDocumentIngestion
from data.ingestion.milvus_retrieval_test import DEFAULT_TEST_QUERIES
from data.ingestion.milvus_retrieval_test import MilvusRetrievalTest
from data.ingestion.neo4j_graph_builder import build_api_knowledge_graph
from scripts.operation_extraction import OperationExtraction
from scripts.product_area_discovery import ProductAreaDiscovery
from scripts.raw_spec_intake import RawSpecIntake
from scripts.relationship_mining import RelationshipMining
from scripts.schema_extraction import SchemaExtraction
from scripts.semantic_indexing import SemanticIndexing
from scripts.validate_semantic_neo4j_links import SemanticNeo4jLinkValidator


PROJECT_SRC = Path(__file__).resolve().parent
BACKEND_DIR = PROJECT_SRC.parent
DATA_DIR = PROJECT_SRC / "data"
SCRIPTS_DIR = PROJECT_SRC / "scripts"
INGESTION_DIR = DATA_DIR / "ingestion"

for import_path in (PROJECT_SRC, SCRIPTS_DIR, INGESTION_DIR):
    if str(import_path) not in sys.path:
        sys.path.append(str(import_path))




DEFAULT_SPEC_PATH = DATA_DIR / "raw_specs" / "openapi.spec3.yaml"
DEFAULT_ENV_PATH = BACKEND_DIR / ".env"
DEFAULT_PROVIDER = "stripe"


class ApiDiscoveryPipeline:
    def __init__(
        self,
        spec_path: Path = DEFAULT_SPEC_PATH,
        env_path: Path = DEFAULT_ENV_PATH,
        provider: str = DEFAULT_PROVIDER,
        logger: logging.Logger | None = None,
    ) -> None:
        self.spec_path = spec_path.resolve()
        self.env_path = env_path.resolve()
        self.provider = provider
        self.logger = logger or logging.getLogger(self.__class__.__name__)

    def run_local_modeling_pipeline(self) -> dict[str, Any]:
        self.logger.info("Running local API data modeling pipeline.")
        api_spec = self.run_raw_spec_intake()
        products = self.run_product_discovery()
        operations = self.run_operation_extraction()
        schemas, fields = self.run_schema_extraction()
        relationships = self.run_relationship_mining()
        search_documents = self.run_semantic_document_indexing()
        return {
            "api_spec_id": api_spec.get("id"),
            "products": len(products),
            "operations": len(operations),
            "schemas": len(schemas),
            "fields": len(fields),
            "relationships": len(relationships),
            "search_documents": len(search_documents),
        }

    def run_raw_spec_intake(self) -> dict[str, Any]:
        record = RawSpecIntake(
            spec_path=self.spec_path,
            provider=self.provider,
        ).ingest()
        self.logger.info("Raw spec intake complete: %s", record.get("id"))
        return record

    def run_product_discovery(self) -> list[dict[str, Any]]:
        products = ProductAreaDiscovery(spec_path=self.spec_path).discover()
        self.logger.info("Product discovery complete: %s products.", len(products))
        return products

    def run_operation_extraction(self) -> list[dict[str, Any]]:
        operations = OperationExtraction(spec_path=self.spec_path).extract()
        self.logger.info("Operation extraction complete: %s operations.", len(operations))
        return operations

    def run_schema_extraction(self) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        schemas, fields = SchemaExtraction(spec_path=self.spec_path).extract()
        self.logger.info(
            "Schema extraction complete: %s schemas, %s fields.",
            len(schemas),
            len(fields),
        )
        return schemas, fields

    def run_relationship_mining(self) -> list[dict[str, Any]]:
        relationships = RelationshipMining().mine()
        self.logger.info("Relationship mining complete: %s relationships.", len(relationships))
        return relationships

    def run_semantic_document_indexing(self) -> list[dict[str, Any]]:
        documents = SemanticIndexing(
            embedding_model=os.getenv("EMBEDDING_MODEL") or None,
        ).build()
        self.logger.info("Semantic document indexing complete: %s documents.", len(documents))
        return documents

    def run_neo4j_graph_ingestion(self, source_url: str | None = None) -> dict[str, Any]:
        self.logger.info("Building Neo4j knowledge graph.")
        spec = self.load_spec()
        raw_spec_url = source_url or str(self.spec_path)
        build_api_knowledge_graph(spec=spec, provider=self.provider, source_url=raw_spec_url)
        return {"neo4j_graph_ingestion": "completed", "source_url": raw_spec_url}

    def run_semantic_neo4j_validation(self) -> dict[str, Any]:
        validator = SemanticNeo4jLinkValidator(env_path=self.env_path)
        try:
            report = validator.validate()
        finally:
            validator.close()
        self.logger.info(
            "Semantic Neo4j validation complete: %s valid, %s invalid.",
            report.get("valid_count"),
            report.get("invalid_count"),
        )
        return report

    def run_embedding_generation(self, batch_size: int = 16) -> dict[str, int]:
        generator = SearchDocumentEmbeddingGenerator.from_env(
            env_path=self.env_path,
            batch_size=batch_size,
        )
        summary = generator.generate()
        self.logger.info("Embedding generation complete: %s", summary)
        return summary

    def run_milvus_collection_setup(self, recreate: bool = False) -> dict[str, Any]:
        setup = MilvusCollectionSetup.from_env(env_path=self.env_path)
        setup.setup_collection(recreate=recreate)
        return {
            "collection_name": setup.config.collection_name,
            "vector_dimension": setup.config.vector_dimension,
            "recreated": recreate,
        }

    def run_milvus_document_ingestion(
        self,
        batch_size: int = 250,
        replace_collection: bool = True,
    ) -> dict[str, int | None]:
        setup = MilvusCollectionSetup.from_env(env_path=self.env_path)
        ingestion = MilvusDocumentIngestion(
            collection_setup=setup,
            batch_size=batch_size,
            replace_collection=replace_collection,
        )
        summary = ingestion.ingest().to_dict()
        self.logger.info("Milvus ingestion complete: %s", summary)
        return summary

    def run_retrieval_test(self, top_k: int = 5, with_filters: bool = True) -> dict[str, Any]:
        filters: list[str | None] = [None]
        if with_filters:
            filters.extend(
                [
                    'entity_type == "operation"',
                    'business_domain == "payments"',
                    'method == "POST"',
                ]
            )
        retrieval_test = MilvusRetrievalTest.from_env(env_path=self.env_path, top_k=top_k)
        return retrieval_test.run(queries=DEFAULT_TEST_QUERIES, filters=filters)

    def run_graph_rag_retrieval(
        self,
        query: str,
        filter_expression: str | None = None,
        top_k: int = 5,
        max_graph_expansions: int | None = 3,
    ) -> dict[str, Any]:
        semantic_retriever = MilvusRetrievalTest.from_env(
            env_path=self.env_path,
            top_k=top_k,
        )
        graph_expander = Neo4jGraphExpander.from_env(env_path=self.env_path)
        try:
            retriever = GraphRagRetriever(
                semantic_retriever=semantic_retriever,
                graph_expander=graph_expander,
            )
            return retriever.retrieve(
                query=query,
                filter_expression=filter_expression,
                max_graph_expansions=max_graph_expansions,
            )
        finally:
            graph_expander.close()

    def answer_query(
        self,
        query: str,
        filter_expression: str | None = None,
        top_k: int = 5,
        max_graph_expansions: int | None = 3,
    ) -> dict[str, Any]:
        context = self.run_graph_rag_retrieval(
            query=query,
            filter_expression=filter_expression,
            top_k=top_k,
            max_graph_expansions=max_graph_expansions,
        )
        return {
            "query": query,
            "answer": ApiDiscoveryAnswerFormatter().format_answer(context),
            "context": context,
        }

    def load_spec(self) -> dict[str, Any]:
        with self.spec_path.open("r", encoding="utf-8") as source:
            spec = yaml.safe_load(source)
        if not isinstance(spec, dict):
            raise ValueError(f"{self.spec_path} did not parse as an OpenAPI object.")
        return spec


class ApiDiscoveryAnswerFormatter:
    def format_answer(self, context: dict[str, Any]) -> str:
        query = context.get("query") or "your query"
        operations = self.clean_rows(context.get("recommended_operations"))
        schemas = self.clean_rows(context.get("schemas"))
        fields = self.clean_rows(context.get("fields"))
        semantic_hits = self.clean_rows(context.get("semantic_hits"))
        confidence_notes = self.clean_values(context.get("confidence_notes"))

        if not semantic_hits:
            return f"I could not find a matching Stripe API for: {query}"

        primary_operation = self.first_operation(operations)
        if primary_operation:
            return self.format_operation_answer(
                query=query,
                operation=primary_operation,
                operations=operations,
                schemas=schemas,
                fields=fields,
                confidence_notes=confidence_notes,
            )

        primary_schema = self.first_schema(schemas)
        if primary_schema:
            return self.format_schema_answer(
                query=query,
                schema=primary_schema,
                operations=operations,
                fields=fields,
                confidence_notes=confidence_notes,
            )

        top_hit = semantic_hits[0]
        return "\n".join(
            self.non_empty(
                [
                    f"For: {query}",
                    f"Best match: {top_hit.get('title')}",
                    f"Type: {top_hit.get('entity_type')}",
                    f"Neo4j node: {top_hit.get('neo4j_node_id')}",
                    self.format_confidence(confidence_notes),
                ]
            )
        )

    def format_operation_answer(
        self,
        query: str,
        operation: dict[str, Any],
        operations: list[dict[str, Any]],
        schemas: list[dict[str, Any]],
        fields: list[dict[str, Any]],
        confidence_notes: list[str],
    ) -> str:
        related_operations = [
            row
            for row in operations[1:6]
            if row.get("method") and row.get("path")
        ]
        request_schemas = [schema for schema in schemas if schema.get("source") == "request_schema"]
        response_schemas = [schema for schema in schemas if schema.get("source") == "response_schema"]
        field_names = self.unique_values(
            field.get("name") or field.get("path") for field in fields[:20]
        )

        lines = [
            f"Answer for: {query}",
            "",
            "Recommended endpoint:",
            f"{operation.get('method') or ''} {operation.get('path') or ''}".strip(),
            f"Operation: {operation.get('title') or operation.get('operation_id')}",
            f"Summary: {operation.get('summary') or operation.get('title') or 'No summary available.'}",
        ]
        if operation.get("description"):
            lines.append(f"Description: {self.preview(operation.get('description'), 450)}")
        if request_schemas:
            lines.append(f"Request schema: {self.join_names(request_schemas)}")
        if response_schemas:
            lines.append(f"Response schema: {self.join_names(response_schemas)}")
        if field_names:
            lines.append(f"Important fields: {', '.join(field_names[:12])}")
        if related_operations:
            lines.extend(["", "Related operations:"])
            for related in related_operations:
                lines.append(
                    f"- {related.get('method') or ''} {related.get('path') or ''} "
                    f"({related.get('title') or related.get('operation_id')})"
                )
        confidence = self.format_confidence(confidence_notes)
        if confidence:
            lines.extend(["", confidence])
        return "\n".join(self.non_empty(lines))

    def format_schema_answer(
        self,
        query: str,
        schema: dict[str, Any],
        operations: list[dict[str, Any]],
        fields: list[dict[str, Any]],
        confidence_notes: list[str],
    ) -> str:
        lines = [
            f"Answer for: {query}",
            "",
            f"Best matching schema: {schema.get('title') or schema.get('name')}",
            f"Type: {schema.get('type') or 'unknown'}",
        ]
        if schema.get("description"):
            lines.append(f"Description: {self.preview(schema.get('description'), 450)}")
        field_names = self.unique_values(field.get("name") or field.get("path") for field in fields)
        if field_names:
            lines.append(f"Fields: {', '.join(field_names[:20])}")
        usable_operations = [row for row in operations if row.get("method") and row.get("path")]
        if usable_operations:
            lines.extend(["", "Operations connected to this schema:"])
            for operation in usable_operations[:6]:
                lines.append(
                    f"- {operation.get('method')} {operation.get('path')} "
                    f"({operation.get('title') or operation.get('operation_id')})"
                )
        confidence = self.format_confidence(confidence_notes)
        if confidence:
            lines.extend(["", confidence])
        return "\n".join(self.non_empty(lines))

    def first_operation(self, operations: list[dict[str, Any]]) -> dict[str, Any] | None:
        for operation in operations:
            if operation.get("method") and operation.get("path"):
                return operation
        return None

    def first_schema(self, schemas: list[dict[str, Any]]) -> dict[str, Any] | None:
        for schema in schemas:
            if schema.get("name") or schema.get("title"):
                return schema
        return None

    def format_confidence(self, confidence_notes: list[str]) -> str | None:
        if not confidence_notes:
            return None
        return "Confidence notes: " + " ".join(confidence_notes[:3])

    def join_names(self, rows: list[dict[str, Any]]) -> str:
        names = self.unique_values(row.get("name") or row.get("title") for row in rows)
        return ", ".join(names) if names else "none"

    def clean_rows(self, value: Any) -> list[dict[str, Any]]:
        return [row for row in value if isinstance(row, dict)] if isinstance(value, list) else []

    def clean_values(self, value: Any) -> list[str]:
        return [str(item) for item in value if item] if isinstance(value, list) else []

    def unique_values(self, values: Any) -> list[str]:
        seen: set[str] = set()
        unique = []
        for value in values:
            if value in (None, "", [], {}):
                continue
            text = str(value)
            if text not in seen:
                seen.add(text)
                unique.append(text)
        return unique

    def non_empty(self, values: list[Any]) -> list[str]:
        return [str(value) for value in values if value not in (None, "", [], {})]

    def preview(self, value: Any, limit: int = 500) -> str:
        text = "" if value is None else str(value)
        return text if len(text) <= limit else f"{text[:limit]}..."


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Orchestrate API Discovery Agent data pipelines.")
    parser.add_argument(
        "--stage",
        choices=[
            "local",
            "neo4j",
            "validate-neo4j",
            "embeddings",
            "milvus-setup",
            "milvus-ingest",
            "retrieval-test",
            "graph-rag",
            "ask",
            "chat",
            "all",
        ],
        default="local",
        help="Pipeline stage to run. 'local' prepares JSON/JSONL artifacts without external DB writes.",
    )
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC_PATH)
    parser.add_argument("--env", type=Path, default=DEFAULT_ENV_PATH)
    parser.add_argument("--provider", default=DEFAULT_PROVIDER)
    parser.add_argument("--source-url", default=None)
    parser.add_argument("--query", default="How do I confirm a Stripe payment?")
    parser.add_argument("--filter", default=None, help="Optional Milvus metadata filter.")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-graph-expansions", type=int, default=3)
    parser.add_argument(
        "--raw-json",
        action="store_true",
        help="For ask/chat, print the full retrieval context as JSON instead of a formatted answer.",
    )
    parser.add_argument("--embedding-batch-size", type=int, default=16)
    parser.add_argument("--milvus-batch-size", type=int, default=250)
    parser.add_argument(
        "--preserve-milvus-collection",
        action="store_true",
        help=(
            "For milvus-ingest, keep the existing collection and replace matching IDs only. "
            "By default the dedicated api_search_documents collection is recreated to avoid duplicates."
        ),
    )
    parser.add_argument(
        "--recreate-milvus",
        action="store_true",
        help="Drop and recreate the Milvus collection before setup. Destructive and explicit.",
    )
    return parser.parse_args()


def main() -> None:
    configure_logging()
    args = parse_args()
    if args.stage in {"ask", "chat"}:
        quiet_interactive_logs()
    pipeline = ApiDiscoveryPipeline(
        spec_path=args.spec,
        env_path=args.env,
        provider=args.provider,
    )

    if args.stage == "local":
        result = pipeline.run_local_modeling_pipeline()
    elif args.stage == "neo4j":
        result = pipeline.run_neo4j_graph_ingestion(source_url=args.source_url)
    elif args.stage == "validate-neo4j":
        result = pipeline.run_semantic_neo4j_validation()
    elif args.stage == "embeddings":
        result = pipeline.run_embedding_generation(batch_size=args.embedding_batch_size)
    elif args.stage == "milvus-setup":
        result = pipeline.run_milvus_collection_setup(recreate=args.recreate_milvus)
    elif args.stage == "milvus-ingest":
        result = pipeline.run_milvus_document_ingestion(
            batch_size=args.milvus_batch_size,
            replace_collection=not args.preserve_milvus_collection,
        )
    elif args.stage == "retrieval-test":
        result = pipeline.run_retrieval_test(top_k=args.top_k, with_filters=True)
    elif args.stage == "graph-rag":
        result = pipeline.run_graph_rag_retrieval(
            query=args.query,
            filter_expression=args.filter,
            top_k=args.top_k,
            max_graph_expansions=args.max_graph_expansions,
        )
    elif args.stage == "ask":
        result = pipeline.answer_query(
            query=args.query,
            filter_expression=args.filter,
            top_k=args.top_k,
            max_graph_expansions=args.max_graph_expansions,
        )
        if not args.raw_json:
            print(result["answer"])
            return
    elif args.stage == "chat":
        run_interactive_chat(pipeline=pipeline, args=args)
        return
    elif args.stage == "all":
        result = {
            "local": pipeline.run_local_modeling_pipeline(),
            "neo4j": pipeline.run_neo4j_graph_ingestion(source_url=args.source_url),
            "semantic_neo4j_validation": pipeline.run_semantic_neo4j_validation(),
            "embeddings": pipeline.run_embedding_generation(batch_size=args.embedding_batch_size),
            "milvus_setup": pipeline.run_milvus_collection_setup(recreate=args.recreate_milvus),
            "milvus_ingestion": pipeline.run_milvus_document_ingestion(
                batch_size=args.milvus_batch_size,
                replace_collection=not args.preserve_milvus_collection,
            ),
            "retrieval_test": pipeline.run_retrieval_test(top_k=args.top_k, with_filters=True),
        }
    else:
        raise ValueError(f"Unsupported stage: {args.stage}")

    print(json.dumps(result, indent=2))


def run_interactive_chat(pipeline: ApiDiscoveryPipeline, args: argparse.Namespace) -> None:
    print("API Discovery Agent")
    print("Ask a Stripe API question. Type 'exit' or 'quit' to stop.")
    while True:
        query = input("\n> ").strip()
        if query.lower() in {"exit", "quit"}:
            break
        if not query:
            continue
        result = pipeline.answer_query(
            query=query,
            filter_expression=args.filter,
            top_k=args.top_k,
            max_graph_expansions=args.max_graph_expansions,
        )
        if args.raw_json:
            print(json.dumps(result, indent=2))
        else:
            print(result["answer"])


def quiet_interactive_logs() -> None:
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    os.environ.setdefault("TQDM_DISABLE", "1")
    logging.getLogger().setLevel(logging.ERROR)
    for logger_name in (
        "neo4j",
        "neo4j.pool",
        "neo4j.notifications",
        "sentence_transformers",
        "sentence_transformers.base.model",
        "GraphRagRetriever",
        "Neo4jGraphExpander",
    ):
        logging.getLogger(logger_name).setLevel(logging.ERROR)


if __name__ == "__main__":
    main()
