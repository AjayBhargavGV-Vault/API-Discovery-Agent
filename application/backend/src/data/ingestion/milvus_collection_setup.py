from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
from pymilvus import DataType
from pymilvus import MilvusClient
from pymilvus.exceptions import MilvusException


DATA_DIR = Path(__file__).resolve().parents[1]
PROJECT_SRC = DATA_DIR.parent
BACKEND_DIR = PROJECT_SRC.parent
DEFAULT_ENV_PATH = BACKEND_DIR / ".env"
DEFAULT_COLLECTION_NAME = "api_search_documents"


@dataclass(frozen=True)
class MilvusCollectionConfig:
    uri: str
    vector_dimension: int
    collection_name: str = DEFAULT_COLLECTION_NAME
    token: str | None = None
    username: str | None = None
    password: str | None = None
    database_name: str | None = None


class MilvusCollectionSetup:
    VARCHAR_MAX_LENGTHS = {
        "id": 512,
        "entity_type": 64,
        "entity_id": 512,
        "neo4j_node_id": 512,
        "title": 4096,
        "body": 65535,
        "provider": 128,
        "spec_id": 512,
        "spec_version": 128,
        "business_domain": 128,
        "product_key": 256,
        "method": 32,
        "path": 4096,
        "embedding_model": 256,
        "indexed_at": 128,
    }

    def __init__(
        self,
        config: MilvusCollectionConfig,
        logger: logging.Logger | None = None,
    ) -> None:
        self.config = config
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        self.client = self.create_client()

    @classmethod
    def from_env(
        cls,
        env_path: Path = DEFAULT_ENV_PATH,
        collection_name: str | None = None,
        logger: logging.Logger | None = None,
    ) -> MilvusCollectionSetup:
        load_dotenv(env_path)
        config = MilvusCollectionConfig(
            uri=cls.required_env("MILVUS_URI"),
            token=os.getenv("MILVUS_TOKEN") or None,
            username=os.getenv("MILVUS_USERNAME") or os.getenv("MILVUS_USER") or None,
            password=os.getenv("MILVUS_PASSWORD") or None,
            database_name=os.getenv("MILVUS_DATABASE") or os.getenv("MILVUS_DB_NAME") or None,
            vector_dimension=cls.required_int_env("MILVUS_VECTOR_DIMENSION"),
            collection_name=collection_name or os.getenv("MILVUS_COLLECTION_NAME") or DEFAULT_COLLECTION_NAME,
        )
        return cls(config=config, logger=logger)

    @staticmethod
    def required_env(key: str) -> str:
        value = os.getenv(key)
        if not value:
            raise ValueError(f"Missing required environment variable: {key}")
        return value

    @staticmethod
    def required_int_env(key: str) -> int:
        value = MilvusCollectionSetup.required_env(key)
        try:
            parsed = int(value)
        except ValueError as exc:
            raise ValueError(f"{key} must be an integer.") from exc
        if parsed <= 0:
            raise ValueError(f"{key} must be greater than zero.")
        return parsed

    def create_client(self) -> MilvusClient:
        kwargs = {"uri": self.config.uri}
        if self.config.token:
            kwargs["token"] = self.config.token
        if self.config.username and self.config.password:
            kwargs["user"] = self.config.username
            kwargs["password"] = self.config.password
        if self.config.database_name:
            kwargs["db_name"] = self.config.database_name

        self.logger.info("Connecting to Milvus.")
        return MilvusClient(**kwargs)

    def setup_collection(self, recreate: bool = False) -> None:
        collection_name = self.config.collection_name
        if recreate:
            self.drop_collection_if_exists()

        if self.client.has_collection(collection_name):
            self.logger.info("Milvus collection already exists: %s", collection_name)
            return

        schema = self.build_schema()
        self.logger.info(
            "Creating Milvus collection %s with vector dimension %s.",
            collection_name,
            self.config.vector_dimension,
        )
        try:
            self.create_collection_with_index(schema=schema, index_type="AUTOINDEX")
        except MilvusException as exc:
            self.logger.warning(
                "AUTOINDEX unavailable for %s; falling back to HNSW. Error: %s",
                collection_name,
                exc,
            )
            self.create_collection_with_index(schema=schema, index_type="HNSW")

    def drop_collection_if_exists(self) -> None:
        collection_name = self.config.collection_name
        if self.client.has_collection(collection_name):
            self.logger.warning("Dropping Milvus collection: %s", collection_name)
            self.client.drop_collection(collection_name)

    def build_schema(self):
        schema = MilvusClient.create_schema(
            auto_id=False,
            enable_dynamic_field=False,
            description="Graph RAG semantic documents for API discovery.",
        )
        schema.add_field(
            field_name="id",
            datatype=DataType.VARCHAR,
            is_primary=True,
            max_length=self.VARCHAR_MAX_LENGTHS["id"],
        )
        schema.add_field(
            field_name="vector",
            datatype=DataType.FLOAT_VECTOR,
            dim=self.config.vector_dimension,
        )
        for field_name in self.VARCHAR_MAX_LENGTHS:
            if field_name == "id":
                continue
            schema.add_field(
                field_name=field_name,
                datatype=DataType.VARCHAR,
                max_length=self.VARCHAR_MAX_LENGTHS[field_name],
            )
        return schema

    def create_collection_with_index(self, schema, index_type: str) -> None:
        index_params = self.build_index_params(index_type=index_type)
        self.client.create_collection(
            collection_name=self.config.collection_name,
            schema=schema,
            index_params=index_params,
        )
        self.logger.info(
            "Created Milvus collection %s with %s index.",
            self.config.collection_name,
            index_type,
        )

    def build_index_params(self, index_type: str):
        index_params = self.client.prepare_index_params()
        params = {} if index_type == "AUTOINDEX" else {"M": 16, "efConstruction": 200}
        index_params.add_index(
            field_name="vector",
            index_type=index_type,
            metric_type="COSINE",
            params=params,
        )
        return index_params


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )
