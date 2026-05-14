from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer


DATA_DIR = Path(__file__).resolve().parents[1]
PROJECT_SRC = DATA_DIR.parent
BACKEND_DIR = PROJECT_SRC.parent
DEFAULT_ENV_PATH = BACKEND_DIR / ".env"
DEFAULT_INPUT_PATH = DATA_DIR / "api_search_documents.jsonl"
DEFAULT_OUTPUT_PATH = DATA_DIR / "api_search_documents_embedded.jsonl"


@dataclass(frozen=True)
class EmbeddingGenerationConfig:
    model_name: str
    vector_dimension: int
    batch_size: int = 16


class SearchDocumentEmbeddingGenerator:
    def __init__(
        self,
        config: EmbeddingGenerationConfig,
        input_path: Path = DEFAULT_INPUT_PATH,
        output_path: Path = DEFAULT_OUTPUT_PATH,
        logger: logging.Logger | None = None,
    ) -> None:
        self.config = config
        self.input_path = input_path.resolve()
        self.output_path = output_path.resolve()
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        self.model = SentenceTransformer(self.config.model_name)

    @classmethod
    def from_env(
        cls,
        env_path: Path = DEFAULT_ENV_PATH,
        input_path: Path = DEFAULT_INPUT_PATH,
        output_path: Path = DEFAULT_OUTPUT_PATH,
        batch_size: int = 16,
        logger: logging.Logger | None = None,
    ) -> SearchDocumentEmbeddingGenerator:
        load_dotenv(env_path)
        model_name = os.getenv("EMBEDDING_MODEL")
        vector_dimension = os.getenv("MILVUS_VECTOR_DIMENSION")
        if not model_name:
            raise ValueError("Missing required environment variable: EMBEDDING_MODEL")
        if not vector_dimension:
            raise ValueError("Missing required environment variable: MILVUS_VECTOR_DIMENSION")
        return cls(
            config=EmbeddingGenerationConfig(
                model_name=model_name,
                vector_dimension=int(vector_dimension),
                batch_size=batch_size,
            ),
            input_path=input_path,
            output_path=output_path,
            logger=logger,
        )

    def generate(self) -> dict[str, int]:
        documents = self.read_jsonl(self.input_path)
        total = len(documents)
        embedded = 0
        failed = 0

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with self.output_path.open("w", encoding="utf-8") as target:
            for batch_start in range(0, total, self.config.batch_size):
                batch = documents[batch_start : batch_start + self.config.batch_size]
                texts = [self.embedding_text(document) for document in batch]
                self.logger.info(
                    "Embedding records %s-%s of %s.",
                    batch_start + 1,
                    batch_start + len(batch),
                    total,
                )
                vectors = self.model.encode(
                    texts,
                    batch_size=self.config.batch_size,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                )

                for document, vector in zip(batch, vectors, strict=True):
                    vector_list = [float(value) for value in vector.tolist()]
                    if len(vector_list) != self.config.vector_dimension:
                        failed += 1
                        self.logger.error(
                            "Skipping %s because vector dimension is %s, expected %s.",
                            document.get("id"),
                            len(vector_list),
                            self.config.vector_dimension,
                        )
                        continue

                    document["embedding_model"] = self.config.model_name
                    document["embedding_vector"] = vector_list
                    document["embedding_status"] = "embedded"
                    target.write(json.dumps(document, separators=(",", ":")))
                    target.write("\n")
                    embedded += 1

        summary = {
            "total_records": total,
            "embedded_records": embedded,
            "failed_records": failed,
        }
        self.logger.info("Embedding generation summary: %s", summary)
        return summary

    def read_jsonl(self, path: Path) -> list[dict[str, Any]]:
        records = []
        with path.open("r", encoding="utf-8") as source:
            for line in source:
                stripped = line.strip()
                if stripped:
                    records.append(json.loads(stripped))
        return records

    def embedding_text(self, document: dict[str, Any]) -> str:
        title = document.get("title") or ""
        body = document.get("body") or ""
        keywords = ", ".join(document.get("keywords") or [])
        return f"{title}\n{body}\nKeywords: {keywords}".strip()


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )
