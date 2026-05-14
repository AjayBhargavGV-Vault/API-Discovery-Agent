from __future__ import annotations

import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

PROJECT_SRC = Path(__file__).resolve().parents[1]
BACKEND_DIR = PROJECT_SRC.parent
DEFAULT_ENV_PATH = BACKEND_DIR / ".env"

if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from data.ingestion.milvus_retrieval_test import MilvusRetrievalTest  # noqa: E402
from api.routes.search import router as search_router  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)
_log = logging.getLogger("api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    _log.info("Loading embedding model and connecting to Milvus…")
    app.state.env_path = DEFAULT_ENV_PATH
    app.state.semantic_retriever = MilvusRetrievalTest.from_env(
        env_path=DEFAULT_ENV_PATH,
        top_k=10,
    )
    _log.info("API ready.")
    yield


app = FastAPI(
    title="API Discovery Agent",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:4173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(search_router, prefix="/api")


if __name__ == "__main__":
    uvicorn.run("api.app:app", host="0.0.0.0", port=8000, reload=True)
