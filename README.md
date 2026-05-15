# API Discovery Agent

API Discovery Agent is a local, agentic Graph RAG system for discovering, explaining, and using APIs from an OpenAPI specification. It is designed for developer questions like:

- "Which endpoint creates a checkout session?"
- "What schema does this operation return?"
- "Show me the related lifecycle operations for a customer."
- "Generate a Python example for this API call."

The current project is wired around the Stripe OpenAPI spec at:

```text
application/backend/src/data/raw_specs/openapi.spec3.yaml
```

From that spec, the backend builds structured API artifacts, generates embeddings, loads semantic records into Milvus, creates a Neo4j knowledge graph, and serves a chat/search experience through FastAPI and a lightweight frontend.

## Highlights

- **OpenAPI ingestion**: extracts products, operations, schemas, fields, auth requirements, and relationships.
- **Semantic retrieval**: indexes API operation and schema documents with sentence-transformer embeddings.
- **Graph retrieval**: expands relevant operations through Neo4j relationships such as request schemas, response schemas, related operations, and resource lifecycles.
- **Agentic orchestration**: combines guardrails, query enrichment, semantic search, graph search, API spec lookup, code generation, validation, and answer synthesis.
- **Grounded answers**: responses are built from local API artifacts and validated against retrieved sources.
- **Developer UI**: browser interface for chat, search, sources, traces, and raw JSON responses.

## System Architecture

```text
OpenAPI Spec
    |
    v
Local Modeling Pipeline
    |-- api_specs.json
    |-- api_products.json
    |-- api_operations.json
    |-- api_schemas.json
    |-- api_schema_fields.json
    |-- api_relationships.json
    v
Semantic Documents
    |
    v
SentenceTransformer Embeddings ---> Milvus Vector Store
    |
    v
Neo4j Knowledge Graph
    |
    v
FastAPI Backend
    |-- /api/search
    |-- /api/chat
    v
Frontend Chat/Search UI
```

At runtime, user questions pass through safety checks, query enrichment, retrieval planning, semantic and graph lookup, optional code generation, answer synthesis, validation, and output guardrails.

## Repository Structure

```text
application/
  backend/
    requirements.txt
    src/
      main.py                         # CLI pipeline and terminal chat entry point
      api/
        app.py                        # FastAPI app setup
        routes/
          chat.py                     # Agentic chat endpoint
          search.py                   # Graph RAG search endpoint
      agents/                         # Guardrail, enrichment, retrieval, code, validation agents
      agent_tools/
        tool_registry.py              # Runtime tool registration and dispatch
      chat/
        chat_engine.py                # Full chat orchestration pipeline
        schemas.py                    # Request/response and agent schemas
        memory.py                     # Conversation memory
      data/
        raw_specs/                    # Source OpenAPI specs
        ingestion/                    # Milvus, Neo4j, embedding, and retrieval utilities
      llm/                            # vLLM, Ollama, and OpenAI-compatible clients
      prompts/                        # Prompt templates for agent behavior
      scripts/                        # Local OpenAPI extraction stages
    test-suite/                       # Backend tests
  frontend/
    index.html
    app.js
    styles.css
    server.mjs                       # Static frontend server
```

## Runtime Flow

The chat pipeline is implemented in `application/backend/src/chat/chat_engine.py`.

1. Receive a `ChatRequest`.
2. Run input guardrails.
3. Load conversation memory.
4. Enrich the user query for retrieval.
5. Create an agent execution plan.
6. Run semantic, graph, API spec, and code agents as needed.
7. Synthesize a draft answer.
8. Validate the draft against retrieved sources.
9. Run output guardrails.
10. Save conversation memory.
11. Return a grounded `ChatResponse`.

The simpler `/api/search` route runs hybrid semantic plus graph retrieval and returns recommended operations, schemas, confidence notes, and a formatted answer.

## Requirements

- Python 3.10+
- Node.js 18+
- Milvus or a Zilliz-compatible Milvus endpoint
- Neo4j
- A local or OpenAI-compatible LLM backend

Supported LLM backend modes in the code:

- `vllm`
- `ollama`
- `openai_compatible`

Python dependencies are listed in:

```text
application/backend/requirements.txt
```

## Backend Setup

From the repository root:

```powershell
cd application\backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Create this file:

```text
application/backend/.env
```

Example configuration:

```env
# LLM backend: vllm, ollama, or openai_compatible
LLM_BACKEND=vllm
VLLM_BASE_URL=http://localhost:8000/v1
OLLAMA_BASE_URL=http://localhost:11434
OPENAI_BASE_URL=http://localhost:1234/v1
OPENAI_API_KEY=

# Agent models
ORCHESTRATOR_MODEL=qwen2.5-32b-instruct
ORCHESTRATOR_FALLBACK_MODEL=qwen2.5-14b-instruct
QUERY_ENRICHMENT_MODEL=qwen2.5-14b-instruct
GUARDRAIL_MODEL=llama-guard-3-8b
VALIDATION_MODEL=qwen2.5-7b-instruct
VALIDATION_FALLBACK_MODEL=qwen2.5-14b-instruct
CODE_MODEL=deepseek-coder-v2-lite-instruct
CODE_FALLBACK_MODEL=qwen2.5-coder-14b-instruct
FALLBACK_MODEL=qwen2.5-7b-instruct

# Embeddings and Milvus
EMBEDDING_MODEL=BAAI/bge-large-en-v1.5
MILVUS_VECTOR_DIMENSION=1024
MILVUS_URI=http://localhost:19530
MILVUS_COLLECTION_NAME=api_search_documents
MILVUS_TOKEN=
MILVUS_USERNAME=
MILVUS_PASSWORD=
MILVUS_DATABASE=

# Neo4j
NEO4J_URI=neo4j://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=password
NEO4J_DATABASE=
NEO4J_VERIFY_CERTIFICATES=false
```

Update credentials, ports, model names, and vector dimension for your environment. For `BAAI/bge-large-en-v1.5`, the expected vector dimension is `1024`.

## Build The API Index

Run pipeline commands from:

```powershell
cd application\backend\src
```

Build local JSON and JSONL artifacts from the OpenAPI spec:

```powershell
python main.py --stage local
```

Load the API knowledge graph into Neo4j:

```powershell
python main.py --stage neo4j
```

Validate semantic document links against Neo4j:

```powershell
python main.py --stage validate-neo4j
```

Generate embeddings:

```powershell
python main.py --stage embeddings
```

Create the Milvus collection:

```powershell
python main.py --stage milvus-setup
```

Ingest embedded search documents into Milvus:

```powershell
python main.py --stage milvus-ingest
```

Run a retrieval smoke test:

```powershell
python main.py --stage retrieval-test
```

Run the full flow:

```powershell
python main.py --stage all
```

## Command-Line Usage

Ask a one-off question:

```powershell
python main.py --stage ask --query "How do I create a checkout session?"
```

Print raw retrieval context:

```powershell
python main.py --stage ask --query "How do I confirm a payment intent?" --raw-json
```

Start terminal chat:

```powershell
python main.py --stage chat
```

Useful options:

```text
--spec                     Path to an OpenAPI spec
--provider                 Provider name, default: stripe
--query                    Question for ask, chat, or graph-rag stages
--top-k                    Number of semantic matches to retrieve
--max-graph-expansions     Number of graph expansion hops
--filter                   Optional Milvus metadata filter
--raw-json                 Print full JSON context instead of formatted text
```

## Run The API Server

Run from:

```powershell
cd application\backend\src
```

The frontend is configured to call `http://127.0.0.1:8080`, so use port `8080` when running the UI:

```powershell
uvicorn api.app:app --host 127.0.0.1 --port 8080 --reload
```

Health checks:

```text
GET /api/health
GET /api/chat/health
```

Search:

```text
POST /api/search
```

```json
{
  "query": "How do I create a checkout session?",
  "filter": null,
  "max_graph_expansions": 3
}
```

Chat:

```text
POST /api/chat
```

```json
{
  "user_message": "Show me a Python example for creating a checkout session",
  "conversation_id": null,
  "chat_history": [],
  "top_k": 10,
  "include_agent_trace": true
}
```

## Run The Frontend

From:

```powershell
cd application\frontend
```

Install dependencies and start the static server:

```powershell
npm install
npm run dev
```

Open:

```text
http://127.0.0.1:5173
```

The frontend includes:

- Chat mode for the full agentic pipeline.
- Search mode for direct Graph RAG lookup.
- Source inspection.
- Agent trace inspection.
- Raw JSON response inspection.
- Conversation reset.

## Main Agents

| Agent | Responsibility |
| --- | --- |
| `GuardrailAgent` | Screens user input and generated output for safety issues. |
| `QueryEnrichmentAgent` | Rewrites vague questions into retrieval-optimized API queries. |
| `OrchestratorAgent` | Plans agent execution and synthesizes final answers. |
| `SemanticRetrieverAgent` | Searches Milvus-backed API documents. |
| `GraphRetrieverAgent` | Expands search results through Neo4j relationships. |
| `ApiSpecAgent` | Looks up operation and schema details from local artifacts. |
| `CodeExampleAgent` | Generates grounded curl, Python, Node.js, or JSON payload examples. |
| `ValidationAgent` | Checks generated answers against retrieved sources. |

## Registered Runtime Tools

`application/backend/src/agent_tools/tool_registry.py` registers the callable tools used by the agentic pipeline, including:

- `check_input_guardrails`
- `check_output_guardrails`
- `enrich_user_query`
- `semantic_api_search`
- `graph_search_by_query`
- `get_related_operations`
- `get_operation_schema_context`
- `get_resource_lifecycle`
- `get_operation_details`
- `get_schema_details`
- `generate_curl`
- `generate_python_example`
- `generate_node_example`
- `generate_sample_payload`
- `validate_response`

FastAPI routes and chat orchestration use these higher-level tools instead of calling model clients directly.

## Generated Data Artifacts

The local pipeline writes API catalog artifacts into `application/backend/src/data`, including:

| File | Purpose |
| --- | --- |
| `api_specs.json` | Source spec metadata. |
| `api_products.json` | Product or tag-level groupings. |
| `api_operations.json` | Operation metadata, paths, methods, summaries, and schema links. |
| `api_schemas.json` | Schema component metadata. |
| `api_schema_fields.json` | Flattened schema field details. |
| `api_relationships.json` | Mined relationships between operations, schemas, fields, and resources. |
| `api_search_documents.jsonl` | Searchable semantic documents before embeddings. |
| `api_search_documents_embedded.jsonl` | Search documents with embedding vectors. |

## Tests

Tests are located in:

```text
application/backend/test-suite
```

Install `pytest` if needed:

```powershell
pip install pytest
```

Run tests from `application/backend`:

```powershell
pytest test-suite
```

Some tests or runtime paths may require local data artifacts, Milvus, Neo4j, or an LLM backend depending on what is being exercised.

## Troubleshooting

**Frontend says the API is unavailable**

Make sure the backend is running on port `8080`:

```powershell
uvicorn api.app:app --host 127.0.0.1 --port 8080 --reload
```

**Milvus semantic search is unavailable**

Check `MILVUS_URI`, credentials, collection name, and whether `python main.py --stage milvus-ingest` completed successfully.

**Neo4j graph expansion fails**

Check `NEO4J_URI`, `NEO4J_USERNAME`, `NEO4J_PASSWORD`, and whether `python main.py --stage neo4j` completed successfully.

**Embedding dimension errors**

Make sure `MILVUS_VECTOR_DIMENSION` matches the embedding model. For `BAAI/bge-large-en-v1.5`, use `1024`.

**LLM calls fail**

Check `LLM_BACKEND`, the matching base URL, model names, and whether your local model server is running.

## Notes

- The default provider is `stripe`.
- The default spec path is `application/backend/src/data/raw_specs/openapi.spec3.yaml`.
- `.env`, virtual environments, Python caches, logs, and `node_modules` are ignored by git.
- Running `python application/backend/src/api/app.py` starts the backend on port `8000`; the included frontend expects port `8080`.
