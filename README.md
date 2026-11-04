# agent-memory-graph

An asynchronous GraphRAG long-term memory engine for AI agents. It stores memories as
a knowledge graph in **Neo4j** and as vector embeddings in **Qdrant**, extracts structured
entities and relations with **instructor** on top of **Pydantic v2** contracts, and serves
them over **FastAPI**.

> Status: service + HTTP API with guardrails. Ingestion, GraphRAG retrieval, the
> FastAPI endpoints, concurrency limits, circuit breakers and telemetry work end to end;
> deployment packaging comes next.

## Domain model

`agent_memory_graph.domain` is pure (no I/O, no driver imports) and every model is
immutable (`frozen=True`, `extra="forbid"`, tuples and read-only mappings all the way down).

| Module      | Contracts                                                                 |
| ----------- | ------------------------------------------------------------------------- |
| `models.py` | `EntityType`, `Entity`, `Relation` triplets, `KnowledgeGraph`, `make_entity_id` |
| `query.py`  | `MemoryQuery`, `GraphHop`, `GraphContext`                                 |
| `errors.py` | `MemoryGraphError` hierarchy, `MemoryProblemDetails` (RFC 9457), `PROBLEM_TAXONOMY` |

Invariants enforced at construction time:

- **Entity ids** are lowercase slugs (`tool-neo4j`) or canonical UUIDs (lower-cased).
  `make_entity_id(name, type)` derives them deterministically, with a UUIDv5 fallback for
  names that have no ASCII slug.
- **Names and text** are NFKC-normalised and stripped of invisible characters (controls,
  zero-width and bidi-override code points). Multiline text keeps markdown indentation.
- **Predicates** normalise to `SCREAMING_SNAKE_CASE` (`dependsOn` → `DEPENDS_ON`). They
  become Neo4j relationship types, which Cypher cannot parameterise, so the allowlist is
  also the injection boundary.
- **`KnowledgeGraph`** rejects duplicate entity ids, dangling relation endpoints and
  duplicate `(source, predicate, target)` triplets.
- **Scalars are strict**: no `"0.5"` → `0.5` or `True` → `1`; weights, scores and
  confidences are finite and in `[0, 1]`; `traversal_depth` is 1–3 and `top_k` 1–20.
- **Errors** map one-to-one onto a problem `type` URI, `title` and HTTP status:

  | Exception                     | `code`                         | Status |
  | ----------------------------- | ------------------------------ | ------ |
  | `MemoryGraphError`            | `AMG_INTERNAL_ERROR`           | 500    |
  | `EntityIntegrityError`        | `AMG_ENTITY_INTEGRITY`         | 422    |
  | `TraversalDepthExceededError` | `AMG_TRAVERSAL_DEPTH_EXCEEDED` | 422    |
  | `VectorSearchTimeoutError`    | `AMG_VECTOR_SEARCH_TIMEOUT`    | 504    |
  | `StoreUnavailableError`       | `AMG_STORE_UNAVAILABLE`        | 503    |
  | `StoreTimeoutError`           | `AMG_STORE_TIMEOUT`            | 504    |
  | `ExtractionFailedError`       | `AMG_EXTRACTION_FAILED`        | 502    |
  | `ExtractionTimeoutError`      | `AMG_EXTRACTION_TIMEOUT`       | 504    |
  | `InvalidInputError`           | `AMG_INVALID_INPUT`            | 422    |

## Storage adapters

`agent_memory_graph.adapters` holds the only code that talks to a database. Each adapter
owns one async client for its lifetime (`async with Store.from_settings(settings) as s:`,
or `connect()` / `close()`, both idempotent), and every driver failure leaves it as a
`MemoryGraphError` subclass with the original exception chained as `__cause__`.

**`Neo4jGraphStore`** (`neo4j.AsyncDriver`)

- `ensure_schema()` creates a uniqueness constraint on `Entity.id`.
- `ingest_graph(graph)` merges entities on `(id, type)` and relations on
  `(source)-[PREDICATE]->(target)` in **one** write transaction, `UNWIND`-batched, so a
  graph is either fully written or not at all. Nested `attributes` / `properties` are
  stored as canonical JSON strings, since Neo4j properties cannot hold maps.
- `traverse_neighborhood(seed_ids, depth=2)` follows relations outward for 1 to 3 hops and
  returns each reached relation once, as an immutable `GraphHop` at its shortest depth.
- Every value is a bound parameter. Relationship types and the `*1..N` bound cannot be
  parameterised in Cypher, so both are re-validated (canonical predicate allowlist, int in
  1..3) right before being spliced in. That check also blocks models built with
  `model_construct()`, which skips validation.

**`QdrantVectorStore`** (`qdrant_client.AsyncQdrantClient`, REST)

- `ensure_collection()` creates a Cosine collection of `embedding_dim` plus a keyword
  index on `entity_type`. It survives creation races and rejects an existing collection
  whose vector configuration differs.
- `upsert_entities(entities)` writes batched points with ids derived (UUIDv5) from entity
  ids and an `{entity_id, name, entity_type}` payload. Entities without embeddings are
  skipped, and dimensions are checked before anything is sent.
- `search_seeds(vector, top_k, score_threshold, allowed_types)` returns entity ids, best
  match first. Timeouts raise `VectorSearchTimeoutError`.
- The client is built with `check_compatibility=False`: the default probe makes a network
  call from a background thread at construction time.

| Driver failure                                     | Domain error                          |
| -------------------------------------------------- | ------------------------------------- |
| Unreachable / session expired / transient / 502-503 | `StoreUnavailableError` (503)        |
| Client deadline, server transaction timeout, 504   | `StoreTimeoutError` / `VectorSearchTimeoutError` (504) |
| Neo4j uniqueness constraint violated / relation count mismatch | `EntityIntegrityError` (422) |
| Rejected credentials / API key, anything else      | `MemoryGraphError` (500), no server message leaked |

## Extraction and embedding services

`agent_memory_graph.services` turns conversation text into memory. Both services depend
on narrow ports, so the LLM and embedding providers are chosen where the app is
assembled, not inside the library.

**`GraphExtractor.extract_triplets(text) -> KnowledgeGraph`** (`extract()` also returns an
`ExtractionReport` of what was pruned)

1. *Guard.* The text is NFKC-normalised and stripped of control, zero-width and
   bidi-override characters. Input over `max_input_chars` (32 000) raises `ValueError`
   instead of being truncated, and empty input returns an empty graph without calling
   the model. Look-alikes of the `<conversation>` delimiter are neutralised, and the
   system prompt treats the text as data, never as instructions.
2. *Extract.* A `StructuredLLM` returns a lenient `GraphExtraction` (entities by name,
   relations by endpoint name). `InstructorStructuredLLM` wraps any
   `instructor.AsyncInstructor`. instructor validates the tool call and re-asks the model
   on schema or JSON errors; exhausting the retries raises `ExtractionFailedError`.
3. *Convert.* Ids come from `make_entity_id(name, type)` and are never taken from the
   model. Predicates are normalised to `SCREAMING_SNAKE_CASE` and unknown entity types
   become `MISC`. Unprintable or oversized entities, invalid predicates, dangling
   endpoints, self-loops and duplicate triplets are pruned and counted, and the
   highest-confidence duplicate is kept. The result always satisfies `KnowledgeGraph`
   integrity.

```python
import instructor
from agent_memory_graph.services import GraphExtractor, InstructorStructuredLLM

llm = InstructorStructuredLLM(
    instructor.from_provider("<provider>/<model>", async_client=True), model="<model>"
)
graph = await GraphExtractor(llm).extract_triplets(conversation_text)
```

**`EmbeddingService`** (`dimension`, `embed_text`, `embed_entities`) embeds entities from
`"<name> (<type>): <description>"`. `MockEmbeddingService(settings.embedding_dim)` is
deterministic and dependency-free: signed feature hashing over word tokens, L2-normalised,
so texts that share words get similar vectors and seed search behaves plausibly in
hermetic tests.

## GraphRAG memory service

`GraphMemoryService` (`services/retrieval.py`) composes the stores, the extractor and
the embedder behind two narrow ports (`GraphStore`, `VectorStore`).

- **`ingest(text)`** extracts, then embeds (before anything is written), then writes
  the graph in one Neo4j transaction, then upserts the vectors. Neo4j and Qdrant can't
  share a transaction, so the graph is written first: a vector never points at a
  missing node. If the upsert then fails, the graph is complete but those entities
  aren't seedable yet. Every step is idempotent (deterministic ids, `MERGE`, upserts),
  so retrying the same ingest converges. Nothing is rolled back across stores, because
  a delete could destroy graph state that earlier conversations share.
- **`query(MemoryQuery)`** embeds the text, takes the best `top_k` seeds above
  `score_threshold` (optionally restricted to `entity_types`), traverses
  `traversal_depth` hops outward, and resolves the seeds and every path endpoint in one
  read. It returns a `GraphContext` whose `summary` is bounded, markdown-escaped
  markdown and whose `confidence` is the best seed's cosine similarity. It returns
  `GraphContext.empty()` when nothing clears the threshold.
- **`health()`** pings both stores concurrently and reports, rather than raises, failures.

## HTTP API

`create_app(service_factory)` builds the FastAPI app. Its lifespan starts the service
(connection pools plus idempotent schema/collection setup) and closes it on shutdown.
`build_memory_service(settings, llm=..., embedder=...)` composes the production service
and refuses an embedder whose dimension differs from `AMG_EMBEDDING_DIM`.

| Endpoint                | Body              | Success                              |
| ----------------------- | ----------------- | ------------------------------------ |
| `POST /v1/memory/ingest` | `{"text": str}`   | `200 {"graph": KnowledgeGraph, "report": ExtractionReport}` |
| `POST /v1/memory/query`  | `MemoryQuery`     | `200 GraphContext`                   |
| `GET /health`            |                   | `200` `{"status": "ok", "checks": {...}}`, or `503` with `"degraded"` |
| `GET /metrics`           |                   | `200` telemetry snapshot (JSON)      |

Every error is `application/problem+json` (RFC 9457):

- `MemoryGraphError` subclasses use their taxonomy entry, with `instance` defaulting to
  the request path.
- Request validation failures become `AMG_INVALID_INPUT` (422), listing each `loc`,
  `msg` and `type` but never the rejected value.
- Framework errors (404, 405, 413) use `type: about:blank`.
- Anything unexpected becomes a generic `AMG_INTERNAL_ERROR` (500), and the exception is
  logged, never returned.

Bodies over 1 MiB are refused with 413 before parsing: by `Content-Length`, or while
streaming when there is none.

```python
from agent_memory_graph.api import build_memory_service, create_app
from agent_memory_graph.config import Settings

settings = Settings()
app = create_app(lambda: build_memory_service(settings, llm=my_llm, embedder=my_embedder))
```

## Guardrails

`agent_memory_graph.guardrails` keeps one slow or hostile dependency from taking the
service down. `build_memory_service` wires all of it from settings.

| Guard | Behaviour |
| ----- | --------- |
| **Bounded queues** (`ConcurrencyLimiter`) | Each store allows at most `AMG_MAX_CONCURRENT_STORE_OPERATIONS` concurrent operations, and extraction allows `AMG_MAX_CONCURRENT_EXTRACTIONS`. A caller that can't get a permit within `AMG_QUEUE_TIMEOUT_SECONDS` gets `AMG_STORE_TIMEOUT` (`operation: "queue"`) or `AMG_EXTRACTION_TIMEOUT` (504) instead of piling up. |
| **Circuit breakers** (`CircuitBreaker`) | A store's breaker opens after `AMG_BREAKER_FAILURE_THRESHOLD` consecutive *store* failures (unavailable or timed out; rejected requests don't count). While open, calls fail fast with 503 and `Retry-After` and never touch the store or the queue. After `AMG_BREAKER_RESET_SECONDS` one probe is let through; success closes the breaker, failure reopens it. Queue timeouts never trip a breaker, since they signal local overload, not store health. |
| **Input guard** (`InputGuard`) | Text is screened before extraction. Forged structure (chat-template tokens such as `<\|im_start\|>`, `System:` role headers, fake `<conversation>`, `[INST]` and `<<SYS>>` delimiters) is defanged in place. Instruction-override phrasing is flagged but kept. `AMG_INPUT_GUARD_MODE=reject` refuses any finding with 422, without echoing the text. |
| **Traversal guard** (`TraversalGuard`) | Traversal results are bounded before synthesis: at most 500 edges per seed, 100 out-edges per node (hubs) and 100 cycle-closing edges. Edges are kept shallowest and strongest first, and anything reachable only through a pruned edge is dropped, so every path stays connected to a seed. |
| **Adaptive fallback** (`RetrievalPolicy`) | If the requested `score_threshold` finds no seeds, it's retried at ×0.8 per step, for up to `AMG_RETRIEVAL_FALLBACK_STEPS` extra searches and never below `AMG_RETRIEVAL_FALLBACK_FLOOR`. `confidence` then reflects the weaker match. |
| **Telemetry** (`Telemetry`) | Request and failure counters, latency (count, sum, min, max, mean, p50/p95/p99, cumulative buckets), entities and relations indexed, pruned candidates and hops, suspicious inputs, fallbacks, and live queue occupancy and circuit states, served at `GET /metrics`. |

## Configuration

Settings are read from `AMG_*` environment variables only; `.env` files are never loaded.

| Variable                       | Default                  | Notes                                  |
| ------------------------------ | ------------------------ | -------------------------------------- |
| `AMG_NEO4J_URI`                | `neo4j://localhost:7687` | `neo4j`, `bolt` (`+s`, `+ssc`); no credentials in the URI |
| `AMG_NEO4J_USER`               | `neo4j`                  |                                        |
| `AMG_NEO4J_PASSWORD`           | **required**             | secret                                 |
| `AMG_NEO4J_DATABASE`           | server home database     |                                        |
| `AMG_NEO4J_MAX_POOL_SIZE`      | `50`                     |                                        |
| `AMG_QDRANT_URL`               | `http://localhost:6333`  | `http` / `https`                       |
| `AMG_QDRANT_API_KEY`           | none                     | secret; requires `https` unless the host is loopback |
| `AMG_QDRANT_COLLECTION`        | `amg_entities`           |                                        |
| `AMG_QDRANT_UPSERT_BATCH_SIZE` | `256`                    |                                        |
| `AMG_EMBEDDING_DIM`            | `384`                    | 1 to 8192                              |
| `AMG_DEFAULT_TIMEOUT_SECONDS`  | `10.0`                   | per operation, client and server side  |
| `AMG_MAX_CONCURRENT_EXTRACTIONS` | `4`                    | concurrent LLM extraction calls        |
| `AMG_MAX_CONCURRENT_STORE_OPERATIONS` | `32`              | per store                              |
| `AMG_QUEUE_TIMEOUT_SECONDS`    | `2.0`                    | max wait for a permit                  |
| `AMG_BREAKER_FAILURE_THRESHOLD` | `5`                     | consecutive store failures to trip     |
| `AMG_BREAKER_RESET_SECONDS`    | `30.0`                   | open time before a half-open probe     |
| `AMG_INPUT_GUARD_MODE`         | `flag`                   | `flag` or `reject`                     |
| `AMG_RETRIEVAL_FALLBACK_STEPS` | `3`                      | 0 disables the fallback                |
| `AMG_RETRIEVAL_FALLBACK_FLOOR` | `0.1`                    | lowest threshold a fallback may use    |

## Quickstart

```bash
direnv allow          # pins caches, venv and temp dirs under ./.cache/, wires git hooks
uv sync               # creates .cache/venv with runtime + dev dependencies

ruff check . && ruff format --check .
mypy                  # strict mode, configured in pyproject.toml
pytest                # hermetic: inet sockets blocked, no live services
scripts/coverage-gate.sh   # domain 100 %; adapters, config, services, api, guardrails >= 95 %
bandit -c pyproject.toml -r src
```

Without direnv, run `git config core.hooksPath .githooks` once and prefix the tool
commands with `uv run`.

## Quality gates

| Gate         | Tool / location                     | Enforces                                                        |
| ------------ | ----------------------------------- | --------------------------------------------------------------- |
| Lint/format  | `ruff` (line length 100)            | Broad strict rule set, including annotations and docstrings     |
| Types        | `mypy --strict` + pydantic plugin   | Fully typed `src/` and `tests/`                                 |
| Tests        | `pytest`, `pytest-asyncio`, `pytest-socket` | Network-free runs, warnings as errors, ≥ 90 % branch coverage overall |
| Layer cov.   | `scripts/coverage-gate.sh`          | 100 % on `domain/`, ≥ 95 % on `adapters/`, `config.py`, `services/`, `api/`, `guardrails/` (line + branch) |
| Security     | `bandit`                            | Static security scan of `src/`                                  |
| `pre-commit` | `.githooks/pre-commit`              | Committer identity, no cache/secret files, no machine paths or credentials, ruff on staged files |
| `commit-msg` | `.githooks/commit-msg`              | Conventional Commits subject, no tool-generated trailers, identity |
| `pre-push`   | `.githooks/pre-push`                | Attribution audit of outgoing commits and a full-tree hygiene audit |

All hook patterns are defined once in `.githooks/_policy.sh`. You can run the audits
on demand with `scripts/audit-attribution.sh [<rev-list args>]` and
`scripts/audit-hygiene.sh`.

## Repository layout

```text
src/agent_memory_graph/   package (typed: ships py.typed)
  domain/                 pure schemas, retrieval contracts, error taxonomy
  adapters/               async Neo4j graph store and Qdrant vector store
  services/               extraction (instructor), embeddings, GraphRAG retrieval
  api/                    FastAPI app, endpoints, RFC 9457 handlers, body limit
  guardrails/             limiters, circuit breakers, input/traversal guards, telemetry
  config.py               AMG_* settings
tests/                    hermetic test suite (conftest scrubs service credentials)
.githooks/                git hooks + shared policy
scripts/                  audits used by the hooks, per-layer coverage gate
.envrc                    direnv: cache isolation under ./.cache/ and git wiring
```

## Contributing

This repository has a single committer. Commits must be authored and committed as
`mikeZ98 <145975972+mikeZ98@users.noreply.github.com>` and use
[Conventional Commits](https://www.conventionalcommits.org/) subjects. The hooks reject
anything else.
