# agent-memory-graph

An asynchronous GraphRAG long-term memory engine for AI agents. It stores memories as
a knowledge graph in **Neo4j** and as vector embeddings in **Qdrant**, extracts structured
entities and relations with **instructor** on top of **Pydantic v2** contracts, and serves
them over **FastAPI**.

> Status: storage layer. Domain contracts, configuration and the async Neo4j / Qdrant
> adapters exist; the retrieval service and the API come next.

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

## Quickstart

```bash
direnv allow          # pins caches, venv and temp dirs under ./.cache/, wires git hooks
uv sync               # creates .cache/venv with runtime + dev dependencies

ruff check . && ruff format --check .
mypy                  # strict mode, configured in pyproject.toml
pytest                # hermetic: inet sockets blocked, no live services
scripts/coverage-gate.sh   # domain 100 %, adapters + config >= 95 % (line + branch)
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
| Layer cov.   | `scripts/coverage-gate.sh`          | 100 % on `domain/`, ≥ 95 % on `adapters/` + `config.py` (line + branch) |
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
