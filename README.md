# agent-memory-graph

An asynchronous GraphRAG long-term memory engine for AI agents. It stores memories as
a knowledge graph in **Neo4j** and as vector embeddings in **Qdrant**, extracts structured
entities and relations with **instructor** on top of **Pydantic v2** contracts, and serves
them over **FastAPI**.

> Status: domain layer. Schemas, retrieval contracts and the error taxonomy exist;
> storage adapters and the API come next.

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

## Quickstart

```bash
direnv allow          # pins caches, venv and temp dirs under ./.cache/, wires git hooks
uv sync               # creates .cache/venv with runtime + dev dependencies

ruff check . && ruff format --check .
mypy                  # strict mode, configured in pyproject.toml
pytest                # hermetic: inet sockets blocked, no live services
scripts/coverage-gate.sh   # 100 % line + branch coverage on the domain layer
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
| Domain cov.  | `scripts/coverage-gate.sh`          | 100 % line + branch coverage on `src/agent_memory_graph/domain/` |
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
tests/                    hermetic test suite (conftest scrubs service credentials)
.githooks/                git hooks + shared policy
scripts/                  audits used by the hooks, domain coverage gate
.envrc                    direnv: cache isolation under ./.cache/ and git wiring
```

## Contributing

This repository has a single committer. Commits must be authored and committed as
`mikeZ98 <145975972+mikeZ98@users.noreply.github.com>` and use
[Conventional Commits](https://www.conventionalcommits.org/) subjects. The hooks reject
anything else.
