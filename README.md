# agent-memory-graph

An asynchronous GraphRAG long-term memory engine for AI agents. It stores memories as
a knowledge graph in **Neo4j** and as vector embeddings in **Qdrant**, extracts structured
entities and relations with **instructor** on top of **Pydantic v2** contracts, and serves
them over **FastAPI**.

> Status: repository bootstrap. Only the toolchain and quality gates exist so far.

## Quickstart

```bash
direnv allow          # pins caches, venv and temp dirs under ./.cache/, wires git hooks
uv sync               # creates .cache/venv with runtime + dev dependencies

ruff check . && ruff format --check .
mypy                  # strict mode, configured in pyproject.toml
pytest                # hermetic: inet sockets blocked, no live services
bandit -c pyproject.toml -r src
```

Without direnv, run `git config core.hooksPath .githooks` once and prefix the tool
commands with `uv run`.

## Quality gates

| Gate         | Tool / location                     | Enforces                                                        |
| ------------ | ----------------------------------- | --------------------------------------------------------------- |
| Lint/format  | `ruff` (line length 100)            | Broad strict rule set, including annotations and docstrings     |
| Types        | `mypy --strict` + pydantic plugin   | Fully typed `src/` and `tests/`                                 |
| Tests        | `pytest`, `pytest-asyncio`, `pytest-socket` | Network-free runs, warnings as errors, ≥ 90 % branch coverage |
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
tests/                    hermetic test suite (conftest scrubs service credentials)
.githooks/                git hooks + shared policy
scripts/                  on-demand audits used by the hooks
.envrc                    direnv: cache isolation under ./.cache/ and git wiring
```

## Contributing

This repository has a single committer. Commits must be authored and committed as
`mikeZ98 <145975972+mikeZ98@users.noreply.github.com>` and use
[Conventional Commits](https://www.conventionalcommits.org/) subjects. The hooks reject
anything else.
