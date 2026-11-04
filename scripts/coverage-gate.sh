#!/usr/bin/env bash
# Enforce 100% line + branch coverage on the pure domain layer.
#
# Run after the test suite has produced coverage data (pytest runs with --cov);
# the overall floor is enforced by pytest-cov itself (fail_under in pyproject.toml).
# Usage (inside the project environment, e.g. direnv or `uv run`):
#   pytest && scripts/coverage-gate.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

readonly CRITICAL=(
  "src/agent_memory_graph/domain/*"
)

include="$(IFS=,; echo "${CRITICAL[*]}")"
python -m coverage report --fail-under=100 --show-missing --include="$include"
