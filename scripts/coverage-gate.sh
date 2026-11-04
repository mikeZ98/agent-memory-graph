#!/usr/bin/env bash
# Enforce per-layer coverage floors (line + branch):
#   * domain layer (pure contracts):        100 %
#   * adapters + configuration (I/O edge):  >= 95 %
#
# Run after the test suite has produced coverage data (pytest runs with --cov);
# the overall floor is enforced by pytest-cov itself (fail_under in pyproject.toml).
# Usage (inside the project environment, e.g. direnv or `uv run`):
#   pytest && scripts/coverage-gate.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

gate() {
  local floor="$1" label="$2"
  shift 2
  local include
  include="$(IFS=,; echo "$*")"
  echo "== ${label} (fail under ${floor} %)"
  python -m coverage report --fail-under="$floor" --show-missing --include="$include"
}

gate 100 "domain" "src/agent_memory_graph/domain/*"
gate 95 "adapters + config" "src/agent_memory_graph/adapters/*" "src/agent_memory_graph/config.py"
