#!/usr/bin/env bash
# Fail if any cache, virtualenv, build or test artifact exists outside ./.cache/.
#
# Run after the gates (lint, types, tests, coverage) to prove every tool honoured the
# cache redirection in .envrc / the CI env block. Tracked content is audited separately
# by scripts/audit-hygiene.sh.
#
# Usage: scripts/audit-cache-leaks.sh
set -euo pipefail

ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"

readonly ARTIFACT_RE='(^|/)(__pycache__|\.pytest_cache|\.mypy_cache|\.ruff_cache|\.hypothesis|\.venv|venv|htmlcov|build|dist|\.tox|\.nox|\.uv-cache)/?$|(^|/)\.coverage(\..+)?$|(^|/)coverage\.xml$|\.egg-info/?$|\.py[cod]$'

leaks="$(
  {
    git ls-files --others --ignored --exclude-standard --directory
    git ls-files --others --exclude-standard
  } | grep -v '^\.cache/' | grep -E "$ARTIFACT_RE" | sort -u || true
)"

if [[ -n "$leaks" ]]; then
  printf '\033[31m✖ artifacts outside ./.cache/:\033[0m\n' >&2
  sed 's/^/    /' <<<"$leaks" >&2
  exit 1
fi
printf '\033[32m✔ no caches or artifacts outside ./.cache/\033[0m\n' >&2
