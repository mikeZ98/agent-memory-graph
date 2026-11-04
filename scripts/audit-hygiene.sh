#!/usr/bin/env bash
# Audit the whole tracked tree (not just a diff) for cache/secret files, absolute
# machine paths and credential shapes. Same patterns as .githooks/pre-commit.
#
# Usage: scripts/audit-hygiene.sh
set -euo pipefail

ROOT="$(git rev-parse --show-toplevel)"
# shellcheck source=../.githooks/_policy.sh
source "$ROOT/.githooks/_policy.sh"
cd "$ROOT"

fail=0

bad_paths="$(git ls-files | grep -E "$AMG_FORBIDDEN_PATH_RE" | grep -vE "$AMG_ALLOWED_PATH_RE" || true)"
if [[ -n "$bad_paths" ]]; then
  amg_err "Tracked cache, virtualenv or secret files:"
  sed 's/^/    /' <<<"$bad_paths" >&2
  fail=1
fi

leaks="$(git grep -nIE "$AMG_PATH_LEAK_RE" -- . ':(exclude).githooks/_policy.sh' || true)"
if [[ -n "$leaks" ]]; then
  amg_err "Tracked content contains absolute machine paths:"
  sed 's/^/    /' <<<"$leaks" >&2
  fail=1
fi

secrets="$(git grep -lIE -e "$AMG_SECRET_RE" -- . ':(exclude).githooks/_policy.sh' || true)"
if [[ -n "$secrets" ]]; then
  amg_err "Tracked files look like they contain credentials (content redacted):"
  sed 's/^/    /' <<<"$secrets" >&2
  fail=1
fi

if ((fail)); then
  amg_err "Hygiene audit FAILED."
  exit 1
fi
amg_ok "Hygiene audit passed ($(git ls-files | wc -l | tr -d ' ') tracked file(s))."
