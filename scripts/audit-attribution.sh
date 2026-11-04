#!/usr/bin/env bash
# Audit commits against the policy the local commit-msg hook enforces: the
# sole-committer identity (author + committer), a Conventional Commits subject
# and no tool-generated meta-trailers.
#
# Usage: scripts/audit-attribution.sh [<git rev-list args>...]   (default: HEAD)
set -euo pipefail

ROOT="$(git rev-parse --show-toplevel)"
# shellcheck source=../.githooks/_policy.sh
source "$ROOT/.githooks/_policy.sh"

(($#)) || set -- HEAD

fail=0
count=0
while IFS= read -r sha; do
  count=$((count + 1))
  IFS=$'\x1f' read -r an ae cn ce <<<"$(git log -1 --format='%an%x1f%ae%x1f%cn%x1f%ce' "$sha")"
  short="${sha:0:12}"
  amg_check_identity "[$short] Author" "$an" "$ae" || fail=1
  amg_check_identity "[$short] Committer" "$cn" "$ce" || fail=1
  message="$(git log -1 --format='%B' "$sha")"
  amg_check_subject "$(amg_subject "$message")" || fail=1
  amg_check_trailers "$message" || {
    amg_err "[$short] message policy violation."
    fail=1
  }
done < <(git rev-list "$@")

if ((fail)); then
  amg_err "Attribution audit FAILED ($count commit(s) scanned)."
  exit 1
fi
amg_ok "Attribution audit passed ($count commit(s) scanned)."
