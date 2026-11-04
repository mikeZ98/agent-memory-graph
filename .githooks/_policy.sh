# shellcheck shell=bash
# Shared commit policy for git hooks and scripts/audit-*.sh:
#   * sole-committer identity (author + committer),
#   * Conventional Commits subject line,
#   * no tool-generated meta-trailers in commit messages,
#   * no cache/secret files, absolute machine paths or credentials in tracked content.
# Sourced, never executed. Git ignores this file as a hook (not a hook name).

readonly AMG_ALLOWED_NAME="mikeZ98"
readonly AMG_ALLOWED_EMAIL="145975972+mikeZ98@users.noreply.github.com"

readonly AMG_CONVENTIONAL_RE='^(feat|fix|chore|docs|test|build|ci|refactor|perf|style|revert)(\([a-z0-9._/-]+\))?!?: [^ ].{0,98}$'

# Case-insensitive patterns rejected anywhere in the message body.
readonly AMG_FORBIDDEN_TRAILER_RE='^(co-authored-by|generated-by|signed-off-by-bot):|generated with \[?[a-z ]+\]?\(|🤖'

# Tracked paths that must never be committed: caches, virtualenvs, secrets.
readonly AMG_FORBIDDEN_PATH_RE='(^|/)(\.cache|\.venv|__pycache__|\.mypy_cache|\.ruff_cache|\.pytest_cache|\.hypothesis)/|(^|/)\.env(\..+)?$|(^|/)\.envrc\.local$|\.(pem|key|p12|pfx)$'
readonly AMG_ALLOWED_PATH_RE='(^|/)\.env\.example$'

# Absolute machine-specific paths that must not appear in tracked content.
readonly AMG_PATH_LEAK_RE='/(Volumes|Users)/[A-Za-z0-9_]|/home/[a-z_][a-z0-9_-]*/|[A-Za-z]:\\Users\\'

# High-signal credential shapes (private keys, cloud/LLM/VCS tokens, inline DSN passwords).
readonly AMG_SECRET_RE='-----BEGIN ([A-Z]+ )?PRIVATE KEY-----|AKIA[0-9A-Z]{16}|sk-(ant-|proj-)?[A-Za-z0-9_-]{20,}|gh[pousr]_[A-Za-z0-9]{36,}|xox[baprs]-[A-Za-z0-9-]{10,}|AIza[0-9A-Za-z_-]{35}|(neo4j|bolt|neo4j\+s|bolt\+s|https?)://[^/[:space:]:@]+:[^/[:space:]@<>{}$]{3,}@'

amg_err() { printf '\033[31m✖ %s\033[0m\n' "$*" >&2; }
amg_ok() { printf '\033[32m✔ %s\033[0m\n' "$*" >&2; }

# amg_check_identity <role> <name> <email>
amg_check_identity() {
  local role="$1" name="$2" email="$3"
  if [[ "$name" != "$AMG_ALLOWED_NAME" || "$email" != "$AMG_ALLOWED_EMAIL" ]]; then
    amg_err "$role identity '$name <$email>' violates the sole-committer invariant."
    amg_err "Expected: '$AMG_ALLOWED_NAME <$AMG_ALLOWED_EMAIL>'."
    return 1
  fi
}

# amg_parse_ident <git-var-ident>  →  prints "name<TAB>email"
amg_parse_ident() {
  sed -E 's/^(.*) <([^>]*)> [0-9]+ [-+][0-9]{4}$/\1\t\2/' <<<"$1"
}

# amg_check_current_identity — validate the identity git will record right now.
amg_check_current_identity() {
  local a_name a_email c_name c_email fail=0
  IFS=$'\t' read -r a_name a_email <<<"$(amg_parse_ident "$(git var GIT_AUTHOR_IDENT)")"
  IFS=$'\t' read -r c_name c_email <<<"$(amg_parse_ident "$(git var GIT_COMMITTER_IDENT)")"
  amg_check_identity "Author" "$a_name" "$a_email" || fail=1
  amg_check_identity "Committer" "$c_name" "$c_email" || fail=1
  return "$fail"
}

# amg_clean_message — strip comments and anything below the scissors line.
amg_clean_message() {
  sed -e '/^# ------------------------ >8 ------------------------$/,$d' -e '/^#/d'
}

# amg_subject <message> — the first non-blank line, which is what gets checked.
amg_subject() {
  sed -n '/[^[:space:]]/{p;q;}' <<<"$1"
}

# amg_check_subject <first-line>
amg_check_subject() {
  local subject="$1"
  case "$subject" in
    Merge\ *|fixup!\ *|squash!\ *|amend!\ *) return 0 ;;
  esac
  if ! grep -qE "$AMG_CONVENTIONAL_RE" <<<"$subject"; then
    amg_err "Subject is not a Conventional Commit (max 100 chars): '$subject'"
    amg_err "Format: <type>(<scope>)?: <summary>   types: feat fix chore docs test build ci refactor perf style revert"
    return 1
  fi
}

# amg_check_trailers <full-message>
amg_check_trailers() {
  local hits
  hits="$(grep -inE "$AMG_FORBIDDEN_TRAILER_RE" <<<"$1" || true)"
  if [[ -n "$hits" ]]; then
    amg_err "Commit message contains tool-generated meta-trailers:"
    sed 's/^/    /' <<<"$hits" >&2
    return 1
  fi
}
