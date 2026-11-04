#!/usr/bin/env bash
# Container smoke verification.
#
#   scripts/smoke-docker.sh --offline   policy + hadolint (if installed) + compose config;
#                                       never touches a Docker daemon or the network
#   scripts/smoke-docker.sh             offline checks, then (only if a daemon answers)
#                                       build the image and inspect it from the inside
#   scripts/smoke-docker.sh --live      all of the above, then boot the full Compose stack,
#                                       probe /health, SIGTERM the API and assert exit 0
#
# Docker CLI state stays under ./.cache/docker (DOCKER_CONFIG), like every other cache.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

MODE="auto"
case "${1:-}" in
  --offline) MODE="offline" ;;
  --live) MODE="live" ;;
  "") ;;
  -h | --help)
    sed -n '2,12p' "$0"
    exit 0
    ;;
  *)
    echo "usage: $0 [--offline|--live]" >&2
    exit 2
    ;;
esac

readonly IMAGE="agent-memory-graph:smoke"
readonly HADOLINT_IMAGE="hadolint/hadolint:v2.14.0@sha256:27086352fd5e1907ea2b934eb1023f217c5ae087992eb59fde121dce9c9ff21e"
readonly PROJECT="amg-smoke"

export DOCKER_CONFIG="${DOCKER_CONFIG:-$ROOT/.cache/docker}"
mkdir -p "$DOCKER_CONFIG"
if [[ ! -e "$DOCKER_CONFIG/cli-plugins" && -d "$HOME/.docker/cli-plugins" ]]; then
  ln -s "$HOME/.docker/cli-plugins" "$DOCKER_CONFIG/cli-plugins"
fi
PYTHON="${UV_PROJECT_ENVIRONMENT:-$ROOT/.cache/venv}/bin/python"
[[ -x "$PYTHON" ]] || PYTHON="python3"

ok() { printf '\033[32m✔ %s\033[0m\n' "$*"; }
skip() { printf '\033[33m- %s\033[0m\n' "$*"; }
die() {
  printf '\033[31m✖ %s\033[0m\n' "$*" >&2
  exit 1
}
step() { printf '\n== %s\n' "$*"; }

have_docker_cli() { command -v docker >/dev/null 2>&1; }
have_compose() { have_docker_cli && docker compose version >/dev/null 2>&1; }
have_daemon() { have_docker_cli && docker info >/dev/null 2>&1; }

# ---------------------------------------------------------------------------
# Offline: static guarantees
# ---------------------------------------------------------------------------
step "policy (Dockerfile, .dockerignore, docker-compose.yml)"
"$PYTHON" scripts/container_policy.py Dockerfile .dockerignore docker-compose.yml ||
  die "container policy violations"
ok "container policy: non-root 10001:10001, digest pins, exec form, no package managers, read-only stack"

step "hadolint"
if command -v hadolint >/dev/null 2>&1; then
  hadolint --failure-threshold warning Dockerfile || die "hadolint findings"
  ok "hadolint (local binary)"
elif [[ "$MODE" != "offline" ]] && have_daemon; then
  docker run --rm -i --network none "$HADOLINT_IMAGE" hadolint --failure-threshold warning - <Dockerfile ||
    die "hadolint findings"
  ok "hadolint (pinned container)"
else
  skip "hadolint not installed (policy check covers DL3000/2/4/6/7/15/20/25/42)"
fi

step "compose configuration"
if have_compose; then
  AMG_NEO4J_PASSWORD="offline-placeholder" docker compose config --quiet || die "compose config invalid"
  if env -u AMG_NEO4J_PASSWORD docker compose config --quiet 2>/dev/null; then
    die "compose accepted a stack without AMG_NEO4J_PASSWORD"
  fi
  ok "compose config valid; AMG_NEO4J_PASSWORD is mandatory"
else
  skip "docker compose not available (structural checks ran in the policy step)"
fi

if [[ "$MODE" == "offline" ]]; then
  printf '\n'
  ok "offline smoke checks passed"
  exit 0
fi

# ---------------------------------------------------------------------------
# Daemon: build and inspect the real image
# ---------------------------------------------------------------------------
if ! have_daemon; then
  printf '\n'
  skip "no Docker daemon reachable: image build and boot checks skipped"
  [[ "$MODE" == "live" ]] && die "--live requires a running Docker daemon"
  ok "offline smoke checks passed"
  exit 0
fi

step "build"
docker build --pull=false -t "$IMAGE" --build-arg REVISION="$(git rev-parse --short HEAD)" . ||
  die "image build failed"
ok "built $IMAGE"

step "image metadata"
user="$(docker image inspect -f '{{.Config.User}}' "$IMAGE")"
[[ "$user" == "10001:10001" ]] || die "image user is '$user', expected 10001:10001"
entrypoint="$(docker image inspect -f '{{json .Config.Entrypoint}}' "$IMAGE")"
[[ "$entrypoint" == '["python","-m","agent_memory_graph"]' ]] || die "unexpected entrypoint $entrypoint"
license="$(docker image inspect -f '{{index .Config.Labels "org.opencontainers.image.licenses"}}' "$IMAGE")"
[[ -n "$license" ]] || die "missing OCI license label"
ok "user 10001:10001, exec entrypoint, OCI labels ($license)"

step "runtime filesystem (read-only root, no caps, no new privileges)"
docker run --rm --network none --read-only --tmpfs /tmp:rw,noexec,nosuid,nodev \
  --cap-drop ALL --security-opt no-new-privileges:true --entrypoint python "$IMAGE" - <<'PY' ||
import os, shutil, stat, sys

problems = []
if (os.getuid(), os.getgid()) != (10001, 10001):
    problems.append(f"running as {os.getuid()}:{os.getgid()}")
for tool in ("apt", "apt-get", "dpkg", "pip", "pip3", "gcc", "cc", "sudo", "su"):
    if shutil.which(tool):
        problems.append(f"{tool} present at {shutil.which(tool)}")
for root, dirs, files in os.walk("/"):
    dirs[:] = [d for d in dirs if os.path.join(root, d) not in ("/proc", "/sys", "/dev")]
    for name in files:
        path = os.path.join(root, name)
        try:
            mode = os.lstat(path).st_mode
        except OSError:
            continue
        if stat.S_ISREG(mode) and mode & (stat.S_ISUID | stat.S_ISGID):
            problems.append(f"setuid/setgid: {path}")
for target in ("/opt/venv/probe", "/probe"):
    try:
        open(target, "w").close()
        problems.append(f"writable: {target}")
    except OSError:
        pass
with open("/tmp/probe", "w") as handle:  # the only writable location
    handle.write("ok")
import agent_memory_graph  # noqa: F401  (imports from a read-only, precompiled venv)
if problems:
    print("\n".join(problems))
    sys.exit(1)
PY
  die "runtime filesystem checks failed"
ok "uid 10001, no package managers/compilers, no setuid bits, root fs read-only, /tmp writable"

if [[ "$MODE" != "live" ]]; then
  printf '\n'
  ok "smoke checks passed (use --live to boot the Compose stack)"
  exit 0
fi

# ---------------------------------------------------------------------------
# Live: boot the stack, probe health, assert graceful SIGTERM
# ---------------------------------------------------------------------------
step "compose stack"
AMG_NEO4J_PASSWORD="$("$PYTHON" -c 'import secrets; print(secrets.token_urlsafe(24))')"
export AMG_NEO4J_PASSWORD
export AMG_HTTP_PORT="${AMG_HTTP_PORT:-18000}"
cleanup() { docker compose -p "$PROJECT" down -v --remove-orphans >/dev/null 2>&1 || true; }
trap cleanup EXIT

docker compose -p "$PROJECT" up -d --build --wait --wait-timeout 180 || {
  docker compose -p "$PROJECT" logs --tail 80
  die "stack did not become healthy"
}
health="$("$PYTHON" - <<PY
import urllib.request
print(urllib.request.urlopen("http://127.0.0.1:${AMG_HTTP_PORT}/health", timeout=5).read().decode())
PY
)"
[[ "$health" == *'"status":"ok"'* ]] || die "unexpected health: $health"
ok "stack healthy: $health"

container="$(docker compose -p "$PROJECT" ps -q agent-memory-graph)"
docker compose -p "$PROJECT" stop -t 30 agent-memory-graph >/dev/null
code="$(docker inspect -f '{{.State.ExitCode}}' "$container")"
[[ "$code" == "0" ]] || die "API exited with $code after SIGTERM"
docker logs "$container" 2>&1 | grep -q "graceful shutdown complete (signals: SIGTERM)" ||
  die "no graceful shutdown in the API logs"
ok "SIGTERM: graceful shutdown, exit code 0"

printf '\n'
ok "live smoke checks passed"
