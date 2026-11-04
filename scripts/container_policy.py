"""Offline container policy checks: Dockerfile, .dockerignore and the Compose stack.

Needs no Docker daemon and no network. Covers the Hadolint rules that matter for this
image (noted as ``DLxxxx``) plus the project's hardening invariants (``AMG-*``).

Usage: python scripts/container_policy.py [Dockerfile] [.dockerignore] [docker-compose.yml]
Exit status: 0 when every check passes, 1 otherwise (each violation is printed).
"""

import json
import re
import shlex
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

import yaml

APP_UID: Final = "10001"
APP_USER: Final = f"{APP_UID}:{APP_UID}"
APP_SERVICE: Final = "agent-memory-graph"
STORE_SERVICES: Final = ("neo4j", "qdrant")
REQUIRED_LABELS: Final = (
    "org.opencontainers.image.title",
    "org.opencontainers.image.description",
    "org.opencontainers.image.licenses",
    "org.opencontainers.image.source",
    "org.opencontainers.image.version",
    "org.opencontainers.image.revision",
)
PROVIDER_CREDENTIALS: Final = frozenset({"ANTHROPIC_API_KEY", "OPENAI_API_KEY"})

_DIGEST_RE: Final = re.compile(r"@sha256:[0-9a-f]{64}$")
_SPDX_RE: Final = re.compile(r"^[A-Za-z0-9.+-]+(?: (?:AND|OR|WITH) [A-Za-z0-9.+-]+)*$")
_SECRET_NAME_RE: Final = re.compile(r"PASSWORD|SECRET|TOKEN|API_KEY|PRIVATE_KEY|_AUTH$", re.I)
_SHELLS: Final = frozenset(
    {"sh", "bash", "dash", "ash", "zsh", "/bin/sh", "/bin/bash", "/usr/bin/env"}
)
_ARG_REF_RE: Final = re.compile(r"\$\{?([A-Za-z_][A-Za-z0-9_]*)\}?")


@dataclass(frozen=True, slots=True)
class Violation:
    """One failed check."""

    rule: str
    message: str
    location: str

    def __str__(self) -> str:
        return f"{self.location}: [{self.rule}] {self.message}"


# ---------------------------------------------------------------------------
# Dockerfile parsing
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class Instruction:
    """One Dockerfile instruction with continuations joined."""

    keyword: str
    value: str
    line: int


@dataclass(slots=True)
class Stage:
    """Instructions from one ``FROM`` up to the next."""

    image: str
    name: str | None
    line: int
    instructions: list[Instruction] = field(default_factory=list)

    def all(self, keyword: str) -> list[Instruction]:
        """Return this stage's instructions of one kind, in order."""
        return [i for i in self.instructions if i.keyword == keyword]


def parse_dockerfile(text: str) -> list[Instruction]:
    """Split ``text`` into instructions (comments dropped, line continuations joined)."""
    instructions: list[Instruction] = []
    buffer: list[str] = []
    start = 0
    for number, raw in enumerate(text.splitlines(), start=1):
        stripped = raw.strip()
        if stripped.startswith("#") or (not stripped and not buffer):
            continue
        if not buffer:
            start = number
        continued = stripped.endswith("\\")
        buffer.append(stripped[:-1].rstrip() if continued else stripped)
        if not continued:
            joined = " ".join(part for part in buffer if part)
            keyword, _, value = joined.partition(" ")
            instructions.append(Instruction(keyword.upper(), value.strip(), start))
            buffer = []
    return instructions


def _arg_defaults(instructions: Iterable[Instruction]) -> dict[str, str]:
    defaults: dict[str, str] = {}
    for instruction in instructions:
        if instruction.keyword == "FROM":
            break
        if instruction.keyword == "ARG" and "=" in instruction.value:
            name, _, default = instruction.value.partition("=")
            defaults[name.strip()] = default.strip().strip('"')
    return defaults


def split_stages(instructions: Sequence[Instruction]) -> list[Stage]:
    """Group instructions by stage, resolving global ``ARG`` defaults in ``FROM``."""
    defaults = _arg_defaults(instructions)
    stages: list[Stage] = []
    for instruction in instructions:
        if instruction.keyword == "FROM":
            parts = [p for p in instruction.value.split() if not p.startswith("--")]
            image = _ARG_REF_RE.sub(lambda m: defaults.get(m.group(1), m.group(0)), parts[0])
            name = parts[2] if len(parts) >= 3 and parts[1].upper() == "AS" else None
            stages.append(Stage(image=image, name=name, line=instruction.line))
        elif stages:
            stages[-1].instructions.append(instruction)
    return stages


def _exec_form(value: str) -> list[str] | None:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, list) and all(isinstance(p, str) for p in parsed) else None


def _removes(run_text: str, path: str) -> bool:
    # A whole shell token, so "/usr/local/bin/pip3" does not count as removing "pip".
    return re.search(rf"(?<!\S){re.escape(path)}(?=[\s;]|$)", run_text) is not None


def _labels(stage: Stage) -> dict[str, str]:
    labels: dict[str, str] = {}
    for instruction in stage.all("LABEL"):
        for token in shlex.split(instruction.value):
            key, sep, value = token.partition("=")
            if sep:
                labels[key] = value
    return labels


# ---------------------------------------------------------------------------
# Dockerfile rules
# ---------------------------------------------------------------------------
def check_dockerfile(text: str, location: str = "Dockerfile") -> list[Violation]:
    """Return every hardening violation in a Dockerfile."""
    out: list[Violation] = []

    def fail(rule: str, message: str, line: int) -> None:
        out.append(Violation(rule, message, f"{location}:{line}"))

    instructions = parse_dockerfile(text)
    stages = split_stages(instructions)
    if len(stages) < 2:
        fail("AMG-D002", "multi-stage build required (builder + runtime)", 1)
    if not stages:
        return out
    stage_names = {s.name for s in stages if s.name}
    for stage in stages:
        if stage.image in stage_names:
            continue
        if not _DIGEST_RE.search(stage.image):
            fail("DL3006/AMG-D001", f"base image not pinned by digest: {stage.image}", stage.line)
        if stage.image.split("@")[0].endswith(":latest"):
            fail("DL3007", f"'latest' tag used: {stage.image}", stage.line)

    for instruction in instructions:
        text_value = instruction.value
        if instruction.keyword == "ADD":
            fail("DL3020", "use COPY instead of ADD", instruction.line)
        if instruction.keyword == "WORKDIR" and not text_value.startswith(("/", "$")):
            fail("DL3000", f"WORKDIR must be absolute: {text_value}", instruction.line)
        if instruction.keyword == "RUN":
            if re.search(r"\bsudo\b", text_value):
                fail("DL3004", "do not use sudo", instruction.line)
            if (
                re.search(r"\bapt-get install\b", text_value)
                and "--no-install-recommends" not in text_value
            ):
                fail("DL3015", "apt-get install without --no-install-recommends", instruction.line)
            if re.search(r"\bpip3? install\b", text_value) and "--no-cache-dir" not in text_value:
                fail("DL3042", "pip install without --no-cache-dir", instruction.line)
        if instruction.keyword in {"ENV", "ARG"}:
            for token in shlex.split(text_value):
                name, sep, value = token.partition("=")
                if sep and value and _SECRET_NAME_RE.search(name):
                    fail(
                        "AMG-D012",
                        f"secret-looking {instruction.keyword} baked in: {name}",
                        instruction.line,
                    )

    runtime = stages[-1]
    if runtime.name != "runtime":
        fail("AMG-D002", "final stage must be named 'runtime'", runtime.line)
    users = runtime.all("USER")
    if not users or users[-1].value != APP_USER:
        fail(
            "DL3002/AMG-D003",
            f"final USER must be {APP_USER}",
            users[-1].line if users else runtime.line,
        )
    run_text = " ".join(i.value for i in runtime.all("RUN"))
    if not re.search(
        rf"groupadd\b[^;]*--gid {APP_UID}\b[^;]*\bappuser\b", run_text
    ) or not re.search(
        rf"useradd\b[^;]*--uid {APP_UID}\b[^;]*--gid {APP_UID}\b[^;]*\bappuser\b", run_text
    ):
        fail(
            "AMG-D004", f"runtime must create appuser with static UID:GID {APP_USER}", runtime.line
        )
    if "-perm /6000" not in run_text or "chmod a-s" not in run_text:
        fail("AMG-D007", "runtime must strip setuid/setgid bits", runtime.line)
    for path in ("/var/lib/dpkg", "/usr/bin/dpkg", "/usr/bin/apt", "/etc/apt"):
        if not _removes(run_text, path):
            fail("AMG-D007", f"runtime must remove the package manager ({path})", runtime.line)
    pip_paths = ("/usr/local/lib/python3.12/site-packages/pip", "/usr/local/bin/pip")
    if not all(_removes(run_text, path) for path in pip_paths):
        fail("AMG-D008", "runtime must remove pip", runtime.line)
    for copy in runtime.all("COPY"):
        if re.search(r"--from=(uv|builder-tools)\b", copy.value) and "/opt/venv" not in copy.value:
            fail("AMG-D017", "build tooling copied into the runtime stage", copy.line)

    env_text = " ".join(i.value for i in runtime.all("ENV"))
    if "PYTHONDONTWRITEBYTECODE=1" not in env_text:
        fail("AMG-D019", "set PYTHONDONTWRITEBYTECODE=1 (read-only root filesystem)", runtime.line)

    for keyword in ("ENTRYPOINT", "CMD"):
        found = runtime.all(keyword)
        if not found:
            fail("AMG-D005", f"runtime needs an exec-form {keyword}", runtime.line)
            continue
        argv = _exec_form(found[-1].value)
        if argv is None:
            fail("DL3025", f"{keyword} must use JSON exec form", found[-1].line)
        elif keyword == "ENTRYPOINT" and ((argv[:1] and argv[0] in _SHELLS) or "-c" in argv[:2]):
            fail("AMG-D005", "ENTRYPOINT must not be a shell wrapper", found[-1].line)

    health = runtime.all("HEALTHCHECK")
    if not health:
        fail("AMG-D009", "runtime needs a HEALTHCHECK", runtime.line)
    else:
        command = health[-1].value.partition("CMD")[2].strip()
        if _exec_form(command) is None:
            fail("AMG-D009", "HEALTHCHECK must use exec form", health[-1].line)
    stop = runtime.all("STOPSIGNAL")
    if not stop or stop[-1].value != "SIGTERM":
        fail("AMG-D010", "STOPSIGNAL SIGTERM required", stop[-1].line if stop else runtime.line)

    labels = _labels(runtime)
    for label in REQUIRED_LABELS:
        if label not in labels:
            fail("AMG-D011", f"missing OCI label {label}", runtime.line)
    license_id = labels.get("org.opencontainers.image.licenses", "")
    if license_id and not _SPDX_RE.fullmatch(license_id):
        fail("AMG-D011", f"licenses is not an SPDX expression: {license_id}", runtime.line)
    source = labels.get("org.opencontainers.image.source", "")
    if source and not source.startswith("https://"):
        fail("AMG-D011", "source label must be an https URL", runtime.line)
    return out


def check_dockerignore(text: str, location: str = ".dockerignore") -> list[Violation]:
    """Require a deny-by-default context that can never include secrets or caches."""
    out: list[Violation] = []
    patterns = [p.strip() for p in text.splitlines() if p.strip() and not p.strip().startswith("#")]
    if not patterns or patterns[0] != "*":
        out.append(Violation("AMG-I001", "first pattern must be '*' (deny by default)", location))
    allowed = {p[1:] for p in patterns if p.startswith("!")}
    out.extend(
        Violation("AMG-I002", f"must not re-include {sensitive}", location)
        for sensitive in (".env", ".git", ".cache", ".envrc", ".git/")
        if sensitive in allowed
    )
    out.extend(
        Violation("AMG-I003", f"missing exclusion {required}", location)
        for required in ("**/.env", "**/.cache")
        if required not in patterns
    )
    return out


# ---------------------------------------------------------------------------
# Compose rules
# ---------------------------------------------------------------------------
def _environment(service: Mapping[str, Any]) -> dict[str, str]:
    env = service.get("environment") or {}
    if isinstance(env, list):
        return {k: v for k, _, v in (str(item).partition("=") for item in env)}
    return {str(k): "" if v is None else str(v) for k, v in env.items()}


def _healthcheck_text(service: Mapping[str, Any]) -> str:
    test = (service.get("healthcheck") or {}).get("test")
    return " ".join(test) if isinstance(test, list) else str(test or "")


def check_compose(data: Mapping[str, Any], location: str = "docker-compose.yml") -> list[Violation]:
    """Return every hardening violation in a parsed Compose document."""
    out: list[Violation] = []

    def fail(rule: str, service: str, message: str) -> None:
        out.append(Violation(rule, message, f"{location}:services.{service}"))

    services: Mapping[str, Any] = data.get("services") or {}
    app = services.get(APP_SERVICE)
    if app is None:
        out.append(Violation("AMG-C001", f"service {APP_SERVICE} missing", location))
        return out
    if (app.get("build") or {}).get("context") != ".":
        fail("AMG-C001", APP_SERVICE, "must build from the local Dockerfile (context '.')")
    if app.get("read_only") is not True:
        fail("AMG-C002", APP_SERVICE, "read_only: true required")
    if "ALL" not in (app.get("cap_drop") or []) or app.get("cap_add"):
        fail("AMG-C003", APP_SERVICE, "cap_drop: [ALL] required and no cap_add allowed")
    if str(app.get("user")) != APP_USER:
        fail("AMG-C005", APP_SERVICE, f"user must be {APP_USER}")
    tmpfs = " ".join(app.get("tmpfs") or [])
    if not re.search(r"(^|\s)/tmp:[^\s]*noexec", tmpfs) or "nosuid" not in tmpfs:
        fail("AMG-C006", APP_SERVICE, "tmpfs /tmp with noexec,nosuid required")
    for port in app.get("ports") or []:
        if not str(port).startswith("127.0.0.1:"):
            fail("AMG-C008", APP_SERVICE, f"port must bind to 127.0.0.1: {port}")
    env = _environment(app)
    for key in env:
        if not key.startswith("AMG_") and key not in PROVIDER_CREDENTIALS:
            fail("AMG-C009", APP_SERVICE, f"unexpected environment variable {key}")
    if not re.fullmatch(r"\$\{AMG_NEO4J_PASSWORD:\?[^}]*\}", env.get("AMG_NEO4J_PASSWORD", "")):
        fail(
            "AMG-C009",
            APP_SERVICE,
            "AMG_NEO4J_PASSWORD must be required via ${AMG_NEO4J_PASSWORD:?...}",
        )
    test = (app.get("healthcheck") or {}).get("test")
    if not (isinstance(test, list) and test[:1] == ["CMD"]):
        fail("AMG-C010", APP_SERVICE, "healthcheck must be exec form ([CMD, ...])")
    depends = app.get("depends_on") or {}
    if not isinstance(depends, Mapping):  # short syntax cannot express a condition
        depends = {}
    for store in STORE_SERVICES:
        if (depends.get(store) or {}).get("condition") != "service_healthy":
            fail("AMG-C011", APP_SERVICE, f"must wait for {store} to be healthy")

    networks: Mapping[str, Any] = data.get("networks") or {}
    for name, service in services.items():
        if service.get("privileged"):
            fail("AMG-C007", name, "privileged containers are forbidden")
        for mode in ("network_mode", "pid", "ipc"):
            if str(service.get(mode, "")).startswith("host"):
                fail("AMG-C007", name, f"{mode}: host is forbidden")
        if "no-new-privileges:true" not in (service.get("security_opt") or []):
            fail("AMG-C004", name, "security_opt: [no-new-privileges:true] required")
        for key, value in _environment(service).items():
            if _SECRET_NAME_RE.search(key) and value and "${" not in value:
                fail("AMG-C030", name, f"literal secret in environment: {key}")
        if name in STORE_SERVICES:
            image = str(service.get("image", ""))
            if not _DIGEST_RE.search(image) or ":" not in image.split("@", maxsplit=1)[0]:
                fail("AMG-C020", name, f"image must be pinned by tag and digest: {image}")
            if service.get("ports"):
                fail("AMG-C022", name, "stores must not publish ports")
            attached = service.get("networks") or []
            if not any((networks.get(n) or {}).get("internal") for n in attached):
                fail("AMG-C024", name, "stores must sit on an internal network")
            if not service.get("healthcheck"):
                fail("AMG-C021", name, "healthcheck required")
    neo4j_probe = _healthcheck_text(services.get("neo4j") or {})
    if "7687" not in neo4j_probe and "7474" not in neo4j_probe:
        fail("AMG-C021", "neo4j", "healthcheck must probe bolt 7687 or http 7474")
    if "/readyz" not in _healthcheck_text(services.get("qdrant") or {}):
        fail("AMG-C021", "qdrant", "healthcheck must probe GET /readyz")
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def run(dockerfile: Path, dockerignore: Path, compose: Path) -> list[Violation]:
    """Check all three files; missing files are violations too."""
    violations = [
        Violation("AMG-F001", "file not found", str(path))
        for path in (dockerfile, dockerignore, compose)
        if not path.is_file()
    ]
    if violations:
        return violations
    violations += check_dockerfile(dockerfile.read_text(), str(dockerfile))
    violations += check_dockerignore(dockerignore.read_text(), str(dockerignore))
    violations += check_compose(yaml.safe_load(compose.read_text()) or {}, str(compose))
    return violations


def main(argv: Sequence[str] | None = None) -> int:
    """Print violations; return 0 when there are none."""
    args = list(sys.argv[1:] if argv is None else argv)
    paths = [Path(p) for p in (args or ["Dockerfile", ".dockerignore", "docker-compose.yml"])]
    if len(paths) != 3:
        print(__doc__, file=sys.stderr)
        return 2
    violations = run(*paths)
    for violation in violations:
        print(f"  x {violation}", file=sys.stderr)
    return 1 if violations else 0


if __name__ == "__main__":
    sys.exit(main())
