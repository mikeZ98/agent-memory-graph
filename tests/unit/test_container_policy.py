"""Container policy: the real files pass, and every rule fires on a violating variant."""

import importlib.util
import os
import shutil
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_policy() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "container_policy", REPO_ROOT / "scripts" / "container_policy.py"
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses resolve annotations through sys.modules
    spec.loader.exec_module(module)
    return module


policy = _load_policy()
DOCKERFILE = (REPO_ROOT / "Dockerfile").read_text()
DOCKERIGNORE = (REPO_ROOT / ".dockerignore").read_text()
COMPOSE: dict[str, Any] = yaml.safe_load((REPO_ROOT / "docker-compose.yml").read_text())


def rules(violations: list[Any]) -> set[str]:
    return {v.rule for v in violations}


def compose_with(mutate: Callable[[dict[str, Any]], None]) -> dict[str, Any]:
    data: dict[str, Any] = yaml.safe_load(yaml.safe_dump(COMPOSE))
    mutate(data)
    return data


# ---------------------------------------------------------------------------
# The shipped files
# ---------------------------------------------------------------------------
def test_shipped_files_pass_every_rule() -> None:
    assert (
        policy.run(
            REPO_ROOT / "Dockerfile", REPO_ROOT / ".dockerignore", REPO_ROOT / "docker-compose.yml"
        )
        == []
    )


def test_cli_exit_codes(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    os.chdir(REPO_ROOT)
    assert policy.main([]) == 0
    assert policy.main(["only-one-file"]) == 2
    missing = [str(tmp_path / name) for name in ("D", "I", "C")]
    assert policy.main(missing) == 1
    assert "AMG-F001" in capsys.readouterr().err


def test_parser_joins_continuations_and_resolves_global_args() -> None:
    stages = policy.split_stages(policy.parse_dockerfile(DOCKERFILE))
    assert [s.name for s in stages] == ["uv", "builder", "runtime"]
    assert all("@sha256:" in s.image for s in stages)
    assert stages[-1].all("USER")[-1].value == "10001:10001"


# ---------------------------------------------------------------------------
# Dockerfile rules
# ---------------------------------------------------------------------------
def mutated(old: str, new: str) -> str:
    assert old in DOCKERFILE, old
    return DOCKERFILE.replace(old, new)


@pytest.mark.parametrize(
    ("text", "rule"),
    [
        (mutated("@sha256:229a2c5bfa27522db7815ea81f9bed70af17ccb9de9fc7ad142b1877b5830d36", ""),
         "DL3006/AMG-D001"),
        (mutated("python:3.12.13-slim-trixie@sha256:229a2c5bfa27522db7815ea81f9bed70af17ccb9de9fc7ad142b1877b5830d36",
                 "python:latest"), "DL3007"),
        (mutated("USER 10001:10001", "USER root"), "DL3002/AMG-D003"),
        (mutated("USER 10001:10001\n", ""), "DL3002/AMG-D003"),
        (mutated("--uid 10001", "--uid 0"), "AMG-D004"),
        (mutated("find / -xdev -type f -perm /6000 -exec chmod a-s {} +;", "true;"), "AMG-D007"),
        (mutated("/var/lib/dpkg /var/cache/debconf", "/var/cache/debconf"), "AMG-D007"),
        (mutated("/usr/local/bin/pip /usr/local/bin/pip3", "/usr/local/bin/pip3"), "AMG-D008"),
        (mutated("/usr/bin/dpkg /usr/bin/dpkg-*", "/usr/bin/dpkg-*"), "AMG-D007"),
        (mutated('ENTRYPOINT ["python", "-m", "agent_memory_graph"]',
                 "ENTRYPOINT python -m agent_memory_graph"), "DL3025"),
        (mutated('ENTRYPOINT ["python", "-m", "agent_memory_graph"]',
                 'ENTRYPOINT ["/bin/sh", "-c", "python -m agent_memory_graph"]'), "AMG-D005"),
        (mutated('CMD ["--host", "0.0.0.0", "--port", "8000"]', ""), "AMG-D005"),
        (mutated("STOPSIGNAL SIGTERM", "STOPSIGNAL SIGKILL"), "AMG-D010"),
        (mutated("    PYTHONDONTWRITEBYTECODE=1 \\\n    HOME=/tmp", "    HOME=/tmp"), "AMG-D019"),
        (mutated('licenses="MIT"', 'licenses="MIT License!"'), "AMG-D011"),
        (mutated('org.opencontainers.image.source="https://github.com/mikeZ98/agent-memory-graph"',
                 'org.opencontainers.image.source="git@github.com:x/y"'), "AMG-D011"),
        (mutated('org.opencontainers.image.revision="${REVISION}" \\\n', ""), "AMG-D011"),
        (mutated("WORKDIR /tmp", "WORKDIR tmp"), "DL3000"),
        (mutated("COPY src ./src", "ADD src ./src"), "DL3020"),
        (mutated("RUN set -eu; \\\n    groupadd", "RUN sudo true; \\\n    groupadd"), "DL3004"),
        (mutated("COPY src ./src", "RUN apt-get install -y gcc\nCOPY src ./src"), "DL3015"),
        (mutated("COPY src ./src", "RUN pip install uv\nCOPY src ./src"), "DL3042"),
        (mutated("ENV PATH=", "ENV AMG_NEO4J_PASSWORD=hunter2 PATH="), "AMG-D012"),
        (mutated("COPY --from=builder --chown=root:root /opt/venv /opt/venv",
                 "COPY --from=uv /uv /usr/local/bin/uv"), "AMG-D017"),
        (mutated("FROM ${PYTHON_IMAGE} AS runtime", "FROM ${PYTHON_IMAGE} AS final"), "AMG-D002"),
    ],
)  # fmt: skip
def test_dockerfile_rules_fire(text: str, rule: str) -> None:
    assert rule in rules(policy.check_dockerfile(text))


@pytest.mark.parametrize(
    ("old", "new"),
    [
        ('--retries=3 \\\n    CMD ["python",', '--retries=3 CMD python -c pass # ["python",'),
    ],
)  # fmt: skip
def test_healthcheck_must_be_exec_form(old: str, new: str) -> None:
    text = mutated(old, new)
    assert "AMG-D009" in rules(policy.check_dockerfile(text))
    no_health = "\n".join(line for line in DOCKERFILE.splitlines() if "HEALTHCHECK" not in line
                          and not line.lstrip().startswith('CMD ["python", "-c"'))  # fmt: skip
    assert "AMG-D009" in rules(policy.check_dockerfile(no_health))


def test_single_stage_and_empty_dockerfiles() -> None:
    assert "AMG-D002" in rules(policy.check_dockerfile("FROM python@sha256:" + "a" * 64))
    assert "AMG-D002" in rules(policy.check_dockerfile("# nothing\n"))


# ---------------------------------------------------------------------------
# .dockerignore rules
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("text", "rule"),
    [
        (DOCKERIGNORE.replace("*\n", "", 1), "AMG-I001"),
        (DOCKERIGNORE + "!.env\n", "AMG-I002"),
        (DOCKERIGNORE.replace("**/.cache\n", ""), "AMG-I003"),
        ("", "AMG-I001"),
    ],
)
def test_dockerignore_rules_fire(text: str, rule: str) -> None:
    assert rule in rules(policy.check_dockerignore(text))


# ---------------------------------------------------------------------------
# Compose rules
# ---------------------------------------------------------------------------
APP = "agent-memory-graph"


def app_set(key: str, value: Any) -> Callable[[dict[str, Any]], None]:
    def mutate(data: dict[str, Any]) -> None:
        data["services"][APP][key] = value

    return mutate


def app_drop(key: str) -> Callable[[dict[str, Any]], None]:
    return lambda data: data["services"][APP].pop(key)


def store_set(store: str, key: str, value: Any) -> Callable[[dict[str, Any]], None]:
    def mutate(data: dict[str, Any]) -> None:
        data["services"][store][key] = value

    return mutate


def env_set(key: str, value: str) -> Callable[[dict[str, Any]], None]:
    def mutate(data: dict[str, Any]) -> None:
        data["services"][APP]["environment"][key] = value

    return mutate


@pytest.mark.parametrize(
    ("mutate", "rule"),
    [
        (app_set("read_only", False), "AMG-C002"),
        (app_drop("cap_drop"), "AMG-C003"),
        (app_set("cap_add", ["NET_ADMIN"]), "AMG-C003"),
        (app_set("security_opt", []), "AMG-C004"),
        (app_set("user", "root"), "AMG-C005"),
        (app_set("tmpfs", ["/tmp"]), "AMG-C006"),  # noqa: S108
        (app_set("privileged", True), "AMG-C007"),
        (app_set("network_mode", "host"), "AMG-C007"),
        (app_set("ports", ["8000:8000"]), "AMG-C008"),
        (env_set("DATABASE_URL", "x"), "AMG-C009"),
        (env_set("AMG_NEO4J_PASSWORD", "${AMG_NEO4J_PASSWORD:-changeme}"), "AMG-C009"),
        (env_set("AMG_SECRET_TOKEN", "literal"), "AMG-C030"),
        (app_set("healthcheck", {"test": "curl localhost"}), "AMG-C010"),
        (app_set("depends_on", ["neo4j", "qdrant"]), "AMG-C011"),
        (app_set("build", {"context": "./other"}), "AMG-C001"),
        (store_set("neo4j", "image", "neo4j:5.26-community"), "AMG-C020"),
        (store_set("qdrant", "image", "qdrant/qdrant@sha256:" + "a" * 64), "AMG-C020"),
        (store_set("qdrant", "ports", ["6333:6333"]), "AMG-C022"),
        (store_set("neo4j", "networks", ["edge"]), "AMG-C024"),
        (store_set("neo4j", "healthcheck", {"test": ["CMD", "true"]}), "AMG-C021"),
        (store_set("qdrant", "healthcheck", {"test": ["CMD", "true"]}), "AMG-C021"),
        (lambda data: data["services"]["qdrant"].pop("healthcheck"), "AMG-C021"),
        (lambda data: data["services"]["neo4j"].update(environment=["NEO4J_AUTH=neo4j/secret"]),
         "AMG-C030"),
    ],
)  # fmt: skip
def test_compose_rules_fire(mutate: Callable[[dict[str, Any]], None], rule: str) -> None:
    assert rule in rules(policy.check_compose(compose_with(mutate)))


def test_compose_without_the_api_service() -> None:
    assert rules(policy.check_compose({"services": {}})) == {"AMG-C001"}


# ---------------------------------------------------------------------------
# The smoke script itself (offline mode: no daemon, no network)
# ---------------------------------------------------------------------------
@pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")
def test_offline_smoke_script_passes() -> None:
    result = subprocess.run(  # noqa: S603
        ["bash", str(REPO_ROOT / "scripts" / "smoke-docker.sh"), "--offline"],  # noqa: S607
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "offline smoke checks passed" in result.stdout


@pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")
def test_smoke_script_rejects_unknown_flags() -> None:
    result = subprocess.run(  # noqa: S603
        ["bash", str(REPO_ROOT / "scripts" / "smoke-docker.sh"), "--yolo"],  # noqa: S607
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=30, check=False,
    )  # fmt: skip
    assert result.returncode == 2
