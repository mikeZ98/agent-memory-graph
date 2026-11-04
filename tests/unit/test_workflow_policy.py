"""Workflow policy: ci.yml passes, and each rule fires on a violating variant."""

import importlib.util
import os
import sys
from pathlib import Path
from types import ModuleType

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
CI_PATH = REPO_ROOT / ".github" / "workflows" / "ci.yml"
CI = CI_PATH.read_text()
CHECKOUT_PIN = "actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1 # v7.0.1"
FIRST_JOB = "    runs-on: ubuntu-24.04\n    timeout-minutes: 5"
HYGIENE_STEP = "      - name: Audit tracked files\n"
MINIMAL = (
    "on: [push]\npermissions:\n  contents: read\n"
    "jobs:\n  x:\n    runs-on: ubuntu-24.04\n    timeout-minutes: 1\n"
)


def _load(name: str, filename: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / "scripts" / filename)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


policy = _load("audit_workflows", "audit-workflows.py")


def rules(text: str, location: str = ".github/workflows/ci.yml") -> set[str]:
    return {v.rule for v in policy.check_workflow(text, location)}


def mutated(old: str, new: str, *, count: int = 1) -> str:
    assert old in CI, old
    return CI.replace(old, new, count)


def test_ci_workflow_passes() -> None:
    assert policy.check_workflow(CI, str(CI_PATH)) == []


def test_every_remote_action_is_sha_pinned_with_a_version_comment() -> None:
    lines = list(policy._uses_lines(CI))
    assert len(lines) >= 9
    assert all(len(ref.split("@")[1]) == 40 and comment for _, ref, comment in lines)


def test_cli(tmp_path: Path) -> None:
    os.chdir(REPO_ROOT)
    assert policy.main([]) == 0
    assert policy.main([str(tmp_path / "missing.yml")]) == 1
    empty = tmp_path / "repo"
    empty.mkdir()
    os.chdir(empty)
    try:
        assert policy.main([]) == 1
    finally:
        os.chdir(REPO_ROOT)


@pytest.mark.parametrize(
    ("text", "rule"),
    [
        ("jobs: [unclosed", "AMG-W001"),
        ("- just a list", "AMG-W001"),
        (mutated("  pull_request:\n", "  pull_request_target:\n"), "AMG-W002"),
        (mutated("permissions:\n  contents: read", "permissions: write-all"), "AMG-W003"),
        (mutated("permissions:\n  contents: read", "permissions:\n  contents: write"), "AMG-W003"),
        (mutated(FIRST_JOB, FIRST_JOB + "\n    permissions: write-all"), "AMG-W003"),
        (mutated(FIRST_JOB, FIRST_JOB + "\n    permissions:\n      packages: write"), "AMG-W003"),
        (mutated("runs-on: ubuntu-24.04", "runs-on: ubuntu-latest"), "AMG-W004"),
        (mutated("    timeout-minutes: 5\n", ""), "AMG-W007"),
        (mutated(CHECKOUT_PIN, "actions/checkout@v7"), "AMG-W005"),
        (mutated(CHECKOUT_PIN, "actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1"),
         "AMG-W005"),
        (mutated(CHECKOUT_PIN, "actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1 # main"),
         "AMG-W005"),
        (mutated(CHECKOUT_PIN, "actions/checkout@" + "a" * 40 + " # v7.0.1"), "AMG-W005"),
        (mutated(HYGIENE_STEP, "      - uses: docker://alpine:3\n" + HYGIENE_STEP), "AMG-W005"),
        (mutated("          persist-credentials: false\n", "", count=1), "AMG-W006"),
        (mutated("run: scripts/audit-hygiene.sh",
                 "run: echo ${{ github.event.pull_request.title }}"), "AMG-W008"),
        (mutated("    timeout-minutes: 5\n", "    timeout-minutes: 5\n    secrets: inherit\n"),
         "AMG-W009"),
        (mutated("    branches: [main]\n", "    branches: [develop]\n"), "AMG-W010"),
        (mutated("  container-security:\n", "  container-scan:\n"), "AMG-W010"),
        (mutated("          fetch-depth: 0\n", ""), "AMG-W011"),
        (mutated("run: scripts/audit-hygiene.sh", "run: true"), "AMG-W011"),
        (mutated("          HYPOTHESIS_PROFILE: ci\n", "          HYPOTHESIS_PROFILE: dev\n"),
         "AMG-W013"),
        (mutated("run: scripts/coverage-gate.sh", "run: true"), "AMG-W013"),
        (mutated('          python-version: "3.12"\n          enable-cache: true',
                 '          python-version: "3.11"\n          enable-cache: true'), "AMG-W013"),
        (mutated("run: scripts/smoke-docker.sh --live", "run: true"), "AMG-W012"),
        (mutated("          file: Dockerfile\n", "          file: other.Dockerfile\n"), "AMG-W012"),
        (mutated('          exit-code: "1"\n', '          exit-code: "0"\n'), "AMG-W012"),
        (mutated("ignore-unfixed: true\n", "ignore-unfixed: false\n"), "AMG-W012"),
        (mutated("          severity: CRITICAL,HIGH\n          ignore-unfixed: true",
                 "          severity: CRITICAL\n          ignore-unfixed: true"), "AMG-W012"),
        (mutated("          version: v0.76.1\n          scan-type: image",
                 "          scan-type: image"), "AMG-W012"),
    ],
)  # fmt: skip
def test_rules_fire(text: str, rule: str) -> None:
    assert rule in rules(text)


def test_inconsistent_pins_for_one_action_are_rejected() -> None:
    other = "actions/checkout@" + "b" * 40 + " # v7.0.0"
    assert "AMG-W005" in rules(CI.replace(CHECKOUT_PIN, other, 1))


def test_setup_uv_is_required_in_code_quality() -> None:
    start = CI.index("      - name: Set up uv and Python 3.12\n")
    end = CI.index("      - name: Install locked dependencies", start)
    assert "AMG-W013" in rules(CI[:start] + CI[end:])


def test_ci_contract_applies_only_to_ci_yml() -> None:
    assert rules(MINIMAL, ".github/workflows/other.yml") == set()
    assert "AMG-W010" in rules(MINIMAL, ".github/workflows/ci.yml")


def test_local_actions_are_allowed() -> None:
    text = mutated(HYGIENE_STEP, "      - uses: ./.github/actions/local\n" + HYGIENE_STEP)
    assert rules(text) == set()
