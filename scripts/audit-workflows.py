"""Offline policy checks for GitHub Actions workflows (no network, no GitHub API).

* YAML parses; no ``pull_request_target`` / ``workflow_run`` triggers.
* Top-level ``permissions`` is exactly ``contents: read``; job overrides are read-only.
* Every job runs on the pinned ``ubuntu-24.04`` image and has ``timeout-minutes``.
* Every remote action is pinned by a full 40-hex commit SHA with a ``# vX.Y.Z`` comment,
  one SHA per action across the workflow; ``docker://`` actions are digest-pinned.
* ``actions/checkout`` never persists credentials.
* No untrusted ``${{ github.event.* }}`` / ``github.head_ref`` / ``inputs.*`` expression
  is interpolated into a ``run:`` script (pass it through ``env:`` instead).
* For ``ci.yml``: the attribution, quality and container gates exist and are wired as
  documented (full history, CI fuzzing profile, coverage gate, fail-closed Trivy).

Usage: python scripts/audit-workflows.py [workflow.yml ...]  (default: .github/workflows/*)
"""

import re
import sys
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import yaml

RUNNER: Final = "ubuntu-24.04"
CI_JOBS: Final = ("attribution-audit", "code-quality", "container-security")
_SHA_PIN_RE: Final = re.compile(r"^[\w.-]+/[\w./-]+@[0-9a-f]{40}$")
_USES_LINE_RE: Final = re.compile(r"^\s*(?:-\s*)?uses:\s*([^\s#]+)\s*(?:#\s*(\S+))?")
_VERSION_COMMENT_RE: Final = re.compile(r"^v\d+(?:\.\d+){0,2}$")
_UNTRUSTED_RE: Final = re.compile(
    r"\$\{\{\s*(?:github\.event\.|github\.head_ref|inputs\.|github\.event_path)[^}]*\}\}"
)


@dataclass(frozen=True, slots=True)
class Violation:
    """One failed check."""

    rule: str
    message: str
    location: str

    def __str__(self) -> str:
        return f"{self.location}: [{self.rule}] {self.message}"


def _steps(job: Mapping[str, Any]) -> Iterator[Mapping[str, Any]]:
    for step in job.get("steps") or []:
        if isinstance(step, Mapping):
            yield step


def _uses_lines(text: str) -> Iterator[tuple[int, str, str | None]]:
    for number, line in enumerate(text.splitlines(), start=1):
        match = _USES_LINE_RE.match(line)
        if match:
            yield number, match.group(1), match.group(2)


def check_workflow(text: str, location: str) -> list[Violation]:
    """Return every policy violation in one workflow file."""
    out: list[Violation] = []

    def fail(rule: str, message: str, where: str = "") -> None:
        out.append(Violation(rule, message, f"{location}{':' + where if where else ''}"))

    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        fail("AMG-W001", f"invalid YAML: {exc}".splitlines()[0])
        return out
    if not isinstance(data, Mapping):
        fail("AMG-W001", "workflow must be a mapping")
        return out

    triggers = data.get("on", data.get(True))  # YAML 1.1 reads a bare `on` key as True
    trigger_names = set(triggers) if isinstance(triggers, Mapping) else set(triggers or [])
    for dangerous in ("pull_request_target", "workflow_run"):
        if dangerous in trigger_names:
            fail("AMG-W002", f"{dangerous} runs untrusted code with privileges")

    if data.get("permissions") != {"contents": "read"}:
        fail("AMG-W003", "top-level permissions must be exactly {contents: read}")

    jobs: Mapping[str, Any] = data.get("jobs") or {}
    for name, job in jobs.items():
        where = f"jobs.{name}"
        if job.get("runs-on") != RUNNER:
            fail("AMG-W004", f"runs-on must be {RUNNER}", where)
        if not isinstance(job.get("timeout-minutes"), int):
            fail("AMG-W007", "timeout-minutes required", where)
        permissions = job.get("permissions")
        if permissions == "write-all" or (
            isinstance(permissions, Mapping) and set(permissions.values()) - {"read", "none"}
        ):
            fail("AMG-W003", "job permissions may only be read or none", where)
        if job.get("secrets") == "inherit":
            fail("AMG-W009", "secrets: inherit is forbidden", where)
        for index, step in enumerate(_steps(job)):
            step_where = f"{where}.steps[{index}]"
            uses = str(step.get("uses", ""))
            persists = (step.get("with") or {}).get("persist-credentials") is not False
            if uses.startswith("actions/checkout@") and persists:
                fail("AMG-W006", "checkout must set persist-credentials: false", step_where)
            run = str(step.get("run", ""))
            if _UNTRUSTED_RE.search(run):
                fail("AMG-W008", "untrusted expression interpolated into run:", step_where)

    pins: dict[str, set[str]] = {}
    for number, reference, comment in _uses_lines(text):
        where = f"{number}"
        if reference.startswith("./"):
            continue
        if reference.startswith("docker://"):
            if "@sha256:" not in reference:
                fail("AMG-W005", f"docker action not pinned by digest: {reference}", where)
            continue
        if not _SHA_PIN_RE.match(reference):
            fail("AMG-W005", f"action not pinned by a full commit SHA: {reference}", where)
            continue
        if not comment or not _VERSION_COMMENT_RE.match(comment):
            fail("AMG-W005", f"pinned action needs a '# vX.Y.Z' comment: {reference}", where)
        action, sha = reference.split("@")
        pins.setdefault(action, set()).add(sha)
    for action, shas in sorted(pins.items()):
        if len(shas) > 1:
            fail("AMG-W005", f"{action} is pinned to {len(shas)} different SHAs")

    if Path(location).name == "ci.yml":
        out += _check_ci_contract(data, jobs, trigger_names, triggers, location)
    return out


def _step_runs(job: Mapping[str, Any]) -> str:
    return "\n".join(str(step.get("run", "")) for step in _steps(job))


def _check_ci_contract(
    data: Mapping[str, Any],
    jobs: Mapping[str, Any],
    trigger_names: set[Any],
    triggers: Any,  # noqa: ANN401 - raw YAML
    location: str,
) -> list[Violation]:
    out: list[Violation] = []

    def fail(rule: str, message: str) -> None:
        out.append(Violation(rule, message, location))

    push = triggers.get("push") if isinstance(triggers, Mapping) else None
    if "pull_request" not in trigger_names or "main" not in ((push or {}).get("branches") or []):
        fail("AMG-W010", "CI must run on pushes to main and on pull requests")
    for name in CI_JOBS:
        if name not in jobs:
            fail("AMG-W010", f"job {name} missing")
    if out:
        return out

    audit = jobs["attribution-audit"]
    checkout = next(
        (s for s in _steps(audit) if str(s.get("uses", "")).startswith("actions/checkout@")), {}
    )
    if (checkout.get("with") or {}).get("fetch-depth") != 0:
        fail("AMG-W011", "attribution-audit needs the full history (fetch-depth: 0)")
    audit_runs = _step_runs(audit)
    for script in ("scripts/audit-attribution.sh", "scripts/audit-hygiene.sh"):
        if script not in audit_runs:
            fail("AMG-W011", f"attribution-audit must run {script}")

    quality = jobs["code-quality"]
    quality_runs = _step_runs(quality)
    for command in ("ruff check", "ruff format --check", "mypy", "bandit -c pyproject.toml -r src",
                    "pytest", "scripts/coverage-gate.sh", "scripts/audit-workflows.py",
                    "scripts/audit-cache-leaks.sh"):  # fmt: skip
        if command not in quality_runs:
            fail("AMG-W013", f"code-quality must run `{command}`")
    pytest_step = next((s for s in _steps(quality) if "pytest" in str(s.get("run", ""))), {})
    if (pytest_step.get("env") or {}).get("HYPOTHESIS_PROFILE") != "ci":
        fail("AMG-W013", "tests must run with HYPOTHESIS_PROFILE=ci")
    if any(str(s.get("uses", "")).startswith("astral-sh/setup-uv@") for s in _steps(quality)):
        setup = next(
            s for s in _steps(quality) if str(s.get("uses", "")).startswith("astral-sh/setup-uv@")
        )
        options = setup.get("with") or {}
        if str(options.get("python-version")) != "3.12" or options.get("enable-cache") is not True:
            fail("AMG-W013", "setup-uv must install Python 3.12 with the cache enabled")
    else:
        fail("AMG-W013", "code-quality must set up Python with astral-sh/setup-uv")

    container = jobs["container-security"]
    container_steps = list(_steps(container))
    if not any("smoke-docker.sh" in str(s.get("run", "")) for s in container_steps):
        fail("AMG-W012", "container-security must run scripts/smoke-docker.sh")
    builds = [
        s for s in container_steps if str(s.get("uses", "")).startswith("docker/build-push-action@")
    ]
    if not builds or (builds[0].get("with") or {}).get("file") != "Dockerfile":
        fail("AMG-W012", "container-security must build the local Dockerfile")
    gating = [
        s.get("with") or {}
        for s in container_steps
        if str(s.get("uses", "")).startswith("aquasecurity/trivy-action@")
        and str((s.get("with") or {}).get("exit-code")) == "1"
    ]
    if not gating:
        fail("AMG-W012", "a Trivy step must fail closed (exit-code: 1)")
    else:
        trivy = gating[0]
        if trivy.get("ignore-unfixed") is not True or not {"CRITICAL", "HIGH"} <= set(
            str(trivy.get("severity", "")).split(",")
        ):
            fail("AMG-W012", "gating Trivy scan must cover CRITICAL,HIGH with ignore-unfixed")
        if not str(trivy.get("version", "")).startswith("v"):
            fail("AMG-W012", "pin the Trivy binary version (with: version: vX.Y.Z)")
    if data.get("permissions") != {"contents": "read"}:
        fail("AMG-W003", "CI must stay read-only")
    return out


def main(argv: Sequence[str] | None = None) -> int:
    """Print violations for each workflow; return 0 when all pass."""
    args = list(sys.argv[1:] if argv is None else argv)
    paths = [Path(p) for p in args] or sorted(Path(".github/workflows").glob("*.y*ml"))
    if not paths:
        print("no workflows found", file=sys.stderr)
        return 1
    violations: list[Violation] = []
    for path in paths:
        if not path.is_file():
            violations.append(Violation("AMG-W001", "file not found", str(path)))
            continue
        violations += check_workflow(path.read_text(), str(path))
    for violation in violations:
        print(f"  x {violation}", file=sys.stderr)
    if not violations:
        print(f"workflow policy passed ({len(paths)} file(s))", file=sys.stderr)
    return 1 if violations else 0


if __name__ == "__main__":
    sys.exit(main())
