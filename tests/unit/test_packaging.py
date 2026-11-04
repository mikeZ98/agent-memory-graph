"""Packaging: everything ``src/`` imports at runtime ships with the runtime install.

The container installs with ``uv sync --no-dev``; a module that is only available
through the dev group (as uvicorn once was) works in every local run and test, then
fails at image build. This test reproduces that check without Docker.
"""

import ast
import sys
import tomllib
from collections.abc import Iterator
from importlib import metadata
from pathlib import Path
from typing import Final

import pytest
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

REPO_ROOT: Final = Path(__file__).resolve().parents[2]
SRC: Final = REPO_ROOT / "src" / "agent_memory_graph"
PYPROJECT: Final = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())


def _is_type_checking_block(node: ast.AST) -> bool:
    return isinstance(node, ast.If) and (
        (isinstance(node.test, ast.Name) and node.test.id == "TYPE_CHECKING")
        or (isinstance(node.test, ast.Attribute) and node.test.attr == "TYPE_CHECKING")
    )


def _runtime_imports(tree: ast.AST) -> Iterator[str]:
    """Top-level module names imported at runtime (TYPE_CHECKING blocks excluded)."""
    stack: list[ast.AST] = [tree]
    while stack:
        node = stack.pop()
        if _is_type_checking_block(node):
            continue
        if isinstance(node, ast.Import):
            yield from (alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            yield node.module.split(".")[0]
        stack.extend(ast.iter_child_nodes(node))


def third_party_runtime_modules() -> dict[str, set[Path]]:
    found: dict[str, set[Path]] = {}
    for path in SRC.rglob("*.py"):
        for module in _runtime_imports(ast.parse(path.read_text())):
            if module in sys.stdlib_module_names or module == "agent_memory_graph":
                continue
            found.setdefault(module, set()).add(path.relative_to(REPO_ROOT))
    return found


def runtime_closure() -> set[str]:
    """Canonical names of every distribution the runtime install brings in (no extras)."""
    pending = [Requirement(spec) for spec in PYPROJECT["project"]["dependencies"]]
    seen: set[str] = set()
    while pending:
        requirement = pending.pop()
        if requirement.marker is not None and not requirement.marker.evaluate({"extra": ""}):
            continue
        name = canonicalize_name(requirement.name)
        if name in seen:
            continue
        seen.add(name)
        pending.extend(Requirement(spec) for spec in metadata.requires(requirement.name) or [])
    return seen


def test_runtime_imports_are_shipped_by_runtime_dependencies() -> None:
    providers = metadata.packages_distributions()
    closure = runtime_closure()
    missing = {
        module: sorted(str(p) for p in paths)
        for module, paths in third_party_runtime_modules().items()
        if not {canonicalize_name(d) for d in providers.get(module, [])} & closure
    }
    assert missing == {}, f"imported at runtime but not installed by `uv sync --no-dev`: {missing}"


def test_dev_tools_are_not_runtime_dependencies() -> None:
    runtime = {canonicalize_name(Requirement(s).name) for s in PYPROJECT["project"]["dependencies"]}
    dev_only = {"pytest", "pytest-asyncio", "pytest-cov", "pytest-socket", "hypothesis", "ruff",
                "mypy", "bandit", "pyyaml", "types-pyyaml"}  # fmt: skip
    assert runtime.isdisjoint(dev_only)


def test_entrypoint_dependencies_are_declared_directly() -> None:
    runtime = {canonicalize_name(Requirement(s).name) for s in PYPROJECT["project"]["dependencies"]}
    assert {"uvicorn", "fastapi", "pydantic-settings"} <= runtime


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("import uvicorn\n", {"uvicorn"}),
        ("from fastapi.responses import JSONResponse\n", {"fastapi"}),
        ("from typing import TYPE_CHECKING\nif TYPE_CHECKING:\n    import openai\n", {"typing"}),
        ("import typing\nif typing.TYPE_CHECKING:\n    import openai\n", {"typing"}),
        ("def f():\n    import instructor\n", {"instructor"}),
        ("from . import sibling\n", set()),
    ],
)
def test_import_scanner(source: str, expected: set[str]) -> None:
    assert set(_runtime_imports(ast.parse(source))) == expected
