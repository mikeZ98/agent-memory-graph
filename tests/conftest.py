"""Suite-wide hermetic configuration.

* Network isolation is enforced by ``--disable-socket`` (pytest-socket) in
  ``pyproject.toml``; only Unix sockets (asyncio internals) are allowed.
* Every test starts without ``AMG_*``, ``NEO4J_*``, ``QDRANT_*`` or LLM-provider
  variables so the developer's shell cannot change results or reach live services.
"""

import os

import pytest

_ISOLATED_PREFIXES = (
    "AMG_",
    "NEO4J_",
    "QDRANT_",
    "OPENAI_",
    "ANTHROPIC_",
    "INSTRUCTOR_",
)


@pytest.fixture(autouse=True)
def _isolate_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in [key for key in os.environ if key.startswith(_ISOLATED_PREFIXES)]:
        monkeypatch.delenv(name)
