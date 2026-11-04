"""Suite-wide hermetic configuration.

* Network isolation is enforced by ``--disable-socket`` (pytest-socket) in
  ``pyproject.toml``; only Unix sockets (asyncio internals, the e2e harness) are allowed.
* Hypothesis state (example database, internal storage) lives under ``./.cache/``,
  derived from this file's location. ``HYPOTHESIS_PROFILE`` selects ``dev`` (default,
  50 examples) or ``ci`` (200 examples, derandomised, no example database).
* Every test starts without ``AMG_*``, ``NEO4J_*``, ``QDRANT_*`` or LLM-provider
  variables so the developer's shell cannot change results or reach live services.
"""

import os
from pathlib import Path

import pytest
from hypothesis import HealthCheck, settings
from hypothesis.configuration import set_hypothesis_home_dir
from hypothesis.database import DirectoryBasedExampleDatabase

REPO_ROOT = Path(__file__).resolve().parent.parent
HYPOTHESIS_HOME = REPO_ROOT / ".cache" / "hypothesis"

set_hypothesis_home_dir(HYPOTHESIS_HOME)
settings.register_profile(
    "dev",
    max_examples=50,
    deadline=None,
    database=DirectoryBasedExampleDatabase(HYPOTHESIS_HOME / "examples"),
    suppress_health_check=[HealthCheck.too_slow],
    print_blob=True,
)
settings.register_profile(
    "ci",
    parent=settings.get_profile("dev"),
    max_examples=200,
    derandomize=True,
    database=None,
)
settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "dev"))

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
