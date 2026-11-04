"""Bootstrap guards: package metadata and the hermetic test harness itself."""

import os
import socket
from pathlib import Path

import pytest
from pytest_socket import SocketBlockedError

import agent_memory_graph

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_package_exposes_installed_version() -> None:
    assert agent_memory_graph.__version__ == "0.1.0"


def test_inet_sockets_are_blocked() -> None:
    with (
        pytest.warns(UserWarning, match="tried to use socket"),
        pytest.raises(SocketBlockedError),
    ):
        socket.socket(socket.AF_INET, socket.SOCK_STREAM)


async def test_async_tests_run_without_markers() -> None:
    assert agent_memory_graph.__all__ == ["__version__"]


def test_service_credentials_are_scrubbed() -> None:
    leaked = [k for k in os.environ if k.startswith(("NEO4J_", "QDRANT_", "OPENAI_"))]
    assert leaked == []


def test_pytest_cache_lives_under_dot_cache(pytestconfig: pytest.Config) -> None:
    cache_dir = Path(pytestconfig.getini("cache_dir"))
    assert cache_dir.parts[0] == ".cache"
    assert not (REPO_ROOT / ".pytest_cache").exists()
