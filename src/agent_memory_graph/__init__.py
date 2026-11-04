"""agent-memory-graph: asynchronous GraphRAG long-term memory engine for AI agents."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__: str = version("agent-memory-graph")
except PackageNotFoundError:  # pragma: no cover - only when run from an uninstalled tree
    __version__ = "0.0.0"

__all__ = ["__version__"]
