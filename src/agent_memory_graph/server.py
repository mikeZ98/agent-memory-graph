"""Process entrypoint: ``python -m agent_memory_graph``.

Runs the ASGI app under uvicorn as the container's PID 1 (no shell wrapper), so
SIGTERM reaches uvicorn's handler directly and triggers a graceful shutdown: in-flight
requests finish, the lifespan closes the connection pools, and the process exits 0.

Uvicorn re-raises a captured shutdown signal once it has finished (so the process
would die *by* that signal). The handler installed here is what uvicorn restores
before re-raising, which turns a completed graceful shutdown into ``exit 0`` and a
failed startup into ``exit 3`` (uvicorn's ``STARTUP_FAILURE``).
"""

import argparse
import logging
import os
import signal
from collections.abc import Callable, Sequence
from types import FrameType
from typing import TYPE_CHECKING, Final, Protocol, cast

import uvicorn
from fastapi import FastAPI

from agent_memory_graph.api.app import build_memory_service, create_app
from agent_memory_graph.config import Settings
from agent_memory_graph.domain.errors import ExtractionFailedError
from agent_memory_graph.services.embedder import MockEmbeddingService
from agent_memory_graph.services.extractor import InstructorStructuredLLM, StructuredLLM

if TYPE_CHECKING:
    from instructor import AsyncInstructor
    from pydantic import BaseModel

    from agent_memory_graph.services.extractor import ChatMessage

__all__ = [
    "DEFAULT_FACTORY",
    "STARTUP_FAILURE",
    "UnconfiguredLLM",
    "create_default_app",
    "llm_from_settings",
    "main",
]

DEFAULT_FACTORY: Final = "agent_memory_graph.server:create_default_app"
STARTUP_FAILURE: Final = 3
_SHUTDOWN_SIGNALS: Final = (signal.SIGTERM, signal.SIGINT)

logger = logging.getLogger("uvicorn.error")


class _Server(Protocol):
    started: bool

    def run(self) -> None: ...


# ---------------------------------------------------------------------------
# Composition
# ---------------------------------------------------------------------------
class UnconfiguredLLM:
    """Placeholder when ``AMG_LLM_MODEL`` is unset: queries work, ingestion fails 502."""

    async def create[T: "BaseModel"](
        self,
        *,
        response_model: type[T],
        messages: "Sequence[ChatMessage]",
        max_retries: int,
    ) -> T:
        """Raise :class:`ExtractionFailedError`: there is no model to call."""
        del response_model, messages, max_retries  # the port's signature; nothing to send
        raise ExtractionFailedError("no extraction model is configured (set AMG_LLM_MODEL)")


def llm_from_settings(
    settings: Settings,
    *,
    from_provider: Callable[..., object] | None = None,
) -> StructuredLLM:
    """Build the extraction model from ``AMG_LLM_MODEL`` (``"<provider>/<model>"``).

    The provider SDK must be installed (``pip install 'agent-memory-graph[anthropic]'``
    or the provider's own package); credentials come from the provider's usual
    environment variables, e.g. ``ANTHROPIC_API_KEY``.
    """
    if settings.llm_model is None:
        logger.warning("AMG_LLM_MODEL is not set: ingestion answers AMG_EXTRACTION_FAILED")
        return UnconfiguredLLM()
    if from_provider is None:
        import instructor  # noqa: PLC0415 - provider SDKs load only when configured

        from_provider = instructor.from_provider
    client = cast("AsyncInstructor", from_provider(settings.llm_model, async_client=True))
    _, model = settings.llm_model.split("/", 1)
    return InstructorStructuredLLM(client, model=model, max_tokens=settings.llm_max_tokens)


def create_default_app() -> FastAPI:
    """Uvicorn factory for the production app, configured from ``AMG_*`` variables.

    Settings are validated here, at startup, so a misconfigured container fails
    immediately instead of on its first request.
    """
    settings = Settings()
    llm = llm_from_settings(settings)
    embedder = MockEmbeddingService.from_settings(settings)
    logger.warning(
        "using the lexical hashing embedder (%d-d); plug a semantic EmbeddingService "
        "into build_memory_service for production-grade recall",
        settings.embedding_dim,
    )
    return create_app(lambda: build_memory_service(settings, llm=llm, embedder=embedder))


# ---------------------------------------------------------------------------
# Process
# ---------------------------------------------------------------------------
def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="agent-memory-graph", description="Serve the agent-memory-graph HTTP API."
    )
    parser.add_argument(
        "--host",
        default=os.environ.get("AMG_HTTP_HOST", "127.0.0.1"),
        help="bind address (AMG_HTTP_HOST, default 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("AMG_HTTP_PORT", "8000")),
        help="TCP port (AMG_HTTP_PORT, default 8000)",
    )
    parser.add_argument("--uds", default=None, help="serve on a Unix domain socket instead")
    parser.add_argument(
        "--factory",
        default=DEFAULT_FACTORY,
        help=f"'module:callable' returning the ASGI app (default {DEFAULT_FACTORY})",
    )
    parser.add_argument(
        "--log-level",
        default=os.environ.get("AMG_LOG_LEVEL", "info"),
        choices=["critical", "error", "warning", "info", "debug"],
    )
    parser.add_argument(
        "--graceful-timeout",
        type=float,
        default=float(os.environ.get("AMG_GRACEFUL_TIMEOUT_SECONDS", "10")),
        help="seconds in-flight requests get after SIGTERM",
    )
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    server_factory: Callable[[uvicorn.Config], _Server] = uvicorn.Server,
) -> int:
    """Serve until SIGTERM/SIGINT; return the process exit code (0, or 3 on failed start)."""
    args = _parser().parse_args(argv)
    received: list[int] = []

    def remember(signum: int, _frame: FrameType | None) -> None:
        received.append(signum)

    for sig in _SHUTDOWN_SIGNALS:
        signal.signal(sig, remember)

    config = uvicorn.Config(
        args.factory,
        factory=True,
        host=args.host,
        port=args.port,
        uds=args.uds,
        lifespan="on",
        log_level=args.log_level,
        access_log=False,
        server_header=False,
        proxy_headers=False,
        timeout_graceful_shutdown=args.graceful_timeout,
    )
    server = server_factory(config)
    server.run()
    if not server.started:
        return STARTUP_FAILURE
    names = ", ".join(signal.Signals(sig).name for sig in received) or "none"
    logger.info("graceful shutdown complete (signals: %s)", names)
    return 0
