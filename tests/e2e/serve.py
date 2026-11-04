"""Run the e2e app under uvicorn on a Unix domain socket.

Exit codes: ``0`` after a graceful shutdown (SIGTERM / SIGINT), ``3`` when startup
fails (uvicorn's ``STARTUP_FAILURE``).

Uvicorn re-raises a captured shutdown signal once it has finished shutting down
(so a supervisor sees the process die *by* that signal). The handler installed here
is what uvicorn restores before re-raising, which turns a completed graceful
shutdown into an explicit, observable ``exit 0``.

Usage: ``python -X dev -m tests.e2e.serve --uds .cache/e2e/<name>.sock``
"""

import argparse
import signal
import sys
from types import FrameType
from typing import Final

import uvicorn

GRACEFUL_MARKER: Final = "amg-e2e: graceful shutdown complete"
STARTUP_FAILURE: Final = 3


def main(argv: list[str] | None = None) -> int:
    """Serve until a shutdown signal arrives; return the process exit code."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--uds", required=True, help="Unix domain socket path to bind")
    args = parser.parse_args(argv)

    received: list[int] = []

    def remember(signum: int, _frame: FrameType | None) -> None:
        received.append(signum)

    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, remember)

    config = uvicorn.Config(
        "tests.e2e.app:create_e2e_app",
        factory=True,
        uds=args.uds,
        lifespan="on",
        log_level="info",
        access_log=False,
        timeout_graceful_shutdown=5,
    )
    server = uvicorn.Server(config)
    server.run()
    if not server.started:
        return STARTUP_FAILURE
    print(GRACEFUL_MARKER, f"(dev_mode={sys.flags.dev_mode}, signals={received})", flush=True)  # noqa: T201
    return 0


if __name__ == "__main__":
    sys.exit(main())
