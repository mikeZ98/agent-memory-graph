"""ASGI middleware bounding request bodies before any of them is parsed."""

from typing import Final

from starlette.exceptions import HTTPException
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from agent_memory_graph.api.problems import problem_response

__all__ = ["DEFAULT_MAX_BODY_BYTES", "BodySizeLimitMiddleware"]

DEFAULT_MAX_BODY_BYTES: Final = 1 << 20  # 1 MiB
_TOO_LARGE: Final = 413


def _too_large(max_bytes: int) -> str:
    return f"Request body exceeds {max_bytes} bytes."


class BodySizeLimitMiddleware:
    """Reject bodies larger than ``max_bytes`` with 413, without buffering them.

    A declared ``Content-Length`` above the limit is refused before the app runs;
    bodies without one (chunked uploads) are counted as they stream and aborted at
    the limit. The streaming case raises :class:`HTTPException`, which FastAPI
    re-raises from body parsing, so it is rendered by the problem handlers.
    """

    def __init__(self, app: ASGIApp, *, max_bytes: int = DEFAULT_MAX_BODY_BYTES) -> None:
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Pass the request through, refusing or truncating oversized bodies."""
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        declared = dict(scope["headers"]).get(b"content-length")
        if declared is not None and (not declared.isdigit() or int(declared) > self.max_bytes):
            path = scope.get("path", "")
            body = {
                "type": "about:blank",
                "title": "Content Too Large",
                "status": _TOO_LARGE,
                "detail": _too_large(self.max_bytes),
                "instance": path,
            }
            await problem_response(body, _TOO_LARGE)(scope, receive, send)
            return

        received = 0

        async def limited_receive() -> Message:
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self.max_bytes:
                    raise HTTPException(_TOO_LARGE, _too_large(self.max_bytes))
            return message

        await self.app(scope, limited_receive, send)
