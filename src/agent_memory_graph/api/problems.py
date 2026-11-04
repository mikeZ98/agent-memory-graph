"""RFC 9457 Problem Details for every error the API can produce.

* :class:`MemoryGraphError` (and subclasses) → its taxonomy entry, with ``instance``
  defaulting to the request path;
* request validation failures → ``AMG_INVALID_INPUT`` (422), rejected values omitted;
* framework HTTP errors (404, 405, 413, ...) → ``type: about:blank`` as RFC 9457 §4.2.1
  prescribes for problems with no further semantics;
* anything else → ``AMG_INTERNAL_ERROR`` (500) with a generic detail; the exception is
  logged, never returned.
"""

import logging
from collections.abc import Mapping
from http import HTTPStatus
from typing import Any, Final, cast

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from agent_memory_graph.domain.errors import (
    PROBLEM_JSON_MEDIA_TYPE,
    ErrorCode,
    InvalidInputError,
    MemoryGraphError,
    MemoryProblemDetails,
)

__all__ = ["PROBLEM_RESPONSES", "install_problem_handlers", "problem_response"]

logger = logging.getLogger(__name__)

_SERVER_ERROR: Final = 500

PROBLEM_RESPONSES: Final[dict[int | str, dict[str, Any]]] = {
    status: {
        "model": MemoryProblemDetails,
        "description": HTTPStatus(status).phrase,
        "content": {PROBLEM_JSON_MEDIA_TYPE: {}},
    }
    for status in (422, 500, 502, 503, 504)
}
"""OpenAPI ``responses`` entries documenting the problem+json error bodies."""


def problem_response(
    body: dict[str, Any], status: int, headers: Mapping[str, str] | None = None
) -> JSONResponse:
    """Serialise ``body`` as ``application/problem+json``."""
    return JSONResponse(
        body, status_code=status, media_type=PROBLEM_JSON_MEDIA_TYPE, headers=headers
    )


def _render(error: MemoryGraphError, request: Request) -> JSONResponse:
    problem = MemoryProblemDetails.from_code(
        error.code,
        error.detail,
        instance=error.instance or request.url.path,
        context=error.context,
    )
    if problem.status >= _SERVER_ERROR:
        logger.error("%s %s → %s: %s", request.method, request.url.path, error.code.value,
                     error.detail, exc_info=error)  # fmt: skip
    else:
        logger.info("%s %s → %s", request.method, request.url.path, error.code.value)
    retry_after = problem.context.get("retry_after_seconds")
    headers = {"Retry-After": str(retry_after)} if isinstance(retry_after, int) else None
    return problem_response(problem.to_json_dict(), problem.status, headers=headers)


async def _memory_error_handler(request: Request, exc: Exception) -> JSONResponse:
    # Starlette types handlers as taking Exception; each is registered for one type.
    return _render(cast("MemoryGraphError", exc), request)


async def _validation_error_handler(request: Request, exc: Exception) -> JSONResponse:
    errors = cast("RequestValidationError", exc).errors()
    error = InvalidInputError("The request does not match the API contract.", errors=errors)
    return _render(error, request)


async def _http_error_handler(request: Request, exc: Exception) -> JSONResponse:
    exc = cast("StarletteHTTPException", exc)
    body = {
        "type": "about:blank",
        "title": HTTPStatus(exc.status_code).phrase,
        "status": exc.status_code,
        "detail": exc.detail,
        "instance": request.url.path,
    }
    return problem_response(body, exc.status_code, headers=exc.headers)


async def _unhandled_error_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("unhandled error on %s %s", request.method, request.url.path, exc_info=exc)
    error = MemoryGraphError("An unexpected error occurred.")
    problem = MemoryProblemDetails.from_code(
        ErrorCode.INTERNAL, error.detail, instance=request.url.path
    )
    return problem_response(problem.to_json_dict(), problem.status)


def install_problem_handlers(app: FastAPI) -> None:
    """Register the RFC 9457 exception handlers on ``app``."""
    app.add_exception_handler(MemoryGraphError, _memory_error_handler)
    app.add_exception_handler(RequestValidationError, _validation_error_handler)
    app.add_exception_handler(StarletteHTTPException, _http_error_handler)
    app.add_exception_handler(Exception, _unhandled_error_handler)
