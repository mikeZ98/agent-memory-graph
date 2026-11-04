"""Domain error taxonomy and its RFC 9457 Problem Details representation.

Each :class:`MemoryGraphError` subclass is bound to one :class:`ErrorCode`, and each
code to exactly one problem ``type`` URI, ``title`` and HTTP ``status`` in
:data:`PROBLEM_TAXONOMY`. A code can therefore never be reported with conflicting
HTTP semantics, and clients can switch on ``type`` (or the ``code`` extension).
"""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import Annotated, Any, ClassVar, Final, Self

from pydantic import Field, Strict, model_validator

from agent_memory_graph.domain._base import EMPTY_JSON_MAPPING, FrozenJsonMapping, StrictModel

__all__ = [
    "PROBLEM_JSON_MEDIA_TYPE",
    "PROBLEM_TAXONOMY",
    "EntityIntegrityError",
    "ErrorCode",
    "MemoryGraphError",
    "MemoryProblemDetails",
    "ProblemSpec",
    "StoreTimeoutError",
    "StoreUnavailableError",
    "TraversalDepthExceededError",
    "VectorSearchTimeoutError",
]

PROBLEM_JSON_MEDIA_TYPE: Final = "application/problem+json"

_TYPE_URI_PREFIX: Final = "urn:agent-memory-graph:problem:"
# URI reference (RFC 3986): no whitespace, controls or always-escaped characters.
_URI_REFERENCE_PATTERN: Final = r"^[^\s\x00-\x1f\x7f<>\"{}|\\^`]+$"


class ErrorCode(StrEnum):
    """Stable machine-readable codes, exposed as the ``code`` extension member."""

    INTERNAL = "AMG_INTERNAL_ERROR"
    ENTITY_INTEGRITY = "AMG_ENTITY_INTEGRITY"
    TRAVERSAL_DEPTH_EXCEEDED = "AMG_TRAVERSAL_DEPTH_EXCEEDED"
    VECTOR_SEARCH_TIMEOUT = "AMG_VECTOR_SEARCH_TIMEOUT"
    STORE_UNAVAILABLE = "AMG_STORE_UNAVAILABLE"
    STORE_TIMEOUT = "AMG_STORE_TIMEOUT"


@dataclass(frozen=True, slots=True)
class ProblemSpec:
    """Canonical Problem Details members bound to one :class:`ErrorCode`."""

    type: str
    title: str
    status: int


PROBLEM_TAXONOMY: Final[Mapping[ErrorCode, ProblemSpec]] = MappingProxyType(
    {
        ErrorCode.INTERNAL: ProblemSpec(
            type=f"{_TYPE_URI_PREFIX}internal-error",
            title="Internal memory graph error",
            status=500,
        ),
        ErrorCode.ENTITY_INTEGRITY: ProblemSpec(
            type=f"{_TYPE_URI_PREFIX}entity-integrity",
            title="Knowledge graph integrity violated",
            status=422,
        ),
        ErrorCode.TRAVERSAL_DEPTH_EXCEEDED: ProblemSpec(
            type=f"{_TYPE_URI_PREFIX}traversal-depth-exceeded",
            title="Graph traversal depth exceeds the allowed maximum",
            status=422,
        ),
        ErrorCode.VECTOR_SEARCH_TIMEOUT: ProblemSpec(
            type=f"{_TYPE_URI_PREFIX}vector-search-timeout",
            title="Vector search exceeded its deadline",
            status=504,
        ),
        ErrorCode.STORE_UNAVAILABLE: ProblemSpec(
            type=f"{_TYPE_URI_PREFIX}store-unavailable",
            title="A backing store is unavailable",
            status=503,
        ),
        ErrorCode.STORE_TIMEOUT: ProblemSpec(
            type=f"{_TYPE_URI_PREFIX}store-timeout",
            title="A backing store operation exceeded its deadline",
            status=504,
        ),
    }
)

UriReference = Annotated[str, Strict(), Field(min_length=1, max_length=2048)]


class MemoryProblemDetails(StrictModel):
    """RFC 9457 Problem Details body with ``code`` and ``context`` extension members.

    ``type``, ``title`` and ``status`` must equal the :data:`PROBLEM_TAXONOMY` entry
    for ``code``; build instances with :meth:`from_code` rather than by hand.
    """

    type: Annotated[UriReference, Field(pattern=_URI_REFERENCE_PATTERN)]
    title: Annotated[str, Strict(), Field(min_length=1, max_length=200)]
    status: Annotated[int, Strict(), Field(ge=400, le=599)]
    detail: Annotated[str, Strict(), Field(min_length=1, max_length=4096)]
    instance: Annotated[UriReference, Field(pattern=_URI_REFERENCE_PATTERN)] | None = None
    code: ErrorCode
    context: FrozenJsonMapping = Field(default_factory=EMPTY_JSON_MAPPING)

    @model_validator(mode="after")
    def _check_taxonomy(self) -> Self:
        spec = PROBLEM_TAXONOMY[self.code]
        if (self.type, self.title, self.status) != (spec.type, spec.title, spec.status):
            msg = (
                f"type/title/status do not match {self.code.value} "
                f"(expected {spec.type!r}, {spec.title!r}, {spec.status})"
            )
            raise ValueError(msg)
        return self

    @classmethod
    def from_code(
        cls,
        code: ErrorCode,
        detail: str,
        *,
        instance: str | None = None,
        context: Mapping[str, Any] | None = None,
    ) -> Self:
        """Build a problem whose ``type``, ``title`` and ``status`` come from the taxonomy."""
        spec = PROBLEM_TAXONOMY[code]
        return cls(
            type=spec.type,
            title=spec.title,
            status=spec.status,
            detail=detail,
            instance=instance,
            code=code,
            context=dict(context or {}),
        )

    def to_json_dict(self) -> dict[str, Any]:
        """Serialise as an ``application/problem+json`` body (absent members omitted)."""
        return self.model_dump(mode="json", exclude_none=True)


# ---------------------------------------------------------------------------
# Exception hierarchy
# ---------------------------------------------------------------------------
class MemoryGraphError(Exception):
    """Root of every domain failure; maps losslessly onto :class:`MemoryProblemDetails`."""

    code: ClassVar[ErrorCode] = ErrorCode.INTERNAL

    def __init__(
        self,
        detail: str,
        *,
        instance: str | None = None,
        context: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(detail)
        self.detail = detail
        self.instance = instance
        self.context: Mapping[str, Any] = MappingProxyType(dict(context or {}))

    @property
    def status(self) -> int:
        """HTTP status code this error is reported with."""
        return PROBLEM_TAXONOMY[self.code].status

    def to_problem(self) -> MemoryProblemDetails:
        """Render this error as an RFC 9457 problem."""
        return MemoryProblemDetails.from_code(
            self.code, self.detail, instance=self.instance, context=self.context
        )


class EntityIntegrityError(MemoryGraphError):
    """A write would leave relations pointing at missing or duplicated entities."""

    code: ClassVar[ErrorCode] = ErrorCode.ENTITY_INTEGRITY

    def __init__(
        self,
        detail: str,
        *,
        dangling_ids: Iterable[str] = (),
        duplicate_ids: Iterable[str] = (),
        instance: str | None = None,
    ) -> None:
        self.dangling_ids = tuple(sorted(set(dangling_ids)))
        self.duplicate_ids = tuple(sorted(set(duplicate_ids)))
        super().__init__(
            detail,
            instance=instance,
            context={"dangling_ids": self.dangling_ids, "duplicate_ids": self.duplicate_ids},
        )


class TraversalDepthExceededError(MemoryGraphError):
    """A traversal was asked to go deeper than the engine allows."""

    code: ClassVar[ErrorCode] = ErrorCode.TRAVERSAL_DEPTH_EXCEEDED

    def __init__(self, requested: int, maximum: int, *, instance: str | None = None) -> None:
        self.requested = requested
        self.maximum = maximum
        super().__init__(
            f"Requested traversal depth {requested} exceeds the maximum of {maximum}.",
            instance=instance,
            context={"requested_depth": requested, "max_depth": maximum},
        )


class VectorSearchTimeoutError(MemoryGraphError):
    """The vector store did not answer the similarity search within its deadline."""

    code: ClassVar[ErrorCode] = ErrorCode.VECTOR_SEARCH_TIMEOUT

    def __init__(
        self,
        timeout_seconds: float,
        *,
        collection: str | None = None,
        instance: str | None = None,
    ) -> None:
        self.timeout_seconds = timeout_seconds
        self.collection = collection
        where = f" on collection {collection!r}" if collection else ""
        context: dict[str, Any] = {"timeout_seconds": timeout_seconds}
        if collection:
            context["collection"] = collection
        super().__init__(
            f"Vector search{where} did not complete within {timeout_seconds:g}s.",
            instance=instance,
            context=context,
        )


class StoreUnavailableError(MemoryGraphError):
    """A backing store (graph or vector) cannot be reached or refused the session."""

    code: ClassVar[ErrorCode] = ErrorCode.STORE_UNAVAILABLE

    def __init__(self, store: str, *, instance: str | None = None) -> None:
        self.store = store
        super().__init__(
            f"The {store} store is unavailable.", instance=instance, context={"store": store}
        )


class StoreTimeoutError(MemoryGraphError):
    """A backing store operation other than vector search exceeded its deadline."""

    code: ClassVar[ErrorCode] = ErrorCode.STORE_TIMEOUT

    def __init__(
        self,
        store: str,
        operation: str,
        timeout_seconds: float,
        *,
        instance: str | None = None,
    ) -> None:
        self.store = store
        self.operation = operation
        self.timeout_seconds = timeout_seconds
        super().__init__(
            f"The {store} {operation} operation did not complete within {timeout_seconds:g}s.",
            instance=instance,
            context={"store": store, "operation": operation, "timeout_seconds": timeout_seconds},
        )
