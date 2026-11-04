"""Asynchronous Qdrant adapter: entity embeddings in, seed entity ids out.

Qdrant point ids must be unsigned ints or UUIDs, so each entity is stored under a
UUIDv5 derived from its domain id; the domain id travels in the ``entity_id``
payload field and is what searches return.
"""

import asyncio
import math
import uuid
from collections.abc import Awaitable, Callable, Sequence
from types import TracebackType
from typing import Annotated, Final, Self

import httpx
from pydantic import Field, SecretStr, Strict, TypeAdapter, ValidationError
from qdrant_client import AsyncQdrantClient
from qdrant_client.http.exceptions import (
    ApiException,
    ResponseHandlingException,
    UnexpectedResponse,
)
from qdrant_client.http.models import (
    CollectionInfo,
    Distance,
    FieldCondition,
    Filter,
    MatchAny,
    PayloadSchemaType,
    PointStruct,
    VectorParams,
)

from agent_memory_graph.config import Settings
from agent_memory_graph.domain._base import Probability
from agent_memory_graph.domain.errors import (
    MemoryGraphError,
    StoreTimeoutError,
    StoreUnavailableError,
    VectorSearchTimeoutError,
)
from agent_memory_graph.domain.models import Embedding, Entity, EntityId, EntityType
from agent_memory_graph.domain.query import MAX_TOP_K

__all__ = [
    "PAYLOAD_ENTITY_ID",
    "PAYLOAD_ENTITY_TYPE",
    "PAYLOAD_NAME",
    "POINT_NAMESPACE",
    "STORE_NAME",
    "QdrantVectorStore",
    "point_id",
]

STORE_NAME: Final = "qdrant"
PAYLOAD_ENTITY_ID: Final = "entity_id"
PAYLOAD_NAME: Final = "name"
PAYLOAD_ENTITY_TYPE: Final = "entity_type"
POINT_NAMESPACE: Final = uuid.uuid5(uuid.NAMESPACE_URL, "urn:agent-memory-graph:qdrant-point")

_UNAVAILABLE_STATUSES: Final = frozenset({502, 503})
_AUTH_STATUSES: Final = frozenset({401, 403})
_CONFLICT: Final = 409
_ENTITY_ID: Final = TypeAdapter(EntityId)
_EMBEDDING: Final = TypeAdapter(Embedding)
_TOP_K: Final = TypeAdapter(Annotated[int, Strict(), Field(ge=1, le=MAX_TOP_K)])
_SCORE_THRESHOLD: Final = TypeAdapter(Probability)
_DRIVER_ERRORS: Final = (TimeoutError, httpx.HTTPError, ApiException)

type ClientFactory = Callable[..., AsyncQdrantClient]


def point_id(entity_id: str) -> str:
    """Return the deterministic Qdrant point id (UUIDv5) for a domain entity id."""
    return str(uuid.uuid5(POINT_NAMESPACE, entity_id))


def _root_cause(exc: BaseException) -> BaseException:
    return exc.source if isinstance(exc, ResponseHandlingException) else exc


def _is_timeout(exc: BaseException) -> bool:
    cause = _root_cause(exc)
    return isinstance(cause, TimeoutError | httpx.TimeoutException) or (
        isinstance(cause, UnexpectedResponse) and cause.status_code == httpx.codes.GATEWAY_TIMEOUT
    )


def _translate(  # noqa: PLR0911 - one return per mapped failure class
    exc: BaseException,
    operation: str,
    timeout_seconds: float,
    *,
    collection: str,
    search: bool,
) -> MemoryGraphError:
    if _is_timeout(exc):
        if search:
            return VectorSearchTimeoutError(timeout_seconds, collection=collection)
        return StoreTimeoutError(STORE_NAME, operation, timeout_seconds)
    cause = _root_cause(exc)
    if isinstance(cause, httpx.TransportError):
        return StoreUnavailableError(STORE_NAME)
    context: dict[str, object] = {
        "store": STORE_NAME,
        "operation": operation,
        "collection": collection,
    }
    if isinstance(cause, UnexpectedResponse):
        if cause.status_code in _UNAVAILABLE_STATUSES:
            return StoreUnavailableError(STORE_NAME)
        if cause.status_code in _AUTH_STATUSES:
            return MemoryGraphError("Qdrant rejected the configured API key.", context=context)
        context["http_status"] = cause.status_code
        return MemoryGraphError(f"Qdrant {operation} failed.", context=context)
    return MemoryGraphError(f"Qdrant client error during {operation}.", context=context)


def _check_vector(vector: Sequence[float], dimension: int, what: str) -> list[float]:
    try:
        values = _EMBEDDING.validate_python(vector)
    except ValidationError as exc:
        msg = f"{what} must be a non-empty sequence of finite numbers"
        raise ValueError(msg) from exc
    if len(values) != dimension:
        msg = f"{what} has {len(values)} dimensions, expected {dimension}"
        raise ValueError(msg)
    return list(values)


class QdrantVectorStore:
    """Owns one :class:`~qdrant_client.AsyncQdrantClient` for its lifetime.

    Use as ``async with QdrantVectorStore.from_settings(settings) as store: ...`` or
    call :meth:`connect` / :meth:`close` explicitly. Both are idempotent.
    """

    def __init__(
        self,
        url: str,
        *,
        collection: str,
        embedding_dim: int,
        api_key: SecretStr | None = None,
        timeout_seconds: float = 10.0,
        batch_size: int = 256,
        client_factory: ClientFactory = AsyncQdrantClient,
    ) -> None:
        self._url = url
        self._api_key = api_key
        self.collection = collection
        self.embedding_dim = embedding_dim
        self._timeout = timeout_seconds
        self._batch_size = batch_size
        self._client_factory = client_factory
        self._client: AsyncQdrantClient | None = None
        self._lock = asyncio.Lock()

    @classmethod
    def from_settings(
        cls, settings: Settings, *, client_factory: ClientFactory = AsyncQdrantClient
    ) -> Self:
        """Build an adapter from the ``AMG_QDRANT_*`` and embedding settings."""
        return cls(
            settings.qdrant_url,
            collection=settings.qdrant_collection,
            embedding_dim=settings.embedding_dim,
            api_key=settings.qdrant_api_key,
            timeout_seconds=settings.default_timeout_seconds,
            batch_size=settings.qdrant_upsert_batch_size,
            client_factory=client_factory,
        )

    @property
    def is_connected(self) -> bool:
        """Whether a client is currently held."""
        return self._client is not None

    @property
    def _request_timeout(self) -> int:
        # The client takes whole seconds; never round a sub-second budget down to 0.
        return max(1, math.ceil(self._timeout))

    # -- lifecycle ----------------------------------------------------------
    async def connect(self) -> None:
        """Create the client. Connections are opened lazily by the first request."""
        async with self._lock:
            if self._client is not None:
                return
            self._client = self._client_factory(
                url=self._url,
                api_key=self._api_key.get_secret_value() if self._api_key else None,
                timeout=self._request_timeout,
                prefer_grpc=False,
                # The compatibility probe runs on a background thread at construction
                # time; it would be an unbounded, unmapped network call.
                check_compatibility=False,
            )

    async def close(self) -> None:
        """Close the client and its HTTP pool; a no-op when not connected."""
        async with self._lock:
            client, self._client = self._client, None
            if client is not None:
                await client.close()

    async def __aenter__(self) -> Self:
        await self.connect()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.close()

    # -- execution ----------------------------------------------------------
    async def _call[T](
        self,
        operation: str,
        call: Callable[[AsyncQdrantClient], Awaitable[T]],
        *,
        collection: str,
        search: bool = False,
    ) -> T:
        client = self._client
        if client is None:
            msg = f"QdrantVectorStore.{operation}() called before connect()"
            raise MemoryGraphError(msg, context={"store": STORE_NAME})
        try:
            async with asyncio.timeout(self._timeout):
                return await call(client)
        except _DRIVER_ERRORS as exc:
            raise _translate(
                exc, operation, self._timeout, collection=collection, search=search
            ) from exc

    # -- operations ---------------------------------------------------------
    async def ensure_collection(self, collection_name: str | None = None) -> None:
        """Create the collection (Cosine, ``embedding_dim``) if absent; verify it if present.

        Also ensures a keyword index on ``entity_type`` so type-filtered searches stay
        index-backed. Safe to call concurrently from several processes.
        """
        name = collection_name or self.collection
        params = VectorParams(size=self.embedding_dim, distance=Distance.COSINE)

        async def create(client: AsyncQdrantClient) -> CollectionInfo:
            if not await client.collection_exists(name):
                try:
                    await client.create_collection(name, vectors_config=params)
                except UnexpectedResponse as exc:
                    if exc.status_code != _CONFLICT:  # 409: another process won the race
                        raise
            await client.create_payload_index(
                name, field_name=PAYLOAD_ENTITY_TYPE, field_schema=PayloadSchemaType.KEYWORD
            )
            return await client.get_collection(name)

        info = await self._call("ensure_collection", create, collection=name)
        vectors = info.config.params.vectors
        if (
            not isinstance(vectors, VectorParams)
            or vectors.size != params.size
            or vectors.distance != params.distance
        ):
            msg = f"Qdrant collection {name!r} exists with an incompatible vector configuration."
            raise MemoryGraphError(
                msg,
                context={
                    "store": STORE_NAME,
                    "collection": name,
                    "expected_size": params.size,
                    "expected_distance": params.distance.value,
                },
            )

    async def upsert_entities(self, entities: Sequence[Entity]) -> int:
        """Upsert the embeddings of ``entities`` in batches; return how many were written.

        Entities without an embedding are skipped (they stay reachable through graph
        traversal only). Dimensions are checked for every entity before anything is sent.
        """
        points = [
            PointStruct(
                id=point_id(entity.id),
                vector=_check_vector(entity.embedding, self.embedding_dim, f"entity {entity.id}"),
                payload={
                    PAYLOAD_ENTITY_ID: entity.id,
                    PAYLOAD_NAME: entity.name,
                    PAYLOAD_ENTITY_TYPE: entity.entity_type.value,
                },
            )
            for entity in entities
            if entity.embedding is not None
        ]
        for start in range(0, len(points), self._batch_size):
            batch = points[start : start + self._batch_size]

            async def upsert(client: AsyncQdrantClient, batch: list[PointStruct] = batch) -> None:
                await client.upsert(
                    self.collection, points=batch, wait=True, timeout=self._request_timeout
                )

            await self._call("upsert_entities", upsert, collection=self.collection)
        return len(points)

    async def search_seeds(
        self,
        query_vector: Sequence[float],
        top_k: int,
        score_threshold: float,
        allowed_types: Sequence[EntityType] | None = None,
    ) -> list[str]:
        """Return the ids of the ``top_k`` most similar entities, best match first.

        ``allowed_types=None`` searches every type; an empty sequence matches nothing.
        Raises :class:`VectorSearchTimeoutError` when the deadline is exceeded.
        """
        vector = _check_vector(query_vector, self.embedding_dim, "query vector")
        top_k = _TOP_K.validate_python(top_k)
        score_threshold = _SCORE_THRESHOLD.validate_python(score_threshold)
        query_filter = None
        if allowed_types is not None:
            types = sorted({EntityType(t).value for t in allowed_types})
            if not types:
                return []
            query_filter = Filter(
                must=[FieldCondition(key=PAYLOAD_ENTITY_TYPE, match=MatchAny(any=types))]
            )

        async def search(client: AsyncQdrantClient) -> list[object]:
            response = await client.query_points(
                self.collection,
                query=vector,
                limit=top_k,
                score_threshold=float(score_threshold),
                query_filter=query_filter,
                with_payload=[PAYLOAD_ENTITY_ID],
                with_vectors=False,
                timeout=self._request_timeout,
            )
            return [(point.payload or {}).get(PAYLOAD_ENTITY_ID) for point in response.points]

        raw_ids = await self._call("search_seeds", search, collection=self.collection, search=True)
        try:
            ids = [_ENTITY_ID.validate_python(raw) for raw in raw_ids]
        except ValidationError as exc:
            msg = "Qdrant returned a point whose entity_id payload is missing or invalid."
            raise MemoryGraphError(
                msg, context={"store": STORE_NAME, "collection": self.collection}
            ) from exc
        return list(dict.fromkeys(ids))
