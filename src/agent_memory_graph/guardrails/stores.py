"""Guarded decorators for the service ports: breaker check → queue → guarded call.

The breaker is consulted first, so while a store is down callers fail fast without
taking (or waiting for) a permit. The permit is then acquired *outside* the breaker's
guard, so a queue-deadline rejection (``StoreTimeoutError(operation="queue")``) is a
local overload signal and never counts against the store's health. Lifecycle calls (``connect`` /
``close``) bypass both; ``ping`` passes through the breaker only, so health checks act
as half-open probes but never queue behind traffic.
"""

from collections.abc import Awaitable, Callable, Sequence
from typing import TYPE_CHECKING

from agent_memory_graph.domain.models import Entity, EntityType, KnowledgeGraph
from agent_memory_graph.domain.query import GraphHop
from agent_memory_graph.guardrails.limiter import CircuitBreaker, ConcurrencyLimiter

if TYPE_CHECKING:
    from pydantic import BaseModel

    from agent_memory_graph.services.extractor import ChatMessage, StructuredLLM
    from agent_memory_graph.services.retrieval import GraphStore, VectorStore

__all__ = ["GuardedGraphStore", "GuardedLLM", "GuardedVectorStore"]


async def _guarded[T](
    breaker: CircuitBreaker, limiter: ConcurrencyLimiter, call: Callable[[], Awaitable[T]]
) -> T:
    breaker.check()
    async with limiter.acquire(), breaker.guard():
        return await call()


class GuardedGraphStore:
    """:class:`GraphStore` decorator adding a circuit breaker and a bounded queue."""

    def __init__(
        self, inner: "GraphStore", *, breaker: CircuitBreaker, limiter: ConcurrencyLimiter
    ) -> None:
        self.inner = inner
        self.breaker = breaker
        self.limiter = limiter

    async def connect(self) -> None:
        """Open the inner store."""
        await self.inner.connect()

    async def close(self) -> None:
        """Close the inner store."""
        await self.inner.close()

    async def ping(self) -> None:
        """Probe the store through the breaker (no queueing)."""
        async with self.breaker.guard():
            await self.inner.ping()

    async def ensure_schema(self) -> None:
        """Create constraints through the breaker and queue."""
        await _guarded(self.breaker, self.limiter, self.inner.ensure_schema)

    async def ingest_graph(self, graph: KnowledgeGraph) -> None:
        """Merge ``graph`` through the breaker and queue."""
        await _guarded(self.breaker, self.limiter, lambda: self.inner.ingest_graph(graph))

    async def traverse_neighborhood(
        self, seed_ids: Sequence[str], depth: int = 2
    ) -> tuple[GraphHop, ...]:
        """Traverse through the breaker and queue."""
        return await _guarded(
            self.breaker,
            self.limiter,
            lambda: self.inner.traverse_neighborhood(seed_ids, depth=depth),
        )

    async def fetch_entities(self, entity_ids: Sequence[str]) -> tuple[Entity, ...]:
        """Fetch entities through the breaker and queue."""
        return await _guarded(
            self.breaker, self.limiter, lambda: self.inner.fetch_entities(entity_ids)
        )


class GuardedVectorStore:
    """:class:`VectorStore` decorator adding a circuit breaker and a bounded queue."""

    def __init__(
        self, inner: "VectorStore", *, breaker: CircuitBreaker, limiter: ConcurrencyLimiter
    ) -> None:
        self.inner = inner
        self.breaker = breaker
        self.limiter = limiter

    async def connect(self) -> None:
        """Create the inner client."""
        await self.inner.connect()

    async def close(self) -> None:
        """Close the inner client."""
        await self.inner.close()

    async def ping(self) -> None:
        """Probe the store through the breaker (no queueing)."""
        async with self.breaker.guard():
            await self.inner.ping()

    async def ensure_collection(self, collection_name: str | None = None) -> None:
        """Create or verify the collection through the breaker and queue."""
        await _guarded(
            self.breaker, self.limiter, lambda: self.inner.ensure_collection(collection_name)
        )

    async def upsert_entities(self, entities: Sequence[Entity]) -> int:
        """Upsert through the breaker and queue."""
        return await _guarded(
            self.breaker, self.limiter, lambda: self.inner.upsert_entities(entities)
        )

    async def search_scored(
        self,
        query_vector: Sequence[float],
        top_k: int,
        score_threshold: float,
        allowed_types: Sequence[EntityType] | None = None,
    ) -> list[tuple[str, float]]:
        """Search through the breaker and queue."""
        return await _guarded(
            self.breaker,
            self.limiter,
            lambda: self.inner.search_scored(query_vector, top_k, score_threshold, allowed_types),
        )


class GuardedLLM:
    """:class:`StructuredLLM` decorator bounding concurrent extraction calls."""

    def __init__(self, inner: "StructuredLLM", *, limiter: ConcurrencyLimiter) -> None:
        self.inner = inner
        self.limiter = limiter

    async def create[T: "BaseModel"](
        self,
        *,
        response_model: type[T],
        messages: "Sequence[ChatMessage]",
        max_retries: int,
    ) -> T:
        """Run the inner completion once a permit is available."""
        async with self.limiter.acquire():
            return await self.inner.create(
                response_model=response_model, messages=messages, max_retries=max_retries
            )
