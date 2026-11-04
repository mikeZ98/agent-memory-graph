"""In-memory stand-ins for the graph and vector stores (no sockets, real semantics).

``InMemoryGraphStore`` performs a genuine outward breadth-first traversal with the
adapter's contract (each relation once, at its shortest depth, ordered by depth then
triplet); ``InMemoryVectorStore`` ranks by cosine similarity with threshold and type
filtering. Every operation can be made to fail (``errors``) or hang (``hangs``).
"""

import asyncio
import math
from collections.abc import Sequence
from typing import Any

from pydantic import BaseModel

from agent_memory_graph.domain.models import Entity, EntityType, KnowledgeGraph, Relation
from agent_memory_graph.domain.query import GraphHop
from agent_memory_graph.services.extractor import ChatMessage


class _Controlled:
    def __init__(self) -> None:
        self.errors: dict[str, BaseException] = {}
        self.hangs: set[str] = set()
        self.calls: list[tuple[str, Any]] = []
        self.connected = False
        self.closed = 0

    async def _enter(self, operation: str, argument: Any = None) -> None:
        self.calls.append((operation, argument))
        if operation in self.hangs:
            await asyncio.Event().wait()
        if operation in self.errors:
            raise self.errors[operation]

    def called(self, operation: str) -> list[Any]:
        return [argument for name, argument in self.calls if name == operation]

    async def connect(self) -> None:
        await self._enter("connect")
        self.connected = True

    async def close(self) -> None:
        await self._enter("close")
        self.connected = False
        self.closed += 1

    async def ping(self) -> None:
        await self._enter("ping")


class InMemoryGraphStore(_Controlled):
    def __init__(self) -> None:
        super().__init__()
        self.entities: dict[str, Entity] = {}
        self.relations: dict[tuple[str, str, str], Relation] = {}

    async def ensure_schema(self) -> None:
        await self._enter("ensure_schema")

    async def ingest_graph(self, graph: KnowledgeGraph) -> None:
        await self._enter("ingest_graph", graph)
        for entity in graph.entities:
            self.entities[entity.id] = entity
        for relation in graph.relations:
            self.relations[relation.key] = relation

    async def traverse_neighborhood(
        self, seed_ids: Sequence[str], depth: int = 2
    ) -> tuple[GraphHop, ...]:
        await self._enter("traverse_neighborhood", (tuple(seed_ids), depth))
        best: dict[tuple[str, str, str], int] = {}
        frontier = set(seed_ids)
        for level in range(1, depth + 1):
            reached: set[str] = set()
            for key in self.relations:
                if key[0] in frontier and key not in best:
                    best[key] = level
                if key[0] in frontier:
                    reached.add(key[2])
            frontier = reached
        ordered = sorted(best.items(), key=lambda item: (item[1], *item[0]))
        return tuple(
            GraphHop.from_relation(self.relations[key], depth=level) for key, level in ordered
        )

    async def fetch_entities(self, entity_ids: Sequence[str]) -> tuple[Entity, ...]:
        await self._enter("fetch_entities", tuple(entity_ids))
        unique = dict.fromkeys(entity_ids)
        return tuple(self.entities[eid] for eid in unique if eid in self.entities)


class InMemoryVectorStore(_Controlled):
    def __init__(self) -> None:
        super().__init__()
        self.points: dict[str, tuple[tuple[float, ...], EntityType]] = {}

    async def ensure_collection(self, collection_name: str | None = None) -> None:
        await self._enter("ensure_collection", collection_name)

    async def upsert_entities(self, entities: Sequence[Entity]) -> int:
        await self._enter("upsert_entities", tuple(entities))
        written = 0
        for entity in entities:
            if entity.embedding is not None:
                self.points[entity.id] = (entity.embedding, entity.entity_type)
                written += 1
        return written

    async def search_scored(
        self,
        query_vector: Sequence[float],
        top_k: int,
        score_threshold: float,
        allowed_types: Sequence[EntityType] | None = None,
    ) -> list[tuple[str, float]]:
        await self._enter("search_scored", (tuple(query_vector), top_k, score_threshold))
        scored = [
            (eid, math.fsum(a * b for a, b in zip(vector, query_vector, strict=True)))
            for eid, (vector, entity_type) in self.points.items()
            if allowed_types is None or entity_type in allowed_types
        ]
        hits = [(eid, score) for eid, score in scored if score >= score_threshold]
        hits.sort(key=lambda item: (-item[1], item[0]))
        return hits[:top_k]


class ScriptedLLM:
    """:class:`StructuredLLM` returning scripted payloads (or raising scripted errors)."""

    def __init__(self, *responses: dict[str, Any] | BaseException, hang: bool = False) -> None:
        self._responses = list(responses)
        self.hang = hang
        self.calls: list[dict[str, Any]] = []

    async def create[T: BaseModel](
        self, *, response_model: type[T], messages: Sequence[ChatMessage], max_retries: int
    ) -> T:
        self.calls.append(
            {
                "response_model": response_model,
                "messages": list(messages),
                "max_retries": max_retries,
            }
        )
        if self.hang:
            await asyncio.Event().wait()
        response = self._responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response_model.model_validate(response)
