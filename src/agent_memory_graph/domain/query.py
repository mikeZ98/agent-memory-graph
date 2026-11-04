"""GraphRAG retrieval contracts: the query an agent asks and the context it gets back."""

from collections import Counter
from typing import Annotated, Final, Self

from pydantic import AfterValidator, Field, Strict, field_validator, model_validator
from pydantic_core import PydanticCustomError

from agent_memory_graph.domain._base import (
    Probability,
    StrictModel,
    sanitize_line,
    sanitize_multiline,
)
from agent_memory_graph.domain.models import Entity, EntityId, EntityType, Predicate, Relation

__all__ = [
    "DEFAULT_TOP_K",
    "DEFAULT_TRAVERSAL_DEPTH",
    "MAX_TOP_K",
    "MAX_TRAVERSAL_DEPTH",
    "GraphContext",
    "GraphHop",
    "MemoryQuery",
]

MAX_QUERY_LENGTH: Final = 4096
MAX_SUMMARY_LENGTH: Final = 32_768
MAX_CONTEXT_HOPS: Final = 1_000
DEFAULT_TOP_K: Final = 5
MAX_TOP_K: Final = 20
DEFAULT_TRAVERSAL_DEPTH: Final = 2
MAX_TRAVERSAL_DEPTH: Final = 3


def _sanitize_query(value: str) -> str:
    return sanitize_line(value, max_length=MAX_QUERY_LENGTH)


def _sanitize_summary(value: str) -> str:
    return sanitize_multiline(value, max_length=MAX_SUMMARY_LENGTH)


TraversalDepth = Annotated[int, Strict(), Field(ge=1, le=MAX_TRAVERSAL_DEPTH)]


class MemoryQuery(StrictModel):
    """A retrieval request: vector-search seeds, then expand them through the graph."""

    text: Annotated[str, Strict(), AfterValidator(_sanitize_query)]
    top_k: Annotated[int, Strict(), Field(ge=1, le=MAX_TOP_K)] = DEFAULT_TOP_K
    score_threshold: Probability = 0.0
    traversal_depth: TraversalDepth = DEFAULT_TRAVERSAL_DEPTH
    entity_types: Annotated[tuple[EntityType, ...], Field(min_length=1)] | None = None
    """Restrict seeds to these types; ``None`` means every type."""

    @field_validator("entity_types", mode="after")
    @classmethod
    def _canonical_entity_types(
        cls, value: tuple[EntityType, ...] | None
    ) -> tuple[EntityType, ...] | None:
        # Deduplicate and order by declaration so equal filters serialise identically.
        if value is None:
            return None
        return tuple(member for member in EntityType if member in value)


class GraphHop(StrictModel):
    """One traversal step ``(source_id)-[predicate]->(target_id)`` at ``depth`` from a seed."""

    source_id: EntityId
    predicate: Predicate
    target_id: EntityId
    weight: Probability = 1.0
    depth: TraversalDepth = 1

    @classmethod
    def from_relation(cls, relation: Relation, *, depth: int) -> Self:
        """Record that ``relation`` was traversed ``depth`` hops away from a seed."""
        return cls(
            source_id=relation.source_id,
            predicate=relation.predicate,
            target_id=relation.target_id,
            weight=relation.weight,
            depth=depth,
        )

    def __str__(self) -> str:
        return f"({self.source_id})-[{self.predicate}]->({self.target_id})"


class GraphContext(StrictModel):
    """Everything retrieved for one :class:`MemoryQuery`, ready to prime an LLM prompt."""

    seed_entities: tuple[Entity, ...] = ()
    paths: Annotated[tuple[GraphHop, ...], Field(max_length=MAX_CONTEXT_HOPS)] = ()
    summary: Annotated[str, Strict(), AfterValidator(_sanitize_summary)] = ""
    """Markdown synthesis of the contextual relationships."""
    confidence: Probability = 0.0

    @model_validator(mode="after")
    def _check_consistency(self) -> Self:
        duplicated = sorted(
            eid for eid, n in Counter(e.id for e in self.seed_entities).items() if n > 1
        )
        if duplicated:
            raise PydanticCustomError(
                "duplicate_seed",
                "seed entities must be unique; duplicated: {duplicate_ids}",
                {"duplicate_ids": duplicated},
            )
        if not self.seed_entities and (self.paths or self.confidence > 0.0):
            raise PydanticCustomError(
                "seedless_context",
                "a context without seed entities cannot carry paths or confidence",
            )
        return self

    @classmethod
    def empty(cls) -> Self:
        """Return the context reported when nothing relevant was found."""
        return cls()
