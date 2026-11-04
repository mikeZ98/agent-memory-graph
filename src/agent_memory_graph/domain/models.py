"""Knowledge-graph core: entities, relation triplets and a referentially sound graph."""

import re
import unicodedata
import uuid
from collections import Counter
from enum import StrEnum
from typing import Annotated, Final, Self

from pydantic import AfterValidator, Field, Strict, model_validator
from pydantic_core import PydanticCustomError

from agent_memory_graph.domain._base import (
    FrozenJsonMapping,
    Probability,
    StrictModel,
    sanitize_line,
    sanitize_multiline,
)

__all__ = [
    "ENTITY_NAMESPACE",
    "MAX_EMBEDDING_DIMENSIONS",
    "MAX_ENTITY_ID_LENGTH",
    "MAX_GRAPH_ENTITIES",
    "MAX_GRAPH_RELATIONS",
    "Embedding",
    "Entity",
    "EntityId",
    "EntityName",
    "EntityType",
    "KnowledgeGraph",
    "Predicate",
    "Relation",
    "make_entity_id",
]

MAX_ENTITY_ID_LENGTH: Final = 128
MAX_ENTITY_NAME_LENGTH: Final = 256
MAX_DESCRIPTION_LENGTH: Final = 4096
MAX_PREDICATE_LENGTH: Final = 64
MAX_EMBEDDING_DIMENSIONS: Final = 8192
MAX_GRAPH_ENTITIES: Final = 10_000
MAX_GRAPH_RELATIONS: Final = 50_000

ENTITY_NAMESPACE: Final = uuid.uuid5(uuid.NAMESPACE_URL, "urn:agent-memory-graph:entity")
"""UUIDv5 namespace for ids of entities whose name has no ASCII slug (e.g. CJK names)."""

_SLUG_RE: Final = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")
_UUID_RE: Final = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", re.I)
_SLUG_SEPARATOR_RE: Final = re.compile(r"[^a-z0-9]+")
_PREDICATE_RE: Final = re.compile(r"[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)*")
_CAMEL_BOUNDARY_RE: Final = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
_PREDICATE_SEPARATOR_RE: Final = re.compile(r"[^A-Za-z0-9]+")


class EntityType(StrEnum):
    """Closed set of node labels; anything else is classified as ``MISC``."""

    PERSON = "PERSON"
    ORGANIZATION = "ORGANIZATION"
    TOOL = "TOOL"
    CONCEPT = "CONCEPT"
    LOCATION = "LOCATION"
    EVENT = "EVENT"
    MISC = "MISC"


# ---------------------------------------------------------------------------
# Field validators
# ---------------------------------------------------------------------------
def _validate_entity_id(value: str) -> str:
    if len(value) > MAX_ENTITY_ID_LENGTH:
        msg = f"entity id exceeds {MAX_ENTITY_ID_LENGTH} characters"
        raise ValueError(msg)
    if _UUID_RE.fullmatch(value):
        return value.lower()
    if _SLUG_RE.fullmatch(value):
        return value
    msg = "entity id must be a lowercase slug (a-z, 0-9, single hyphens) or a canonical UUID"
    raise ValueError(msg)


def _sanitize_name(value: str) -> str:
    return sanitize_line(value, max_length=MAX_ENTITY_NAME_LENGTH)


def _sanitize_description(value: str) -> str:
    return sanitize_multiline(value, max_length=MAX_DESCRIPTION_LENGTH)


def _normalize_predicate(value: str) -> str:
    # Predicates become Neo4j relationship types, which Cypher cannot parameterise:
    # this allowlist is what keeps them out of query-injection territory.
    normalized = unicodedata.normalize("NFKC", value).strip()
    if not normalized.isascii():
        msg = "predicate must be ASCII"
        raise ValueError(msg)
    snake = _CAMEL_BOUNDARY_RE.sub("_", normalized)
    candidate = _PREDICATE_SEPARATOR_RE.sub("_", snake).strip("_").upper()
    if len(candidate) > MAX_PREDICATE_LENGTH or not _PREDICATE_RE.fullmatch(candidate):
        msg = (
            "predicate must normalise to SCREAMING_SNAKE_CASE starting with a letter "
            f"(at most {MAX_PREDICATE_LENGTH} characters), e.g. DEPENDS_ON"
        )
        raise ValueError(msg)
    return candidate


EntityId = Annotated[str, Strict(), AfterValidator(_validate_entity_id)]
"""Lowercase slug (``tool-neo4j``) or canonical UUID, normalised to lowercase."""

EntityName = Annotated[str, Strict(), AfterValidator(_sanitize_name)]
"""Single-line display name: NFKC, no invisible characters, collapsed whitespace."""

Description = Annotated[str, Strict(), AfterValidator(_sanitize_description)]

Predicate = Annotated[str, Strict(), AfterValidator(_normalize_predicate)]
"""Relationship type in SCREAMING_SNAKE_CASE; ``depends on``/``dependsOn`` → ``DEPENDS_ON``."""

Embedding = Annotated[
    tuple[Annotated[float, Strict(), Field(allow_inf_nan=False)], ...],
    Field(min_length=1, max_length=MAX_EMBEDDING_DIMENSIONS),
]


def make_entity_id(name: str, entity_type: EntityType) -> str:
    """Derive the deterministic id of an entity from its type and (sanitised) name.

    The id is ``<type>-<ascii-slug>`` when the name has an ASCII transliteration that
    fits; otherwise a UUIDv5 over the type and case-folded name, so the same entity
    always maps to the same node regardless of which extraction produced it.
    """
    clean = _sanitize_name(name)
    ascii_name = unicodedata.normalize("NFKD", clean).encode("ascii", "ignore").decode()
    slug = _SLUG_SEPARATOR_RE.sub("-", ascii_name.lower()).strip("-")
    candidate = f"{entity_type.value.lower()}-{slug}"
    if slug and len(candidate) <= MAX_ENTITY_ID_LENGTH:
        return candidate
    return str(uuid.uuid5(ENTITY_NAMESPACE, f"{entity_type.value}:{clean.casefold()}"))


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
class Entity(StrictModel):
    """A node of the memory graph."""

    id: EntityId
    name: EntityName
    entity_type: EntityType
    description: Description = ""
    attributes: FrozenJsonMapping
    embedding: Embedding | None = None


class Relation(StrictModel):
    """A directed, weighted triplet ``(source_id)-[predicate]->(target_id)``."""

    source_id: EntityId
    predicate: Predicate
    target_id: EntityId
    weight: Probability = 1.0
    properties: FrozenJsonMapping

    @property
    def key(self) -> tuple[str, str, str]:
        """Identity of the triplet; weight and properties are payload."""
        return (self.source_id, self.predicate, self.target_id)


class KnowledgeGraph(StrictModel):
    """A self-contained subgraph: every relation endpoint is one of its entities.

    Entity ids and relation triplets are unique, so the graph can be merged into a
    store idempotently.
    """

    entities: Annotated[tuple[Entity, ...], Field(max_length=MAX_GRAPH_ENTITIES)] = ()
    relations: Annotated[tuple[Relation, ...], Field(max_length=MAX_GRAPH_RELATIONS)] = ()

    @model_validator(mode="after")
    def _check_integrity(self) -> Self:
        id_counts = Counter(entity.id for entity in self.entities)
        duplicate_ids = sorted(eid for eid, count in id_counts.items() if count > 1)
        if duplicate_ids:
            raise PydanticCustomError(
                "duplicate_entity",
                "entity ids must be unique; duplicated: {duplicate_ids}",
                {"duplicate_ids": duplicate_ids},
            )

        dangling_ids = sorted(
            {
                endpoint
                for relation in self.relations
                for endpoint in (relation.source_id, relation.target_id)
                if endpoint not in id_counts
            }
        )
        if dangling_ids:
            raise PydanticCustomError(
                "dangling_relation",
                "relations reference entities missing from the graph: {dangling_ids}",
                {"dangling_ids": dangling_ids},
            )

        key_counts = Counter(relation.key for relation in self.relations)
        duplicate_triplets = sorted("|".join(k) for k, count in key_counts.items() if count > 1)
        if duplicate_triplets:
            raise PydanticCustomError(
                "duplicate_relation",
                "relation triplets must be unique; duplicated: {duplicate_triplets}",
                {"duplicate_triplets": duplicate_triplets},
            )
        return self
