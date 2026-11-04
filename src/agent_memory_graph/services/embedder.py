"""Embedding services that turn text and entities into vectors for seed search.

:class:`EmbeddingService` is the port; :class:`MockEmbeddingService` is a
deterministic, dependency-free implementation for tests and local runs. It uses
signed feature hashing over word tokens, so texts that share words get similar
vectors (unlike a hash of the whole string) and seed search behaves plausibly in
end-to-end tests. Results are identical across processes and platforms.
"""

import hashlib
import math
import re
import unicodedata
from collections.abc import Sequence
from typing import Final, Protocol, Self

from agent_memory_graph.config import Settings
from agent_memory_graph.domain._base import sanitize_line
from agent_memory_graph.domain.models import MAX_EMBEDDING_DIMENSIONS, Entity

__all__ = [
    "MAX_EMBEDDING_INPUT_CHARS",
    "EmbeddingService",
    "MockEmbeddingService",
    "entity_embedding_text",
    "with_embeddings",
]

MAX_EMBEDDING_INPUT_CHARS: Final = 8_192

_TOKEN_RE: Final = re.compile(r"\w+")


def entity_embedding_text(entity: Entity) -> str:
    """Return the canonical text an entity is embedded from.

    ``"<name> (<type>): <description>"``; the type disambiguates homonyms such as
    the tool and the concept both called "Python".
    """
    label = f"{entity.name} ({entity.entity_type.value.lower()})"
    return f"{label}: {entity.description}" if entity.description else label


def with_embeddings(
    entities: Sequence[Entity], vectors: Sequence[Sequence[float]], *, dimension: int
) -> tuple[Entity, ...]:
    """Return copies of ``entities`` carrying ``vectors``, re-validated by the domain."""
    if len(entities) != len(vectors):
        msg = f"got {len(vectors)} vectors for {len(entities)} entities"
        raise ValueError(msg)
    embedded = []
    for entity, vector in zip(entities, vectors, strict=True):
        if len(vector) != dimension:
            msg = f"vector for {entity.id} has {len(vector)} dimensions, expected {dimension}"
            raise ValueError(msg)
        embedded.append(Entity.model_validate({**entity.model_dump(), "embedding": tuple(vector)}))
    return tuple(embedded)


class EmbeddingService(Protocol):
    """Port for embedding providers; every vector has exactly ``dimension`` floats."""

    @property
    def dimension(self) -> int:
        """Length of every vector this service returns."""
        ...

    async def embed_text(self, text: str) -> tuple[float, ...]:
        """Embed one query or passage."""
        ...

    async def embed_entities(self, entities: Sequence[Entity]) -> tuple[Entity, ...]:
        """Return ``entities`` (same order) with ``embedding`` set."""
        ...


class MockEmbeddingService:
    """Deterministic, hermetic :class:`EmbeddingService` producing L2-normalised vectors."""

    def __init__(self, dimension: int) -> None:
        if isinstance(dimension, bool) or not 1 <= dimension <= MAX_EMBEDDING_DIMENSIONS:
            msg = f"dimension must be an int in 1..{MAX_EMBEDDING_DIMENSIONS}"
            raise ValueError(msg)
        self._dimension = dimension

    @classmethod
    def from_settings(cls, settings: Settings) -> Self:
        """Build a service whose vectors match ``AMG_EMBEDDING_DIM``."""
        return cls(settings.embedding_dim)

    @property
    def dimension(self) -> int:
        """Length of every vector this service returns."""
        return self._dimension

    def _bucket(self, feature: str) -> tuple[int, float]:
        digest = hashlib.sha256(feature.encode()).digest()
        index = int.from_bytes(digest[:8], "big") % self._dimension
        return index, 1.0 if digest[8] & 1 else -1.0

    def _vector(self, text: str) -> tuple[float, ...]:
        clean = sanitize_line(text, max_length=MAX_EMBEDDING_INPUT_CHARS)
        folded = unicodedata.normalize("NFKC", clean).casefold()
        values = [0.0] * self._dimension
        for token in _TOKEN_RE.findall(folded):
            index, sign = self._bucket(token)
            values[index] += sign
        norm = math.sqrt(math.fsum(v * v for v in values))
        if norm == 0.0:
            # No word tokens (punctuation only) or all features cancelled out:
            # fall back to a one-hot of the whole text so the vector stays unit-length.
            index, sign = self._bucket(f"\x00text:{folded}")
            values = [0.0] * self._dimension
            values[index] = sign
            norm = 1.0
        return tuple(v / norm for v in values)

    async def embed_text(self, text: str) -> tuple[float, ...]:
        """Embed ``text``; raises ``ValueError`` when it has no printable content."""
        return self._vector(text)

    async def embed_entities(self, entities: Sequence[Entity]) -> tuple[Entity, ...]:
        """Embed each entity from :func:`entity_embedding_text`."""
        vectors = [self._vector(entity_embedding_text(entity)) for entity in entities]
        return with_embeddings(entities, vectors, dimension=self._dimension)
