"""GraphRAG memory orchestration: ingest conversations, answer memory queries.

Ingestion: extract → embed → write the graph (one Neo4j transaction) → upsert vectors.
Neo4j and Qdrant cannot share a transaction, so the write order is deliberate: the
graph goes first, so a vector can never point at a node that does not exist. If the
vector upsert fails afterwards, the graph is complete but those entities are not
seedable yet. Every step is idempotent (deterministic ids, ``MERGE``, upserts), so
retrying the same ingest converges. Nothing is rolled back across stores: a failed
upsert must not delete graph state that earlier conversations may share.

Retrieval: embed the query → vector-search seeds → expand them through the graph →
resolve every entity on the paths in one read → render a markdown summary.
"""

import asyncio
import re
from collections.abc import Awaitable, Sequence
from dataclasses import dataclass
from typing import Final, Literal, Protocol

from agent_memory_graph.domain.errors import MemoryGraphError
from agent_memory_graph.domain.models import Entity, EntityType, KnowledgeGraph
from agent_memory_graph.domain.query import (
    MAX_SUMMARY_LENGTH,
    GraphContext,
    GraphHop,
    MemoryQuery,
)
from agent_memory_graph.services.embedder import EmbeddingService
from agent_memory_graph.services.extractor import ExtractionReport, GraphExtractor

__all__ = [
    "MAX_SUMMARY_DESCRIPTION_CHARS",
    "GraphMemoryService",
    "GraphStore",
    "HealthReport",
    "VectorStore",
    "render_summary",
]

MAX_SUMMARY_DESCRIPTION_CHARS: Final = 280
_MARKDOWN_SPECIAL_RE: Final = re.compile(r"([\\`*_\[\]<>#|~])")

type CheckStatus = Literal["ok", "unavailable"]


# ---------------------------------------------------------------------------
# Ports (structurally satisfied by Neo4jGraphStore / QdrantVectorStore)
# ---------------------------------------------------------------------------
class GraphStore(Protocol):
    """The graph-side operations the memory service needs."""

    async def connect(self) -> None:
        """Open the connection pool."""
        ...

    async def close(self) -> None:
        """Release the connection pool."""
        ...

    async def ping(self) -> None:
        """Raise a :class:`MemoryGraphError` unless the store answers."""
        ...

    async def ensure_schema(self) -> None:
        """Create constraints and indexes (idempotent)."""
        ...

    async def ingest_graph(self, graph: KnowledgeGraph) -> None:
        """Merge ``graph`` atomically."""
        ...

    async def traverse_neighborhood(
        self, seed_ids: Sequence[str], depth: int = 2
    ) -> tuple[GraphHop, ...]:
        """Return the relations reachable from ``seed_ids`` within ``depth`` hops."""
        ...

    async def fetch_entities(self, entity_ids: Sequence[str]) -> tuple[Entity, ...]:
        """Return the stored entities among ``entity_ids``, in request order."""
        ...


class VectorStore(Protocol):
    """The vector-side operations the memory service needs."""

    async def connect(self) -> None:
        """Create the client."""
        ...

    async def close(self) -> None:
        """Release the client."""
        ...

    async def ping(self) -> None:
        """Raise a :class:`MemoryGraphError` unless the store answers."""
        ...

    async def ensure_collection(self, collection_name: str | None = None) -> None:
        """Create or verify the collection (idempotent)."""
        ...

    async def upsert_entities(self, entities: Sequence[Entity]) -> int:
        """Upsert entity embeddings; return how many were written."""
        ...

    async def search_scored(
        self,
        query_vector: Sequence[float],
        top_k: int,
        score_threshold: float,
        allowed_types: Sequence[EntityType] | None = None,
    ) -> list[tuple[str, float]]:
        """Return ``(entity_id, similarity)`` pairs, best match first."""
        ...


@dataclass(frozen=True, slots=True)
class HealthReport:
    """Connectivity of each backing store; ``ok`` only when every check passes."""

    graph_store: CheckStatus
    vector_store: CheckStatus

    @property
    def ok(self) -> bool:
        """Whether every store answered."""
        return self.graph_store == "ok" and self.vector_store == "ok"


# ---------------------------------------------------------------------------
# Summary rendering
# ---------------------------------------------------------------------------
def _escape(text: str) -> str:
    return _MARKDOWN_SPECIAL_RE.sub(r"\\\1", text)


def _one_line(text: str, limit: int) -> str:
    flat = " ".join(text.split())
    return flat if len(flat) <= limit else f"{flat[: limit - 3].rstrip()}..."


def render_summary(
    seeds: Sequence[Entity],
    hops: Sequence[GraphHop],
    names: dict[str, str],
    *,
    max_length: int = MAX_SUMMARY_LENGTH,
) -> str:
    """Render seeds and traversal hops as markdown bounded to ``max_length`` characters.

    Names are markdown-escaped so stored text cannot restructure the summary. When the
    budget runs out, the remaining relationships are counted instead of listed. Output
    is plain ASCII punctuation, so the NFKC normalisation ``GraphContext`` applies cannot
    lengthen it past the budget (it would expand "…" to "...").
    """
    lines = ["## Relevant memories", ""]
    for entity in seeds:
        line = f"- **{_escape(entity.name)}** ({entity.entity_type.value.lower()})"
        if entity.description:
            line += f": {_escape(_one_line(entity.description, MAX_SUMMARY_DESCRIPTION_CHARS))}"
        lines.append(line)
    if hops:
        lines += ["", "## Relationships", ""]

    def label(entity_id: str) -> str:
        return f"**{_escape(names.get(entity_id, entity_id))}**"

    rendered = "\n".join(lines)
    for index, hop in enumerate(hops):
        line = (
            f"\n- {label(hop.source_id)} {hop.predicate} {label(hop.target_id)}"
            f" (weight {hop.weight:.2f}, depth {hop.depth})"
        )
        remaining = len(hops) - index
        footer = f"\n- ... and {remaining} more relationships"
        if len(rendered) + len(line) + len(footer) > max_length:
            rendered += footer
            break
        rendered += line
    return rendered[:max_length]


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------
class GraphMemoryService:
    """Orchestrates extraction, embedding, the graph store and the vector store."""

    def __init__(
        self,
        *,
        graph_store: GraphStore,
        vector_store: VectorStore,
        extractor: GraphExtractor,
        embedder: EmbeddingService,
    ) -> None:
        self._graph = graph_store
        self._vectors = vector_store
        self._extractor = extractor
        self._embedder = embedder

    # -- lifecycle ----------------------------------------------------------
    async def start(self, *, ensure_schema: bool = True) -> None:
        """Connect both stores and (by default) create their schema idempotently.

        If anything fails, both stores are closed again before the error propagates,
        so a failed start leaks no connection pool.
        """
        try:
            await self._graph.connect()
            await self._vectors.connect()
            if ensure_schema:
                await self._graph.ensure_schema()
                await self._vectors.ensure_collection()
        except BaseException:
            await self.close()
            raise

    async def close(self) -> None:
        """Close both stores; safe to call repeatedly."""
        try:
            await self._graph.close()
        finally:
            await self._vectors.close()

    async def health(self) -> HealthReport:
        """Ping both stores concurrently; failures are reported, never raised."""

        async def check(ping: Awaitable[None]) -> CheckStatus:
            try:
                await ping
            except MemoryGraphError:
                return "unavailable"
            return "ok"

        graph_status, vector_status = await asyncio.gather(
            check(self._graph.ping()), check(self._vectors.ping())
        )
        return HealthReport(graph_store=graph_status, vector_store=vector_status)

    # -- ingestion ----------------------------------------------------------
    async def ingest(self, text: str) -> tuple[KnowledgeGraph, ExtractionReport]:
        """Extract a graph from ``text``, store it, and index its entities.

        Embeddings are computed before anything is written, so an embedding failure
        leaves both stores untouched. The returned graph carries no embeddings; they
        live in the vector store.
        """
        graph, report = await self._extractor.extract(text)
        if not graph.entities:
            return graph, report
        embedded = await self._embedder.embed_entities(graph.entities)
        await self._graph.ingest_graph(graph)
        await self._vectors.upsert_entities(embedded)
        return graph, report

    # -- retrieval ----------------------------------------------------------
    async def query(self, query: MemoryQuery) -> GraphContext:
        """Answer ``query`` with seed entities, their neighbourhood and a summary.

        Returns :meth:`GraphContext.empty` when no stored entity clears the
        similarity threshold. ``confidence`` is the best seed's cosine similarity.
        """
        vector = await self._embedder.embed_text(query.text)
        scored = await self._vectors.search_scored(
            vector, query.top_k, query.score_threshold, query.entity_types
        )
        if not scored:
            return GraphContext.empty()
        seed_ids = [entity_id for entity_id, _ in scored]
        hops = await self._graph.traverse_neighborhood(seed_ids, depth=query.traversal_depth)
        endpoint_ids = [eid for hop in hops for eid in (hop.source_id, hop.target_id)]
        known = {e.id: e for e in await self._graph.fetch_entities([*seed_ids, *endpoint_ids])}
        seeds = tuple(known[eid] for eid in seed_ids if eid in known)
        if not seeds:
            # Vectors without graph nodes: an ingest whose graph write never landed.
            return GraphContext.empty()
        best_score = max(score for entity_id, score in scored if entity_id in known)
        names = {entity_id: entity.name for entity_id, entity in known.items()}
        return GraphContext(
            seed_entities=seeds,
            paths=hops,
            summary=render_summary(seeds, hops, names),
            confidence=min(max(best_score, 0.0), 1.0),
        )
