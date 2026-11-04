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
import logging
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
from agent_memory_graph.guardrails.graph_guard import InputGuard, TraversalGuard
from agent_memory_graph.guardrails.telemetry import Telemetry
from agent_memory_graph.services.embedder import EmbeddingService
from agent_memory_graph.services.extractor import ExtractionReport, GraphExtractor

__all__ = [
    "MAX_SUMMARY_DESCRIPTION_CHARS",
    "GraphMemoryService",
    "GraphStore",
    "HealthReport",
    "RetrievalPolicy",
    "VectorStore",
    "fallback_thresholds",
    "render_summary",
]

logger = logging.getLogger(__name__)

MAX_SUMMARY_DESCRIPTION_CHARS: Final = 280
_MARKDOWN_SPECIAL_RE: Final = re.compile(r"([\\`*_\[\]<>#|~])")


@dataclass(frozen=True, slots=True)
class RetrievalPolicy:
    """Adaptive fallback when the requested threshold finds no seeds.

    Each retry multiplies the threshold by ``fallback_factor`` (never going below
    ``fallback_floor``), for at most ``max_fallback_steps`` extra vector searches.
    ``max_fallback_steps=0`` disables the fallback.
    """

    fallback_factor: float = 0.8
    fallback_floor: float = 0.1
    max_fallback_steps: int = 3

    def __post_init__(self) -> None:
        if not 0.0 < self.fallback_factor < 1.0:
            msg = "fallback_factor must be in (0, 1)"
            raise ValueError(msg)
        if not 0.0 <= self.fallback_floor <= 1.0:
            msg = "fallback_floor must be in [0, 1]"
            raise ValueError(msg)
        if self.max_fallback_steps < 0:
            msg = "max_fallback_steps must be >= 0"
            raise ValueError(msg)


def fallback_thresholds(threshold: float, policy: RetrievalPolicy) -> tuple[float, ...]:
    """Return the thresholds to try in order: the requested one, then relaxed ones.

    A threshold already at or below the floor is never relaxed (the caller asked for
    exactly that), and the sequence stops as soon as it reaches the floor.
    """
    thresholds = [threshold]
    current = threshold
    for _ in range(policy.max_fallback_steps):
        if current <= policy.fallback_floor:
            break
        current = max(policy.fallback_floor, round(current * policy.fallback_factor, 6))
        thresholds.append(current)
    return tuple(thresholds)


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
    """Orchestrates extraction, embedding, the graph store and the vector store.

    Guardrails are on by default: ingested text is screened by ``input_guard``,
    traversal results are bounded by ``traversal_guard`` before synthesis, empty
    retrievals are retried per ``policy``, and everything is recorded in ``telemetry``.
    """

    def __init__(
        self,
        *,
        graph_store: GraphStore,
        vector_store: VectorStore,
        extractor: GraphExtractor,
        embedder: EmbeddingService,
        telemetry: Telemetry | None = None,
        input_guard: InputGuard | None = None,
        traversal_guard: TraversalGuard | None = None,
        policy: RetrievalPolicy | None = None,
    ) -> None:
        self._graph = graph_store
        self._vectors = vector_store
        self._extractor = extractor
        self._embedder = embedder
        self.telemetry = telemetry or Telemetry()
        self._input_guard = input_guard or InputGuard()
        self._traversal_guard = traversal_guard or TraversalGuard()
        self._policy = policy or RetrievalPolicy()

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
        """Screen, extract, embed and store ``text``; return the graph and report.

        Embeddings are computed before anything is written, so an embedding failure
        leaves both stores untouched. The returned graph carries no embeddings; they
        live in the vector store.
        """
        with self.telemetry.track("ingest"):
            scan = self._input_guard.check(text)
            if scan.suspicious:
                self.telemetry.increment("suspicious_inputs")
                logger.warning(
                    "ingest text contains prompt-injection markers: %s",
                    ", ".join(scan.kinds),
                )
            graph, report = await self._extractor.extract(scan.text)
            self.telemetry.increment("candidates_pruned", report.pruned)
            if not graph.entities:
                return graph, report
            embedded = await self._embedder.embed_entities(graph.entities)
            await self._graph.ingest_graph(graph)
            await self._vectors.upsert_entities(embedded)
            self.telemetry.increment("entities_indexed", len(graph.entities))
            self.telemetry.increment("relations_indexed", len(graph.relations))
            return graph, report

    # -- retrieval ----------------------------------------------------------
    async def _search_seeds(
        self, vector: Sequence[float], query: MemoryQuery
    ) -> list[tuple[str, float]]:
        thresholds = fallback_thresholds(query.score_threshold, self._policy)
        for attempt, threshold in enumerate(thresholds):
            if attempt:
                self.telemetry.increment("query_fallbacks")
            scored = await self._vectors.search_scored(
                vector, query.top_k, threshold, query.entity_types
            )
            if scored:
                if attempt:
                    self.telemetry.increment("fallback_hits")
                    logger.info(
                        "fallback threshold %.3f found %d seed(s) (requested %.3f)",
                        threshold, len(scored), query.score_threshold,
                    )  # fmt: skip
                return scored
        return []

    async def query(self, query: MemoryQuery) -> GraphContext:
        """Answer ``query`` with seed entities, their neighbourhood and a summary.

        When the requested ``score_threshold`` finds nothing, the threshold is relaxed
        per the :class:`RetrievalPolicy` before giving up; ``confidence`` (the best
        seed's cosine similarity) then reflects the weaker match. Returns
        :meth:`GraphContext.empty` when even the fallback finds nothing.
        """
        with self.telemetry.track("query"):
            context = await self._query(query)
            if not context.seed_entities:
                self.telemetry.increment("empty_contexts")
            return context

    async def _query(self, query: MemoryQuery) -> GraphContext:
        vector = await self._embedder.embed_text(query.text)
        scored = await self._search_seeds(vector, query)
        if not scored:
            return GraphContext.empty()
        seed_ids = [entity_id for entity_id, _ in scored]
        raw_hops = await self._graph.traverse_neighborhood(seed_ids, depth=query.traversal_depth)
        hops, containment = self._traversal_guard.contain(seed_ids, raw_hops)
        if containment.pruned:
            self.telemetry.increment("traversal_hops_pruned", containment.pruned)
            logger.info(
                "traversal containment pruned %d hop(s): %s", containment.pruned, containment
            )
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
