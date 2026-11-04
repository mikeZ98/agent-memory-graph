"""GraphMemoryService: lifecycle, ingestion order, retrieval and summary rendering.

Runs on in-memory stores, the deterministic mock embedder and a scripted LLM, so the
whole ingest → query loop executes for real without a single socket.
"""

from collections.abc import Sequence
from typing import Any

import pytest
from pydantic import SecretStr

from agent_memory_graph.adapters import Neo4jGraphStore, QdrantVectorStore
from agent_memory_graph.domain import (
    Entity,
    EntityType,
    GraphContext,
    GraphHop,
    KnowledgeGraph,
    MemoryQuery,
    StoreTimeoutError,
    StoreUnavailableError,
    VectorSearchTimeoutError,
    make_entity_id,
)
from agent_memory_graph.services import GraphExtractor, MockEmbeddingService
from agent_memory_graph.services.retrieval import (
    MAX_SUMMARY_DESCRIPTION_CHARS,
    GraphMemoryService,
    GraphStore,
    HealthReport,
    VectorStore,
    render_summary,
)
from tests.unit.doubles import InMemoryGraphStore, InMemoryVectorStore, ScriptedLLM

DIM = 64
TEXT = "User: our memory service uses Neo4j, and Neo4j runs on the JVM."


def ent(name: str, entity_type: str = "TOOL", description: str = "") -> dict[str, Any]:
    return {"name": name, "entity_type": entity_type, "description": description}


def rel(source: str, predicate: str, target: str, confidence: float = 1.0) -> dict[str, Any]:
    return {"source": source, "predicate": predicate, "target": target, "confidence": confidence}


PAYLOAD: dict[str, Any] = {
    "entities": [
        ent("memory service", "CONCEPT", "The agents' long-term memory."),
        ent("Neo4j", description="Graph database."),
        ent("JVM", description="Java virtual machine."),
        ent("Ada Lovelace", "PERSON", "Engineer."),
    ],
    "relations": [
        rel("memory service", "uses", "Neo4j", 0.9),
        rel("Neo4j", "runs on", "JVM", 0.8),
        rel("Ada Lovelace", "BUILDS", "memory service"),
    ],
}

MEMORY = make_entity_id("memory service", EntityType.CONCEPT)
NEO4J = make_entity_id("Neo4j", EntityType.TOOL)
JVM = make_entity_id("JVM", EntityType.TOOL)
ADA = make_entity_id("Ada Lovelace", EntityType.PERSON)


class Setup:
    def __init__(self, *responses: dict[str, Any] | BaseException) -> None:
        self.graph = InMemoryGraphStore()
        self.vectors = InMemoryVectorStore()
        self.llm = ScriptedLLM(*(responses or (PAYLOAD,)))
        self.embedder = MockEmbeddingService(DIM)
        self.service = GraphMemoryService(
            graph_store=self.graph,
            vector_store=self.vectors,
            extractor=GraphExtractor(self.llm),
            embedder=self.embedder,
        )


@pytest.fixture
def setup() -> Setup:
    return Setup()


# ---------------------------------------------------------------------------
# Ports and lifecycle
# ---------------------------------------------------------------------------
def test_real_adapters_and_doubles_satisfy_the_ports() -> None:
    graph_stores: list[GraphStore] = [
        InMemoryGraphStore(),
        Neo4jGraphStore("neo4j://localhost:7687", user="u", password=SecretStr("p")),
    ]
    vector_stores: list[VectorStore] = [
        InMemoryVectorStore(),
        QdrantVectorStore("http://localhost:6333", collection="c", embedding_dim=DIM),
    ]
    assert len(graph_stores) == len(vector_stores) == 2


class TestLifecycle:
    async def test_start_connects_and_prepares_both_stores(self, setup: Setup) -> None:
        await setup.service.start()
        assert setup.graph.connected
        assert setup.vectors.connected
        assert [op for op, _ in setup.graph.calls] == ["connect", "ensure_schema"]
        assert [op for op, _ in setup.vectors.calls] == ["connect", "ensure_collection"]

    async def test_schema_setup_can_be_skipped(self, setup: Setup) -> None:
        await setup.service.start(ensure_schema=False)
        assert setup.graph.called("ensure_schema") == []
        assert setup.vectors.called("ensure_collection") == []

    @pytest.mark.parametrize(
        ("store", "operation"),
        [
            ("graph", "connect"),
            ("vectors", "connect"),
            ("graph", "ensure_schema"),
            ("vectors", "ensure_collection"),
        ],
    )
    async def test_failed_start_closes_everything(
        self, setup: Setup, store: str, operation: str
    ) -> None:
        error = StoreUnavailableError("neo4j")
        getattr(setup, store).errors[operation] = error
        with pytest.raises(StoreUnavailableError) as exc:
            await setup.service.start()
        assert exc.value is error
        assert (setup.graph.closed, setup.vectors.closed) == (1, 1)

    async def test_close_releases_the_vector_store_even_if_the_graph_close_fails(
        self, setup: Setup
    ) -> None:
        setup.graph.errors["close"] = RuntimeError("boom")
        with pytest.raises(RuntimeError):
            await setup.service.close()
        assert setup.vectors.closed == 1

    @pytest.mark.parametrize(
        ("down", "report"),
        [
            ((), HealthReport("ok", "ok")),
            (("graph",), HealthReport("unavailable", "ok")),
            (("vectors",), HealthReport("ok", "unavailable")),
            (("graph", "vectors"), HealthReport("unavailable", "unavailable")),
        ],
    )
    async def test_health_reports_each_store(
        self, setup: Setup, down: tuple[str, ...], report: HealthReport
    ) -> None:
        for store in down:
            getattr(setup, store).errors["ping"] = StoreTimeoutError("x", "ping", 1.0)
        result = await setup.service.health()
        assert result == report
        assert result.ok is (not down)

    async def test_health_does_not_mask_programming_errors(self, setup: Setup) -> None:
        setup.graph.errors["ping"] = KeyError("bug")
        with pytest.raises(KeyError):
            await setup.service.health()


# ---------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------
class TestIngest:
    async def test_extract_embed_and_write_both_stores(self, setup: Setup) -> None:
        graph, report = await setup.service.ingest(TEXT)

        assert {e.id for e in graph.entities} == {MEMORY, NEO4J, JVM, ADA}
        assert report.relations == 3
        assert all(e.embedding is None for e in graph.entities)  # vectors live in Qdrant

        assert setup.graph.called("ingest_graph") == [graph]
        assert set(setup.graph.entities) == {MEMORY, NEO4J, JVM, ADA}
        (upserted,) = setup.vectors.called("upsert_entities")
        assert [e.id for e in upserted] == [e.id for e in graph.entities]
        assert all(e.embedding is not None and len(e.embedding) == DIM for e in upserted)
        assert [op for op, _ in setup.graph.calls + setup.vectors.calls] == [
            "ingest_graph",
            "upsert_entities",
        ]

    async def test_graph_is_written_before_vectors(self, setup: Setup) -> None:
        order: list[str] = []
        original_graph, original_vectors = setup.graph.ingest_graph, setup.vectors.upsert_entities

        async def graph_write(graph: KnowledgeGraph) -> None:
            order.append("graph")
            await original_graph(graph)

        async def vector_write(entities: Sequence[Entity]) -> int:
            order.append("vectors")
            return await original_vectors(entities)

        setup.graph.ingest_graph = graph_write  # type: ignore[method-assign]
        setup.vectors.upsert_entities = vector_write  # type: ignore[method-assign]
        await setup.service.ingest(TEXT)
        assert order == ["graph", "vectors"]

    async def test_nothing_extracted_means_nothing_written(self) -> None:
        setup = Setup({"entities": [], "relations": []})
        graph, _ = await setup.service.ingest("User: hello")
        assert graph == KnowledgeGraph()
        assert setup.graph.calls == setup.vectors.calls == []

    async def test_embedding_failure_writes_nothing(self, setup: Setup) -> None:
        async def broken(entities: Sequence[Entity]) -> tuple[Entity, ...]:
            raise ValueError("embedding provider down")

        setup.embedder.embed_entities = broken  # type: ignore[method-assign]
        with pytest.raises(ValueError, match="embedding provider down"):
            await setup.service.ingest(TEXT)
        assert setup.graph.calls == setup.vectors.calls == []

    async def test_graph_failure_skips_the_vector_upsert(self, setup: Setup) -> None:
        setup.graph.errors["ingest_graph"] = StoreUnavailableError("neo4j")
        with pytest.raises(StoreUnavailableError):
            await setup.service.ingest(TEXT)
        assert setup.vectors.calls == []

    async def test_vector_failure_leaves_the_graph_and_a_retry_converges(self) -> None:
        setup = Setup(PAYLOAD, PAYLOAD)
        setup.vectors.errors["upsert_entities"] = StoreTimeoutError("qdrant", "upsert", 1.0)
        with pytest.raises(StoreTimeoutError):
            await setup.service.ingest(TEXT)
        assert set(setup.graph.entities) == {MEMORY, NEO4J, JVM, ADA}
        assert setup.vectors.points == {}

        del setup.vectors.errors["upsert_entities"]
        await setup.service.ingest(TEXT)
        assert set(setup.vectors.points) == {MEMORY, NEO4J, JVM, ADA}
        assert len(setup.graph.relations) == 3  # MERGE semantics: no duplicates


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------
class TestQuery:
    async def test_end_to_end_ingest_then_query(self, setup: Setup) -> None:
        await setup.service.ingest(TEXT)
        context = await setup.service.query(
            MemoryQuery(text="What does the memory service use?", top_k=1, traversal_depth=2)
        )

        assert [e.id for e in context.seed_entities] == [MEMORY]
        assert [str(hop) for hop in context.paths] == [
            f"({MEMORY})-[USES]->({NEO4J})",
            f"({NEO4J})-[RUNS_ON]->({JVM})",
        ]
        assert [hop.depth for hop in context.paths] == [1, 2]
        assert 0.0 < context.confidence <= 1.0
        assert context.summary == (
            "## Relevant memories\n\n"
            "- **memory service** (concept): The agents' long-term memory.\n\n"
            "## Relationships\n\n"
            "- **memory service** USES **Neo4j** (weight 0.90, depth 1)\n"
            "- **Neo4j** RUNS_ON **JVM** (weight 0.80, depth 2)"
        )

    async def test_query_parameters_reach_the_stores(self, setup: Setup) -> None:
        await setup.service.ingest(TEXT)
        query = MemoryQuery(
            text="neo4j", top_k=3, score_threshold=0.1, traversal_depth=1,
            entity_types=(EntityType.TOOL,),
        )  # fmt: skip
        context = await setup.service.query(query)
        ((vector, top_k, threshold),) = setup.vectors.called("search_scored")
        assert (len(vector), top_k, threshold) == (DIM, 3, 0.1)
        assert vector == await setup.embedder.embed_text("neo4j")
        ((seeds, depth),) = setup.graph.called("traverse_neighborhood")
        assert depth == 1
        assert seeds[0] == NEO4J
        assert all(e.entity_type is EntityType.TOOL for e in context.seed_entities)

    async def test_one_entity_read_resolves_seeds_and_path_endpoints(self, setup: Setup) -> None:
        await setup.service.ingest(TEXT)
        await setup.service.query(MemoryQuery(text="memory service", top_k=1))
        (requested,) = setup.graph.called("fetch_entities")
        assert set(requested) == {MEMORY, NEO4J, JVM}

    async def test_no_seeds_returns_the_empty_context(self, setup: Setup) -> None:
        context = await setup.service.query(MemoryQuery(text="anything at all"))
        assert context == GraphContext.empty()
        assert setup.graph.calls == []

    async def test_threshold_filters_everything(self, setup: Setup) -> None:
        await setup.service.ingest(TEXT)
        context = await setup.service.query(
            MemoryQuery(text="completely unrelated words", score_threshold=0.99)
        )
        assert context == GraphContext.empty()

    async def test_vectors_without_graph_nodes_return_empty(self, setup: Setup) -> None:
        await setup.service.ingest(TEXT)
        setup.graph.entities.clear()
        context = await setup.service.query(MemoryQuery(text="memory service"))
        assert context == GraphContext.empty()

    async def test_unresolvable_hop_endpoints_fall_back_to_ids(self, setup: Setup) -> None:
        await setup.service.ingest(TEXT)
        del setup.graph.entities[JVM]
        context = await setup.service.query(MemoryQuery(text="memory service", top_k=1))
        assert f"**{JVM}**" in context.summary
        assert len(context.paths) == 2

    @pytest.mark.parametrize(
        ("score", "confidence"), [(1.0000004, 1.0), (-0.25, 0.0), (0.42, 0.42)]
    )
    async def test_confidence_is_the_best_seed_score_clamped(
        self, setup: Setup, score: float, confidence: float
    ) -> None:
        await setup.service.ingest(TEXT)

        async def fixed(*_args: Any, **_kwargs: Any) -> list[tuple[str, float]]:
            return [("ghost-entity", 0.99), (NEO4J, score)]

        setup.vectors.search_scored = fixed  # type: ignore[method-assign]
        context = await setup.service.query(MemoryQuery(text="neo4j"))
        assert [e.id for e in context.seed_entities] == [NEO4J]
        assert context.confidence == confidence

    async def test_store_errors_propagate(self, setup: Setup) -> None:
        setup.vectors.errors["search_scored"] = VectorSearchTimeoutError(1.0, collection="c")
        with pytest.raises(VectorSearchTimeoutError):
            await setup.service.query(MemoryQuery(text="neo4j"))


# ---------------------------------------------------------------------------
# Summary rendering
# ---------------------------------------------------------------------------
def entity(name: str, description: str = "") -> Entity:
    return Entity(
        id=make_entity_id(name, EntityType.TOOL),
        name=name,
        entity_type=EntityType.TOOL,
        description=description,
    )


def hop(source: str, target: str, depth: int = 1) -> GraphHop:
    return GraphHop(source_id=source, predicate="USES", target_id=target, weight=0.5, depth=depth)


class TestRenderSummary:
    def test_markdown_in_stored_text_is_escaped(self) -> None:
        seed = entity("*bold* [link](x) `code` <b>", "# Heading | table _x_")
        summary = render_summary([seed], [], {})
        assert "\\*bold\\* \\[link\\](x) \\`code\\` \\<b\\>" in summary
        assert "\\# Heading \\| table \\_x\\_" in summary
        assert "## Relationships" not in summary

    def test_long_descriptions_are_flattened_and_shortened(self) -> None:
        seed = entity("Neo4j", "line one\n\nline two " + "x" * 400)
        line = render_summary([seed], [], {}).splitlines()[-1]
        assert "\n" not in line
        assert line.endswith("...")
        assert len(line.split(": ", 1)[1]) == MAX_SUMMARY_DESCRIPTION_CHARS

    def test_relationships_beyond_the_budget_are_counted(self) -> None:
        seed = entity("Neo4j")
        hops = [hop("tool-neo4j", f"tool-{i}") for i in range(50)]
        summary = render_summary([seed], hops, {"tool-neo4j": "Neo4j"}, max_length=400)
        assert len(summary) <= 400
        assert summary.endswith("more relationships")
        listed = summary.count(" USES ")
        assert f"... and {50 - listed} more relationships" in summary

    def test_summary_fits_the_context_contract(self) -> None:
        seeds = tuple(entity(f"Tool {i}", "d" * 4000) for i in range(20))
        hops = tuple(hop(s.id, "tool-x") for s in seeds) * 50
        context = GraphContext(
            seed_entities=seeds,
            paths=hops[:1000],
            summary=render_summary(seeds, hops[:1000], {}),
            confidence=0.5,
        )
        assert len(context.summary) <= 32_768
