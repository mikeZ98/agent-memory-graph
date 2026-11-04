"""Hermetic adapter tests: Neo4j and Qdrant are replaced by in-process doubles.

The doubles record every statement, parameter and call so the tests can assert on
exactly what would have crossed the wire. ``--disable-socket`` stays in force; the
last tests prove that the real clients are blocked rather than silently reaching out.
"""

import asyncio
import math
import uuid
from collections.abc import Callable, Mapping, Sequence
from types import SimpleNamespace
from typing import Any, Self, cast

import httpx
import neo4j
import pytest
from neo4j import AsyncDriver
from neo4j.exceptions import ConfigurationError, Neo4jError, ServiceUnavailable, SessionExpired
from pydantic import SecretStr, ValidationError
from pytest_socket import SocketBlockedError
from qdrant_client import AsyncQdrantClient
from qdrant_client.http.exceptions import (
    ApiException,
    ResponseHandlingException,
    UnexpectedResponse,
)
from qdrant_client.http.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchAny,
    PayloadSchemaType,
    PointStruct,
    VectorParams,
)

from agent_memory_graph.adapters import Neo4jGraphStore, QdrantVectorStore
from agent_memory_graph.adapters.neo4j import (
    FETCH_ENTITIES,
    MERGE_ENTITIES,
    SCHEMA_STATEMENTS,
    WRITE_BATCH_SIZE,
    merge_relations_statement,
    traversal_statement,
)
from agent_memory_graph.adapters.qdrant import POINT_NAMESPACE, point_id
from agent_memory_graph.config import Settings
from agent_memory_graph.domain import (
    Entity,
    EntityIntegrityError,
    EntityType,
    GraphHop,
    KnowledgeGraph,
    MemoryGraphError,
    Relation,
    StoreTimeoutError,
    StoreUnavailableError,
    TraversalDepthExceededError,
    VectorSearchTimeoutError,
)

DIM = 4
TIMEOUT = 0.05
UUID_UPPER = "123E4567-E89B-12D3-A456-426614174000"

type Rows = list[dict[str, Any]]
type Responder = Callable[[str, Mapping[str, Any]], Rows]


def neo4j_error(code: str, message: str = "server said no") -> Neo4jError:
    return Neo4jError._hydrate_neo4j(code=code, message=message)


def unexpected(status: int) -> UnexpectedResponse:
    return UnexpectedResponse(
        status_code=status, reason_phrase="", content=b"{}", headers=httpx.Headers()
    )


def entity(eid: str, *, embedding: Sequence[float] | None = None, **kwargs: Any) -> Entity:
    return Entity(
        id=eid,
        name=kwargs.pop("name", eid),
        entity_type=kwargs.pop("entity_type", EntityType.TOOL),
        embedding=None if embedding is None else tuple(embedding),
        **kwargs,
    )


async def _forever() -> None:
    await asyncio.Event().wait()


# ---------------------------------------------------------------------------
# Neo4j doubles
# ---------------------------------------------------------------------------
def count_responder(query: str, parameters: Mapping[str, Any]) -> Rows:
    """Behave like a healthy server: every MERGE writes one row per input row."""
    rows = parameters.get("rows")
    return [{"written": len(rows)}] if rows is not None else []


class FakeResult:
    def __init__(self, rows: Rows) -> None:
        self._rows = rows

    async def data(self) -> Rows:
        return self._rows


class FakeTransaction:
    def __init__(self, responder: Responder) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self._responder = responder

    async def run(self, query: str, parameters: dict[str, Any] | None = None) -> FakeResult:
        params = dict(parameters or {})
        self.calls.append((query, params))
        return FakeResult(self._responder(query, params))


class FakeSession:
    def __init__(self, driver: "FakeDriver") -> None:
        self._driver = driver

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *_exc: object) -> None:
        self._driver.sessions_closed += 1

    async def _execute(self, mode: str, fn: Callable[[Any], Any]) -> Any:
        self._driver.modes.append(mode)
        self._driver.tx_timeouts.append(getattr(fn, "timeout", None))
        if self._driver.hang:
            await _forever()
        if self._driver.error is not None:
            raise self._driver.error
        return await fn(self._driver.tx)

    async def execute_write(self, fn: Callable[[Any], Any]) -> Any:
        return await self._execute("write", fn)

    async def execute_read(self, fn: Callable[[Any], Any]) -> Any:
        return await self._execute("read", fn)


class FakeDriver:
    def __init__(
        self,
        responder: Responder = count_responder,
        *,
        verify_error: BaseException | None = None,
        verify_hangs: bool = False,
    ) -> None:
        self.tx = FakeTransaction(responder)
        self.verify_error = verify_error
        self.verify_hangs = verify_hangs
        self.error: BaseException | None = None
        self.hang = False
        self.verified = 0
        self.closed = 0
        self.sessions_closed = 0
        self.databases: list[str | None] = []
        self.modes: list[str] = []
        self.tx_timeouts: list[float | None] = []

    def session(self, *, database: str | None = None) -> FakeSession:
        self.databases.append(database)
        return FakeSession(self)

    async def verify_connectivity(self) -> None:
        self.verified += 1
        if self.verify_hangs:
            await _forever()
        if self.verify_error is not None:
            raise self.verify_error

    async def close(self) -> None:
        self.closed += 1


class DriverFactory:
    def __init__(self, driver: FakeDriver) -> None:
        self.driver = driver
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def __call__(self, uri: str, **config: Any) -> AsyncDriver:
        self.calls.append((uri, config))
        return cast("AsyncDriver", self.driver)


def make_graph_store(
    driver: FakeDriver | None = None, **kwargs: Any
) -> tuple[Neo4jGraphStore, DriverFactory]:
    factory = DriverFactory(driver or FakeDriver())
    store = Neo4jGraphStore(
        "neo4j://graph.internal:7687",
        user="memory",
        password=SecretStr("s3cret-pw"),
        database=kwargs.pop("database", "memories"),
        timeout_seconds=kwargs.pop("timeout_seconds", TIMEOUT),
        driver_factory=factory,
        **kwargs,
    )
    return store, factory


@pytest.fixture
async def graph_store() -> tuple[Neo4jGraphStore, FakeDriver]:
    driver = FakeDriver()
    store, _ = make_graph_store(driver)
    await store.connect()
    return store, driver


@pytest.fixture
def knowledge_graph() -> KnowledgeGraph:
    return KnowledgeGraph(
        entities=(
            entity("tool-neo4j", attributes={"z": 1, "a": {"nested": [True, None]}}),
            entity("tool-qdrant"),
            entity("concept-graphrag", entity_type=EntityType.CONCEPT, description="RAG."),
        ),
        relations=(
            Relation(source_id="concept-graphrag", predicate="uses", target_id="tool-qdrant"),
            Relation(
                source_id="concept-graphrag",
                predicate="USES",
                target_id="tool-neo4j",
                weight=0.5,
                properties={"since": 2024},
            ),
            Relation(source_id="tool-qdrant", predicate="dependsOn", target_id="tool-neo4j"),
        ),
    )


# ---------------------------------------------------------------------------
# Neo4j statement composition and injection guards
# ---------------------------------------------------------------------------
class TestCypherComposition:
    def test_relation_statement_backquotes_a_canonical_predicate(self) -> None:
        statement = merge_relations_statement("DEPENDS_ON")
        assert "MERGE (s)-[r:`DEPENDS_ON`]->(t)" in statement
        assert "$rows" in statement
        assert "{id: row.source_id}" in statement

    @pytest.mark.parametrize(
        "predicate",
        [
            "uses",  # not canonical: normalisation is the domain's job, not the adapter's
            "USES`]->(x) DETACH DELETE x //",
            "USES]->() MATCH (n) DETACH DELETE n //",
            "USES\nDELETE",
            "",
            "P" * 65,
            "_USES",
        ],
    )
    def test_relation_statement_refuses_non_canonical_predicates(self, predicate: str) -> None:
        with pytest.raises(ValueError, match="non-canonical predicate"):
            merge_relations_statement(predicate)

    def test_relation_statement_refuses_non_strings(self) -> None:
        with pytest.raises(ValueError, match="non-canonical"):
            merge_relations_statement(cast("str", 42))

    @pytest.mark.parametrize("depth", [1, 2, 3])
    def test_traversal_bound_is_spliced_as_validated_int(self, depth: int) -> None:
        statement = traversal_statement(depth)
        assert f"-[*1..{depth}]->" in statement
        assert "WHERE seed.id IN $seed_ids" in statement
        assert "LIMIT $limit" in statement

    def test_traversal_rejects_excessive_depth_with_domain_error(self) -> None:
        with pytest.raises(TraversalDepthExceededError) as exc:
            traversal_statement(4)
        assert (exc.value.requested, exc.value.maximum) == (4, 3)

    @pytest.mark.parametrize(
        ("depth", "error"),
        [(0, ValueError), (-1, ValueError), (True, TypeError), (2.0, TypeError), ("2", TypeError)],
    )
    def test_traversal_rejects_invalid_depth(self, depth: object, error: type[Exception]) -> None:
        with pytest.raises(error):
            traversal_statement(cast("int", depth))

    def test_entity_statement_is_fully_parameterised(self) -> None:
        assert "MERGE (e:Entity {id: row.id, type: row.type})" in MERGE_ENTITIES
        assert "UNWIND $rows AS row" in MERGE_ENTITIES


# ---------------------------------------------------------------------------
# Neo4j lifecycle
# ---------------------------------------------------------------------------
class TestNeo4jLifecycle:
    async def test_connect_builds_driver_with_bounded_pool_and_timeouts(self) -> None:
        store, factory = make_graph_store(max_pool_size=7)
        await store.connect()
        ((uri, config),) = factory.calls
        assert uri == "neo4j://graph.internal:7687"
        assert config == {
            "auth": ("memory", "s3cret-pw"),
            "max_connection_pool_size": 7,
            "connection_timeout": TIMEOUT,
            "connection_acquisition_timeout": TIMEOUT,
            "max_transaction_retry_time": TIMEOUT,
        }
        assert factory.driver.verified == 1
        assert store.is_connected

    async def test_connect_is_idempotent_under_concurrency(self) -> None:
        store, factory = make_graph_store()
        await asyncio.gather(*(store.connect() for _ in range(5)))
        assert len(factory.calls) == 1

    async def test_close_is_idempotent(self) -> None:
        store, factory = make_graph_store()
        await store.close()
        await store.connect()
        await store.close()
        await store.close()
        assert factory.driver.closed == 1
        assert not store.is_connected

    async def test_async_context_manager(self) -> None:
        store, factory = make_graph_store()
        async with store as entered:
            assert entered is store
            assert store.is_connected
        assert factory.driver.closed == 1

    async def test_from_settings(self) -> None:
        settings = Settings(
            neo4j_uri="neo4j+s://db.example:7687",
            neo4j_password=SecretStr("pw"),
            neo4j_database="graph",
            neo4j_max_pool_size=3,
            default_timeout_seconds=2.0,
        )
        factory = DriverFactory(FakeDriver())
        store = Neo4jGraphStore.from_settings(settings, driver_factory=factory)
        await store.connect()
        await store.traverse_neighborhood(["tool-x"])
        uri, config = factory.calls[0]
        assert (uri, config["auth"], config["max_connection_pool_size"]) == (
            "neo4j+s://db.example:7687",
            ("neo4j", "pw"),
            3,
        )
        assert factory.driver.databases == ["graph"]
        assert factory.driver.tx_timeouts == [2.0]

    @pytest.mark.parametrize(
        ("error", "expected"),
        [
            (ServiceUnavailable("connection refused"), StoreUnavailableError),
            (neo4j_error("Neo.ClientError.Security.Unauthorized"), MemoryGraphError),
        ],
    )
    async def test_failed_verification_closes_driver_and_maps_error(
        self, error: Exception, expected: type[MemoryGraphError]
    ) -> None:
        driver = FakeDriver(verify_error=error)
        store, _ = make_graph_store(driver)
        with pytest.raises(expected) as exc:
            await store.connect()
        assert type(exc.value) is expected
        assert exc.value.__cause__ is error
        assert driver.closed == 1
        assert not store.is_connected

    async def test_rejected_credentials_do_not_leak_server_message(self) -> None:
        error = neo4j_error("Neo.ClientError.Security.Unauthorized", "user 'agent-x9' bad pw")
        store, _ = make_graph_store(FakeDriver(verify_error=error))
        with pytest.raises(MemoryGraphError) as exc:
            await store.connect()
        assert exc.value.detail == "Neo4j rejected the configured credentials."
        rendered = str(exc.value.to_problem().to_json_dict())
        assert "agent-x9" not in rendered
        assert "bad pw" not in rendered

    async def test_connect_timeout(self) -> None:
        driver = FakeDriver(verify_hangs=True)
        store, _ = make_graph_store(driver)
        with pytest.raises(StoreTimeoutError) as exc:
            await store.connect()
        assert exc.value.operation == "connect"
        assert driver.closed == 1

    async def test_operations_require_connect(self, knowledge_graph: KnowledgeGraph) -> None:
        store, _ = make_graph_store()
        with pytest.raises(MemoryGraphError, match=r"ingest_graph\(\) called before connect"):
            await store.ingest_graph(knowledge_graph)


# ---------------------------------------------------------------------------
# Neo4j ingestion
# ---------------------------------------------------------------------------
class TestIngestGraph:
    async def test_single_write_transaction_with_bound_parameters(
        self, graph_store: tuple[Neo4jGraphStore, FakeDriver], knowledge_graph: KnowledgeGraph
    ) -> None:
        store, driver = graph_store
        await store.ingest_graph(knowledge_graph)

        assert driver.modes == ["write"]
        assert driver.databases == ["memories"]
        assert driver.tx_timeouts == [TIMEOUT]
        assert driver.sessions_closed == 1

        (entity_call, *relation_calls) = driver.tx.calls
        assert entity_call[0] == MERGE_ENTITIES
        assert entity_call[1]["rows"][0] == {
            "id": "tool-neo4j",
            "type": "TOOL",
            "name": "tool-neo4j",
            "description": "",
            "attributes_json": '{"a":{"nested":[true,null]},"z":1}',
        }
        assert [row["type"] for row in entity_call[1]["rows"]] == ["TOOL", "TOOL", "CONCEPT"]

        # One statement per predicate, in sorted order, each fully parameterised.
        assert [query for query, _ in relation_calls] == [
            merge_relations_statement("DEPENDS_ON"),
            merge_relations_statement("USES"),
        ]
        assert relation_calls[1][1]["rows"] == [
            {
                "source_id": "concept-graphrag",
                "target_id": "tool-qdrant",
                "weight": 1.0,
                "properties_json": "{}",
            },
            {
                "source_id": "concept-graphrag",
                "target_id": "tool-neo4j",
                "weight": 0.5,
                "properties_json": '{"since":2024}',
            },
        ]

    async def test_values_never_reach_statement_text(
        self, graph_store: tuple[Neo4jGraphStore, FakeDriver]
    ) -> None:
        store, driver = graph_store
        hostile = "x'}) DETACH DELETE n //"
        await store.ingest_graph(
            KnowledgeGraph(entities=(entity("tool-x", name=hostile, description=hostile),))
        )
        ((query, params),) = driver.tx.calls
        assert hostile not in query
        assert params["rows"][0]["name"] == hostile

    async def test_large_graphs_are_batched(
        self, graph_store: tuple[Neo4jGraphStore, FakeDriver]
    ) -> None:
        store, driver = graph_store
        count = WRITE_BATCH_SIZE + 1
        nodes = tuple(entity(f"tool-{i}") for i in range(count))
        edges = tuple(
            Relation(source_id="tool-0", predicate="LINKS", target_id=f"tool-{i}")
            for i in range(count)
        )
        await store.ingest_graph(KnowledgeGraph(entities=nodes, relations=edges))
        sizes = [len(params["rows"]) for _, params in driver.tx.calls]
        assert sizes == [WRITE_BATCH_SIZE, 1, WRITE_BATCH_SIZE, 1]
        assert driver.modes == ["write"]

    async def test_empty_graph_opens_no_session(
        self, graph_store: tuple[Neo4jGraphStore, FakeDriver]
    ) -> None:
        store, driver = graph_store
        await store.ingest_graph(KnowledgeGraph())
        assert driver.databases == []

    @pytest.mark.parametrize("written", [[{"written": 0}], [{"written": 2}], []])
    async def test_relation_count_mismatch_aborts_transaction(
        self, knowledge_graph: KnowledgeGraph, written: Rows
    ) -> None:
        def responder(query: str, parameters: Mapping[str, Any]) -> Rows:
            return written if "MATCH (s:Entity" in query else count_responder(query, parameters)

        driver = FakeDriver(responder)
        store, _ = make_graph_store(driver)
        await store.connect()
        with pytest.raises(EntityIntegrityError, match="DEPENDS_ON: merged"):
            await store.ingest_graph(knowledge_graph)

    async def test_bypassed_validation_cannot_inject_cypher(
        self, graph_store: tuple[Neo4jGraphStore, FakeDriver]
    ) -> None:
        store, driver = graph_store
        smuggled = Relation.model_construct(
            source_id="tool-a",
            predicate="USES]->(x) MATCH (n) DETACH DELETE n //",
            target_id="tool-b",
            weight=1.0,
            properties={},
        )
        graph = KnowledgeGraph.model_construct(
            entities=(entity("tool-a"), entity("tool-b")), relations=(smuggled,)
        )
        with pytest.raises(ValueError, match="non-canonical predicate"):
            await store.ingest_graph(graph)
        assert driver.databases == []  # rejected before any session was opened

    async def test_ensure_schema_creates_uniqueness_constraint(
        self, graph_store: tuple[Neo4jGraphStore, FakeDriver]
    ) -> None:
        store, driver = graph_store
        await store.ensure_schema()
        assert [query for query, _ in driver.tx.calls] == list(SCHEMA_STATEMENTS)
        assert "IF NOT EXISTS" in SCHEMA_STATEMENTS[0]
        assert driver.modes == ["write"]


# ---------------------------------------------------------------------------
# Neo4j traversal
# ---------------------------------------------------------------------------
HOP_ROWS: Rows = [
    {"source_id": "concept-graphrag", "predicate": "USES", "target_id": "tool-neo4j",
     "weight": 0.5, "depth": 1},
    {"source_id": "tool-neo4j", "predicate": "WRITTEN_IN", "target_id": "concept-java",
     "weight": 1.0, "depth": 2},
]  # fmt: skip


class TestTraverseNeighborhood:
    async def test_returns_immutable_hops_from_a_read_transaction(self) -> None:
        driver = FakeDriver(lambda _q, _p: HOP_ROWS)
        store, _ = make_graph_store(driver)
        await store.connect()

        hops = await store.traverse_neighborhood(["concept-graphrag", UUID_UPPER], depth=3)

        assert isinstance(hops, tuple)
        assert all(isinstance(hop, GraphHop) for hop in hops)
        assert [str(hop) for hop in hops] == [
            "(concept-graphrag)-[USES]->(tool-neo4j)",
            "(tool-neo4j)-[WRITTEN_IN]->(concept-java)",
        ]
        ((query, params),) = driver.tx.calls
        assert query == traversal_statement(3)
        assert params == {"seed_ids": ["concept-graphrag", UUID_UPPER.lower()], "limit": 1000}
        assert driver.modes == ["read"]

    async def test_default_depth_is_two(
        self, graph_store: tuple[Neo4jGraphStore, FakeDriver]
    ) -> None:
        store, driver = graph_store
        assert await store.traverse_neighborhood(["tool-x"]) == ()
        assert driver.tx.calls[0][0] == traversal_statement(2)

    async def test_seed_ids_are_deduplicated_in_order(
        self, graph_store: tuple[Neo4jGraphStore, FakeDriver]
    ) -> None:
        store, driver = graph_store
        await store.traverse_neighborhood(("b", "a", "b"), depth=1, limit=10)
        assert driver.tx.calls[0][1] == {"seed_ids": ["b", "a"], "limit": 10}

    async def test_no_seeds_means_no_query(
        self, graph_store: tuple[Neo4jGraphStore, FakeDriver]
    ) -> None:
        store, driver = graph_store
        assert await store.traverse_neighborhood([]) == ()
        assert driver.databases == []

    async def test_depth_is_checked_before_anything_else(
        self, graph_store: tuple[Neo4jGraphStore, FakeDriver]
    ) -> None:
        store, driver = graph_store
        with pytest.raises(TraversalDepthExceededError):
            await store.traverse_neighborhood(["tool-x"], depth=4)
        assert driver.databases == []

    @pytest.mark.parametrize(
        ("seed_ids", "limit", "error", "match"),
        [
            ("tool-x", 10, TypeError, "single string"),
            (["Not An Id"], 10, ValueError, "invalid entity id"),
            ([f"tool-{i}" for i in range(1001)], 10, ValueError, "at most 1000"),
            (["tool-x"], 0, ValidationError, "greater than or equal"),
            (["tool-x"], 1001, ValidationError, "less than or equal"),
            (["tool-x"], True, ValidationError, "valid integer"),
        ],
    )
    async def test_invalid_arguments(
        self,
        graph_store: tuple[Neo4jGraphStore, FakeDriver],
        seed_ids: Sequence[str],
        limit: int,
        error: type[Exception],
        match: str,
    ) -> None:
        store, driver = graph_store
        with pytest.raises(error, match=match):
            await store.traverse_neighborhood(seed_ids, limit=limit)
        assert driver.databases == []

    @pytest.mark.parametrize(
        "row",
        [
            {**HOP_ROWS[0], "weight": None},  # foreign edge without a weight
            {**HOP_ROWS[0], "depth": 9},
            {**HOP_ROWS[0], "unexpected": 1},
        ],
    )
    async def test_rows_violating_the_contract_fail_closed(self, row: dict[str, Any]) -> None:
        store, _ = make_graph_store(FakeDriver(lambda _q, _p: [row]))
        await store.connect()
        with pytest.raises(MemoryGraphError, match="violates the domain contract") as exc:
            await store.traverse_neighborhood(["tool-x"])
        assert isinstance(exc.value.__cause__, ValidationError)


# ---------------------------------------------------------------------------
# Neo4j error translation
# ---------------------------------------------------------------------------
class TestNeo4jErrorMapping:
    @pytest.mark.parametrize(
        ("error", "expected", "context"),
        [
            (ServiceUnavailable("routing failed"), StoreUnavailableError, {"store": "neo4j"}),
            (SessionExpired("gone"), StoreUnavailableError, {"store": "neo4j"}),
            (
                neo4j_error("Neo.TransientError.General.MemoryPoolOutOfMemoryError"),
                StoreUnavailableError,
                {"store": "neo4j"},
            ),
            (
                neo4j_error("Neo.ClientError.Transaction.TransactionTimedOut"),
                StoreTimeoutError,
                {"store": "neo4j", "operation": "ingest_graph", "timeout_seconds": TIMEOUT},
            ),
            (
                neo4j_error("Neo.ClientError.Transaction.TransactionTimedOutClientConfiguration"),
                StoreTimeoutError,
                {"store": "neo4j", "operation": "ingest_graph", "timeout_seconds": TIMEOUT},
            ),
            (
                neo4j_error("Neo.ClientError.Schema.ConstraintValidationFailed"),
                EntityIntegrityError,
                {"dangling_ids": (), "duplicate_ids": ()},
            ),
            (
                neo4j_error("Neo.ClientError.Security.Unauthorized"),
                MemoryGraphError,
                {"store": "neo4j", "operation": "ingest_graph"},
            ),
            (
                neo4j_error("Neo.ClientError.Statement.SyntaxError"),
                MemoryGraphError,
                {
                    "store": "neo4j",
                    "operation": "ingest_graph",
                    "neo4j_code": "Neo.ClientError.Statement.SyntaxError",
                },
            ),
            (
                ConfigurationError("bad config"),
                MemoryGraphError,
                {"store": "neo4j", "operation": "ingest_graph"},
            ),
        ],
    )
    async def test_driver_errors_become_domain_errors(
        self,
        graph_store: tuple[Neo4jGraphStore, FakeDriver],
        knowledge_graph: KnowledgeGraph,
        error: Exception,
        expected: type[MemoryGraphError],
        context: dict[str, Any],
    ) -> None:
        store, driver = graph_store
        driver.error = error
        with pytest.raises(MemoryGraphError) as exc:
            await store.ingest_graph(knowledge_graph)
        assert type(exc.value) is expected
        assert dict(exc.value.context) == context
        assert exc.value.__cause__ is error
        assert driver.sessions_closed == 1

    async def test_client_side_deadline(
        self, graph_store: tuple[Neo4jGraphStore, FakeDriver]
    ) -> None:
        store, driver = graph_store
        driver.hang = True
        with pytest.raises(StoreTimeoutError) as exc:
            await store.traverse_neighborhood(["tool-x"])
        assert (exc.value.operation, exc.value.status) == ("traverse_neighborhood", 504)

    async def test_unrelated_exceptions_are_not_masked(
        self, graph_store: tuple[Neo4jGraphStore, FakeDriver]
    ) -> None:
        store, driver = graph_store
        driver.error = KeyError("bug")
        with pytest.raises(KeyError):
            await store.traverse_neighborhood(["tool-x"])


# ---------------------------------------------------------------------------
# Qdrant doubles
# ---------------------------------------------------------------------------
class FakeQdrant:
    def __init__(
        self,
        *,
        exists: bool = False,
        vectors: object = None,
        points: Sequence[Mapping[str, Any] | None] = (),
    ) -> None:
        self.exists = exists
        self.vectors = (
            vectors if vectors is not None else VectorParams(size=DIM, distance=Distance.COSINE)
        )
        self.points = points
        self.calls: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []
        self.errors: dict[str, BaseException] = {}
        self.hangs: set[str] = set()
        self.closed = 0

    async def _record(self, method: str, *args: Any, **kwargs: Any) -> None:
        self.calls.append((method, args, kwargs))
        if method in self.hangs:
            await _forever()
        if method in self.errors:
            raise self.errors[method]

    def called(self, method: str) -> list[tuple[tuple[Any, ...], dict[str, Any]]]:
        return [(args, kwargs) for name, args, kwargs in self.calls if name == method]

    async def collection_exists(self, name: str) -> bool:
        await self._record("collection_exists", name)
        return self.exists

    async def create_collection(self, name: str, **kwargs: Any) -> bool:
        await self._record("create_collection", name, **kwargs)
        self.exists = True
        return True

    async def create_payload_index(self, name: str, **kwargs: Any) -> None:
        await self._record("create_payload_index", name, **kwargs)

    async def get_collection(self, name: str) -> SimpleNamespace:
        await self._record("get_collection", name)
        return SimpleNamespace(config=SimpleNamespace(params=SimpleNamespace(vectors=self.vectors)))

    async def upsert(self, collection: str, **kwargs: Any) -> None:
        await self._record("upsert", collection, **kwargs)

    async def query_points(self, collection: str, **kwargs: Any) -> SimpleNamespace:
        await self._record("query_points", collection, **kwargs)
        return SimpleNamespace(points=[SimpleNamespace(payload=p, score=0.9) for p in self.points])

    async def close(self) -> None:
        self.closed += 1


class ClientFactory:
    def __init__(self, client: FakeQdrant) -> None:
        self.client = client
        self.calls: list[dict[str, Any]] = []

    def __call__(self, **kwargs: Any) -> AsyncQdrantClient:
        self.calls.append(kwargs)
        return cast("AsyncQdrantClient", self.client)


def make_vector_store(
    client: FakeQdrant | None = None, **kwargs: Any
) -> tuple[QdrantVectorStore, ClientFactory]:
    factory = ClientFactory(client or FakeQdrant())
    store = QdrantVectorStore(
        "https://vectors.internal:6333",
        collection=kwargs.pop("collection", "memories"),
        embedding_dim=DIM,
        api_key=kwargs.pop("api_key", SecretStr("qdrant-key")),
        timeout_seconds=kwargs.pop("timeout_seconds", TIMEOUT),
        client_factory=factory,
        **kwargs,
    )
    return store, factory


async def connected_vector_store(
    client: FakeQdrant | None = None, **kwargs: Any
) -> tuple[QdrantVectorStore, FakeQdrant]:
    store, factory = make_vector_store(client, **kwargs)
    await store.connect()
    return store, factory.client


VECTOR = (0.1, 0.2, 0.3, 0.4)


# ---------------------------------------------------------------------------
# Qdrant lifecycle
# ---------------------------------------------------------------------------
class TestQdrantLifecycle:
    def test_point_ids_are_deterministic_uuid5(self) -> None:
        assert point_id("tool-neo4j") == str(uuid.uuid5(POINT_NAMESPACE, "tool-neo4j"))
        assert point_id("tool-neo4j") != point_id("tool-qdrant")
        assert uuid.UUID(point_id("tool-neo4j")).version == 5

    @pytest.mark.parametrize(("budget", "request_timeout"), [(0.05, 1), (2.0, 2), (2.5, 3)])
    async def test_connect_builds_rest_client_without_background_probe(
        self, budget: float, request_timeout: int
    ) -> None:
        store, factory = make_vector_store(timeout_seconds=budget)
        await store.connect()
        await store.connect()
        assert factory.calls == [
            {
                "url": "https://vectors.internal:6333",
                "api_key": "qdrant-key",
                "timeout": request_timeout,
                "prefer_grpc": False,
                "check_compatibility": False,
            }
        ]
        assert store.is_connected

    async def test_connect_without_api_key(self) -> None:
        store, factory = make_vector_store(api_key=None)
        await store.connect()
        assert factory.calls[0]["api_key"] is None

    async def test_close_and_context_manager(self) -> None:
        store, factory = make_vector_store()
        await store.close()
        async with store as entered:
            assert entered is store
        await store.close()
        assert factory.client.closed == 1
        assert not store.is_connected

    async def test_from_settings(self) -> None:
        settings = Settings(
            neo4j_password=SecretStr("pw"),
            qdrant_url="https://q.example",
            qdrant_api_key=SecretStr("k"),
            qdrant_collection="facts",
            embedding_dim=1536,
            qdrant_upsert_batch_size=8,
        )
        factory = ClientFactory(FakeQdrant())
        store = QdrantVectorStore.from_settings(settings, client_factory=factory)
        await store.connect()
        assert (store.collection, store.embedding_dim) == ("facts", 1536)
        assert factory.calls[0]["url"] == "https://q.example"
        assert factory.calls[0]["timeout"] == 10

    async def test_operations_require_connect(self) -> None:
        store, _ = make_vector_store()
        with pytest.raises(MemoryGraphError, match=r"search_seeds\(\) called before connect"):
            await store.search_seeds(VECTOR, top_k=1, score_threshold=0.0)


# ---------------------------------------------------------------------------
# Qdrant collection management
# ---------------------------------------------------------------------------
class TestEnsureCollection:
    async def test_creates_cosine_collection_and_type_index(self) -> None:
        store, client = await connected_vector_store()
        await store.ensure_collection()
        assert [name for name, _, _ in client.calls] == [
            "collection_exists",
            "create_collection",
            "create_payload_index",
            "get_collection",
        ]
        ((args, kwargs),) = client.called("create_collection")
        assert args == ("memories",)
        assert kwargs == {"vectors_config": VectorParams(size=DIM, distance=Distance.COSINE)}
        ((_, index_kwargs),) = client.called("create_payload_index")
        assert index_kwargs == {
            "field_name": "entity_type",
            "field_schema": PayloadSchemaType.KEYWORD,
        }

    async def test_existing_compatible_collection_is_left_alone(self) -> None:
        store, client = await connected_vector_store(FakeQdrant(exists=True))
        await store.ensure_collection("other")
        assert client.called("create_collection") == []
        assert client.called("get_collection") == [(("other",), {})]

    async def test_losing_a_creation_race_is_fine(self) -> None:
        client = FakeQdrant()
        client.errors["create_collection"] = unexpected(409)
        store, _ = await connected_vector_store(client)
        await store.ensure_collection()
        assert len(client.called("get_collection")) == 1

    async def test_other_creation_failures_propagate_as_domain_errors(self) -> None:
        client = FakeQdrant()
        client.errors["create_collection"] = unexpected(400)
        store, _ = await connected_vector_store(client)
        with pytest.raises(MemoryGraphError) as exc:
            await store.ensure_collection()
        assert exc.value.context["http_status"] == 400

    @pytest.mark.parametrize(
        "vectors",
        [
            VectorParams(size=DIM + 1, distance=Distance.COSINE),
            VectorParams(size=DIM, distance=Distance.DOT),
            {"named": VectorParams(size=DIM, distance=Distance.COSINE)},
        ],
    )
    async def test_incompatible_existing_collection_is_reported(self, vectors: object) -> None:
        store, _ = await connected_vector_store(FakeQdrant(exists=True, vectors=vectors))
        with pytest.raises(MemoryGraphError, match="incompatible vector configuration") as exc:
            await store.ensure_collection()
        assert exc.value.context["expected_size"] == DIM
        assert exc.value.context["expected_distance"] == "Cosine"

    async def test_timeout_maps_to_store_timeout(self) -> None:
        client = FakeQdrant()
        client.hangs.add("collection_exists")
        store, _ = await connected_vector_store(client)
        with pytest.raises(StoreTimeoutError) as exc:
            await store.ensure_collection()
        assert exc.value.context == {
            "store": "qdrant",
            "operation": "ensure_collection",
            "timeout_seconds": TIMEOUT,
        }


# ---------------------------------------------------------------------------
# Qdrant upserts
# ---------------------------------------------------------------------------
class TestUpsertEntities:
    async def test_points_carry_uuid_ids_and_domain_payload(self) -> None:
        store, client = await connected_vector_store()
        written = await store.upsert_entities(
            [entity("tool-neo4j", name="Neo4j", embedding=VECTOR), entity("tool-no-vector")]
        )
        assert written == 1
        ((args, kwargs),) = client.called("upsert")
        assert args == ("memories",)
        assert kwargs["wait"] is True
        assert kwargs["timeout"] == 1
        assert kwargs["points"] == [
            PointStruct(
                id=point_id("tool-neo4j"),
                vector=list(VECTOR),
                payload={"entity_id": "tool-neo4j", "name": "Neo4j", "entity_type": "TOOL"},
            )
        ]

    async def test_upserts_are_batched(self) -> None:
        store, client = await connected_vector_store(batch_size=2)
        entities = [entity(f"tool-{i}", embedding=VECTOR) for i in range(5)]
        assert await store.upsert_entities(entities) == 5
        batches = [kwargs["points"] for _, kwargs in client.called("upsert")]
        assert [len(batch) for batch in batches] == [2, 2, 1]
        assert [p.payload["entity_id"] for batch in batches for p in batch] == [
            f"tool-{i}" for i in range(5)
        ]

    async def test_nothing_to_upsert(self) -> None:
        store, client = await connected_vector_store()
        assert await store.upsert_entities([entity("tool-x")]) == 0
        assert client.calls == []

    async def test_dimension_mismatch_is_rejected_before_any_write(self) -> None:
        store, client = await connected_vector_store(batch_size=1)
        entities = [entity("tool-a", embedding=VECTOR), entity("tool-b", embedding=(1.0, 2.0))]
        with pytest.raises(ValueError, match="entity tool-b has 2 dimensions, expected 4"):
            await store.upsert_entities(entities)
        assert client.calls == []

    async def test_unvalidated_non_finite_vectors_are_rejected(self) -> None:
        store, client = await connected_vector_store()
        smuggled = Entity.model_construct(
            id="tool-x",
            name="X",
            entity_type=EntityType.TOOL,
            description="",
            attributes={},
            embedding=(math.nan, 0.0, 0.0, 0.0),
        )
        with pytest.raises(ValueError, match="finite numbers"):
            await store.upsert_entities([smuggled])
        assert client.calls == []

    async def test_upsert_failures_are_mapped(self) -> None:
        client = FakeQdrant()
        client.errors["upsert"] = ResponseHandlingException(httpx.ConnectError("refused"))
        store, _ = await connected_vector_store(client)
        with pytest.raises(StoreUnavailableError):
            await store.upsert_entities([entity("tool-x", embedding=VECTOR)])


# ---------------------------------------------------------------------------
# Qdrant seed search
# ---------------------------------------------------------------------------
class TestSearchSeeds:
    async def test_query_parameters_and_ranked_unique_ids(self) -> None:
        points = [
            {"entity_id": "tool-neo4j"},
            {"entity_id": UUID_UPPER},
            {"entity_id": "tool-neo4j"},
        ]
        store, client = await connected_vector_store(FakeQdrant(points=points))
        ids = await store.search_seeds(
            [1, 0, 0, 0], top_k=3, score_threshold=0.25, allowed_types=None
        )
        assert ids == ["tool-neo4j", UUID_UPPER.lower()]
        ((args, kwargs),) = client.called("query_points")
        assert args == ("memories",)
        assert kwargs == {
            "query": [1.0, 0.0, 0.0, 0.0],
            "limit": 3,
            "score_threshold": 0.25,
            "query_filter": None,
            "with_payload": ["entity_id"],
            "with_vectors": False,
            "timeout": 1,
        }

    async def test_type_filter_is_index_friendly_match_any(self) -> None:
        store, client = await connected_vector_store()
        await store.search_seeds(
            VECTOR,
            top_k=5,
            score_threshold=0,
            allowed_types=[EntityType.TOOL, EntityType.PERSON, EntityType.TOOL],
        )
        ((_, kwargs),) = client.called("query_points")
        assert kwargs["query_filter"] == Filter(
            must=[FieldCondition(key="entity_type", match=MatchAny(any=["PERSON", "TOOL"]))]
        )

    async def test_empty_type_filter_matches_nothing_without_a_request(self) -> None:
        store, client = await connected_vector_store()
        assert await store.search_seeds(VECTOR, 5, 0.0, allowed_types=[]) == []
        assert client.calls == []

    @pytest.mark.parametrize(
        ("vector", "top_k", "threshold", "match"),
        [
            ((0.1, 0.2), 5, 0.0, "2 dimensions"),
            ((), 5, 0.0, "non-empty sequence"),
            ((math.inf, 0, 0, 0), 5, 0.0, "finite numbers"),
            ((True, 0, 0, 0), 5, 0.0, "finite numbers"),
            (VECTOR, 0, 0.0, "greater than or equal"),
            (VECTOR, 21, 0.0, "less than or equal"),
            (VECTOR, True, 0.0, "valid integer"),
            (VECTOR, 5, 1.5, "less than or equal"),
            (VECTOR, 5, -0.1, "greater than or equal"),
            (VECTOR, 5, "0.5", "valid number"),
        ],
    )
    async def test_invalid_arguments_never_reach_qdrant(
        self, vector: Sequence[float], top_k: int, threshold: float, match: str
    ) -> None:
        store, client = await connected_vector_store()
        with pytest.raises(ValueError, match=match):
            await store.search_seeds(vector, top_k, threshold)
        assert client.calls == []

    @pytest.mark.parametrize("payload", [None, {}, {"entity_id": 7}, {"entity_id": "Not An Id"}])
    async def test_points_with_bad_payload_fail_closed(
        self, payload: Mapping[str, Any] | None
    ) -> None:
        store, _ = await connected_vector_store(FakeQdrant(points=[payload]))
        with pytest.raises(MemoryGraphError, match="entity_id payload is missing or invalid"):
            await store.search_seeds(VECTOR, 5, 0.0)

    @pytest.mark.parametrize(
        "error",
        [
            ResponseHandlingException(httpx.ReadTimeout("read timed out")),
            ResponseHandlingException(TimeoutError()),
            httpx.WriteTimeout("write timed out"),
            unexpected(504),
        ],
    )
    async def test_timeouts_raise_vector_search_timeout(self, error: BaseException) -> None:
        client = FakeQdrant()
        client.errors["query_points"] = error
        store, _ = await connected_vector_store(client)
        with pytest.raises(VectorSearchTimeoutError) as exc:
            await store.search_seeds(VECTOR, 5, 0.0)
        assert exc.value.context == {"timeout_seconds": TIMEOUT, "collection": "memories"}
        assert exc.value.status == 504
        assert exc.value.__cause__ is error

    async def test_client_side_deadline_raises_vector_search_timeout(self) -> None:
        client = FakeQdrant()
        client.hangs.add("query_points")
        store, _ = await connected_vector_store(client)
        with pytest.raises(VectorSearchTimeoutError):
            await store.search_seeds(VECTOR, 5, 0.0)


class TestQdrantErrorMapping:
    @pytest.mark.parametrize(
        ("error", "expected", "detail"),
        [
            (ResponseHandlingException(httpx.ConnectError("refused")), StoreUnavailableError, None),
            (httpx.RemoteProtocolError("reset"), StoreUnavailableError, None),
            (unexpected(503), StoreUnavailableError, None),
            (unexpected(502), StoreUnavailableError, None),
            (unexpected(401), MemoryGraphError, "Qdrant rejected the configured API key."),
            (unexpected(403), MemoryGraphError, "Qdrant rejected the configured API key."),
            (unexpected(500), MemoryGraphError, "Qdrant search_seeds failed."),
            (
                ResponseHandlingException(ValueError("bad json")),
                MemoryGraphError,
                "Qdrant client error during search_seeds.",
            ),
            (ApiException(), MemoryGraphError, "Qdrant client error during search_seeds."),
        ],
    )
    async def test_client_errors_become_domain_errors(
        self, error: BaseException, expected: type[MemoryGraphError], detail: str | None
    ) -> None:
        client = FakeQdrant()
        client.errors["query_points"] = error
        store, _ = await connected_vector_store(client)
        with pytest.raises(MemoryGraphError) as exc:
            await store.search_seeds(VECTOR, 5, 0.0)
        assert type(exc.value) is expected
        if detail is not None:
            assert exc.value.detail == detail
        assert exc.value.__cause__ is error

    async def test_unrelated_exceptions_are_not_masked(self) -> None:
        client = FakeQdrant()
        client.errors["query_points"] = KeyError("bug")
        store, _ = await connected_vector_store(client)
        with pytest.raises(KeyError):
            await store.search_seeds(VECTOR, 5, 0.0)


# ---------------------------------------------------------------------------
# Hermeticity: the real clients are constructed, and blocked if they reach out
# ---------------------------------------------------------------------------
class TestRealClientsStayOffline:
    async def test_real_neo4j_driver_is_created_without_network(self) -> None:
        store = Neo4jGraphStore("neo4j://localhost:7687", user="u", password=SecretStr("p"))
        driver = store._create_driver()
        assert isinstance(driver, neo4j.AsyncDriver)
        await driver.close()

    async def test_real_qdrant_client_is_created_without_network(self) -> None:
        store = QdrantVectorStore("http://localhost:6333", collection="c", embedding_dim=DIM)
        async with store:
            assert isinstance(store._client, AsyncQdrantClient)

    async def test_real_qdrant_request_is_blocked_by_socket_guard(self) -> None:
        store = QdrantVectorStore("http://localhost:6333", collection="c", embedding_dim=DIM)
        async with store:
            with (
                pytest.warns(UserWarning, match="tried to use socket"),
                pytest.raises(MemoryGraphError) as exc,
            ):
                await store.search_seeds(VECTOR, 1, 0.0)
        cause = exc.value.__cause__
        assert isinstance(cause, ResponseHandlingException)
        assert isinstance(cause.source, SocketBlockedError)


# ---------------------------------------------------------------------------
# Neo4j entity lookup and health probe
# ---------------------------------------------------------------------------
ENTITY_ROWS: Rows = [
    {"id": "tool-qdrant", "name": "Qdrant", "entity_type": "TOOL", "description": "",
     "attributes_json": "{}"},
    {"id": "tool-neo4j", "name": "Neo4j", "entity_type": "TOOL", "description": "Graph DB.",
     "attributes_json": '{"license":"GPL","tags":["graph"]}'},
]  # fmt: skip


class TestFetchEntities:
    async def test_returns_entities_in_requested_order_skipping_unknown_ids(self) -> None:
        driver = FakeDriver(lambda _q, _p: ENTITY_ROWS)
        store, _ = make_graph_store(driver)
        await store.connect()
        entities = await store.fetch_entities(["tool-neo4j", "tool-missing", "tool-qdrant"])
        assert [e.id for e in entities] == ["tool-neo4j", "tool-qdrant"]
        assert entities[0].attributes == {"license": "GPL", "tags": ("graph",)}
        assert entities[0].embedding is None
        ((query, params),) = driver.tx.calls
        assert query == FETCH_ENTITIES
        assert params == {"ids": ["tool-neo4j", "tool-missing", "tool-qdrant"]}
        assert driver.modes == ["read"]

    async def test_no_ids_means_no_query(
        self, graph_store: tuple[Neo4jGraphStore, FakeDriver]
    ) -> None:
        store, driver = graph_store
        assert await store.fetch_entities([]) == ()
        assert driver.databases == []

    @pytest.mark.parametrize(
        ("ids", "error", "match"),
        [
            ("tool-x", TypeError, "entity_ids must be a sequence"),
            (["Bad Id"], ValueError, "entity_ids contains an invalid entity id"),
            ([f"t-{i}" for i in range(3001)], ValueError, "at most 3000 entity_ids"),
        ],
    )
    async def test_invalid_ids(
        self,
        graph_store: tuple[Neo4jGraphStore, FakeDriver],
        ids: Sequence[str],
        error: type[Exception],
        match: str,
    ) -> None:
        store, _ = graph_store
        with pytest.raises(error, match=match):
            await store.fetch_entities(ids)

    @pytest.mark.parametrize(
        "row",
        [
            {**ENTITY_ROWS[0], "attributes_json": "{not json"},
            {**ENTITY_ROWS[0], "entity_type": "ANIMAL"},
            {**ENTITY_ROWS[0], "name": None},
            {k: v for k, v in ENTITY_ROWS[0].items() if k != "name"},
        ],
    )
    async def test_stored_nodes_violating_the_contract_fail_closed(
        self, row: dict[str, Any]
    ) -> None:
        store, _ = make_graph_store(FakeDriver(lambda _q, _p: [row]))
        await store.connect()
        with pytest.raises(MemoryGraphError, match="entity that violates the domain contract"):
            await store.fetch_entities(["tool-qdrant"])


class TestPing:
    async def test_neo4j_ping_verifies_connectivity(
        self, graph_store: tuple[Neo4jGraphStore, FakeDriver]
    ) -> None:
        store, driver = graph_store
        await store.ping()
        assert driver.verified == 2  # connect + ping

    async def test_neo4j_ping_maps_failures(
        self, graph_store: tuple[Neo4jGraphStore, FakeDriver]
    ) -> None:
        store, driver = graph_store
        driver.verify_error = ServiceUnavailable("down")
        with pytest.raises(StoreUnavailableError):
            await store.ping()
        driver.verify_error, driver.verify_hangs = None, True
        with pytest.raises(StoreTimeoutError) as exc:
            await store.ping()
        assert exc.value.operation == "ping"

    async def test_neo4j_ping_requires_connect(self) -> None:
        store, _ = make_graph_store()
        with pytest.raises(MemoryGraphError, match=r"ping\(\) called before connect"):
            await store.ping()

    async def test_qdrant_ping_checks_the_collection(self) -> None:
        store, client = await connected_vector_store()
        await store.ping()
        assert client.called("collection_exists") == [(("memories",), {})]

    async def test_qdrant_ping_maps_failures(self) -> None:
        client = FakeQdrant()
        client.errors["collection_exists"] = unexpected(503)
        store, _ = await connected_vector_store(client)
        with pytest.raises(StoreUnavailableError):
            await store.ping()


class TestScoredSearch:
    async def test_scores_are_returned_best_first_and_deduplicated(self) -> None:
        client = FakeQdrant(points=[{"entity_id": "tool-a"}, {"entity_id": "tool-a"}])
        store, _ = await connected_vector_store(client)
        assert await store.search_scored(VECTOR, 5, 0.0) == [("tool-a", 0.9)]
