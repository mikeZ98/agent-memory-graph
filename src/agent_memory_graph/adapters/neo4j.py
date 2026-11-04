"""Asynchronous Neo4j adapter: idempotent graph ingestion and bounded traversal.

Statement safety
----------------
Every value reaches Neo4j as a bound parameter. Cypher cannot parameterise two
things this adapter needs, so both are spliced in only after a strict check:

* relationship types: re-checked with :func:`is_canonical_predicate` (``[A-Z0-9_]``)
  and back-quoted, so they can never close the pattern or start a new clause;
* the variable-length bound ``*1..N``: an ``int`` in ``1..MAX_TRAVERSAL_DEPTH``.

Neo4j properties cannot hold nested maps, so ``attributes``/``properties`` are stored
as canonical JSON strings (``attributes_json``/``properties_json``).
"""

import asyncio
import json
from collections import defaultdict
from collections.abc import Awaitable, Callable, Iterator, Mapping, Sequence
from types import TracebackType
from typing import Annotated, Any, Final, Literal, Self

import neo4j
from neo4j import AsyncDriver, AsyncGraphDatabase, AsyncManagedTransaction
from neo4j.exceptions import (
    AuthError,
    ConstraintError,
    DriverError,
    Neo4jError,
    ServiceUnavailable,
    SessionExpired,
    TransientError,
)
from pydantic import Field, SecretStr, Strict, TypeAdapter, ValidationError

from agent_memory_graph.config import Settings
from agent_memory_graph.domain._base import thaw_json
from agent_memory_graph.domain.errors import (
    EntityIntegrityError,
    MemoryGraphError,
    StoreTimeoutError,
    StoreUnavailableError,
    TraversalDepthExceededError,
)
from agent_memory_graph.domain.models import (
    Entity,
    EntityId,
    KnowledgeGraph,
    Relation,
    is_canonical_predicate,
)
from agent_memory_graph.domain.query import (
    DEFAULT_TRAVERSAL_DEPTH,
    MAX_CONTEXT_HOPS,
    MAX_TRAVERSAL_DEPTH,
    GraphHop,
)

__all__ = [
    "FETCH_ENTITIES",
    "MERGE_ENTITIES",
    "SCHEMA_STATEMENTS",
    "STORE_NAME",
    "Neo4jGraphStore",
    "merge_relations_statement",
    "traversal_statement",
]

STORE_NAME: Final = "neo4j"
WRITE_BATCH_SIZE: Final = 1_000
MAX_SEED_IDS: Final = 1_000
MAX_FETCH_IDS: Final = MAX_SEED_IDS + 2 * MAX_CONTEXT_HOPS

_TIMEOUT_CODES: Final = frozenset(
    {
        "Neo.ClientError.Transaction.TransactionTimedOut",
        "Neo.ClientError.Transaction.TransactionTimedOutClientConfiguration",
    }
)
_ENTITY_ID: Final = TypeAdapter(EntityId)
_HOP_LIMIT: Final = TypeAdapter(Annotated[int, Strict(), Field(ge=1, le=MAX_CONTEXT_HOPS)])

SCHEMA_STATEMENTS: Final = (
    "CREATE CONSTRAINT amg_entity_id IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE",
)

MERGE_ENTITIES: Final = """
UNWIND $rows AS row
MERGE (e:Entity {id: row.id, type: row.type})
SET e.name = row.name,
    e.description = row.description,
    e.attributes_json = row.attributes_json
RETURN count(e) AS written
"""

_MERGE_RELATIONS_TEMPLATE: Final = """
UNWIND $rows AS row
MATCH (s:Entity {{id: row.source_id}})
MATCH (t:Entity {{id: row.target_id}})
MERGE (s)-[r:`{predicate}`]->(t)
SET r.weight = row.weight,
    r.properties_json = row.properties_json
RETURN count(r) AS written
"""

FETCH_ENTITIES: Final = """
MATCH (e:Entity)
WHERE e.id IN $ids
RETURN e.id AS id,
       e.name AS name,
       e.type AS entity_type,
       coalesce(e.description, '') AS description,
       coalesce(e.attributes_json, '{}') AS attributes_json
"""

_TRAVERSAL_TEMPLATE: Final = """
MATCH path = (seed:Entity)-[*1..{depth}]->(:Entity)
WHERE seed.id IN $seed_ids
UNWIND range(0, length(path) - 1) AS i
WITH relationships(path)[i] AS r, i + 1 AS hop_depth
WITH r, min(hop_depth) AS depth
RETURN startNode(r).id AS source_id,
       type(r) AS predicate,
       endNode(r).id AS target_id,
       r.weight AS weight,
       depth
ORDER BY depth, source_id, predicate, target_id
LIMIT $limit
"""

type DriverFactory = Callable[..., AsyncDriver]
type _Work[T] = Callable[[AsyncManagedTransaction], Awaitable[T]]


# ---------------------------------------------------------------------------
# Statement composition (the only places where text is spliced into Cypher)
# ---------------------------------------------------------------------------
def merge_relations_statement(predicate: str) -> str:
    """Return the batched MERGE statement for relations of one ``predicate``."""
    if not is_canonical_predicate(predicate):
        msg = f"refusing to compose Cypher for non-canonical predicate {predicate!r}"
        raise ValueError(msg)
    return _MERGE_RELATIONS_TEMPLATE.format(predicate=predicate)


def _check_depth(depth: object) -> int:
    if isinstance(depth, bool) or not isinstance(depth, int):
        msg = "traversal depth must be an int"
        raise TypeError(msg)
    if depth > MAX_TRAVERSAL_DEPTH:
        raise TraversalDepthExceededError(depth, MAX_TRAVERSAL_DEPTH)
    if depth < 1:
        msg = "traversal depth must be at least 1"
        raise ValueError(msg)
    return depth


def traversal_statement(depth: int) -> str:
    """Return the outward traversal statement bounded to ``1..depth`` hops."""
    return _TRAVERSAL_TEMPLATE.format(depth=_check_depth(depth))


# ---------------------------------------------------------------------------
# Row mapping
# ---------------------------------------------------------------------------
def _canonical_json(payload: object) -> str:
    return json.dumps(thaw_json(payload), sort_keys=True, separators=(",", ":"))


def _entity_row(entity: Entity) -> dict[str, Any]:
    return {
        "id": entity.id,
        "type": entity.entity_type.value,
        "name": entity.name,
        "description": entity.description,
        "attributes_json": _canonical_json(entity.attributes),
    }


def _relation_row(relation: Relation) -> dict[str, Any]:
    return {
        "source_id": relation.source_id,
        "target_id": relation.target_id,
        "weight": relation.weight,
        "properties_json": _canonical_json(relation.properties),
    }


def _chunks[T](rows: Sequence[T], size: int) -> Iterator[Sequence[T]]:
    for start in range(0, len(rows), size):
        yield rows[start : start + size]


def _normalize_ids(ids: Sequence[str], *, limit: int, what: str = "seed_ids") -> list[str]:
    if isinstance(ids, str):
        msg = f"{what} must be a sequence of ids, not a single string"
        raise TypeError(msg)
    try:
        normalized = list(dict.fromkeys(_ENTITY_ID.validate_python(eid) for eid in ids))
    except ValidationError as exc:
        msg = f"{what} contains an invalid entity id"
        raise ValueError(msg) from exc
    if len(normalized) > limit:
        msg = f"at most {limit} {what} are allowed"
        raise ValueError(msg)
    return normalized


def _entity_from_row(row: Mapping[str, Any]) -> Entity:
    attributes = json.loads(row["attributes_json"])
    return Entity.model_validate(
        {
            "id": row["id"],
            "name": row["name"],
            "entity_type": row["entity_type"],
            "description": row["description"],
            "attributes": attributes,
        }
    )


async def _run_rows(
    tx: AsyncManagedTransaction, statement: str, parameters: dict[str, Any]
) -> list[dict[str, Any]]:
    result = await tx.run(statement, parameters)
    return await result.data()


# ---------------------------------------------------------------------------
# Error translation
# ---------------------------------------------------------------------------
def _translate(  # noqa: PLR0911 - one return per mapped failure class
    exc: Exception, operation: str, timeout_seconds: float
) -> MemoryGraphError:
    context = {"store": STORE_NAME, "operation": operation}
    if isinstance(exc, TimeoutError):
        return StoreTimeoutError(STORE_NAME, operation, timeout_seconds)
    if isinstance(exc, ServiceUnavailable | SessionExpired):
        return StoreUnavailableError(STORE_NAME)
    if isinstance(exc, AuthError):
        return MemoryGraphError("Neo4j rejected the configured credentials.", context=context)
    if isinstance(exc, ConstraintError):
        return EntityIntegrityError("Neo4j rejected the write: an entity id is already in use.")
    if isinstance(exc, Neo4jError):
        code = exc.code or "unknown"
        if code in _TIMEOUT_CODES:
            return StoreTimeoutError(STORE_NAME, operation, timeout_seconds)
        if isinstance(exc, TransientError):
            return StoreUnavailableError(STORE_NAME)
        return MemoryGraphError(
            f"Neo4j {operation} failed.", context={**context, "neo4j_code": code}
        )
    return MemoryGraphError(f"Neo4j driver error during {operation}.", context=context)


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------
class Neo4jGraphStore:
    """Owns one :class:`neo4j.AsyncDriver` (and its connection pool) for its lifetime.

    Use as ``async with Neo4jGraphStore.from_settings(settings) as store: ...`` or call
    :meth:`connect` / :meth:`close` explicitly. Both are idempotent and safe to call
    concurrently.
    """

    def __init__(
        self,
        uri: str,
        *,
        user: str,
        password: SecretStr,
        database: str | None = None,
        timeout_seconds: float = 10.0,
        max_pool_size: int = 50,
        driver_factory: DriverFactory = AsyncGraphDatabase.driver,
    ) -> None:
        self._uri = uri
        self._user = user
        self._password = password
        self._database = database
        self._timeout = timeout_seconds
        self._max_pool_size = max_pool_size
        self._driver_factory = driver_factory
        self._driver: AsyncDriver | None = None
        self._lock = asyncio.Lock()

    @classmethod
    def from_settings(
        cls, settings: Settings, *, driver_factory: DriverFactory = AsyncGraphDatabase.driver
    ) -> Self:
        """Build an adapter from the ``AMG_NEO4J_*`` settings."""
        return cls(
            settings.neo4j_uri,
            user=settings.neo4j_user,
            password=settings.neo4j_password,
            database=settings.neo4j_database,
            timeout_seconds=settings.default_timeout_seconds,
            max_pool_size=settings.neo4j_max_pool_size,
            driver_factory=driver_factory,
        )

    @property
    def is_connected(self) -> bool:
        """Whether a verified driver is currently held."""
        return self._driver is not None

    # -- lifecycle ----------------------------------------------------------
    def _create_driver(self) -> AsyncDriver:
        return self._driver_factory(
            self._uri,
            auth=(self._user, self._password.get_secret_value()),
            max_connection_pool_size=self._max_pool_size,
            connection_timeout=self._timeout,
            connection_acquisition_timeout=self._timeout,
            max_transaction_retry_time=self._timeout,
        )

    async def connect(self) -> None:
        """Create the driver and verify the server is reachable with these credentials."""
        async with self._lock:
            if self._driver is not None:
                return
            driver = self._create_driver()
            try:
                async with asyncio.timeout(self._timeout):
                    await driver.verify_connectivity()
            except (TimeoutError, Neo4jError, DriverError) as exc:
                await driver.close()
                raise _translate(exc, "connect", self._timeout) from exc
            self._driver = driver

    async def close(self) -> None:
        """Close the driver and its pool; a no-op when not connected."""
        async with self._lock:
            driver, self._driver = self._driver, None
            if driver is not None:
                await driver.close()

    async def ping(self) -> None:
        """Check that the server answers with the configured credentials."""
        driver = self._driver
        if driver is None:
            msg = "Neo4jGraphStore.ping() called before connect()"
            raise MemoryGraphError(msg, context={"store": STORE_NAME})
        try:
            async with asyncio.timeout(self._timeout):
                await driver.verify_connectivity()
        except (TimeoutError, Neo4jError, DriverError) as exc:
            raise _translate(exc, "ping", self._timeout) from exc

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
    async def _execute[T](
        self, mode: Literal["read", "write"], operation: str, work: _Work[T]
    ) -> T:
        driver = self._driver
        if driver is None:
            msg = f"Neo4jGraphStore.{operation}() called before connect()"
            raise MemoryGraphError(msg, context={"store": STORE_NAME})
        transaction_fn = neo4j.unit_of_work(timeout=self._timeout)(work)
        try:
            async with (
                asyncio.timeout(self._timeout),
                driver.session(database=self._database) as session,
            ):
                if mode == "write":
                    return await session.execute_write(transaction_fn)
                return await session.execute_read(transaction_fn)
        except (TimeoutError, Neo4jError, DriverError) as exc:
            raise _translate(exc, operation, self._timeout) from exc

    # -- operations ---------------------------------------------------------
    async def ensure_schema(self) -> None:
        """Create the ``Entity.id`` uniqueness constraint (idempotent)."""

        async def work(tx: AsyncManagedTransaction) -> None:
            for statement in SCHEMA_STATEMENTS:
                await _run_rows(tx, statement, {})

        await self._execute("write", "ensure_schema", work)

    async def ingest_graph(self, graph: KnowledgeGraph) -> None:
        """Merge every entity and relation of ``graph`` in one atomic transaction.

        Entities merge on ``(id, type)``, relations on ``(source)-[PREDICATE]->(target)``;
        re-ingesting the same graph is a no-op apart from refreshed properties.
        """
        entity_rows = [_entity_row(entity) for entity in graph.entities]
        by_predicate: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
        for relation in graph.relations:
            by_predicate[relation.predicate].append(_relation_row(relation))
        statements = {pred: merge_relations_statement(pred) for pred in sorted(by_predicate)}
        if not entity_rows:
            return

        async def work(tx: AsyncManagedTransaction) -> None:
            for chunk in _chunks(entity_rows, WRITE_BATCH_SIZE):
                await _run_rows(tx, MERGE_ENTITIES, {"rows": list(chunk)})
            for predicate, statement in statements.items():
                for chunk in _chunks(by_predicate[predicate], WRITE_BATCH_SIZE):
                    rows = await _run_rows(tx, statement, {"rows": list(chunk)})
                    written = rows[0]["written"] if rows else 0
                    if written != len(chunk):
                        # Only possible when stored ids are ambiguous (no uniqueness
                        # constraint) or missing; abort so the transaction rolls back.
                        msg = (
                            f"{predicate}: merged {written} of {len(chunk)} relations; "
                            "run ensure_schema() and check for duplicate entity ids"
                        )
                        raise EntityIntegrityError(msg)

        await self._execute("write", "ingest_graph", work)

    async def traverse_neighborhood(
        self,
        seed_ids: Sequence[str],
        depth: int = DEFAULT_TRAVERSAL_DEPTH,
        *,
        limit: int = MAX_CONTEXT_HOPS,
    ) -> tuple[GraphHop, ...]:
        """Return the relations reachable from ``seed_ids`` within ``depth`` outward hops.

        Each relation appears once, at the shortest depth it was reached, ordered by
        ``(depth, source_id, predicate, target_id)`` and capped at ``limit`` hops.
        """
        statement = traversal_statement(depth)
        limit = _HOP_LIMIT.validate_python(limit)
        ids = _normalize_ids(seed_ids, limit=MAX_SEED_IDS)
        if not ids:
            return ()

        async def work(tx: AsyncManagedTransaction) -> list[dict[str, Any]]:
            return await _run_rows(tx, statement, {"seed_ids": ids, "limit": limit})

        rows = await self._execute("read", "traverse_neighborhood", work)
        try:
            return tuple(GraphHop.model_validate(row) for row in rows)
        except ValidationError as exc:
            msg = "Neo4j returned a relation that violates the domain contract."
            raise MemoryGraphError(msg, context={"store": STORE_NAME}) from exc

    async def fetch_entities(self, entity_ids: Sequence[str]) -> tuple[Entity, ...]:
        """Return the stored entities among ``entity_ids``, in the order requested.

        Unknown ids are skipped. Embeddings live in the vector store and are not
        included. A stored node that violates the domain contract fails closed.
        """
        ids = _normalize_ids(entity_ids, limit=MAX_FETCH_IDS, what="entity_ids")
        if not ids:
            return ()

        async def work(tx: AsyncManagedTransaction) -> list[dict[str, Any]]:
            return await _run_rows(tx, FETCH_ENTITIES, {"ids": ids})

        rows = await self._execute("read", "fetch_entities", work)
        try:
            found = {entity.id: entity for entity in map(_entity_from_row, rows)}
        except (ValueError, TypeError, KeyError) as exc:  # ValidationError, JSONDecodeError
            msg = "Neo4j returned an entity that violates the domain contract."
            raise MemoryGraphError(msg, context={"store": STORE_NAME}) from exc
        return tuple(found[eid] for eid in ids if eid in found)
