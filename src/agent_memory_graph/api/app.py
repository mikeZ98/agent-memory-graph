"""FastAPI application: lifespan-managed stores, memory endpoints, RFC 9457 errors."""

from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import Annotated, Any, cast

from fastapi import APIRouter, Depends, FastAPI, Request
from fastapi.responses import JSONResponse

from agent_memory_graph import __version__
from agent_memory_graph.adapters.neo4j import Neo4jGraphStore
from agent_memory_graph.adapters.qdrant import QdrantVectorStore
from agent_memory_graph.api.middleware import DEFAULT_MAX_BODY_BYTES, BodySizeLimitMiddleware
from agent_memory_graph.api.problems import PROBLEM_RESPONSES, install_problem_handlers
from agent_memory_graph.api.schemas import (
    HealthChecks,
    HealthResponse,
    IngestRequest,
    IngestResponse,
)
from agent_memory_graph.config import Settings
from agent_memory_graph.domain.errors import ExtractionTimeoutError, StoreTimeoutError
from agent_memory_graph.domain.query import GraphContext, MemoryQuery
from agent_memory_graph.guardrails.graph_guard import InputGuard
from agent_memory_graph.guardrails.limiter import CircuitBreaker, ConcurrencyLimiter
from agent_memory_graph.guardrails.stores import GuardedGraphStore, GuardedLLM, GuardedVectorStore
from agent_memory_graph.guardrails.telemetry import Telemetry
from agent_memory_graph.services.embedder import EmbeddingService
from agent_memory_graph.services.extractor import GraphExtractor, StructuredLLM
from agent_memory_graph.services.retrieval import GraphMemoryService, RetrievalPolicy

__all__ = ["build_memory_service", "create_app", "router"]

type ServiceFactory = Callable[[], GraphMemoryService]

_SERVICE_UNAVAILABLE = 503


def _store_guards(
    settings: Settings, store: str, telemetry: Telemetry
) -> tuple[CircuitBreaker, ConcurrencyLimiter]:
    breaker = CircuitBreaker(
        store,
        failure_threshold=settings.breaker_failure_threshold,
        reset_timeout_seconds=settings.breaker_reset_seconds,
    )
    limiter = ConcurrencyLimiter(
        store,
        max_concurrency=settings.max_concurrent_store_operations,
        queue_timeout_seconds=settings.queue_timeout_seconds,
        on_queue_timeout=lambda waited: StoreTimeoutError(store, "queue", waited),
    )
    telemetry.register_breaker(breaker)
    telemetry.register_limiter(limiter)
    return breaker, limiter


def build_memory_service(
    settings: Settings, *, llm: StructuredLLM, embedder: EmbeddingService
) -> GraphMemoryService:
    """Compose the production service: real adapters behind breakers and queues.

    * each store gets a circuit breaker and a bounded queue (``AMG_STORE_TIMEOUT``
      after ``AMG_QUEUE_TIMEOUT_SECONDS``);
    * extraction calls share a smaller queue (``AMG_EXTRACTION_TIMEOUT`` on expiry);
    * every breaker and queue reports into the service's telemetry (``GET /metrics``).
    """
    if embedder.dimension != settings.embedding_dim:
        msg = (
            f"embedder produces {embedder.dimension}-d vectors but AMG_EMBEDDING_DIM is "
            f"{settings.embedding_dim}"
        )
        raise ValueError(msg)
    telemetry = Telemetry()
    graph_breaker, graph_limiter = _store_guards(settings, "neo4j", telemetry)
    vector_breaker, vector_limiter = _store_guards(settings, "qdrant", telemetry)
    llm_limiter = ConcurrencyLimiter(
        "extraction",
        max_concurrency=settings.max_concurrent_extractions,
        queue_timeout_seconds=settings.queue_timeout_seconds,
        on_queue_timeout=ExtractionTimeoutError,
    )
    telemetry.register_limiter(llm_limiter)
    return GraphMemoryService(
        graph_store=GuardedGraphStore(
            Neo4jGraphStore.from_settings(settings), breaker=graph_breaker, limiter=graph_limiter
        ),
        vector_store=GuardedVectorStore(
            QdrantVectorStore.from_settings(settings),
            breaker=vector_breaker,
            limiter=vector_limiter,
        ),
        extractor=GraphExtractor(GuardedLLM(llm, limiter=llm_limiter)),
        embedder=embedder,
        telemetry=telemetry,
        input_guard=InputGuard(mode=settings.input_guard_mode),
        policy=RetrievalPolicy(
            fallback_floor=settings.retrieval_fallback_floor,
            max_fallback_steps=settings.retrieval_fallback_steps,
        ),
    )


def get_memory_service(request: Request) -> GraphMemoryService:
    """Dependency: the service the lifespan started for this application."""
    return cast("GraphMemoryService", request.app.state.memory_service)


MemoryService = Annotated[GraphMemoryService, Depends(get_memory_service)]

router = APIRouter()


@router.post(
    "/v1/memory/ingest",
    response_model=IngestResponse,
    responses=PROBLEM_RESPONSES,
    summary="Extract a conversation into the memory graph",
)
async def ingest(body: IngestRequest, service: MemoryService) -> IngestResponse:
    """Extract entities and relations from ``text`` and store them idempotently."""
    graph, report = await service.ingest(body.text)
    return IngestResponse(graph=graph, report=report)


@router.post(
    "/v1/memory/query",
    response_model=GraphContext,
    responses=PROBLEM_RESPONSES,
    summary="Retrieve graph context for a question",
)
async def query(body: MemoryQuery, service: MemoryService) -> GraphContext:
    """Return seed entities, their graph neighbourhood and a markdown summary."""
    return await service.query(body)


@router.get(
    "/health",
    response_model=HealthResponse,
    responses={_SERVICE_UNAVAILABLE: {"model": HealthResponse, "description": "Degraded"}},
    summary="Connectivity of the graph and vector stores",
)
async def health(service: MemoryService) -> JSONResponse:
    """``200`` when both stores answer, ``503`` (same body shape) otherwise."""
    report = await service.health()
    body = HealthResponse(
        status="ok" if report.ok else "degraded",
        checks=HealthChecks(graph_store=report.graph_store, vector_store=report.vector_store),
    )
    return JSONResponse(
        body.model_dump(mode="json"), status_code=200 if report.ok else _SERVICE_UNAVAILABLE
    )


@router.get("/metrics", summary="Counters, latencies, queue occupancy and circuit states")
async def metrics(service: MemoryService) -> dict[str, Any]:
    """In-process telemetry snapshot (JSON)."""
    return service.telemetry.snapshot()


def create_app(
    service_factory: ServiceFactory,
    *,
    ensure_schema: bool = True,
    max_body_bytes: int = DEFAULT_MAX_BODY_BYTES,
) -> FastAPI:
    """Build the ASGI app; ``service_factory`` is called once per lifespan.

    The lifespan starts the service (connection pools, idempotent schema setup) before
    the first request and closes it on shutdown.
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        service = service_factory()
        await service.start(ensure_schema=ensure_schema)
        app.state.memory_service = service
        try:
            yield
        finally:
            await service.close()

    app = FastAPI(
        title="agent-memory-graph",
        version=__version__,
        summary="Asynchronous GraphRAG long-term memory for AI agents.",
        lifespan=lifespan,
    )
    install_problem_handlers(app)
    app.add_middleware(BodySizeLimitMiddleware, max_bytes=max_body_bytes)
    app.include_router(router)
    return app
