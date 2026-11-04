"""FastAPI application: lifespan-managed stores, memory endpoints, RFC 9457 errors."""

from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import Annotated, cast

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
from agent_memory_graph.domain.query import GraphContext, MemoryQuery
from agent_memory_graph.services.embedder import EmbeddingService
from agent_memory_graph.services.extractor import GraphExtractor, StructuredLLM
from agent_memory_graph.services.retrieval import GraphMemoryService

__all__ = ["build_memory_service", "create_app", "router"]

type ServiceFactory = Callable[[], GraphMemoryService]

_SERVICE_UNAVAILABLE = 503


def build_memory_service(
    settings: Settings, *, llm: StructuredLLM, embedder: EmbeddingService
) -> GraphMemoryService:
    """Compose the production service from settings and the chosen model providers."""
    if embedder.dimension != settings.embedding_dim:
        msg = (
            f"embedder produces {embedder.dimension}-d vectors but AMG_EMBEDDING_DIM is "
            f"{settings.embedding_dim}"
        )
        raise ValueError(msg)
    return GraphMemoryService(
        graph_store=Neo4jGraphStore.from_settings(settings),
        vector_store=QdrantVectorStore.from_settings(settings),
        extractor=GraphExtractor(llm),
        embedder=embedder,
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
