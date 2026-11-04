"""FastAPI layer over ASGITransport: endpoints, lifespan and exact RFC 9457 bodies."""

import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

import httpx
import pytest
from fastapi import FastAPI
from pydantic import SecretStr
from starlette.types import Message, Receive, Scope, Send

from agent_memory_graph.adapters import Neo4jGraphStore, QdrantVectorStore
from agent_memory_graph.api import build_memory_service, create_app
from agent_memory_graph.api.middleware import BodySizeLimitMiddleware
from agent_memory_graph.api.schemas import MAX_RAW_INGEST_CHARS
from agent_memory_graph.config import Settings
from agent_memory_graph.domain import (
    EntityType,
    ExtractionFailedError,
    MemoryGraphError,
    StoreUnavailableError,
    VectorSearchTimeoutError,
    make_entity_id,
)
from agent_memory_graph.services import GraphExtractor, MockEmbeddingService
from agent_memory_graph.services.retrieval import GraphMemoryService
from tests.unit.doubles import InMemoryGraphStore, InMemoryVectorStore, ScriptedLLM

PROBLEM = "application/problem+json"
URN = "urn:agent-memory-graph:problem:"
TEXT = "User: the memory service uses Neo4j."
PAYLOAD: dict[str, Any] = {
    "entities": [
        {"name": "memory service", "entity_type": "CONCEPT", "description": "Agent memory."},
        {"name": "Neo4j", "entity_type": "TOOL", "description": "Graph database."},
    ],
    "relations": [{"source": "memory service", "predicate": "uses", "target": "Neo4j"}],
}
MEMORY = make_entity_id("memory service", EntityType.CONCEPT)
NEO4J = make_entity_id("Neo4j", EntityType.TOOL)


@dataclass
class Api:
    client: httpx.AsyncClient
    app: FastAPI
    graph: InMemoryGraphStore
    vectors: InMemoryVectorStore
    llm: ScriptedLLM


@pytest.fixture
async def api() -> AsyncIterator[Api]:
    graph, vectors = InMemoryGraphStore(), InMemoryVectorStore()
    llm = ScriptedLLM(PAYLOAD, PAYLOAD)
    service = GraphMemoryService(
        graph_store=graph,
        vector_store=vectors,
        extractor=GraphExtractor(llm),
        embedder=MockEmbeddingService(32),
    )
    app = create_app(lambda: service, max_body_bytes=256 * 1024)
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
        async with httpx.AsyncClient(transport=transport, base_url="http://memory.test") as client:
            yield Api(client, app, graph, vectors, llm)


def invalid_input(instance: str, *errors: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": f"{URN}invalid-input",
        "title": "The request input is invalid",
        "status": 422,
        "detail": "The request does not match the API contract.",
        "instance": instance,
        "code": "AMG_INVALID_INPUT",
        "context": {"errors": list(errors)},
    }


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------
async def test_lifespan_starts_and_closes_the_service() -> None:
    graph, vectors = InMemoryGraphStore(), InMemoryVectorStore()
    service = GraphMemoryService(
        graph_store=graph,
        vector_store=vectors,
        extractor=GraphExtractor(ScriptedLLM()),
        embedder=MockEmbeddingService(8),
    )
    built: list[GraphMemoryService] = []

    def factory() -> GraphMemoryService:
        built.append(service)
        return service

    app = create_app(factory, ensure_schema=False)
    assert built == []  # nothing is created at import/build time
    async with app.router.lifespan_context(app):
        assert built == [service]
        assert graph.connected
        assert vectors.connected
        assert graph.called("ensure_schema") == []
    assert (graph.closed, vectors.closed) == (1, 1)


def test_build_memory_service_composes_real_adapters_offline() -> None:
    settings = Settings(neo4j_password=SecretStr("pw"), embedding_dim=16)
    service = build_memory_service(settings, llm=ScriptedLLM(), embedder=MockEmbeddingService(16))
    assert isinstance(service._graph, Neo4jGraphStore)
    assert isinstance(service._vectors, QdrantVectorStore)


def test_build_memory_service_rejects_mismatched_embedding_dimensions() -> None:
    settings = Settings(neo4j_password=SecretStr("pw"), embedding_dim=384)
    with pytest.raises(ValueError, match="16-d vectors but AMG_EMBEDDING_DIM is 384"):
        build_memory_service(settings, llm=ScriptedLLM(), embedder=MockEmbeddingService(16))


# ---------------------------------------------------------------------------
# Ingest
# ---------------------------------------------------------------------------
class TestIngestEndpoint:
    async def test_ingest_returns_graph_and_report(self, api: Api) -> None:
        response = await api.client.post("/v1/memory/ingest", json={"text": TEXT})
        assert response.status_code == 200
        assert response.headers["content-type"] == "application/json"
        body = response.json()
        assert [e["id"] for e in body["graph"]["entities"]] == [MEMORY, NEO4J]
        assert body["graph"]["entities"][0]["embedding"] is None
        assert body["graph"]["relations"] == [
            {
                "source_id": MEMORY,
                "predicate": "USES",
                "target_id": NEO4J,
                "weight": 1.0,
                "properties": {},
            }
        ]
        assert body["report"] == {
            "entities": 2,
            "relations": 1,
            "merged_entities": 0,
            "invalid_entities": 0,
            "dangling_relations": 0,
            "invalid_predicates": 0,
            "self_loops": 0,
            "duplicate_relations": 0,
            "dangling_names": [],
        }
        assert set(api.graph.entities) == set(api.vectors.points) == {MEMORY, NEO4J}

    @pytest.mark.parametrize(
        ("payload", "error"),
        [
            ({}, {"loc": ["body", "text"], "msg": "Field required", "type": "missing"}),
            (
                {"text": 42},
                {"loc": ["body", "text"], "msg": "Input should be a valid string",
                 "type": "string_type"},
            ),
            (
                {"text": ""},
                {
                    "loc": ["body", "text"],
                    "msg": "String should have at least 1 character",
                    "type": "string_too_short",
                },
            ),
            (
                {"text": TEXT, "admin": True},
                {"loc": ["body", "admin"], "msg": "Extra inputs are not permitted",
                 "type": "extra_forbidden"},
            ),
        ],
    )  # fmt: skip
    async def test_contract_violations_are_problem_details(
        self, api: Api, payload: dict[str, Any], error: dict[str, Any]
    ) -> None:
        response = await api.client.post("/v1/memory/ingest", json=payload)
        assert response.status_code == 422
        assert response.headers["content-type"] == PROBLEM
        assert response.json() == invalid_input("/v1/memory/ingest", error)
        assert api.llm.calls == []

    async def test_rejected_values_are_not_echoed(self, api: Api) -> None:
        response = await api.client.post(
            "/v1/memory/ingest", json={"text": "x", "api_key": "sk-secret-value"}
        )
        assert response.status_code == 422
        assert "sk-secret-value" not in response.text

    async def test_raw_text_above_the_edge_limit(self, api: Api) -> None:
        response = await api.client.post(
            "/v1/memory/ingest", json={"text": "a" * (MAX_RAW_INGEST_CHARS + 1)}
        )
        assert response.status_code == 422
        (error,) = response.json()["context"]["errors"]
        assert error["type"] == "string_too_long"

    async def test_cleaned_text_above_the_extractor_limit(self, api: Api) -> None:
        response = await api.client.post("/v1/memory/ingest", json={"text": "word " * 7_000})
        assert response.status_code == 422
        assert response.json() == {
            "type": f"{URN}invalid-input",
            "title": "The request input is invalid",
            "status": 422,
            "detail": "Conversation text exceeds 32000 characters; split it into smaller turns.",
            "instance": "/v1/memory/ingest",
            "code": "AMG_INVALID_INPUT",
            "context": {
                "errors": [
                    {
                        "loc": ["text"],
                        "msg": "text exceeds 32000 characters",
                        "type": "string_too_long",
                    }
                ]
            },
        }

    async def test_malformed_json(self, api: Api) -> None:
        response = await api.client.post(
            "/v1/memory/ingest",
            content=b'{"text": "unterminated',
            headers={"content-type": "application/json"},
        )
        assert response.status_code == 422
        (error,) = response.json()["context"]["errors"]
        assert error["type"] == "json_invalid"

    async def test_graph_store_outage_is_503(self, api: Api) -> None:
        api.graph.errors["ingest_graph"] = StoreUnavailableError("neo4j")
        response = await api.client.post("/v1/memory/ingest", json={"text": TEXT})
        assert response.status_code == 503
        assert response.headers["content-type"] == PROBLEM
        assert response.json() == {
            "type": f"{URN}store-unavailable",
            "title": "A backing store is unavailable",
            "status": 503,
            "detail": "The neo4j store is unavailable.",
            "instance": "/v1/memory/ingest",
            "code": "AMG_STORE_UNAVAILABLE",
            "context": {"store": "neo4j"},
        }

    async def test_extraction_failure_is_502(self, api: Api) -> None:
        api.llm._responses[:] = [ExtractionFailedError("schema validation", attempts=3)]
        response = await api.client.post("/v1/memory/ingest", json={"text": TEXT})
        assert response.status_code == 502
        assert response.json()["code"] == "AMG_EXTRACTION_FAILED"
        assert response.json()["context"] == {"reason": "schema validation", "attempts": 3}


# ---------------------------------------------------------------------------
# Query
# ---------------------------------------------------------------------------
class TestQueryEndpoint:
    async def test_query_returns_graph_context(self, api: Api) -> None:
        await api.client.post("/v1/memory/ingest", json={"text": TEXT})
        response = await api.client.post(
            "/v1/memory/query", json={"text": "What does the memory service use?", "top_k": 1}
        )
        assert response.status_code == 200
        body = response.json()
        assert [e["id"] for e in body["seed_entities"]] == [MEMORY]
        assert body["paths"] == [
            {
                "source_id": MEMORY,
                "predicate": "USES",
                "target_id": NEO4J,
                "weight": 1.0,
                "depth": 1,
            }
        ]
        assert body["summary"].startswith("## Relevant memories")
        assert 0.0 < body["confidence"] <= 1.0

    async def test_no_memories_returns_the_empty_context(self, api: Api) -> None:
        response = await api.client.post("/v1/memory/query", json={"text": "anything"})
        assert response.status_code == 200
        assert response.json() == {
            "seed_entities": [],
            "paths": [],
            "summary": "",
            "confidence": 0.0,
        }

    @pytest.mark.parametrize(
        ("payload", "loc", "error_type"),
        [
            ({"text": "q", "traversal_depth": 4}, ["body", "traversal_depth"], "less_than_equal"),
            ({"text": "q", "top_k": 0}, ["body", "top_k"], "greater_than_equal"),
            ({"text": "q", "top_k": "5"}, ["body", "top_k"], "int_type"),
            ({"text": "q", "score_threshold": 2}, ["body", "score_threshold"], "less_than_equal"),
            ({"text": "q", "entity_types": ["ANIMAL"]}, ["body", "entity_types", 0], "enum"),
            ({"text": "\u200b"}, ["body", "text"], "value_error"),
        ],
    )
    async def test_invalid_queries_are_422(
        self, api: Api, payload: dict[str, Any], loc: list[Any], error_type: str
    ) -> None:
        response = await api.client.post("/v1/memory/query", json=payload)
        assert response.status_code == 422
        assert response.headers["content-type"] == PROBLEM
        errors = response.json()["context"]["errors"]
        assert (loc, error_type) in [(error["loc"], error["type"]) for error in errors]

    async def test_vector_search_timeout_is_504(self, api: Api) -> None:
        api.vectors.errors["search_scored"] = VectorSearchTimeoutError(1.0, collection="memories")
        response = await api.client.post("/v1/memory/query", json={"text": "neo4j"})
        assert response.status_code == 504
        assert response.headers["content-type"] == PROBLEM
        assert response.json() == {
            "type": f"{URN}vector-search-timeout",
            "title": "Vector search exceeded its deadline",
            "status": 504,
            "detail": "Vector search on collection 'memories' did not complete within 1s.",
            "instance": "/v1/memory/query",
            "code": "AMG_VECTOR_SEARCH_TIMEOUT",
            "context": {"timeout_seconds": 1.0, "collection": "memories"},
        }

    async def test_explicit_problem_instance_is_preserved(self, api: Api) -> None:
        api.vectors.errors["search_scored"] = MemoryGraphError("boom", instance="urn:trace:42")
        response = await api.client.post("/v1/memory/query", json={"text": "neo4j"})
        assert response.status_code == 500
        assert response.json()["instance"] == "urn:trace:42"


# ---------------------------------------------------------------------------
# Health, framework errors, unexpected errors
# ---------------------------------------------------------------------------
class TestHealthAndFramework:
    async def test_health_ok(self, api: Api) -> None:
        response = await api.client.get("/health")
        assert response.status_code == 200
        assert response.json() == {
            "status": "ok",
            "checks": {"graph_store": "ok", "vector_store": "ok"},
        }

    async def test_health_degraded_is_503(self, api: Api) -> None:
        api.vectors.errors["ping"] = StoreUnavailableError("qdrant")
        response = await api.client.get("/health")
        assert response.status_code == 503
        assert response.json() == {
            "status": "degraded",
            "checks": {"graph_store": "ok", "vector_store": "unavailable"},
        }

    async def test_unknown_route_is_about_blank_problem(self, api: Api) -> None:
        response = await api.client.get("/v1/nope")
        assert response.status_code == 404
        assert response.headers["content-type"] == PROBLEM
        assert response.json() == {
            "type": "about:blank",
            "title": "Not Found",
            "status": 404,
            "detail": "Not Found",
            "instance": "/v1/nope",
        }

    async def test_wrong_method_keeps_allow_header(self, api: Api) -> None:
        response = await api.client.get("/v1/memory/query")
        assert response.status_code == 405
        assert response.headers["allow"] == "POST"
        assert response.json()["title"] == "Method Not Allowed"

    async def test_unexpected_errors_are_generic_500s(
        self, api: Api, caplog: pytest.LogCaptureFixture
    ) -> None:
        api.vectors.errors["search_scored"] = RuntimeError("connection string pw=hunter2")
        with caplog.at_level(logging.ERROR, logger="agent_memory_graph.api.problems"):
            response = await api.client.post("/v1/memory/query", json={"text": "neo4j"})
        assert response.status_code == 500
        assert response.headers["content-type"] == PROBLEM
        assert response.json() == {
            "type": f"{URN}internal-error",
            "title": "Internal memory graph error",
            "status": 500,
            "detail": "An unexpected error occurred.",
            "instance": "/v1/memory/query",
            "code": "AMG_INTERNAL_ERROR",
            "context": {},
        }
        assert "hunter2" not in response.text
        assert "unhandled error on POST /v1/memory/query" in caplog.text

    async def test_server_side_domain_errors_are_logged(
        self, api: Api, caplog: pytest.LogCaptureFixture
    ) -> None:
        api.graph.errors["ingest_graph"] = StoreUnavailableError("neo4j")
        with caplog.at_level(logging.ERROR, logger="agent_memory_graph.api.problems"):
            await api.client.post("/v1/memory/ingest", json={"text": TEXT})
        assert "AMG_STORE_UNAVAILABLE" in caplog.text

    async def test_openapi_documents_problem_responses(self, api: Api) -> None:
        response = await api.client.get("/openapi.json")
        assert response.status_code == 200
        spec = response.json()
        query_responses = spec["paths"]["/v1/memory/query"]["post"]["responses"]
        assert {"200", "422", "500", "502", "503", "504"} <= set(query_responses)
        assert PROBLEM in query_responses["504"]["content"]
        assert set(spec["paths"]) == {"/v1/memory/ingest", "/v1/memory/query", "/health"}


# ---------------------------------------------------------------------------
# Body size limit
# ---------------------------------------------------------------------------
class TestBodySizeLimit:
    async def test_declared_oversized_body_is_refused_before_the_app_runs(self, api: Api) -> None:
        response = await api.client.post("/v1/memory/ingest", json={"text": "x" * 300_000})
        assert response.status_code == 413
        assert response.headers["content-type"] == PROBLEM
        assert response.json() == {
            "type": "about:blank",
            "title": "Content Too Large",
            "status": 413,
            "detail": "Request body exceeds 262144 bytes.",
            "instance": "/v1/memory/ingest",
        }
        assert api.llm.calls == []

    async def test_streamed_oversized_body_is_aborted(self, api: Api) -> None:
        async def chunks() -> AsyncIterator[bytes]:
            yield b'{"text": "'
            for _ in range(400):
                yield b"x" * 1024
            yield b'"}'

        response = await api.client.post(
            "/v1/memory/ingest", content=chunks(), headers={"content-type": "application/json"}
        )
        assert response.status_code == 413
        assert response.json()["detail"] == "Request body exceeds 262144 bytes."
        assert api.llm.calls == []

    async def test_streamed_body_within_the_limit_passes(self, api: Api) -> None:
        async def chunks() -> AsyncIterator[bytes]:
            yield b'{"text": "'
            yield TEXT.encode()
            yield b'"}'

        response = await api.client.post(
            "/v1/memory/ingest", content=chunks(), headers={"content-type": "application/json"}
        )
        assert response.status_code == 200

    async def test_non_numeric_content_length_is_refused(self) -> None:
        sent: list[Message] = []

        async def app(scope: Scope, receive: Receive, send: Send) -> None:
            raise AssertionError("must not be reached")

        async def receive() -> Message:
            return {"type": "http.request", "body": b"", "more_body": False}

        async def send(message: Message) -> None:
            sent.append(message)

        middleware = BodySizeLimitMiddleware(app, max_bytes=10)
        scope = {"type": "http", "path": "/x", "headers": [(b"content-length", b"1e9")]}
        await middleware(scope, receive, send)
        assert sent[0]["status"] == 413

    async def test_non_http_scopes_pass_through(self) -> None:
        seen: list[str] = []

        async def app(scope: Scope, receive: Receive, send: Send) -> None:
            seen.append(scope["type"])

        async def receive() -> Message:
            return {"type": "lifespan.startup"}

        async def send(message: Message) -> None:
            raise AssertionError("must not be reached")

        await BodySizeLimitMiddleware(app, max_bytes=1)({"type": "lifespan"}, receive, send)
        assert seen == ["lifespan"]


async def test_client_disconnect_while_streaming_is_passed_through() -> None:
    messages: list[Message] = []

    async def app(scope: Scope, receive: Receive, send: Send) -> None:
        messages.append(await receive())
        messages.append(await receive())

    queue: list[Message] = [
        {"type": "http.request", "body": b"abc", "more_body": True},
        {"type": "http.disconnect"},
    ]

    async def receive() -> Message:
        return queue.pop(0)

    async def send(message: Message) -> None:
        raise AssertionError("must not be reached")

    await BodySizeLimitMiddleware(app, max_bytes=10)(
        {"type": "http", "path": "/x", "headers": []}, receive, send
    )
    assert [m["type"] for m in messages] == ["http.request", "http.disconnect"]
