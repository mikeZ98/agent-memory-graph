"""Hermetic application for the e2e harness.

Everything above the storage and model ports is production code: the FastAPI app,
RFC 9457 handlers, body limit, ``GraphMemoryService``, guarded store decorators,
circuit breakers, queues and telemetry. Below the ports, stores are in memory, the
embedder is the deterministic mock, and the LLM is a small rule-based extractor, so
the subprocess needs no network and no external service.
"""

import re
import sys
from collections.abc import Sequence
from typing import Any, Final

from fastapi import FastAPI
from pydantic import BaseModel

from agent_memory_graph.api import create_app
from agent_memory_graph.domain.errors import ExtractionTimeoutError, StoreTimeoutError
from agent_memory_graph.guardrails.limiter import CircuitBreaker, ConcurrencyLimiter
from agent_memory_graph.guardrails.stores import GuardedGraphStore, GuardedLLM, GuardedVectorStore
from agent_memory_graph.guardrails.telemetry import Telemetry
from agent_memory_graph.services.embedder import MockEmbeddingService
from agent_memory_graph.services.extractor import ChatMessage, GraphExtractor
from agent_memory_graph.services.retrieval import GraphMemoryService
from tests.unit.doubles import InMemoryGraphStore, InMemoryVectorStore

__all__ = ["CLOSED_MARKER", "DEV_MODE_MARKER", "RuleBasedLLM", "create_e2e_app"]

CLOSED_MARKER: Final = "amg-e2e: stores closed"
DEV_MODE_MARKER: Final = "amg-e2e: dev_mode="
EMBEDDING_DIM: Final = 64

# Capitalised words; a dot only *inside* a word ("Node.js"), never the sentence's full stop.
_WORD = r"[A-Z][\w+-]*(?:\.[\w+-]+)*"
_PROPER_NOUN = rf"{_WORD}(?: {_WORD})*"
_FACT_RE: Final = re.compile(
    rf"\b({_PROPER_NOUN}) (uses|depends on|runs on|works at|lives in) ({_PROPER_NOUN})"
)
_TYPES: Final = {
    "works at": ("PERSON", "ORGANIZATION"),
    "lives in": ("PERSON", "LOCATION"),
}


class RuleBasedLLM:
    """Extracts ``<Proper Noun> <verb> <Proper Noun>`` facts; stands in for a real model."""

    async def create[T: BaseModel](
        self, *, response_model: type[T], messages: Sequence[ChatMessage], max_retries: int
    ) -> T:
        """Return the facts in the last message as a ``response_model`` instance."""
        entities: dict[str, dict[str, Any]] = {}
        relations = []
        for source, verb, target in _FACT_RE.findall(messages[-1]["content"]):
            source_type, target_type = _TYPES.get(verb, ("TOOL", "TOOL"))
            entities.setdefault(source, {"name": source, "entity_type": source_type})
            entities.setdefault(target, {"name": target, "entity_type": target_type})
            relations.append({"source": source, "predicate": verb, "target": target})
        return response_model.model_validate(
            {"entities": list(entities.values()), "relations": relations}
        )


class _ReportingService(GraphMemoryService):
    async def close(self) -> None:
        await super().close()
        print(CLOSED_MARKER, flush=True)  # noqa: T201 - the harness asserts on it


def _guards(store: str, telemetry: Telemetry) -> tuple[CircuitBreaker, ConcurrencyLimiter]:
    breaker = CircuitBreaker(store, failure_threshold=3, reset_timeout_seconds=5)
    limiter = ConcurrencyLimiter(
        store,
        max_concurrency=8,
        queue_timeout_seconds=2.0,
        on_queue_timeout=lambda waited: StoreTimeoutError(store, "queue", waited),
    )
    telemetry.register_breaker(breaker)
    telemetry.register_limiter(limiter)
    return breaker, limiter


def _build_service() -> GraphMemoryService:
    telemetry = Telemetry()
    graph_breaker, graph_limiter = _guards("neo4j", telemetry)
    vector_breaker, vector_limiter = _guards("qdrant", telemetry)
    llm_limiter = ConcurrencyLimiter(
        "extraction",
        max_concurrency=2,
        queue_timeout_seconds=2.0,
        on_queue_timeout=ExtractionTimeoutError,
    )
    telemetry.register_limiter(llm_limiter)
    return _ReportingService(
        graph_store=GuardedGraphStore(
            InMemoryGraphStore(), breaker=graph_breaker, limiter=graph_limiter
        ),
        vector_store=GuardedVectorStore(
            InMemoryVectorStore(), breaker=vector_breaker, limiter=vector_limiter
        ),
        extractor=GraphExtractor(GuardedLLM(RuleBasedLLM(), limiter=llm_limiter)),
        embedder=MockEmbeddingService(EMBEDDING_DIM),
        telemetry=telemetry,
    )


def create_e2e_app() -> FastAPI:
    """Uvicorn factory (``python -m agent_memory_graph --factory tests.e2e.app:create_e2e_app``)."""
    print(f"{DEV_MODE_MARKER}{sys.flags.dev_mode}", flush=True)  # noqa: T201
    return create_app(_build_service)
