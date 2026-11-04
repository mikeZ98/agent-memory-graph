"""HTTP request and response bodies (domain models are reused wherever they fit)."""

from typing import Annotated, Final, Literal

from pydantic import Field, Strict

from agent_memory_graph.domain._base import StrictModel
from agent_memory_graph.domain.models import KnowledgeGraph
from agent_memory_graph.services.extractor import DEFAULT_MAX_INPUT_CHARS, ExtractionReport
from agent_memory_graph.services.retrieval import CheckStatus

__all__ = ["MAX_RAW_INGEST_CHARS", "HealthResponse", "IngestRequest", "IngestResponse"]

MAX_RAW_INGEST_CHARS: Final = DEFAULT_MAX_INPUT_CHARS * 4
"""Raw-text ceiling at the edge; the extractor enforces the cleaned-text limit."""


class IngestRequest(StrictModel):
    """A conversation excerpt to remember."""

    text: Annotated[str, Strict(), Field(min_length=1, max_length=MAX_RAW_INGEST_CHARS)]


class IngestResponse(StrictModel):
    """What was stored, and what the extractor pruned on the way."""

    graph: KnowledgeGraph
    report: ExtractionReport


class HealthChecks(StrictModel):
    """Per-store connectivity."""

    graph_store: CheckStatus
    vector_store: CheckStatus


class HealthResponse(StrictModel):
    """``ok`` when every store answered, ``degraded`` otherwise (served with 503)."""

    status: Literal["ok", "degraded"]
    checks: HealthChecks
