"""Application services: knowledge extraction and embedding.

Services depend on the domain and on narrow ports (:class:`StructuredLLM`,
:class:`EmbeddingService`); concrete providers are chosen at the composition root.
"""

from agent_memory_graph.services.embedder import (
    EmbeddingService,
    MockEmbeddingService,
    entity_embedding_text,
)
from agent_memory_graph.services.extractor import (
    ExtractionReport,
    GraphExtraction,
    GraphExtractor,
    InstructorStructuredLLM,
    StructuredLLM,
)

__all__ = [
    "EmbeddingService",
    "ExtractionReport",
    "GraphExtraction",
    "GraphExtractor",
    "InstructorStructuredLLM",
    "MockEmbeddingService",
    "StructuredLLM",
    "entity_embedding_text",
]
