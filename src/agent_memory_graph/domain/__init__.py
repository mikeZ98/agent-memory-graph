"""Pure domain layer: immutable graph schemas, retrieval contracts and error taxonomy.

Nothing here performs I/O or imports a driver; adapters depend on this package,
never the other way round.
"""

from agent_memory_graph.domain.errors import (
    PROBLEM_JSON_MEDIA_TYPE,
    PROBLEM_TAXONOMY,
    EntityIntegrityError,
    ErrorCode,
    MemoryGraphError,
    MemoryProblemDetails,
    TraversalDepthExceededError,
    VectorSearchTimeoutError,
)
from agent_memory_graph.domain.models import (
    Entity,
    EntityType,
    KnowledgeGraph,
    Relation,
    make_entity_id,
)
from agent_memory_graph.domain.query import GraphContext, GraphHop, MemoryQuery

__all__ = [
    "PROBLEM_JSON_MEDIA_TYPE",
    "PROBLEM_TAXONOMY",
    "Entity",
    "EntityIntegrityError",
    "EntityType",
    "ErrorCode",
    "GraphContext",
    "GraphHop",
    "KnowledgeGraph",
    "MemoryGraphError",
    "MemoryProblemDetails",
    "MemoryQuery",
    "Relation",
    "TraversalDepthExceededError",
    "VectorSearchTimeoutError",
    "make_entity_id",
]
