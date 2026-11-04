"""Storage adapters: the only modules that talk to Neo4j and Qdrant.

Every driver failure leaves this package as a :class:`~agent_memory_graph.domain.errors.
MemoryGraphError` subclass, so callers never handle driver-specific exceptions.
"""

from agent_memory_graph.adapters.neo4j import Neo4jGraphStore
from agent_memory_graph.adapters.qdrant import QdrantVectorStore

__all__ = ["Neo4jGraphStore", "QdrantVectorStore"]
