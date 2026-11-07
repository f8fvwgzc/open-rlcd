"""Connectors: vector and graph stores the documents pack retrieves from.

`base` defines the protocols (`VectorStore`, `GraphStore`, `Hit`) and the error convention (`ConnectorError`,
turned into a 503 by `unavailable`); `security.check_host` keeps every connector on loopback unless remote hosts
are explicitly allowed. Local implementations: `local_vector.LocalVectorStore` (faiss -> hnswlib -> numpy) and
`local_graph.LocalGraphStore` (networkx or a dict fallback). Remote ones (Qdrant, pgvector, Neo4j) are not part
of this package yet.
"""
from .base import ConnectorError, GraphStore, Hit, VectorStore, unavailable
from .security import check_host, remote_allowed

__all__ = ["ConnectorError", "GraphStore", "Hit", "VectorStore", "check_host", "remote_allowed", "unavailable"]
