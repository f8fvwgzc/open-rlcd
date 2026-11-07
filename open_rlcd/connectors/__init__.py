"""Connectors: vector and graph stores the documents pack retrieves from.

`base` defines the protocols (`VectorStore`, `GraphStore`, `Hit`) and the error convention (a store that cannot be
reached is a 503 `connector_unavailable`); `security.check_host` keeps every connector on loopback unless remote
hosts are explicitly allowed. Local implementations: `local_vector.LocalVectorStore` (faiss -> hnswlib -> numpy) and
`local_graph.LocalGraphStore` (networkx or a dict fallback). Remote ones, each behind an optional extra and imported
only when used: `qdrant.QdrantVectorStore` (extra `qdrant`), `pgvector.PgVectorStore` (extra `pgvector`, psycopg 3)
and `neo4j.Neo4jGraphStore` (extra `neo4j`), built from config by `factory`.
"""
from .base import ConnectorError, GraphStore, Hit, VectorStore, unavailable
from .security import check_host, remote_allowed

__all__ = ["ConnectorError", "GraphStore", "Hit", "VectorStore", "check_host", "remote_allowed", "unavailable"]
