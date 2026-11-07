"""Connector protocols: what the documents pack needs from a vector store and from a graph store.

Local implementations live next to this file (`local_vector.LocalVectorStore`, `local_graph.LocalGraphStore`);
remote ones (Qdrant, pgvector, Neo4j) implement the same protocols. A connector that cannot be reached raises
`ConnectorError`; the API boundary turns it into a 503 with `unavailable(...)`, so a dead database is reported
as "service unavailable", never as a bad request or a silent empty result.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np

from ..core.contract import ContractError


@dataclass
class Hit:
    """One vector search result: the stored id, its cosine similarity to the query, and its payload."""

    id: str
    score: float
    payload: dict = field(default_factory=dict)


@runtime_checkable
class VectorStore(Protocol):
    def search(self, vector: np.ndarray, k: int = 10, filter: dict | None = None) -> list[Hit]: ...

    def upsert(self, ids: list[str], vectors: np.ndarray, payloads: list[dict]) -> None: ...

    def count(self) -> int: ...


@runtime_checkable
class GraphStore(Protocol):
    def expand(self, terms: list[str], hops: int = 2, limit: int = 50) -> list[dict]:
        """Chunks reachable from the entities named by `terms`: [{id (chunk id), path_length, degree, via}]."""
        ...

    def neighbors(self, ids: list[str], hops: int = 1) -> list[dict]: ...


class ConnectorError(Exception):
    """A store is unreachable, misconfigured or missing its client library."""


def unavailable(msg: str, **details: object) -> ContractError:
    """The API-boundary form of a connector failure: HTTP 503, kind `connector_unavailable`."""
    return ContractError(msg, kind="connector_unavailable", status=503, **details)
