"""Build the remote stores the documents pack's config names (`pack_config.vector`, `pack_config.graph`).

    vector: {kind: local | qdrant | pgvector, mode: index | attach, ...}
    graph:  {kind: local | neo4j, mode: index | attach, ...}

Only configuration is checked here (keys, credentials-in-config, hosts, templates): building a store opens no
connection. Each store connects on first use, or when `connect()` is called. The store modules import their client
libraries only then, so this module (and the whole package) imports without the optional extras.
"""
from __future__ import annotations

from typing import Any

from .base import config_error

VECTOR_KINDS = ("local", "qdrant", "pgvector")
GRAPH_KINDS = ("local", "neo4j")
REMOTE_VECTOR_KINDS = ("qdrant", "pgvector")
REMOTE_GRAPH_KINDS = ("neo4j",)


def kind_of(cfg: dict | None, what: str) -> str:
    kind = str((cfg or {}).get("kind") or "local").lower()
    kinds = VECTOR_KINDS if what == "vector" else GRAPH_KINDS
    if kind not in kinds:
        raise config_error(f"unknown {what} connector kind {kind!r}; use one of {list(kinds)}")
    return kind


def mode_of(cfg: dict | None, what: str) -> str:
    """'local' for a local store, else the configured mode (default 'index')."""
    if kind_of(cfg, what) == "local":
        return "local"
    mode = str((cfg or {}).get("mode") or "index").lower()
    if mode not in ("index", "attach"):
        raise config_error(f"pack_config.{what}.mode must be 'index' or 'attach', got {mode!r}")
    return mode


def make_vector_store(cfg: dict, dim: int, allow_remote: bool = False) -> Any:
    """A remote vector store from `pack_config.vector` (kind qdrant or pgvector)."""
    kind = kind_of(cfg, "vector")
    if kind == "qdrant":
        from .qdrant import QdrantVectorStore

        return QdrantVectorStore.from_config(cfg, dim, allow_remote=allow_remote)
    if kind == "pgvector":
        from .pgvector import PgVectorStore

        return PgVectorStore.from_config(cfg, dim, allow_remote=allow_remote)
    raise config_error(f"vector kind {kind!r} is not a remote store")


def make_graph_store(cfg: dict, allow_remote: bool = False, terms: list[str] | None = None) -> Any:
    """A remote graph store from `pack_config.graph` (kind neo4j)."""
    kind = kind_of(cfg, "graph")
    if kind == "neo4j":
        from .neo4j import Neo4jGraphStore

        return Neo4jGraphStore.from_config(cfg, allow_remote=allow_remote, terms=terms)
    raise config_error(f"graph kind {kind!r} is not a remote store")
