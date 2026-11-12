"""Build the remote stores the documents pack's config names (`pack_config.vector`, `pack_config.graph`).

    vector: {kind: local | qdrant | pgvector, mode: index | attach, ...}
    graph:  {kind: local | neo4j, mode: index | attach, ...}

Only configuration is checked here (keys, credentials-in-config, hosts, templates): building a store opens no
connection. Each store connects on first use, or when `connect()` is called. The store modules import their client
libraries only then, so this module (and the whole package) imports without the optional extras.

`check_store` runs the same checks offline for `rlcd config check` (no client library, no environment variable,
any host), and `store_keys` says which keys each kind reads.
"""
from __future__ import annotations

import importlib
from typing import Any

from .base import config_error

VECTOR_KINDS = ("local", "qdrant", "pgvector")
GRAPH_KINDS = ("local", "neo4j")
KINDS = {"vector": VECTOR_KINDS, "graph": GRAPH_KINDS}
REMOTE_VECTOR_KINDS = ("qdrant", "pgvector")
REMOTE_GRAPH_KINDS = ("neo4j",)
MODES = ("index", "attach")
# the keys the local stores read (packs/documents/corpus.py `Corpus`); a remote store's are its module's `KEYS`
LOCAL_KEYS = {"vector": frozenset({"kind", "backend"}),
              "graph": frozenset({"kind", "terms", "max_fanout", "hub_share", "hub_min"})}


def kind_of(cfg: dict | None, what: str) -> str:
    kind = str((cfg or {}).get("kind") or "local").lower()
    kinds = KINDS[what]
    if kind not in kinds:
        raise config_error(f"unknown {what} connector kind {kind!r}; use one of {list(kinds)}")
    return kind


def mode_of(cfg: dict | None, what: str) -> str:
    """'local' for a local store, else the configured mode (default 'index')."""
    if kind_of(cfg, what) == "local":
        return "local"
    mode = str((cfg or {}).get("mode") or "index").lower()
    if mode not in MODES:
        raise config_error(f"pack_config.{what}.mode must be 'index' or 'attach', got {mode!r}")
    return mode


def store_keys(what: str, kind: str) -> frozenset[str]:
    """The `pack_config.<what>` keys the store of `kind` reads: the local store's, or the remote store module's
    `KEYS` (importing it imports no client library). Other keys are ignored by a local store and refused by a
    remote one."""
    kind = kind_of({"kind": kind}, what)
    if kind == "local":
        return LOCAL_KEYS[what]
    return frozenset(importlib.import_module(f"{__package__}.{kind}").KEYS)


def check_store(cfg: dict, what: str) -> None:
    """What building the store `cfg` names would refuse, checked offline for `rlcd config check`: no client library
    is imported, no environment variable read, no connection opened, and any host passes (whether a remote host is
    allowed depends on the environment the store is built in, which checks it then). Raises the store's
    ContractError (`connector_config`); a local store has nothing to check here."""
    kind = kind_of(cfg, what)
    mode_of(cfg, what)
    if kind == "qdrant":
        from .qdrant import QdrantVectorStore

        QdrantVectorStore.check_config(cfg)
    elif kind == "pgvector":
        from .pgvector import PgVectorStore

        PgVectorStore.check_config(cfg)
    elif kind == "neo4j":
        from .neo4j import Neo4jGraphStore

        Neo4jGraphStore.check_config(cfg)


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
