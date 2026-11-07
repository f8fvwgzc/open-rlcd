"""Neo4j as the documents pack's entity graph (optional extra `neo4j`: `uv sync --extra neo4j`).

    pack_config:
      graph: {kind: neo4j, uri: bolt://localhost:7687, user: neo4j, password_env: NEO4J_PASSWORD}   # mode: index
      graph: {kind: neo4j, mode: attach, uri: bolt://localhost:7687, password_env: NEO4J_PASSWORD,
              database: kb, expand_cypher: "<your read-only Cypher>"}

Two modes:

* **index**: open-rlcd writes its own entity graph, the same one the local graph holds: `(:RlcdDoc {id})
  -[:HAS_CHUNK]->(:RlcdChunk {id, doc_id})-[:MENTIONS]->(:RlcdEntity {key, label})` and `(:RlcdDoc)-[:LINKS]->
  (:RlcdDoc)`, with uniqueness constraints on the ids / keys. Entities come from the same surface patterns as
  the local graph (`local_graph.entities`, plus the configured `terms`). Re-ingesting a document replaces its
  chunks and the links it declared; entities nobody mentions any more are deleted. The labels are prefixed
  (`Rlcd...`) so they never mix with a graph that already lives in the same database.
* **attach**: read-only expansion over a graph that already exists, through a Cypher template you configure
  (`expand_cypher`, optionally `neighbors_cypher` and `degree_cypher`). Templates run verbatim in read
  transactions (`execute_read` on a READ session: the server refuses writes) and are refused at configuration time
  when they contain a write clause (CREATE, MERGE, SET, DELETE, REMOVE, DROP, FOREACH, LOAD CSV). Nothing is ever
  string-formatted into a query: the query's terms reach it only as parameters.

Parameters every template receives: `$terms` (entity names found in the query, as written), `$keys` (the same,
normalised: lower case, single spaces, possessives stripped, as `local_graph.normalize_entity`), `$hops` (1-3),
`$max_edges` (2 x hops), `$limit` and `$max_fanout`; `neighbors_cypher` gets `$ids` instead of terms / keys,
`degree_cypher` gets `$id`. An expand / neighbors template returns one row per chunk with `id` (the chunk id as
the vector store or the local corpus knows it) and `path_length` (chunk hops, 1 = mentions a query entity), and
optionally `degree`, `via`, and `text`, `doc_id`, `title`, `updated_at`, `authority` (then a graph hit can become
a candidate even when the vector store does not return it). Cypher cannot take a parameter as the upper bound of a
variable-length pattern, so a template carries a literal bound and filters with `length(p) <= $max_edges` (the
defaults are one constant query per bound: `*1..4` for up to 2 hops, `*1..6` for 3).

Hosts and credentials: `uri` goes through `security.check_host`; the password is read from the environment
variable named by `password_env` (unset = no authentication), the user from `user` (default "neo4j") or the
variable named by `user_env`; a URI with `user:password@` is refused. Failures become 503 `connector_unavailable`
naming the database and the exception type, never the URI or the password.

What this does not do: the default templates walk variable-length paths and skip hub entities (more than
`max_fanout` mentions) by counting their mentions per path, which is fine for a corpus open-rlcd indexed but slow
on a graph with millions of densely connected nodes: write a template that uses your own indexes and relationship
types there. `stats()` makes no request (no node counts).
"""
from __future__ import annotations

import contextlib
import re
import threading
from typing import Any

from ..core.contract import ContractError
from .base import (
    CANDIDATE_FIELDS,
    check_keys,
    config_error,
    connector_failure,
    env_name,
    env_secret,
    json_safe,
    read_only,
    refuse_inline_secrets,
    require_extra,
    url_has_credentials,
)
from .local_graph import ENTITY_VERSION, entities, normalize_entity
from .security import check_host

KEYS = {"kind", "mode", "uri", "user", "user_env", "password_env", "database", "expand_cypher", "neighbors_cypher",
        "degree_cypher", "max_fanout", "terms", "timeout"}
MODES = ("index", "attach")
MAX_HOPS = 3

# ---------------------------------------------------------------------------- open-rlcd's own schema (index mode)

# The variable-length bound must be a literal (Cypher takes no parameter there), and Neo4j expands up to it before
# filtering on `length(p) <= $max_edges`: so there is one constant query per bound, 4 edges for up to 2 chunk hops
# (the pack's default) and 6 for 3, picked by the requested hops. Nothing variable is ever formatted into a query.
_RETURN_ROWS = """
WITH c, p ORDER BY length(p) ASC
WITH c, head(collect(p)) AS best
RETURN c.id AS id, (length(best) + 1) / 2 AS path_length, size([(c)--(z) | z]) AS degree,
       coalesce([n IN nodes(best) WHERE n:RlcdEntity][-1].label, c.doc_id) AS via
ORDER BY path_length ASC, degree DESC, id ASC
LIMIT $limit"""
_NOT_THROUGH_HUBS = "none(n IN nodes(p)[1..-1] WHERE n:RlcdEntity AND size([(n)<-[:MENTIONS]-(y) | y]) > $max_fanout)"


def _expand(bound: int) -> str:
    return (f"MATCH (s:RlcdEntity) WHERE s.key IN $keys AND size([(s)<-[:MENTIONS]-(x) | x]) <= $max_fanout\n"
            f"MATCH p = (s)-[:MENTIONS|HAS_CHUNK|LINKS*1..{bound}]-(c:RlcdChunk)\n"
            f"WHERE length(p) <= $max_edges AND {_NOT_THROUGH_HUBS}{_RETURN_ROWS}")


def _neighbors(bound: int) -> str:
    return (f"MATCH (start:RlcdChunk) WHERE start.id IN $ids\n"
            f"MATCH p = (start)-[:MENTIONS|HAS_CHUNK|LINKS*1..{bound}]-(c:RlcdChunk)\n"
            f"WHERE length(p) <= $max_edges AND NOT c.id IN $ids AND {_NOT_THROUGH_HUBS}{_RETURN_ROWS}")


EXPAND_CYPHER, EXPAND_CYPHER_DEEP = _expand(4), _expand(6)            # built once, at import time
NEIGHBORS_CYPHER, NEIGHBORS_CYPHER_DEEP = _neighbors(4), _neighbors(6)

DEGREE_CYPHER = "MATCH (c:RlcdChunk {id: $id}) RETURN size([(c)--(z) | z]) AS degree"

SCHEMA = (
    "CREATE CONSTRAINT rlcd_doc_id IF NOT EXISTS FOR (d:RlcdDoc) REQUIRE d.id IS UNIQUE",
    "CREATE CONSTRAINT rlcd_chunk_id IF NOT EXISTS FOR (c:RlcdChunk) REQUIRE c.id IS UNIQUE",
    "CREATE CONSTRAINT rlcd_entity_key IF NOT EXISTS FOR (e:RlcdEntity) REQUIRE e.key IS UNIQUE",
)

DROP_CHUNKS = """
MATCH (:RlcdDoc {id: $doc_id})-[:HAS_CHUNK]->(c:RlcdChunk)
OPTIONAL MATCH (c)-[:MENTIONS]->(e:RlcdEntity)
WITH collect(DISTINCT c) AS chunks, collect(DISTINCT e) AS ents
FOREACH (c IN chunks | DETACH DELETE c)
WITH ents
UNWIND ents AS e
WITH e WHERE NOT EXISTS { (e)<-[:MENTIONS]-() }
DELETE e
""".strip()

DROP_LINKS = "MATCH (:RlcdDoc {id: $doc_id})-[r:LINKS]->() DELETE r"

UPSERT_DOC = """
MERGE (d:RlcdDoc {id: $doc_id}) SET d.placeholder = false
WITH d
UNWIND $chunks AS ch
MERGE (c:RlcdChunk {id: ch.id}) SET c.doc_id = $doc_id
MERGE (d)-[:HAS_CHUNK]->(c)
WITH c, ch
UNWIND ch.entities AS ent
MERGE (e:RlcdEntity {key: ent.key}) ON CREATE SET e.label = ent.label
MERGE (c)-[:MENTIONS]->(e)
""".strip()

UPSERT_LINKS = """
MATCH (d:RlcdDoc {id: $doc_id})
UNWIND $links AS other
MERGE (o:RlcdDoc {id: other}) ON CREATE SET o.placeholder = true
MERGE (d)-[:LINKS]->(o)
""".strip()

DROP_DOC = """
MATCH (d:RlcdDoc {id: $doc_id})
WITH d, EXISTS { (d)<-[:LINKS]-() } AS linked
FOREACH (_ IN CASE WHEN linked THEN [1] ELSE [] END | SET d.placeholder = true)
FOREACH (_ IN CASE WHEN linked THEN [] ELSE [1] END | DETACH DELETE d)
""".strip()

# ---------------------------------------------------------------------------- template checks

_LITERALS = re.compile(r"'(?:[^'\\]|\\.)*'|\"(?:[^\"\\]|\\.)*\"|`[^`]*`|//[^\n]*|/\*.*?\*/", re.S)
_WRITES = re.compile(r"\b(CREATE|MERGE|DELETE|DETACH|SET|REMOVE|DROP|FOREACH|LOAD\s+CSV|IN\s+TRANSACTIONS)\b", re.I)


def check_template(cypher: Any, key: str, needs: tuple[str, ...]) -> str:
    """A configured Cypher template: a non-empty string, without write clauses (outside string literals and
    comments), that uses one of the parameters in `needs`."""
    if not isinstance(cypher, str) or not cypher.strip():
        raise config_error(f"pack_config.graph.{key} must be a Cypher query (a non-empty string)")
    code = _LITERALS.sub(" ", cypher)
    m = _WRITES.search(code)
    if m:
        raise config_error(f"pack_config.graph.{key} contains the write clause {m.group(1).upper()!r}: templates are "
                           f"read-only (they run in read transactions)")
    if not any(re.search(rf"\${n}\b", code) for n in needs):
        raise config_error(f"pack_config.graph.{key} must use the parameter {' or '.join('$' + n for n in needs)} "
                           f"(query values are passed as parameters, never formatted into the query)")
    return cypher.strip()


class Neo4jGraphStore:
    """A `GraphStore` on Neo4j (see the module docstring for the modes, templates and security)."""

    backend = "neo4j"

    def __init__(self, uri: str, *, mode: str = "index", user: str | None = None, user_env: str | None = None,
                 password_env: str | None = None, database: str | None = None, expand_cypher: str | None = None,
                 neighbors_cypher: str | None = None, degree_cypher: str | None = None, max_fanout: int = 200,
                 terms: list[str] | None = None, timeout: int = 10, allow_remote: bool = False):
        if not isinstance(uri, str) or not uri.strip():
            raise config_error("the neo4j connector needs `uri` (e.g. bolt://localhost:7687)")
        if url_has_credentials(uri):
            raise config_error("the neo4j uri carries credentials: rlcd.yaml never holds credentials; set user and "
                               "name the password's environment variable with password_env")
        try:
            check_host(uri, allow_remote=allow_remote)
        except ValueError as e:
            raise config_error(f"neo4j: {e}") from None
        self.uri = uri.strip()
        self.mode = str(mode or "index").lower()
        if self.mode not in MODES:
            raise config_error(f"neo4j mode must be one of {MODES}, got {mode!r}")
        self._user_env = env_name(user_env, "user_env")
        self._password_env = env_name(password_env, "password_env")
        self.user = str(user) if user else "neo4j"
        self.database = str(database) if database else None
        if self.mode == "attach" and expand_cypher is None:
            raise config_error("neo4j attach mode needs expand_cypher: a read-only Cypher template that maps the "
                               "query's entities ($terms / $keys) to chunk ids with path_length (see "
                               "open_rlcd/connectors/neo4j.py)")
        # None = open-rlcd's own schema: the default query for the requested hops (EXPAND_CYPHER / _DEEP)
        self.expand_cypher: str | None = check_template(expand_cypher, "expand_cypher", ("terms", "keys")) \
            if expand_cypher is not None else None
        self.neighbors_cypher: str | None = check_template(neighbors_cypher, "neighbors_cypher", ("ids",)) \
            if neighbors_cypher is not None else None
        self._default_neighbors = neighbors_cypher is None and self.mode == "index"
        if degree_cypher is not None:
            self.degree_cypher: str | None = check_template(degree_cypher, "degree_cypher", ("id",))
        else:
            self.degree_cypher = DEGREE_CYPHER if self.mode == "index" else None
        self.max_fanout = int(max_fanout)
        self.terms = [str(t) for t in (terms or [])]
        self.timeout = int(timeout)
        self.entity_version = ENTITY_VERSION
        self._driver: Any = None
        self._neo4j: Any = None
        self._ready = False
        self._schema_done = False
        self._lock = threading.RLock()

    @classmethod
    def from_config(cls, cfg: dict, allow_remote: bool = False, terms: list[str] | None = None) -> Neo4jGraphStore:
        refuse_inline_secrets(cfg, "pack_config.graph")
        check_keys(cfg, KEYS, "pack_config.graph (neo4j)")
        return cls(cfg.get("uri"), mode=cfg.get("mode") or "index", user=cfg.get("user"),
                   user_env=cfg.get("user_env"), password_env=cfg.get("password_env"), database=cfg.get("database"),
                   expand_cypher=cfg.get("expand_cypher"), neighbors_cypher=cfg.get("neighbors_cypher"),
                   degree_cypher=cfg.get("degree_cypher"), max_fanout=int(cfg.get("max_fanout") or 200),
                   terms=terms if terms is not None else cfg.get("terms"), timeout=int(cfg.get("timeout") or 10),
                   allow_remote=allow_remote)

    @property
    def target(self) -> str:
        return f"database {self.database!r}" if self.database else "default database"

    @property
    def writable(self) -> bool:
        return self.mode == "index"

    def describe(self) -> dict:
        return {"kind": "neo4j", "database": self.database, "mode": self.mode}

    def stats(self) -> dict:
        return {"backend": "neo4j", "mode": self.mode, "database": self.database}

    def entities(self, text: str, extra: list[str] | None = None) -> list[str]:
        return entities(text, [*self.terms, *(extra or [])])

    # ------------------------------------------------------------------ connection

    def _secrets(self) -> tuple[str | None, ...]:
        return (env_secret(self._password_env),)

    def _failure(self, e: BaseException) -> ContractError:
        hint = ""
        if type(e).__name__ in ("AuthError", "AuthConfigurationError"):
            hint = (f"authentication failed: check user and that {self._password_env} holds the password"
                    if self._password_env else "authentication failed: name the password's environment variable "
                                               "with password_env")
        return connector_failure("neo4j", self.target, e, self._secrets(), hint)

    def connect(self) -> None:
        """Create the driver and verify connectivity (503 when the server cannot be reached or refuses the login)."""
        with self._lock:
            if self._ready:
                return
            neo4j = require_extra("neo4j", "neo4j", "neo4j")
            password = env_secret(self._password_env)
            user = env_secret(self._user_env) or self.user
            try:
                drv = neo4j.GraphDatabase.driver(self.uri, auth=(user, password) if password else None,
                                                 connection_timeout=self.timeout)
                drv.verify_connectivity()
            except Exception as e:
                raise self._failure(e) from None
            self._driver, self._neo4j, self._ready = drv, neo4j, True

    def close(self) -> None:
        with self._lock:
            if self._driver is not None:
                with contextlib.suppress(Exception):
                    self._driver.close()
            self._driver, self._ready = None, False

    def _read(self, query: str, params: dict) -> list[dict]:
        self.connect()
        try:
            with self._driver.session(database=self.database, default_access_mode=self._neo4j.READ_ACCESS) as s:
                return s.execute_read(lambda tx: [r.data() for r in tx.run(query, params)])
        except ContractError:
            raise
        except Exception as e:
            raise self._failure(e) from None

    def _write(self, statements: list[tuple[str, dict]]) -> None:
        """Run `statements` in one write transaction (index mode only)."""
        if not self.writable:
            raise read_only(f"neo4j {self.target} is attached read-only (mode: attach): open-rlcd never writes to it")
        self.connect()

        def work(tx: Any) -> None:
            for q, p in statements:
                tx.run(q, p).consume()

        try:
            with self._driver.session(database=self.database, default_access_mode=self._neo4j.WRITE_ACCESS) as s:
                if not self._schema_done:
                    for q in SCHEMA:          # schema changes cannot share a transaction with data writes
                        s.execute_write(lambda tx, q=q: tx.run(q).consume())
                    self._schema_done = True
                s.execute_write(work)
        except ContractError:
            raise
        except Exception as e:
            raise self._failure(e) from None

    # ------------------------------------------------------------------ reads

    @staticmethod
    def _int(v: Any) -> int | None:
        try:
            return int(v) if v is not None else None
        except (TypeError, ValueError):
            return None

    def _rows(self, records: list[dict], limit: int) -> list[dict]:
        out, seen = [], set()
        for r in records:
            cid = r.get("id")
            if cid is None or str(cid) in seen:
                continue
            seen.add(str(cid))
            row = {"id": str(cid), "path_length": self._int(r.get("path_length")),
                   "degree": self._int(r.get("degree")), "via": str(r["via"]) if r.get("via") is not None else None}
            for f in CANDIDATE_FIELDS:
                if r.get(f) is not None:
                    row[f] = json_safe(r[f])
            out.append(row)
        return out[: max(0, int(limit))]

    def expand(self, terms: list[str], hops: int = 2, limit: int = 50) -> list[dict]:
        """Chunks within `hops` chunk-hops of the entities named by `terms`, from `expand_cypher` (parameters only):
        [{id, path_length, degree, via, (text, doc_id, title, updated_at, authority when the template returns
        them)}]."""
        terms = [str(t) for t in (terms or []) if str(t).strip()]
        keys = list(dict.fromkeys(k for k in (normalize_entity(t) for t in terms) if len(k) >= 2))
        limit = max(0, int(limit))
        if not keys or limit == 0:
            return []
        hops = max(1, min(MAX_HOPS, int(hops)))
        params = {"terms": terms, "keys": keys, "hops": hops, "max_edges": 2 * hops, "limit": limit,
                  "max_fanout": self.max_fanout}
        query = self.expand_cypher or (EXPAND_CYPHER if hops <= 2 else EXPAND_CYPHER_DEEP)
        return self._rows(self._read(query, params), limit)

    def neighbors(self, ids: list[str], hops: int = 1, limit: int = 50) -> list[dict]:
        """Chunks within `hops` chunk-hops of the given chunks (`neighbors_cypher`; [] in attach mode without one)."""
        if self.neighbors_cypher is None and not self._default_neighbors:
            return []
        ids = [str(i) for i in ids or []]
        limit = max(0, int(limit))
        if not ids or limit == 0:
            return []
        hops = max(1, min(MAX_HOPS, int(hops)))
        params = {"ids": ids, "hops": hops, "max_edges": 2 * hops, "limit": limit, "max_fanout": self.max_fanout}
        query = self.neighbors_cypher or (NEIGHBORS_CYPHER if hops <= 2 else NEIGHBORS_CYPHER_DEEP)
        return self._rows(self._read(query, params), limit)

    def chunk_degree(self, chunk_id: str) -> int | None:
        """The chunk's degree from `degree_cypher` (None in attach mode without one, or for an unknown chunk)."""
        if self.degree_cypher is None:
            return None
        rows = self._read(self.degree_cypher, {"id": str(chunk_id)})
        return self._int(rows[0].get("degree")) if rows else None

    # ------------------------------------------------------------------ writes (index mode only)

    def add_document(self, doc_id: str, chunks: list[tuple[str, str]], links: list[str] | None = None) -> None:
        """(Re)place a document: its chunks, the entities they mention and the links it declares."""
        doc_id = str(doc_id)
        rows = []
        for cid, text in chunks:
            ents: dict[str, str] = {}
            for e in self.entities(text):
                key = normalize_entity(e)
                if key:
                    ents.setdefault(key, e)
            rows.append({"id": str(cid), "entities": [{"key": k, "label": v} for k, v in ents.items()]})
        others = [str(o) for o in dict.fromkeys(links or []) if str(o) != doc_id]
        self._write([(DROP_CHUNKS, {"doc_id": doc_id}), (DROP_LINKS, {"doc_id": doc_id}),
                     (UPSERT_DOC, {"doc_id": doc_id, "chunks": rows}),
                     (UPSERT_LINKS, {"doc_id": doc_id, "links": others})])

    def remove_document(self, doc_id: str, keep_node: bool = False) -> None:
        doc_id = str(doc_id)
        statements = [(DROP_CHUNKS, {"doc_id": doc_id}), (DROP_LINKS, {"doc_id": doc_id})]
        if not keep_node:
            statements.append((DROP_DOC, {"doc_id": doc_id}))
        self._write(statements)

    def save(self) -> None:
        """Nothing to save: every write is committed."""
