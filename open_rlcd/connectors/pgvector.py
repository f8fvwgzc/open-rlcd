"""Postgres + pgvector as the documents pack's vector store (optional extra `pgvector`: psycopg 3).

    pack_config:
      vector: {kind: pgvector, dsn_env: RLCD_PG_DSN, table: rlcd_chunks}                     # mode: index (default)
      vector: {kind: pgvector, mode: attach, dsn_env: KB_DSN, table: public.kb_chunks,       # an existing table
               id_field: chunk_id, vector_field: embedding, text_field: body, doc_id_field: document_id,
               title_field: heading, updated_at_field: modified_at, authority_field: null}

The DSN (`postgresql://user:password@host/db` or `host=... dbname=...`) is read from the environment variable
named by `dsn_env` (default `RLCD_PG_DSN`), never from rlcd.yaml. Every host it names (and `PGHOST` when it names
none) goes through `security.check_host`: loopback and unix sockets only unless remote hosts are allowed.

Two modes:

* **index**: open-rlcd owns the table. On first use it runs `CREATE EXTENSION IF NOT EXISTS vector` and creates
  the table if missing (`id text primary key, doc_id, title, text, updated_at bigint, authority double precision,
  embedding vector(<encoder dim>)`) plus an HNSW index with `vector_cosine_ops`; `rlcd ingest` upserts chunks
  (`INSERT ... ON CONFLICT (id) DO UPDATE`) and deletes the old chunks of a changed document.
* **attach**: read-only search of a table that already exists: the connection's transactions are `READ ONLY`
  (psycopg `read_only = True`, so Postgres itself refuses any write), `upsert` / `delete` raise before a statement
  is sent, and nothing is created. The `*_field` keys name the columns (`id_field` and `vector_field` too; `null`
  = the table has no such column). The vector column must be a `vector` whose dimension (its type modifier, or
  `vector_dims` of a stored row) equals the encoder's, otherwise the store refuses to search
  (`encoder_mismatch`). **Query vectors come from open-rlcd's encoder** (pack_config.encoder): the table must have
  been embedded with that same model; equal dimensions are necessary, not sufficient. Otherwise pass
  `candidates` in the request.

SQL: identifiers (table, columns) are composed with `psycopg.sql.Identifier` (quoted, never pasted in as text);
every value (query vector, limit, ids, filter values) is a bound parameter. Vectors travel as pgvector text
literals cast with `::vector`, so no client-side type registration is needed. Search orders by cosine distance
(`<=>`), and `score = 1 - distance` is the cosine similarity. Failures become 503 `connector_unavailable` naming
the table and the exception type, never the DSN or password; the connection is reopened on the next call.

What this does not do: no connection pool (one connection per store, calls serialised by a lock), and lookups by
id cast the id column to text (`id::text = ANY(...)`), which skips a non-text id column's index; that lookup only
serves graph hits and candidates the vector search did not return.
"""
from __future__ import annotations

import contextlib
import json
import os
import threading
from typing import Any

import numpy as np

from ..core.contract import ContractError
from .base import (
    FieldMap,
    Hit,
    as_query,
    check_keys,
    config_error,
    connector_failure,
    dim_mismatch,
    env_name,
    env_secret,
    read_only,
    refuse_inline_secrets,
    require_extra,
)
from .security import check_host

KEYS = {"kind", "mode", "dsn_env", "table", "timeout", "create_index", "batch_size", "vector_field",
        "id_field", "text_field", "doc_id_field", "title_field", "updated_at_field", "authority_field"}
MODES = ("index", "attach")
DEFAULT_DSN_ENV = "RLCD_PG_DSN"
SELECT_FIELDS = ("doc_id", "title", "text", "updated_at", "authority")


def vector_literal(v: np.ndarray) -> str:
    """A pgvector text literal: '[0.1,0.2,...]' (sent as a bound parameter and cast with ::vector)."""
    return "[" + ",".join(format(float(x), ".9g") for x in np.asarray(v, dtype=np.float32).reshape(-1)) + "]"


def parse_vector(s: Any) -> np.ndarray | None:
    """A pgvector value read back as text ('[1,2,3]') -> float32 array."""
    if s is None:
        return None
    if isinstance(s, (list, tuple, np.ndarray)):
        return np.asarray(s, dtype=np.float32)
    try:
        return np.asarray(json.loads(str(s)), dtype=np.float32)
    except (ValueError, TypeError):
        return None


class PgVectorStore:
    """A `VectorStore` on a Postgres table with a pgvector column (see the module docstring)."""

    backend = "pgvector"

    def __init__(self, table: str, dim: int, *, dsn_env: str | None = DEFAULT_DSN_ENV, mode: str = "index",
                 fields: dict | None = None, timeout: int = 10, create_index: bool = True, batch_size: int = 500,
                 allow_remote: bool = False):
        self._pg = require_extra("psycopg", "pgvector", "pgvector")
        self._sql = require_extra("psycopg.sql", "pgvector", "pgvector")
        if not isinstance(table, str) or not table.strip():
            raise config_error("the pgvector connector needs `table` (optionally schema-qualified: schema.table)")
        parts = [p.strip() for p in table.strip().split(".")]
        if len(parts) > 2 or not all(parts):
            raise config_error(f"pgvector table must be `table` or `schema.table`, got {table!r}")
        self.table = ".".join(parts)
        self._schema, self._name = (parts[0], parts[1]) if len(parts) == 2 else (None, parts[0])
        self.dim = int(dim)
        self.mode = str(mode or "index").lower()
        if self.mode not in MODES:
            raise config_error(f"pgvector mode must be one of {MODES}, got {mode!r}")
        self._dsn_env = env_name(dsn_env or DEFAULT_DSN_ENV, "dsn_env")
        dsn = env_secret(self._dsn_env)
        if not dsn:
            raise config_error(f"the environment variable {self._dsn_env} (pack_config.vector.dsn_env) is not set: "
                               f"put the Postgres connection string there (never in rlcd.yaml)")
        self._dsn = dsn
        self._password = self._check_hosts(dsn, allow_remote)
        f = dict(fields or {})
        self.vector_field = str(f.pop("vector_field", None) or "embedding")
        base = FieldMap(id="id")
        self.fields = base if self.mode == "index" else FieldMap.from_config(f, base)
        if self.fields.id is None:
            raise config_error("pgvector needs id_field: the column that holds the chunk id")
        self.timeout = int(timeout)
        self.create_index = bool(create_index)
        self.batch_size = max(1, int(batch_size))
        self._conn: Any = None
        self._ready = False
        self._lock = threading.RLock()

    @classmethod
    def from_config(cls, cfg: dict, dim: int, allow_remote: bool = False) -> PgVectorStore:
        refuse_inline_secrets(cfg, "pack_config.vector")
        check_keys(cfg, KEYS, "pack_config.vector (pgvector)")
        return cls(cfg.get("table"), dim, dsn_env=cfg.get("dsn_env") or DEFAULT_DSN_ENV,
                   mode=cfg.get("mode") or "index", fields={k: v for k, v in cfg.items() if k.endswith("_field")},
                   timeout=int(cfg.get("timeout") or 10), create_index=cfg.get("create_index", True) is not False,
                   batch_size=int(cfg.get("batch_size") or 500), allow_remote=allow_remote)

    @property
    def target(self) -> str:
        return f"table {self.table!r}"

    def describe(self) -> dict:
        return {"kind": "pgvector", "table": self.table}

    def _check_hosts(self, dsn: str, allow_remote: bool) -> str | None:
        """Every host of the DSN through check_host; returns the DSN's password (to scrub it from logs)."""
        try:
            params = self._pg.conninfo.conninfo_to_dict(dsn)
        except Exception:
            raise config_error(f"the connection string in {self._dsn_env} cannot be parsed") from None
        hosts = [h for h in str(params.get("host") or os.environ.get("PGHOST") or "").split(",") if h.strip()]
        hosts += [h for h in str(params.get("hostaddr") or "").split(",") if h.strip()]
        for h in hosts:
            h = h.strip()
            if h.startswith(("/", "@")):        # a unix socket directory / an abstract socket
                continue
            try:
                check_host(h, allow_remote=allow_remote)
            except ValueError as e:
                raise config_error(f"pgvector ({self._dsn_env}): {e}") from None
        return params.get("password") or None

    # ------------------------------------------------------------------ SQL pieces

    def _ident(self, *names: str) -> Any:
        return self._sql.Identifier(*names)

    def _table(self) -> Any:
        return self._ident(self._schema, self._name) if self._schema else self._ident(self._name)

    def _select_list(self) -> Any:
        S = self._sql
        items = [S.SQL("{}::text AS {}").format(self._ident(self.fields.id), self._ident("id"))]
        for name in SELECT_FIELDS:
            col = getattr(self.fields, name)
            items.append(S.SQL("{} AS {}").format(self._ident(col), self._ident(name)) if col
                         else S.SQL("NULL AS {}").format(self._ident(name)))
        return S.SQL(", ").join(items)

    def _where(self, flt: dict | None) -> tuple[Any, list]:
        if not flt:
            return self._sql.SQL(""), []
        S = self._sql
        parts, params = [], []
        for k, v in flt.items():
            col = getattr(self.fields, str(k), None) if str(k) in SELECT_FIELDS else str(k)
            if not col:
                raise ValueError(f"cannot filter on {k!r}: the table has no such column")
            if isinstance(v, (list, tuple, set)):
                parts.append(S.SQL("{}::text = ANY(%s::text[])").format(self._ident(col)))
                params.append([str(x) for x in v])
            else:
                parts.append(S.SQL("{} = %s").format(self._ident(col)))
                params.append(v)
        return S.SQL(" WHERE ") + S.SQL(" AND ").join(parts), params

    def _row(self, row: tuple) -> tuple[str, dict]:
        rec = dict(zip(("id", *SELECT_FIELDS), row[:6], strict=True))
        return str(rec["id"]), FieldMap(id="id").candidate(rec)

    # ------------------------------------------------------------------ connection

    def _secrets(self) -> tuple[str | None, ...]:
        return (self._dsn, self._password)

    def _column(self, cur: Any) -> tuple[str, int] | None:
        """(type name, dimension or -1) of the vector column, or None when the table / column does not exist."""
        cur.execute("SELECT t.typname, a.atttypmod FROM pg_attribute a "
                    "JOIN pg_class c ON c.oid = a.attrelid "
                    "JOIN pg_namespace n ON n.oid = c.relnamespace "
                    "JOIN pg_type t ON t.oid = a.atttypid "
                    "WHERE c.relname = %s AND n.nspname = COALESCE(%s::text, current_schema()) "
                    "AND a.attname = %s AND NOT a.attisdropped",
                    (self._name, self._schema, self.vector_field))
        row = cur.fetchone()
        return None if row is None else (str(row[0]), int(row[1]))

    def _missing_columns(self, cur: Any) -> list[str]:
        want = [c for c in (self.fields.id, *(getattr(self.fields, n) for n in SELECT_FIELDS)) if c]
        cur.execute("SELECT a.attname FROM pg_attribute a "
                    "JOIN pg_class c ON c.oid = a.attrelid "
                    "JOIN pg_namespace n ON n.oid = c.relnamespace "
                    "WHERE c.relname = %s AND n.nspname = COALESCE(%s::text, current_schema()) "
                    "AND a.attnum > 0 AND NOT a.attisdropped", (self._name, self._schema))
        have = {str(r[0]) for r in cur.fetchall()}
        return [c for c in want if c not in have]

    def _create(self, cur: Any) -> None:
        S = self._sql
        f = self.fields
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
        cur.execute(S.SQL("CREATE TABLE IF NOT EXISTS {t} ({id} text PRIMARY KEY, {doc} text, {title} text, "
                          "{text} text, {upd} bigint, {auth} double precision, {emb} vector({dim}))").format(
            t=self._table(), id=self._ident(f.id), doc=self._ident(f.doc_id), title=self._ident(f.title),
            text=self._ident(f.text), upd=self._ident(f.updated_at), auth=self._ident(f.authority),
            emb=self._ident(self.vector_field), dim=S.Literal(int(self.dim))))
        if self.create_index:
            idx = f"{self._name}_{self.vector_field}_hnsw"[:63]
            cur.execute(S.SQL("CREATE INDEX IF NOT EXISTS {i} ON {t} USING hnsw ({emb} vector_cosine_ops)").format(
                i=self._ident(idx), t=self._table(), emb=self._ident(self.vector_field)))

    def connect(self) -> None:
        """Open the connection and check the table (index: create it if missing; attach: it must exist, with the
        mapped columns and a vector column of the encoder's dimension)."""
        with self._lock:
            if self._ready and self._conn is not None and not getattr(self._conn, "closed", False):
                return
            self._close()
            try:
                conn = self._pg.connect(self._dsn, connect_timeout=self.timeout)
            except Exception as e:
                raise connector_failure("pgvector", self.target, e, self._secrets()) from None
            try:
                if self.mode == "attach":
                    conn.read_only = True          # BEGIN READ ONLY: Postgres refuses any write in these transactions
                with conn.cursor() as cur:
                    col = self._column(cur)
                    if col is None and self.mode == "index":
                        self._create(cur)
                        col = self._column(cur)
                    if col is None:
                        raise config_error(f"pgvector {self.target} has no column {self.vector_field!r} (or the "
                                           f"table does not exist); attach mode never creates anything")
                    missing = self._missing_columns(cur)
                    if missing:
                        raise config_error(f"pgvector {self.target} has no column(s) {missing}: fix the *_field "
                                           f"mapping in pack_config.vector")
                    typ, dims = col
                    if typ != "vector":
                        raise config_error(f"pgvector {self.target}: column {self.vector_field!r} has type {typ!r}, "
                                           f"not vector")
                    if dims <= 0:
                        S = self._sql
                        cur.execute(S.SQL("SELECT vector_dims({e}) FROM {t} WHERE {e} IS NOT NULL LIMIT 1").format(
                            e=self._ident(self.vector_field), t=self._table()))
                        row = cur.fetchone()
                        dims = int(row[0]) if row else self.dim      # an empty untyped column: nothing to mismatch
                if self.mode == "index":
                    conn.commit()
                else:
                    conn.rollback()
            except Exception as e:
                with contextlib.suppress(Exception):
                    conn.close()
                if isinstance(e, ContractError):
                    raise
                raise connector_failure("pgvector", self.target, e, self._secrets()) from None
            if dims != self.dim:
                conn.close()
                raise dim_mismatch("pgvector", self.target, dims, self.dim)
            self._conn, self._ready = conn, True

    def _close(self) -> None:
        if self._conn is not None:
            with contextlib.suppress(Exception):
                self._conn.close()
        self._conn, self._ready = None, False

    def close(self) -> None:
        with self._lock:
            self._close()

    def _run(self, fn: Any, write: bool = False) -> Any:
        self.connect()
        with self._lock:
            conn = self._conn
            try:
                with conn.cursor() as cur:
                    out = fn(cur)
                if write:
                    conn.commit()
                else:
                    conn.rollback()           # end the read transaction (nothing to commit)
                return out
            except Exception as e:
                self._close()
                if isinstance(e, ContractError):
                    raise
                raise connector_failure("pgvector", self.target, e, self._secrets()) from None

    # ------------------------------------------------------------------ reads

    def search(self, vector: np.ndarray, k: int = 10, filter: dict | None = None) -> list[Hit]:
        k = int(k)
        q = as_query(vector, self.dim)
        if k <= 0:
            return []
        S = self._sql
        where, wparams = self._where(filter)
        emb = self._ident(self.vector_field)
        query = S.SQL("SELECT {cols}, 1 - ({emb} <=> %s::vector) AS score FROM {t}{where} "
                      "ORDER BY {emb} <=> %s::vector LIMIT %s").format(cols=self._select_list(), emb=emb,
                                                                       t=self._table(), where=where)
        lit = vector_literal(q)

        def run(cur: Any) -> list:
            cur.execute(query, [lit, *wparams, lit, k])
            return cur.fetchall()

        hits = []
        for row in self._run(run):
            cid, cand = self._row(row)
            score = float(row[6]) if row[6] is not None else float("nan")
            hits.append(Hit(cid, score, cand))
        return hits

    def fetch(self, ids: list[str]) -> dict[str, tuple[np.ndarray | None, dict]]:
        ids = list(dict.fromkeys(str(i) for i in ids))
        if not ids:
            return {}
        S = self._sql
        query = S.SQL("SELECT {cols}, {emb}::text FROM {t} WHERE {id}::text = ANY(%s::text[])").format(
            cols=self._select_list(), emb=self._ident(self.vector_field), t=self._table(),
            id=self._ident(self.fields.id))

        def run(cur: Any) -> list:
            cur.execute(query, [ids])
            return cur.fetchall()

        out: dict[str, tuple[np.ndarray | None, dict]] = {}
        for row in self._run(run):
            cid, cand = self._row(row)
            out.setdefault(cid, (parse_vector(row[6]), cand))
        return out

    def vectors(self, ids: list[str]) -> np.ndarray:
        out = np.full((len(ids), self.dim), np.nan, dtype=np.float32)
        got = self.fetch(ids)
        for r, i in enumerate(ids):
            v = got.get(str(i), (None, {}))[0]
            if v is not None and v.shape == (self.dim,):
                n = float(np.linalg.norm(v))
                out[r] = v / n if n > 1e-12 else 0.0
        return out

    def count(self) -> int:
        S = self._sql

        def run(cur: Any) -> int:
            cur.execute(S.SQL("SELECT count(*) FROM {t}").format(t=self._table()))
            return int(cur.fetchone()[0])

        return self._run(run)

    # ------------------------------------------------------------------ writes (index mode only)

    def _writable(self, what: str) -> None:
        if self.mode != "index":
            raise read_only(f"pgvector {self.target} is attached read-only (mode: attach): open-rlcd never {what} "
                            f"it. To let open-rlcd index documents, give it a table of its own with mode: index")

    def upsert(self, ids: list[str], vectors: np.ndarray, payloads: list[dict] | None = None) -> None:
        self._writable("writes to")
        ids = [str(i) for i in ids]
        vecs = np.asarray(vectors, dtype=np.float32)
        if vecs.ndim == 1:
            vecs = vecs[None, :]
        if vecs.shape != (len(ids), self.dim):
            raise ValueError(f"vectors must have shape ({len(ids)}, {self.dim}), got {vecs.shape}")
        if not np.all(np.isfinite(vecs)):
            raise ValueError("vectors have NaN or infinite values")
        payloads = [dict(p or {}) for p in (payloads if payloads is not None else [{}] * len(ids))]
        if len(payloads) != len(ids) or len(set(ids)) != len(ids):
            raise ValueError("ids and payloads differ in length, or ids repeat")
        S, f = self._sql, self.fields
        cols = [f.id, f.doc_id, f.title, f.text, f.updated_at, f.authority, self.vector_field]
        query = S.SQL("INSERT INTO {t} ({cols}) VALUES (%s, %s, %s, %s, %s, %s, %s::vector) ON CONFLICT ({id}) DO "
                      "UPDATE SET {sets}").format(
            t=self._table(), cols=S.SQL(", ").join(self._ident(c) for c in cols), id=self._ident(f.id),
            sets=S.SQL(", ").join(S.SQL("{c} = EXCLUDED.{c}").format(c=self._ident(c)) for c in cols[1:]))
        rows = []
        for i, v, p in zip(ids, vecs, payloads, strict=True):
            upd = p.get("updated_at")
            auth = p.get("authority")
            rows.append((i, p.get("doc_id"), p.get("title"), p.get("text"),
                         int(upd) if isinstance(upd, (int, float)) and np.isfinite(upd) else None,
                         float(auth) if isinstance(auth, (int, float)) and np.isfinite(auth) else None,
                         vector_literal(v)))

        def run(cur: Any) -> None:
            for s in range(0, len(rows), self.batch_size):
                cur.executemany(query, rows[s:s + self.batch_size])

        self._run(run, write=True)

    def delete(self, ids: list[str]) -> int:
        self._writable("deletes from")
        ids = [str(i) for i in ids]
        if not ids:
            return 0
        S = self._sql
        query = S.SQL("DELETE FROM {t} WHERE {id} = ANY(%s::text[])").format(t=self._table(),
                                                                             id=self._ident(self.fields.id))

        def run(cur: Any) -> int:
            cur.execute(query, [ids])
            return int(getattr(cur, "rowcount", 0) or 0)

        return self._run(run, write=True)

    def save(self) -> None:
        """Nothing to save: every write is committed."""

