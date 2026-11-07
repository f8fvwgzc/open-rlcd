"""Connector protocols: what the documents pack needs from a vector store and from a graph store.

Local implementations live next to this file (`local_vector.LocalVectorStore`, `local_graph.LocalGraphStore`);
remote ones (`qdrant.QdrantVectorStore`, `pgvector.PgVectorStore`, `neo4j.Neo4jGraphStore`) implement the same
protocols. A store that cannot be reached is reported as HTTP 503 (`connector_unavailable`): the local stores raise
`ConnectorError` (the API boundary turns it into a 503 with `unavailable(...)`); the remote stores raise that
ContractError themselves, with a message that names the store and the exception type but never a URL, DSN,
password or path (the details go to the server log, scrubbed of the secrets the store read). A dead database is
"service unavailable", never a bad request or a silent empty result.

The helpers below are shared by the remote stores: the lazy import that names the missing extra, credentials read
only from environment variables named in config (never from rlcd.yaml values), and the mapping of a store's
payload fields / columns onto candidate fields.
"""
from __future__ import annotations

import importlib
import logging
import math
import os
import re
from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import Decimal
from typing import Any, Protocol, runtime_checkable
from urllib.parse import urlparse

import numpy as np

from ..core.contract import ContractError

log = logging.getLogger("open_rlcd.connectors")

# the candidate fields a remote store can fill (besides the chunk id and the cosine score)
CANDIDATE_FIELDS = ("doc_id", "title", "text", "updated_at", "authority")
# config keys that hold a credential: refused in rlcd.yaml, with the `*_env` key that replaces them
SECRET_KEYS = {"password": "password_env", "passwd": "password_env", "api_key": "api_key_env",
               "apikey": "api_key_env", "token": "api_key_env", "secret": "api_key_env", "dsn": "dsn_env",
               "conninfo": "dsn_env", "auth": "password_env"}
_ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


@dataclass
class Hit:
    """One vector search result: the stored id, its cosine similarity to the query, and its payload.

    Local stores return their stored payload; remote stores return the mapped candidate fields (`doc_id`, `title`,
    `text`, `updated_at`, `authority`) so that a hit can become a candidate without another round trip."""

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


def config_error(msg: str, **details: object) -> ContractError:
    """A connector configuration that cannot work (a secret in rlcd.yaml, a missing setting, a refused host): 422."""
    return ContractError(msg, kind="connector_config", **details)


def read_only(msg: str, **details: object) -> ContractError:
    """A write asked of a store opened in attach mode (read-only): 422 `read_only_connector`."""
    return ContractError(msg, kind="read_only_connector", **details)


# ----------------------------------------------------------------------------- lazy imports

def require_extra(module: str, extra: str, connector: str) -> Any:
    """Import `module` (the store's client library) or fail with a 503 that names the extra to install."""
    try:
        return importlib.import_module(module)
    except ImportError:
        raise unavailable(f"the {connector} connector needs its client library: install the '{extra}' extra "
                          f"(`uv sync --extra {extra}`, or `pip install 'open-rlcd[{extra}]'`)",
                          connector=connector, extra=extra) from None


# ----------------------------------------------------------------------------- config and credentials

def refuse_inline_secrets(cfg: dict, where: str) -> None:
    """rlcd.yaml never holds credentials: a `password` / `api_key` / `dsn` / ... key is refused, naming the `*_env`
    key that points at an environment variable instead. The value is never echoed."""
    for key in cfg:
        alt = SECRET_KEYS.get(str(key).lower())
        if alt:
            raise config_error(f"{where}.{key} looks like a credential, and rlcd.yaml never holds credentials: put it "
                               f"in an environment variable and name that variable with {where}.{alt}")


def check_keys(cfg: dict, allowed: set[str], where: str) -> None:
    """Unknown keys are refused (a typo such as `colection` must not silently fall back to a default)."""
    unknown = sorted(str(k) for k in cfg if k not in allowed)
    if unknown:
        raise config_error(f"{where}: unknown key(s) {unknown}; allowed: {sorted(allowed)}")


def env_name(value: Any, key: str) -> str | None:
    """A validated environment variable name from config (`api_key_env`, `password_env`, `dsn_env`), or None."""
    if value is None or value == "":
        return None
    if not isinstance(value, str) or not _ENV_NAME.match(value):
        raise config_error(f"{key} must be the name of an environment variable (letters, digits, underscores), "
                           f"not its value")
    return value


def env_secret(name: str | None) -> str | None:
    """The value of the environment variable `name` (None when unnamed or unset / empty)."""
    return (os.environ.get(name) or None) if name else None


def url_has_credentials(url: str) -> bool:
    """True when a URL carries `user[:password]@` (credentials belong in the environment, not in rlcd.yaml)."""
    try:
        u = urlparse(str(url))
        return bool(u.username or u.password)
    except ValueError:
        return False


_USERINFO = re.compile(r"(?<=://)[^/@\s]+@")


def scrub(text: str, secrets: tuple[str | None, ...] = ()) -> str:
    """`text` with the given secret values and any `user:password@` of a URL removed (for the server log)."""
    out = str(text)
    for s in secrets:
        if s:
            out = out.replace(s, "***")
    return _USERINFO.sub("***@", out)


def connector_failure(connector: str, target: str, exc: BaseException, secrets: tuple[str | None, ...] = (),
                      hint: str = "") -> ContractError:
    """A 503 for a store that failed: the message names the store and the exception type only; the scrubbed details
    go to the server log."""
    log.warning("%s %s unavailable: %s: %s", connector, target, type(exc).__name__, scrub(str(exc), secrets)[:500])
    extra = f"; {hint}" if hint else ""
    return unavailable(f"the {connector} store ({target}) is unavailable: {type(exc).__name__}{extra}; details are "
                       f"in the server log", connector=connector)


# ----------------------------------------------------------------------------- payload / column mapping

def json_safe(v: Any) -> Any:
    """A payload value as strict JSON (decisions are logged as JSON): datetimes as ISO 8601, Decimal as float,
    NaN / inf as None, anything else unusual as its string."""
    if v is None or isinstance(v, (bool, int, str)):
        return v
    if isinstance(v, float):
        return v if math.isfinite(v) else None
    if isinstance(v, Decimal):
        f = float(v)
        return f if math.isfinite(f) else None
    if isinstance(v, (datetime, date)):
        return v.isoformat()
    if isinstance(v, (np.integer, np.floating)):
        return json_safe(v.item())
    if isinstance(v, (list, tuple)):
        return [json_safe(x) for x in v]
    if isinstance(v, dict):
        return {str(k): json_safe(x) for k, x in v.items()}
    return str(v)


FIELD_KEYS = ("id_field", "text_field", "doc_id_field", "title_field", "updated_at_field", "authority_field")


@dataclass(frozen=True)
class FieldMap:
    """Where a store keeps each candidate field: a payload key (Qdrant) or a column (pgvector). None = the store
    does not have it (the candidate field stays empty); `id` None = the point id / primary key is the chunk id."""

    id: str | None = None
    text: str = "text"
    doc_id: str | None = "doc_id"
    title: str | None = "title"
    updated_at: str | None = "updated_at"
    authority: str | None = "authority"

    @classmethod
    def from_config(cls, cfg: dict, base: FieldMap | None = None) -> FieldMap:
        base = base or cls()
        vals: dict[str, Any] = {}
        for key in FIELD_KEYS:
            name = key[: -len("_field")]
            if key not in cfg:
                vals[name] = getattr(base, name)
                continue
            v = cfg[key]
            if v is not None and (not isinstance(v, str) or not v.strip()):
                raise config_error(f"{key} must be a field / column name or null")
            vals[name] = v.strip() if isinstance(v, str) else None
        if not vals["text"]:
            raise config_error("text_field cannot be null: a candidate without text cannot be judged")
        return cls(**vals)

    def candidate(self, payload: dict) -> dict:
        """The candidate fields of one stored record (JSON-safe; text as a string)."""
        out: dict[str, Any] = {}
        for name in CANDIDATE_FIELDS:
            key = getattr(self, name)
            v = payload.get(key) if key else None
            if name == "text":
                v = "" if v is None else v if isinstance(v, str) else str(v)
            elif name in ("doc_id", "title") and v is not None and not isinstance(v, str):
                v = str(v)
            out[name] = json_safe(v)
        return out


def cosine(v: Any, q: np.ndarray) -> float:
    """Cosine similarity of a stored vector and a query vector (NaN when either is missing or all zero)."""
    if v is None:
        return float("nan")
    a = np.asarray(v, dtype=np.float32).reshape(-1)
    if a.shape != q.shape:
        return float("nan")
    na, nq = float(np.linalg.norm(a)), float(np.linalg.norm(q))
    if na < 1e-12 or nq < 1e-12:
        return float("nan") if nq < 1e-12 else 0.0
    return float(a @ q / (na * nq))


def as_query(vector: Any, dim: int) -> np.ndarray:
    q = np.asarray(vector, dtype=np.float32).reshape(-1)
    if q.shape[0] != dim:
        raise ValueError(f"query vector has dimension {q.shape[0]}, the store holds {dim}")
    if not np.all(np.isfinite(q)):
        raise ValueError("query vector has NaN or infinite values")
    return q


def dim_mismatch(connector: str, target: str, have: int, want: int) -> ContractError:
    """The store's vectors are not the encoder's: searching them with this encoder's queries would be meaningless."""
    return ContractError(
        f"the {connector} {target} holds {have}-dimensional vectors, but the configured encoder makes {want}-"
        f"dimensional ones. Query vectors come from open-rlcd's encoder, so the store must have been embedded with "
        f"that same model: refusing to search it. Use a collection / table embedded with the encoder in "
        f"pack_config.encoder, or pass candidates in the request instead.", kind="encoder_mismatch",
        connector=connector)
