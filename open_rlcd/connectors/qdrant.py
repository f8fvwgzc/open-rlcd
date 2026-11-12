"""Qdrant as the documents pack's vector store (optional extra `qdrant`: `uv sync --extra qdrant`).

    pack_config:
      vector: {kind: qdrant, url: http://localhost:6333, collection: rlcd_chunks}            # mode: index (default)
      vector: {kind: qdrant, mode: attach, url: http://localhost:6333, collection: kb,      # an existing collection
               text_field: body, doc_id_field: source_id, title_field: heading,
               updated_at_field: modified, authority_field: trust, id_field: null, vector_name: null}

Two modes:

* **index**: open-rlcd owns the collection. `rlcd ingest` creates it if missing (cosine distance, the encoder's
  dimension) and upserts its chunks: point id = uuid5 of the chunk id (Qdrant ids are unsigned integers or UUIDs),
  payload `{chunk_id, doc_id, title, text, updated_at, authority}`. Re-ingesting a changed document deletes its
  old points. An existing collection whose vector size is not the encoder's is refused.
* **attach**: read-only search of a collection that already exists. Nothing is ever created, upserted or deleted
  (`upsert` / `delete` raise before any request is sent). The `*_field` keys map payload keys onto candidate
  fields (`text_field` is required to hold text; `null` means the payload has no such field); `id_field: null`
  means the point id is the chunk id. The collection's vector size must equal the encoder's dimension, otherwise
  the store refuses to search (`encoder_mismatch`). **Query vectors come from open-rlcd's encoder**
  (pack_config.encoder), so the collection must have been embedded with that same model: equal dimensions are
  necessary, not sufficient, and a collection embedded with another model gives meaningless neighbours. When your
  vectors come from another model, pass `candidates` in the request instead.

Scores: `Hit.score` is always a cosine similarity. For a cosine collection that is Qdrant's score; for a dot /
euclidean / manhattan collection (attach mode) the vectors are fetched with the hits and the cosine is computed
here (the order stays Qdrant's).

Hosts and credentials: `url` goes through `security.check_host` (loopback only unless remote hosts are allowed);
an API key is read from the environment variable named by `api_key_env` (unset = no key) and never from
rlcd.yaml; a URL with `user:password@` is refused. `url: ":memory:"` (one in-process store shared by every user of
it in this process, nothing persists) and `path: ./dir` (Qdrant's embedded on-disk mode) never open a socket;
they are for tests, demos and single-process use. Failures become 503 `connector_unavailable` naming the
collection and the exception type, never the URL or the key.

What this does not do: it does not create payload indexes (for `id_field` lookups on a large attached collection,
create a keyword index on that field yourself), and it does not check that an attached collection was embedded
with the same model (only that the dimensions agree).
"""
from __future__ import annotations

import contextlib
import threading
import uuid
from pathlib import Path
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
    cosine,
    dim_mismatch,
    env_name,
    env_secret,
    read_only,
    refuse_inline_secrets,
    require_extra,
    url_has_credentials,
)
from .security import check_host

# fixed namespace: chunk id -> point id is stable across processes and machines
ID_NAMESPACE = uuid.UUID("6f1c2b0e-8a7d-4f43-9c35-2d4a0e5b7c11")
INDEX_FIELDS = FieldMap(id="chunk_id")
KEYS = {"kind", "mode", "url", "path", "collection", "api_key_env", "vector_name", "timeout", "batch_size",
        "id_field", "text_field", "doc_id_field", "title_field", "updated_at_field", "authority_field"}
MODES = ("index", "attach")

_EMBEDDED: dict[str, Any] = {}          # ":memory:" / resolved path -> the one embedded client of this process
_EMBEDDED_LOCK = threading.Lock()
_EMBEDDED_CALLS = threading.RLock()     # the embedded engine is not thread-safe: one call at a time


def point_id(chunk_id: str) -> str:
    """The Qdrant point id of a chunk id in index mode (uuid5, so any string id fits)."""
    return str(uuid.uuid5(ID_NAMESPACE, str(chunk_id)))


def _as_point_id(s: str) -> int | str | None:
    """A chunk id read as a Qdrant point id (attach mode without id_field): unsigned integer or UUID, else None."""
    if s.isdigit():
        return int(s)
    try:
        return str(uuid.UUID(s))
    except ValueError:
        return None


class QdrantVectorStore:
    """A `VectorStore` on a Qdrant collection (see the module docstring for the modes, fields and security)."""

    backend = "qdrant"

    def __init__(self, collection: str, dim: int, *, url: str | None = None, path: str | None = None,
                 mode: str = "index", api_key_env: str | None = None, fields: dict | None = None,
                 vector_name: str | None = None, timeout: int = 10, batch_size: int = 256,
                 allow_remote: bool = False):
        if not isinstance(collection, str) or not collection.strip():
            raise config_error("the qdrant connector needs `collection` (the collection to search / write)")
        self.collection = collection.strip()
        self.dim = int(dim)
        self.mode = str(mode or "index").lower()
        if self.mode not in MODES:
            raise config_error(f"qdrant mode must be one of {MODES}, got {mode!r}")
        if bool(url) == bool(path):
            raise config_error("the qdrant connector needs exactly one of `url` (http://localhost:6333, or "
                               "\":memory:\") and `path` (embedded on-disk mode)")
        if url == ":memory:":
            self._where: tuple[str, str] = ("embedded", ":memory:")
        elif url:
            if not isinstance(url, str):
                raise config_error("qdrant url must be a string")
            if url_has_credentials(url):
                raise config_error("the qdrant url carries credentials: rlcd.yaml never holds credentials; put the "
                                   "API key in an environment variable and name it with api_key_env")
            try:
                check_host(url, allow_remote=allow_remote)
            except ValueError as e:
                raise config_error(f"qdrant: {e}") from None
            self._where = ("url", url)
        else:
            self._where = ("embedded", str(Path(str(path)).expanduser().resolve()))
        self._api_key_env = env_name(api_key_env, "api_key_env")
        self.fields = INDEX_FIELDS if self.mode == "index" else FieldMap.from_config(fields or {},
                                                                                    FieldMap(id=None))
        self.vector_name = vector_name or None
        self.timeout = int(timeout)
        self.batch_size = max(1, int(batch_size))
        self._client: Any = None
        self._models: Any = None
        self._cosine = True
        self._ready = False
        self._lock = threading.RLock()

    @classmethod
    def from_config(cls, cfg: dict, dim: int, allow_remote: bool = False) -> QdrantVectorStore:
        refuse_inline_secrets(cfg, "pack_config.vector")
        check_keys(cfg, KEYS, "pack_config.vector (qdrant)")
        return cls(cfg.get("collection"), dim, url=cfg.get("url"), path=cfg.get("path"),
                   mode=cfg.get("mode") or "index", api_key_env=cfg.get("api_key_env"),
                   fields={k: v for k, v in cfg.items() if k.endswith("_field")},
                   vector_name=cfg.get("vector_name"), timeout=int(cfg.get("timeout") or 10),
                   batch_size=int(cfg.get("batch_size") or 256), allow_remote=allow_remote)

    @classmethod
    def check_config(cls, cfg: dict) -> None:
        """What `from_config` refuses, checked offline for `rlcd config check` (factory.check_store): building a
        store opens no connection and imports no client library, and the dimension only matters once connected.
        Any host passes here; the pack builds the store with the project's `allow_remote`, which decides then."""
        cls.from_config(cfg, 1, allow_remote=True)

    @property
    def target(self) -> str:
        return f"collection {self.collection!r}"

    def describe(self) -> dict:
        """What the corpus meta records about this store (no URL, no key)."""
        return {"kind": "qdrant", "collection": self.collection}

    # ------------------------------------------------------------------ connection

    def _secrets(self) -> tuple[str | None, ...]:
        return (env_secret(self._api_key_env),)

    def _make_client(self) -> Any:
        qc = require_extra("qdrant_client", "qdrant", "qdrant")
        kind, where = self._where
        if kind == "url":
            return qc.QdrantClient(url=where, api_key=env_secret(self._api_key_env), timeout=self.timeout,
                                   check_compatibility=False)
        with _EMBEDDED_LOCK:            # embedded mode allows one client per storage in a process: share it
            client = _EMBEDDED.get(where)
            if client is None:
                client = qc.QdrantClient(location=":memory:") if where == ":memory:" else \
                    qc.QdrantClient(path=where, force_disable_check_same_thread=True)
                _EMBEDDED[where] = client
            return client

    def _vector_params(self, info: Any) -> tuple[int, Any]:
        vectors = info.config.params.vectors
        if isinstance(vectors, dict):
            if self.vector_name is None:
                if len(vectors) != 1:
                    raise config_error(f"qdrant {self.target} has named vectors {sorted(vectors)}: set vector_name")
                self.vector_name = next(iter(vectors))
            if self.vector_name not in vectors:
                raise config_error(f"qdrant {self.target} has no vector named {self.vector_name!r}; it has "
                                   f"{sorted(vectors)}")
            vp = vectors[self.vector_name]
        else:
            if self.vector_name is not None:
                raise config_error(f"qdrant {self.target} has one unnamed vector: remove vector_name")
            vp = vectors
        return int(vp.size), vp.distance

    def connect(self) -> None:
        """Open the client and check the collection (index: create it if missing; attach: it must exist). Raises
        ContractError: 503 `connector_unavailable` when Qdrant cannot be reached, `encoder_mismatch` when the
        vector size is not the encoder's dimension, `connector_config` for a missing collection in attach mode."""
        with self._lock:
            if self._ready:
                return
            client = self._make_client()
            models = require_extra("qdrant_client.models", "qdrant", "qdrant")
            embedded = self._where[0] == "embedded"
            try:
                with _EMBEDDED_CALLS if embedded else contextlib.nullcontext():
                    if not client.collection_exists(self.collection):
                        if self.mode == "attach":
                            raise config_error(f"qdrant {self.target} does not exist; attach mode searches an "
                                               f"existing collection and never creates one")
                        vp = models.VectorParams(size=self.dim, distance=models.Distance.COSINE)
                        client.create_collection(self.collection,
                                                 vectors_config={self.vector_name: vp} if self.vector_name else vp)
                    size, distance = self._vector_params(client.get_collection(self.collection))
                if size != self.dim:
                    raise dim_mismatch("qdrant", self.target, size, self.dim)
                cosine_distance = distance == models.Distance.COSINE
                if self.mode == "index" and not cosine_distance:
                    raise config_error(f"qdrant {self.target} uses {distance} distance; index mode writes to a cosine "
                                       f"collection of its own: use another collection name")
            except Exception as e:
                if not embedded:
                    client.close()
                if isinstance(e, ContractError):
                    raise
                raise self._failure(e) from None
            self._cosine = cosine_distance
            self._client, self._models, self._ready = client, models, True

    def _failure(self, e: BaseException) -> ContractError:
        hint = ""
        status = getattr(e, "status_code", None)
        if status in (401, 403):
            hint = f"unauthorised: is {self._api_key_env or 'api_key_env'} set to a valid API key?"
        return connector_failure("qdrant", self.target, e, self._secrets(), hint)

    def _call(self, fn: Any) -> Any:
        """Run `fn(client, models)`; failures become 503s. The HTTP client is thread-safe, so concurrent requests
        run concurrently; calls into the embedded engine (shared by every store of this process) are serialised."""
        self.connect()
        client, models = self._client, self._models
        with _EMBEDDED_CALLS if self._where[0] == "embedded" else contextlib.nullcontext():
            try:
                return fn(client, models)
            except ContractError:
                raise
            except Exception as e:
                raise self._failure(e) from None

    def close(self) -> None:
        with self._lock:
            if self._client is not None and self._where[0] == "url":
                self._client.close()
            self._client, self._ready = None, False

    # ------------------------------------------------------------------ reads

    def _filter(self, flt: dict | None) -> Any:
        if not flt:
            return None
        m = self._models
        must = []
        for k, v in flt.items():
            if isinstance(v, (list, tuple, set)):
                must.append(m.FieldCondition(key=str(k), match=m.MatchAny(any=list(v))))
            else:
                must.append(m.FieldCondition(key=str(k), match=m.MatchValue(value=v)))
        return m.Filter(must=must)

    def _vec(self, v: Any) -> np.ndarray | None:
        if isinstance(v, dict):
            v = v.get(self.vector_name) if self.vector_name else next(iter(v.values()), None)
        return None if v is None else np.asarray(v, dtype=np.float32)

    def _chunk_id(self, pid: Any, payload: dict) -> str | None:
        if self.fields.id is None:
            return str(pid)
        v = payload.get(self.fields.id)
        return None if v is None or v == "" else str(v)

    def search(self, vector: np.ndarray, k: int = 10, filter: dict | None = None) -> list[Hit]:
        k = int(k)
        q = as_query(vector, self.dim)
        if k <= 0:
            return []

        def run(client: Any, models: Any) -> Any:
            return client.query_points(collection_name=self.collection, query=q.tolist(), using=self.vector_name,
                                       query_filter=self._filter(filter), limit=k, with_payload=True,
                                       with_vectors=not self._cosine)

        hits = []
        for p in self._call(run).points:
            payload = p.payload or {}
            cid = self._chunk_id(p.id, payload)
            if cid is None:
                continue
            score = float(p.score) if self._cosine else cosine(self._vec(p.vector), q)
            hits.append(Hit(cid, score, self.fields.candidate(payload)))
        return hits

    def fetch(self, ids: list[str]) -> dict[str, tuple[np.ndarray | None, dict]]:
        """Stored vectors and candidate fields of the given chunk ids (unknown ids are absent from the result)."""
        ids = list(dict.fromkeys(str(i) for i in ids))
        if not ids:
            return {}
        with_vectors: Any = [self.vector_name] if self.vector_name else True

        def run(client: Any, models: Any) -> list:
            if self.mode == "index":
                return client.retrieve(self.collection, ids=[point_id(i) for i in ids], with_payload=True,
                                       with_vectors=with_vectors)
            if self.fields.id is None:
                pids = [p for p in (_as_point_id(i) for i in ids) if p is not None]
                return client.retrieve(self.collection, ids=pids, with_payload=True,
                                       with_vectors=with_vectors) if pids else []
            nums = [int(i) for i in ids if i.isdigit()]
            should = [models.FieldCondition(key=self.fields.id, match=models.MatchAny(any=ids))]
            if nums:
                should.append(models.FieldCondition(key=self.fields.id, match=models.MatchAny(any=nums)))
            pts, _ = client.scroll(self.collection, scroll_filter=models.Filter(should=should), limit=2 * len(ids),
                                   with_payload=True, with_vectors=with_vectors)
            return pts

        want = set(ids)
        out: dict[str, tuple[np.ndarray | None, dict]] = {}
        for p in self._call(run):
            payload = p.payload or {}
            cid = self._chunk_id(p.id, payload)
            if cid in want and cid not in out:
                out[cid] = (self._vec(p.vector), self.fields.candidate(payload))
        return out

    def vectors(self, ids: list[str]) -> np.ndarray:
        """Stored vectors for `ids`, L2-normalised; unknown ids give NaN rows (as `LocalVectorStore.vectors`)."""
        out = np.full((len(ids), self.dim), np.nan, dtype=np.float32)
        got = self.fetch(ids)
        for r, i in enumerate(ids):
            v = got.get(str(i), (None, {}))[0]
            if v is not None and v.shape == (self.dim,):
                n = float(np.linalg.norm(v))
                out[r] = v / n if n > 1e-12 else 0.0
        return out

    def count(self) -> int:
        return int(self._call(lambda c, m: c.count(self.collection, exact=self.mode == "index")).count)

    # ------------------------------------------------------------------ writes (index mode only)

    def _writable(self, what: str) -> None:
        if self.mode != "index":
            raise read_only(f"qdrant {self.target} is attached read-only (mode: attach): open-rlcd never {what} "
                            f"it. To let open-rlcd index documents, give it a collection of its own with mode: index")

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
        if len(payloads) != len(ids):
            raise ValueError("ids and payloads differ in length")
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate ids in one upsert")

        def run(client: Any, models: Any) -> None:
            for s in range(0, len(ids), self.batch_size):
                points = []
                for i, v, p in zip(ids[s:s + self.batch_size], vecs[s:s + self.batch_size],
                                   payloads[s:s + self.batch_size], strict=True):
                    vec: Any = {self.vector_name: v.tolist()} if self.vector_name else v.tolist()
                    points.append(models.PointStruct(id=point_id(i), vector=vec, payload={**p, "chunk_id": i}))
                client.upsert(self.collection, points=points, wait=True)

        self._call(run)

    def delete(self, ids: list[str]) -> int:
        self._writable("deletes from")
        ids = [str(i) for i in ids]
        if not ids:
            return 0
        self._call(lambda c, m: c.delete(self.collection, points_selector=m.PointIdsList(
            points=[point_id(i) for i in ids]), wait=True))
        return len(ids)

    def save(self) -> None:
        """Nothing to save: Qdrant persists every write itself."""
