"""A local vector store: cosine search over L2-normalised float32 vectors, persisted without pickle.

Backends, in order of preference for `backend="auto"`:

* `faiss`   - `IndexFlatIP` (exact); `IndexHNSWFlat` (approximate, M=32) once the store holds more than
  50,000 vectors;
* `hnswlib` - an HNSW graph (approximate);
* `numpy`   - exact inner product with `argpartition`.

The vectors themselves always live in a numpy array (the source of truth); the ANN index is derived from it and
rebuilt on load or after deletes / overwrites. Searching with a `filter` (payload equality, a list value meaning
"one of") is an exact scan over the matching rows, whatever the backend.

On disk (`path` is a directory): `vectors.npy` (`np.save(allow_pickle=False)`), `items.jsonl` (id + payload per
line, same order), `meta.json`. No index file is written, so nothing loaded can execute code. This is for local
corpora (up to a few million vectors on a laptop with faiss); it is not a database: calls within one process are
serialised by a lock, but there is one writer process at a time and no transactions.
"""
from __future__ import annotations

import importlib
import json
import os
import threading
from pathlib import Path
from typing import Any

import numpy as np

from .base import ConnectorError, Hit

FORMAT = "open-rlcd-vectors/1"
HNSW_THRESHOLD = 50_000
BACKENDS = ("faiss", "hnswlib", "numpy")


def _importable(name: str) -> Any | None:
    try:
        return importlib.import_module(name)
    except ImportError:
        return None


def _normalize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return np.ascontiguousarray(np.where(n > 1e-12, x / np.maximum(n, 1e-12), 0.0), dtype=np.float32)


def _matches(payload: dict, flt: dict) -> bool:
    for k, v in flt.items():
        have = payload.get(k)
        if isinstance(v, (list, tuple, set)):
            if have not in v:
                return False
        elif have != v:
            return False
    return True


class LocalVectorStore:
    """Cosine-similarity vector store on the local disk (see the module docstring for backends and format)."""

    def __init__(self, path: Path, dim: int, backend: str = "auto"):
        self.path = Path(path)
        self.dim = int(dim)
        if self.dim <= 0:
            raise ValueError("dim must be positive")
        self._backend = self._choose(backend)
        self._vecs = np.zeros((0, self.dim), dtype=np.float32)
        self._ids: list[str] = []
        self._payloads: list[dict] = []
        self._pos: dict[str, int] = {}
        self._index: Any = None
        self._indexed = 0          # rows already in the ANN index (appends are added incrementally)
        self._index_kind = ""
        self._lock = threading.RLock()   # the API answers from a thread pool; the index is built lazily

    @staticmethod
    def _choose(backend: str) -> str:
        backend = (backend or "auto").lower()
        if backend == "auto":
            for name in ("faiss", "hnswlib"):
                if _importable(name) is not None:
                    return name
            return "numpy"
        if backend not in BACKENDS:
            raise ValueError(f"unknown vector backend {backend!r}; use one of {BACKENDS} or 'auto'")
        if backend != "numpy" and _importable(backend) is None:
            raise ConnectorError(f"vector backend {backend!r} is not installed; install the 'documents' extra "
                                 f"(`uv sync --extra documents`) or use backend 'numpy'")
        return backend

    @property
    def backend(self) -> str:
        return self._backend

    def count(self) -> int:
        return len(self._ids)

    def __len__(self) -> int:
        return len(self._ids)

    def __contains__(self, item: object) -> bool:
        return item in self._pos

    def ids(self) -> list[str]:
        return list(self._ids)

    def payload(self, id: str) -> dict | None:
        i = self._pos.get(id)
        return None if i is None else dict(self._payloads[i])

    def vector(self, id: str) -> np.ndarray | None:
        i = self._pos.get(id)
        return None if i is None else self._vecs[i].copy()

    def vectors(self, ids: list[str]) -> np.ndarray:
        """Stored (normalised) vectors for `ids`; unknown ids give NaN rows."""
        out = np.full((len(ids), self.dim), np.nan, dtype=np.float32)
        for r, i in enumerate(ids):
            p = self._pos.get(i)
            if p is not None:
                out[r] = self._vecs[p]
        return out

    # ------------------------------------------------------------------ writes

    def upsert(self, ids: list[str], vectors: np.ndarray, payloads: list[dict] | None = None) -> None:
        with self._lock:
            self._upsert(ids, vectors, payloads)

    def _upsert(self, ids: list[str], vectors: np.ndarray, payloads: list[dict] | None) -> None:
        ids = [str(i) for i in ids]
        vecs = np.asarray(vectors, dtype=np.float32)
        if vecs.ndim == 1:
            vecs = vecs[None, :]
        if vecs.shape != (len(ids), self.dim):
            raise ValueError(f"vectors must have shape ({len(ids)}, {self.dim}), got {vecs.shape}")
        payloads = [dict(p or {}) for p in (payloads if payloads is not None else [{}] * len(ids))]
        if len(payloads) != len(ids):
            raise ValueError("ids and payloads differ in length")
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate ids in one upsert")
        vecs = _normalize(vecs)
        new_rows, new_ids, new_payloads = [], [], []
        for i, v, p in zip(ids, vecs, payloads, strict=True):
            pos = self._pos.get(i)
            if pos is None:
                new_rows.append(v)
                new_ids.append(i)
                new_payloads.append(p)
            else:
                self._vecs[pos] = v
                self._payloads[pos] = p
                self._invalidate()
        if new_ids:
            start = len(self._ids)
            self._vecs = np.concatenate([self._vecs, np.asarray(new_rows, dtype=np.float32)], axis=0)
            self._ids.extend(new_ids)
            self._payloads.extend(new_payloads)
            for j, i in enumerate(new_ids):
                self._pos[i] = start + j

    def delete(self, ids: list[str]) -> int:
        with self._lock:
            return self._delete(ids)

    def _delete(self, ids: list[str]) -> int:
        drop = {self._pos[i] for i in ids if i in self._pos}
        if not drop:
            return 0
        keep = [r for r in range(len(self._ids)) if r not in drop]
        self._vecs = np.ascontiguousarray(self._vecs[keep])
        self._ids = [self._ids[r] for r in keep]
        self._payloads = [self._payloads[r] for r in keep]
        self._pos = {i: r for r, i in enumerate(self._ids)}
        self._invalidate()
        return len(drop)

    def _invalidate(self) -> None:
        self._index, self._indexed, self._index_kind = None, 0, ""

    # ------------------------------------------------------------------ index

    def _want_kind(self) -> str:
        if self._backend == "faiss":
            return "faiss-hnsw" if len(self._ids) > HNSW_THRESHOLD else "faiss-flat"
        return self._backend

    def _ensure_index(self) -> None:
        n = len(self._ids)
        kind = self._want_kind()
        if kind == "numpy":
            return
        if self._index is not None and self._index_kind == kind and self._indexed == n:
            return
        if self._index is None or self._index_kind != kind or kind == "hnswlib":
            self._index, self._indexed, self._index_kind = self._new_index(kind, n), 0, kind
        new = self._vecs[self._indexed:n]
        if len(new):
            if kind == "hnswlib":
                self._index.add_items(new, np.arange(self._indexed, n))
            else:
                self._index.add(new)
        self._indexed = n

    def _new_index(self, kind: str, n: int) -> Any:
        if kind == "faiss-flat":
            import faiss

            return faiss.IndexFlatIP(self.dim)
        if kind == "faiss-hnsw":
            import faiss

            idx = faiss.IndexHNSWFlat(self.dim, 32, faiss.METRIC_INNER_PRODUCT)
            idx.hnsw.efConstruction = 200
            return idx
        import hnswlib

        idx = hnswlib.Index(space="ip", dim=self.dim)
        idx.init_index(max_elements=max(n, 1), ef_construction=200, M=16)
        return idx

    # ------------------------------------------------------------------ search

    def search(self, vector: np.ndarray, k: int = 10, filter: dict | None = None) -> list[Hit]:
        with self._lock:
            return self._search(vector, k, filter)

    def _search(self, vector: np.ndarray, k: int, filter: dict | None) -> list[Hit]:
        n = len(self._ids)
        k = int(k)
        if n == 0 or k <= 0:
            return []
        q = _normalize(np.asarray(vector, dtype=np.float32).reshape(-1))
        if q.shape[0] != self.dim:
            raise ValueError(f"query vector has dimension {q.shape[0]}, the store holds {self.dim}")
        if filter:
            rows = np.asarray([r for r, p in enumerate(self._payloads) if _matches(p, filter)], dtype=np.int64)
            if rows.size == 0:
                return []
            return self._exact(q, k, rows)
        kind = self._want_kind()
        if kind == "numpy":
            return self._exact(q, k, None)
        self._ensure_index()
        kk = min(k, n)
        if kind == "hnswlib":
            self._index.set_ef(max(64, 2 * kk))
            labels, dist = self._index.knn_query(q[None, :], k=kk)
            pairs = [(int(r), 1.0 - float(d)) for r, d in zip(labels[0], dist[0], strict=True)]
        else:
            if kind == "faiss-hnsw":
                self._index.hnsw.efSearch = max(64, 2 * kk)
            scores, rows = self._index.search(q[None, :], kk)
            pairs = [(int(r), float(s)) for r, s in zip(rows[0], scores[0], strict=True) if r >= 0]
        pairs.sort(key=lambda t: (-t[1], t[0]))
        return [Hit(self._ids[r], s, dict(self._payloads[r])) for r, s in pairs]

    def _exact(self, q: np.ndarray, k: int, rows: np.ndarray | None) -> list[Hit]:
        mat = self._vecs if rows is None else self._vecs[rows]
        scores = mat @ q
        kk = min(k, len(scores))
        if kk < len(scores):
            cand = np.argpartition(-scores, kk - 1)[:kk]
        else:
            cand = np.arange(len(scores))
        order = cand[np.lexsort((cand, -scores[cand]))]
        out = []
        for j in order:
            r = int(j if rows is None else rows[j])
            out.append(Hit(self._ids[r], float(scores[j]), dict(self._payloads[r])))
        return out

    # ------------------------------------------------------------------ persistence

    def save(self) -> Path:
        with self._lock:
            return self._save()

    def _save(self) -> Path:
        self.path.mkdir(parents=True, exist_ok=True)
        tmp_vec = self.path / "vectors.npy.tmp"
        with open(tmp_vec, "wb") as f:
            np.save(f, self._vecs, allow_pickle=False)
        tmp_items = self.path / "items.jsonl.tmp"
        with open(tmp_items, "w", encoding="utf-8") as f:
            for i, p in zip(self._ids, self._payloads, strict=True):
                f.write(json.dumps({"id": i, "payload": p}, ensure_ascii=False) + "\n")
        tmp_meta = self.path / "meta.json.tmp"
        tmp_meta.write_text(json.dumps({"format": FORMAT, "dim": self.dim, "count": len(self._ids),
                                        "metric": "cosine", "backend": self._backend}, indent=1), encoding="utf-8")
        os.replace(tmp_vec, self.path / "vectors.npy")
        os.replace(tmp_items, self.path / "items.jsonl")
        os.replace(tmp_meta, self.path / "meta.json")
        return self.path

    @classmethod
    def load(cls, path: Path, backend: str = "auto") -> LocalVectorStore:
        path = Path(path)
        meta_path = path / "meta.json"
        if not meta_path.is_file():
            raise ConnectorError(f"no vector store at {path} (meta.json missing)")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("format") != FORMAT:
            raise ConnectorError(f"{path}: unknown vector store format {meta.get('format')!r}")
        store = cls(path, int(meta["dim"]), backend=backend)
        vecs = np.load(path / "vectors.npy", allow_pickle=False)
        ids, payloads = [], []
        with open(path / "items.jsonl", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    ids.append(str(rec["id"]))
                    payloads.append(dict(rec.get("payload") or {}))
        if vecs.ndim != 2 or vecs.shape != (len(ids), store.dim) or len(ids) != int(meta.get("count", len(ids))):
            raise ConnectorError(f"{path}: vectors {vecs.shape} do not match {len(ids)} items of dimension "
                                 f"{store.dim}")
        store._vecs = np.ascontiguousarray(vecs, dtype=np.float32)
        store._ids, store._payloads = ids, payloads
        store._pos = {i: r for r, i in enumerate(ids)}
        return store
