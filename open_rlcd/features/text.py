"""Text encoders: turn strings into fixed-size, L2-normalised float32 vectors.

Two encoders ship:

* `HashingEncoder` - offline and deterministic: signed feature hashing of word unigrams and character 2-4 grams
  (within words) after NFKC + casefold. No download, no model, works for any script. It captures surface
  overlap only (shared words / character runs), not meaning: "car" and "automobile" are unrelated to it. Tests
  and the demo bundle use it.
* `OnnxEncoder` - a frozen multilingual sentence encoder (default `intfloat/multilingual-e5-small`, MIT) run
  through ONNX Runtime on CPU. It is downloaded once from the Hugging Face Hub into `<home>/encoders` and
  never fine-tuned here.

`CachedEncoder` wraps either with an on-disk sqlite cache keyed by content hash (float32 bytes, no pickle), so
re-ingesting or re-featurising the same text costs a lookup. `make_encoder` / `encoder_config` build an encoder
from config and back: a bundle stores `encoder_config(enc)` (including the sha256 of the model file actually
used) so it is never served with a different encoder by accident.

What this module does not do: it does not fine-tune, does not run on GPU, and does not download anything unless
an `OnnxEncoder` is constructed while online.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import threading
import unicodedata
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np

DEFAULT_MODEL_ID = "intfloat/multilingual-e5-small"
DEFAULT_ONNX_FILE = "onnx/model_qint8_avx512_vnni.onnx"
EXTRA_HINT = ("install the 'documents' extra: `uv sync --extra documents` (or `pip install 'open-rlcd[documents]'`)")
_WORD = re.compile(r"[^\W_]+")


class EncoderError(RuntimeError):
    """An encoder could not be built or used (missing model file, missing dependency, wrong hash)."""


class EncoderUnavailable(EncoderError):
    """The model is not available locally and may not be downloaded (offline)."""


class EncoderMismatch(EncoderError):
    """The model file on disk is not the one a bundle / corpus was built with."""


@runtime_checkable
class Encoder(Protocol):
    id: str
    dim: int
    sha256: str

    def encode(self, texts: list[str], kind: str = "passage") -> np.ndarray: ...


# ----------------------------------------------------------------------------- maths helpers

def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Row-wise L2 normalisation as float32. All-zero rows stay all-zero (there is no direction to keep)."""
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 1:
        n = float(np.linalg.norm(x))
        return x / n if n > eps else x.copy()
    norms = np.linalg.norm(x, axis=-1, keepdims=True)
    return np.where(norms > eps, x / np.maximum(norms, eps), 0.0).astype(np.float32)


def mean_pool(last_hidden: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Average token vectors over real (mask = 1) tokens: (batch, seq, dim) x (batch, seq) -> (batch, dim).

    Padding tokens contribute nothing; a row with no real tokens comes out all-zero instead of NaN."""
    h = np.asarray(last_hidden, dtype=np.float32)
    m = np.asarray(mask, dtype=np.float32)[..., None]
    summed = (h * m).sum(axis=1)
    counts = np.clip(m.sum(axis=1), 1e-9, None)
    return (summed / counts).astype(np.float32)


def normalize_text(text: str) -> str:
    """NFKC + casefold: full-width / half-width forms and case variants map to one form in every script."""
    return unicodedata.normalize("NFKC", str(text)).casefold()


def _sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


# ----------------------------------------------------------------------------- hashing encoder

class HashingEncoder:
    """Signed feature hashing of word unigrams + character 2-4 grams within words (NFKC + casefold).

    Deterministic across processes and machines: it uses blake2b, not Python's salted `hash()`. Word unigrams
    weigh 1, each character n-gram 0.5; counts are dampened with log1p before hashing collisions are summed, so
    one repeated word cannot dominate. CJK text without spaces is one "word" per run, so its character n-grams
    carry the signal. `kind` is ignored: queries and passages share one space.
    """

    VERSION = 1
    NGRAMS = (2, 4)

    def __init__(self, dim: int = 384, seed: int = 0):
        if int(dim) < 8:
            raise ValueError("HashingEncoder dim must be >= 8")
        self.dim, self.seed = int(dim), int(seed)
        self.id = f"hashing-{self.dim}-s{self.seed}"
        self.sha256 = _sha256_bytes(json.dumps(self.config(), sort_keys=True).encode())
        self._key = self.seed.to_bytes(8, "little", signed=True)
        self._cache: dict[str, tuple[int, float]] = {}

    def config(self) -> dict:
        return {"kind": "hashing", "dim": self.dim, "seed": self.seed, "version": self.VERSION,
                "ngrams": list(self.NGRAMS), "norm": "nfkc+casefold"}

    def _slot(self, feature: str) -> tuple[int, float]:
        hit = self._cache.get(feature)
        if hit is None:
            d = hashlib.blake2b(feature.encode("utf-8"), digest_size=8, key=self._key).digest()
            hit = (int.from_bytes(d[:6], "little") % self.dim, 1.0 if d[7] & 1 else -1.0)
            if len(self._cache) > 2_000_000:
                self._cache.clear()
            self._cache[feature] = hit
        return hit

    def _features(self, text: str) -> dict[str, float]:
        feats: dict[str, float] = {}
        lo, hi = self.NGRAMS
        for w in _WORD.findall(normalize_text(text)):
            feats["w\x01" + w] = feats.get("w\x01" + w, 0.0) + 1.0
            padded = f"<{w}>"
            for n in range(lo, hi + 1):
                for i in range(len(padded) - n + 1):
                    g = "c\x01" + padded[i:i + n]
                    feats[g] = feats.get(g, 0.0) + 0.5
        return feats

    def encode(self, texts: list[str], kind: str = "passage") -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for r, t in enumerate(texts):
            feats = self._features(t)
            if not feats:
                continue
            idx = np.empty(len(feats), dtype=np.int64)
            val = np.empty(len(feats), dtype=np.float64)
            for j, (f, c) in enumerate(feats.items()):
                slot, sign = self._slot(f)
                idx[j], val[j] = slot, sign * np.log1p(c)
            out[r] = np.bincount(idx, weights=val, minlength=self.dim)
        return l2_normalize(out)


# ----------------------------------------------------------------------------- ONNX encoder

def _offline() -> bool:
    return os.environ.get("HF_HUB_OFFLINE", "").strip().lower() in ("1", "true", "yes", "on")


def _lazy(module: str) -> Any:
    import importlib

    try:
        return importlib.import_module(module)
    except ImportError as e:
        raise EncoderError(f"the ONNX encoder needs '{module}', which is not installed: {EXTRA_HINT}. "
                           f"Or use the offline hashing encoder (encoder: {{kind: hashing}}).") from e


class OnnxEncoder:
    """A frozen sentence encoder run with ONNX Runtime on CPU (mean pooling + L2 normalisation).

    The model file and `tokenizer.json` come from the Hugging Face Hub (`hf_hub_download`) into
    `<home>/encoders` (the Hub's default cache when `home` is None). With `HF_HUB_OFFLINE=1` nothing is
    downloaded: a model that is not cached raises `EncoderUnavailable` with instructions. `sha256` is the hash
    of the ONNX file actually loaded; when `expected_sha256` is given and differs, construction fails, because
    a head trained on one encoder's vectors is meaningless on another's.

    `quantize_locally=True` downloads the fp32 `onnx/model.onnx` and writes an int8 copy with
    `onnxruntime.quantization.quantize_dynamic` (that needs the `onnx` package too). e5 models get their
    "query: " / "passage: " prefixes; other models get none.
    """

    def __init__(self, model_id: str = DEFAULT_MODEL_ID, file: str = DEFAULT_ONNX_FILE, home: Path | None = None,
                 max_length: int = 512, batch_size: int = 32, quantize_locally: bool = False,
                 expected_sha256: str | None = None):
        self.model_id, self.max_length, self.batch_size = model_id, int(max_length), max(1, int(batch_size))
        self.quantize_locally = bool(quantize_locally)
        self.file = "onnx/model.onnx" if self.quantize_locally else file
        self.cache_dir = (Path(home).expanduser() / "encoders") if home is not None else None
        self.prefixes = ("query: ", "passage: ") if "e5" in model_id.lower() else ("", "")
        ort = _lazy("onnxruntime")
        tokenizers = _lazy("tokenizers")
        model_path = self._download(self.file)
        if self.quantize_locally:
            model_path = self._quantize(model_path)
        tok_path = self._download("tokenizer.json")
        self.model_path = model_path
        self.sha256 = file_sha256(model_path)
        if expected_sha256 and expected_sha256 != self.sha256:
            raise EncoderMismatch(
                f"encoder file {model_path} has sha256 {self.sha256[:16]}..., but this bundle / corpus was built "
                f"with {expected_sha256[:16]}...: a model trained on one encoder's vectors must not be served "
                f"with another. Use the same model file, or retrain / re-ingest with this one.")
        self.tokenizer = tokenizers.Tokenizer.from_file(str(tok_path))
        self.tokenizer.enable_truncation(max_length=self.max_length)
        pad_id = self.tokenizer.token_to_id("<pad>")
        pad_token = "<pad>"
        if pad_id is None:
            pad_id, pad_token = (self.tokenizer.token_to_id("[PAD]") or 0), "[PAD]"
        self.tokenizer.enable_padding(pad_id=pad_id, pad_token=pad_token)
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = os.cpu_count() or 1
        self.session = ort.InferenceSession(str(model_path), sess_options=opts, providers=["CPUExecutionProvider"])
        self.input_names = [i.name for i in self.session.get_inputs()]
        dim = self.session.get_outputs()[0].shape[-1]
        self.dim = int(dim) if isinstance(dim, int) else int(self._run(["dimension probe"]).shape[1])
        suffix = "#int8-local" if self.quantize_locally else ""
        self.id = f"onnx:{model_id}/{self.file}{suffix}"

    def _download(self, filename: str) -> Path:
        hub = _lazy("huggingface_hub")
        offline = _offline()
        where = str(self.cache_dir) if self.cache_dir else "the Hugging Face cache"
        try:
            return Path(hub.hf_hub_download(repo_id=self.model_id, filename=filename,
                                            cache_dir=str(self.cache_dir) if self.cache_dir else None,
                                            local_files_only=offline))
        except Exception as e:  # the hub raises several types (LocalEntryNotFound, HTTP, OS)
            if offline:
                raise EncoderUnavailable(
                    f"encoder file {self.model_id}/{filename} is not in {where} and HF_HUB_OFFLINE is set, so "
                    f"nothing was downloaded. Run once online (unset HF_HUB_OFFLINE) to download it, or use the "
                    f"offline hashing encoder (pack_config.encoder: {{kind: hashing}}).") from e
            raise EncoderUnavailable(
                f"could not download encoder file {self.model_id}/{filename} into {where}: {e}. Check the network "
                f"and the model id, or use the offline hashing encoder (pack_config.encoder: {{kind: hashing}}).") \
                from e

    def _quantize(self, fp32: Path) -> Path:
        base = self.cache_dir or Path(fp32).parent
        out = base / "quantized" / self.model_id.replace("/", "--") / "model_int8_local.onnx"
        if out.is_file():
            return out
        try:
            import onnx  # noqa: F401  (quantize_dynamic loads the graph with it)
        except ImportError as e:
            raise EncoderError("quantize_locally needs the 'onnx' package, which the 'documents' extra does not "
                               "include: `uv pip install onnx` (or the 'finetune' extra), or keep the default "
                               "pre-quantized file (quantize_locally: false).") from e
        quant = _lazy("onnxruntime.quantization")
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(".tmp.onnx")
        quant.quantize_dynamic(str(fp32), str(tmp), weight_type=quant.QuantType.QInt8)
        os.replace(tmp, out)
        return out

    def _run(self, texts: list[str]) -> np.ndarray:
        encs = self.tokenizer.encode_batch(texts)
        ids = np.asarray([e.ids for e in encs], dtype=np.int64)
        mask = np.asarray([e.attention_mask for e in encs], dtype=np.int64)
        feed: dict[str, np.ndarray] = {}
        for name in self.input_names:
            if name == "input_ids":
                feed[name] = ids
            elif name == "attention_mask":
                feed[name] = mask
            elif name == "token_type_ids":
                feed[name] = np.zeros_like(ids)
            else:
                raise EncoderError(f"encoder {self.model_id} expects an unsupported input {name!r}")
        hidden = self.session.run(None, feed)[0]
        if hidden.ndim == 2:          # already pooled by the export
            return l2_normalize(hidden)
        return l2_normalize(mean_pool(hidden, mask))

    def encode(self, texts: list[str], kind: str = "passage") -> np.ndarray:
        if not texts:
            return np.zeros((0, getattr(self, "dim", 0)), dtype=np.float32)
        prefix = self.prefixes[0] if kind == "query" else self.prefixes[1]
        prepared = [prefix + str(t) for t in texts]
        order = np.argsort([len(t) for t in prepared], kind="stable")   # similar lengths share a batch: less padding
        out = np.zeros((len(prepared), self.dim), dtype=np.float32)
        for s in range(0, len(order), self.batch_size):
            idx = order[s:s + self.batch_size]
            out[idx] = self._run([prepared[i] for i in idx])
        return out


# ----------------------------------------------------------------------------- cache

class CachedEncoder:
    """Any encoder plus a sqlite cache at `cache_dir/<sha256[:16]>.sqlite` (key = sha256(kind NUL text)).

    Only cache misses are sent to the wrapped encoder, in one batch, deduplicated. Vectors are stored as raw
    float32 bytes (no pickle). The file is per encoder hash, so switching encoders never returns stale vectors.
    Safe to share between threads of one process; not designed for many writer processes at once.
    """

    def __init__(self, encoder: Encoder, cache_dir: Path):
        self.inner = encoder
        self.id, self.dim, self.sha256 = encoder.id, encoder.dim, encoder.sha256
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.cache_dir / f"{encoder.sha256[:16]}.sqlite"
        self._lock = threading.Lock()
        self._db = sqlite3.connect(str(self.path), check_same_thread=False)
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("CREATE TABLE IF NOT EXISTS emb (key TEXT PRIMARY KEY, vec BLOB NOT NULL)")
        self._db.commit()
        self.hits = self.misses = 0

    @staticmethod
    def key(text: str, kind: str) -> str:
        return hashlib.sha256(f"{kind}\0{text}".encode()).hexdigest()

    def encode(self, texts: list[str], kind: str = "passage") -> np.ndarray:
        texts = [str(t) for t in texts]
        keys = [self.key(t, kind) for t in texts]
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        found: dict[str, np.ndarray] = {}
        uniq = list(dict.fromkeys(keys))
        with self._lock:
            for s in range(0, len(uniq), 500):
                part = uniq[s:s + 500]
                q = f"SELECT key, vec FROM emb WHERE key IN ({','.join('?' * len(part))})"
                for k, blob in self._db.execute(q, part):
                    v = np.frombuffer(blob, dtype=np.float32)
                    if v.shape[0] == self.dim:
                        found[k] = v
        missing: dict[str, str] = {}
        for k, t in zip(keys, texts, strict=True):
            if k not in found and k not in missing:
                missing[k] = t
        if missing:
            vecs = np.asarray(self.inner.encode(list(missing.values()), kind=kind), dtype=np.float32)
            rows = []
            for k, v in zip(missing, vecs, strict=True):
                found[k] = v
                rows.append((k, np.ascontiguousarray(v, dtype=np.float32).tobytes()))
            with self._lock:
                self._db.executemany("INSERT OR REPLACE INTO emb (key, vec) VALUES (?, ?)", rows)
                self._db.commit()
        self.misses += len(missing)
        self.hits += len(uniq) - len(missing)
        for i, k in enumerate(keys):
            out[i] = found[k]
        return out

    def __len__(self) -> int:
        with self._lock:
            return int(self._db.execute("SELECT COUNT(*) FROM emb").fetchone()[0])

    def close(self) -> None:
        with self._lock:
            self._db.close()


# ----------------------------------------------------------------------------- config round trip

def _unwrap(enc: Encoder) -> Encoder:
    while isinstance(enc, CachedEncoder):
        enc = enc.inner
    return enc


def encoder_config(enc: Encoder) -> dict:
    """JSON description of an encoder; `make_encoder(encoder_config(e), home)` rebuilds an equivalent one and
    refuses a model file whose hash differs."""
    base = _unwrap(enc)
    out: dict[str, Any] = {"id": enc.id, "dim": int(enc.dim), "sha256": enc.sha256}
    if isinstance(base, HashingEncoder):
        return {"kind": "hashing", **out, "seed": base.seed}
    if isinstance(base, OnnxEncoder):
        return {"kind": "onnx", **out, "model_id": base.model_id,
                "file": DEFAULT_ONNX_FILE if base.quantize_locally else base.file,
                "quantize_locally": base.quantize_locally, "max_length": base.max_length}
    return {"kind": "custom", **out}


def make_encoder(config: dict | None, home: Path) -> Encoder:
    """Build an encoder from config and wrap it in a `CachedEncoder` under `<home>/cache/embeddings`.

    `{"kind": "hashing", "dim": 384, "seed": 0}` or `{"kind": "onnx", "model_id", "file", "quantize_locally",
    "sha256": expected or None, "max_length", "batch_size"}`. None means the default ONNX e5-small encoder
    (downloads once). `"cache": false` skips the cache. A `sha256` that does not match the rebuilt encoder
    raises `EncoderMismatch`."""
    cfg = dict(config or {"kind": "onnx"})
    kind = str(cfg.get("kind") or "onnx").lower()
    home = Path(home)
    enc: Encoder
    if kind == "hashing":
        enc = HashingEncoder(dim=int(cfg.get("dim") or 384), seed=int(cfg.get("seed") or 0))
        want = cfg.get("sha256")
        if want and want != enc.sha256:
            raise EncoderMismatch(f"hashing encoder {enc.id} has sha256 {enc.sha256[:16]}..., expected "
                                  f"{str(want)[:16]}... (built by a different encoder version)")
    elif kind == "onnx":
        enc = OnnxEncoder(model_id=str(cfg.get("model_id") or DEFAULT_MODEL_ID),
                          file=str(cfg.get("file") or DEFAULT_ONNX_FILE), home=home,
                          max_length=int(cfg.get("max_length") or 512), batch_size=int(cfg.get("batch_size") or 32),
                          quantize_locally=bool(cfg.get("quantize_locally", False)),
                          expected_sha256=cfg.get("sha256") or None)
    else:
        raise ValueError(f"unknown encoder kind {kind!r}; use 'hashing' (offline) or 'onnx'")
    if cfg.get("cache", True) is False:
        return enc
    return CachedEncoder(enc, home / "cache" / "embeddings")
