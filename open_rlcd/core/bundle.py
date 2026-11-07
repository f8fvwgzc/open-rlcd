"""The open-weight bundle: one trained model version as a directory of plain files.

    <bundle>/
      model.json                      format, version, created_at, pack {name, version, config}, feature_version,
                                      encoder, policy, heads {name: summary + dir + files with sha256}, data_card
      heads/<dir>/ensemble.json       spec, members (kind, params, calibrator JSON)
      heads/<dir>/member_<k>.safetensors | member_<k>.txt     logreg weights | LightGBM native text model
      heads/<dir>/head.json           metrics, verdict, calibration, conformal, data, notes
      heads/<dir>/report.json         the full holdout report
      heads/<dir>/bernoulli.json      the holdout table the Bernoulli ledger restarts from
      MODEL_CARD.md                   (written by `rlcd report` / `rlcd export`)

Loading reads only JSON, safetensors and LightGBM text, so opening a bundle never executes code. `validate`
refuses symlinks, path traversal, unexpected file types, oversized files and files whose hash does not match.
"""
from __future__ import annotations

import hashlib
import json
import re
import shutil
from pathlib import Path
from typing import Any

from .head import CalibratedEnsemble, HeadSpec

FORMAT = "open-rlcd-bundle/1"
ALLOWED_SUFFIXES = {".json", ".safetensors", ".txt", ".md", ".html", ".svg"}
MAX_FILE_BYTES = 1 << 30          # 1 GiB per file
MAX_TOTAL_BYTES = 4 << 30         # 4 GiB per bundle
HEAD_NAME = re.compile(r"^[A-Za-z0-9_.:\-]{1,128}$")


class BundleError(Exception):
    pass


def safe_dir(name: str) -> str:
    if not HEAD_NAME.match(name):
        raise BundleError(f"head name {name!r} must match {HEAD_NAME.pattern}")
    return re.sub(r"[^A-Za-z0-9_.\-]", "_", name)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _dump(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=1, ensure_ascii=False, default=_json_default), encoding="utf-8")


def _json_default(o: Any):
    import numpy as np

    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (set, tuple)):
        return list(o)
    raise TypeError(f"not JSON serialisable: {type(o).__name__}")


def write_head(head_dir: Path, trained: Any) -> dict:
    """Write one freshly trained head (core.training.TrainedHead); returns its model.json entry."""
    head_dir.mkdir(parents=True, exist_ok=True)
    files = trained.ensemble.save(head_dir)
    summary = trained.summary()
    _dump(head_dir / "head.json", summary)
    _dump(head_dir / "report.json", trained.report)
    _dump(head_dir / "bernoulli.json", trained.ledger)
    files += ["head.json", "report.json", "bernoulli.json"]
    return {**summary, "spec": trained.spec.to_dict(), "dir": f"heads/{head_dir.name}",
            "files": {f: sha256(head_dir / f) for f in files}}


def write_bundle(dest: Path, *, version: str, created_at: str, pack: dict, feature_version: str,
                 encoder: dict | None, policy: dict, trained: dict[str, Any],
                 carried: dict[str, tuple[Path, dict]] | None = None, data_card: dict | None = None,
                 notes: list[str] | None = None) -> dict:
    """Write a bundle directory. `trained`: name -> TrainedHead; `carried`: name -> (old head dir, old entry)
    for heads copied unchanged from the previous version."""
    dest = Path(dest)
    if dest.exists():
        raise BundleError(f"{dest} already exists")
    (dest / "heads").mkdir(parents=True)
    heads: dict[str, dict] = {}
    for name, (old_dir, entry) in (carried or {}).items():
        new_dir = dest / "heads" / safe_dir(name)
        shutil.copytree(old_dir, new_dir, symlinks=False)
        heads[name] = {**entry, "dir": f"heads/{new_dir.name}", "carried_over": True}
    for name, th in trained.items():
        heads[name] = write_head(dest / "heads" / safe_dir(name), th)
    meta = {"format": FORMAT, "version": version, "created_at": created_at, "pack": pack,
            "feature_version": feature_version, "encoder": encoder, "policy": policy,
            "retrained": sorted(trained), "heads": heads, "data_card": data_card or {}, "notes": notes or []}
    _dump(dest / "model.json", meta)
    return meta


def read_meta(path: Path) -> dict:
    path = Path(path)
    mj = path / "model.json"
    if not mj.is_file() or mj.is_symlink():
        raise BundleError(f"{path} is not a bundle (no model.json)")
    meta = json.loads(mj.read_text(encoding="utf-8"))
    if meta.get("format") != FORMAT:
        raise BundleError(f"{path}: unknown bundle format {meta.get('format')!r} (expected {FORMAT})")
    return meta


def validate(path: Path, check_hashes: bool = True) -> dict:
    """Check a bundle directory before it is imported or loaded. Returns its model.json."""
    path = Path(path)
    if path.is_symlink() or not path.is_dir():
        raise BundleError(f"{path} is not a directory")
    root = path.resolve()
    total = 0
    for p in path.rglob("*"):
        if p.is_symlink():
            raise BundleError(f"symlink in bundle: {p.relative_to(path)}")
        if not p.resolve().is_relative_to(root):
            raise BundleError(f"path escapes the bundle: {p}")
        if p.is_dir():
            continue
        if p.suffix.lower() not in ALLOWED_SUFFIXES:
            raise BundleError(f"unexpected file type in bundle: {p.relative_to(path)} (allowed: "
                              f"{sorted(ALLOWED_SUFFIXES)}; bundles never contain pickles or code)")
        size = p.stat().st_size
        if size > MAX_FILE_BYTES:
            raise BundleError(f"file too large: {p.relative_to(path)} ({size} bytes)")
        total += size
    if total > MAX_TOTAL_BYTES:
        raise BundleError(f"bundle too large: {total} bytes")
    meta = read_meta(path)
    for name, entry in meta.get("heads", {}).items():
        safe_dir(name)
        d = (path / entry["dir"]).resolve()
        if not d.is_relative_to(root / "heads"):
            raise BundleError(f"head {name!r}: directory escapes the bundle")
        for f, digest in (entry.get("files") or {}).items():
            fp = (d / f).resolve()
            if not fp.is_relative_to(d) or not fp.is_file():
                raise BundleError(f"head {name!r}: missing or invalid file {f!r}")
            if check_hashes and sha256(fp) != digest:
                raise BundleError(f"head {name!r}: {f} does not match its sha256 in model.json")
    return meta


class LoadedHead:
    """A head of a bundle, loaded lazily: the ensemble is read on first use."""

    def __init__(self, root: Path, name: str, entry: dict):
        self.root, self.name, self.entry = root, name, entry
        self.spec = HeadSpec.from_dict(entry["spec"])
        self._ensemble: CalibratedEnsemble | None = None
        self._report: dict | None = None

    @property
    def dir(self) -> Path:
        d = (self.root / self.entry["dir"]).resolve()
        if not d.is_relative_to(self.root.resolve()):
            raise BundleError(f"head {self.name!r}: directory escapes the bundle")
        return d

    @property
    def ensemble(self) -> CalibratedEnsemble:
        if self._ensemble is None:
            self._ensemble = CalibratedEnsemble.load(self.dir)
        return self._ensemble

    @property
    def report(self) -> dict:
        if self._report is None:
            self._report = json.loads((self.dir / "report.json").read_text(encoding="utf-8"))
        return self._report

    def holdout_ledger(self) -> dict:
        return json.loads((self.dir / "bernoulli.json").read_text(encoding="utf-8"))

    @property
    def verdict(self) -> dict:
        return self.entry.get("verdict") or {}

    @property
    def conformal_threshold(self) -> float | None:
        c = self.entry.get("conformal") or {}
        return c.get("threshold")

    @property
    def set_quantile(self) -> float | None:
        c = self.entry.get("conformal") or {}
        return c.get("set_quantile")


class Bundle:
    def __init__(self, path: Path, meta: dict | None = None):
        self.path = Path(path)
        self.meta = meta or read_meta(self.path)
        self._heads: dict[str, LoadedHead] = {}

    @property
    def version(self) -> str:
        return self.meta["version"]

    def names(self) -> list[str]:
        return list(self.meta.get("heads", {}))

    def head(self, name: str) -> LoadedHead | None:
        entry = self.meta.get("heads", {}).get(name)
        if entry is None:
            return None
        if name not in self._heads:
            self._heads[name] = LoadedHead(self.path, name, entry)
        return self._heads[name]
