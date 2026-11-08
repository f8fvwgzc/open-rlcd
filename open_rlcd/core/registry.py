"""Versioned models under RLCD_HOME.

    RLCD_HOME/
      registry.json              {"latest": "rlcd-0.1.3", "next": 4, "versions": {id: {...}}}
      models/<version>/          one bundle per version (see core/bundle.py)
      bernoulli.json             the live Beta-Bernoulli ledger (restarts from each new holdout)
      feedback.jsonl, decisions.jsonl

`rlcd-latest` is an alias of the newest version; old versions stay addressable through `model`. A retrain that
touches some heads carries the others over unchanged, so a version always describes the whole model. Bundles
are only ever loaded from inside RLCD_HOME (registry.json is data: a path in it that escapes the home is refused).

A weight-edit candidate (core/patch.py, `commit_edit`) is a version like any other, plus an `edit` record. It
becomes rlcd-latest only when its holdout gate passed and its base is still rlcd-latest; one whose gate failed is
kept for inspection and never served (`servable`).
"""
from __future__ import annotations

import json
import os
import shutil
import threading
from pathlib import Path
from typing import Any

from .. import LATEST, MODEL_PREFIX
from . import bundle as BD
from .bernoulli import Ledger
from .locks import FileLock, write_atomic
from .training import now_iso


class UnknownModel(Exception):
    pass


def default_home() -> Path:
    return Path(os.environ.get("RLCD_HOME") or "var").expanduser().resolve()


class Registry:
    def __init__(self, home: Path | None = None):
        self.home = Path(home).expanduser().resolve() if home else default_home()
        (self.home / "models").mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        # Version ids are handed out and `latest` is moved by several processes (`rlcd train`, `import`, `promote`,
        # the server's own training jobs). registry.json is never cached: each of those read-modify-writes reads it
        # inside this inter-process lock and holds the lock until the new file is in place.
        self._file_lock = FileLock(self.home / "registry.json")
        self._bundles: dict[str, BD.Bundle] = {}
        self.ledger = Ledger(self.home / "bernoulli.json")

    # ------------------------------------------------------------------ registry.json
    @property
    def _path(self) -> Path:
        return self.home / "registry.json"

    def registry(self) -> dict:
        with self._lock:
            if self._path.exists():
                return json.loads(self._path.read_text(encoding="utf-8"))
            return {"latest": None, "next": 1, "versions": {}}

    def _write(self, reg: dict) -> None:
        write_atomic(self._path, json.dumps(reg, indent=1))

    def latest_id(self) -> str:
        """The versioned id `rlcd-latest` points at. `rlcd-0.1.0` is the untrained service."""
        return self.registry()["latest"] or f"{MODEL_PREFIX}0"

    def resolve(self, model: str | None) -> str:
        reg = self.registry()
        latest = reg["latest"] or f"{MODEL_PREFIX}0"
        if model in (None, "", LATEST, latest):
            return latest
        if model in reg["versions"]:
            return model
        raise UnknownModel(model)

    def available(self) -> list[str]:
        return [LATEST, *sorted(self.registry()["versions"], key=_vkey)]

    def entry(self, version: str) -> dict | None:
        return self.registry()["versions"].get(version)

    def servable(self, version: str) -> bool:
        """False only for a weight-edit candidate whose holdout gate failed: it stays addressable for reports and
        exports, but no request is ever answered with it."""
        e = self.entry(version) or {}
        return (e.get("edit") or {}).get("servable", True) is not False

    def edit_attempts(self, head: str, holdout_sha256: str) -> int:
        """How many edit candidates of `head` this home has already scored on the holdout with this id hash. Each
        scoring spends some of the holdout's credibility (core/patch.py tightens the bound accordingly)."""
        n = 0
        for e in self.registry()["versions"].values():
            if ((e.get("edit") or {}).get("holdouts") or {}).get(head) == holdout_sha256:
                n += 1
        return n

    # ------------------------------------------------------------------ bundles
    def _dir(self, version: str) -> Path:
        e = self.entry(version)
        if e is None:
            raise UnknownModel(version)
        d = (self.home / e["path"]).resolve()
        if not d.is_relative_to(self.home):
            raise RuntimeError(f"model path escapes RLCD_HOME: {e['path']}")
        return d

    def bundle(self, version: str) -> BD.Bundle | None:
        """The bundle of a version, or None for the untrained rlcd-0.1.0."""
        if version == f"{MODEL_PREFIX}0" and self.entry(version) is None:
            return None
        with self._lock:
            if version not in self._bundles:
                self._bundles[version] = BD.Bundle(self._dir(version))
            return self._bundles[version]

    def _register(self, reg: dict, vid: str, meta: dict, extra: dict | None = None) -> None:
        reg["versions"][vid] = {"path": f"models/{vid}", "created_at": meta.get("created_at"),
                                "pack": (meta.get("pack") or {}).get("name"),
                                "feature_version": meta.get("feature_version"),
                                "retrained": meta.get("retrained", []),
                                "heads": {n: {"trained_at": h.get("trained_at"), "metrics": h.get("metrics"),
                                              "verdict": h.get("verdict"), "model": h.get("model")}
                                          for n, h in meta.get("heads", {}).items()},
                                **(extra or {})}
        n = int(vid.rsplit(".", 1)[1])
        reg["latest"] = vid if reg["latest"] is None or _vkey(vid) >= _vkey(reg["latest"]) else reg["latest"]
        reg["next"] = max(int(reg["next"]), n + 1)

    def commit(self, trained: dict[str, Any], *, pack: dict, feature_version: str, encoder: dict | None,
               policy: dict, data_card: dict | None = None, notes: list[str] | None = None) -> str:
        """Save freshly trained heads as a new version, carrying over the previous version's other heads when
        they were trained on the same pack and feature version."""
        with self._lock, self._file_lock:
            reg = self.registry()
            vid = f"{MODEL_PREFIX}{int(reg['next'])}"
            carried: dict[str, tuple[Path, dict]] = {}
            dropped: list[str] = []
            prev = self.bundle(reg["latest"]) if reg["latest"] else None
            if prev is not None:
                same = (prev.meta.get("feature_version") == feature_version
                        and (prev.meta.get("pack") or {}).get("name") == pack.get("name"))
                for name in prev.names():
                    if name in trained:
                        continue
                    if same:
                        h = prev.head(name)
                        # `trained_in`: the version whose training produced this head (kept as it is carried on)
                        carried[name] = (h.dir, {**h.entry, "trained_in": h.entry.get("trained_in") or reg["latest"]})
                    else:
                        dropped.append(name)
            if dropped:
                notes = [*(notes or []), f"not carried over (other feature version or pack): {sorted(dropped)}"]
            meta = BD.write_bundle(self.home / "models" / vid, version=vid, created_at=now_iso(), pack=pack,
                                   feature_version=feature_version, encoder=encoder, policy=policy, trained=trained,
                                   carried=carried, data_card=data_card, notes=notes)
            prev_vid = reg["latest"]
            self._register(reg, vid, meta)
            if reg["latest"] == vid and prev_vid != vid:
                reg.setdefault("history", []).append({"at": now_iso(), "from": prev_vid, "to": vid, "reason": None,
                                                      "by": None, "kind": "train"})
            self._write(reg)
            self._restart_ledger(vid, prev_vid, trained, carried)
            return vid

    def _restart_ledger(self, vid: str, prev_vid: str | None, trained: dict[str, Any],
                        carried: dict[str, tuple[Path, dict]]) -> None:
        """`vid` just became rlcd-latest: the Bernoulli ledger of each new head restarts from its new holdout."""
        for name, th in trained.items():
            self.ledger.reset(name, {**th.ledger, "model_version": vid, "valid_for": [vid]})
        # A carried-over head is the very same model, so its evidence (holdout counts plus the outcomes scored
        # since) carries over too, and the table stays valid for every version that served this head: the
        # Bernoulli gate keeps its evidence for the new version, and an outcome posted now for a decision made
        # before the retrain still updates it (`valid_for`, which feedback and the gate check).
        # read and reset under one hold of the ledger: an outcome another process scores meanwhile is not lost
        with self.ledger.exclusive():
            for name, (old_dir, entry) in carried.items():
                table = self.ledger.table(name)
                if table and (table.get("model_version") == prev_vid or prev_vid in (table.get("valid_for") or [])):
                    # live for the version being replaced (also through valid_for, e.g. after a rollback): its
                    # outcomes scored since the holdout carry over with the very same head
                    valid = list(dict.fromkeys([*(table.get("valid_for") or []), table.get("model_version"), prev_vid]))
                    valid = [v for v in valid if v]
                else:
                    table = json.loads((old_dir / "bernoulli.json").read_text(encoding="utf-8"))
                    valid = [v for v in (entry.get("trained_in"), prev_vid) if v]
                valid = list(dict.fromkeys([*valid, vid]))
                self.ledger.reset(name, {**table, "model_version": vid, "valid_for": valid})

    def commit_edit(self, base_version: str, trained: dict[str, Any], *, edit: dict, carry: bool = True,
                    pack: dict | None = None, feature_version: str | None = None, encoder: dict | None = None,
                    data_card: dict | None = None, notes: list[str] | None = None) -> dict:
        """Save a weight-edit candidate as a new version and decide, under the lock, whether it serves.

        `trained`: the edited (head edit) or retrained (encoder edit) heads, already scored on the base's holdout;
        `carry`: copy the base version's other heads unchanged (a head edit; an encoder edit changes every head's
        features, so nothing is carried); `edit["passed"]` / `edit["reasons"]`: the gate's result (core/patch.py).
        The candidate becomes rlcd-latest only if the gate passed AND `base_version` is still rlcd-latest (an edit
        is judged against its base, not against whatever is newer). Otherwise it is kept as a non-latest version
        with the reason; a candidate whose gate failed is never served. Returns the final edit record + version."""
        with self._lock:
            reg = self.registry()
            base = self.bundle(base_version)
            if base is None:
                raise UnknownModel(base_version)
            vid = f"{MODEL_PREFIX}{int(reg['next'])}"
            carried: dict[str, tuple[Path, dict]] = {}
            if carry:
                for name in base.names():
                    if name not in trained:
                        h = base.head(name)
                        carried[name] = (h.dir, {**h.entry, "trained_in": h.entry.get("trained_in") or base_version})
            latest_before = reg["latest"]
            passed = bool(edit.get("passed"))
            promoted = passed and latest_before == base_version
            if promoted:
                reason = "the paired holdout test shows a reliable gain over the base and no verdict is weaker"
            elif passed:
                reason = (f"the gate passed, but the base {base_version} is no longer rlcd-latest ({latest_before}): "
                          f"an edit is only promoted over the version it was judged against; edit {latest_before}")
            else:
                reason = "; ".join(edit.get("reasons") or ["the gate did not pass"])
            edit = {**edit, "promoted": promoted, "servable": passed, "reason": reason}
            meta = BD.write_bundle(self.home / "models" / vid, version=vid, created_at=now_iso(),
                                   pack=pack or base.meta.get("pack") or {},
                                   feature_version=feature_version or base.meta.get("feature_version"),
                                   encoder=encoder if encoder is not None else base.meta.get("encoder"),
                                   policy=base.meta.get("policy") or {}, trained=trained, carried=carried,
                                   data_card=data_card if data_card is not None else base.meta.get("data_card"),
                                   notes=notes, extra={"edit": edit})
            self._register(reg, vid, meta, {"edit": edit_summary(edit)})
            if not promoted:
                reg["latest"] = latest_before
            self._write(reg)
            if promoted:
                self._restart_ledger(vid, base_version, trained, carried)
            return {"version": vid, **edit}

    def set_latest(self, version: str, *, reason: str | None = None, by: str | None = None,
                   kind: str = "promote") -> dict:
        """Point `rlcd-latest` at an existing version (rollback or promotion), recorded in registry.json `history`.
        Each head of that version then gates on evidence that belongs to it: a Bernoulli table that is valid for the
        version (a carried-over head) is kept, any other restarts from the version's own holdout table (outcomes
        scored for the version that was latest stay with that version's table, which is replaced)."""
        with self._lock, self._file_lock:
            reg = self.registry()
            if version not in reg["versions"]:
                raise UnknownModel(version)
            prev = reg["latest"]
            if prev == version:
                return {"latest": version, "previous": prev, "changed": False}
            reg["latest"] = version
            reg.setdefault("history", []).append({"at": now_iso(), "from": prev, "to": version, "reason": reason,
                                                  "by": by, "kind": kind})
            self._write(reg)
            b = self.bundle(version)
            restarted = []
            # one hold of the ledger from reading a table to resetting it (the server may be scoring outcomes)
            with self.ledger.exclusive():
                for name in b.names():
                    table = self.ledger.table(name) or {}
                    if table.get("model_version") == version or version in (table.get("valid_for") or []):
                        continue
                    restored = self.ledger.archived(name, version)    # its own evidence, outcomes included
                    base = restored if restored is not None else b.head(name).holdout_ledger()
                    self.ledger.reset(name, {**base, "model_version": version,
                                             "valid_for": list(dict.fromkeys([*(base.get("valid_for") or []), version]))})
                    restarted.append(name)
            return {"latest": version, "previous": prev, "changed": True, "ledger_restarted": restarted}

    def previous_latest(self) -> str | None:
        """The version to roll back to (`rlcd rollback` without --to): the version that was latest before the
        current one became latest, skipping every version that was rolled back from and not made latest again
        since (a second rollback must not re-promote the version the first one removed, but a version promoted back
        after a rollback is a good one again). Without history: the newest older version."""
        reg = self.registry()
        hist = reg.get("history") or []
        latest = reg["latest"]
        rolled_back: set = set()
        for h in hist:                      # in order: a version made latest again no longer counts as rolled back
            rolled_back.discard(h.get("to"))
            if h.get("kind") == "rollback":
                rolled_back.add(h.get("from"))
        cur, seen = latest, {latest}
        for h in reversed(hist):
            if h.get("to") != cur:
                continue
            cand = h.get("from")
            if cand in reg["versions"] and cand not in rolled_back and cand not in seen:
                return cand
            if cand is None or cand in seen:
                break
            seen.add(cand)
            cur = cand
        older = [v for v in reg["versions"] if _vkey(v) < _vkey(latest or "") and v not in rolled_back]
        return max(older, key=_vkey) if older else None

    # ------------------------------------------------------------------ export / import
    def export(self, version: str, dest: Path) -> Path:
        src = self._dir(version)
        dest = Path(dest).expanduser().resolve()
        if dest.exists() and any(dest.iterdir()):
            raise BD.BundleError(f"{dest} exists and is not empty")
        if dest.exists():
            dest.rmdir()
        shutil.copytree(src, dest, symlinks=False)
        BD.validate(dest)
        return dest

    def import_(self, src: Path, *, make_latest: bool = True) -> str:
        """Validate a bundle directory and register it as a new version of this home.

        The source is validated, then copied without following symlinks (a symlink planted after the first check
        lands as a symlink, not as the bytes it points at), and the copy is validated again, LightGBM members
        included (each is parsed in a short-lived subprocess, so a malformed model aborts that subprocess and fails
        the import instead of crashing the server later). Whatever fails, the copy is removed."""
        src = Path(src).expanduser()
        BD.validate(src)
        with self._lock, self._file_lock:
            reg = self.registry()
            vid = f"{MODEL_PREFIX}{int(reg['next'])}"
            dest = self.home / "models" / vid
            if dest.exists():
                raise BD.BundleError(f"{dest} already exists: refusing to overwrite it")
            try:   # the source could have changed between validate and copy: check what actually landed
                shutil.copytree(src, dest, symlinks=True)
                meta = BD.validate(dest, check_models=True)
            except BaseException:
                shutil.rmtree(dest, ignore_errors=True)
                raise
            heads = {n: {**e, "trained_in": vid} for n, e in (meta.get("heads") or {}).items()}
            meta = {**meta, "version": vid, "imported_from": meta.get("version"), "heads": heads}
            (dest / "model.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False), encoding="utf-8")
            latest_before = reg["latest"]
            extra = {"imported_from": meta["imported_from"], "imported_at": now_iso()}
            if isinstance(meta.get("edit"), dict):   # a weight-edit candidate keeps its status across export / import
                extra["edit"] = edit_summary(meta["edit"])
                if meta["edit"].get("servable") is False:
                    make_latest = False
            self._register(reg, vid, meta, extra)
            if not make_latest:
                reg["latest"] = latest_before
            if reg["latest"] == vid and latest_before != vid:
                # recorded like a train or a promote, so `rlcd rollback` returns to the version that was serving
                # (without it, previous_latest falls back to the newest older version: a demoted one, or an
                # import kept out of service with --keep-latest)
                reg.setdefault("history", []).append({"at": now_iso(), "from": latest_before, "to": vid,
                                                      "reason": None, "by": None, "kind": "import"})
            self._write(reg)
            bundle = BD.Bundle(dest, meta)
            for name in bundle.names():
                table = bundle.head(name).holdout_ledger()
                if make_latest:
                    self.ledger.reset(name, {**table, "model_version": vid, "valid_for": [vid]})
            return vid


def edit_summary(edit: dict) -> dict:
    """The registry's copy of a weight edit's record: enough to list it, serve it or not, and count how often a
    holdout has judged an edit (`holdouts`: head -> sha256 of the holdout row ids)."""
    keep = ("kind", "base_version", "sha256", "author", "parameters_touched", "passed", "promoted", "servable",
            "reason", "holdouts")
    out = {k: edit.get(k) for k in keep if k in edit}
    if "author" not in out and isinstance(edit.get("provenance"), dict):
        out["author"] = edit["provenance"].get("author")
    return out


def _vkey(vid: str) -> int:
    try:
        return int(vid.rsplit(".", 1)[1])
    except (IndexError, ValueError):
        return -1
