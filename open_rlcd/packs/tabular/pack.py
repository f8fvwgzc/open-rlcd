"""The tabular pack: one calibrated head per label column of a CSV / Parquet table.

    pack: tabular
    pack_config:
      data: data/churn.csv
      group: customer_id          # a customer's rows never straddle train / calibration / holdout
      time: month
      targets:
        churned: {label: churned}                       # 2 values -> noul (true = "yes", say which in the head)
        next_plan: {label: plan_tier_next}              # 3+ values -> choice
        satisfaction: {label: satisfaction, type: score}

Every head reads the same feature view, "row": a request's `state` is one row as a JSON object
`{column: value}`. Labels from the file are `ground_truth` (trusted) unless `label_source` says otherwise, so
they calibrate and are scored on the holdout; feedback rows posted to /v1/feedback with a raw state are
re-featurised at the next train and only train the base models (weight x feedback_weight).

What it does not do: it does not fetch data (the file must be local), does not join tables or build
time-window aggregates (give it one row per decision), and does not deduplicate rows. A head whose classes must
be read from the file is skipped by `heads()` while the file is missing; trained heads are still served from
their bundle, which stores their specs and the provider's schema.
"""
from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

import numpy as np

from ...core.config import Project
from ...core.contract import ContractError
from ...core.dataset import Dataset
from ...core.head import HeadSpec
from ..base import BasePack, epoch, precomputed, spec_from_yaml
from .data import (
    DataUnavailable,
    Table,
    TabularConfig,
    TabularConfigError,
    Target,
    canon,
    label_columns,
    merged_targets,
    parse_config,
    parse_time,
    read_table,
    resolve_target,
)
from .provider import VIEW, TabularProvider

MISSING_SEGMENT = "__missing__"


class TabularPack(BasePack):
    name = "tabular"
    version = "0.1.0"
    yaml_path = Path(__file__).with_name("pack.yaml")
    abstain = "hold"

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._table: tuple[tuple, Table] | None = None
        self._heads: tuple[tuple, dict[str, HeadSpec]] | None = None
        self._defaults: dict | None = None

    def defaults(self) -> dict:
        if self._defaults is None:
            self._defaults = super().defaults()
        return self._defaults

    # ------------------------------------------------------------------ config and data
    def config(self, project: Project) -> TabularConfig:
        return parse_config(project.pack_config, project.resolve_path)

    def table(self, cfg: TabularConfig) -> Table:
        """The training table, re-read only when the file (path, size, mtime) or the reading options change."""
        return self._load(cfg)[1]

    def _load(self, cfg: TabularConfig) -> tuple[tuple, Table]:
        try:
            st = cfg.data.stat()
        except OSError as e:
            raise DataUnavailable(f"tabular data file not found or unreadable: {cfg.data} ({e.strerror})") from None
        key = (*cfg.signature(), st.st_mtime_ns, st.st_size)
        with self._lock:
            if self._table is None or self._table[0] != key:
                self._table = (key, read_table(cfg.data, cfg.delimiter, cfg.encoding))
            return self._table

    def targets(self, project: Project) -> tuple[TabularConfig, dict[str, dict]]:
        """The parsed pack_config and every target (pack_config targets merged with rlcd.yaml head overrides)."""
        cfg = self.config(project)
        return cfg, merged_targets(cfg, project.head_overrides)

    # ------------------------------------------------------------------ Pack protocol
    def heads(self, project: Project) -> dict[str, HeadSpec]:
        """One head per enabled target. Called on every request by the service, so the result is cached until
        pack_config, the head overrides or the data file change."""
        cfg, merged = self.targets(project)
        table: Table | None
        try:
            file_key, table = self._load(cfg)
        except DataUnavailable:
            file_key, table = None, None
        key = (json.dumps([project.pack_config, project.head_overrides], sort_keys=True, default=str), file_key)
        with self._lock:
            if self._heads is not None and self._heads[0] == key:
                return dict(self._heads[1])
        out: dict[str, HeadSpec] = {}
        for name, d in merged.items():
            if d.get("enabled", True) is False:
                continue
            try:
                t = resolve_target(name, d, table, cfg.na_values)
            except DataUnavailable:
                continue  # its classes come from the file: known again once the file is there
            out[name] = spec_from_yaml(name, t.spec)
        with self._lock:
            self._heads = (key, out)
        return dict(out)

    def provider(self, project: Project, config: dict | None = None) -> TabularProvider:
        """From a bundle's config: the exact trained schema. Without: learned from the training file."""
        if config:
            return TabularProvider.from_config(config)
        cfg, merged = self.targets(project)
        return TabularProvider.learn(self.table(cfg), cfg, label_columns(merged))

    def datasets(self, project: Project, provider: Any, heads: list[str], feedback: list[dict] | None = None,
                 feedback_weight: float = 5.0) -> dict[str, Dataset]:
        cfg, merged = self.targets(project)
        table = self.table(cfg)
        na = cfg.na_values
        table.require(cfg.meta_columns, "pack_config id / group / time / segment")
        table.require(provider.required(), f"feature columns of the provider ({provider.feature_version})")
        n = table.n
        X_all = provider.transform(VIEW, table.rows(provider.required()))
        common: list[str] = list(getattr(provider, "notes", []))
        ids = np.asarray([_key(table, cfg.id, i, na) or f"row:{i}" for i in range(n)], dtype=object)
        t_all = None
        if cfg.time:
            t_all = np.asarray([_t(parse_time(v, na)) for v in table.data[cfg.time]], np.int64)
            bad = int((t_all < 0).sum())
            if bad == n:
                raise TabularConfigError(f"time column {cfg.time!r}: no value parses as ISO-8601 or epoch seconds")
            if bad:
                common.append(f"{bad} rows have no parseable time in {cfg.time!r}: they only ever train")
        g_all = None
        if cfg.group:
            g_all = np.asarray([_key(table, cfg.group, i, na) or f"row:{i}" for i in range(n)], dtype=object)
            lone = sum(1 for i in range(n) if _key(table, cfg.group, i, na) is None)
            if lone:
                common.append(f"{lone} rows have no {cfg.group!r}: each is its own group")
        s_all = None
        if cfg.segment:
            s_all = np.asarray([_key(table, cfg.segment, i, na) or MISSING_SEGMENT for i in range(n)], dtype=object)
        out: dict[str, Dataset] = {}
        for name in heads:
            if name not in merged:
                raise TabularConfigError(f"no target for head {name!r}; targets: {sorted(merged)}")
            target = resolve_target(name, merged[name], table, na)
            spec = spec_from_yaml(name, target.spec)
            out[name] = self._dataset(spec, target, cfg, table, provider, X_all, ids, t_all, g_all, s_all,
                                      feedback, feedback_weight, common)
        return out

    def _dataset(self, spec: HeadSpec, target: Target, cfg: TabularConfig, table: Table, provider: Any,
                 X_all: np.ndarray, ids: np.ndarray, t_all: np.ndarray | None, g_all: np.ndarray | None,
                 s_all: np.ndarray | None, feedback: list[dict] | None, feedback_weight: float,
                 common: list[str]) -> Dataset:
        na = cfg.na_values
        table.require([target.label], f"label of head {spec.name!r}")
        y_raw = [target.index(v, na) for v in table.data[target.label]]
        keep = np.asarray([i for i, y in enumerate(y_raw) if y is not None], dtype=int)
        n_missing = sum(1 for v in table.data[target.label] if canon(v, na) is None)
        n_unknown = len(y_raw) - len(keep) - n_missing
        notes = list(common)
        if n_missing:
            notes.append(f"{n_missing} rows have no label in {target.label!r}: skipped")
        if n_unknown:
            notes.append(f"{n_unknown} rows have a {target.label!r} value outside the head's classes "
                         f"{list(spec.labels if spec.type != 'score' else spec.classes)}: skipped")
        X = [X_all[keep]]
        y = [np.asarray([y_raw[i] for i in keep], np.int64)]
        rid = [ids[keep]]
        t = [t_all[keep]] if t_all is not None else None
        g = [g_all[keep]] if g_all is not None else None
        seg = [s_all[keep]] if s_all is not None else None
        src = [np.full(len(keep), cfg.label_source, dtype=object)]
        w = [np.ones(len(keep))]
        fb = _feedback_rows(spec, cfg, provider, feedback, feedback_weight)
        if fb["n"]:
            X.append(fb["X"])
            y.append(fb["y"])
            rid.append(fb["ids"])
            if t is not None:
                t.append(fb["t"])
            if g is not None:
                g.append(fb["group"])
            if seg is not None:
                seg.append(fb["segment"])
            src.append(np.full(fb["n"], "feedback", dtype=object))
            w.append(fb["weight"])
            notes.append(f"{fb['n']} feedback rows added to training (weight x {feedback_weight:g})")
        if fb["skipped"]:
            notes.append(f"{fb['skipped']} feedback rows for head {spec.name!r} skipped ({fb['why']})")
        return Dataset(X=np.vstack(X), y=np.concatenate(y), feature_names=provider.names(VIEW),
                       feature_version=provider.feature_version, ids=np.concatenate(rid),
                       t=None if t is None else np.concatenate(t), group=None if g is None else np.concatenate(g),
                       weight=np.concatenate(w), source=np.concatenate(src),
                       segment=None if seg is None else np.concatenate(seg),
                       feature_groups=provider.groups(VIEW), notes=notes)

    # ------------------------------------------------------------------ optional hooks
    def synth(self, seed: int = 7, n: int = 4000, path: str | Path | None = None, signal: float = 1.0,
              noise: bool = False) -> dict:
        """Write the synthetic churn table (see synth.py) and return its path plus a ready pack_config.
        Default path: ./data/synthetic_churn.csv under the current directory."""
        from .synth import synthetic_pack_config, write_synthetic_csv

        dest = Path(path) if path is not None else Path.cwd() / "data" / "synthetic_churn.csv"
        info = write_synthetic_csv(dest, n=n, seed=seed, signal=signal, noise=noise)
        return {**info, "pack_config": synthetic_pack_config(dest)}


def _t(v: int | None) -> int:
    return -1 if v is None else int(v)


def _key(table: Table, column: str | None, i: int, na: frozenset[str]) -> str | None:
    return None if column is None else canon(table.data[column][i], na)


def _feedback_rows(spec: HeadSpec, cfg: TabularConfig, provider: Any, feedback: list[dict] | None,
                   feedback_weight: float) -> dict:
    """Feedback rows for one head, featurised with the current provider. A raw state is re-featurised (so it
    survives a schema change as long as its columns are still there); a precomputed {x, feature_version} is used
    only when it matches the current feature version. Rows that cannot be used are counted, never fatal."""
    na = cfg.na_values
    width = len(provider.names(VIEW))
    X, y, ids, ts, groups, segs, w = [], [], [], [], [], [], []
    skipped, why = 0, set()
    for r in feedback or []:
        if r.get("head") != spec.name:
            continue
        state = r.get("state")
        try:
            yi = spec.label_index(r.get("label"))
        except ValueError:
            skipped += 1
            why.add("label not valid for the head")
            continue
        raw = isinstance(state, dict) and not precomputed(state)
        if precomputed(state):
            x = state.get("x")
            if state.get("feature_version") != provider.feature_version or not isinstance(x, list) or len(x) != width:
                skipped += 1
                why.add("precomputed features of another feature version")
                continue
            try:
                row = np.asarray([np.nan if v is None else float(v) for v in x], np.float32)
            except (TypeError, ValueError):
                skipped += 1
                why.add("bad feature vector")
                continue
        elif raw:
            try:
                row = provider.transform(VIEW, [state])[0]
            except ContractError:
                skipped += 1
                why.add("state lacks columns of the current schema")
                continue
        else:
            skipped += 1
            why.add("no state")
            continue
        fid = f"feedback:{r.get('id')}"
        X.append(row)
        y.append(yi)
        ids.append(fid)
        t = parse_time(state.get(cfg.time), na) if raw and cfg.time else None
        ts.append(_t(t if t is not None else epoch(r.get("time") or r.get("received_at"))))
        g = canon(state.get(cfg.group), na) if raw and cfg.group else None
        groups.append(g or f"feedback:{r.get('decision_id') or r.get('id')}")
        s = canon(state.get(cfg.segment), na) if raw and cfg.segment else None
        segs.append(s or MISSING_SEGMENT)
        w.append(float(r.get("weight") or 1.0) * float(feedback_weight))
    n = len(X)
    return {"n": n, "X": np.vstack(X) if n else np.zeros((0, width), np.float32), "y": np.asarray(y, np.int64),
            "ids": np.asarray(ids, dtype=object), "t": np.asarray(ts, np.int64),
            "group": np.asarray(groups, dtype=object), "segment": np.asarray(segs, dtype=object),
            "weight": np.asarray(w, float), "skipped": skipped, "why": ", ".join(sorted(why))}
