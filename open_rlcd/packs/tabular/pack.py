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

Columns that are only known after the decision (a second outcome, a review level, a closing date) must not be
features. The pack keeps every declared label column out; the others it cannot know: list them under
`pack_config.exclude`. As a safety net, `datasets()` adds a note when one feature on its own almost determines a
head's label (`leak_notes`): a note, never a refusal.

What it does not do: it does not fetch data (the file must be local), does not join tables or build
time-window aggregates (give it one row per decision), and does not deduplicate rows of the file (feedback rows
are deduplicated: only the latest outcome posted for a decision counts). A head whose classes must
be read from the file is skipped by `heads()` while the file is missing; trained heads are still served from
their bundle, which stores their specs and the provider's schema.
"""
from __future__ import annotations

import json
import os
import re
import threading
from collections import Counter
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
    config_keys,
    label_columns,
    merged_targets,
    parse_config,
    parse_time,
    read_table,
    resolve_target,
    to_float,
)
from .provider import VIEW, TabularProvider, new_salt

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

    def config_keys(self) -> dict:
        return config_keys()

    def head_override_keys(self) -> set[str]:
        return {"label"}      # `heads:` may add a head, which then brings its own label column

    def check_config(self, project: Project) -> list[dict]:
        """What `parse_config` and `merged_targets` refuse, as one finding (the data file is not opened)."""
        try:
            merged_targets(parse_config(project.pack_config, project.resolve_path), project.head_overrides)
        except TabularConfigError as e:
            return [{"level": "error", "path": "pack_config", "message": str(e)}]
        return []

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
        """From a bundle's config: the exact trained schema. Without: learned from the training file (with
        `hash_categories`, hashed with this project home's salt, see `category_salt`)."""
        if config:
            return TabularProvider.from_config(config)
        cfg, merged = self.targets(project)
        salt = category_salt(Path(project.home)) if cfg.hash_categories else None
        return TabularProvider.learn(self.table(cfg), cfg, label_columns(merged), salt=salt)

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
            t_all = np.asarray([_t(parse_time(v, na, cfg.time_format)) for v in table.data[cfg.time]], np.int64)
            bad = int((t_all < 0).sum())
            if bad == n:
                raise TabularConfigError(_time_error(cfg, table))
            if bad:
                common.append(f"{bad} of {n} rows ({_share(bad, n)}) have no parseable time in {cfg.time!r}: they "
                              f"only ever train" + ("" if cfg.time_format else
                                                    "; if they are written in another notation, set "
                                                    "pack_config.time_format"))
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
        notes = [*common, *target.notes]
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
        notes.extend(leak_notes(spec, target, table, provider, keep, y[0], na))
        fb = _feedback_rows(spec, cfg, provider, feedback, feedback_weight)
        if fb["replaced"]:
            notes.append(f"{fb['replaced']} feedback rows replaced by a later outcome posted for the same "
                         f"decision_id")
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


def category_salt(home: Path) -> str:
    """The salt for hashed vocabularies of this project home: <home>/tabular/category_salt, created once with a
    random value (owner-only permissions). One salt per home rather than per training run, so a retrain of the same
    table keeps its feature schema and the heads it does not retrain are carried over; every bundle stores the salt
    it was trained with, so a bundle never depends on this file."""
    path = Path(home) / "tabular" / "category_salt"
    for _ in range(2):
        try:
            text = path.read_text(encoding="ascii").strip()
        except (OSError, UnicodeDecodeError):
            text = ""
        if re.fullmatch(r"[0-9a-f]{32}", text):
            return text
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError:
            if text:   # present but not a salt: refuse to guess (replacing it would change every hash)
                raise TabularConfigError(f"{path} does not hold a salt (32 hex characters); fix or delete it") from None
            continue   # another process is writing it right now: read it again
        with os.fdopen(fd, "w", encoding="ascii") as f:
            f.write(new_salt() + "\n")
    text = path.read_text(encoding="ascii").strip()
    if not re.fullmatch(r"[0-9a-f]{32}", text):
        raise TabularConfigError(f"{path} does not hold a salt (32 hex characters); fix or delete it")
    return text


def _t(v: int | None) -> int:
    return -1 if v is None else int(v)


def _share(k: int, n: int) -> str:
    """k of n as a percentage that never rounds a non-zero share to 0 % or a partial one to 100 %."""
    pct = 100.0 * k / max(n, 1)
    if 0 < pct < 0.1:
        return "under 0.1 %"
    if 99.9 < pct < 100:
        return "over 99.9 %"
    return f"{pct:.1f} %"


def _time_error(cfg: TabularConfig, table: Table) -> str:
    """Why no time was read, with two of the column's values and the option that fixes it."""
    na = cfg.na_values
    samples = list(dict.fromkeys(v for v in (canon(x, na) for x in table.data[cfg.time]) if v is not None))[:2]
    shown = " and ".join(repr(v[:40]) for v in samples) or "(every value is empty)"
    if cfg.time_format:
        return (f"time column {cfg.time!r}: no value matches pack_config.time_format {cfg.time_format!r} (nor "
                f"ISO-8601 or epoch seconds); the column holds values like {shown}. Fix time_format (Python "
                f"strptime codes, e.g. '%d/%m/%Y %I:%M %p' for 31/01/2025 02:15 PM)")
    return (f"time column {cfg.time!r}: no value parses as ISO-8601 or epoch seconds; the column holds values like "
            f"{shown}. Set pack_config.time_format to their strptime format (e.g. '%d/%m/%Y %I:%M %p' for "
            f"31/01/2025 02:15 PM); times without a zone are read as UTC")


# ----------------------------------------------------------------------------- the leakage note

LEAK_MIN_ROWS = 200           # labelled rows below which nothing is said (too few to tell a leak from luck)
LEAK_MIN_MINORITY = 20        # rows outside the most frequent class below which nothing is said
LEAK_MIN_PER_VALUE = 10       # categorical: average rows per value below which pure cells are expected by chance
LEAK_ERROR_SHARE = 0.10       # categorical: flagged when guessing from the column leaves at most this share of the
#                               errors of guessing the most frequent class ...
LEAK_MAX_ERROR = 0.02         # ... and is wrong on at most this share of the rows
LEAK_AUC = 0.99               # numeric: flagged when the column ranks one class against the rest this well
LEAK_MAX_ROWS = 50_000        # rows looked at (evenly spaced over the file), to keep the check cheap


def leak_notes(spec: HeadSpec, target: Target, table: Table, provider: Any, keep: np.ndarray, y: np.ndarray,
               na: frozenset[str]) -> list[str]:
    """Notes about feature columns that, each on its own, almost determine this head's label: the usual sign of a
    column that is only filled in after the outcome is known (a review level, a refund amount, a closing reason).

    The test, per feature column, on the file's labelled rows (never feedback rows):

    * categorical column (also "is the value missing" for any column): guess every row's class as the most
      frequent class among the rows with the same value. Flag when those guesses are wrong on at most
      LEAK_MAX_ERROR of the rows AND on at most LEAK_ERROR_SHARE of the rows that "always guess the most frequent
      class" gets wrong. Values are grouped as the provider encodes them (vocabulary, __other__, __missing__)
      and the column needs LEAK_MIN_PER_VALUE rows per value on average, so a column with many rare values is
      not flagged for cells that are pure by chance.
    * numeric column: flag when, for some class with at least LEAK_MIN_MINORITY rows inside and outside it, the
      column's values rank that class against the rest with an AUC of LEAK_AUC or more (or 1 - LEAK_AUC or
      less), among the rows that have a value.

    Nothing is said with fewer than LEAK_MIN_ROWS labelled rows or fewer than LEAK_MIN_MINORITY rows outside
    the most frequent class. Text columns are not tested. What this cannot see: leaks spread over several
    columns, weaker leaks (a column decided after the fact that is 90 % aligned with the outcome), and it
    cannot tell a leak from a genuinely decisive input known at decision time, which is why it is a note."""
    columns = getattr(provider, "columns", None)
    n = len(keep)
    if not columns or n < LEAK_MIN_ROWS:
        return []
    if n > LEAK_MAX_ROWS:
        pick = np.linspace(0, n - 1, LEAK_MAX_ROWS).astype(int)
        keep, y, n = keep[pick], y[pick], LEAK_MAX_ROWS
    k = int(y.max()) + 1
    class_n = np.bincount(y, minlength=k)
    base_err = n - int(class_n.max())
    if base_err < LEAK_MIN_MINORITY:
        return []
    values: dict[int, str] = {}           # class index -> the label value as the column writes it
    for value, ci in target.mapping.items():
        values.setdefault(ci, value)

    def name_of(ci: int) -> str:
        if target.type == "noul":
            return repr(target.positive) if ci == 1 else f"not {target.positive!r}"
        return repr(values.get(ci, str(ci)))

    def cell_errors(codes: np.ndarray) -> tuple[int, int]:
        """(rows a per-value majority guess gets wrong, number of values)."""
        _, inv = np.unique(codes, return_inverse=True)
        cells = int(inv.max()) + 1
        table_ = np.zeros((cells, k), np.int64)
        np.add.at(table_, (inv, y), 1)
        return n - int(table_.max(axis=1).sum()), cells

    def decisive(err: int, cells: int) -> bool:
        return (cells >= 2 and n / cells >= LEAK_MIN_PER_VALUE and err <= LEAK_MAX_ERROR * n
                and err <= LEAK_ERROR_SHARE * base_err)

    out: list[str] = []
    advice = ("If it is only known after the decision, list it under pack_config.exclude (or under targets if it is "
              "an outcome itself): a model trained on it looks far better on the holdout than it can be in use. If "
              "it is known when the decision is made, nothing needs to change")
    for col in columns:
        if col.kind == "text" or col.name not in table.data:
            continue
        raw = table.data[col.name]
        vals = [canon(raw[i], na) for i in keep]
        missing = np.asarray([v is None for v in vals])
        why = None
        if col.kind == "categorical":
            if col.hashed:      # group by raw value: the grouping only matters up to the vocabulary size
                top = {v for v, _ in Counter(v for v in vals if v is not None).most_common(len(col.vocabulary))}
            else:
                top = set(col.vocabulary) | {v.lower() for v in col.vocabulary}
            codes = np.asarray(["\x00missing" if v is None else (v if v in top or v.lower() in top else "\x00other")
                                for v in vals], dtype=object)
            err, cells = cell_errors(codes)
            if decisive(err, cells):
                why = (f"its value alone gives the right label for {_share(n - err, n)} of {n} rows; always guessing "
                       f"{name_of(int(class_n.argmax()))} gives {_share(n - base_err, n)}")
        else:
            x = np.asarray([to_float(raw[i], na) for i in keep], float)
            have = ~np.isnan(x)
            for ci in range(k):
                pos = have & (y == ci)
                neg = have & (y != ci)
                if pos.sum() < LEAK_MIN_MINORITY or neg.sum() < LEAK_MIN_MINORITY:
                    continue
                auc = _auc(x[have], (y == ci)[have])
                if auc >= LEAK_AUC or auc <= 1 - LEAK_AUC:
                    side = "higher" if auc >= LEAK_AUC else "lower"
                    why = (f"{side} values separate {target.label} = {name_of(ci)} from the rest almost perfectly: "
                           f"rank AUC {max(auc, 1 - auc):.3f} over {int(have.sum())} rows")
                    break
            if why is None and missing.any() and not missing.all():
                err, cells = cell_errors(missing.astype(int))
                if decisive(err, cells):
                    why = (f"whether it is filled in at all gives the right label for {_share(n - err, n)} of {n} "
                           f"rows")
        if why:
            out.append(f"possible label leak: feature column {col.name!r} on its own almost determines "
                       f"{target.label!r} ({why}). {advice}")
    return out


def _auc(x: np.ndarray, positive: np.ndarray) -> float:
    """Rank AUC of `x` for `positive` (ties get their average rank): P(x of a positive row > x of another row)."""
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), float)
    xs = x[order]
    starts = np.flatnonzero(np.r_[True, xs[1:] != xs[:-1]])
    ends = np.r_[starts[1:], len(xs)]
    avg = (starts + ends - 1) / 2.0 + 1.0
    ranks[order] = np.repeat(avg, ends - starts)
    n_pos = int(positive.sum())
    n_neg = len(x) - n_pos
    return float((ranks[positive].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


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
    skipped, why, replaced = 0, set(), 0
    rows = [r for r in feedback or [] if r.get("head") == spec.name]
    # a decision's outcome counts once, as in the Bernoulli ledger and in packs.base.dataset_from_labels: a retried
    # or corrected POST /v1/feedback for the same decision_id replaces the earlier row instead of adding weight
    # (a /v1/systemone receipt can ask one head in several questions: each question is its own outcome)
    last = {(str(r["decision_id"]), r.get("question")): i for i, r in enumerate(rows) if r.get("decision_id")}
    for i, r in enumerate(rows):
        if r.get("decision_id") and last[(str(r["decision_id"]), r.get("question"))] != i:
            replaced += 1
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
        t = parse_time(state.get(cfg.time), na, cfg.time_format) if raw and cfg.time else None
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
            "weight": np.asarray(w, float), "skipped": skipped, "why": ", ".join(sorted(why)),
            "replaced": replaced}
