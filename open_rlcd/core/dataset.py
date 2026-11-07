"""The rows one head is trained on, already featurised.

Label provenance matters: `trusted` rows (human-verified or ground truth) are the only ones a calibrator is
fitted on and the only ones the holdout scores. `judge` rows (weak labels from an LLM judge) and `feedback`
rows (production outcomes, up-weighted) help the base model learn the shape of the problem; they never decide
what a probability means.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

SOURCES = ("human", "ground_truth", "synthetic", "judge", "feedback")
TRUSTED_SOURCES = ("human", "ground_truth", "synthetic")


@dataclass
class Dataset:
    X: np.ndarray                         # (n, f) float32, NaN = missing
    y: np.ndarray                         # (n,) int class index into spec.labels
    feature_names: list[str]
    feature_version: str
    ids: np.ndarray | None = None         # row ids (str)
    t: np.ndarray | None = None           # epoch seconds: enables time ordering
    t_end: np.ndarray | None = None       # end of each row's label window (purge); defaults to t
    group: np.ndarray | None = None       # document / customer / ticket: enables group ordering
    weight: np.ndarray | None = None      # sample weights (feedback up-weight only; never class weights)
    trusted: np.ndarray | None = None     # bool
    source: np.ndarray | None = None      # one of SOURCES per row
    judge_y: np.ndarray | None = None     # judge label (class index, -1 = none) for agreement reporting
    segment: np.ndarray | None = None     # optional slice key for per-segment reports and baselines
    feature_groups: dict | None = None    # feature name -> group (reason_groups)
    holdout_ok: np.ndarray | None = None  # bool: may be scored in the holdout (False for actively sampled rows)
    notes: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.X = np.ascontiguousarray(np.asarray(self.X, np.float32))
        if self.X.ndim != 2:
            raise ValueError(f"X must be 2-D, got shape {self.X.shape}")
        n = len(self.X)
        self.y = np.asarray(self.y, np.int64).reshape(-1)
        if len(self.y) != n:
            raise ValueError(f"y has {len(self.y)} rows, X has {n}")
        if len(self.feature_names) != self.X.shape[1]:
            raise ValueError(f"{len(self.feature_names)} feature names for {self.X.shape[1]} columns")
        if self.ids is None:
            self.ids = np.array([str(i) for i in range(n)], dtype=object)
        self.ids = np.asarray(self.ids, dtype=object)
        if self.weight is None:
            self.weight = np.ones(n)
        self.weight = np.asarray(self.weight, float)
        if self.source is None:
            self.source = np.full(n, "ground_truth", dtype=object)
        self.source = np.asarray(self.source, dtype=object)
        bad = sorted(set(self.source.tolist()) - set(SOURCES))
        if bad:
            raise ValueError(f"unknown label sources {bad}; use one of {SOURCES}")
        if self.trusted is None:
            self.trusted = np.isin(self.source, TRUSTED_SOURCES)
        self.trusted = np.asarray(self.trusted, bool) & (self.source != "feedback") & (self.source != "judge")
        self.holdout_ok = np.ones(n, bool) if self.holdout_ok is None else np.asarray(self.holdout_ok, bool)
        if self.t is not None:
            self.t = np.asarray(self.t, np.int64)
            self.t_end = self.t.copy() if self.t_end is None else np.maximum(np.asarray(self.t_end, np.int64), self.t)
        if self.group is not None:
            self.group = np.asarray([str(g) for g in self.group], dtype=object)
        if self.segment is not None:
            self.segment = np.asarray([str(s) for s in self.segment], dtype=object)
        if self.judge_y is not None:
            self.judge_y = np.asarray(self.judge_y, np.int64)
        for name in ("ids", "weight", "source", "trusted", "t", "t_end", "group", "segment", "judge_y", "holdout_ok"):
            a = getattr(self, name)
            if a is not None and len(a) != n:
                raise ValueError(f"{name} has {len(a)} rows, X has {n}")

    def __len__(self) -> int:
        return len(self.y)

    def subset(self, idx: np.ndarray) -> Dataset:
        idx = np.asarray(idx)
        take = {}
        for name in ("X", "y", "ids", "t", "t_end", "group", "weight", "trusted", "source", "judge_y", "segment",
                     "holdout_ok"):
            a = getattr(self, name)
            take[name] = None if a is None else a[idx]
        return replace(self, **take, notes=list(self.notes))

    def counts(self) -> dict:
        src, n = self.source.tolist(), len(self)
        return {"rows": n, "trusted": int(self.trusted.sum()),
                "by_source": {s: src.count(s) for s in SOURCES if s in src},
                "groups": None if self.group is None else int(len(set(self.group.tolist()))),
                "time_from": None if self.t is None or n == 0 else int(self.t.min()),
                "time_to": None if self.t is None or n == 0 else int(self.t.max())}


def concat(parts: list[Dataset]) -> Dataset:
    """Stack datasets of the same feature schema (e.g. labelled rows + feedback rows)."""
    parts = [p for p in parts if p is not None and len(p)]
    if not parts:
        raise ValueError("nothing to concatenate")
    first = parts[0]
    for p in parts[1:]:
        if p.feature_names != first.feature_names or p.feature_version != first.feature_version:
            raise ValueError("datasets have different feature schemas")

    def cat(name: str, fill=None):
        arrs = [getattr(p, name) for p in parts]
        if all(a is None for a in arrs):
            return None
        out = []
        for p, a in zip(parts, arrs, strict=True):
            if a is None:
                if fill is None:
                    return None
                a = np.full(len(p), fill, dtype=object if isinstance(fill, str) else type(fill))
            out.append(a)
        return np.concatenate(out)

    return Dataset(X=np.vstack([p.X for p in parts]), y=np.concatenate([p.y for p in parts]),
                   feature_names=first.feature_names, feature_version=first.feature_version,
                   # rows without a time (e.g. feedback without a timestamp) get t = -1: train side only
                   ids=cat("ids"), t=cat("t", -1), t_end=cat("t_end", -1), group=cat("group", "-"),
                   weight=np.concatenate([p.weight for p in parts]),
                   trusted=np.concatenate([p.trusted for p in parts]),
                   source=np.concatenate([p.source for p in parts]), judge_y=cat("judge_y", -1),
                   segment=cat("segment", "-"), feature_groups=first.feature_groups,
                   holdout_ok=np.concatenate([p.holdout_ok for p in parts]),
                   notes=[n for p in parts for n in p.notes])
