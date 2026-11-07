"""A minimal pack for core tests: numeric feature vectors, labels from a known-calibrated generator.

`generate(n, k, signal)` draws X ~ N(0, I) and y ~ softmax(X @ W) with a fixed W per k: the true class
probabilities are known exactly, so a well-calibrated head must reproduce them on rows it never saw. signal = 0
is pure noise (y independent of X). Feedback rows posted through the service are added to training exactly as a
real pack would add them (source "feedback", up-weighted, a note), so the reinforcement loop can be tested.
"""
from __future__ import annotations

import numpy as np

from open_rlcd.core.contract import ContractError
from open_rlcd.core.dataset import Dataset, concat
from open_rlcd.core.head import HeadSpec
from open_rlcd.packs.base import BasePack

NAMES = [f"f{i}" for i in range(6)]


class ToyProvider:
    feature_version = "toy-v1"

    def names(self, view):
        return list(NAMES)

    def transform(self, view, states):
        rows = []
        for s in states:
            if not isinstance(s, dict) or not isinstance(s.get("f"), list) or len(s["f"]) != len(NAMES):
                raise ContractError("toy heads need state {f: [6 numbers]}")
            rows.append([float(v) for v in s["f"]])
        return np.asarray(rows, np.float32).reshape(len(rows), len(NAMES))

    def groups(self, view):
        return {n: ("a" if i < 3 else "b") for i, n in enumerate(NAMES)}

    def describe(self, view):
        return {"f": "array of 6 numbers"}

    def to_config(self):
        return {"kind": "toy"}


def generate(n, k, signal, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, len(NAMES))).astype(np.float32)
    W = np.random.default_rng(99 + k).normal(size=(len(NAMES), k)) * signal
    L = X @ W
    P = np.exp(L - L.max(1, keepdims=True))
    P /= P.sum(1, keepdims=True)
    y = (rng.random(n)[:, None] > np.cumsum(P, 1)).sum(1)
    return X, y, P


def feedback_dataset(spec: HeadSpec, provider: ToyProvider, feedback: list[dict] | None,
                     feedback_weight: float) -> Dataset | None:
    """Feedback rows for one head (raw `{f: [...]}` states only), latest row per decision_id."""
    rows = [r for r in feedback or [] if r.get("head") == spec.name and isinstance(r.get("state"), dict)
            and "f" in r["state"]]
    last = {r["decision_id"]: i for i, r in enumerate(rows) if r.get("decision_id")}
    rows = [r for i, r in enumerate(rows) if not r.get("decision_id") or last[r["decision_id"]] == i]
    if not rows:
        return None
    X = provider.transform(spec.view, [r["state"] for r in rows])
    y = [spec.label_index(r["label"]) for r in rows]
    w = [float(r.get("weight") or 1.0) * float(feedback_weight) for r in rows]
    return Dataset(X=X, y=y, feature_names=NAMES, feature_version=provider.feature_version,
                   ids=[f"feedback:{r.get('id')}" for r in rows],
                   group=[f"feedback:{r.get('decision_id') or r.get('id')}" for r in rows],
                   weight=w, source=["feedback"] * len(rows),
                   notes=[f"{len(rows)} feedback rows added to training (weight x {feedback_weight:g})"])


class ToyPack(BasePack):
    name = "toy"
    abstain = "escalate"
    rank_head = "good"

    def __init__(self, signal=1.5, n=3000):
        self.signal, self.n = signal, n

    def heads(self, project):
        return {"good": HeadSpec("good", "noul", view="v", decision="gate", conformal=0.1),
                "color": HeadSpec("color", "choice", ("red", "green", "blue"), view="v"),
                "level": HeadSpec("level", "score", ("0", "1", "2", "3"), view="v",
                                  levels=("none", "low", "mid", "high"))}

    def provider(self, project, config=None):
        return ToyProvider()

    def datasets(self, project, provider, heads, feedback=None, feedback_weight=5.0):
        out = {}
        specs = self.heads(project)
        for i, name in enumerate(heads):
            spec = specs[name]
            X, y, _ = generate(self.n, len(spec.labels), self.signal, seed=i)
            base = Dataset(X=X, y=y, feature_names=NAMES, feature_version="toy-v1",
                           group=[f"g{j // 3}" for j in range(self.n)], source=["synthetic"] * self.n)
            fb = feedback_dataset(spec, provider, feedback, feedback_weight)
            out[name] = base if fb is None else concat([base, fb])
        return out
