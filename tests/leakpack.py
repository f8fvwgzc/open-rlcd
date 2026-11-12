"""A pack for weight-edit tests: the toy generator plus one planted spurious feature.

`leak_rows(n, holds)` draws the six genuine features and a binary label from `toypack.generate` (true
probabilities known), adds a column `leak` = (2y - 1) * strength + noise, and orders the rows in time (one row per
hour), so the holdout is the latest 20 %. With `holds=False` the leak breaks exactly where the holdout starts (it
becomes noise with the same spread): a head trained on development rows leans on a shortcut that the holdout
does not have, and an edit that removes the leak's coefficients should help there. With `holds=True` the leak is
real everywhere, so removing it should hurt. The `clean` head has no leak at all.
"""
from __future__ import annotations

import numpy as np

from open_rlcd.core.dataset import Dataset
from open_rlcd.core.head import HeadSpec
from open_rlcd.packs.base import BasePack
from tests.toypack import NAMES, ToyProvider, generate

LEAK_NAMES = [*NAMES, "leak"]
T0 = 1_700_000_000


def leak_rows(n: int = 3000, holds: bool = False, strength: float = 1.5, seed: int = 0) -> Dataset:
    X, y, _ = generate(n, 2, 1.0, seed=seed)
    rng = np.random.default_rng(seed + 1000)
    leak = (2 * y - 1) * strength + rng.normal(size=n)
    cut = int(n * 0.8)
    if not holds:
        leak[cut:] = rng.normal(size=n - cut) * np.sqrt(strength ** 2 + 1)
    return Dataset(X=np.column_stack([X, leak]), y=y, feature_names=LEAK_NAMES, feature_version="leak-v1",
                   t=T0 + 3600 * np.arange(n), source=["synthetic"] * n)


def clean_rows(n: int = 3000, seed: int = 3) -> Dataset:
    X, y, _ = generate(n, 2, 1.5, seed=seed)
    return Dataset(X=X, y=y, feature_names=list(NAMES), feature_version="leak-v1",
                   group=[f"g{j // 3}" for j in range(n)], source=["synthetic"] * n)


class LeakProvider(ToyProvider):
    feature_version = "leak-v1"

    def names(self, view):
        return list(LEAK_NAMES) if view == "leaky" else list(NAMES)

    def transform(self, view, states):
        X = super().transform("v", [{"f": s["f"][:6]} if isinstance(s, dict) and "f" in s else s for s in states])
        if view != "leaky":
            return X
        leak = np.asarray([float(s.get("leak", 0.0)) for s in states], np.float32).reshape(-1, 1)
        return np.hstack([X, leak])


class LeakPack(BasePack):
    """`good`: noul trained on leak_rows (the leak breaks in the holdout unless the class attribute `holds` is
    set); `clean`: noul without a leak."""
    name = "leak"
    abstain = "escalate"
    holds = False

    def heads(self, project):
        return {"good": HeadSpec("good", "noul", view="leaky"), "clean": HeadSpec("clean", "noul", view="v")}

    def provider(self, project, config=None):
        return LeakProvider()

    def datasets(self, project, provider, heads, feedback=None, feedback_weight=5.0):
        rows = {"good": lambda: leak_rows(holds=self.holds), "clean": clean_rows}
        return {h: rows[h]() for h in heads}


class HoldingLeakPack(LeakPack):
    holds = True
