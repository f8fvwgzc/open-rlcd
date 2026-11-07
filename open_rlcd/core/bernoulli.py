"""Every decision as a Bernoulli trial.

A decision picks a class (an option of a choice head, a level of a score head, true/false of a noul head).
It is either right (success) or wrong. The calibrated probability says what the model believes; the
Beta-Bernoulli posterior says what has actually happened to predictions like this one.

Counts are kept per (head, chosen class, probability bin), start from the trusted holdout rows of the model
version being served, and move with every outcome posted to /v1/feedback with a decision_id. No retrain is
involved: that online update is the reinforcement step, and it is exactly as modest as it sounds. `act`
requires the posterior to put at least `min_p_above` of its mass above the stakes level's `min_accuracy`.
"""
from __future__ import annotations

import json
import math
import threading
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from scipy import stats

BINS = 10


@dataclass(frozen=True)
class BernoulliConfig:
    prior_alpha: float = 1.0      # Beta(1, 1): uniform prior on the true accuracy
    prior_beta: float = 1.0
    min_p_above: float = 0.9      # the gate for `act`
    alpha: float = 0.05           # one-sided significance for the binomial test and the sample size
    power: float = 0.8

    def to_dict(self) -> dict:
        return asdict(self)


def _r(x: float | None, nd: int = 6) -> float | None:
    return None if x is None or not math.isfinite(x) else round(float(x), nd)


def posterior(wins: float, losses: float, p0: float, cfg: BernoulliConfig = BernoulliConfig()) -> dict:
    """Beta(prior_alpha + wins, prior_beta + losses) for the true accuracy, against the required level p0."""
    a, b = cfg.prior_alpha + wins, cfg.prior_beta + losses
    lo, hi = stats.beta.ppf([0.025, 0.975], a, b)
    return {"alpha": _r(a, 3), "beta": _r(b, 3), "n": int(round(wins + losses)), "mean": _r(a / (a + b)),
            "ci95": [_r(lo), _r(hi)], "required": _r(p0),
            "p_above_required": _r(float(stats.beta.sf(p0, a, b)))}


def thompson(wins: float, losses: float, rng: np.random.Generator,
             cfg: BernoulliConfig = BernoulliConfig()) -> float:
    """One draw of the accuracy from its posterior (exploration: under-sampled bins get tried sometimes)."""
    return float(rng.beta(cfg.prior_alpha + wins, cfg.prior_beta + losses))


def binomial_test(wins: int, n: int, p0: float) -> float | None:
    """Exact one-sided p-value: P(X >= wins) for X ~ Binomial(n, p0). Small = the rate beats p0."""
    if n <= 0:
        return None
    return float(stats.binom.sf(wins - 1, n, p0))


def binomial_test_below(wins: int, n: int, p0: float) -> float | None:
    """Exact one-sided p-value: P(X <= wins) for X ~ Binomial(n, p0). Small = the rate falls short of p0."""
    if n <= 0:
        return None
    return float(stats.binom.cdf(wins, n, p0))


def decisions_needed(p1: float | None, p0: float, cfg: BernoulliConfig = BernoulliConfig()) -> int | None:
    """Decisions needed to tell a true accuracy p1 from the required level p0 (one-sided test at `alpha`,
    `power`): n = ((z_a*sqrt(p0*(1-p0)) + z_b*sqrt(p1*(1-p1))) / (p1 - p0))^2. None when p1 is not above p0."""
    if p1 is None or not p0 < p1 < 1:
        return None
    za, zb = stats.norm.ppf(1 - cfg.alpha), stats.norm.ppf(cfg.power)
    n = ((za * math.sqrt(p0 * (1 - p0)) + zb * math.sqrt(p1 * (1 - p1))) / (p1 - p0)) ** 2
    return int(math.ceil(n))


def bin_of(p: float) -> int:
    return int(min(BINS - 1, max(0, math.floor(float(p) * BINS))))


# ----------------------------------------------------------------------------- the durable table

class Ledger:
    """Right/wrong counts per head: `bins[class][k]` for decisions whose probability of the chosen class fell
    in bin k. Stored as bernoulli.json under RLCD_HOME; one table per head, tied to one model version."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self._lock = threading.RLock()

    def _read(self) -> dict:
        if self.path.exists():
            try:
                return json.loads(self.path.read_text())
            except json.JSONDecodeError:
                return {}
        return {}

    def _write(self, data: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(data))
        tmp.replace(self.path)

    @staticmethod
    def empty(classes: Sequence[str] = (), model_version: str | None = None) -> dict:
        return {"model_version": model_version, "classes": list(classes),
                "bins": {c: [[0, 0] for _ in range(BINS)] for c in classes},
                "holdout_trials": 0, "online_trials": 0, "applied": []}

    def table(self, head: str) -> dict | None:
        with self._lock:
            return self._read().get(head)

    def reset(self, head: str, table: dict) -> None:
        """Start a head's table again from the holdout of a freshly trained model."""
        with self._lock:
            data = self._read()
            data[head] = table
            self._write(data)

    def update(self, head: str, decision_id: str, chosen: str, p: float, label: str) -> dict | None:
        """Score one decision against its realised outcome. Each decision_id counts once per head."""
        with self._lock:
            data = self._read()
            t = data.get(head)
            if t is None or chosen not in t["bins"]:
                return None
            if decision_id in t["applied"]:
                return {"updated": False, "reason": "this decision_id was already scored"}
            k = bin_of(p)
            cell = t["bins"][chosen][k]
            cell[0 if label == chosen else 1] += 1
            t["applied"].append(decision_id)
            t["online_trials"] = int(t.get("online_trials", 0)) + 1
            data[head] = t
            self._write(data)
            return {"updated": True, "class": chosen, "bin": k, "wins": cell[0], "losses": cell[1],
                    "correct": label == chosen}


def holdout_table(P: np.ndarray, y: np.ndarray, classes: Sequence[str], chosen: np.ndarray,
                  model_version: str | None = None) -> dict:
    """Initial counts from the trusted holdout. `P` is (n, K) in `classes` order, `y` the true class index,
    `chosen` the class index the policy picks for each row."""
    P, y, chosen = np.asarray(P, float), np.asarray(y, int), np.asarray(chosen, int)
    t = Ledger.empty(classes, model_version)
    if len(y):
        p = P[np.arange(len(y)), chosen]
        k = np.minimum((p * BINS).astype(int), BINS - 1)
        win = chosen == y
        for ci, c in enumerate(classes):
            for b in range(BINS):
                m = (chosen == ci) & (k == b)
                t["bins"][c][b] = [int((m & win).sum()), int((m & ~win).sum())]
    t["holdout_trials"] = int(len(y))
    return t


def table_view(table: dict, required: float, cfg: BernoulliConfig = BernoulliConfig()) -> dict:
    """The posterior table for /v1/bernoulli: one row per (chosen class, probability bin)."""
    rows = []
    for c in table.get("classes", []):
        for k, (w, l_) in enumerate(table["bins"][c]):
            rows.append({"class": c, "bin": k, "lo": round(k / BINS, 2), "hi": round((k + 1) / BINS, 2),
                         "wins": w, "losses": l_, **posterior(w, l_, required, cfg)})
    return {"bins": rows, "holdout_trials": table.get("holdout_trials", 0),
            "online_trials": table.get("online_trials", 0), "model_version": table.get("model_version")}
