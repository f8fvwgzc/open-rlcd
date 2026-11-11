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
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from scipy import stats

from .locks import FileLock, write_atomic

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
    in bin k. Stored as bernoulli.json under RLCD_HOME; one table per head, tied to one model version.

    Several processes share the file (the server scores outcomes while `rlcd train` / `promote` restart tables).
    Nothing is cached here: every call reads the file, and every read-modify-write (`reset`, `update`) holds the
    inter-process lock bernoulli.json.lock from its read to its atomic replace, so no writer works on a stale copy.
    The archive file is written under the same lock. Readers take no file lock: they see a whole file, old or new."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self._lock = threading.RLock()
        self._file_lock = FileLock(self.path)

    def _read(self) -> dict:
        if self.path.exists():
            try:
                return json.loads(self.path.read_text())
            except json.JSONDecodeError:
                return {}
        return {}

    def _write(self, data: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        write_atomic(self.path, json.dumps(data))

    @contextmanager
    def exclusive(self) -> Iterator[None]:
        """Hold the ledger over several calls that belong together (read a table, then `reset` it from what was
        read): no other thread or process writes in between. `reset` and `update` take the same lock themselves."""
        with self._lock, self._file_lock:
            yield

    @staticmethod
    def empty(classes: Sequence[str] = (), model_version: str | None = None) -> dict:
        return {"model_version": model_version, "classes": list(classes),
                "bins": {c: [[0, 0] for _ in range(BINS)] for c in classes},
                "holdout_trials": 0, "online_trials": 0, "applied": [], "outcomes": {}}

    def table(self, head: str) -> dict | None:
        with self._lock:
            return self._read().get(head)

    def reset(self, head: str, table: dict) -> None:
        """Start a head's table again from the holdout of a freshly trained model. The table it replaces is kept in
        bernoulli.archive.json under every version it was valid for, so a version that is pinned or rolled back to
        later gates on its own evidence (holdout + the outcomes scored for it), not on its bare holdout."""
        with self._lock, self._file_lock:
            data = self._read()
            old = data.get(head)
            if isinstance(old, dict):
                arch = self._archive_read()
                versions = {v for v in [old.get("model_version"), *(old.get("valid_for") or [])] if v}
                for v in versions:
                    arch.setdefault(head, {})[v] = old
                if versions:
                    self._archive_write(arch)
            data[head] = table
            self._write(data)

    @property
    def _archive_path(self) -> Path:
        return self.path.with_name("bernoulli.archive.json")

    def _archive_read(self) -> dict:
        try:
            return json.loads(self._archive_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, ValueError):
            return {}

    def _archive_write(self, data: dict) -> None:
        write_atomic(self._archive_path, json.dumps(data))

    def archived(self, head: str, version: str) -> dict | None:
        """The last live table of `head` that was valid for `version`, if one was archived."""
        with self._lock:
            return (self._archive_read().get(head) or {}).get(version)

    def update(self, head: str, decision_id: str, chosen: str, p: float, label: str) -> dict | None:
        """Score one decision against its realised outcome. Each decision_id counts once per head.

        A later outcome for the same decision_id with a different label is a correction: its count moves from the
        old cell (win / loss) to the new one, so the ledger keeps the latest outcome, as training does (the latest
        feedback row per decision_id). The same outcome posted again changes nothing."""
        with self._lock, self._file_lock:
            data = self._read()
            t = data.get(head)
            if t is None or chosen not in t["bins"]:
                return None
            outcomes = t.setdefault("outcomes", {})
            k = bin_of(p)
            cell = t["bins"][chosen][k]
            correct = label == chosen
            if decision_id in t["applied"]:
                prev = outcomes.get(decision_id)
                if prev is None or bool(prev.get("correct")) == correct:
                    return {"updated": False, "reason": "this decision_id was already scored"}
                old = t["bins"].get(prev["class"], [[0, 0]] * BINS)[int(prev["bin"])]
                j = 0 if prev["correct"] else 1
                old[j] = max(0, int(old[j]) - 1)
                cell[0 if correct else 1] += 1
                outcomes[decision_id] = {"class": chosen, "bin": k, "correct": correct}
                data[head] = t
                self._write(data)
                return {"updated": True, "corrected": True, "class": chosen, "bin": k, "wins": cell[0],
                        "losses": cell[1], "correct": correct}
            cell[0 if correct else 1] += 1
            t["applied"].append(decision_id)
            outcomes[decision_id] = {"class": chosen, "bin": k, "correct": correct}
            t["online_trials"] = int(t.get("online_trials", 0)) + 1
            data[head] = t
            self._write(data)
            return {"updated": True, "class": chosen, "bin": k, "wins": cell[0], "losses": cell[1],
                    "correct": correct}


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
