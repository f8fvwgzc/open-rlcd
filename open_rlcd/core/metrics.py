"""Calibration metrics. Everything here is computed on rows the model never saw (the holdout).

Definitions used throughout:
  * Brier (multiclass)  mean over rows of sum_k (p_k - y_k)^2   (0 = perfect, 2 = worst)
  * Brier (binary)      mean (p - y)^2
  * ECE                 sum over 10 equal-width bins of (count/n) * |mean predicted - observed rate|
  * baseline            the base-rate forecaster: predicts the class frequencies of the trusted training rows
  * Brier skill score   1 - brier / brier_baseline   (> 0 = better than the base rate)
  * skill test          block bootstrap of the per-row Brier gain over the baseline; skill is claimed only when
                        the one-sided 95% lower bound of the mean gain is > 0
"""
from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

EPS = 1e-12
BINS = 10


def _f(x: float | None) -> float | None:
    if x is None:
        return None
    x = float(x)
    return round(x, 6) if math.isfinite(x) else None


def wilson(k: int, n: int, z: float = 1.959964) -> tuple[float | None, float | None]:
    """Wilson score interval for a binomial rate (95% by default)."""
    if n <= 0:
        return None, None
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    # the interval always contains p mathematically; the clamp keeps rounding (k = n gives 0.9999999999999999)
    # from excluding the observed rate
    return max(0.0, min(p, c - h)), min(1.0, max(p, c + h))


def reliability_table(p: np.ndarray, y: np.ndarray, bins: int = BINS) -> list[dict]:
    """10 equal-width probability bins: mean predicted probability, observed rate, count."""
    p, y = np.asarray(p, float), np.asarray(y, float)
    idx = np.minimum((p * bins).astype(int), bins - 1)
    out = []
    for b in range(bins):
        m = idx == b
        n = int(m.sum())
        out.append({"bin": b, "lo": round(b / bins, 2), "hi": round((b + 1) / bins, 2), "count": n,
                    "mean_predicted": _f(p[m].mean()) if n else None,
                    "observed_rate": _f(y[m].mean()) if n else None})
    return out


def ece(p: np.ndarray, y: np.ndarray, bins: int = BINS) -> float:
    p, y = np.asarray(p, float), np.asarray(y, float)
    if len(p) == 0:
        return float("nan")
    idx = np.minimum((p * bins).astype(int), bins - 1)
    total = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            total += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(total)


def brier_rows(P: np.ndarray, Y: np.ndarray) -> np.ndarray:
    return ((P - Y) ** 2).sum(axis=1)


def log_loss_rows(P: np.ndarray, Y: np.ndarray) -> np.ndarray:
    return -(Y * np.log(np.clip(P, EPS, 1.0))).sum(axis=1)


def binary_report(p: np.ndarray, y: np.ndarray, base_rate) -> dict:
    """Reliability of one yes-probability against its 0/1 outcome, next to the base-rate forecaster
    (`base_rate`: a scalar, or one baseline probability per row)."""
    p, y = np.asarray(p, float), np.asarray(y, float)
    b = np.broadcast_to(np.asarray(base_rate, float), p.shape).astype(float)
    brier, brier_b = float(((p - y) ** 2).mean()), float(((b - y) ** 2).mean())

    def ll(q):
        q = np.clip(q, EPS, 1 - EPS)
        return float(-(y * np.log(q) + (1 - y) * np.log(1 - q)).mean())

    return {"n": int(len(p)), "observed_rate": _f(y.mean()), "base_rate_train": _f(b.mean()),
            "brier": _f(brier), "log_loss": _f(ll(p)), "ece": _f(ece(p, y)),
            "baseline": {"brier": _f(brier_b), "log_loss": _f(ll(b)), "ece": _f(ece(b, y))},
            "brier_skill_score": _f(1 - brier / brier_b) if brier_b > 0 else None,
            "reliability": reliability_table(p, y)}


def skill_test(gain: np.ndarray, blocks: np.ndarray | int = 20, n_boot: int = 2000, seed: int = 7) -> dict:
    """Is the per-row Brier gain over the base rate reliably positive?

    Rows are not independent (neighbouring rows in time share context; rows of one document or customer share
    everything), so a per-row standard error would be far too confident. The gain is averaged per block and the
    blocks are resampled with replacement (block bootstrap). `blocks` is either the number of contiguous blocks
    to cut the (already ordered) rows into, or one block key per row (e.g. the group id), in which case whole
    groups are resampled. Skill is claimed only when the one-sided 95% lower bound of the mean gain is > 0.
    """
    gain = np.asarray(gain, float)
    n = len(gain)
    if isinstance(blocks, (int, np.integer)):
        k = max(2, min(int(blocks), n // 5)) if n >= 10 else 0
        keys = np.repeat(np.arange(k), [len(c) for c in np.array_split(np.arange(n), k)]) if k else None
        how = f"{k} contiguous blocks"
    else:
        keys = np.asarray(blocks)
        k = len(np.unique(keys)) if n else 0
        how = f"{k} groups"
    if keys is None or k < 5 or n < 10:
        return {"has_skill": False, "mean_brier_gain": _f(gain.mean()) if n else None, "lower_95": None,
                "upper_95": None, "blocks": int(k), "n": int(n),
                "method": "too few holdout rows or blocks to test (need at least 10 rows in 5 blocks)"}
    _, inv = np.unique(keys, return_inverse=True)
    sums = np.bincount(inv, weights=gain)
    counts = np.bincount(inv).astype(float)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, k, size=(n_boot, k))
    boot = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
    m = float(gain.mean())
    lower, upper = float(np.quantile(boot, 0.05)), float(np.quantile(boot, 0.95))
    return {"has_skill": bool(m > 0 and lower > 0), "mean_brier_gain": _f(m), "lower_95": _f(lower),
            "upper_95": _f(upper), "blocks": int(k), "n": int(n), "n_boot": int(n_boot),
            "method": f"block bootstrap ({how}, {n_boot} resamples): one-sided 95% lower bound of the mean Brier "
                      f"gain over the base rate"}


def _block_keys(n: int, blocks: np.ndarray | int) -> tuple[np.ndarray | None, int, str]:
    if isinstance(blocks, (int, np.integer)):
        k = max(2, min(int(blocks), n // 5)) if n >= 10 else 0
        keys = np.repeat(np.arange(k), [len(c) for c in np.array_split(np.arange(n), k)]) if k else None
        return keys, k, f"{k} contiguous blocks"
    keys = np.asarray(blocks)
    k = len(np.unique(keys)) if n else 0
    return keys, k, f"{k} groups"


def resolution_test(P: np.ndarray, Y: np.ndarray, blocks: np.ndarray | int = 20, n_boot: int = 2000,
                    seed: int = 7) -> dict:
    """Brier gain of the forecasts over the holdout's OWN class frequencies, recomputed inside every block
    bootstrap resample. A constant forecast can never beat that baseline (its expected gain is minus its squared
    distance from the holdout frequencies), so a forecaster that merely sits closer to the holdout's base rate
    than the training base rate did cannot pass. A positive gain means resolution exceeds miscalibration."""
    P, Y = np.asarray(P, float), np.asarray(Y, float)
    n = len(P)
    keys, k, how = _block_keys(n, blocks)
    if keys is None or k < 5 or n < 10:
        return {"mean_brier_gain": None, "lower_95": None, "upper_95": None, "blocks": int(k)}
    model = ((P - Y) ** 2).sum(axis=1)
    _, inv = np.unique(keys, return_inverse=True)
    nb = np.bincount(inv).astype(float)
    mb = np.bincount(inv, weights=model)
    Sb = np.stack([np.bincount(inv, weights=Y[:, j], minlength=k) for j in range(Y.shape[1])], axis=1)

    def gain(S: np.ndarray, nn: np.ndarray, m: np.ndarray) -> np.ndarray:
        # sum_i sum_k (qbar_k - y_ik)^2 with qbar = S / n equals sum_k (S_k - S_k^2 / n)
        base = (S - S ** 2 / nn[..., None]).sum(axis=-1)
        return (base - m) / nn

    full = float(gain(Sb.sum(0), np.array(nb.sum()), np.array(mb.sum())))
    rng = np.random.default_rng(seed + 1)
    draws = rng.integers(0, k, size=(n_boot, k))
    boot = gain(Sb[draws].sum(axis=1), nb[draws].sum(axis=1), mb[draws].sum(axis=1))
    return {"mean_brier_gain": _f(full), "lower_95": _f(float(np.quantile(boot, 0.05))),
            "upper_95": _f(float(np.quantile(boot, 0.95))), "blocks": int(k), "resampled": how}


def combined_skill(gain_vs_train: np.ndarray, P: np.ndarray, Y: np.ndarray, blocks: np.ndarray | int = 20,
                   seed: int = 7) -> dict:
    """Skill = the Brier gain over the training base rate AND over the holdout's own class frequencies both
    have a one-sided 95% block-bootstrap lower bound above 0."""
    out = skill_test(gain_vs_train, blocks, seed=seed)
    res = resolution_test(P, Y, blocks, seed=seed)
    out["vs_holdout_base_rate"] = res
    res_ok = res.get("lower_95") is not None and res["lower_95"] > 0
    if out["has_skill"] and not res_ok:
        out["has_skill"] = False
        mean = res.get("mean_brier_gain")
        if mean is not None and mean > 0:
            out["why_not"] = (f"the gain over the holdout's own class frequencies averages {mean} but is not reliably "
                              f"positive across the {res.get('blocks')} blocks resampled (95% lower bound "
                              f"{res.get('lower_95')}): too few independent groups or time blocks to establish it; "
                              f"more trusted rows from more groups would settle it")
        else:
            out["why_not"] = ("the gain over the training base rate is not matched by a gain over the holdout's own "
                              "class frequencies: the forecasts do not separate outcomes, they only sit nearer to "
                              "this holdout's base rate")
    if out.get("lower_95") is not None:
        out["method"] = (out["method"] + "; and the same for the gain over the holdout's own class frequencies "
                         "(recomputed in every resample), which no constant forecast can beat")
    return out


def paired_gain_test(P_base: np.ndarray, P_cand: np.ndarray, y: np.ndarray, blocks: np.ndarray | int = 20,
                     alpha: float = 0.05, n_boot: int | None = None, seed: int = 7) -> dict:
    """Is a candidate model reliably better than a base model on the SAME holdout rows?

    Per row, the Brier gain and the log-loss gain of the candidate over the base (base loss minus candidate loss;
    > 0 = the candidate is better). Pairing removes the row-to-row difficulty both models share, so the test sees
    only how they differ. Rows are not independent, so the gains are summed per block (contiguous blocks of the
    scoring order, or one block key per row, e.g. the group) and whole blocks are resampled with replacement; both
    gains use the same resamples. `passed` needs the one-sided (1 - alpha) lower bound of BOTH mean gains > 0."""
    P_base, P_cand, y = np.asarray(P_base, float), np.asarray(P_cand, float), np.asarray(y, int)
    n, k = P_base.shape
    if P_cand.shape != P_base.shape or len(y) != n:
        raise ValueError("paired_gain_test needs both models' forecasts for the same rows")
    Y = np.eye(k)[y]
    gains = {"brier": brier_rows(P_base, Y) - brier_rows(P_cand, Y),
             "log_loss": log_loss_rows(P_base, Y) - log_loss_rows(P_cand, Y)}
    keys, nb, how = _block_keys(n, blocks)
    out: dict = {"n": int(n), "blocks": int(nb), "alpha": _f(alpha), "confidence": _f(1 - alpha)}
    if keys is None or nb < 5 or n < 10:
        for name, g in gains.items():
            out[name] = {"mean_gain": _f(g.mean()) if n else None, "lower": None, "upper": None}
        out.update(passed=False, method="too few holdout rows or blocks to test (need at least 10 rows in 5 blocks)")
        return out
    n_boot = int(n_boot or max(4000, math.ceil(200 / alpha)))
    _, inv = np.unique(keys, return_inverse=True)
    counts = np.bincount(inv).astype(float)
    draws = np.random.default_rng(seed).integers(0, nb, size=(n_boot, nb))
    denom = counts[draws].sum(axis=1)
    ok = True
    for name, g in gains.items():
        boot = np.bincount(inv, weights=g)[draws].sum(axis=1) / denom
        lower, upper = float(np.quantile(boot, alpha)), float(np.quantile(boot, 1 - alpha))
        out[name] = {"mean_gain": _f(g.mean()), "lower": _f(lower), "upper": _f(upper)}
        ok = ok and float(g.mean()) > 0 and lower > 0
    out.update(passed=bool(ok), n_boot=n_boot,
               method=f"paired block bootstrap ({how}, {n_boot} resamples) of the per-row Brier and log-loss gain "
                      f"of the candidate over the base on the same holdout rows: both one-sided {1 - alpha:.4g} "
                      f"lower bounds must be > 0")
    return out


def calibration_report(P: np.ndarray, y: np.ndarray, classes: Sequence[str], prior: np.ndarray,
                       baseline_note: str = "always predicts the class frequencies of the trusted training rows",
                       blocks: np.ndarray | int = 20, seed: int = 7) -> dict:
    """Full holdout report of a calibrated K-class forecaster. `y` holds class indices. `prior` is the
    baseline forecast: one row of training class frequencies, or one such row per holdout row."""
    P, y, prior = np.asarray(P, float), np.asarray(y, int), np.asarray(prior, float)
    n, k = P.shape
    Y = np.eye(k)[y]
    B = np.tile(prior, (n, 1)) if prior.ndim == 1 else prior
    brier, brier_b = brier_rows(P, Y), brier_rows(B, Y)
    hit = (P.argmax(axis=1) == y).astype(float)
    hit_b = (B.argmax(axis=1) == y).astype(float)
    conf, conf_b = P.max(axis=1), B.max(axis=1)
    per_class = {c: binary_report(P[:, i], Y[:, i], B[:, i]) for i, c in enumerate(classes)}
    bm, bbm = float(brier.mean()), float(brier_b.mean())
    return {
        "n_holdout": int(n), "classes": list(classes),
        "class_rates_holdout": {c: _f(Y[:, i].mean()) for i, c in enumerate(classes)},
        "base_rates_train": {c: _f(B[:, i].mean()) for i, c in enumerate(classes)},
        "brier": _f(bm), "log_loss": _f(log_loss_rows(P, Y).mean()), "accuracy": _f(hit.mean()),
        "ece": _f(ece(conf, hit)),
        "ece_classwise": _f(np.mean([per_class[c]["ece"] for c in classes])),
        "baseline": {"brier": _f(bbm), "log_loss": _f(log_loss_rows(B, Y).mean()), "accuracy": _f(hit_b.mean()),
                     "ece": _f(ece(conf_b, hit_b)),
                     "ece_classwise": _f(np.mean([per_class[c]["baseline"]["ece"] for c in classes]))},
        "brier_skill_score": _f(1 - bm / bbm) if bbm > 0 else None,
        "skill": combined_skill(brier_b - brier, P, Y, blocks, seed=seed),
        "reliability": {"top_label": reliability_table(conf, hit),
                        "per_class": {c: per_class[c]["reliability"] for c in classes}},
        "per_class": {c: {k_: v for k_, v in per_class[c].items() if k_ != "reliability"} for c in classes},
        "definitions": {
            "brier": "mean over rows of sum_k (p_k - y_k)^2",
            "ece": "top-label expected calibration error over 10 equal-width bins",
            "baseline": baseline_note,
            "brier_skill_score": "1 - brier / baseline brier; > 0 means better than the base rate",
        },
    }


def summary(report: dict) -> dict:
    """The few numbers shown in /v1/models, /v1/heads and decisions."""
    keys = ("n_holdout", "brier", "log_loss", "accuracy", "ece", "brier_skill_score")
    out = {k: report.get(k) for k in keys}
    out["baseline_brier"] = report.get("baseline", {}).get("brier")
    out["baseline_log_loss"] = report.get("baseline", {}).get("log_loss")
    out["baseline_accuracy"] = report.get("baseline", {}).get("accuracy")
    out["has_skill"] = bool(report.get("skill", {}).get("has_skill"))
    out["brier_gain_lower_95"] = report.get("skill", {}).get("lower_95")
    return out


def cohen_kappa(a: np.ndarray, b: np.ndarray, k: int) -> float | None:
    """Agreement between two labelers beyond chance (1 = perfect, 0 = chance level)."""
    a, b = np.asarray(a, int), np.asarray(b, int)
    n = len(a)
    if n == 0:
        return None
    cm = np.zeros((k, k))
    np.add.at(cm, (a, b), 1)
    po = np.trace(cm) / n
    pe = float((cm.sum(0) * cm.sum(1)).sum()) / (n * n)
    return None if pe >= 1 else float((po - pe) / (1 - pe))


def agreement(judge: np.ndarray, human: np.ndarray, classes: Sequence[str]) -> dict:
    """How often the LLM judge agrees with the human label, on rows that have both."""
    judge, human = np.asarray(judge, int), np.asarray(human, int)
    m = (judge >= 0) & (human >= 0)
    n = int(m.sum())
    if n == 0:
        return {"n": 0, "note": "no rows carry both a judge label and a human label"}
    k = len(classes)
    cm = np.zeros((k, k), int)
    np.add.at(cm, (human[m], judge[m]), 1)
    acc = float((judge[m] == human[m]).mean())
    lo, hi = wilson(int((judge[m] == human[m]).sum()), n)
    return {"n": n, "accuracy": _f(acc), "accuracy_wilson_95": [_f(lo), _f(hi)],
            "cohen_kappa": _f(cohen_kappa(human[m], judge[m], k)),
            "confusion": {"rows": "human label", "columns": "judge label", "classes": list(classes),
                          "counts": cm.tolist()}}
