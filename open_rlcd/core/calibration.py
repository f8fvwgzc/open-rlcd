"""Calibrators and conformal abstention. Every fitted object here is plain JSON (no pickle).

Calibrators map a base model's raw scores (logits) to probabilities:

  * sigmoid      Platt scaling p = 1 / (1 + exp(-(a*s + b))), fitted by maximum likelihood on Platt's smoothed
                 targets ((N+ + 1)/(N+ + 2) for positives, 1/(N- + 2) for negatives), as scikit-learn does.
  * isotonic     a monotone step function (knots stored), linear interpolation between knots, clipped.
  * temperature  one parameter T: softmax(S / T) for multiclass, sigmoid(s / T) for binary.

Binary heads calibrate the positive-class score. Multiclass heads calibrate each class one-vs-rest and
renormalise the rows (the scheme `CalibratedClassifierCV` uses), but on the log-softmax of the scores rather
than the raw logit: a class's own logit ignores the competing classes and comes out underconfident (measured:
top-label ECE 0.07-0.08 vs about 0.03 on a known multinomial generator). Temperature scales the softmax.

Method choice ("auto"): isotonic only when every class has at least `isotonic_min_per_class` rows in every
calibration block (isotonic overfits small samples), otherwise sigmoid.

Conformal (split-conformal, on rows not used to fit the base model):
  * selective_threshold   for gate heads: the lowest threshold t on P(true) such that, with probability
                          >= 1 - delta, the error rate among rows with P(true) >= t is <= target_error.
                          Fixed-sequence testing (Learn-then-Test) over thresholds of decreasing strictness,
                          each tested with an exact one-sided binomial test; stops at the first failure.
  * set_quantile          for choice / score heads: the split-conformal quantile q of 1 - P(true class);
                          the prediction set {k : 1 - P_k <= q} covers the truth with probability >= 1 - alpha.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy import optimize, stats

ISOTONIC_MIN_PER_CLASS = 3000
METHODS = ("sigmoid", "isotonic", "temperature")


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + np.tanh(0.5 * np.asarray(z, float)))


def _softmax(S: np.ndarray) -> np.ndarray:
    S = np.asarray(S, float)
    S = S - S.max(axis=1, keepdims=True)
    E = np.exp(S)
    return E / E.sum(axis=1, keepdims=True)


def _log_softmax(S: np.ndarray) -> np.ndarray:
    S = np.asarray(S, float)
    S = S - S.max(axis=1, keepdims=True)
    return S - np.log(np.exp(S).sum(axis=1, keepdims=True))


# ----------------------------------------------------------------------------- 1-D calibrators

def fit_platt(s: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """(a, b) of p = sigmoid(a*s + b), maximum likelihood on Platt's smoothed targets."""
    s, y = np.asarray(s, float), np.asarray(y, int)
    n_pos, n_neg = int((y == 1).sum()), int((y == 0).sum())
    t = np.where(y == 1, (n_pos + 1.0) / (n_pos + 2.0), 1.0 / (n_neg + 2.0))
    scale = float(np.std(s)) or 1.0

    def nll(theta):
        a, b = theta
        z = a * s / scale + b
        # log(1 + exp(z)) computed stably
        lse = np.logaddexp(0.0, z)
        loss = float(np.sum(lse - t * z))
        p = _sigmoid(z)
        g = p - t
        return loss, np.array([float(np.sum(g * s / scale)), float(np.sum(g))])

    b0 = math.log((n_pos + 1.0) / (n_neg + 1.0))
    res = optimize.minimize(nll, np.array([1.0, b0]), jac=True, method="L-BFGS-B")
    a, b = res.x
    return float(a / scale), float(b)


def fit_isotonic(s: np.ndarray, y: np.ndarray) -> tuple[list[float], list[float]]:
    from sklearn.isotonic import IsotonicRegression

    s, y = np.asarray(s, float), np.asarray(y, float)
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0, increasing=True).fit(s, y)
    return [float(v) for v in iso.X_thresholds_], [float(v) for v in iso.y_thresholds_]


def _apply_1d(params: dict, s: np.ndarray) -> np.ndarray:
    s = np.asarray(s, float)
    if params["method"] == "sigmoid":
        return _sigmoid(params["a"] * s + params["b"])
    if params["method"] == "isotonic":
        xs, ys = np.asarray(params["x"], float), np.asarray(params["y"], float)
        if len(xs) == 0:
            return np.full(len(s), 0.5)
        if len(xs) == 1:
            return np.full(len(s), ys[0])
        return np.interp(s, xs, ys)
    raise ValueError(f"unknown 1-D calibration method {params['method']!r}")


def _fit_1d(method: str, s: np.ndarray, y01: np.ndarray) -> dict:
    if method == "isotonic":
        x, yy = fit_isotonic(s, y01)
        return {"method": "isotonic", "x": x, "y": yy}
    a, b = fit_platt(s, y01)
    return {"method": "sigmoid", "a": a, "b": b}


def fit_temperature(S: np.ndarray, y: np.ndarray, binary: bool) -> float:
    S, y = np.asarray(S, float), np.asarray(y, int)

    def nll(log_t):
        T = math.exp(float(log_t[0]))
        if binary:
            z = S / T
            return float(np.sum(np.logaddexp(0.0, z) - y * z))
        P = _softmax(S / T)
        return float(-np.sum(np.log(np.clip(P[np.arange(len(y)), y], 1e-12, 1.0))))

    res = optimize.minimize(nll, np.array([0.0]), method="L-BFGS-B", bounds=[(-5.0, 5.0)])
    return float(math.exp(res.x[0]))


# ----------------------------------------------------------------------------- the calibrator

class Calibrator:
    """Maps raw scores to calibrated probabilities over `n_classes` classes.

    `scores` is 1-D (the positive-class logit) for binary heads and (n, K) for multiclass heads.
    `predict` always returns (n, K) probabilities in class order (binary: [P(false), P(true)])."""

    def __init__(self, method: str, n_classes: int, params: Any = None):
        if method not in METHODS:
            raise ValueError(f"calibration method must be one of {METHODS}, got {method!r}")
        self.method, self.n_classes, self.params = method, int(n_classes), params

    @property
    def binary(self) -> bool:
        return self.n_classes == 2

    def fit(self, scores: np.ndarray, y: np.ndarray) -> Calibrator:
        y = np.asarray(y, int)
        S = np.asarray(scores, float)
        if self.method == "temperature":
            self.params = {"T": fit_temperature(S, y, self.binary)}
        elif self.binary:
            self.params = _fit_1d(self.method, S.reshape(-1), (y == 1).astype(int))
        else:
            L = _log_softmax(S)
            self.params = [_fit_1d(self.method, L[:, k], (y == k).astype(int)) for k in range(self.n_classes)]
        return self

    def predict(self, scores: np.ndarray) -> np.ndarray:
        S = np.asarray(scores, float)
        if self.method == "temperature":
            T = float(self.params["T"])
            if self.binary:
                p = _sigmoid(S.reshape(-1) / T)
                return np.column_stack([1.0 - p, p])
            return _softmax(S / T)
        if self.binary:
            p = np.clip(_apply_1d(self.params, S.reshape(-1)), 0.0, 1.0)
            return np.column_stack([1.0 - p, p])
        L = _log_softmax(S)
        P = np.column_stack([_apply_1d(self.params[k], L[:, k]) for k in range(self.n_classes)])
        tot = P.sum(axis=1, keepdims=True)
        uniform = np.full_like(P, 1.0 / self.n_classes)
        return np.where(tot > 0, P / np.where(tot > 0, tot, 1.0), uniform)

    def to_dict(self) -> dict:
        return {"method": self.method, "n_classes": self.n_classes, "params": self.params}

    @classmethod
    def from_dict(cls, d: dict) -> Calibrator:
        return cls(d["method"], int(d["n_classes"]), d["params"])


def choose_method(requested: str, calibration_labels: list[np.ndarray], n_classes: int,
                  isotonic_min_per_class: int = ISOTONIC_MIN_PER_CLASS) -> tuple[str, str]:
    """(method, why). `calibration_labels`: the trusted labels of each calibration block."""
    if requested in METHODS:
        return requested, "requested"
    least = min((int(np.bincount(np.asarray(y, int), minlength=n_classes).min()) for y in calibration_labels),
                default=0)
    if least >= isotonic_min_per_class:
        return "isotonic", f"every class has >= {isotonic_min_per_class} trusted rows in every calibration block"
    return "sigmoid", (f"sigmoid: the smallest class has {least} trusted rows in some calibration block "
                       f"(isotonic needs >= {isotonic_min_per_class})")


# ----------------------------------------------------------------------------- conformal

def min_rows_for(target_error: float, delta: float) -> int:
    """Fewest selected rows with which zero errors can establish error <= target at level delta."""
    return int(math.ceil(math.log(delta) / math.log(1.0 - target_error)))


def selective_threshold(p: np.ndarray, correct: np.ndarray, target_error: float, delta: float = 0.1,
                        max_candidates: int = 200) -> dict:
    """Learn-then-Test fixed-sequence search for the lowest threshold on `p` whose selective error rate is
    <= `target_error` with probability >= 1 - delta. `correct[i]` = the row's decision would be right.

    Candidate thresholds depend only on the scores (never on the labels): the j-th largest score for
    j = n_min, ... (n_min = fewest rows that could ever pass the test), in decreasing order. Each is tested
    with an exact one-sided binomial test of H0: error >= target; the search stops at the first failure."""
    p, correct = np.asarray(p, float), np.asarray(correct, bool)
    n = len(p)
    n_min = min_rows_for(target_error, delta)
    base = {"target_error": target_error, "delta": delta, "n_calibration": int(n), "min_rows": n_min,
            "method": "split-conformal selective threshold: Learn-then-Test fixed-sequence search, exact "
                      "one-sided binomial test per threshold"}
    if n < n_min:
        return {**base, "threshold": None,
                "reason": f"{n} calibration rows; at least {n_min} are needed to promise an error rate <= "
                          f"{target_error} at confidence {1 - delta:g}"}
    order = np.sort(p)[::-1]
    js = np.unique(np.linspace(n_min, n, num=min(max_candidates, n - n_min + 1)).round().astype(int))
    best = None
    for j in js:
        t = float(order[j - 1])
        sel = p >= t
        k = int(sel.sum())
        errors = int((~correct[sel]).sum())
        pv = float(stats.binom.cdf(errors, k, target_error))
        if pv > delta:
            break
        best = {"threshold": t, "n_selected": k, "errors": errors, "coverage": k / n, "p_value": pv}
    if best is None:
        return {**base, "threshold": None,
                "reason": f"no threshold keeps the error rate <= {target_error} at confidence {1 - delta:g} "
                          f"on {n} calibration rows"}
    return {**base, **{k: (round(v, 6) if isinstance(v, float) else v) for k, v in best.items()}}


def selective_check(p: np.ndarray, correct: np.ndarray, threshold: float | None, target_error: float) -> dict:
    """The promised error rate checked on rows the threshold was not chosen on (the holdout)."""
    from .metrics import _f, wilson

    p, correct = np.asarray(p, float), np.asarray(correct, bool)
    if threshold is None or len(p) == 0:
        return {"n": int(len(p)), "n_selected": 0, "coverage": None, "error_rate": None}
    sel = p >= threshold
    k, errors = int(sel.sum()), int((~correct[sel]).sum())
    lo, hi = wilson(errors, k)
    pv = float(stats.binom.sf(errors - 1, k, target_error)) if k else None  # H0: error <= target
    return {"n": int(len(p)), "n_selected": k, "coverage": _f(k / len(p)),
            "error_rate": _f(errors / k) if k else None, "error_wilson_95": [_f(lo), _f(hi)],
            "p_value_exceeds_target": _f(pv),
            "consistent": None if not k else bool(pv is None or pv > 0.05),
            "note": "consistent = the holdout does not show (one-sided p <= 0.05) an error rate above the target"}


def set_quantile(p_true: np.ndarray, alpha: float) -> dict:
    """Split-conformal quantile of the nonconformity 1 - P(true class) at miscoverage alpha."""
    s = 1.0 - np.asarray(p_true, float)
    n = len(s)
    if n == 0:
        return {"alpha": alpha, "q": None, "n_calibration": 0}
    level = math.ceil((n + 1) * (1 - alpha)) / n
    q = 1.0 if level > 1 else float(np.quantile(s, level, method="higher"))
    return {"alpha": alpha, "q": round(q, 6), "n_calibration": int(n),
            "method": "split-conformal: prediction set {k : 1 - P_k <= q}"}


def prediction_set(P: np.ndarray, q: float | None) -> list[int] | None:
    if q is None:
        return None
    P = np.asarray(P, float).reshape(-1)
    return [int(k) for k in np.flatnonzero(q + 1e-12 >= 1.0 - P)]
