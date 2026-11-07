"""Calibrators (sigmoid / isotonic / temperature) and split-conformal abstention, on synthetic data whose true
probabilities are known."""
import json
import math

import numpy as np
import pytest

from open_rlcd.core import calibration as C


def _sigmoid(z):
    return 1 / (1 + np.exp(-z))


def _softmax(S):
    E = np.exp(S - S.max(1, keepdims=True))
    return E / E.sum(1, keepdims=True)


# ----------------------------------------------------------------------------- 1-D calibrators

def test_platt_recovers_slope_and_intercept():
    rng = np.random.default_rng(0)
    s = rng.normal(0, 2, 60000)
    y = (rng.random(60000) < _sigmoid(1.7 * s - 0.4)).astype(int)
    a, b = C.fit_platt(s, y)
    assert a == pytest.approx(1.7, abs=0.05) and b == pytest.approx(-0.4, abs=0.05)


def test_platt_survives_degenerate_inputs():
    a, b = C.fit_platt(np.zeros(50), np.r_[np.ones(10), np.zeros(40)].astype(int))   # constant scores
    assert math.isfinite(a) and _sigmoid(b) == pytest.approx(11 / 52, abs=0.02)         # Platt's smoothed rate
    a, b = C.fit_platt(np.random.default_rng(1).normal(size=40), np.zeros(40, int))     # one class only
    assert math.isfinite(a) and math.isfinite(b) and _sigmoid(b) < 0.1


def test_isotonic_is_monotone_and_clipped():
    rng = np.random.default_rng(2)
    s = rng.normal(size=4000)
    y = (rng.random(4000) < _sigmoid(2 * s)).astype(int)
    cal = C.Calibrator("isotonic", 2).fit(s, y)
    grid = np.linspace(-6, 6, 2001)
    p = cal.predict(grid)[:, 1]
    assert np.all(np.diff(p) >= -1e-12) and p.min() >= 0 and p.max() <= 1
    assert p[0] == pytest.approx(cal.predict(np.array([s.min()]))[0, 1])      # clipped below the data
    assert p[-1] == pytest.approx(cal.predict(np.array([s.max()]))[0, 1])     # clipped above the data
    assert cal.predict(np.array([-100.0, 100.0]))[:, 1].tolist() == [p[0], p[-1]]
    assert abs(cal.predict(np.array([0.0]))[0, 1] - 0.5) < 0.1


def test_temperature_recovers_T_on_softmax_scores():
    rng = np.random.default_rng(3)
    S = rng.normal(size=(30000, 4)) * 3
    P = _softmax(S / 2.5)
    y = (rng.random(30000)[:, None] > np.cumsum(P, 1)).sum(1)
    assert C.fit_temperature(S, y, binary=False) == pytest.approx(2.5, rel=0.05)
    s = rng.normal(size=30000) * 3
    yb = (rng.random(30000) < _sigmoid(s / 0.6)).astype(int)
    assert C.fit_temperature(s, yb, binary=True) == pytest.approx(0.6, rel=0.05)


@pytest.mark.parametrize("method", C.METHODS)
def test_multiclass_rows_sum_to_one(method):
    rng = np.random.default_rng(4)
    S = rng.normal(size=(5000, 3)) * 2
    y = (rng.random(5000)[:, None] > np.cumsum(_softmax(S), 1)).sum(1)
    cal = C.Calibrator(method, 3).fit(S, y)
    P = cal.predict(rng.normal(size=(200, 3)) * 4)
    assert P.shape == (200, 3) and np.allclose(P.sum(1), 1) and (P >= 0).all()


@pytest.mark.parametrize("method", C.METHODS)
def test_binary_predictions_are_false_true_columns(method):
    rng = np.random.default_rng(5)
    s = rng.normal(size=3000)
    y = (rng.random(3000) < _sigmoid(s)).astype(int)
    P = C.Calibrator(method, 2).fit(s, y).predict(np.array([-3.0, 0.0, 3.0]))
    assert P.shape == (3, 2) and np.allclose(P.sum(1), 1) and P[0, 1] < P[1, 1] < P[2, 1]


def test_sigmoid_calibration_fixes_an_overconfident_model():
    rng = np.random.default_rng(6)
    z = rng.normal(size=20000)
    y = (rng.random(20000) < _sigmoid(z)).astype(int)
    raw = 3 * z                                       # a model whose logits are 3x too large
    p = C.Calibrator("sigmoid", 2).fit(raw[:10000], y[:10000]).predict(raw[10000:])[:, 1]
    from open_rlcd.core.metrics import ece

    assert ece(_sigmoid(raw[10000:]), y[10000:]) > 0.08 and ece(p, y[10000:]) < 0.02


@pytest.mark.parametrize("method,k", [(m, k) for m in C.METHODS for k in (2, 3)])
def test_to_dict_round_trip_gives_identical_predictions(method, k):
    rng = np.random.default_rng(7)
    S = rng.normal(size=(2000, k)) if k > 2 else rng.normal(size=2000)
    y = rng.integers(0, k, 2000)
    cal = C.Calibrator(method, k).fit(S, y)
    back = C.Calibrator.from_dict(json.loads(json.dumps(cal.to_dict())))
    X = rng.normal(size=(300, k)) * 3 if k > 2 else rng.normal(size=300) * 3
    assert np.array_equal(cal.predict(X), back.predict(X))
    assert back.method == method and back.n_classes == k


def test_unknown_method_is_refused():
    with pytest.raises(ValueError, match="calibration method"):
        C.Calibrator("beta", 2)


def test_choose_method():
    small = [np.array([0, 1] * 100), np.array([0, 1] * 150)]
    m, why = C.choose_method("auto", small, 2)
    assert m == "sigmoid" and "100" in why and "3000" in why
    big = [np.array([0] * 3000 + [1] * 3100), np.array([0] * 3200 + [1] * 3000)]
    assert C.choose_method("auto", big, 2)[0] == "isotonic"
    one_short = [*big, np.array([0] * 5000 + [1] * 2999)]
    assert C.choose_method("auto", one_short, 2)[0] == "sigmoid"            # every block must qualify
    missing = [np.array([0] * 4000 + [1] * 4000)]
    assert C.choose_method("auto", missing, 3)[0] == "sigmoid"               # a class absent from a block
    assert C.choose_method("auto", small, 2, isotonic_min_per_class=50)[0] == "isotonic"
    for m in C.METHODS:
        assert C.choose_method(m, small, 2) == (m, "requested")


# ----------------------------------------------------------------------------- conformal: selective threshold

def _calibrated_gate(n, rng):
    """P(true) calibrated by construction, with the mass near 0 and 1 that a model with real signal produces
    (with uniform scores few rows have an error rate well under the target, and the search has little power)."""
    p = _sigmoid(3 * rng.normal(size=n))
    return p, rng.random(n) < p


def test_min_rows_for():
    assert C.min_rows_for(0.05, 0.1) == 45      # 0.95^45 < 0.1 <= 0.95^44
    assert 0.95 ** 45 < 0.1 <= 0.95 ** 44
    assert C.min_rows_for(0.1, 0.1) == 22


@pytest.mark.parametrize("target", [0.1, 0.05])
def test_selective_threshold_keeps_its_promise_on_fresh_rows(target):
    """With probability >= 1 - delta the error rate among rows passing the threshold is <= the target. P(true)
    is calibrated, so the true selective error above t is the mean of 1 - p over fresh rows with p >= t."""
    delta = 0.1
    fresh_p, fresh_ok = _calibrated_gate(400_000, np.random.default_rng(12345))
    violations, missing, thresholds = 0, 0, []
    for seed in range(30):
        p, correct = _calibrated_gate(1500, np.random.default_rng(seed))
        r = C.selective_threshold(p, correct, target, delta)
        t = r["threshold"]
        if t is None:                                             # allowed (low power), never a violation
            missing += 1
            assert r["reason"]
            continue
        assert r["coverage"] == pytest.approx(r["n_selected"] / 1500, abs=1e-6)
        assert r["errors"] / r["n_selected"] <= target            # on its own calibration rows
        thresholds.append(t)
        sel = fresh_p >= t
        violations += (1 - fresh_p[sel]).mean() > target          # the true selective error
        assert (~fresh_ok[sel]).mean() <= target + 0.01           # observed on fresh rows
    assert missing <= 3
    assert violations / 30 <= delta + 0.1                         # about delta at most (slack for 30 seeds)
    srt = np.sort(fresh_p)[::-1]
    selective_error = np.cumsum(1 - srt) / np.arange(1, len(srt) + 1)
    ideal = srt[np.searchsorted(selective_error, target) - 1]    # the lowest threshold that truly meets the target
    assert ideal < np.median(thresholds) < ideal + 0.15           # conservative, but not trivially so


def test_selective_threshold_with_too_few_rows_says_why():
    rng = np.random.default_rng(1)
    p, correct = _calibrated_gate(30, rng)
    r = C.selective_threshold(p, correct, 0.05, 0.1)
    assert r["threshold"] is None and r["min_rows"] == 45 and r["n_calibration"] == 30
    assert "30 calibration rows" in r["reason"] and "at least 45" in r["reason"]


def test_selective_threshold_refuses_pure_noise():
    rng = np.random.default_rng(2)
    p = rng.random(2000)
    correct = rng.random(2000) < 0.5                # P(true) carries no information
    r = C.selective_threshold(p, correct, 0.05, 0.1)
    assert r["threshold"] is None and "no threshold" in r["reason"]


def test_selective_check_on_the_holdout():
    rng = np.random.default_rng(3)
    p, correct = _calibrated_gate(3000, rng)
    ok = C.selective_check(p, correct, 0.85, 0.1)
    assert ok["consistent"] is True and ok["error_rate"] < 0.1 and ok["n"] == 3000
    assert ok["coverage"] == pytest.approx((p >= 0.85).mean(), abs=1e-6)
    bad = C.selective_check(p, np.zeros(3000, bool), 0.85, 0.1)      # every selected row wrong
    assert bad["consistent"] is False and bad["error_rate"] == 1.0
    none = C.selective_check(p, correct, None, 0.1)
    assert none["n_selected"] == 0 and none["error_rate"] is None
    empty = C.selective_check(p, correct, 2.0, 0.1)                  # nothing passes
    assert empty["n_selected"] == 0 and empty["consistent"] is None


# ----------------------------------------------------------------------------- conformal: prediction sets

def _calibrated_choice(n, k, rng):
    P = rng.dirichlet(np.ones(k) * 0.7, size=n)
    y = (rng.random(n)[:, None] > np.cumsum(P, 1)).sum(1)
    return P, np.minimum(y, k - 1)


@pytest.mark.parametrize("alpha", [0.1, 0.2])
def test_set_quantile_covers_one_minus_alpha_on_fresh_rows(alpha):
    covers, sizes = [], []
    for seed in range(10):
        rng = np.random.default_rng(seed)
        P, y = _calibrated_choice(1000, 4, rng)
        q = C.set_quantile(P[np.arange(1000), y], alpha)
        assert q["n_calibration"] == 1000 and q["alpha"] == alpha and 0 <= q["q"] <= 1
        Pf, yf = _calibrated_choice(5000, 4, rng)
        sets = [C.prediction_set(row, q["q"]) for row in Pf]
        covers.append(np.mean([int(t) in s for t, s in zip(yf, sets, strict=True)]))
        sizes.append(np.mean([len(s) for s in sets]))
    assert np.mean(covers) == pytest.approx(1 - alpha, abs=0.02)
    assert min(covers) >= 1 - alpha - 0.04 and np.mean(sizes) < 4


def test_set_quantile_edge_cases():
    assert C.set_quantile(np.array([]), 0.1)["q"] is None
    assert C.set_quantile(np.array([0.9, 0.8, 0.7]), 0.1)["q"] == 1.0     # too few rows: the full set
    assert C.prediction_set(np.array([0.2, 0.5, 0.3]), None) is None
    assert C.prediction_set(np.array([0.2, 0.5, 0.3]), 1.0) == [0, 1, 2]
    assert C.prediction_set(np.array([0.2, 0.5, 0.3]), 0.5) == [1]
    assert C.prediction_set(np.array([0.2, 0.5, 0.3]), 0.7) == [1, 2]
