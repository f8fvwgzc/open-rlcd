"""Holdout metrics: Wilson, reliability, ECE, the block-bootstrap skill test, judge agreement."""
import math

import numpy as np
import pytest
from sklearn.metrics import cohen_kappa_score

from open_rlcd.core import metrics as M

# ----------------------------------------------------------------------------- Wilson

def test_wilson_matches_the_closed_form():
    z = 1.959964
    for k, n in [(5, 10), (0, 10), (10, 10), (37, 120), (1, 3)]:
        p = k / n
        centre = (p + z * z / (2 * n)) / (1 + z * z / n)
        half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
        lo, hi = M.wilson(k, n)
        assert lo == pytest.approx(max(0.0, centre - half)) and hi == pytest.approx(min(1.0, centre + half))
        assert 0 <= lo <= p <= hi <= 1
    assert M.wilson(5, 10) == (pytest.approx(0.2366, abs=1e-4), pytest.approx(0.7634, abs=1e-4))
    assert M.wilson(0, 0) == (None, None)
    assert M.wilson(0, 10)[0] == 0.0 and M.wilson(10, 10)[1] == 1.0


def test_wilson_interval_narrows_with_more_trials():
    widths = [np.subtract(*M.wilson(n // 2, n)[::-1]) for n in (10, 100, 1000)]
    assert widths[0] > widths[1] > widths[2]


def test_wilson_covers_the_true_rate_about_95_percent():
    rng = np.random.default_rng(3)
    p, n = 0.3, 80
    ks = rng.binomial(n, p, size=4000)
    cover = np.mean([lo <= p <= hi for lo, hi in (M.wilson(int(k), n) for k in ks)])
    assert 0.93 <= cover <= 0.97


# ----------------------------------------------------------------------------- reliability and ECE

def test_reliability_table_bins():
    p = np.array([0.0, 0.05, 0.15, 0.55, 0.95, 1.0])
    y = np.array([0, 0, 1, 1, 1, 1])
    t = M.reliability_table(p, y)
    assert len(t) == 10 and [b["bin"] for b in t] == list(range(10)) and sum(b["count"] for b in t) == 6
    assert t[0]["count"] == 2 and t[0]["observed_rate"] == 0.0 and t[0]["mean_predicted"] == pytest.approx(0.025)
    assert t[9]["count"] == 2 and t[9]["mean_predicted"] == pytest.approx(0.975)   # p = 1.0 lands in the last bin
    assert t[3]["count"] == 0 and t[3]["mean_predicted"] is None and t[3]["observed_rate"] is None
    for b in t:
        assert b["lo"] == pytest.approx(b["bin"] / 10) and b["hi"] == pytest.approx((b["bin"] + 1) / 10)
        if b["count"]:
            assert b["lo"] <= b["mean_predicted"] <= b["hi"]


def test_ece_worked_examples():
    # one bin, predicted 0.7, observed 0.5 -> 0.2
    assert M.ece(np.full(10, 0.7), np.array([1] * 5 + [0] * 5)) == pytest.approx(0.2)
    # two bins of equal weight: |0.15 - 0| and |0.85 - 1| -> 0.15
    p = np.array([0.15] * 4 + [0.85] * 4)
    y = np.array([0] * 4 + [1] * 4)
    assert M.ece(p, y) == pytest.approx(0.15)
    assert math.isnan(M.ece(np.array([]), np.array([])))


def test_ece_is_small_for_calibrated_and_large_for_overconfident_forecasts():
    rng = np.random.default_rng(0)
    p = rng.random(20000)
    y = (rng.random(20000) < p).astype(int)
    assert M.ece(p, y) < 0.02
    sharp = np.clip(0.5 + 1.8 * (p - 0.5), 0, 1)    # same ranking, overconfident
    assert M.ece(sharp, y) > 0.1


# ----------------------------------------------------------------------------- reports

def test_binary_report_against_the_base_rate():
    rng = np.random.default_rng(1)
    p = rng.random(5000)
    y = (rng.random(5000) < p).astype(int)
    r = M.binary_report(p, y, y.mean())
    assert r["n"] == 5000 and r["brier"] < r["baseline"]["brier"] and r["brier_skill_score"] > 0.2
    assert r["brier"] == pytest.approx(float(((p - y) ** 2).mean()), abs=1e-6)
    assert r["baseline"]["brier"] == pytest.approx(float(((y.mean() - y) ** 2).mean()), abs=1e-6)
    assert len(r["reliability"]) == 10


def test_calibration_report_shapes_and_definitions():
    rng = np.random.default_rng(2)
    n, k = 600, 3
    logits = rng.normal(size=(n, k)) * 2
    P = np.exp(logits) / np.exp(logits).sum(1, keepdims=True)
    y = np.array([rng.choice(k, p=row) for row in P])
    prior = np.bincount(y, minlength=k) / n
    rep = M.calibration_report(P, y, ["a", "b", "c"], prior, blocks=20, seed=1)
    Y = np.eye(k)[y]
    assert rep["n_holdout"] == n and rep["classes"] == ["a", "b", "c"]
    assert rep["brier"] == pytest.approx(float(((P - Y) ** 2).sum(1).mean()), abs=1e-6)
    assert rep["log_loss"] == pytest.approx(float(-np.log(P[np.arange(n), y]).mean()), abs=1e-6)
    assert rep["accuracy"] == pytest.approx(float((P.argmax(1) == y).mean()), abs=1e-6)
    assert rep["brier_skill_score"] == pytest.approx(1 - rep["brier"] / rep["baseline"]["brier"], abs=1e-5)
    assert rep["skill"]["has_skill"] is True and rep["skill"]["lower_95"] > 0
    assert len(rep["reliability"]["top_label"]) == 10 and set(rep["reliability"]["per_class"]) == {"a", "b", "c"}
    assert set(rep["per_class"]["a"]) >= {"brier", "ece", "baseline", "brier_skill_score"}
    assert "reliability" not in rep["per_class"]["a"]
    assert rep["base_rates_train"]["a"] == pytest.approx(prior[0], abs=1e-6)
    s = M.summary(rep)
    assert set(s) == {"n_holdout", "brier", "log_loss", "accuracy", "ece", "brier_skill_score", "baseline_brier",
                      "baseline_log_loss", "baseline_accuracy", "has_skill", "brier_gain_lower_95"}
    assert s["has_skill"] is True and s["baseline_brier"] == rep["baseline"]["brier"]


def test_calibration_report_accepts_one_baseline_row_per_holdout_row():
    P = np.array([[0.8, 0.2], [0.3, 0.7], [0.6, 0.4], [0.1, 0.9]])
    y = np.array([0, 1, 0, 1])
    prior = np.array([[0.5, 0.5], [0.4, 0.6], [0.5, 0.5], [0.4, 0.6]])
    rep = M.calibration_report(P, y, ["false", "true"], prior, blocks=2)
    assert rep["baseline"]["brier"] == pytest.approx(float(((prior - np.eye(2)[y]) ** 2).sum(1).mean()), abs=1e-6)
    assert rep["skill"]["has_skill"] is False and "too few" in rep["skill"]["method"]


# ----------------------------------------------------------------------------- the skill test

def test_skill_test_finds_a_real_gain():
    rng = np.random.default_rng(0)
    gain = rng.normal(0.05, 0.2, size=1000)
    r = M.skill_test(gain, 20, seed=1)
    assert r["has_skill"] is True and r["lower_95"] > 0 and r["lower_95"] < r["mean_brier_gain"] < r["upper_95"]
    assert r["blocks"] == 20 and r["n"] == 1000 and "block bootstrap" in r["method"]


def test_zero_mean_noise_shows_no_skill_in_the_vast_majority_of_seeds():
    claims = 0
    for seed in range(100):
        gain = np.random.default_rng(seed).normal(0.0, 0.2, size=600)
        claims += M.skill_test(gain, 20, n_boot=500, seed=seed)["has_skill"]
    assert claims <= 12          # a one-sided 95 % bound: about 5 % false claims, with slack for 20 blocks


def test_a_negative_gain_never_claims_skill():
    gain = np.random.default_rng(4).normal(-0.02, 0.1, size=800)
    assert M.skill_test(gain, 20)["has_skill"] is False


@pytest.mark.parametrize("n", [0, 5, 9, 20])
def test_too_few_rows_cannot_claim_skill(n):
    r = M.skill_test(np.full(n, 0.5), 20)
    assert r["has_skill"] is False and r["lower_95"] is None and "too few" in r["method"]
    assert r["n"] == n


def test_too_few_groups_cannot_claim_skill():
    gain = np.full(400, 0.3)
    r = M.skill_test(gain, np.repeat(["a", "b", "c", "d"], 100))
    assert r["has_skill"] is False and r["blocks"] == 4 and "too few" in r["method"]


def test_group_blocks_resample_whole_groups():
    """Rows of one group share their gain (one document, one customer). Cutting shuffled rows into contiguous
    blocks treats them as independent and claims skill whenever the mean is positive; resampling whole groups
    sees there are only 8 independent units and rarely claims anything."""
    contiguous = grouped = 0
    for seed in range(40):
        rng = np.random.default_rng(seed)
        groups = np.repeat(np.arange(8), 100)
        gain = rng.normal(0, 1, 8)[groups] + rng.normal(0, 0.1, 800)
        order = rng.permutation(800)
        gain, groups = gain[order], groups[order]
        contiguous += M.skill_test(gain, 20, n_boot=400, seed=seed)["has_skill"]
        r = M.skill_test(gain, groups.astype(str), n_boot=400, seed=seed)
        assert r["blocks"] == 8 and "8 groups" in r["method"]
        grouped += r["has_skill"]
    assert contiguous >= 12 and grouped <= 8 and grouped < contiguous


# ----------------------------------------------------------------------------- agreement

def test_cohen_kappa_matches_scikit_learn():
    rng = np.random.default_rng(5)
    a = rng.integers(0, 3, 300)
    b = np.where(rng.random(300) < 0.7, a, rng.integers(0, 3, 300))
    assert M.cohen_kappa(a, b, 3) == pytest.approx(cohen_kappa_score(a, b), abs=1e-9)
    assert M.cohen_kappa(a, a, 3) == pytest.approx(1.0)
    assert M.cohen_kappa(np.array([], int), np.array([], int), 2) is None
    assert M.cohen_kappa(np.zeros(5, int), np.zeros(5, int), 2) is None     # chance agreement is total
    # textbook example: 20 yes-yes, 5 yes-no, 10 no-yes, 15 no-no -> po = 0.7, pe = 0.5, kappa = 0.4
    a = np.array([1] * 25 + [0] * 25)
    b = np.array([1] * 20 + [0] * 5 + [1] * 10 + [0] * 15)
    assert M.cohen_kappa(a, b, 2) == pytest.approx(0.4)


def test_agreement_report():
    human = np.array([0, 0, 1, 1, 1, 2, 2, -1])
    judge = np.array([0, 1, 1, 1, 0, 2, -1, 2])
    r = M.agreement(judge, human, ["a", "b", "c"])
    assert r["n"] == 6 and r["accuracy"] == pytest.approx(4 / 6, abs=1e-6)       # rows with -1 are left out
    assert r["confusion"]["rows"] == "human label" and r["confusion"]["columns"] == "judge label"
    assert r["confusion"]["counts"] == [[1, 1, 0], [1, 2, 0], [0, 0, 1]]
    lo, hi = r["accuracy_wilson_95"]
    assert lo < r["accuracy"] < hi
    assert r["cohen_kappa"] == pytest.approx(cohen_kappa_score(human[:6], judge[:6]), abs=1e-6)
    assert M.agreement(np.array([-1]), np.array([0]), ["a", "b"])["n"] == 0
