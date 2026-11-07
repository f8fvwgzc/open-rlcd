"""Every decision as a Bernoulli trial: posterior arithmetic, exact tests, sample size, the durable ledger
(ported from the reference test_bernoulli, without the trading-only Kelly sizing)."""
import json
import math

import numpy as np
import pytest
from scipy import stats

from open_rlcd.core import bernoulli as B


def test_beta_update_arithmetic():
    cfg = B.BernoulliConfig(prior_alpha=1, prior_beta=1)
    p = B.posterior(7, 3, 0.3, cfg)
    assert (p["alpha"], p["beta"], p["n"]) == (8.0, 4.0, 10)
    assert p["mean"] == pytest.approx(8 / 12) and p["required"] == 0.3
    assert p["p_above_required"] == pytest.approx(stats.beta.sf(0.3, 8, 4), abs=1e-6)
    lo, hi = p["ci95"]
    assert lo < p["mean"] < hi
    assert stats.beta.cdf(hi, 8, 4) - stats.beta.cdf(lo, 8, 4) == pytest.approx(0.95, abs=1e-4)
    empty = B.posterior(0, 0, 0.3, cfg)                 # uniform prior: 70 % of the mass above 0.3
    assert empty["p_above_required"] == pytest.approx(0.7) and empty["n"] == 0 and empty["mean"] == 0.5
    jeffreys = B.posterior(2, 1, 0.5, B.BernoulliConfig(prior_alpha=0.5, prior_beta=0.5))
    assert (jeffreys["alpha"], jeffreys["beta"]) == (2.5, 1.5)


def test_p_above_required_is_monotonic_in_wins():
    ps = [B.posterior(w, 20 - w, 0.3)["p_above_required"] for w in range(21)]
    assert all(b >= a for a, b in zip(ps, ps[1:], strict=False)) and ps[0] < 0.01 and ps[-1] > 0.99
    more = [B.posterior(4 * k, 6 * k, 0.3)["p_above_required"] for k in (1, 5, 25)]   # 40 % right, more trials
    assert more[0] < more[1] < more[2]


def test_binomial_p_values():
    assert B.binomial_test(3, 3, 0.5) == pytest.approx(0.125)          # by hand: 0.5^3
    assert B.binomial_test(2, 3, 0.5) == pytest.approx(0.5)            # P(X >= 2) = 3/8 + 1/8
    assert B.binomial_test(45, 100, 0.3) == pytest.approx(
        stats.binomtest(45, 100, 0.3, alternative="greater").pvalue)
    assert B.binomial_test(0, 10, 0.3) == pytest.approx(1.0)
    assert B.binomial_test(0, 0, 0.3) is None
    assert B.binomial_test_below(0, 3, 0.5) == pytest.approx(0.125)    # P(X <= 0)
    assert B.binomial_test_below(70, 100, 0.8) == pytest.approx(
        stats.binomtest(70, 100, 0.8, alternative="less").pvalue)
    assert B.binomial_test_below(10, 10, 0.9) == pytest.approx(1.0) and B.binomial_test_below(0, 0, 0.5) is None


def test_decisions_needed_formula():
    za, zb = stats.norm.ppf(0.95), stats.norm.ppf(0.8)
    want = ((za * math.sqrt(0.3 * 0.7) + zb * math.sqrt(0.4 * 0.6)) / 0.1) ** 2
    assert B.decisions_needed(0.4, 0.3) == math.ceil(want) == 136
    assert B.decisions_needed(0.32, 0.3) > B.decisions_needed(0.4, 0.3) > B.decisions_needed(0.6, 0.3)
    assert B.decisions_needed(0.3, 0.3) is None and B.decisions_needed(0.2, 0.3) is None
    assert B.decisions_needed(1.0, 0.3) is None and B.decisions_needed(None, 0.3) is None
    stricter = B.BernoulliConfig(alpha=0.01, power=0.9)
    assert B.decisions_needed(0.4, 0.3, stricter) > B.decisions_needed(0.4, 0.3)


@pytest.mark.parametrize("p,k", [(0.0, 0), (0.05, 0), (0.1, 1), (0.55, 5), (0.999, 9), (1.0, 9), (-0.2, 0),
                                 (1.3, 9)])
def test_bin_of(p, k):
    assert B.bin_of(p) == k


def test_thompson_draws_follow_the_posterior():
    rng = np.random.default_rng(0)
    draws = np.array([B.thompson(30, 10, rng) for _ in range(6000)])
    a, b = 31, 11
    assert draws.mean() == pytest.approx(a / (a + b), abs=0.005)
    assert draws.var() == pytest.approx(a * b / ((a + b) ** 2 * (a + b + 1)), rel=0.08)
    assert stats.kstest(draws, stats.beta(a, b).cdf).pvalue > 0.01
    cfg = B.BernoulliConfig(prior_alpha=2, prior_beta=5)
    assert np.mean([B.thompson(0, 0, rng, cfg) for _ in range(4000)]) == pytest.approx(2 / 7, abs=0.01)


# ----------------------------------------------------------------------------- the ledger

def test_ledger_counts_each_decision_id_once(tmp_path):
    led = B.Ledger(tmp_path / "bernoulli.json")
    assert led.table("h") is None and led.update("h", "a", "true", 0.9, "true") is None   # no table yet
    led.reset("h", B.Ledger.empty(["false", "true"], "rlcd-0.1.1"))
    first = led.update("h", "a", "true", 0.87, "true")
    assert first == {"updated": True, "class": "true", "bin": 8, "wins": 1, "losses": 0, "correct": True}
    again = led.update("h", "a", "true", 0.87, "true")                       # the same outcome again: no change
    assert again == {"updated": False, "reason": "this decision_id was already scored"}
    # a corrected outcome replaces the first one (training keeps the latest row per decision_id too)
    fixed = led.update("h", "a", "true", 0.87, "false")
    assert fixed == {"updated": True, "corrected": True, "class": "true", "bin": 8, "wins": 0, "losses": 1,
                     "correct": False}
    back = led.update("h", "a", "true", 0.87, "true")
    assert back["corrected"] and back["wins"] == 1 and back["losses"] == 0
    wrong = led.update("h", "b", "true", 0.83, "false")
    assert wrong["updated"] and wrong["wins"] == 1 and wrong["losses"] == 1 and wrong["correct"] is False
    other = led.update("h", "c", "false", 0.62, "false")
    assert other["bin"] == 6 and other["wins"] == 1
    t = led.table("h")
    assert t["bins"]["true"][8] == [1, 1] and t["bins"]["false"][6] == [1, 0]
    assert t["online_trials"] == 3 and t["applied"] == ["a", "b", "c"] and t["model_version"] == "rlcd-0.1.1"
    assert led.update("h", "d", "maybe", 0.9, "maybe") is None              # not a class of this head
    # durable: a new Ledger on the same file sees the same counts and still refuses "a"
    again = B.Ledger(tmp_path / "bernoulli.json")
    assert again.table("h") == t and again.update("h", "a", "true", 0.87, "true")["updated"] is False
    assert json.loads((tmp_path / "bernoulli.json").read_text())["h"]["online_trials"] == 3


def test_ledger_tables_are_per_head_and_restart_at_reset(tmp_path):
    led = B.Ledger(tmp_path / "bernoulli.json")
    led.reset("x", B.Ledger.empty(["false", "true"], "v1"))
    led.reset("y", B.Ledger.empty(["a", "b"], "v1"))
    led.update("x", "d1", "true", 0.9, "true")
    led.update("y", "d1", "a", 0.7, "b")                         # the same decision_id may score another head
    assert led.table("x")["online_trials"] == led.table("y")["online_trials"] == 1
    led.reset("x", B.Ledger.empty(["false", "true"], "v2"))
    assert led.table("x")["online_trials"] == 0 and led.table("x")["model_version"] == "v2"
    assert led.table("y")["online_trials"] == 1


def test_a_corrupt_ledger_file_reads_as_empty(tmp_path):
    path = tmp_path / "bernoulli.json"
    path.write_text("{torn")
    led = B.Ledger(path)
    assert led.table("h") is None
    led.reset("h", B.Ledger.empty(["false", "true"]))
    assert led.table("h")["classes"] == ["false", "true"]


def test_holdout_table_counts_every_row_in_its_bin():
    P = np.array([[0.1, 0.9], [0.2, 0.8], [0.85, 0.15], [0.45, 0.55], [0.05, 0.95]])
    y = np.array([1, 0, 0, 1, 1])
    chosen = P.argmax(1)
    t = B.holdout_table(P, y, ["false", "true"], chosen, "v3")
    assert t["holdout_trials"] == 5 and t["online_trials"] == 0 and t["model_version"] == "v3"
    assert t["bins"]["true"][9] == [2, 0] and t["bins"]["true"][8] == [0, 1] and t["bins"]["true"][5] == [1, 0]
    assert t["bins"]["false"][8] == [1, 0]
    assert sum(w + l_ for c in t["bins"] for w, l_ in t["bins"][c]) == 5
    gate = B.holdout_table(P, y, ["false", "true"], np.ones(5, int))   # a gate head always commits to true
    assert gate["bins"]["true"][1] == [0, 1] and sum(w + l_ for w, l_ in gate["bins"]["false"]) == 0


def test_table_view():
    t = B.Ledger.empty(["a", "b", "c"], "v1")
    t["bins"]["b"][9] = [40, 2]
    v = B.table_view(t, 0.75)
    assert len(v["bins"]) == 30 and v["model_version"] == "v1"
    row = next(r for r in v["bins"] if r["class"] == "b" and r["bin"] == 9)
    assert row["wins"] == 40 and row["losses"] == 2 and row["lo"] == 0.9 and row["hi"] == 1.0
    assert row["p_above_required"] == pytest.approx(stats.beta.sf(0.75, 41, 3), abs=1e-6)
    empty = next(r for r in v["bins"] if r["class"] == "a" and r["bin"] == 0)
    assert empty["n"] == 0 and empty["p_above_required"] == pytest.approx(0.25)
