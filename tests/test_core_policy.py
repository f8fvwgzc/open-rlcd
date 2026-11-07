"""The decision policy: thresholds -> tiers, the evidence caps (verify, Bernoulli gate), exploration, weakest link."""
import numpy as np
import pytest

from open_rlcd.core import bernoulli as B
from open_rlcd.core import policy as P
from open_rlcd.core.head import HeadSpec

GATE = HeadSpec("answerable", "noul", decision="gate")
GATE_CONFORMAL = HeadSpec("answerable", "noul", decision="gate", conformal=0.05)
SYM = HeadSpec("needs_graph", "noul")
COLOR = HeadSpec("color", "choice", ("red", "green", "blue"))
CFG = P.PolicyConfig()


def strong_ledger(classes, wins=400, losses=4):
    t = B.Ledger.empty(classes, "v1")
    for c in classes:
        t["bins"][c] = [[wins, losses] for _ in range(10)]
    return t


# ----------------------------------------------------------------------------- tiers

def test_tiers_follow_the_stakes_thresholds():
    p = np.array([0.40, 0.50, 0.69, 0.70, 0.84, 0.85, 0.94, 0.95, 1.0])
    assert P.tiers(p, CFG.level("low")).tolist() == [0, 1, 1, 2, 2, 2, 2, 2, 2]
    assert P.tiers(p, CFG.level("medium")).tolist() == [0, 0, 1, 1, 1, 2, 2, 2, 2]
    assert P.tiers(p, CFG.level("high")).tolist() == [0, 0, 0, 0, 1, 1, 1, 2, 2]
    # a conformal threshold only ever restricts act
    assert P.tiers(p, CFG.level("low"), 0.9).tolist() == [0, 1, 1, 1, 1, 1, 2, 2, 2]


def test_stakes_config_validation_and_round_trip():
    with pytest.raises(ValueError, match="confirm <= act"):
        P.StakesLevel(act=0.6, confirm=0.7, min_accuracy=0.5)
    with pytest.raises(ValueError, match="min_accuracy"):
        P.StakesLevel(act=0.9, confirm=0.7, min_accuracy=1.5)
    cfg = P.PolicyConfig.from_dict({"abstain": "hold", "stakes": {"medium": {"act": 0.9}, "vip": {
        "act": 0.99, "confirm": 0.9, "min_accuracy": 0.95}}, "min_verify_rows": 50, "unknown_key": 1})
    assert cfg.abstain == "hold" and cfg.min_verify_rows == 50
    assert cfg.level("medium") == P.StakesLevel(0.9, 0.65, 0.75) and cfg.level("vip").act == 0.99
    assert P.PolicyConfig.from_dict(cfg.to_dict()) == cfg
    with pytest.raises(ValueError, match="unknown stakes level"):
        cfg.level("extreme")
    assert [cfg.tier_name(i) for i in range(3)] == ["hold", "confirm", "act"]


# ----------------------------------------------------------------------------- gate vs argmax

def test_gate_commits_to_true_while_argmax_commits_to_the_likeliest_class():
    probs = np.array([[0.9, 0.1], [0.2, 0.8]])
    c, p = P.chosen(GATE, probs)
    assert c.tolist() == [1, 1] and p.tolist() == [0.1, 0.8]
    c, p = P.chosen(SYM, probs)
    assert c.tolist() == [0, 1] and p.tolist() == [0.9, 0.8]
    c, p = P.chosen(COLOR, np.array([[0.2, 0.5, 0.3]]))
    assert c.tolist() == [1] and p.tolist() == [0.5]


def test_a_gate_head_never_acts_on_false():
    led = strong_ledger(["false", "true"])
    g = P.decide(GATE, [0.97, 0.03], CFG, ledger=led)
    assert g["class"] == "true" and g["p"] == 0.03 and g["tier"] == "escalate" and g["commit"] is None
    s = P.decide(SYM, [0.97, 0.03], CFG, ledger=led)
    assert s["class"] == "false" and s["tier"] == "act" and s["commit"] == "false"
    ok = P.decide(GATE, [0.03, 0.97], CFG, ledger=led)
    assert ok["tier"] == "act" and ok["commit"] == "true"


def test_decision_shape():
    d = P.decide(COLOR, [0.1, 0.8, 0.1], CFG, ledger=strong_ledger(["red", "green", "blue"]))
    assert set(d) == {"head", "decision", "stakes", "class", "commit", "p", "tier", "uncapped_tier", "max_tier",
                      "thresholds", "bernoulli", "gate", "calibrated", "warnings"}
    assert d["thresholds"] == {"act": 0.85, "confirm": 0.65, "min_accuracy": 0.75, "conformal": None}
    assert d["stakes"] == "medium" and d["class"] == "green" and d["tier"] == "confirm"   # 0.8 < 0.85
    assert d["bernoulli"]["bin"] == 8 and d["bernoulli"]["wins"] == 400
    assert P.decide(COLOR, [0.1, 0.8, 0.1], CFG, stakes="low")["thresholds"]["act"] == 0.70


def test_conformal_threshold_restricts_act_for_gate_heads_only():
    led = strong_ledger(["false", "true"])
    d = P.decide(GATE_CONFORMAL, [0.1, 0.9], CFG, conformal_threshold=0.95, ledger=led)
    assert d["uncapped_tier"] == "confirm" and d["thresholds"]["conformal"] == 0.95
    d = P.decide(GATE_CONFORMAL, [0.03, 0.97], CFG, conformal_threshold=0.95, ledger=led)
    assert d["tier"] == "act"
    d = P.decide(SYM, [0.1, 0.9], CFG, conformal_threshold=0.95, ledger=led)
    assert d["tier"] == "act" and d["thresholds"]["conformal"] is None


# ----------------------------------------------------------------------------- caps

def test_untrained_heads_always_escalate():
    d = P.decide(COLOR, [1.0, 0.0, 0.0], CFG, ledger=strong_ledger(["red", "green", "blue"]), calibrated=False)
    assert d["tier"] == "escalate" and d["uncapped_tier"] == "act" and d["max_tier"] == "escalate"
    assert d["calibrated"] is False and any("not trained" in w for w in d["warnings"])


def test_the_verdict_caps_the_tier_and_says_why():
    led = strong_ledger(["red", "green", "blue"])
    verdict = {"medium": {"max_tier": "confirm", "tier": 1, "reason": "too few act-tier rows"},
               "low": {"max_tier": "act", "tier": 2, "reason": "fine"}}
    d = P.decide(COLOR, [0.95, 0.03, 0.02], CFG, verdict=verdict, ledger=led)
    assert d["tier"] == "confirm" and d["uncapped_tier"] == "act" and d["max_tier"] == "confirm"
    assert any("too few act-tier rows" in w for w in d["warnings"])
    d = P.decide(COLOR, [0.95, 0.03, 0.02], CFG, verdict=verdict, ledger=led, stakes="low")
    assert d["tier"] == "act" and d["warnings"] == []
    none = {"medium": {"max_tier": "escalate", "tier": 0, "reason": "no skill over the base rate"}}
    d = P.decide(COLOR, [0.95, 0.03, 0.02], CFG, verdict=none, ledger=led)
    assert d["tier"] == "escalate" and any("no skill" in w for w in d["warnings"])


def test_bernoulli_gate_turns_act_into_confirm_without_evidence():
    """A uniform prior puts only 25 % of its mass above the 0.75 accuracy medium stakes require."""
    d = P.decide(COLOR, [0.95, 0.03, 0.02], CFG, ledger=None)
    assert d["uncapped_tier"] == "act" and d["tier"] == "confirm"
    assert d["gate"]["passed"] is False and d["gate"]["mode"] == "posterior"
    assert d["bernoulli"]["n"] == 0 and d["bernoulli"]["p_above_required"] == pytest.approx(0.25)
    assert any("Bernoulli gate" in w and "act becomes confirm" in w for w in d["warnings"])
    weak = strong_ledger(["red", "green", "blue"], wins=6, losses=4)
    assert P.decide(COLOR, [0.95, 0.03, 0.02], CFG, ledger=weak)["tier"] == "confirm"
    strong = P.decide(COLOR, [0.95, 0.03, 0.02], CFG, ledger=strong_ledger(["red", "green", "blue"]))
    assert strong["tier"] == "act" and strong["gate"]["passed"] is True and strong["warnings"] == []


def test_exploration_is_off_by_default_and_labelled_when_on():
    led = strong_ledger(["red", "green", "blue"], wins=3, losses=1)
    d = P.decide(COLOR, [0.95, 0.03, 0.02], CFG, ledger=led)
    assert "exploration" not in d["gate"] and d["gate"]["mode"] == "posterior"
    rng = np.random.default_rng(0)
    draws = [P.decide(COLOR, [0.95, 0.03, 0.02], CFG, ledger=led, explore=True, rng=rng) for _ in range(300)]
    for e in draws:
        assert e["gate"]["mode"] == "thompson_sampling" and e["gate"]["exploration"] is True
        assert 0 < e["gate"]["sampled_accuracy"] < 1
        assert any("exploration mode" in w for w in e["warnings"])
        assert e["gate"]["passed"] == (e["gate"]["sampled_accuracy"] > 0.75)
    acted = np.mean([e["tier"] == "act" for e in draws])
    want = 1 - __import__("scipy").stats.beta.cdf(0.75, 4, 2)    # P(draw > 0.75) under Beta(1 + 3, 1 + 1)
    assert acted == pytest.approx(want, abs=0.08)
    # exploration never lifts a cap
    capped = P.decide(COLOR, [0.95, 0.03, 0.02], CFG, ledger=led, explore=True, rng=rng, calibrated=False)
    assert capped["tier"] == "escalate"


# ----------------------------------------------------------------------------- holdout evaluation and verify

def calibrated_holdout(n, rng, sharpness=4.0, overstate=0.0):
    """Gate-head holdout: P(true) from a logistic score; `overstate` makes the stated P(true) too high."""
    q = 1 / (1 + np.exp(-sharpness * rng.normal(size=n)))
    y = (rng.random(n) < q).astype(int)
    p = np.clip(q + overstate * (q > 0.5), 0, 1)
    return np.column_stack([1 - p, p]), y


def test_evaluate_counts_every_row_per_stakes():
    rng = np.random.default_rng(1)
    probs, y = calibrated_holdout(800, rng)
    ev = P.evaluate(GATE, probs, y, CFG)
    assert ev["rows"] == 800 and set(ev) >= {"low", "medium", "high", "note"}
    for name in ("low", "medium", "high"):
        assert sum(ev[name]["tiers"].values()) == 800 and set(ev[name]["tiers"]) == {"escalate", "confirm", "act"}
        assert ev[name]["act"]["n"] == ev[name]["tiers"]["act"]
        assert ev[name]["act_or_confirm"]["n"] == ev[name]["tiers"]["act"] + ev[name]["tiers"]["confirm"]
    act = ev["medium"]["act"]
    assert act["wilson_95"][0] <= act["accuracy"] <= act["wilson_95"][1] and act["mean_predicted"] >= 0.85


def test_verify_levels():
    rng = np.random.default_rng(2)
    probs, y = calibrated_holdout(1500, rng)
    good = P.evaluate(SYM, probs, y, CFG)
    v = P.verify(True, 1500, good, CFG)
    assert all(v[s]["max_tier"] == "act" and v[s]["tier"] == 2 for s in ("low", "medium", "high"))
    assert "Wilson" in v["medium"]["reason"]
    # no skill -> escalate, whatever the rest says
    v = P.verify(False, 1500, good, CFG)
    assert all(v[s]["max_tier"] == "escalate" and "no skill" in v[s]["reason"] for s in v)
    # too few trusted holdout rows -> confirm
    v = P.verify(True, 20, good, CFG)
    assert all(v[s]["max_tier"] == "confirm" and "only 20 trusted holdout rows" in v[s]["reason"] for s in v)
    # too few act-tier rows at one stakes level -> confirm there only
    few = P.evaluate(SYM, *calibrated_holdout(200, np.random.default_rng(3), sharpness=1.0), CFG)
    v = P.verify(True, 200, few, CFG)
    assert v["high"]["max_tier"] == "confirm" and "act-tier rows at high stakes" in v["high"]["reason"]


def test_verify_caps_act_tier_rows_that_came_true_less_often_than_predicted():
    rng = np.random.default_rng(4)
    probs, y = calibrated_holdout(3000, rng, overstate=0.08)
    v = P.verify(True, 3000, P.evaluate(SYM, probs, y, CFG), CFG)
    assert v["medium"]["max_tier"] == "confirm" and "overstate" in v["medium"]["reason"]


def test_verify_treats_a_p_value_rounded_to_zero_as_significant():
    """A grossly overconfident act tier has a p-value that rounds to 0.0; it must still cap at confirm."""
    ev = {s: {"act": {"n": 400, "correct": 200, "accuracy": 0.5, "wilson_95": [0.45, 0.55],
                      "mean_predicted": 0.97, "p_value_below_predicted": 0.0}} for s in ("low", "medium", "high")}
    v = P.verify(True, 1000, ev, CFG)
    assert all(v[s]["max_tier"] == "confirm" and "overstate" in v[s]["reason"] for s in v)
    ev["medium"]["act"]["p_value_below_predicted"] = None         # unknown is not evidence of a problem
    assert P.verify(True, 1000, ev, CFG)["medium"]["max_tier"] == "act"


def test_verify_caps_a_gate_head_without_a_conformal_threshold():
    rng = np.random.default_rng(5)
    probs, y = calibrated_holdout(1500, rng)
    ev = P.evaluate(GATE_CONFORMAL, probs, y, CFG)
    no_thr = {"selective": {"threshold": None, "target_error": 0.05, "reason": "no threshold keeps the error"},
              "holdout": {"n_selected": 0}}
    v = P.verify(True, 1500, ev, CFG, no_thr, GATE_CONFORMAL)
    assert all(v[s]["max_tier"] == "confirm" and "conformal" in v[s]["reason"] for s in v)
    broken = {"selective": {"threshold": 0.9, "target_error": 0.05},
              "holdout": {"consistent": False, "error_rate": 0.2}}
    v = P.verify(True, 1500, ev, CFG, broken, GATE_CONFORMAL)
    assert all(v[s]["max_tier"] == "confirm" and "exceeds the target" in v[s]["reason"] for s in v)
    fine = {"selective": {"threshold": 0.9, "target_error": 0.05}, "holdout": {"consistent": True}}
    assert P.verify(True, 1500, ev, CFG, fine, GATE_CONFORMAL)["medium"]["max_tier"] == "act"
    # a symmetric head ignores the conformal block
    assert P.verify(True, 1500, ev, CFG, no_thr, SYM)["medium"]["max_tier"] == "act"


def test_weakest_link():
    names = lambda *ts: [{"tier": t} for t in ts]  # noqa: E731
    assert P.weakest(names("act", "confirm", "act"), CFG) == "confirm"
    assert P.weakest(names("act", "act"), CFG) == "act"
    assert P.weakest(names("act", "escalate", "confirm"), CFG) == "escalate"
    assert P.weakest([], CFG) == "escalate"
    hold = P.PolicyConfig(abstain="hold")
    assert P.weakest(names("act", "hold"), hold) == "hold" and P.weakest([], hold) == "hold"


def test_abstain_name_is_used_in_decisions():
    hold = P.PolicyConfig(abstain="hold")
    d = P.decide(COLOR, [0.4, 0.3, 0.3], hold)
    assert d["tier"] == "hold" and d["uncapped_tier"] == "hold"
