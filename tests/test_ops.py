"""Operations metrics and the review queue (core/ops.py) over hand-built decision, feedback and outcome rows."""
import itertools
import json

import numpy as np
import pytest

from open_rlcd.core import metrics as M
from open_rlcd.core import ops as O
from open_rlcd.core import policy as P
from open_rlcd.core.bernoulli import binomial_test_below
from open_rlcd.core.contract import ContractError
from open_rlcd.core.head import HeadSpec
from open_rlcd.core.uncertainty import entropy

V1, V2 = "rlcd-0.1.1", "rlcd-0.1.2"
_ids = itertools.count()


def H(cls, p, tier, *, stakes="medium", probs=None, unc=None, proposed=None, head=None, state=None):
    """One head entry of a decision record."""
    h = {"class": cls, "p": p, "tier": tier, "stakes": stakes}
    for k, v in (("probabilities", probs), ("uncertainty", unc), ("proposed_tier", proposed), ("head", head),
                 ("state", state)):
        if v is not None:
            h[k] = v
    return h


def D(heads, *, did=None, kind="decide", at="2026-11-02T10:00:00Z", model=V1, action=None, latency=10.0,
      rollout=None, warnings=(), receipt=True, **extra):
    """One decision record (with a receipt unless receipt=False: a line written before receipts existed)."""
    row = {"decision_id": did or f"d{next(_ids)}", "created_at": at, "kind": kind, "model": model, "heads": heads}
    if action is not None:
        row["action"] = action
    row.update(extra)
    if receipt:
        row["receipt"] = {"format": "open-rlcd-receipt/1", "latency_ms": latency, "rollout": rollout,
                          "warnings": list(warnings), "state_digest": "sha256:s", "digest": "sha256:r",
                          "questions": {"digest": "sha256:q", "battery": None}}
    return row


def F(did, head, label, **kw):
    return {"id": f"f{next(_ids)}", "received_at": "2026-11-05T00:00:00Z", "decision_id": did, "head": head,
            "label": label, "weight": 1.0, "source": "test", **kw}


def OUT(did, **kw):
    return {"id": f"o{next(_ids)}", "received_at": "2026-11-05T00:00:00Z", "decision_id": did, **kw}


def RO(*, capped=False, proposed=None, mode="full", enabled=True):
    out = {"mode": mode, "automation": {"enabled": enabled}, "capped": capped}
    if proposed is not None:
        out["proposed_action"] = proposed
    return out


def dumps(x):
    """The report must survive the API's JSON encoding: no NaN, no numpy scalars, string keys only."""
    return json.dumps(x, allow_nan=False)


def labelled(head, n, k, *, tier="act", proposed=None, p=0.95, stakes="medium", model=V1, cls="x", wrong="y"):
    """n decisions of one head at `tier` with feedback: the first k labelled right."""
    rows, fb = [], []
    for i in range(n):
        d = D({head: H(cls, p, tier, stakes=stakes, proposed=proposed)}, action=tier, model=model)
        rows.append(d)
        fb.append(F(d["decision_id"], head, cls if i < k else wrong))
    return rows, fb


# ----------------------------------------------------------------------------- empty input and the window

def test_empty_logs_give_an_empty_report():
    rep = O.metrics([], [], [], review_capacity_per_day=10)
    assert rep["window"]["decisions"] == 0 and rep["heads"] == {} and rep["actions"]["n"] == 0
    assert rep["actions"]["automatic"] == {"count": 0, "n": 0, "rate": None}
    assert rep["coverage"]["feedback"]["rate"] is None and rep["coverage"]["outcome"]["rate"] is None
    assert rep["outcomes"]["task_success"]["rate"] is None and rep["outcomes"]["cost"]["n"] == 0
    assert rep["review_load"]["days"] is None and rep["review_load"]["review_over_capacity"] is None
    assert rep["latency_ms"] == {} and rep["plans"] is None and rep["evaluate"] is None
    assert rep["binding"]["records"] == 0 and rep["drift_suspected"] == []
    assert "random sample" in rep["notes"][0]
    dumps(rep)
    assert O.review_queue([], []) == []
    assert O.metrics(iter([]), None, None)["window"]["decisions"] == 0      # generators and missing logs


def test_window_bounds_and_filters():
    rows = [D({"a": H("x", 0.9, "act")}, did="early", at="2026-11-01T00:00:00Z", action="act"),
            D({"a": H("x", 0.9, "act")}, did="mid", at="2026-11-02T12:00:00Z", action="act"),
            D({"b": H("x", 0.9, "act")}, did="late", at="2026-11-03T00:00:00Z", action="act", model=V2),
            D({"a": H("x", 0.9, None, head="a")}, did="sys", kind="systemone", at="2026-11-02T13:00:00Z"),
            {**D({"a": H("x", 0.9, "act")}, did="undated", action="act"), "created_at": None}]
    assert O.metrics(rows, [], [])["window"]["decisions"] == 5      # no bound: an undated record still counts
    w = O.metrics(rows, [], [], since="2026-11-02", until="2026-11-03")["window"]
    assert (w["decisions"], w["undated_excluded"]) == (2, 1)        # until is exclusive, undated left out
    assert w["since"] == "2026-11-02T00:00:00Z" and w["until"] == "2026-11-03T00:00:00Z"
    assert w["first"] == "2026-11-02T12:00:00Z" and w["last"] == "2026-11-02T13:00:00Z"
    assert O.metrics(rows, [], [], since="2026-11-01T00:00:00Z", until="2026-11-02")["window"]["decisions"] == 1
    # a naive time is UTC
    assert O.metrics(rows, [], [], since="2026-11-02T12:00:00")["window"]["decisions"] == 3
    assert O.metrics(rows, [], [], model=V2)["window"]["decisions"] == 1
    assert O.metrics(rows, [], [], kind="systemone")["window"]["kinds"] == {"systemone": 1}
    rep = O.metrics(rows, [], [], head="b")
    assert rep["window"]["decisions"] == 1 and list(rep["heads"]) == ["b"]
    assert rep["window"]["filters"] == {"model": None, "kind": None, "head": "b"}
    assert O.metrics([*rows, "torn", None], [], [])["window"]["unreadable"] == 2
    with pytest.raises(ContractError) as e:
        O.metrics(rows, [], [], since="last tuesday")
    assert e.value.details["field"] == "since" and e.value.status == 422
    with pytest.raises(ContractError):
        O.metrics(rows, [], [], since="2026-11-03", until="2026-11-02")


# ----------------------------------------------------------------------------- actions and rollout

def test_action_mix_counts_and_shares():
    rows = ([D({"a": H("x", 0.9, "act")}, action="act") for _ in range(3)]
            + [D({"a": H("x", 0.7, "confirm")}, action="confirm") for _ in range(2)]
            + [D({"a": H("x", 0.3, "escalate")}, action="escalate")]
            + [D({"grounded": H("true", 0.8, "confirm")}, kind="evaluate", action="confirm", verdict="review"),
               D({"grounded": H("true", 0.2, "escalate")}, kind="evaluate", action="escalate", verdict="reject"),
               D({"q": H("x", 0.9, None, head="a")}, kind="systemone")])
    a = O.metrics(rows, [], [])["actions"]
    assert a["n"] == 8 and a["derived"] == 0 and a["counts"] == {"confirm": 3, "act": 3, "escalate": 2}
    assert a["automatic"] == {"count": 3, "n": 8, "rate": 0.375}
    assert a["review"] == {"count": 3, "n": 8, "rate": 0.375} and a["abstain"]["count"] == 2
    assert "wilson_95" not in a["automatic"]          # a count of what happened, not an estimate
    assert a["by_kind"]["decide"]["automatic"]["rate"] == 0.5 and a["by_kind"]["evaluate"]["abstain"]["count"] == 1
    assert "unrecognised" not in a


def test_an_unknown_abstain_name_is_reported_not_guessed():
    rows = [D({"a": H("x", 0.3, "defer")}, action="defer"), D({"a": H("x", 0.3, "hold")}, action="hold")]
    rep = O.metrics(rows, [], [])
    assert rep["actions"]["unrecognised"] == {"defer": 1} and rep["actions"]["abstain"]["count"] == 1  # hold
    assert any("abstain_names" in n for n in rep["notes"])
    rep = O.metrics(rows, [], [], abstain_names=("defer",))
    assert rep["actions"]["abstain"]["count"] == 1 and rep["actions"]["unrecognised"] == {"hold": 1}
    assert [i["decision_id"] for i in O.review_queue(rows, [], abstain_names=("defer",))] == [rows[0]["decision_id"]]


def test_rollout_capped_proposed_and_effective_actions():
    rows = [D({"a": H("x", 0.95, "confirm", proposed="act")}, action="confirm",
              rollout=RO(capped=True, proposed="act", mode="shadow")),
            D({"a": H("x", 0.95, "act")}, action="act", rollout=RO(proposed="act")),
            # a pack whose action comes from a chain: a non-chain head's weaker tier became the "proposal"
            D({"a": H("x", 0.95, "confirm", proposed="act"), "b": H("y", 0.2, "escalate")}, action="confirm",
              rollout=RO(capped=True, proposed="escalate", mode="shadow")),
            D({"a": H("x", 0.95, "confirm", proposed="act")}, action="confirm",
              rollout=RO(capped=True, proposed="act", enabled=False)),
            D({"a": H("x", 0.95, "act")}, action="act", receipt=False)]
    ro = O.metrics(rows, [], [])["rollout"]
    assert ro["receipts"] == 4 and ro["capped"] == 3 and ro["automation_off"] == 1
    assert ro["modes"] == {"full": 2, "shadow": 2}
    assert ro["proposed"] == {"act": 4, "escalate": 1}
    assert ro["transitions"] == {"act -> confirm": 2, "escalate -> confirm": 1} and ro["incoherent"] == 1
    assert any("proposed_action" in n for n in O.metrics(rows, [], [])["notes"])


# ----------------------------------------------------------------------------- legacy, systemone and rank records

def test_legacy_records_without_receipts_are_read_and_their_action_derived():
    legacy_decide = {"decision_id": "old1", "created_at": "2026-10-01T00:00:00Z", "kind": "decide", "model": V1,
                     "heads": {"a": {"class": "x", "p": 0.9, "tier": "act", "state": {"q": 1}},
                               "b": {"class": "true", "p": 0.6, "tier": "confirm", "state": {"q": 1}}}}
    legacy_docs = {"decision_id": "old2", "created_at": "2026-10-01T01:00:00Z", "kind": "decide", "model": V1,
                   "heads": {"route": {"class": "vector", "p": 0.9, "tier": "act"}},
                   "plan": {"next": "answer", "call_llm": True, "route": "vector"}, "action": "act"}
    legacy_eval = {"decision_id": "old3", "created_at": "2026-10-01T02:00:00Z", "kind": "evaluate", "model": V1,
                   "heads": {"grounded": {"class": "true", "p": 0.4, "tier": "escalate"}}, "verdict": "reject",
                   "action": "escalate", "stakes": "high"}
    odd = {"decision_id": "old4", "created_at": "2026-10-01T03:00:00Z", "kind": "decide", "heads": ["not", "a", "dict"]}
    rows = [legacy_decide, legacy_docs, legacy_eval, odd, D({"a": H("x", 0.9, "act")}, action="act", latency=42.0)]
    rep = O.metrics(rows, [F("old1", "b", "true"), F("old1", "a", "z")], [OUT("old2", task_success=True)])
    dumps(rep)
    assert rep["actions"]["n"] == 4 and rep["actions"]["derived"] == 1 and rep["actions"]["counts"]["confirm"] == 1
    assert rep["latency_ms"]["decide"]["n"] == 1 and rep["latency_ms"]["decide"]["p50"] == 42.0
    assert rep["heads"]["b"]["feedback"]["accuracy"]["count"] == 1
    assert rep["heads"]["a"]["feedback"]["false_automation"] == {"count": 1, "n": 1, "rate": 1.0,
                                                                 "wilson_95": [M._f(M.wilson(1, 1)[0]), 1.0]}
    assert rep["plans"]["next"] == {"answer": 1} and rep["evaluate"]["next"] == {"unset": 1}
    assert rep["outcomes"]["task_success"]["count"] == 1
    assert any("before receipts" in n and "derived" in n for n in rep["notes"])
    queue = O.review_queue(rows, [])
    assert [i["decision_id"] for i in queue] == ["old1", "old3"]
    assert queue[0]["action"] == "confirm" and queue[0]["action_derived"] is True and "receipt" not in queue[0]
    assert queue[0]["state"] == {"q": 1} and "state" not in queue[0]["heads"]["a"]    # one shared state, shown once
    assert any("derived" in w for w in queue[0]["why"])


def test_systemone_and_rank_records_have_no_tier_and_never_queue():
    sys1 = D({"q1": H("red", 0.6, None, head="color", probs={"red": 0.6, "green": 0.3, "blue": 0.1}),
              "q2": H("true", 0.8, None, head="good")}, kind="systemone", did="s1")
    sys2 = D({"a": H("red", 0.5, None, head="color"), "b": H("blue", 0.4, None, head="color")}, kind="systemone",
             did="s2")
    rank = D({}, kind="rank", did="r1", head="good", option=None, ranking=[{"id": "c1", "probability": 0.7}])
    rep = O.metrics([sys1, sys2, rank], [F("s1", "color", "red"), F("s2", "color", "red")], [])
    dumps(rep)
    assert rep["actions"]["n"] == 0 and rep["window"]["kinds"] == {"systemone": 2, "rank": 1}
    color = rep["heads"]["color"]
    assert color["decisions"] == 3 and color["untiered"] == 3 and color["low_confidence"] == {"count": 0, "n": 0,
                                                                                               "rate": None}
    assert color["feedback"]["outcomes"] == 1 and color["feedback"]["accuracy"]["rate"] == 1.0
    assert color["feedback"]["calibration"]["n"] == 1
    # s2 asked head color in two questions: the feedback row (it names the head, not the question) is ambiguous
    assert rep["feedback"]["ambiguous"] == 1 and any("several questions" in n for n in rep["notes"])
    assert rep["latency_ms"]["rank"]["n"] == 1
    assert O.metrics([sys1, sys2, rank], [], [], head="good")["window"]["decisions"] == 2
    assert O.review_queue([sys1, sys2, rank], []) == []


# ----------------------------------------------------------------------------- per head

def test_per_head_mix_entropy_low_confidence_and_uncertainty_sources():
    uniform = {"red": 1 / 3, "green": 1 / 3, "blue": 1 / 3}
    rows = [D({"color": H("red", 1 / 3, "escalate", probs=uniform, unc=["ambiguous", "diffuse"]),
               "good": H("true", 0.5, "escalate", unc=["ambiguous"])}, action="escalate"),
            D({"color": H("red", 1.0, "act", probs={"red": 1.0, "green": 0.0, "blue": 0.0}),
               "good": H("true", 0.9, "act")}, action="act"),
            # two allowed options listed with their unrenormalised probabilities: renormalised before the entropy
            D({"color": H("green", 0.3, "confirm", probs={"red": 0.3, "green": 0.3}, unc=["options_removed"],
                          proposed="act"),
               "good": H(None, 0.0, "escalate")}, action="escalate"),
            D({"color": H("red", 0.4, "confirm", probs={"red": 0.4})}, action="confirm")]   # one option: no entropy
    heads = O.metrics(rows, [], [])["heads"]
    c, g = heads["color"], heads["good"]
    assert c["decisions"] == 4 and c["tiers"] == {"confirm": 2, "act": 1, "escalate": 1}
    assert c["classes"] == {"red": 3, "green": 1} and c["no_class"] == 0
    assert c["entropy"] == {"mean": pytest.approx(2 / 3, abs=1e-6), "n": 3}
    # low confidence reads the policy tier before the rollout cap: a capped act is not a low-confidence decision
    assert c["low_confidence"] == {"count": 2, "n": 4, "rate": 0.5}
    assert c["uncertainty"] == {"ambiguous": 1, "diffuse": 1, "options_removed": 1} and c["rollout_capped"] == 1
    assert g["no_class"] == 1 and g["classes"] == {"true": 2}
    assert g["entropy"]["n"] == 2 and g["entropy"]["mean"] == pytest.approx((1.0 + entropy([0.9, 0.1])) / 2, abs=1e-6)


def test_feedback_join_the_last_row_wins_and_labels_compare_as_logged():
    d1 = D({"color": H("red", 0.8, "act"), "good": H("true", 0.9, "act"), "level": H("2", 0.7, "confirm")},
           did="d1", action="confirm")
    d2 = D({"color": H("blue", 0.6, "confirm")}, did="d2", action="confirm")
    fb = [F("d1", "color", "green"), F("d1", "color", "red"),        # a later row corrects the earlier one
          F("d1", "good", True), F("d1", "level", 2.0),              # bool / whole-float labels as the log has them
          F("d1", "route", "vector"),                               # a head d1 does not have
          F("gone", "color", "red"), {**F(None, "color", "red"), "decision_id": None}, "torn"]
    rep = O.metrics([d1, d2], fb, [])
    assert rep["feedback"] == {"rows": 7, "joined": 3, "corrected": 1, "unlinked": 1, "not_in_window": 1,
                               "unmatched": 1, "ambiguous": 0}
    assert rep["heads"]["color"]["feedback"]["accuracy"]["count"] == 1    # red, not green
    assert rep["heads"]["good"]["feedback"]["accuracy"]["rate"] == 1.0
    assert rep["heads"]["level"]["feedback"]["accuracy"]["rate"] == 1.0
    assert rep["coverage"]["feedback"] == {"count": 1, "n": 2, "rate": 0.5}
    # with a head filter only that head's feedback counts
    assert O.metrics([d1, d2], fb, [], head="color")["feedback"]["rows"] == 4


def test_false_automation_false_rejection_and_review_agreement():
    spec = [("act", "x", 3), ("act", "y", 1), ("confirm", "x", 2), ("confirm", "y", 1), ("escalate", "x", 1),
            ("escalate", "y", 2)]
    rows, fb = [], []
    for tier, label, times in spec:
        for _ in range(times):
            d = D({"a": H("x", 0.8, tier)}, action=tier)
            rows.append(d)
            fb.append(F(d["decision_id"], "a", label))
    none = D({"a": H(None, 0.0, "escalate")}, action="escalate")   # committed to nothing: no correctness
    rows.append(none)
    fb.append(F(none["decision_id"], "a", "x"))
    f = O.metrics(rows, fb, [])["heads"]["a"]["feedback"]

    def rate(k, n):
        lo, hi = M.wilson(k, n)
        return {"count": k, "n": n, "rate": M._f(k / n), "wilson_95": [M._f(lo), M._f(hi)]}

    assert f["outcomes"] == 11 and f["no_class"] == 1
    assert f["accuracy"] == rate(6, 10)
    assert f["false_automation"] == rate(1, 4)
    assert f["false_rejection"] == rate(1, 3)
    assert f["review_agreement"] == rate(2, 3)


def test_online_calibration_is_the_core_metrics_on_committed_class_and_outcome():
    rng = np.random.default_rng(5)
    p = rng.uniform(0.3, 1.0, size=300).round(6)
    y = rng.uniform(size=300) < p
    rows, fb = [], []
    for pi, yi in zip(p, y, strict=True):
        d = D({"a": H("x", float(pi), "act" if pi >= 0.85 else "confirm")}, action="act" if pi >= 0.85 else "confirm")
        rows.append(d)
        fb.append(F(d["decision_id"], "a", "x" if yi else "y"))
    cal = O.metrics(rows, fb, [])["heads"]["a"]["feedback"]["calibration"]
    yf = y.astype(float)
    assert cal["n"] == 300
    assert cal["ece"] == M._f(M.ece(p, yf)) and cal["brier"] == M._f(float(((p - yf) ** 2).mean()))
    assert cal["reliability"] == M.reliability_table(p, yf)
    assert cal["mean_predicted"] == M._f(p.mean()) and cal["observed_rate"] == M._f(yf.mean())


def test_policy_act_counts_the_tier_before_the_rollout_cap():
    rows, fb = labelled("a", 12, 9, tier="confirm", proposed="act", p=0.9)       # shadow mode: nothing automated
    f = O.metrics(rows, fb, [])["heads"]["a"]["feedback"]
    assert f["false_automation"]["n"] == 0 and f["review_agreement"] == {**f["review_agreement"], "count": 9, "n": 12}
    pa = f["policy_act"]["medium"]
    assert (pa["n"], pa["correct"], pa["accuracy"], pa["mean_predicted"]) == (12, 9, 0.75, 0.9)
    assert pa["p_value_below_predicted"] == M._f(binomial_test_below(9, 12, 0.9))
    assert pa["wilson_95"] == [M._f(x) for x in M.wilson(9, 12)]


# ----------------------------------------------------------------------------- holdout comparison and drift

HOLDOUT = {"model": V1, "n": 500, "accuracy": 0.9, "ece": 0.02, "brier": None,
           "act": {"medium": {"n": 200, "correct": 190, "accuracy": 0.95, "wilson_95": [0.91, 0.973]}}}


def test_drift_is_not_suspected_on_too_few_outcomes():
    rows, fb = labelled("a", 10, 0)
    h = O.metrics(rows, fb, [], holdout={"a": HOLDOUT})["heads"]["a"]
    assert h["drift_suspected"] is False
    check = h["vs_holdout"]["drift"][0]
    assert check["enough_outcomes"] is False and "at least 30" in check["why"]


def test_drift_is_suspected_when_the_intervals_separate():
    rows, fb = labelled("a", 40, 24)
    rep = O.metrics(rows, fb, [], holdout={"a": HOLDOUT, "absent": HOLDOUT})
    h = rep["heads"]["a"]
    assert M.wilson(24, 40)[1] < M.wilson(190, 200)[0]
    assert h["drift_suspected"] is True and rep["drift_suspected"] == ["a"] and "absent" not in rep["heads"]
    check = h["vs_holdout"]["drift"][0]
    assert check["stakes"] == "medium" and check["online"]["n"] == 40 and check["holdout"]["n"] == 200
    assert "lower bound" in check["compared_with"]
    assert h["vs_holdout"]["holdout"]["ece"] == 0.02 and h["vs_holdout"]["online"]["outcomes"] == 40


def test_drift_is_not_suspected_when_the_intervals_overlap():
    rows, fb = labelled("a", 40, 33)
    assert M.wilson(33, 40)[1] > M.wilson(190, 200)[0]       # overlapping 95% intervals: no claim
    assert O.metrics(rows, fb, [], holdout={"a": HOLDOUT})["heads"]["a"]["drift_suspected"] is False
    # without a holdout row count the holdout accuracy is taken as known, and the same data cross it
    point = {"model": V1, "act": {"medium": {"accuracy": 0.95}}}
    assert M.wilson(33, 40)[1] < 0.95
    h = O.metrics(rows, fb, [], holdout={"a": point})["heads"]["a"]
    assert h["drift_suspected"] is True and "no holdout row count" in h["vs_holdout"]["drift"][0]["compared_with"]


def test_drift_compares_only_the_holdouts_own_model_version_and_stakes():
    good, fb1 = labelled("a", 40, 39, model=V1)
    bad, fb2 = labelled("a", 40, 10, model=V2)
    h = O.metrics(good + bad, fb1 + fb2, [], holdout={"a": HOLDOUT})["heads"]["a"]
    assert h["feedback"]["outcomes"] == 80 and h["vs_holdout"]["online"]["outcomes"] == 40
    assert h["drift_suspected"] is False
    other, fb3 = labelled("a", 40, 10, stakes="high")
    h = O.metrics(other, fb3, [], holdout={"a": HOLDOUT})["heads"]["a"]
    assert h["vs_holdout"]["drift"] == [] and h["drift_suspected"] is False


def test_holdout_summary_reads_a_real_holdout_report():
    rng = np.random.default_rng(1)
    n = 300
    P3 = rng.dirichlet([1.0, 1.0, 1.0], size=n)
    y3 = np.array([rng.choice(3, p=row) for row in P3])
    rep = M.calibration_report(P3, y3, ["red", "green", "blue"], np.full(3, 1 / 3))
    spec = HeadSpec("color", "choice", ("red", "green", "blue"), view="v")
    rep["policy"] = {"holdout": P.evaluate(spec, P3, y3, P.PolicyConfig())}
    s = O.holdout_summary(rep, decision="argmax", model=V1)
    assert (s["model"], s["n"], s["accuracy"], s["ece"], s["brier"]) == (V1, n, rep["accuracy"], rep["ece"], None)
    assert set(s["act"]) == {"low", "medium", "high"}
    assert s["act"]["medium"] == {k: rep["policy"]["holdout"]["medium"]["act"][k]
                                  for k in ("n", "correct", "accuracy", "wilson_95", "mean_predicted")}
    pt = rng.uniform(size=n)
    yb = (rng.uniform(size=n) < pt).astype(int)
    P2 = np.c_[1 - pt, pt]
    rep2 = M.calibration_report(P2, yb, ["false", "true"], np.array([0.5, 0.5]))
    rep2["policy"] = {"holdout": P.evaluate(HeadSpec("good", "noul", view="v", decision="gate"), P2, yb,
                                            P.PolicyConfig())}
    g = O.holdout_summary(rep2, decision="gate", model=V1)
    assert g["accuracy"] == rep2["class_rates_holdout"]["true"]
    assert g["ece"] == rep2["per_class"]["true"]["ece"] and g["brier"] == rep2["per_class"]["true"]["brier"]
    assert O.holdout_summary(None) == {"model": None, "decision": "argmax", "n": None, "accuracy": None,
                                       "ece": None, "brier": None, "act": {}}
    rows, fb = labelled("color", 5, 5, cls="red", wrong="blue")
    h = O.metrics(rows, fb, [], holdout={"color": s})["heads"]["color"]["vs_holdout"]
    assert h["holdout"]["ece"] == rep["ece"] and h["model"] == V1
    dumps(h)


# ----------------------------------------------------------------------------- binding links

def test_binding_links_heads_rollout_rules_and_evaluate():
    chain = ["response_type", "route", "answerable"]
    rows = [D({"a": H("x", 0.9, "act"), "b": H("y", 0.7, "confirm")}, action="confirm"),       # b alone
            D({"a": H("x", 0.7, "confirm"), "b": H("y", 0.7, "confirm")}, action="confirm"),   # a and b tie
            D({"a": H("x", 0.9, "act"), "b": H("y", 0.9, "act")}, action="act"),               # nothing held it
            D({"a": H("x", 0.9, "confirm", proposed="act"), "b": H("y", 0.9, "confirm", proposed="act")},
              action="confirm", rollout=RO(capped=True, proposed="act", mode="shadow")),       # the rollout cap
            # documents decide: needs_graph is not in the chain; no evidence made the plan abstain (a pack rule)
            D({"response_type": H("direct_answer", 0.9, "act"), "route": H("vector", 0.9, "act"),
               "answerable": H("true", 0.9, "act"), "needs_graph": H("false", 0.6, "escalate")},
              action="escalate", plan={"chain": chain, "next": "retrieve_more", "call_llm": False}),
            D({}, kind="evaluate", action="escalate", decided_by="checks", verdict="reject", failed=[]),
            # evaluate: complete acted but its level (2) cannot ship: its failure explains the review
            D({"grounded": H("true", 0.95, "act"), "complete": H("2", 0.9, "act"),
               "risk_tier": H("0", 0.9, "act")}, kind="evaluate", action="confirm", decided_by="heads",
              verdict="review", failed=["complete"])]
    b = O.metrics(rows, [], [])["binding"]
    assert b["records"] == 7 and b["unbound"] == 1
    assert b["by"] == {"a": 1, "b": 2, "complete": 1, "decided_by:checks": 1, "pack_rule": 1, "rollout_cap": 1}
    assert b["heads"]["b"] == {"candidate": 4, "binds": 2, "sole": 1, "rate": 0.5}
    assert b["heads"]["a"] == {"candidate": 4, "binds": 1, "sole": 0, "rate": 0.25}
    assert b["heads"]["complete"]["sole"] == 1 and b["heads"]["risk_tier"]["binds"] == 0
    assert "needs_graph" not in b["heads"] and b["heads"]["answerable"]["candidate"] == 1
    assert b["never_bind"] == []           # fewer than MIN_RECORDS records: no head is listed yet


def test_heads_that_never_bind_nor_vary_are_latency_only_candidates():
    rows = []
    for i in range(O.MIN_RECORDS):
        b_tier = "confirm" if i % 2 else "act"
        rows.append(D({"guard": H("ok", 0.99, "act"), "b": H("y" if i % 3 else "z", 0.8, b_tier),
                       "flag": H("false", 0.9, "act")}, action=b_tier))
        rows.append(D({"route": H("vector", 0.9, "act"), "needs_graph": H("false", 0.9, "act" if i % 2 else "confirm"),
                       "side": H("u" if i % 2 else "v", 0.9, "act")}, action="act",
                      plan={"chain": ["route"], "next": "answer"}))
    b = O.metrics(rows, [], [])["binding"]
    assert b["heads"]["guard"] == {"candidate": 30, "binds": 0, "sole": 0, "rate": 0.0}
    assert set(b["never_bind"]) == {"guard", "flag", "route"} and "b" not in b["never_bind"]
    assert set(b["never_vary"]) == {"guard", "flag", "route", "needs_graph"}
    # needs_graph is never a candidate link and always answers false; side is never a link but varies
    assert set(b["latency_only_candidates"]) == {"guard", "flag", "route", "needs_graph"}
    assert any("never binds" in n for n in O.metrics(rows, [], [])["notes"])
    assert O.metrics(rows, [], [], head="guard")["binding"]["latency_only_candidates"] == ["guard"]


# ----------------------------------------------------------------------------- plans, evaluations, outcomes

def test_documents_plans_and_evaluate_distributions():
    def plan(nxt, route):
        return {"response_type": "direct_answer", "route": route, "next": nxt, "call_llm": nxt == "answer",
                "chain": ["route"]}

    rows = [D({"route": H("vector", 0.9, "act")}, action="act", plan=plan("answer", "vector")),
            D({"route": H("vector", 0.9, "act")}, action="act", plan=plan("answer", "vector")),
            D({"route": H("graph", 0.6, "confirm")}, action="escalate", plan=plan("retrieve_more", "graph")),
            D({"route": H("vector", 0.5, "escalate")}, action="escalate", plan=plan("escalate_to_human", "vector")),
            D({"grounded": H("true", 0.9, "act")}, kind="evaluate", action="act", verdict="ship", next="ship",
              decided_by="heads", failed=[]),
            D({"grounded": H("true", 0.4, "escalate")}, kind="evaluate", action="escalate", verdict="reject",
              next="revise", decided_by="heads", failed=["grounded", "complete"]),
            D({}, kind="evaluate", action="escalate", verdict="reject", next="escalate_to_human", decided_by="checks",
              failed=[])]
    rep = O.metrics(rows, [], [])
    assert rep["plans"] == {"n": 4, "next": {"answer": 2, "escalate_to_human": 1, "retrieve_more": 1},
                            "call_llm": {"false": 2, "true": 2}, "response_type": {"direct_answer": 4},
                            "route": {"vector": 3, "graph": 1}}
    assert rep["evaluate"] == {"n": 3, "verdict": {"reject": 2, "ship": 1},
                               "next": {"escalate_to_human": 1, "revise": 1, "ship": 1},
                               "decided_by": {"heads": 2, "checks": 1}, "failed": {"complete": 1, "grounded": 1}}


def test_outcomes_join_task_success_cost_latency_and_retries():
    d1 = D({"route": H("vector", 0.9, "act")}, did="o1", action="act", plan={"route": "vector", "next": "answer"})
    d2 = D({"route": H("graph", 0.7, "confirm")}, did="o2", action="confirm", plan={"route": "graph", "next": "answer"})
    d3 = D({"grounded": H("true", 0.9, "act")}, did="o3", kind="evaluate", action="act", next="ship")
    d4 = D({"q": H("x", 0.9, None, head="a")}, did="o4", kind="systemone")
    outs = [OUT("o1", task_success=False, route_taken="vector"),
            OUT("o1", task_success=True, route_taken="vector", provider="p1", fallback=False, cost=0.01,
                latency_ms=1200, retries={"transport": 2, "workflow": 0}, tokens={"input": 100, "output": 20}),
            OUT("o2", task_success=False, route_taken="vector", provider="p2", fallback=True, cost=0.03,
                latency_ms=3000, retries={"transport": 0, "workflow": 1}, tokens={"input": 50}),
            OUT("o3", task_success=False, cost=0.02, latency_ms=800),
            OUT("o4", route_taken="cache"),
            OUT("missing", task_success=True), {"id": "x", "task_success": True}]
    rep = O.metrics([d1, d2, d3, d4], [], outs)
    o = rep["outcomes"]
    dumps(o)
    assert (o["rows"], o["joined"], o["corrected"], o["not_in_window"], o["unlinked"]) == (7, 4, 1, 1, 1)
    assert o["task_success"]["count"] == 1 and o["task_success"]["n"] == 3
    assert o["task_success"]["wilson_95"] == [M._f(x) for x in M.wilson(1, 3)]
    assert o["task_success_by_kind"]["decide"]["rate"] == 0.5 and o["task_success_by_kind"]["evaluate"]["rate"] == 0.0
    assert set(o["task_success_by_action"]) == {"act", "confirm"} and o["task_success_by_action"]["act"]["n"] == 2
    assert o["task_success_by_next"] == {"answer": o["task_success_by_next"]["answer"],
                                         "ship": o["task_success_by_next"]["ship"]}
    assert o["task_success_by_route"]["vector"]["n"] == 2 and o["task_success_by_provider"]["p1"]["rate"] == 1.0
    assert o["false_automation"]["count"] == 1 and o["false_automation"]["n"] == 2      # o3 acted and failed
    assert o["routes_taken"] == {"vector": 2, "cache": 1}
    assert o["planned_vs_taken"] == {"graph -> vector": 1, "vector -> vector": 1}
    assert o["fallback"]["count"] == 1 and o["fallback"]["n"] == 2
    assert o["cost"]["n"] == 3 and o["cost"]["sum"] == pytest.approx(0.06)
    assert o["cost"]["p50"] == M._f(np.percentile([0.01, 0.03, 0.02], 50))
    assert o["latency_ms"]["p95"] == M._f(np.percentile([1200, 3000, 800], 95)) and o["latency_ms"]["max"] == 3000
    assert o["retries"] == {"reported": 2, "transport": {"sum": 2, "outcomes": 1},
                            "workflow": {"sum": 1, "outcomes": 1}}
    assert o["tokens"] == {"input": 150, "output": 20}
    assert rep["coverage"]["outcome"] == {"count": 4, "n": 4, "rate": 1.0}


def test_latency_percentiles_by_kind():
    rows = [D({"a": H("x", 0.9, "act")}, action="act", latency=float(ms)) for ms in range(1, 201)]
    rows += [D({"g": H("true", 0.9, "act")}, kind="evaluate", action="act", latency=ms) for ms in (5.0, 7.0, 9.0)]
    rep = O.metrics(rows, [], [])
    dec = rep["latency_ms"]["decide"]
    xs = np.arange(1, 201, dtype=float)
    assert dec["n"] == 200 and dec["p50"] == M._f(np.percentile(xs, 50)) and dec["p99"] == M._f(np.percentile(xs, 99))
    assert dec["p95"] == M._f(np.percentile(xs, 95)) and dec["max"] == 200.0
    assert rep["latency_ms"]["evaluate"]["p50"] == 7.0
    assert any("p99 of fewer than 100" in n for n in rep["notes"])


def test_by_model_and_by_stakes():
    rows = [D({"a": H("x", 0.9, "act")}, action="act", model=V1, stakes="low"),
            D({"a": H("x", 0.6, "confirm")}, action="confirm", model=V1, stakes="low"),
            D({"a": H("x", 0.3, "escalate")}, action="escalate", model=V2, stakes="high"),
            D({"a": H("x", 0.9, None, head="a")}, kind="systemone", model=V2)]
    rep = O.metrics(rows, [], [])
    assert rep["by_model"][V1]["automatic"] == {"count": 1, "n": 2, "rate": 0.5}
    assert rep["by_model"][V2]["decisions"] == 2 and rep["by_model"][V2]["abstain"]["count"] == 1
    assert rep["by_stakes"]["low"]["actions"] == {"act": 1, "confirm": 1} and set(rep["by_stakes"]) == {"low", "high"}
    assert any("2 model versions are pooled" in n for n in rep["notes"])
    assert not any("pooled" in n for n in O.metrics(rows, [], [], model=V1)["notes"])


# ----------------------------------------------------------------------------- review load and coverage

def _review_rows():
    rows = [D({"a": H("x", 0.7, "confirm")}, did=f"c{i}", at=f"2026-11-01T0{i}:00:00Z", action="confirm")
            for i in range(6)]
    rows += [D({"a": H("x", 0.7, "confirm")}, did=f"k{i}", at=f"2026-11-02T0{i}:00:00Z", action="confirm")
             for i in range(2)]
    rows += [D({"a": H("x", 0.2, "escalate")}, did="e0", at="2026-11-02T05:00:00Z", action="escalate"),
             D({"a": H("x", 0.9, "act")}, did="x0", at="2026-11-02T06:00:00Z", action="act")]
    return rows


def test_review_load_against_capacity():
    rows, fb = _review_rows(), [F("k0", "a", "x"), F("k1", "a", "y")]
    load = O.metrics(rows, fb, [], since="2026-11-01", until="2026-11-03", review_capacity_per_day=3)["review_load"]
    assert (load["confirm"], load["abstain"], load["days"], load["confirm_per_day"]) == (8, 1, 2.0, 4.0)
    assert load["review_over_capacity"] is True and load["utilisation"] == pytest.approx(4 / 3, abs=1e-6)
    assert load["days_over_capacity"] == 1 and load["peak_day"] == {"date": "2026-11-01", "confirm": 6, "abstain": 0}
    assert load["per_day"] == {"2026-11-01": {"confirm": 6, "abstain": 0}, "2026-11-02": {"confirm": 2, "abstain": 1}}
    assert load["pending"] == {"confirm": 6, "abstain": 1} and load["abstain_per_day"] == 0.5
    load = O.metrics(rows, fb, [], since="2026-11-01", until="2026-11-03", review_capacity_per_day=5)["review_load"]
    assert load["review_over_capacity"] is False and load["days_over_capacity"] == 1
    load = O.metrics(rows, fb, [])["review_load"]          # no bounds: first to last decision, at least a day
    assert load["days"] == 1.25 and load["capacity_per_day"] is None and load["review_over_capacity"] is None
    with pytest.raises(ContractError):
        O.metrics(rows, fb, [], review_capacity_per_day=0)


def test_coverage_counts_decisions_with_any_feedback_or_outcome():
    rows = _review_rows()[:4]
    fb = [F("c0", "a", "x"), F("c1", "nope", "x")]       # an unmatched head still is feedback on the decision
    cov = O.metrics(rows, fb, [OUT("c2", task_success=True)])["coverage"]
    assert cov == {"feedback": {"count": 2, "n": 4, "rate": 0.5}, "outcome": {"count": 1, "n": 4, "rate": 0.25}}


# ----------------------------------------------------------------------------- the review queue

def test_review_queue_lists_confirm_and_abstain_decisions_without_feedback():
    rows = _review_rows()
    q = O.review_queue(rows, [F("c0", "a", "x")])
    assert [i["decision_id"] for i in q] == ["c1", "c2", "c3", "c4", "c5", "k0", "k1", "e0"]
    assert q[-1]["queue"] == "abstain" and q[0]["queue"] == "confirm"
    assert [i["decision_id"] for i in O.review_queue(rows, [], oldest_first=False)][:3] == ["e0", "k1", "k0"]
    assert "e0" not in [i["decision_id"] for i in O.review_queue(rows, [], include_abstain=False)]
    assert len(O.review_queue(rows, [], limit=2)) == 2 and len(O.review_queue(rows, [], limit=None)) == 9
    assert O.review_queue(rows, [], limit=0) == []


def test_review_queue_filters_by_kind_and_by_the_head_still_unreviewed():
    two = D({"a": H("x", 0.9, "act"), "b": H("y", 0.6, "confirm")}, did="two", action="confirm")
    ev = D({"grounded": H("true", 0.7, "confirm")}, did="ev", kind="evaluate", action="confirm", verdict="review",
           next="human_review", decided_by="heads", failed=[])
    fb = [F("two", "a", "x")]
    assert O.review_queue([two, ev], fb, kind="evaluate")[0]["decision_id"] == "ev"
    assert [i["decision_id"] for i in O.review_queue([two, ev], fb)] == ["ev"]          # two has feedback
    assert [i["decision_id"] for i in O.review_queue([two, ev], fb, head="b")] == ["two"]   # but not for head b
    assert O.review_queue([two, ev], fb, head="a") == []


def test_review_queue_items_carry_what_a_reviewer_needs():
    state = {"query": "refund my duplicate charge"}
    capped = D({"route": H("vector", 0.95, "confirm", proposed="act", state=state),
                "answerable": H("true", 0.9, "confirm", proposed="act", state={**state, "evidence": ["c1"]})},
               did="cap", action="confirm", rollout=RO(capped=True, proposed="act", mode="shadow"),
               plan={"next": "answer", "call_llm": True, "chain": ["route", "answerable"], "reason": None},
               evidence=[{"id": "c1", "digest": "sha256:c"}], warnings=["rollout mode shadow (act became confirm)"])
    capped["receipt"]["audit"] = {"baseline": {"route": "llm_only"}, "tags": ["tenant:a"]}
    abstained = D({"answerable": H("true", 0.3, "escalate", unc=["below_threshold", "missing_evidence"],
                                   probs=None, state=state)},
                  did="abs", action="escalate", plan={"next": "retrieve_more", "reason": "answerable: the evidence "
                                                                                      "is not sufficient (P 0.30)"})
    ev = D({"grounded": H("true", 0.6, "confirm", state=state), "complete": H("3", 0.9, "act", state=state)},
           did="ev", kind="evaluate", action="confirm", verdict="review", next="human_review", decided_by="heads",
           failed=[], stakes="high")
    q = {i["decision_id"]: i for i in O.review_queue([capped, abstained, ev], [])}
    dumps(list(q.values()))
    c = q["cap"]
    assert c["plan"]["next"] == "answer" and c["evidence"] == [{"id": "c1", "digest": "sha256:c"}]
    assert c["heads"]["route"] == {"class": "vector", "p": 0.95, "tier": "confirm", "proposed_tier": "act",
                                   "stakes": "medium", "uncertainty": [], "state": state}
    assert "state" not in c                           # the heads read different states: each keeps its own
    assert c["rollout"] == {"mode": "shadow", "capped": True, "proposed_action": "act"}
    assert c["warnings"] == ["rollout mode shadow (act became confirm)"] and c["audit"]["baseline"]
    assert c["receipt"] == {"digest": "sha256:r", "state_digest": "sha256:s"}
    assert "the rollout (mode shadow) capped route, answerable from act at confirm" in c["why"]
    assert not any(w.startswith("route") for w in c["why"])      # a capped head did not bind by itself
    a = q["abs"]
    assert a["queue"] == "abstain" and a["heads"]["answerable"]["uncertainty"] == ["below_threshold",
                                                                                   "missing_evidence"]
    assert any(w.startswith("plan: answerable") for w in a["why"])
    assert "answerable at escalate (p 0.30, below_threshold, missing_evidence)" in a["why"]
    e = q["ev"]
    assert (e["verdict"], e["next"], e["decided_by"], e["stakes"]) == ("review", "human_review", "heads", "high")
    assert e["state"] == state and all("state" not in h for h in e["heads"].values())
    assert "grounded at confirm (p 0.60)" in e["why"]


def test_review_queue_names_capped_heads_from_the_heads_not_the_receipt_proposal():
    """A receipt's proposed_action can be one the rollout cannot explain; the heads' own proposed_tier is exact."""
    inc = D({"route": H("vector", 0.95, "confirm", proposed="act"), "needs_graph": H("false", 0.6, "escalate")},
            did="inc", action="confirm", rollout=RO(capped=True, proposed="escalate", mode="shadow"),
            plan={"next": "answer", "chain": ["route"]})
    off = D({"a": H("x", 0.95, "confirm", proposed="act")}, did="off", action="confirm",
            rollout=RO(capped=True, proposed="act", enabled=False))
    q = {i["decision_id"]: i["why"] for i in O.review_queue([inc, off], [])}
    assert "the rollout (mode shadow) capped route from act at confirm" in q["inc"]
    assert "the rollout (mode full, automation switched off) capped a from act at confirm" in q["off"]


# ----------------------------------------------------------------------------- real rows from the service

def test_metrics_and_queue_read_the_rows_the_service_writes(tmp_path):
    """No training: an untrained toy service still decides (every head abstains), logs receipts, takes feedback
    and outcomes; the report and the queue must read those exact rows."""
    from open_rlcd.core.config import Project
    from open_rlcd.core.contract import DecideRequest, FeedbackRequest, OutcomeRequest, RankRequest, SystemOneRequest
    from open_rlcd.core.service import Service
    from tests.toypack import ToyPack

    svc = Service(Project(root=tmp_path), home=tmp_path, pack=ToyPack())
    state = {"f": [0.1] * 6}
    d = svc.decide(DecideRequest(state=state, stakes="low"))
    svc.systemone(SystemOneRequest(state=state, receipt=True, questions={"c": {"type": "choice", "head": "color"}}))
    svc.rank(RankRequest(state=state, candidates=[{"id": "x", "state": {"f": [1.0] * 6}}], receipt=True))
    svc.feedback(FeedbackRequest(decision_id=d["decision_id"], head="color", label="red", source="test"))
    svc.outcome(OutcomeRequest(decision_id=d["decision_id"], task_success=False, retries={"transport": 1}))
    rep = O.metrics(svc.logs.decisions(), svc.logs.feedback(), svc.logs.outcomes(), review_capacity_per_day=50)
    dumps(rep)
    assert rep["window"]["kinds"] == {"decide": 1, "rank": 1, "systemone": 1}
    assert rep["actions"]["counts"] == {d["action"]: 1} and rep["actions"]["abstain"]["count"] == 1
    color = rep["heads"]["color"]
    assert color["decisions"] == 2 and color["feedback"]["outcomes"] == 1
    assert color["feedback"]["false_rejection"]["n"] == 1 and color["uncertainty"]["untrained"] == 1
    assert rep["outcomes"]["retries"]["transport"] == {"sum": 1, "outcomes": 1}
    assert rep["latency_ms"]["decide"]["n"] == 1 and rep["rollout"]["receipts"] == 1
    assert rep["review_load"]["pending"] == {"confirm": 0, "abstain": 0}       # it has feedback
    queue = O.review_queue(svc.logs.decisions(), [])
    assert [i["decision_id"] for i in queue] == [d["decision_id"]]
    assert queue[0]["state"] == state and queue[0]["receipt"]["digest"] and queue[0]["warnings"]


def test_a_proposed_action_is_one_the_rollout_cap_can_explain(tmp_path):
    from open_rlcd.core.config import Project
    from open_rlcd.core.service import Service
    from tests.toypack import ToyPack

    svc = Service(Project(root=tmp_path), home=tmp_path, pack=ToyPack())
    cfg = svc.policy_config()
    capped = {"decision": {"tier": "confirm", "rollout": {"capped": True, "proposed_tier": "act"}}}
    side = {"decision": {"tier": "escalate", "rollout": {"capped": False}}}
    # shadow mode, documents decide: the chain heads would act (capped at confirm); needs_graph is not in the chain
    # and abstains. The plan confirms; without the cap it would act. A proposal weaker than the action is impossible.
    assert svc.rollout_summary([capped, side], "confirm", cfg)["proposed_action"] in ("confirm", "act")
    # no evidence: the plan abstains by rule, which the rollout did not cause, so it proposed nothing stronger
    assert svc.rollout_summary([capped], "escalate", cfg)["proposed_action"] == "escalate"


def test_labels_for_two_questions_on_one_head_both_train(tmp_path):
    from open_rlcd.core.config import Project
    from open_rlcd.core.contract import FeedbackRequest, SystemOneRequest
    from open_rlcd.core.service import Service
    from open_rlcd.packs.base import dataset_from_labels
    from tests.toypack import ToyPack, ToyProvider

    pack = ToyPack()
    svc = Service(Project(root=tmp_path), home=tmp_path, pack=pack)
    state = {"x": {"f": [0.1] * 6}, "y": {"f": [-0.5] * 6}}
    r = svc.systemone(SystemOneRequest(state=state, receipt=True, questions={
        "a": {"type": "choice", "head": "color", "path": "x"}, "b": {"type": "choice", "head": "color", "path": "y"}}))
    did = r["receipt"]["id"]
    svc.feedback(FeedbackRequest(decision_id=did, head="a", label="red", source="review"))
    svc.feedback(FeedbackRequest(decision_id=did, head="b", label="blue", source="review"))
    ds = dataset_from_labels(pack.heads(svc.project)["color"], ToyProvider(), [], svc.logs.feedback())
    assert len(ds.y) == 2       # two predictions on two different states, two labels: both are training rows
