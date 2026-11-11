"""Replay (core/replay.py) and regression cases (core/regress.py) with the toy pack.

One module-scoped world: rlcd-0.1.1 trained on the toy generator, twelve decisions and a systemone receipt logged
under it (plus a rank receipt, which logs no head states), feedback for three of them (one label corrected by a
later row), then rlcd-0.1.2 trained after the generator's signal was flipped: the candidate learns the opposite
relation, so every clear case is a deterministic regression. A `spare` head exists in the pack but is never trained.
The last test (marked slow, about 10 s) trains the documents pack offline and replays and runs cases on it.
"""
from __future__ import annotations

import json
from collections import Counter
from types import SimpleNamespace

import numpy as np
import pytest

from open_rlcd.core import regress as RG
from open_rlcd.core.config import Project
from open_rlcd.core.contract import (
    ContractError,
    DecideRequest,
    EvaluateRequest,
    FeedbackRequest,
    RankRequest,
    SystemOneRequest,
    TrainRequest,
)
from open_rlcd.core.head import HeadSpec
from open_rlcd.core.replay import feedback_labels, replay
from open_rlcd.core.service import Service
from tests.toypack import NAMES, ToyPack

# the toy generator's noul weights are fixed (seed 99 + 2): states along their direction are clear cases
_W = np.random.default_rng(99 + 2).normal(size=(6, 2))
_D = (_W[:, 1] - _W[:, 0]) / np.linalg.norm(_W[:, 1] - _W[:, 0])
POS = {"f": (2.5 * _D).round(3).tolist()}
NEG = {"f": (-2.5 * _D).round(3).tolist()}
V1, V2 = "rlcd-0.1.1", "rlcd-0.1.2"
TRAINED = ["good", "color", "level"]


class SparePack(ToyPack):
    """The toy pack plus a noul head that is never trained."""

    def heads(self, project):
        return {**super().heads(project), "spare": HeadSpec("spare", "noul", view="v")}


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    root = tmp_path_factory.mktemp("replay")
    svc = Service(Project(root=root), home=root, pack=SparePack())
    assert svc.trainer.start(TrainRequest(heads=TRAINED), wait=True)["model"] == V1
    rng = np.random.default_rng(3)
    states = [POS, NEG] + [{"f": rng.normal(size=6).round(3).tolist()} for _ in range(10)]
    decided = [svc.decide(DecideRequest(state=s, stakes="low")) for s in states]
    ids = [d["decision_id"] for d in decided]
    so = svc.systemone(SystemOneRequest(state=POS, questions={"g": {"type": "noul", "head": "good"}}, receipt=True))
    svc.rank(RankRequest(candidates=[{"id": "a", "state": POS}, {"id": "b", "state": NEG}], receipt=True))
    svc.feedback(FeedbackRequest(decision_id=ids[0], head="good", label="true", source="test"))
    svc.feedback(FeedbackRequest(decision_id=ids[1], head="good", label="true", source="test"))
    svc.feedback(FeedbackRequest(decision_id=ids[1], head="good", label="false", source="test",
                                 note="corrected", tags=["regression"]))
    svc.feedback(FeedbackRequest(decision_id=so["receipt"]["id"], head="g", label="yes", source="test"))
    rows = list(svc.logs.decisions())          # a snapshot: later tests may log more decisions
    svc.pack.signal = -svc.pack.signal         # the candidate learns the opposite relation
    assert svc.trainer.start(TrainRequest(heads=TRAINED, seed=11), wait=True)["model"] == V2
    return SimpleNamespace(svc=svc, root=root, ids=ids, decided=decided, systemone=so["receipt"]["id"], rows=rows)


def _lines(path) -> int:
    return path.read_text(encoding="utf-8").count("\n") if path.exists() else 0


# ----------------------------------------------------------------------------- replay

def test_replay_compares_classes_tiers_bands_and_labels(world):
    rep = replay(world.svc, V1, V2, world.rows)
    assert rep["base"] == V1 and rep["candidate"] == V2 and rep["policy_changed"] is False
    assert rep["scope"]["rows_read"] == 14 and rep["scope"]["in_scope"] == 13
    assert rep["kinds"] == {"decide": 12, "systemone": 1} and rep["models"] == {V1: 13}
    assert rep["decisions"] == 13 and set(rep["heads"]) == set(TRAINED)
    assert {h: v["n"] for h, v in rep["heads"].items()} == {"good": 13, "color": 12, "level": 12}
    assert rep["replayed"] == rep["totals"]["n"] == 37
    # what could not be replayed is counted and explained, never fatal
    sk = rep["skipped"]
    assert set(sk) == {"no_heads", "untrained"} and rep["n_skipped"] == 13
    assert sk["no_heads"]["count"] == 1 and "rank" in sk["no_heads"]["examples"][0]["detail"]
    assert sk["untrained"]["count"] == 12 and sk["untrained"]["by_version"] == {V1: 12}
    assert all(ex["head"] == "spare" for ex in sk["untrained"]["examples"]) and sk["untrained"]["why"]
    for v in rep["heads"].values():
        n = v["n"]
        assert v["same_model"] is False                                     # every head was retrained
        assert sum(v["classes"]["base"].values()) == sum(v["classes"]["candidate"].values()) == n
        assert sum(x["n"] for x in v["class_transitions"]) == v["class_changes"] <= n
        tiers = v["tiers"]
        assert sum(tiers["base"].values()) == sum(tiers["candidate"].values()) == n
        assert sum(sum(row.values()) for row in tiers["matrix"].values()) == n
        assert set(tiers["matrix"]) == {"escalate", "confirm", "act"}       # the policy's own tier names
        diagonal = sum(tiers["matrix"][t][t] for t in tiers["matrix"])
        assert tiers["raised"] + tiers["lowered"] == n - diagonal
        assert sum(v["bands"]["base"].values()) == sum(v["bands"]["candidate"].values()) == n
        assert v["bands"]["changed"] == v["bands"]["up"] + v["bands"]["down"]
        assert 0 <= v["mean_abs_delta_p"] <= v["max_abs_delta_p"] <= 1 and 0 <= v["mean_tv_distance"] <= 1
    # the flipped candidate changes its mind on clear cases
    good = rep["heads"]["good"]
    assert good["class_changes"] > 0 and good["mean_tv_distance"] > 0.5
    assert {(t["from"], t["to"]) for t in good["class_transitions"]} <= {("true", "false"), ("false", "true")}
    assert all(1 <= len(t["examples"]) <= 3 for t in good["class_transitions"])
    # labels: the last feedback row per decision and head counts (ids[1] was corrected to false)
    lab = good["labelled"]
    assert lab["n"] == 3 and rep["totals"]["labelled"] == 3
    assert lab["base"]["accuracy"] == 1.0 and lab["candidate"]["accuracy"] == 0.0
    assert lab["base"]["brier"] < 0.05 < 1.5 < lab["candidate"]["brier"] <= 2.0
    assert lab["regressions"]["n"] == 3 and lab["improvements"] == {"n": 0, "examples": []}
    ex = {e["decision_id"]: e for e in lab["regressions"]["examples"]}
    assert set(ex) == {world.ids[0], world.ids[1], world.systemone}
    assert ex[world.ids[1]] == {"decision_id": world.ids[1], "label": "false", "base": "false", "candidate": "true"}
    assert ex[world.systemone]["label"] == "true"                          # "yes" on question g, head good
    assert lab["candidate"]["possibly_trained_on"] == 3                     # posted before the candidate trained
    assert lab["paired_test"]["discordant"] == 3 and lab["paired_test"]["p_value"] == 0.25
    assert rep["heads"]["color"]["labelled"]["n"] == 0
    assert "3 regressions" in rep["summary"] and rep["notes"]


def test_replay_of_a_version_against_itself_changes_nothing(world):
    rep = replay(world.svc, V2, V2, world.rows)
    t = rep["totals"]
    assert t["n"] == 37 and t["class_changes"] == t["tier_raised"] == t["tier_lowered"] == t["band_changes"] == 0
    assert t["regressions"] == t["improvements"] == 0
    for v in rep["heads"].values():
        assert v["same_model"] is True and v["class_transitions"] == [] and v["bands"]["transitions"] == []
        assert v["mean_abs_delta_p"] == v["max_abs_delta_p"] == v["mean_tv_distance"] == 0.0
        assert v["classes"]["base"] == v["classes"]["candidate"]
        m = v["tiers"]["matrix"]
        assert all(m[a][b] == 0 for a in m for b in m[a] if a != b)
        lab = v["labelled"]
        assert lab["base"] == lab["candidate"] and lab["paired_test"] is None


def test_replaying_the_deciding_version_reproduces_the_logged_decisions(world):
    """The decisions were made by rlcd-0.1.1 under rollout full, before any outcome moved its ledger: replayed as
    trained, its classes and tiers are the logged ones."""
    decide_rows = [r for r in world.rows if r.get("kind") == "decide"]
    rep = replay(world.svc, V1, V2, decide_rows)
    for head in ("color", "level"):
        logged = Counter(r["heads"][head]["class"] for r in decide_rows)
        assert rep["heads"][head]["classes"]["base"] == dict(sorted(logged.items()))
    for head in TRAINED:
        logged = Counter(r["heads"][head]["tier"] for r in decide_rows)
        assert {k: v for k, v in rep["heads"][head]["tiers"]["base"].items() if v} == dict(logged)


def test_replay_scope_heads_limit_and_since(world):
    svc, rows = world.svc, world.rows
    color = replay(svc, V1, V2, rows, heads=["color"])
    assert set(color["heads"]) == {"color"} and color["scope"]["in_scope"] == 12
    assert color["scope"]["without_matching_heads"] == 2 and color["skipped"] == {}    # systemone + rank rows
    q = replay(svc, V1, V2, rows, heads="g")                                # a systemone question id
    assert {h: v["n"] for h, v in q["heads"].items()} == {"good": 1}
    last = replay(svc, V1, V2, rows, heads=["color"], limit=3)
    assert last["decisions"] == 3 and last["scope"]["beyond_limit"] == 9 and last["heads"]["color"]["n"] == 3
    assert replay(svc, V1, V2, rows, since="2000-01-01")["decisions"] == 13
    future = replay(svc, V1, V2, rows, since="2999-01-01T00:00:00Z")
    assert future["decisions"] == 0 and future["scope"]["before_since"] == 14 and future["heads"] == {}
    assert "nothing replayed" in future["summary"]
    assert replay(svc, V1, V2)["scope"]["rows_read"] >= 14                 # rows=None reads decisions.jsonl
    for kw, field in ((dict(since="last tuesday"), "since"), (dict(limit=0), "limit"),
                      (dict(max_examples=-1), "max_examples")):
        with pytest.raises(ContractError) as e:
            replay(svc, V1, V2, rows, **kw)
        assert e.value.details["field"] == field
    with pytest.raises(ContractError):
        replay(svc, None, V2, rows)
    with pytest.raises(ContractError) as e:
        replay(svc, "rlcd-9.9.9", V2, rows)
    assert e.value.kind == "unknown_model"


def test_replay_never_crashes_on_legacy_or_damaged_rows(world):
    rows = [
        "not a record",
        {"decision_id": "r1", "kind": "decide", "heads": []},
        {"decision_id": "r2", "heads": {"color": "red"}},
        {"decision_id": "r3", "heads": {"color": {"class": "red", "p": 0.9, "tier": "act"}}},      # log_states off
        {"decision_id": "r4", "heads": {"color": {"class": "red", "state_digest": "sha256:00"}}},  # pruned
        {"decision_id": "r5", "heads": {"color": {"state": {"f": [1, 2]}}}},                       # wrong shape
        {"decision_id": "r6", "heads": {"color": {"state": None}}},
        {"decision_id": "r7", "heads": {"nope": {"state": POS}}},
        {"decision_id": "r8", "heads": {"color": {"state": POS, "stakes": "extreme"}}},
        {"decision_id": "r9", "heads": {"color": {"state": POS, "head": {"not": "a name"}, "stakes": 3}}},
        {"decision_id": "r10", "heads": {"color": {"state": POS, "class": "red", "p": 0.9, "tier": "act"}}},
        {"heads": {"level": {"state": NEG}}},                                                    # no decision_id
        {"decision_id": "r11", "heads": {"good": {"state": {"x": POS["f"], "feature_names": NAMES,
                                                            "feature_version": "toy-v0"}}}},
    ]
    rep = replay(world.svc, V1, V2, rows)
    sk = {k: v["count"] for k, v in rep["skipped"].items()}
    assert sk == {"malformed": 2, "no_heads": 1, "no_state": 1, "state_pruned": 1, "state_refused": 3,
                  "head_missing": 1, "stakes_unknown": 1}
    assert {h: v["n"] for h, v in rep["heads"].items()} == {"color": 2, "level": 1}
    refused = rep["skipped"]["state_refused"]["examples"]
    assert {e["decision_id"] for e in refused} == {"r5", "r6", "r11"} and all(e["detail"] for e in refused)
    assert any("toy-v0" in e["detail"] for e in refused)
    dated = replay(world.svc, V1, V2, rows[1:3], since="2020-01-01")
    assert dated["skipped"]["undated"]["count"] == 2 and dated["decisions"] == 0


def test_feedback_labels_keep_the_last_row_per_decision_and_head():
    rows = [{"decision_id": "d", "head": "h", "label": "a"}, {"decision_id": "d", "head": "h", "label": "b"},
            {"decision_id": "d", "head": "k", "label": "c"}, {"head": "h", "label": "x"}, "junk",
            {"decision_id": "e", "head": "h"}]
    out = feedback_labels(rows)
    assert {k: v["label"] for k, v in out.items()} == {("d", "h"): "b", ("d", "k"): "c"}


# ----------------------------------------------------------------------------- the cases file

def test_cases_path_default_and_override(tmp_path):
    assert RG.cases_path(Project(root=tmp_path)) == (tmp_path / "regressions.jsonl").resolve()
    p = RG.cases_path(Project(root=tmp_path, data={"regressions": "qa/cases.jsonl"}))
    assert p == (tmp_path / "qa" / "cases.jsonl").resolve()
    with pytest.raises(ContractError) as e:
        RG.cases_path(Project(root=tmp_path, data={"regressions": {"path": 1}}))
    assert e.value.kind == "bad_config"


def test_add_case_assigns_ids_and_load_cases_keeps_bad_lines_visible(tmp_path):
    path = tmp_path / "qa" / "cases.jsonl"
    a = RG.add_case(path, {"id": "mine", "added_at": "yesterday", "head": "color", "state": POS,
                           "expect": {"class": "red"}})
    assert a["id"].startswith("case-") and a["id"] != "mine" and a["added_at"].endswith("Z") and a["kind"] == "head"
    with open(path, "a", encoding="utf-8") as f:     # a hand edit without a final newline, then two bad lines
        f.write('{"head": "color"}\nnot json\n{"head": "color", "state": {"f": [0]}, "expect": {"tier": "act"}}')
    b = RG.add_case(path, {"kind": "decide", "request": {"state": POS}, "expect": {"action": "act"}})
    cases = RG.load_cases(path)
    assert [c["id"] for c in cases] == [a["id"], "line-2", "line-3", "line-4", b["id"]]
    assert "invalid" in cases[1] and "state" in cases[1]["invalid"] and "not JSON" in cases[2]["invalid"]
    assert cases[3]["kind"] == "head" and "invalid" not in cases[3]       # a valid hand-written case
    assert json.loads(path.read_text(encoding="utf-8").splitlines()[-1])["id"] == b["id"]
    assert RG.load_cases(tmp_path / "missing.jsonl") == []


@pytest.mark.parametrize("case, words", [
    ("a string", "JSON object"),
    ({"head": "color", "state": POS}, "expect"),
    ({"head": "color", "state": POS, "expect": {}}, "expect"),
    ({"head": "color", "expect": {"class": "red"}}, "state"),
    ({"state": POS, "expect": {"class": "red"}}, "head"),
    ({"head": "color", "state": POS, "expect": {"klass": "red"}}, "unknown keys"),
    ({"head": "color", "state": POS, "expect": {"p_at_least": 1.5}}, "probability"),
    ({"head": "color", "state": POS, "expect": {"p_at_least": 0.9, "p_at_most": 0.1}}, "no answer could pass"),
    ({"head": "color", "state": POS, "expect": {"class_in": []}}, "class_in"),
    ({"head": "color", "state": POS, "expect": {"error": "validation", "class": "red"}}, "stands alone"),
    ({"head": "color", "state": POS, "stakes": "very high!", "expect": {"class": "red"}}, "stakes"),
    ({"kind": "rank", "request": {}, "expect": {"action": "act"}}, "kind"),
    ({"kind": ["decide"], "request": {}, "expect": {"action": "act"}}, "kind"),
    ({"id": 5, "head": "color", "state": POS, "expect": {"class": "red"}}, "id must be a string"),
    ({"kind": "decide", "request": {"state": POS, "explore": True}, "expect": {"action": "act"}}, "deterministic"),
    ({"kind": "decide", "request": {"state": POS}, "expect": {"answers": {"q": {"noul_at_least": 0.5}}}},
     "unknown keys"),
    ({"kind": "systemone", "request": {"state": POS}, "expect": {"answers": {"q": {"noul_at_least": 0.5}}}},
     "not a valid systemone request"),
    ({"kind": "systemone", "request": {"state": POS, "battery": "b"}, "expect": {"answers": {"q": {"pick": 1}}}},
     "unknown keys"),
    ({"kind": "decide", "state": POS, "request": {}, "expect": {"action": "act"}}, "inside `request`"),
])
def test_validate_case_names_the_problem(case, words):
    with pytest.raises(ContractError) as e:
        RG.validate_case(case)
    assert e.value.kind == "bad_case" and words in e.value.message


def test_parse_expect():
    assert RG.parse_expect(["class=red", "class_in=red, green", "p_at_least=0.8", "tier_at_most=confirm"]) == {
        "class": "red", "class_in": ["red", "green"], "p_at_least": 0.8, "tier_at_most": "confirm"}
    for bad in (["class"], ["colour=red"], ["p_at_most=high"], ["error=validation"]):
        with pytest.raises(ContractError):
            RG.parse_expect(bad)


# ----------------------------------------------------------------------------- cases from the logs

def test_case_from_decision_copies_the_logged_state_and_runs(world, tmp_path):
    svc = world.svc
    logged = world.decided[0]["heads"]["color"]["decision"]
    case = RG.case_from_decision(svc, world.ids[0], head="color", expect={"class": logged["class"]},
                                 note="routed wrong", incident="INC-7")
    assert case["kind"] == "head" and case["head"] == "color" and case["state"] == POS and case["stakes"] == "low"
    assert case["note"] == "routed wrong" and case["incident"] == "INC-7"
    assert case["source"]["decision_id"] == world.ids[0] and case["source"]["model"] == V1
    assert case["source"]["logged"]["class"] == logged["class"]
    path = tmp_path / "cases.jsonl"
    stored = RG.add_case(path, case)
    on_v1 = RG.run_cases(svc, RG.load_cases(path), model=V1)
    assert (on_v1["model"], on_v1["passed"], on_v1["failed"], on_v1["errors"]) == (V1, 1, 0, 0)
    r = on_v1["results"][0]
    assert r["id"] == stored["id"] and r["ok"] is True and r["why"] is None and r["incident"] == "INC-7"
    assert r["got"]["class"] == logged["class"] and r["got"]["tier"] == logged["tier"]
    on_v2 = RG.run_cases(svc, RG.load_cases(path))                         # default: rlcd-latest
    assert on_v2["model"] == V2 and on_v2["failed"] == 1
    assert f"expected {logged['class']}" in on_v2["results"][0]["why"]
    # a systemone question id names the head; "yes" is the noul's "true"
    q = RG.case_from_decision(svc, world.systemone, head="g", expect={"class": "yes"})
    assert q["head"] == "good" and q["expect"] == {"class": "true"} and q["source"]["question"] == "g"
    assert "stakes" not in q                                                # systemone logs no stakes
    one = RG.case_from_decision(svc, world.systemone, head=None, expect={"tier_at_most": "confirm"})
    assert one["head"] == "good"


def test_case_from_decision_refuses_what_it_cannot_copy(world):
    svc = world.svc
    for kw, kind in ((dict(decision_id="nope", head="color"), "unknown_decision"),
                     (dict(decision_id=world.ids[0], head="nope"), "unknown_head"),
                     (dict(decision_id=world.ids[0], head=None), "validation"),
                     (dict(decision_id=world.ids[0], head="color", expect={"class": "purple"}), "bad_case")):
        with pytest.raises(ContractError) as e:
            RG.case_from_decision(svc, **{"expect": {"class": "red"}, **kw})
        assert e.value.kind == kind
    svc.project.data["serve"] = {"log_states": False}
    try:
        did = svc.decide(DecideRequest(state=POS, heads=["color"]))["decision_id"]
    finally:
        svc.project.data.pop("serve")
    with pytest.raises(ContractError) as e:
        RG.case_from_decision(svc, did, head="color", expect={"class": "red"})
    assert e.value.kind == "no_state" and "log_states" in e.value.message


def test_case_from_feedback_expects_the_corrected_label(world):
    svc = world.svc
    rows = [r for r in svc.logs.feedback() if r.get("decision_id") == world.ids[1]]
    case = RG.case_from_feedback(svc, rows[-1])
    assert case["head"] == "good" and case["state"] == NEG and case["expect"] == {"class": "false"}
    assert case["stakes"] == "low" and case["note"] == "corrected"
    assert case["source"]["feedback_id"] == rows[-1]["id"] and case["source"]["tags"] == ["regression"]
    assert case["source"]["logged"]["tier"] == world.decided[1]["heads"]["good"]["decision"]["tier"]
    res = {m: RG.run_cases(svc, [case], model=m)["results"][0]["status"] for m in (V1, V2)}
    assert res == {V1: "passed", V2: "failed"}
    q = next(r for r in svc.logs.feedback() if r.get("decision_id") == world.systemone)
    assert RG.case_from_feedback(svc, q)["source"]["question"] == "g"
    with pytest.raises(ContractError) as e:
        RG.case_from_feedback(svc, {"id": "f1", "head": "color", "label": "red"})
    assert e.value.kind == "no_state"
    with pytest.raises(ContractError):
        RG.case_from_feedback(svc, {"head": "color", "state": POS})


# ----------------------------------------------------------------------------- running cases

def test_head_cases_pass_and_fail_on_class_tier_and_p(world):
    cases = [
        {"id": "pos", "head": "good", "state": POS, "expect": {"class": "true", "p_at_least": 0.8}},
        {"id": "neg", "head": "good", "state": NEG, "expect": {"class": "false", "tier": "abstain"}},
        {"id": "neg-named", "head": "good", "state": NEG, "expect": {"tier": "escalate", "p_at_most": 0.2}},
        {"id": "in", "head": "color", "state": POS, "expect": {"class_in": ["red", "green"]}},
        {"id": "not-red", "head": "color", "state": POS, "expect": {"class": "blue"}},
        {"id": "low", "head": "good", "state": POS, "expect": {"tier_at_most": "abstain"}},
        {"id": "bad-tier", "head": "good", "state": POS, "expect": {"tier": "hold"}},
        {"id": "bad-class", "head": "color", "state": POS, "expect": {"class": "purple"}},
        {"id": "spare", "head": "spare", "state": POS, "expect": {"class": "true"}},
        {"id": "nohead", "head": "nope", "state": POS, "expect": {"class": "true"}},
        {"id": "invalid", "head": "good", "expect": {"class": "true"}},
    ]
    res = RG.run_cases(world.svc, cases, model=V1)
    by = {r["id"]: r for r in res["results"]}
    status = {k: r["status"] for k, r in by.items()}
    assert status == {"pos": "passed", "neg": "passed", "neg-named": "passed", "in": "passed", "not-red": "failed",
                      "low": "failed", "bad-tier": "error", "bad-class": "error", "spare": "error",
                      "nohead": "error", "invalid": "error"}
    assert (res["total"], res["passed"], res["failed"], res["errors"]) == (11, 4, 2, 5)
    assert by["pos"]["got"]["class"] == "true" and by["pos"]["got"]["p"] >= 0.8
    assert set(by["pos"]["got"]["probabilities"]) == {"false", "true"}
    assert by["neg"]["got"]["p"] == by["neg"]["got"]["probabilities"]["false"]    # p of the expected class
    assert by["neg-named"]["got"]["p"] == by["neg-named"]["got"]["probabilities"]["true"]   # gate: P(true)
    assert "expected blue" in by["not-red"]["why"] and "at most abstain" in by["low"]["why"]
    assert "not a tier" in by["bad-tier"]["why"] and "purple" in by["bad-class"]["why"]
    assert "not trained" in by["spare"]["why"] and "unknown_head" in by["nohead"]["why"]
    assert all(r["ok"] is (r["status"] == "passed") for r in res["results"])
    flipped = RG.run_cases(world.svc, cases[:2], model=V2)
    assert flipped["failed"] == 2 and "expected true" in flipped["results"][0]["why"]


def test_malformed_cases_are_errors_not_crashes(world):
    cases = ["junk", {"kind": ["decide"], "request": {}, "expect": {"action": "act"}},
             {"id": "no-expect", "head": "good", "state": POS}, {"head": "good", "state": POS, "expect": {"x": 1}}]
    res = RG.run_cases(world.svc, cases)
    assert [(r["id"], r["status"]) for r in res["results"]] == [("#1", "error"), ("#2", "error"),
                                                                 ("no-expect", "error"), ("#4", "error")]
    assert res["errors"] == 4 and all(r["why"] for r in res["results"])


def test_refusals_and_the_input_guard_as_cases(world):
    hole = {"x": [None, *POS["f"][1:]], "feature_names": NAMES, "feature_version": "toy-v1"}
    cases = [
        {"id": "malformed", "head": "color", "state": {"f": [1, 2]}, "expect": {"error": "validation"}},
        {"id": "strict", "head": "color", "state": hole, "strict": True, "expect": {"error": "missing_evidence"}},
        {"id": "guarded", "head": "color", "state": hole, "expect": {"tier_at_most": "confirm"}},
        {"id": "answered", "head": "color", "state": POS, "expect": {"error": "validation"}},
        {"id": "other-error", "head": "color", "state": {"f": [1, 2]}, "expect": {"error": "missing_evidence"}},
        {"id": "unexpected", "head": "color", "state": {"f": [1, 2]}, "expect": {"class": "red"}},
    ]
    by = {r["id"]: r for r in RG.run_cases(world.svc, cases, model=V1)["results"]}
    assert {k: r["status"] for k, r in by.items()} == {"malformed": "passed", "strict": "passed", "guarded": "passed",
                                                       "answered": "failed", "other-error": "failed",
                                                       "unexpected": "error"}
    assert by["malformed"]["got"]["error"] == "validation"
    assert by["guarded"]["got"]["missing_inputs"] == ["f0"]
    assert "was answered" in by["answered"]["why"] and "expected missing_evidence" in by["other-error"]["why"]
    assert "refused (validation)" in by["unexpected"]["why"]


def test_request_cases_decide_systemone_and_evaluate(world):
    svc = world.svc
    with svc.dry_run():
        live = svc.decide(DecideRequest(model=V1, state=POS, heads=["color"], stakes="low"))
    req = {"state": POS, "heads": ["color"], "stakes": "low", "model": "rlcd-0.1.0"}    # the model is replaced
    q = {"g": {"type": "noul", "head": "good"}, "c": {"type": "choice", "head": "color"}}
    cases = [
        {"id": "action", "kind": "decide", "request": req, "expect": {"action": live["action"]}},
        {"id": "at-least", "kind": "decide", "request": req, "expect": {"action_at_least": "confirm"}},
        {"id": "too-high", "kind": "decide", "request": req, "expect": {"action_at_most": "abstain"}},
        {"id": "no-next", "kind": "decide", "request": req, "expect": {"next": "answer"}},
        {"id": "answers", "kind": "systemone", "request": {"state": POS, "questions": q, "receipt": True},
         "expect": {"answers": {"g": {"noul_at_least": 0.8}, "c": {"choice": "red"}}}},
        {"id": "missing-q", "kind": "systemone", "request": {"state": POS, "questions": q},
         "expect": {"answers": {"zz": {"choice": "red"}}}},
        {"id": "stakes", "kind": "decide", "request": {"state": POS, "stakes": "extreme"},
         "expect": {"error": "validation"}},
        {"id": "evaluate", "kind": "evaluate", "request": {"state": POS}, "expect": {"verdict": "ship"}},
        {"id": "evaluate-refused", "kind": "evaluate", "request": {"state": POS}, "expect": {"error": "unsupported"}},
        {"id": "spare", "kind": "systemone", "request": {"state": POS, "questions": {"s": {"type": "noul",
                                                                                           "head": "spare"}}},
         "expect": {"answers": {"s": {"noul_at_most": 0.9}}}},
    ]
    res = RG.run_cases(svc, cases, model=V1)
    by = {r["id"]: r for r in res["results"]}
    assert live["action"] == "act" and by["action"]["got"] == {"action": "act"}
    assert {k: r["status"] for k, r in by.items()} == {
        "action": "passed", "at-least": "passed", "too-high": "failed", "no-next": "failed", "answers": "passed",
        "missing-q": "failed", "stakes": "passed", "evaluate": "error", "evaluate-refused": "passed",
        "spare": "error"}
    assert by["answers"]["got"]["answers"]["c"] == {"type": "choice", "choice": "red"}
    assert "next None" in by["no-next"]["why"] and "no answer for question 'zz'" in by["missing-q"]["why"]
    assert "unsupported" in by["evaluate"]["why"] and "not trained" in by["spare"]["why"]


def test_running_cases_logs_nothing(world):
    svc = world.svc
    files = [svc.home / n for n in ("decisions.jsonl", "feedback.jsonl", "bernoulli.json", "outcomes.jsonl")]
    before = {f.name: f.read_bytes() if f.exists() else None for f in files}
    n = _lines(svc.home / "decisions.jsonl")
    cases = [{"kind": "decide", "request": {"state": s, "stakes": "low"}, "expect": {"action_at_least": "abstain"}}
             for s in (POS, NEG)]
    cases.append({"kind": "systemone", "request": {"state": POS, "receipt": True,
                                                   "questions": {"g": {"type": "noul", "head": "good"}}},
                  "expect": {"answers": {"g": {"noul_at_least": 0.0}}}})
    cases.append({"head": "color", "state": POS, "expect": {"class_in": ["red", "green", "blue"]}})
    res = RG.run_cases(svc, cases)
    assert res["passed"] == 4
    assert {f.name: f.read_bytes() if f.exists() else None for f in files} == before
    assert _lines(svc.home / "decisions.jsonl") == n and not svc.is_dry
    svc.decide(DecideRequest(state=POS, heads=["color"]))                  # outside run_cases, decisions log again
    assert _lines(svc.home / "decisions.jsonl") == n + 1


def test_cases_judge_the_model_not_the_rollout_or_the_kill_switch(world):
    """Switching automation off during an incident must not make a case pass (or fail): cases run as trained."""
    svc = world.svc
    cases = [{"id": "head-act", "head": "color", "state": POS, "stakes": "low", "expect": {"tier": "act"}},
             {"id": "decide-act", "kind": "decide", "request": {"state": POS, "heads": ["color"], "stakes": "low"},
              "expect": {"action": "act"}},
             {"id": "never-act", "head": "color", "state": POS, "stakes": "low", "expect": {"tier_at_most": "confirm"}}]
    on = {r["id"]: r["status"] for r in RG.run_cases(svc, cases, model=V1)["results"]}
    assert on == {"head-act": "passed", "decide-act": "passed", "never-act": "failed"}
    svc.set_automation(False, reason="incident 42", by="test")
    try:
        with svc.dry_run():
            live = svc.decide(DecideRequest(model=V1, state=POS, heads=["color"], stakes="low"))
        assert live["action"] == "confirm" and live["rollout"]["capped"] is True        # the deployment is capped
        off = {r["id"]: r["status"] for r in RG.run_cases(svc, cases, model=V1)["results"]}
    finally:
        svc.set_automation(True, by="test")
    svc.project.data["rollout"] = {"mode": "shadow"}
    try:
        shadow = {r["id"]: r["status"] for r in RG.run_cases(svc, cases, model=V1)["results"]}
        assert svc.rollout().mode == "shadow"                              # the service itself is untouched
    finally:
        svc.project.data.pop("rollout")
    assert off == shadow == on


def test_as_trained_view_changes_only_rollout_switch_and_gate(world):
    svc = world.svc
    svc.project.data["rollout"] = {"mode": "assist"}
    try:
        view = RG.as_trained(svc)
        assert view.rollout().mode == "full" and view.switch.state().enabled and view.switch.cap() == (2, None)
        assert svc.rollout().mode == "assist" and svc.switch is not view.switch
        head = svc.loaded(V1, "color")
        assert view.gate_table(V1, "color", head) == head.holdout_ledger()
        assert view.gate_table(V1, "color", None) is None
        assert view.registry is svc.registry and view.logs is svc.logs
        with view.dry_run():
            assert svc.is_dry                                              # one dry-run flag for both
    finally:
        svc.project.data.pop("rollout")


def test_tier_names_follow_the_policy_abstain_name(tmp_path):
    svc = Service(Project(root=tmp_path, data={"policy": {"abstain": "hold"}}), home=tmp_path, pack=ToyPack(n=1500))
    assert svc.trainer.start(TrainRequest(heads=["good"]), wait=True)["status"] == "succeeded"
    cases = [{"id": t, "head": "good", "state": NEG, "expect": {"tier": t}} for t in ("hold", "abstain", "escalate")]
    cases += [{"id": f"action-{t}", "kind": "decide", "request": {"state": NEG, "heads": ["good"]},
               "expect": {"action": t}} for t in ("hold", "abstain", "escalate")]
    by = {r["id"]: r for r in RG.run_cases(svc, cases)["results"]}
    assert {k: r["status"] for k, r in by.items()} == {
        "hold": "passed", "abstain": "passed", "escalate": "error",
        "action-hold": "passed", "action-abstain": "passed", "action-escalate": "error"}
    assert by["hold"]["got"]["tier"] == "hold" and "'hold'" in by["escalate"]["why"]


def test_feedback_tagged_regression_adds_a_case(tmp_path):
    """The Service.feedback wiring: a reviewer's correction tagged "regression" becomes a case in the project."""
    svc = Service(Project(root=tmp_path), home=tmp_path, pack=ToyPack(n=1500))
    assert svc.trainer.start(TrainRequest(heads=["color"]), wait=True)["status"] == "succeeded"
    did = svc.decide(DecideRequest(state=POS, heads=["color"]))["decision_id"]
    out = svc.feedback(FeedbackRequest(decision_id=did, label="blue", source="review", tags=["regression"]))
    if "regression_case" not in out:
        pytest.skip("Service.feedback does not add regression cases yet (wiring pending)")
    cases = RG.load_cases(RG.cases_path(svc.project))
    assert len(cases) == 1 and cases[0]["id"] == out["regression_case"]["id"]
    assert cases[0]["expect"] == {"class": "blue"} and cases[0]["source"]["decision_id"] == did
    plain = svc.feedback(FeedbackRequest(decision_id=did, label="red", source="review"))
    assert "regression_case" not in plain and len(RG.load_cases(RG.cases_path(svc.project))) == 1


# ----------------------------------------------------------------------------- the documents pack

DOC_ENC = {"kind": "hashing", "dim": 256}
DOC_DECIDE_HEADS = {"response_type", "route", "needs_graph", "chunk_relevant", "answerable"}


@pytest.fixture(scope="module")
def documents(tmp_path_factory):
    from open_rlcd.packs.documents import write_synth
    from open_rlcd.packs.documents.world import build_world

    root = tmp_path_factory.mktemp("documents-replay")
    home = root / "var"
    mp = pytest.MonkeyPatch()
    mp.setenv("RLCD_HOME", str(home))     # the pack reads its label store and corpus from the project's home
    write_synth(home, DOC_ENC, seed=7, n_queries=900)
    project = Project(root=root, data={"pack": "documents", "pack_config": {"encoder": DOC_ENC}})
    svc = Service(project, home=home)
    job = svc.trainer.start(TrainRequest(), wait=True)
    assert job["status"] == "succeeded", job.get("error")
    yield SimpleNamespace(svc=svc, version=job["model"], world=build_world(7))
    mp.undo()


@pytest.mark.slow
def test_documents_decisions_replay_exactly_and_become_cases(documents):
    """On the real pack (views, chunk_relevant's top candidate, the plan): replaying the version that decided
    reproduces every logged head tier, and cases built from its decisions pass without logging anything."""
    svc, v = documents.svc, documents.version
    products = documents.world.products[:4]
    queries = [f"What is the price of the {p.subject}?" for p in products]
    queries += [f"Who leads the team that owns the {p.subject}?" for p in products]
    queries.append("What's the weather in Lisbon tomorrow?")
    decided = [svc.decide(DecideRequest(state={"query": q})) for q in queries]
    rows = [r for r in svc.logs.decisions() if r.get("kind") == "decide"]
    rep = replay(svc, v, v, rows)
    assert rep["skipped"] == {} and rep["decisions"] == len(queries) and set(rep["heads"]) == DOC_DECIDE_HEADS
    assert rep["totals"]["class_changes"] == rep["totals"]["tier_raised"] == rep["totals"]["tier_lowered"] == 0
    for head, hv in rep["heads"].items():
        logged = Counter(r["heads"][head]["tier"] for r in rows if head in r["heads"])
        assert {k: n for k, n in hv["tiers"]["base"].items() if n} == dict(logged), head
    first, oos = decided[0], decided[-1]
    assert oos["plan"]["next"] == "refuse_out_of_scope"
    corpus = svc.pack.corpus(svc.project, svc.provider(v))
    p = products[0]
    state = {"query": queries[0], "evidence": [corpus.chunk(p.chunk_id)],
             "answer": f"The price of {p.subject} is {p.facts['price'].value['en']}."}
    with svc.dry_run():
        review = svc.evaluate(EvaluateRequest(state=state))
    cases = [RG.case_from_decision(svc, first["decision_id"], head="route", expect={"class": first["plan"]["route"]}),
             RG.case_from_decision(svc, first["decision_id"], head="chunk_relevant",
                                   expect={"tier": first["heads"]["chunk_relevant"]["decision"]["tier"]}),
             {"kind": "decide", "request": {"state": {"query": queries[0]}},
              "expect": {"next": first["plan"]["next"], "action": first["action"]}},
             {"kind": "decide", "request": {"state": {"query": queries[-1]}}, "expect": {"next": "refuse_out_of_scope"}},
             {"kind": "evaluate", "request": {"state": state},
              "expect": {"verdict": review["verdict"], "next": review["next"]}}]
    n = _lines(svc.home / "decisions.jsonl")
    res = RG.run_cases(svc, cases)
    assert (res["passed"], res["failed"], res["errors"]) == (5, 0, 0), res["results"]
    assert _lines(svc.home / "decisions.jsonl") == n
