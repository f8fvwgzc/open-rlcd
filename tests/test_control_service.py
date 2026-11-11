"""The control layer as wired into the service and the HTTP API, with the toy pack (tests/toypack.py).

What: batteries in /v1/systemone (pinned versions, unknown names, the ref in the response and the receipt), question
inputs and paths (fail closed on a field that is not there), `allowed` options on a choice question and on
/v1/decide, receipts (the hash chain, lookup, the staleness check against state, alias, kill switch and age),
rollout modes, the kill switch with its environment lock and the admin key, promote / rollback and the Bernoulli
tables that follow them, feedback on /v1/systemone receipts, redaction of logged states, /v1/outcome, retryable
error bodies, /health, dry runs and `serve.receipts: all`.

Why: every one of these either takes automation away or records what was decided. A regression would let a decision
act where the operator said it must not, or let a receipt approve a state it never judged, and nothing else in the
suite would notice. The toy pack trains in well under a second, so a test that changes shared state (the switch, the
registry, the ledger, the logs) gets its own RLCD_HOME; read-only tests share one trained service.

What it does not cover: the documents pack's plan and evaluate decision table (tests/test_control_documents.py)
and each core module's unit-level behaviour. Where a trained head's exact numbers matter, the test finds a state that
gives the decision it needs instead of pinning a probability.
"""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from fastapi.testclient import TestClient

from open_rlcd.core import uncertainty as U
from open_rlcd.core.config import Project
from open_rlcd.core.contract import (
    ContractError,
    DecideRequest,
    FeedbackRequest,
    RankRequest,
    SystemOneRequest,
    TrainRequest,
)
from open_rlcd.core.digest import digest
from open_rlcd.core.service import Service
from open_rlcd.serve.api import create_app
from tests.toypack import NAMES, ToyPack


def _direction(k: int, cls: int) -> np.ndarray:
    """The unit direction along which the toy generator makes class `cls` of a k-class head likely (its weights are
    fixed: numpy seed 99 + k)."""
    w = np.random.default_rng(99 + k).normal(size=(len(NAMES), k))
    v = w[:, cls] - w[:, [j for j in range(k) if j != cls]].mean(axis=1)
    return v / np.linalg.norm(v)


def _state(direction: np.ndarray, t: float) -> dict:
    return {"f": (t * direction).round(3).tolist()}


GOOD_DIR, RED_DIR, BLUE_DIR = _direction(2, 1), _direction(3, 0), _direction(3, 2)
POS, NEG, MID = _state(GOOD_DIR, 2.5), _state(GOOD_DIR, -2.5), {"f": [0.0] * 6}
RED = _state(RED_DIR, 3.0)         # the color head: red with p close to 1, act at every stakes level
BLUE = _state(BLUE_DIR, 6.0)       # blue with p close to 1: red and green round to 0 in an answer
SECRET = "deploy with AKIAIOSFODNN7EXAMPLE, password=hunter2hunter2"
BATTERY = {"version": "2", "owner": "ops", "description": "intake questions on one state",
           "changelog": ["1: ok", "2: col and lv added"],
           "consumers": {"ok": "the gate before any action", "col": "which queue gets the item"},
           "questions": {"ok": {"type": "noul", "head": "good"}, "col": {"type": "choice", "head": "color"},
                         "lv": {"type": "score", "head": "level"}}}


def service(home: Path, data: dict | None = None, pack: ToyPack | None = None) -> Service:
    """A toy service on `home` with `data` as its rlcd.yaml (several services may share one trained home)."""
    return Service(Project(root=home, data=dict(data or {})), home=home, pack=pack or ToyPack())


def train(svc: Service, heads: list[str] | None = None) -> str:
    job = svc.trainer.start(TrainRequest(heads=heads), wait=True)
    assert job["status"] == "succeeded", job
    return job["model"]


def client(svc: Service, **kw: Any) -> TestClient:
    return TestClient(create_app(svc), **kw)


def decide(c: TestClient, state: dict, *, heads: list[str] | None = None, **body: Any) -> dict:
    r = c.post("/v1/decide", json={"state": state, "heads": heads or ["color"], **body})
    assert r.status_code == 200, r.text
    return r.json()


def first_with_action(c: TestClient, action: str, *, stakes: str = "low") -> tuple[dict, dict]:
    """The first state on the way from the origin towards RED whose color decision is `action` (the trained head's
    exact probabilities are not pinned, only that such a state exists)."""
    for t in np.linspace(0.0, 3.0, 61):
        state = _state(RED_DIR, float(t))
        d = decide(c, state, stakes=stakes)
        if d["action"] == action:
            return state, d
    raise AssertionError(f"no state between the origin and RED gives a {action} decision")


def log_lines(home: Path, name: str = "decisions.jsonl") -> list[str]:
    path = Path(home) / name
    return [x for x in path.read_text(encoding="utf-8").splitlines() if x.strip()] if path.exists() else []


@pytest.fixture(scope="module")
def toy(tmp_path_factory):
    """One trained toy service for tests that never switch automation, move the registry or post feedback."""
    home = tmp_path_factory.mktemp("control")
    svc = service(home)
    model = train(svc)
    return SimpleNamespace(svc=svc, client=client(svc), model=model, home=home)


@pytest.fixture
def fresh(tmp_path):
    """A trained toy service with its own RLCD_HOME."""
    svc = service(tmp_path)
    model = train(svc)
    return SimpleNamespace(svc=svc, client=client(svc), model=model, home=tmp_path)


# ----------------------------------------------------------------------------- batteries

def test_systemone_answers_a_project_battery_and_returns_its_ref(tmp_path):
    svc = service(tmp_path, {"batteries": {"intake": BATTERY}})
    model = train(svc)
    c = client(svc)
    ref = {"name": "intake", "version": "2", "digest": svc.batteries(model)["intake"].digest}
    assert ref["digest"].startswith("sha256:")
    r = c.post("/v1/systemone", json={"battery": "intake", "state": POS})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["battery"] == ref and set(body["answers"]) == {"ok", "col", "lv"}
    adhoc = c.post("/v1/systemone", json={"questions": BATTERY["questions"], "state": POS}).json()
    assert adhoc["answers"] == body["answers"] and "battery" not in adhoc       # same questions, same answers
    assert c.post("/v1/systemone", json={"battery": "intake@2", "state": POS}).json()["battery"] == ref
    # a pinned version that is not the current one: the caller's thresholds were chosen for another meaning
    old = c.post("/v1/systemone", json={"battery": "intake@1", "state": POS})
    assert old.status_code == 422 and old.json()["kind"] == "battery_version_mismatch"
    assert old.json()["expected"] == "1" and old.json()["current"] == "2"
    assert old.json()["changelog"] == BATTERY["changelog"] and old.json()["retryable"] is False
    unknown = c.post("/v1/systemone", json={"battery": "outtake", "state": POS})
    assert unknown.status_code == 422 and unknown.json()["kind"] == "unknown_battery"
    assert unknown.json()["available"] == ["intake"]
    both = c.post("/v1/systemone", json={"battery": "intake", "questions": BATTERY["questions"], "state": POS})
    assert both.status_code == 422 and both.json()["kind"] == "validation"
    assert c.post("/v1/systemone", json={"state": POS}).status_code == 422
    malformed = c.post("/v1/systemone", json={"battery": "in take", "state": POS})
    assert malformed.status_code == 422 and malformed.json()["field"] == "battery"
    listed = {b["name"]: b for b in c.get("/v1/batteries").json()["batteries"]}
    assert listed["intake"]["source"] == "project" and listed["intake"]["problems"] == []
    assert listed["intake"]["digest"] == ref["digest"] and listed["intake"]["consumers"] == BATTERY["consumers"]
    assert all(q["trained"] for q in listed["intake"]["questions"].values())


def test_a_battery_receipt_goes_stale_when_the_battery_version_changes(tmp_path):
    svc = service(tmp_path, {"batteries": {"intake": BATTERY}})
    train(svc)
    r = client(svc).post("/v1/systemone", json={"battery": "intake", "state": POS, "receipt": True}).json()
    rid = r["receipt"]["id"]
    row = svc.receipt_view(rid)
    assert row["kind"] == "systemone" and row["receipt"]["questions"]["battery"] == r["battery"]
    assert row["receipt"]["questions"]["digest"] == r["battery"]["digest"] == r["receipt"]["questions_digest"]
    assert svc.check_receipt(rid, state=POS)["current"] is True
    # the operator ships version 3 of the battery (same home, restarted with the edited rlcd.yaml)
    bumped = service(tmp_path, {"batteries": {"intake": {**BATTERY, "version": "3"}}})
    out = client(bumped).post(f"/v1/receipts/{rid}/check", json={"state": POS}).json()
    assert out["current"] is False and out["checks"]["battery"] == "changed"
    assert any("battery intake was version 2, now version 3" in s for s in out["stale_reasons"])


def test_battery_problems_are_listed_and_a_broken_battery_is_never_answered(tmp_path):
    svc = service(tmp_path, {"batteries": {"broken": {"version": "1", "questions": {
        "x": {"type": "noul", "head": "nope"}}}}})
    train(svc)
    c = client(svc)
    listed = c.get("/v1/batteries").json()["batteries"]
    assert len(listed) == 1 and "no head named 'nope'" in listed[0]["problems"][0]
    r = c.post("/v1/systemone", json={"battery": "broken", "state": POS})
    assert r.status_code == 422 and r.json()["kind"] == "unknown_head"
    unversioned = service(tmp_path, {"batteries": {"b": {"questions": {"x": {"type": "noul", "head": "good"}}}}})
    r = client(unversioned).post("/v1/systemone", json={"battery": "b", "state": POS})
    assert r.status_code == 500 and r.json()["kind"] == "bad_config" and r.json()["retryable"] is False


# ----------------------------------------------------------------------------- question inputs

def test_question_inputs_read_named_fields_of_one_state_and_fail_closed(toy):
    c = toy.client
    state = {"request": {"features": POS["f"]}, "items": [{"f": NEG["f"]}], "flat": POS}
    q = {"type": "noul", "head": "good"}
    direct = {name: c.post("/v1/systemone", json={"state": s, "questions": {"q": q}}).json()["answers"]["q"]
              for name, s in (("pos", POS), ("neg", NEG))}
    questions = {"by_path": {**q, "path": "flat"},
                 "by_inputs": {**q, "inputs": {"f": "request.features"}},
                 "indexed": {**q, "path": "items.0"},
                 "below_path": {**q, "path": "items", "inputs": {"f": "0.f"}}}
    r = c.post("/v1/systemone", json={"state": state, "questions": questions})
    assert r.status_code == 200, r.text
    a = r.json()["answers"]
    assert a["by_path"] == a["by_inputs"] == direct["pos"]
    assert a["indexed"] == a["below_path"] == direct["neg"]
    for path, why in (("request.missing", "request.missing is not in the state"),
                      ("items.3.f", "items.3 is out of range"),
                      ("flat.f.x", "flat.f.x cannot be read")):
        r = c.post("/v1/systemone", json={"state": state, "questions": {
            "ok": questions["by_path"], "gap": {**q, "inputs": {"f": path}}}})
        body = r.json()
        assert r.status_code == 422 and body["kind"] == "missing_evidence", (path, body)
        assert body["question"] == "gap" and body["path"] == path and why in body["error"], body
    r = c.post("/v1/systemone", json={"state": state, "questions": {"q": {**q, "path": "a..b"}}})
    assert r.status_code == 422 and r.json()["kind"] == "validation" and "empty segment" in r.json()["error"]
    # the paths a question reads are part of its contract: same answer, another question digest
    one = c.post("/v1/systemone", json={"state": state, "questions": {"q": questions["by_path"]},
                                        "receipt": True}).json()
    two = c.post("/v1/systemone", json={"state": state, "questions": {"q": questions["by_inputs"]},
                                        "receipt": True}).json()
    assert one["answers"] == two["answers"]
    assert one["receipt"]["questions_digest"] != two["receipt"]["questions_digest"]
    assert one["receipt"]["state_digest"] == two["receipt"]["state_digest"] == digest(state)


def test_strict_refuses_a_state_without_the_inputs_training_had(toy):
    c = toy.client
    vec = {"x": [None, *POS["f"][1:]], "feature_names": NAMES, "feature_version": "toy-v1"}
    q = {"g": {"type": "noul", "head": "good"}}
    lax = c.post("/v1/systemone", json={"state": vec, "questions": q})
    assert lax.status_code == 200 and any("lacks 1 input feature" in w for w in lax.json()["warnings"])
    strict = c.post("/v1/systemone", json={"state": vec, "questions": q, "strict": True})
    body = strict.json()
    assert strict.status_code == 422 and body["kind"] == "missing_evidence" and body["retryable"] is False
    assert body["head"] == "good" and body["question"] == "g" and body["missing"] == ["f0"]
    capped = decide(c, vec, stakes="low")["heads"]["color"]
    assert capped["decision"]["tier"] != "act" and "missing_evidence" in capped["uncertainty"]["sources"]
    r = c.post("/v1/decide", json={"state": vec, "heads": ["color"], "strict": True})
    assert r.status_code == 422 and r.json()["kind"] == "missing_evidence" and r.json()["head"] == "color"


# ----------------------------------------------------------------------------- allowed options

def test_allowed_on_a_choice_question_keeps_unrenormalised_probabilities(toy):
    c = toy.client
    full = c.post("/v1/systemone", json={"state": RED, "questions": {"c": {"type": "choice", "head": "color"}}})
    p = full.json()["answers"]["c"]["probabilities"]
    assert full.json()["answers"]["c"]["choice"] == "red" and p["red"] > 0.9
    r = c.post("/v1/systemone", json={"state": RED, "questions": {
        "c": {"type": "choice", "head": "color", "allowed": ["green", "blue"]}}})
    assert r.status_code == 200, r.text
    a = r.json()["answers"]["c"]
    assert a["allowed"] == ["green", "blue"] and a["probabilities"] == {"green": p["green"], "blue": p["blue"]}
    assert a["choice"] == max(("green", "blue"), key=lambda o: p[o])
    assert a["excluded_mass"] == pytest.approx(p["red"], abs=2e-6) and a["excluded_mass"] >= U.LARGE_REMOVED_MASS
    assert any("question 'c'" in w and "options the request removed" in w for w in r.json()["warnings"])
    kept = c.post("/v1/systemone", json={"state": RED, "questions": {
        "c": {"type": "choice", "head": "color", "allowed": ["red", "green"]}}}).json()
    assert kept["answers"]["c"]["choice"] == "red"
    assert kept["answers"]["c"]["excluded_mass"] == pytest.approx(p["blue"], abs=2e-6)
    assert not any("options the request removed" in w for w in kept.get("warnings", []))
    for allowed, needle in ((["purple"], "does not have"), (["red", "red"], "twice"), ([], "non-empty")):
        r = c.post("/v1/systemone", json={"state": RED, "questions": {
            "c": {"type": "choice", "head": "color", "allowed": allowed}}})
        assert r.status_code == 422 and needle in r.json()["error"], (allowed, r.text)
    r = c.post("/v1/systemone", json={"state": RED, "questions": {
        "g": {"type": "noul", "head": "good", "allowed": ["true"]}}})
    assert r.status_code == 422 and r.json()["field"] == "allowed"


def test_decide_allowed_commits_to_the_best_allowed_class_with_its_unconditional_p(toy):
    c = toy.client
    base = decide(c, RED, stakes="medium")
    p = base["heads"]["color"]["answer"]["probabilities"]
    assert base["heads"]["color"]["decision"]["class"] == "red"
    d = decide(c, RED, stakes="medium", allowed={"color": ["green", "blue"]})
    h = d["heads"]["color"]
    best = max(("green", "blue"), key=lambda o: p[o])
    assert h["decision"]["class"] == h["answer"]["choice"] == best
    assert h["decision"]["p"] == pytest.approx(p[best], abs=1e-6)          # not renormalised over the allowed
    assert h["answer"]["allowed"] == ["green", "blue"]
    assert h["answer"]["excluded_mass"] == pytest.approx(p["red"], abs=2e-6)
    assert "options_removed" in h["uncertainty"]["sources"]
    assert h["decision"]["tier"] == d["action"] == "escalate"            # nothing allowed is likely: it abstains
    assert any("head 'color'" in w and "options the request removed" in w for w in d["warnings"])
    assert toy.svc.receipt_view(d["decision_id"])["heads"]["color"]["class"] == best
    assert d["receipt"]["questions_digest"] != base["receipt"]["questions_digest"]   # allowed is in the contract
    top = decide(c, RED, stakes="medium", allowed={"color": ["red", "blue"]})["heads"]["color"]
    assert top["decision"]["class"] == "red" and top["decision"]["tier"] == base["heads"]["color"]["decision"]["tier"]
    assert "options_removed" not in top["uncertainty"]["sources"]
    noul = c.post("/v1/decide", json={"state": RED, "heads": ["color", "good"], "allowed": {"good": ["true"]}})
    assert noul.status_code == 422 and noul.json()["field"] == "allowed" and "noul head" in noul.json()["error"]
    other = c.post("/v1/decide", json={"state": RED, "heads": ["color"], "allowed": {"level": ["1"]}})
    assert other.status_code == 422 and other.json()["heads"] == ["color"]
    unknown = c.post("/v1/decide", json={"state": RED, "heads": ["color"], "allowed": {"color": ["purple"]}})
    assert unknown.status_code == 422 and unknown.json()["classes"] == ["red", "green", "blue"]
    listed = c.post("/v1/decide", json={"state": RED, "allowed": ["red"]})
    assert listed.status_code == 422 and listed.json()["field"] == "allowed"


def test_a_systemone_receipt_commits_to_the_answers_choice_when_the_allowed_mass_is_tiny(fresh):
    r = fresh.client.post("/v1/systemone", json={"state": BLUE, "receipt": True, "questions": {
        "c": {"type": "choice", "head": "color", "allowed": ["green"]}}}).json()
    a = r["answers"]["c"]
    assert a["choice"] == "green" and a["probabilities"]["green"] < 5e-7 and a["excluded_mass"] == 1.0
    fb = fresh.client.post("/v1/feedback", json={"decision_id": r["receipt"]["id"], "label": "green", "source": "t"})
    assert fb.status_code == 200, fb.text
    assert fresh.svc.receipt_view(r["receipt"]["id"])["heads"]["c"]["class"] == "green"
    assert fb.json()["bernoulli"]["class"] == "green" and fb.json()["bernoulli"]["correct"] is True


def test_generic_decide_filters_options_by_route_contracts(tmp_path):
    routes = {"color": {"red": {"requires": ["paint"], "fallback": "green", "queue": "r"},
                        "green": {"allowed_data": ["public"], "queue": "g"},
                        "blue": {"latency_ms": 5000, "queue": "b"}}}
    svc = service(tmp_path, {"routes": routes})
    train(svc)
    c = client(svc)
    h = decide(c, RED, capabilities=["paint"], sensitivity="public")["heads"]["color"]
    assert h["decision"]["class"] == "red" and all(x["kept"] for x in h["considered"])
    assert h["contract"] == {"option": "red", "contract": routes["color"]["red"], "fallback": "green",
                             "fallback_contract": routes["color"]["green"]}
    # a fallback must pass the same filters as the option it replaces
    nofb = decide(c, RED, capabilities=["paint"], sensitivity="internal")["heads"]["color"]
    assert nofb["decision"]["class"] == "red" and nofb["contract"]["fallback"] is None
    gone = decide(c, RED, capabilities=[], sensitivity="secret", latency_budget_ms=100)
    g = gone["heads"]["color"]
    assert {x["option"]: x["reason"] for x in g["considered"]} == {
        "red": "needs paint, which the request does not list as available",
        "green": "may not receive secret data (allowed: public)",
        "blue": "expected latency 5000 ms exceeds the budget 100 ms"}
    assert g["decision"]["class"] is None and g["decision"]["tier"] == gone["action"] == "escalate"
    assert g["answer"]["allowed"] == [] and g["answer"]["excluded_mass"] == 1.0 and g["contract"] is None
    assert g["uncertainty"]["sources"] == ["options_removed"]
    assert any("deterministic filters removed every option" in w for w in gone["warnings"])
    assert svc.receipt_view(gone["decision_id"])["heads"]["color"]["class"] is None
    bad = client(service(tmp_path, {"routes": {"good": {"true": {}}}}))
    r = bad.post("/v1/decide", json={"state": RED, "heads": ["good"]})
    assert r.status_code == 500 and r.json()["kind"] == "bad_config"
    r = c.post("/v1/decide", json={"state": RED, "heads": ["color"], "latency_budget_ms": -5})
    assert r.status_code == 422 and r.json()["field"] == "latency_budget_ms"


def test_a_head_without_options_reports_its_own_rollout_mode(tmp_path):
    routes = {"color": {o: {"requires": ["paint"]} for o in ("red", "green", "blue")}}
    svc = service(tmp_path, {"routes": routes, "rollout": {"mode": "full", "heads": {"color": "shadow"}}})
    train(svc)
    c = client(svc)
    assert decide(c, RED, capabilities=["paint"])["heads"]["color"]["decision"]["rollout"]["mode"] == "shadow"
    dec = decide(c, RED, capabilities=[])["heads"]["color"]["decision"]
    assert dec["class"] is None and dec["tier"] == "escalate"
    assert dec["rollout"]["mode"] == "shadow"


# ----------------------------------------------------------------------------- receipts

def test_every_decision_is_a_chained_receipt(fresh):
    c, svc = fresh.client, fresh.svc
    states = [POS, NEG, MID, RED, BLUE]
    prev, ids = None, []
    for i in range(30):
        st = states[i % len(states)]
        if i % 3 == 0:
            r = c.post("/v1/decide", json={"state": st}).json()
            assert r["receipt"]["id"] == r["decision_id"] and r["receipt"]["state_digest"] == digest(st)
        elif i % 3 == 1:
            r = c.post("/v1/systemone", json={"state": st, "questions": {"q": {"type": "noul", "head": "good"}},
                                              "receipt": True}).json()
            assert r["receipt"]["state_digest"] == digest(st)
        else:
            r = c.post("/v1/rank", json={"candidates": [{"id": "a", "state": st}, {"id": "b", "state": MID}],
                                         "receipt": True}).json()
        rec = r["receipt"]
        assert rec["digest"].startswith("sha256:") and rec["prev"] == prev and rec["contract_digest"]
        prev = rec["digest"]
        ids.append(rec["id"])
    out = svc.verify_receipts()
    assert out["ok"] is True and out["receipts"] == 30 and out["legacy"] == 0 and out["breaks"] == []
    assert out["head"] == prev
    row = c.get(f"/v1/receipts/{ids[0]}").json()
    assert row["decision_id"] == ids[0] and row["kind"] == "decide" and row["model"] == fresh.model
    rc = row["receipt"]
    assert rc["format"] == "open-rlcd-receipt/1" and rc["pack"]["name"] == "toy" and rc["feature_version"] == "toy-v1"
    assert rc["policy_digest"] == svc.policy_digest(fresh.model) and rc["rollout"]["mode"] == "full"
    assert rc["prev"] is None and set(row["heads"]) == {"good", "color", "level"}
    for name, h in row["heads"].items():
        assert {"class", "p", "tier", "stakes", "uncertainty", "state"} <= set(h)
        assert ("probabilities" in h) is (name != "good")         # a noul's distribution is its class and p
    missing = c.get("/v1/receipts/nope")
    assert missing.status_code == 404 and missing.json()["kind"] == "unknown_decision"
    assert missing.json()["retryable"] is False
    # an edited line breaks the chain at that line
    path = Path(svc.home) / "decisions.jsonl"
    lines = path.read_text(encoding="utf-8").splitlines()
    edited = json.loads(lines[3])
    edited["model"] = "rlcd-0.1.9"
    lines[3] = json.dumps(edited, separators=(",", ":"), ensure_ascii=False)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    broken = svc.verify_receipts()
    assert broken["ok"] is False and broken["breaks"][0]["decision_id"] == edited["decision_id"]


def test_receipt_check_compares_state_alias_and_age(tmp_path):
    svc = service(tmp_path)
    v1 = train(svc)
    v2 = train(svc, ["color"])
    c = client(svc)
    alias = decide(c, RED)
    pinned = decide(c, RED, model=v2)
    assert alias["model"] == pinned["model"] == v2
    url = f"/v1/receipts/{alias['decision_id']}/check"
    same = c.post(url, json={"state": RED}).json()
    assert same["current"] is True and same["stale_reasons"] == []
    assert (same["checks"]["state"], same["checks"]["model"], same["checks"]["policy"]) == ("same", "same", "same")
    assert same["decision_id"] == alias["decision_id"] and same["kind"] == "decide"
    assert same["action"] == alias["action"] and same["checks"]["age_s"] >= 0
    assert c.post(url, json={"state_digest": digest(RED)}).json()["current"] is True
    assert c.post(url, json={"state": RED, "max_age_s": 3600}).json()["current"] is True
    changed = c.post(url, json={"state": {"f": [*RED["f"][:5], 9.0]}}).json()
    assert changed["current"] is False and changed["checks"]["state"] == "changed"
    assert "state differs" in changed["stale_reasons"][0]
    nothing = c.post(url, json={})
    assert nothing.status_code == 422 and nothing.json()["field"] == "state"
    assert c.post(url, json={"state": RED, "max_age_s": 0}).status_code == 422
    assert c.post("/v1/receipts/nope/check", json={"state": RED}).status_code == 404
    # rlcd-latest moves back to v1: a decision made through the alias no longer describes what the alias serves
    assert svc.promote(v1, reason="test rollback")["changed"] is True
    moved = c.post(url, json={"state": RED}).json()
    assert moved["current"] is False and moved["checks"]["model"] == "moved"
    assert f"rlcd-latest now points at {v1}, the decision was made by {v2}" in moved["stale_reasons"]
    held = c.post(f"/v1/receipts/{pinned['decision_id']}/check", json={"state": RED}).json()
    assert held["current"] is True and held["checks"]["model"] == "pinned"
    time.sleep(1.1)        # created_at has whole seconds: the receipt is now older than one second
    old = c.post(f"/v1/receipts/{pinned['decision_id']}/check", json={"state": RED, "max_age_s": 1}).json()
    assert old["current"] is False and old["checks"]["age_s"] > 1
    assert any("older than max_age_s=1" in s for s in old["stale_reasons"])


def test_a_legacy_decision_line_stays_readable_but_cannot_be_checked(fresh):
    c, svc = fresh.client, fresh.svc
    decide(c, RED)
    legacy = {"decision_id": "legacy-1", "created_at": "2026-01-01T00:00:00Z", "kind": "decide",
              "model": fresh.model, "heads": {"color": {"class": "red", "p": 0.9, "tier": "act", "state": RED}}}
    with open(Path(svc.home) / "decisions.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(legacy) + "\n")
    decide(c, NEG)                                         # the chain continues across the legacy line
    out = svc.verify_receipts()
    # a line without a receipt after the chain started can only be an insertion (or an older server writing to the
    # same home): the chain reports it, while the line itself stays readable
    assert out["receipts"] == 2 and out["ok"] is False
    assert [b["decision_id"] for b in out["breaks"]] == ["legacy-1"]
    assert c.get("/v1/receipts/legacy-1").json()["kind"] == "decide"
    r = c.post("/v1/receipts/legacy-1/check", json={"state": RED})
    assert r.status_code == 422 and r.json()["kind"] == "no_receipt"
    fb = c.post("/v1/feedback", json={"decision_id": "legacy-1", "label": "red", "source": "t"})
    assert fb.status_code == 200 and fb.json()["head"] == "color"


# ----------------------------------------------------------------------------- kill switch and keys

def test_the_kill_switch_caps_new_decisions_and_stales_act_receipts_only(fresh):
    c, svc = fresh.client, fresh.svc
    act = decide(c, RED, stakes="medium")
    assert act["action"] == "act"
    state, confirm = first_with_action(c, "confirm")
    # a confirm decision approves an action only once a person confirmed it
    pending = c.post(f"/v1/receipts/{confirm['decision_id']}/check", json={"state": state}).json()
    assert pending["current"] is False and pending["checks"]["confirmation"] == "missing"
    assert c.post(f"/v1/receipts/{confirm['decision_id']}/confirm", json={"by": "alice"}).status_code == 200
    off = c.post("/v1/automation", json={"enabled": False, "reason": "incident 42", "by": "oncall"})
    assert off.status_code == 200, off.text
    view = off.json()
    assert view["automation"]["enabled"] is False and view["automation"]["reason"] == "incident 42"
    assert view["automation"]["by"] == "oncall" and view["history"][-1]["enabled"] is False
    assert c.get("/health").json()["automation"] is False
    stale = c.post(f"/v1/receipts/{act['decision_id']}/check", json={"state": RED}).json()
    assert stale["current"] is False and stale["checks"]["automation"] == "capped"
    assert any("automation is switched off or capped" in s for s in stale["stale_reasons"])
    ok = c.post(f"/v1/receipts/{confirm['decision_id']}/check", json={"state": state}).json()
    assert ok["current"] is True and "automation" not in ok["checks"]
    assert ok["checks"]["confirmation"] == "confirmed"
    # decisions are still made, logged and returned: capped at confirm, saying what they would have done
    now = decide(c, RED, stakes="medium")
    dec = now["heads"]["color"]["decision"]
    assert now["action"] == dec["tier"] == "confirm"
    assert dec["rollout"]["capped"] is True and dec["rollout"]["proposed_tier"] == "act"
    assert dec["rollout"]["automation"] is False and dec["rollout"]["cap"] == "confirm"
    assert "incident 42" in dec["rollout"]["reason"]
    assert any("automation is switched off" in w and "act became confirm" in w for w in now["warnings"])
    assert now["rollout"]["capped"] is True and now["rollout"]["proposed_action"] == "act"
    assert now["rollout"]["automation"]["enabled"] is False
    row = svc.receipt_view(now["decision_id"])
    assert row["heads"]["color"]["proposed_tier"] == "act" and row["receipt"]["rollout"]["capped"] is True
    # abstaining stays the model's own statement: the switch never lifts it to confirm
    low = decide(c, MID, stakes="high")
    assert low["action"] == "escalate" and low["heads"]["color"]["decision"]["rollout"]["capped"] is False
    on = c.post("/v1/automation", json={"enabled": True, "reason": "resolved"})
    assert on.json()["automation"]["enabled"] is True and len(c.get("/v1/automation").json()["history"]) == 2
    assert decide(c, RED, stakes="medium")["action"] == "act"
    assert c.post(f"/v1/receipts/{act['decision_id']}/check", json={"state": RED}).json()["current"] is True


def test_automation_forced_off_by_the_environment_cannot_be_switched_on(fresh, monkeypatch):
    monkeypatch.setenv("RLCD_AUTOMATION", "off")
    c = fresh.client
    st = c.get("/v1/automation").json()["automation"]
    assert st["enabled"] is False and st["forced_by_env"] is True and st["by"] == "environment"
    r = c.post("/v1/automation", json={"enabled": True, "reason": "try"})
    assert r.status_code == 409 and r.json()["kind"] == "automation_locked"
    assert r.json()["retryable"] is False and "retry-after" not in r.headers
    assert c.post("/v1/automation", json={"enabled": False, "reason": "belt and braces"}).status_code == 200
    assert c.get("/health").json()["automation"] is False
    d = decide(c, RED, stakes="medium")
    assert d["action"] == "confirm" and "RLCD_AUTOMATION=off" in d["heads"]["color"]["decision"]["rollout"]["reason"]


def test_the_admin_key_guards_training_and_the_kill_switch(fresh, monkeypatch):
    c = fresh.client
    monkeypatch.setenv("RLCD_API_KEY", "caller-key")
    monkeypatch.setenv("RLCD_ADMIN_KEY", "admin-key")
    caller, admin = {"Authorization": "Bearer caller-key"}, {"x-api-key": "admin-key"}
    body = {"state": RED, "heads": ["color"]}
    r = c.post("/v1/decide", json=body)
    assert r.status_code == 401 and r.json()["kind"] == "unauthorized" and r.json()["retryable"] is False
    assert c.post("/v1/decide", json=body, headers=caller).status_code == 200
    assert c.post("/v1/decide", json=body, headers=admin).status_code == 200          # the admin key passes too
    for path, payload in (("/v1/automation", {"enabled": True}), ("/v1/train", {"heads": ["nope"]})):
        for headers in ({}, caller, {"Authorization": "Bearer wrong"}):
            r = c.post(path, json=payload, headers=headers)
            assert r.status_code == 401 and "admin key" in r.json()["error"], (path, headers, r.text)
    assert c.post("/v1/automation", json={"enabled": False, "reason": "drill"}, headers=admin).status_code == 200
    r = c.post("/v1/train", json={"heads": ["nope"]}, headers=admin)
    assert r.status_code == 422 and "unknown heads" in r.json()["error"]       # past the key check; nothing trained
    assert c.get("/v1/automation", headers=caller).status_code == 200
    assert c.get("/v1/automation").status_code == 401
    assert c.get("/health").status_code == 200
    # only the admin key set: decision routes are open, the operator's routes still need it
    monkeypatch.delenv("RLCD_API_KEY")
    assert c.post("/v1/decide", json=body).status_code == 200
    assert c.post("/v1/automation", json={"enabled": True}).status_code == 401
    assert c.post("/v1/automation", json={"enabled": True}, headers=admin).json()["automation"]["enabled"] is True


# ----------------------------------------------------------------------------- rollout

def test_shadow_and_assist_cap_act_at_confirm_and_say_so(tmp_path):
    base = service(tmp_path)
    train(base)
    act = decide(client(base), RED, stakes="medium")
    assert act["action"] == "act" and act["rollout"]["mode"] == "full" and act["rollout"]["capped"] is False
    assert act["rollout"]["proposed_action"] == "act"
    for mode in ("shadow", "assist"):
        svc = service(tmp_path, {"rollout": {"mode": mode}})
        c = client(svc)
        assert c.get("/health").json()["rollout"] == mode
        d = decide(c, RED, stakes="medium")
        ro = d["heads"]["color"]["decision"]["rollout"]
        assert d["action"] == "confirm"
        assert {k: ro[k] for k in ("mode", "automation", "cap", "capped", "proposed_tier")} == {
            "mode": mode, "automation": True, "cap": "confirm", "capped": True, "proposed_tier": "act"}
        assert f"rollout mode {mode}" in ro["reason"]
        assert any(f"rollout mode {mode}" in w and "act became confirm" in w for w in d["warnings"])
        assert d["rollout"]["mode"] == mode and d["rollout"]["capped"] is True
        assert d["rollout"]["proposed_action"] == "act"
        row = svc.receipt_view(d["decision_id"])
        assert row["heads"]["color"]["proposed_tier"] == "act" and row["receipt"]["rollout"]["mode"] == mode
        # an act receipt made under `full` no longer authorises an automatic action
        stale = c.post(f"/v1/receipts/{act['decision_id']}/check", json={"state": RED}).json()
        assert stale["current"] is False and stale["checks"]["automation"] == "capped"
        # the cap is never the abstain action
        esc = decide(c, MID, stakes="high")
        assert esc["action"] == "escalate" and esc["rollout"]["capped"] is False


def test_limited_and_per_head_rollout(tmp_path):
    train(service(tmp_path))
    lim = client(service(tmp_path, {"rollout": {"mode": "limited", "limited": {"stakes": ["low"]}}}))
    low = decide(lim, RED, stakes="low")
    assert low["action"] == "act" and low["heads"]["color"]["decision"]["rollout"]["capped"] is False
    med = decide(lim, RED, stakes="medium")
    assert med["action"] == "confirm" and med["rollout"]["limited"] == {"stakes": ["low"], "heads": None}
    assert "only stakes ['low'] are automated" in med["heads"]["color"]["decision"]["rollout"]["reason"]
    sliced = client(service(tmp_path, {"rollout": {"mode": "limited",
                                                   "limited": {"stakes": ["low", "medium"], "heads": ["level"]}}}))
    d = decide(sliced, RED, stakes="low")
    assert d["action"] == "confirm"
    assert "only heads ['level'] are automated" in d["heads"]["color"]["decision"]["rollout"]["reason"]
    per = client(service(tmp_path, {"rollout": {"mode": "full", "heads": {"color": "shadow"}}}))
    d = decide(per, RED, heads=["color", "level"], stakes="medium")
    col, lev = d["heads"]["color"]["decision"], d["heads"]["level"]["decision"]
    assert col["tier"] == "confirm" and col["rollout"]["mode"] == "shadow" and col["rollout"]["proposed_tier"] == "act"
    assert "head 'color' is in rollout mode shadow" in col["rollout"]["reason"]
    assert lev["rollout"]["mode"] == "full" and lev["rollout"]["capped"] is False
    assert d["rollout"]["mode"] == "full" and d["rollout"]["heads"] == {"color": "shadow"}
    broken = client(service(tmp_path, {"rollout": {"mode": "yolo"}}))
    r = broken.post("/v1/decide", json={"state": RED, "heads": ["color"]})
    assert r.status_code == 500 and r.json()["kind"] == "bad_config"
    assert broken.get("/health").json()["rollout"] == "invalid"


def test_health_reports_the_rollout_mode_and_the_kill_switch(fresh):
    h = fresh.client.get("/health").json()
    assert h["rollout"] == "full" and h["automation"] is True and h["model"] == fresh.model


# ----------------------------------------------------------------------------- promote, rollback, pinned versions

def test_promote_and_rollback_move_the_alias_and_restart_foreign_ledgers(tmp_path):
    svc = service(tmp_path)
    v1 = train(svc)
    v2 = train(svc, ["color"])
    c = client(svc)
    assert svc.registry.previous_latest() == v1                 # no history yet: the newest older version
    assert svc.ledger.table("color")["valid_for"] == [v2]
    assert svc.ledger.table("good")["valid_for"] == [v1, v2]   # carried over: the very same head
    out = svc.promote(v1, reason="bad color head", by="oncall")
    assert out == {"latest": v1, "previous": v2, "changed": True, "ledger_restarted": ["color"]}
    assert svc.resolve(None) == v1 and c.get("/health").json()["model"] == v1
    hist = svc.registry.registry()["history"][-1]
    assert (hist["from"], hist["to"], hist["reason"], hist["by"]) == (v2, v1, "bad color head", "oncall")
    color = svc.ledger.table("color")
    assert color["model_version"] == v1 and color["valid_for"] == [v1] and color["online_trials"] == 0
    assert color["bins"] == svc.loaded(v1, "color").holdout_ledger()["bins"]
    assert svc.ledger.table("good")["valid_for"] == [v1, v2]   # its evidence is kept
    assert svc.registry.previous_latest() == v2
    assert svc.promote(v1)["changed"] is False
    with pytest.raises(ContractError) as e:
        svc.promote("rlcd-0.1.9")
    assert e.value.kind == "unknown_model"
    assert decide(c, RED)["model"] == v1
    back = svc.promote(svc.registry.previous_latest())
    assert back["latest"] == v2 and back["ledger_restarted"] == ["color"] and svc.resolve(None) == v2


def test_a_train_after_a_rollback_keeps_the_outcomes_scored_for_carried_heads(tmp_path):
    svc = service(tmp_path)
    v1 = train(svc)
    train(svc, ["color"])
    svc.promote(v1)
    d = svc.decide(DecideRequest(state=POS, heads=["good"]))
    assert d["model"] == v1
    fb = svc.feedback(FeedbackRequest(decision_id=d["decision_id"], label="true", source="test"))
    assert fb["bernoulli"]["updated"] is True and svc.ledger.table("good")["online_trials"] == 1
    train(svc, ["level"])                        # `good` is carried over from v1, unchanged
    table = svc.ledger.table("good")
    assert table["online_trials"] == 1 and d["decision_id"] in table["applied"]


def test_a_request_pinned_to_an_older_version_gates_on_that_versions_holdout_table(tmp_path):
    svc = service(tmp_path)
    v1 = train(svc)
    v2 = train(svc, ["color"])
    live = svc.ledger.table("color")
    assert live["model_version"] == v2 and v1 not in live["valid_for"]
    old = svc.loaded(v1, "color")
    # the table v1 had while it was live (archived when v2 replaced it): its holdout counts, no outcomes since
    assert svc.gate_table(v1, "color", old)["bins"] == old.holdout_ledger()["bins"]
    assert svc.gate_table(v2, "color", svc.loaded(v2, "color")) == live
    d = decide(client(svc), RED, stakes="medium", model=v1)
    dec = d["heads"]["color"]["decision"]
    assert d["model"] == v1 and dec["tier"] == "act" and dec["gate"]["passed"] is True
    wins, losses = old.holdout_ledger()["bins"][dec["class"]][dec["bernoulli"]["bin"]]
    assert (dec["bernoulli"]["wins"], dec["bernoulli"]["losses"]) == (wins, losses) and wins > 0


# ----------------------------------------------------------------------------- feedback, redaction, outcomes

def test_feedback_on_a_systemone_receipt_scores_the_ledger_by_question_or_head(fresh):
    c, svc = fresh.client, fresh.svc
    r = c.post("/v1/systemone", json={"state": POS, "receipt": True, "questions": {
        "ok": {"type": "noul", "head": "good"}, "col": {"type": "choice", "head": "color"}}}).json()
    rid, a = r["receipt"]["id"], r["answers"]
    row = c.get(f"/v1/receipts/{rid}").json()
    assert row["kind"] == "systemone" and "action" not in row
    ok, col = row["heads"]["ok"], row["heads"]["col"]
    assert (ok["head"], ok["class"], ok["p"], ok["tier"]) == ("good", "true", a["ok"]["noul"], None)
    assert (col["head"], col["class"]) == ("color", a["col"]["choice"])
    assert col["p"] == a["col"]["probabilities"][a["col"]["choice"]]
    before = svc.ledger.table("good")["online_trials"]
    fb = c.post("/v1/feedback", json={"decision_id": rid, "head": "ok", "label": "yes", "source": "review"}).json()
    assert fb["head"] == "good" and fb["label"] == "true"
    assert fb["bernoulli"]["updated"] is True and fb["bernoulli"]["correct"] is True
    table = svc.ledger.table("good")
    assert table["online_trials"] == before + 1 and rid in table["applied"]
    stored = svc.logs.feedback()[-1]
    assert stored["decision_id"] == rid and stored["state"] == POS and stored["time"] == row["created_at"]
    by_head = c.post("/v1/feedback", json={"decision_id": rid, "head": "color", "label": a["col"]["choice"],
                                           "source": "review"}).json()
    assert by_head["head"] == "color" and by_head["bernoulli"]["updated"] is True
    unnamed = c.post("/v1/feedback", json={"decision_id": rid, "label": "true", "source": "review"})
    assert unnamed.status_code == 422 and unnamed.json()["heads"] == ["ok", "col"]
    # one head asked twice: only the question id says which answer the outcome is for
    twice = c.post("/v1/systemone", json={"state": POS, "receipt": True, "questions": {
        "a": {"type": "noul", "head": "good"}, "b": {"type": "noul", "head": "good"}}}).json()["receipt"]["id"]
    amb = c.post("/v1/feedback", json={"decision_id": twice, "head": "good", "label": "true", "source": "review"})
    assert amb.status_code == 422 and "several questions" in amb.json()["error"]
    named = c.post("/v1/feedback", json={"decision_id": twice, "head": "b", "label": "true", "source": "review"})
    assert named.status_code == 200 and named.json()["bernoulli"]["updated"] is True
    plain = c.post("/v1/systemone", json={"state": POS, "questions": {"q": {"type": "noul", "head": "good"}}}).json()
    assert "receipt" not in plain                      # serve.receipts defaults to decisions only


def test_secrets_in_a_state_are_redacted_from_the_logs_by_default(fresh):
    c, svc = fresh.client, fresh.svc
    state = {**POS, "note": SECRET}
    d = decide(c, state, heads=["good"])
    raw = (Path(svc.home) / "decisions.jsonl").read_text(encoding="utf-8")
    assert "AKIAIOSFODNN7EXAMPLE" not in raw and "hunter2hunter2" not in raw
    logged = svc.receipt_view(d["decision_id"])["heads"]["good"]["state"]
    assert logged["f"] == POS["f"]                    # numbers and structure are untouched
    assert logged["note"] == "deploy with [REDACTED:aws_key], password=[REDACTED:assignment]"
    assert d["receipt"]["state_digest"] == digest(state)            # the digest is of the state as sent
    assert c.post(f"/v1/receipts/{d['decision_id']}/check", json={"state": state}).json()["checks"]["state"] == "same"
    assert svc.verify_receipts()["ok"] is True
    for body in ({"head": "good", "state": state}, {"decision_id": d["decision_id"]}):
        assert c.post("/v1/feedback", json={**body, "label": "true", "source": "t"}).status_code == 200, body
    fb = (Path(svc.home) / "feedback.jsonl").read_text(encoding="utf-8")
    assert "AKIAIOSFODNN7EXAMPLE" not in fb and "hunter2hunter2" not in fb and fb.count("[REDACTED:aws_key]") == 2


def test_redaction_follows_serve_redact(tmp_path):
    svc = service(tmp_path, {"serve": {"redact": False}})
    train(svc)
    c = client(svc)
    state = {**POS, "note": SECRET}
    decide(c, state, heads=["good"])
    fb = c.post("/v1/feedback", json={"head": "good", "state": state, "label": "true", "source": "t"})
    assert fb.status_code == 200, fb.text
    assert SECRET in (tmp_path / "decisions.jsonl").read_text(encoding="utf-8")
    assert SECRET in (tmp_path / "feedback.jsonl").read_text(encoding="utf-8")
    pii = service(tmp_path, {"serve": {"redact": ["secrets", "pii"]}})
    e = decide(client(pii), {**POS, "note": "mail jane.doe@example.com"}, heads=["good"])
    assert pii.receipt_view(e["decision_id"])["heads"]["good"]["state"]["note"] == "mail [REDACTED:email]"
    bad = client(service(tmp_path, {"serve": {"redact": ["passwords"]}}))
    r = bad.post("/v1/decide", json={"state": POS, "heads": ["good"]})
    assert r.status_code == 500 and r.json()["kind"] == "bad_config"


def test_outcome_rows_join_a_decision_and_never_touch_the_ledger(fresh):
    c, svc = fresh.client, fresh.svc
    d = decide(c, RED)
    before = svc.ledger.table("color")
    r = c.post("/v1/outcome", json={"decision_id": d["decision_id"], "task_success": True, "route_taken": "vector",
                                     "fallback": False, "latency_ms": 812.5, "cost": 0.0021,
                                     "tokens": {"input": 1200, "output": 85},
                                     "retries": {"transport": 1, "workflow": 0}, "verified_by": "ci",
                                     "tags": ["canary"]})
    assert r.status_code == 200, r.text
    out = r.json()
    assert out["stored"] is True and out["decision_id"] == d["decision_id"] and out["id"]
    row = svc.logs.outcomes()[-1]
    assert row["id"] == out["id"] and row["received_at"] and (row["kind"], row["model"]) == ("decide", d["model"])
    assert row["task_success"] is True and row["fallback"] is False and row["retries"] == {"transport": 1,
                                                                                           "workflow": 0}
    assert "provider" not in row and "note" not in row          # fields the caller did not send are not stored
    assert svc.ledger.table("color") == before
    unknown = c.post("/v1/outcome", json={"decision_id": "nope", "task_success": False})
    assert unknown.status_code == 422 and unknown.json()["kind"] == "unknown_decision"
    for bad in ({"task_success": True}, {"decision_id": d["decision_id"], "cost": -1},
                {"decision_id": d["decision_id"], "retries": {"transport": "many"}}):
        assert c.post("/v1/outcome", json=bad).status_code == 422, bad


# ----------------------------------------------------------------------------- errors, dry runs, receipts: all

class BlockingPack(ToyPack):
    """Holds a training job inside `datasets` until the test releases it."""

    def __init__(self) -> None:
        super().__init__(n=1200)
        self.entered, self.release = threading.Event(), threading.Event()

    def datasets(self, *a: Any, **kw: Any):
        self.entered.set()
        assert self.release.wait(30)
        return super().datasets(*a, **kw)


def test_error_bodies_say_whether_the_same_request_may_succeed_later(tmp_path):
    pack = BlockingPack()
    svc = service(tmp_path, pack=pack)
    c = client(svc)
    first = c.post("/v1/train", json={"heads": ["good"]})
    assert first.status_code == 202
    try:
        assert pack.entered.wait(10)
        busy = c.post("/v1/train", json={})
        assert busy.status_code == 409 and busy.json()["kind"] == "busy" and busy.json()["retryable"] is True
        assert busy.headers["retry-after"] == "5" and busy.json()["job_id"] == first.json()["job_id"]
    finally:
        pack.release.set()
    deadline = time.time() + 30
    while svc.trainer.running() and time.time() < deadline:
        time.sleep(0.02)
    assert svc.trainer.get(first.json()["job_id"])["status"] == "succeeded"
    bad = c.post("/v1/decide", json={"state": POS, "stakes": "extreme"})
    assert bad.status_code == 422 and bad.json()["retryable"] is False and "retry-after" not in bad.headers
    missing = c.get("/v1/receipts/nope")
    assert missing.status_code == 404 and missing.json()["retryable"] is False


def test_a_dry_run_decides_without_logging_in_its_own_thread_only(fresh):
    svc, c = fresh.svc, fresh.client
    decide(c, POS)
    before = (Path(svc.home) / "decisions.jsonl").read_bytes()
    with svc.dry_run():
        assert svc.is_dry
        d = svc.decide(DecideRequest(state=RED, heads=["color"]))
        s = svc.systemone(SystemOneRequest(state=POS, questions={"q": {"type": "noul", "head": "good"}}, receipt=True))
        k = svc.rank(RankRequest(candidates=[{"id": "a", "state": POS}], receipt=True))
        with svc.dry_run():
            pass
        assert svc.is_dry                                  # a nested block restores the outer one
        served = decide(c, NEG)                            # the server's own threads keep logging meanwhile
    assert not svc.is_dry
    assert d["decision_id"] == d["receipt"]["id"] == "dry-run" and d["heads"]["color"]["decision"]["class"] == "red"
    assert s["receipt"]["id"] == k["receipt"]["id"] == "dry-run"
    after = (Path(svc.home) / "decisions.jsonl").read_bytes()
    assert after.startswith(before) and after.count(b"\n") == before.count(b"\n") + 1
    assert svc.receipt_view(served["decision_id"])["heads"]["color"]["state"] == NEG
    assert c.get("/v1/receipts/dry-run").status_code == 404
    assert svc.verify_receipts()["ok"] is True


def test_serve_receipts_all_records_systemone_and_rank_calls(tmp_path):
    svc = service(tmp_path, {"serve": {"receipts": "all"}})
    train(svc)
    c = client(svc)
    assert svc.receipts_mode() == "all"
    q = {"q": {"type": "noul", "head": "good"}}
    s = c.post("/v1/systemone", json={"state": POS, "questions": q}).json()
    assert svc.receipt_view(s["receipt"]["id"])["kind"] == "systemone"
    cands = [{"id": "neg", "state": NEG}, {"id": "pos", "state": POS}]
    k = c.post("/v1/rank", json={"candidates": cands}).json()
    row = svc.receipt_view(k["receipt"]["id"])
    assert row["kind"] == "rank" and row["head"] == "good" and row["option"] is None and row["heads"] == {}
    assert row["ranking"] == [{"id": x["id"], "probability": x["probability"]} for x in k["candidates"]]
    assert [x["id"] for x in row["ranking"]] == ["pos", "neg"]
    assert row["receipt"]["state_digest"] == digest({"state": None, "candidates": cands})
    assert len(log_lines(tmp_path)) == 2 and svc.verify_receipts()["ok"] is True
    plain = client(service(tmp_path)).post("/v1/systemone", json={"state": POS, "questions": q}).json()
    assert "receipt" not in plain and len(log_lines(tmp_path)) == 2
    broken = client(service(tmp_path, {"serve": {"receipts": "sometimes"}}))
    r = broken.post("/v1/systemone", json={"state": POS, "questions": q})
    assert r.status_code == 500 and r.json()["kind"] == "bad_config"


def test_a_state_with_a_lone_surrogate_is_still_decided(tmp_path):
    svc = service(tmp_path, {"serve": {"log_states": False}})
    train(svc)
    c = client(svc, raise_server_exceptions=False)
    body = '{"state": {"f": [0, 0, 0, 0, 0, 0], "note": "\\ud800"}, "heads": ["color"]}'
    r = c.post("/v1/decide", content=body.encode(), headers={"content-type": "application/json"})
    assert r.status_code == 200, r.text
