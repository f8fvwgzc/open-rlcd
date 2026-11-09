"""The control layer of the documents pack, through the service and over HTTP, on one trained model.

What: /v1/decide with `allowed` options for route / response_type; route contracts (rlcd.yaml `routes:` filtered by
the request's capabilities, sensitivity and latency budget, the chosen contract and its usable fallback in
plan.contracts, a head with no option left escalating the plan); `return_evidence`; stale or undated evidence
(pack_config.max_evidence_age_days / require_evidence_dates) capping answerable; the evidence digests in the receipt;
/v1/evaluate's decision table (failed checks, a partial or missing artifact, citations outside the evidence, the
binding to a plan's evidence, generated evidence, the finite revision loop, `revise` and `next`); `strict`.

Why: these are the deterministic rules that must win over a favourable semantic answer. A regression would ship an
answer judged against other evidence than the plan selected, or keep a generator revising forever, and the heads'
own tests would not notice.

How: one module-scoped training as in tests/test_documents_pack.py (synthetic corpus, hashing encoder; the slow
part of this module), except that the two gate heads are trained without a conformal target: their holdout verdict
can then allow act / ship, so the caps under test actually lower something. Other rlcd.yaml sections (routes, rollout,
pack_config, evaluate) are served by further Service instances on the same trained home. The baseline (a product's
SKU question whose plan acts, and its correct answer, which ships) is found by scanning the synthetic products
instead of pinning a trained head's numbers; where a changed request's outcome still depends on those numbers, a test
asserts the invariant ("never stronger than review") rather than an exact tier.

What it does not cover: the generic /v1/decide, receipts, rollout and the kill switch with the toy pack
(tests/test_control_service.py), and the documents heads' accuracy (tests/test_documents_pack.py).
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from fastapi.testclient import TestClient

from open_rlcd.core.config import Project
from open_rlcd.core.contract import DecideRequest, TrainRequest
from open_rlcd.core.digest import digest
from open_rlcd.core.service import Service
from open_rlcd.packs.documents import write_synth
from open_rlcd.packs.documents.decide import EVALUATE_HEADS, EVIDENCE_FIELDS
from open_rlcd.packs.documents.world import build_world
from open_rlcd.serve.api import create_app

ENC = {"kind": "hashing", "dim": 256}
GATES_WITHOUT_CONFORMAL = {"answerable": {"conformal": None}, "grounded": {"conformal": None}}
ROUTES = {
    "response_type": {
        "direct_answer": {"llm": "small", "context_budget_tokens": 2000, "verify": "postflight",
                          "fallback": "escalate_human"},
        "summarize": {"llm": "large", "requires": ["llm_large"], "allowed_data": ["public", "internal"],
                      "latency_ms": 4000, "verify": "postflight", "fallback": "escalate_human"},
        "escalate_human": {"human": True}},
    "route": {
        "vector": {"requires": ["vector"], "retriever": "dense", "fallback": "hybrid"},
        "hybrid": {"requires": ["vector", "graph"], "fallback": "graph"},
        "graph": {"requires": ["graph"]},
        "none": {"enabled": False}},
}
OFF_TOPIC = "The weather is nice today."
NOT_SAME_EVIDENCE = "the answer is judged against other evidence than plan"


def _merge(a: dict, b: dict) -> dict:
    out = dict(a)
    for k, v in b.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def http(svc: Service) -> TestClient:
    return TestClient(create_app(svc))


def evaluate(c: TestClient, body: dict) -> dict:
    r = c.post("/v1/evaluate", json=body)
    assert r.status_code == 200, r.text
    return r.json()


@pytest.fixture(scope="module")
def docs(tmp_path_factory):
    root = tmp_path_factory.mktemp("control-documents")
    home = root / "var"
    mp = pytest.MonkeyPatch()
    # packs read the label store and the corpus from project.home, which RLCD_HOME (set by conftest) overrides
    mp.setenv("RLCD_HOME", str(home))
    write_synth(home, ENC, seed=7, n_queries=900)
    base = {"pack": "documents", "pack_config": {"encoder": ENC}, "heads": GATES_WITHOUT_CONFORMAL}
    svc = Service(Project(root=root, data=base), home=home)
    job = svc.trainer.start(TrainRequest(), wait=True)
    assert job["status"] == "succeeded", job.get("error")

    def make(**sections: Any) -> Service:
        """Another service on the trained home, with these rlcd.yaml sections merged into the base config."""
        return Service(Project(root=root, data=_merge(base, sections)), home=home)

    corpus = svc.pack.corpus(svc.project, svc.provider(svc.resolve(None)))
    yield SimpleNamespace(root=root, home=home, svc=svc, client=http(svc), job=job, make=make, corpus=corpus,
                          world=build_world(7))
    mp.undo()


@pytest.fixture(scope="module")
def case(docs):
    """A product whose SKU question gets a plan that acts (direct_answer over the vector route) and whose correct
    answer, judged against that plan's evidence at low stakes, ships: the baseline every cap below lowers."""
    for p in docs.world.products:
        query = f"What is the SKU of the {p.subject}?"
        answer = f"The SKU of the {p.subject} is {p.facts['sku'].value['en']}."
        d = docs.client.post("/v1/decide", json={"state": {"query": query}, "return_evidence": True}).json()
        if (d["action"], d["plan"]["response_type"], d["plan"]["route"]) != ("act", "direct_answer", "vector"):
            continue
        state = {"query": query, "evidence": d["evidence"], "answer": answer}
        review = evaluate(docs.client, {"state": state, "stakes": "low"})
        if review["verdict"] == "ship":
            return SimpleNamespace(product=p, query=query, answer=answer, plan=d, state=state, review=review)
    pytest.fail("no product's SKU question gives a plan that acts and an answer that ships")


# ----------------------------------------------------------------------------- decide: deterministic filters

def test_decide_allowed_for_route_and_response_type(docs, case):
    c = docs.client
    base = case.plan
    rp = base["heads"]["route"]["answer"]["probabilities"]
    others = [o for o in ("vector", "graph", "hybrid", "none") if o != "vector"]
    d = c.post("/v1/decide", json={"state": {"query": case.query}, "allowed": {"route": others}}).json()
    route = d["heads"]["route"]
    best = max(others, key=lambda o: rp[o])
    assert route["decision"]["class"] == d["plan"]["route"] == best
    assert route["decision"]["p"] == pytest.approx(rp[best], abs=1e-6)        # unconditional, not renormalised
    assert route["answer"]["allowed"] == others
    assert route["answer"]["excluded_mass"] == pytest.approx(rp["vector"], abs=2e-6)
    assert "options_removed" in route["uncertainty"]["sources"]
    # the head put its mass on the removed route: nothing allowed is likely, the plan escalates
    assert route["decision"]["tier"] == d["action"] == "escalate"
    assert d["plan"]["next"] == "escalate_to_human" and "route" in d["plan"]["reason"]
    assert any("head 'route'" in w and "options the request removed" in w for w in d["warnings"])
    kept = c.post("/v1/decide", json={"state": {"query": case.query},
                                      "allowed": {"response_type": ["direct_answer", "escalate_human"]}}).json()
    rt = kept["heads"]["response_type"]
    assert rt["decision"]["class"] == "direct_answer" and rt["answer"]["allowed"] == ["direct_answer", "escalate_human"]
    assert rt["decision"]["p"] == base["heads"]["response_type"]["decision"]["p"] and kept["action"] == base["action"]
    for allowed, needle in (({"needs_graph": ["true"]}, "take no part"), ({"route": ["teleport"]}, "does not have"),
                            ({"route": []}, "non-empty")):
        r = c.post("/v1/decide", json={"state": {"query": case.query}, "allowed": allowed})
        assert r.status_code == 422 and r.json()["field"] == "allowed", (allowed, r.text)
        assert needle in r.json()["error"], (allowed, r.text)


def test_route_contracts_filter_options_before_the_heads_choose(docs, case):
    c = http(docs.make(routes=ROUTES))
    state = {"query": case.query}
    only_vector = c.post("/v1/decide", json={"state": state, "capabilities": ["vector"]}).json()
    route = only_vector["heads"]["route"]
    assert {x["option"]: (x["kept"], x["reason"]) for x in route["considered"]} == {
        "vector": (True, None),
        "graph": (False, "needs graph, which the request does not list as available"),
        "hybrid": (False, "needs graph, which the request does not list as available"),
        "none": (False, "disabled by configuration")}
    assert route["decision"]["class"] == only_vector["plan"]["route"] == "vector"
    contracts = only_vector["plan"]["contracts"]
    assert contracts["route"] == {"option": "vector", "contract": ROUTES["route"]["vector"], "fallback": None,
                                  "fallback_contract": None}           # every route along its chain needs graph
    assert contracts["response_type"] == {"option": "direct_answer",
                                          "contract": ROUTES["response_type"]["direct_answer"],
                                          "fallback": "escalate_human", "fallback_contract": {"human": True}}
    summarize = {x["option"]: x for x in only_vector["heads"]["response_type"]["considered"]}["summarize"]
    assert summarize == {"option": "summarize", "kept": False,
                         "reason": "needs llm_large, which the request does not list as available"}
    # the fallback is the first option along the chain that survives the same filters
    both = c.post("/v1/decide", json={"state": state, "capabilities": ["vector", "graph"]}).json()
    assert both["plan"]["contracts"]["route"]["fallback"] == "hybrid"
    assert both["plan"]["contracts"]["route"]["fallback_contract"] == ROUTES["route"]["hybrid"]
    narrowed = c.post("/v1/decide", json={"state": state, "capabilities": ["vector", "graph"],
                                          "allowed": {"route": ["vector", "graph"]}}).json()
    assert narrowed["plan"]["contracts"]["route"]["fallback"] == "graph"
    hybrid = {x["option"]: x for x in narrowed["heads"]["route"]["considered"]}["hybrid"]
    assert hybrid == {"option": "hybrid", "kept": False, "reason": "not in the request's allowed options"}
    rich = ["vector", "graph", "llm_large"]
    for extra, reason in (({"sensitivity": "restricted"},
                           "may not receive restricted data (allowed: public, internal)"),
                          ({"sensitivity": "internal", "latency_budget_ms": 1000},
                           "expected latency 4000 ms exceeds the budget 1000 ms")):
        r = c.post("/v1/decide", json={"state": state, "capabilities": rich, **extra}).json()
        considered = {x["option"]: x for x in r["heads"]["response_type"]["considered"]}
        assert considered["summarize"] == {"option": "summarize", "kept": False, "reason": reason}, extra
    fits = c.post("/v1/decide", json={"state": state, "capabilities": rich, "sensitivity": "internal"}).json()
    assert all(x["kept"] for x in fits["heads"]["response_type"]["considered"])
    # what is available right now is part of the decision's contract, not of its state
    assert only_vector["receipt"]["state_digest"] == both["receipt"]["state_digest"]
    assert only_vector["receipt"]["questions_digest"] != both["receipt"]["questions_digest"]
    for bad in ({"capabilities": "vector"}, {"sensitivity": 3}, {"latency_budget_ms": 0}):
        r = c.post("/v1/decide", json={"state": state, **bad})
        assert r.status_code == 422 and r.json()["field"] == next(iter(bad)), bad
    broken = http(docs.make(routes={"route": {"teleport": {}}}))
    r = broken.post("/v1/decide", json={"state": state})
    assert r.status_code == 500 and r.json()["kind"] == "bad_config" and "teleport" in r.json()["error"]


def test_a_head_with_every_option_removed_escalates_the_plan(docs, case):
    svc = docs.make(routes=ROUTES)
    d = http(svc).post("/v1/decide", json={"state": {"query": case.query}, "capabilities": ["graph"],
                                           "allowed": {"route": ["vector"]}}).json()
    route = d["heads"]["route"]
    assert {x["option"]: x["reason"] for x in route["considered"]} == {
        "vector": "needs vector, which the request does not list as available",
        "graph": "not in the request's allowed options",
        "hybrid": "needs vector, which the request does not list as available",
        "none": "disabled by configuration"}
    assert route["decision"]["class"] is None and route["decision"]["tier"] == "escalate" and route["contract"] is None
    assert route["answer"]["allowed"] == [] and route["answer"]["excluded_mass"] == 1.0
    assert route["uncertainty"]["sources"] == ["options_removed"]
    plan = d["plan"]
    assert d["action"] == "escalate" and plan["route"] is None and plan["next"] == "escalate_to_human"
    assert plan["call_llm"] is False and "route" not in plan.get("contracts", {})
    assert plan["reason"] == "no allowed option for route survives the deterministic filters: a person decides"
    assert any("head 'route': deterministic filters removed every option" in w for w in d["warnings"])
    row = svc.receipt_view(d["decision_id"])
    assert row["heads"]["route"]["class"] is None and row["action"] == "escalate"
    assert row["plan"]["next"] == "escalate_to_human"


# ----------------------------------------------------------------------------- decide: evidence

def test_return_evidence_gives_exactly_the_chunks_the_plan_kept(docs, case):
    ev = case.plan["evidence"]
    assert ev and [e["id"] for e in ev] == case.plan["plan"]["use_chunks"]
    for e in ev:
        assert set(e) <= set(EVIDENCE_FIELDS) and {"id", "doc_id", "title", "text", "updated_at"} <= set(e)
        chunk = docs.corpus.chunk(e["id"])
        assert (e["text"], e["title"], e["doc_id"], e["updated_at"]) == (chunk["text"], chunk["title"],
                                                                        chunk["doc_id"], chunk["updated_at"])
    plain = docs.client.post("/v1/decide", json={"state": {"query": case.query}}).json()
    assert "evidence" not in plain and plain["plan"]["use_chunks"] == case.plan["plan"]["use_chunks"]


def test_the_receipt_binds_the_plan_and_its_evidence_by_digest(docs, case):
    row = docs.svc.receipt_view(case.plan["decision_id"])
    assert row["kind"] == "decide" and row["action"] == case.plan["action"] == "act"
    assert row["plan"] == case.plan["plan"]
    assert [e["id"] for e in row["evidence"]] == case.plan["plan"]["use_chunks"]
    for logged, e in zip(row["evidence"], case.plan["evidence"], strict=True):
        assert logged["digest"] == digest({"id": e["id"], "title": e.get("title") or "", "text": e.get("text") or ""})
    assert row["receipt"]["state_digest"] == case.plan["receipt"]["state_digest"] == digest({"query": case.query})
    assert set(row["heads"]) == {"response_type", "route", "needs_graph", "chunk_relevant", "answerable"}
    assert [c["id"] for c in row["heads"]["answerable"]["state"]["evidence"]] == case.plan["plan"]["use_chunks"]
    url = f"/v1/receipts/{case.plan['decision_id']}/check"
    same = docs.client.post(url, json={"state": {"query": case.query}}).json()
    assert same["current"] is True and same["checks"]["automation"] == "allowed"
    other = docs.client.post(url, json={"state": {"query": case.query + " Thanks!"}}).json()
    assert other["current"] is False and other["checks"]["state"] == "changed"


def test_stale_evidence_caps_answerable_at_confirm(docs, case):
    state = {"query": case.query}
    fresh = docs.make(pack_config={"max_evidence_age_days": 100000}).decide(DecideRequest(state=state))
    stale = docs.make(pack_config={"max_evidence_age_days": 30}).decide(DecideRequest(state=state))
    fa, sa = fresh["heads"]["answerable"], stale["heads"]["answerable"]
    assert fresh["action"] == fa["decision"]["tier"] == "act" and "caps" not in fa["decision"]
    assert fa["evidence_age"]["max_age_days"] == 100000 and fa["evidence_age"]["undated"] == []
    assert stale["action"] == sa["decision"]["tier"] == "confirm"
    caps = sa["decision"]["caps"]
    assert [(x["from"], x["to"]) for x in caps] == [("act", "confirm")]
    assert "max_evidence_age_days=30" in caps[0]["reason"]
    assert any("evidence older than max_evidence_age_days=30" in w for w in stale["warnings"])
    assert set(sa["evidence_age"]["age_days"]) == set(stale["plan"]["use_chunks"])
    assert all(age > 30 for age in sa["evidence_age"]["age_days"].values())
    assert stale["plan"]["next"] == "answer" and stale["plan"]["call_llm"] is True   # it answers, a person confirms


def test_undated_evidence_caps_answerable_when_dates_are_required(docs, case):
    cands = [{"id": e["id"], "title": e["title"], "text": e["text"]} for e in case.plan["evidence"]]
    d = docs.make(pack_config={"require_evidence_dates": True}).decide(
        DecideRequest(state={"query": case.query, "candidates": cands}, min_relevance=0.0))
    ans = d["heads"]["answerable"]
    assert d["plan"]["use_chunks"] and set(ans["evidence_age"]["undated"]) == set(d["plan"]["use_chunks"])
    assert ans["decision"]["tier"] != "act" and d["action"] != "act"
    assert any("evidence without updated_at" in w for w in d["warnings"])


# ----------------------------------------------------------------------------- evaluate: the decision table

def test_failed_checks_reject_without_asking_the_heads(docs, case):
    checks = {"schema": True, "links": {"passed": False, "detail": "404 on /x"}}
    e = evaluate(docs.client, {"state": case.state, "stakes": "low", "checks": checks})
    assert (e["verdict"], e["decided_by"], e["next"], e["action"]) == ("reject", "checks", "escalate_to_human",
                                                                       "escalate")
    assert e["heads"] == {} and "revise" not in e and any("links: 404 on /x" in w for w in e["warnings"])
    row = docs.svc.receipt_view(e["decision_id"])
    assert row["decided_by"] == "checks" and row["failed"] == [] and row["heads"] == {}
    assert row["receipt"]["audit"]["checks"] == checks
    passing = evaluate(docs.client, {"state": case.state, "stakes": "low", "checks": {"schema": True}})
    assert passing["verdict"] == "ship" and passing["decided_by"] == "heads"
    bad = docs.client.post("/v1/evaluate", json={"state": case.state, "checks": {"schema": "yes"}})
    assert bad.status_code == 422 and bad.json()["field"] == "checks.schema"


def test_a_partial_or_missing_artifact_asks_for_a_new_generation(docs, case):
    c = docs.client
    for status, needle in (("partial", "the answer is partial (finish_reason length)"),
                           ("none", "no answer was produced (finish_reason length)")):
        e = evaluate(c, {"state": case.state, "stakes": "low", "artifact": {
            "status": status, "finish_reason": "length", "model": "gen-1", "provider": "local"}})
        assert (e["verdict"], e["decided_by"], e["next"]) == ("reject", "artifact", "retry_generation")
        assert e["heads"] == {} and "revise" not in e and any(needle in w for w in e["warnings"])
        assert docs.svc.receipt_view(e["decision_id"])["receipt"]["audit"]["artifact"]["status"] == status
    unanswered = {k: v for k, v in case.state.items() if k != "answer"}
    assert evaluate(c, {"state": unanswered, "artifact": {"status": "none"}})["next"] == "retry_generation"
    complete = evaluate(c, {"state": case.state, "stakes": "low", "artifact": {"status": "complete", "model": "gen-1"}})
    assert complete["verdict"] == "ship"
    for artifact in ({"status": "half"}, {"status": "complete", "latency_ms": "slow"}, "complete"):
        r = c.post("/v1/evaluate", json={"state": case.state, "artifact": artifact})
        assert r.status_code == 422 and r.json()["field"].startswith("artifact"), artifact
    both = evaluate(c, {"state": case.state, "checks": {"tests": False}, "artifact": {"status": "partial"}})
    assert both["decided_by"] == "checks" and both["next"] == "escalate_to_human"     # failed checks come first


def test_citations_outside_the_evidence_reject(docs, case):
    c = docs.client
    ids = [e["id"] for e in case.state["evidence"]]
    out = evaluate(c, {"state": case.state, "stakes": "low", "citations": [ids[0], "invented#9"]})
    assert (out["verdict"], out["decided_by"], out["next"]) == ("reject", "citations", "escalate_to_human")
    assert out["heads"] == {} and any("['invented#9']" in w for w in out["warnings"])
    inside = evaluate(c, {"state": case.state, "stakes": "low", "citations": ids[:1]})
    assert inside["verdict"] == "ship" and inside["decided_by"] == "heads"
    unknown = evaluate(c, {"state": case.state, "stakes": "low", "citations": None})
    assert unknown["verdict"] == "ship" and any("no citations" in w for w in unknown["warnings"])
    bad = c.post("/v1/evaluate", json={"state": case.state, "citations": "e0"})
    assert bad.status_code == 422 and bad.json()["field"] == "citations"


def test_plan_binding_caps_a_review_of_other_evidence_at_review(docs, case):
    c = docs.client
    pid = case.plan["decision_id"]
    same = evaluate(c, {"state": case.state, "stakes": "low", "plan": pid})
    assert same["verdict"] == "ship" and not any(NOT_SAME_EVIDENCE in w for w in same["warnings"])
    assert docs.svc.receipt_view(same["decision_id"])["receipt"]["audit"]["plan"] == pid
    ev = case.state["evidence"]
    fewer = evaluate(c, {"state": {**case.state, "evidence": ev[:1]}, "stakes": "low", "plan": pid})
    assert not any(NOT_SAME_EVIDENCE in w for w in fewer["warnings"])      # a subset is the same evidence
    changed = [{**ev[0], "text": ev[0]["text"] + " Discontinued."}, *ev[1:]]
    extra = [*ev, {"id": "notes#1", "text": "Internal notes about something else."}]
    for evidence, why in ((changed, f"evidence changed since the plan: ['{ev[0]['id']}']"),
                          (extra, "evidence the plan did not select: ['notes#1']")):
        e = evaluate(c, {"state": {**case.state, "evidence": evidence}, "stakes": "low", "plan": pid})
        assert e["verdict"] != "ship" and e["action"] != "act", why
        assert any(NOT_SAME_EVIDENCE in w and why in w for w in e["warnings"]), e["warnings"]
    unknown = c.post("/v1/evaluate", json={"state": case.state, "plan": "nope"})
    assert unknown.status_code == 422 and unknown.json()["kind"] == "unknown_decision"
    not_a_plan = c.post("/v1/evaluate", json={"state": case.state, "plan": same["decision_id"]})
    assert not_a_plan.status_code == 422 and not_a_plan.json()["field"] == "plan"
    assert c.post("/v1/evaluate", json={"state": case.state, "plan": 7}).json()["field"] == "plan"


def test_generated_evidence_caps_the_verdict_at_review(docs, case):
    ev = [*case.state["evidence"], {"id": "llm#1", "text": "A summary another model wrote.", "generated": True}]
    e = evaluate(docs.client, {"state": {**case.state, "evidence": ev}, "stakes": "low"})
    assert e["verdict"] != "ship" and e["action"] != "act"
    assert any("evidence ['llm#1'] is marked generated" in w for w in e["warnings"])
    sourced = [*case.state["evidence"], {"id": "llm#1", "text": "A summary another model wrote.", "generated": False}]
    s = evaluate(docs.client, {"state": {**case.state, "evidence": sourced}, "stakes": "low"})
    assert not any("marked generated" in w for w in s["warnings"])


def test_the_revision_loop_is_finite(docs, case):
    c = docs.client
    bad = {**case.state, "answer": OFF_TOPIC}
    first = evaluate(c, {"state": bad, "stakes": "low"})
    assert (first["verdict"], first["decided_by"], first["next"]) == ("reject", "heads", "revise")
    rv = first["revise"]
    assert [x["property"] for x in rv["failed"]] == ["grounded", "complete"]
    assert rv["evidence_ids"] == [e["id"] for e in case.state["evidence"]]
    assert (rv["attempt"], rv["attempts_left"]) == (1, 2)
    assert rv["keep"].startswith("answer only from the listed evidence ids")
    grounded, complete = rv["failed"]
    spec = docs.svc.spec("grounded", docs.svc.resolve(None))
    assert grounded["meaning"] == spec.positive and grounded["tier"] == "escalate" and grounded["p"] < 0.5
    assert complete["needed"] == 3 and complete["level"] < 3 and complete["meaning"]
    row = docs.svc.receipt_view(first["decision_id"])
    assert (row["failed"], row["next"], row["verdict"]) == (["grounded", "complete"], "revise", "reject")
    # asking again over exactly the same state is not a recovery
    again = evaluate(c, {"state": bad, "stakes": "low", "revision": {"of": first["decision_id"]}})
    assert (again["verdict"], again["decided_by"], again["next"]) == ("reject", "revision", "escalate_to_human")
    assert again["revision"] == {"of": first["decision_id"], "attempt": 2, "max_revisions": 2}
    assert any("sent exactly the state of evaluation" in w for w in again["warnings"]) and "revise" not in again
    # the same property failing in two consecutive attempts stops the loop
    twice = evaluate(c, {"state": {**bad, "answer": "Our office is closed on Sundays."}, "stakes": "low",
                         "revision": {"of": first["decision_id"]}})
    assert (twice["decided_by"], twice["next"]) == ("revision", "escalate_to_human")
    assert any("failed in two consecutive attempts" in w and "grounded" in w for w in twice["warnings"])
    # a revision that fixes the answer ships, and its receipt says which evaluation it revises
    fixed = evaluate(c, {"state": case.state, "stakes": "low", "revision": {"of": first["decision_id"]}})
    assert (fixed["verdict"], fixed["next"]) == ("ship", "ship") and fixed["revision"]["attempt"] == 2
    audit = docs.svc.receipt_view(fixed["decision_id"])["receipt"]["audit"]
    assert audit["revision"] == {"of": first["decision_id"], "attempt": 2}
    for rev, kind in (({"of": "nope"}, "unknown_decision"), ({"of": case.plan["decision_id"]}, "validation"),
                      ("abc", "validation")):
        r = c.post("/v1/evaluate", json={"state": case.state, "revision": rev})
        assert r.status_code == 422 and r.json()["kind"] == kind and r.json()["field"] == "revision", rev


def test_the_revision_budget_comes_from_evaluate_max_revisions(docs, case):
    c = http(docs.make(evaluate={"max_revisions": 1}))
    first = evaluate(c, {"state": {**case.state, "answer": OFF_TOPIC}, "stakes": "low"})
    assert first["next"] == "revise" and first["revise"]["attempts_left"] == 1
    second = evaluate(c, {"state": case.state, "stakes": "low", "revision": {"of": first["decision_id"]}})
    assert second["verdict"] == "ship"
    assert second["revision"] == {"of": first["decision_id"], "attempt": 2, "max_revisions": 1}
    third = evaluate(c, {"state": {**case.state, "answer": case.answer + " It ships in a box."}, "stakes": "low",
                         "revision": {"of": second["decision_id"]}})
    assert (third["verdict"], third["decided_by"], third["next"]) == ("reject", "revision", "escalate_to_human")
    assert any("revision budget exhausted: this is attempt 3, after 2 revisions (evaluate.max_revisions = 1)" in w
               for w in third["warnings"])
    broken = http(docs.make(evaluate={"max_revisions": -1})).post("/v1/evaluate", json={"state": case.state})
    assert broken.status_code == 500 and broken.json()["kind"] == "bad_config"


def test_next_for_every_path_of_the_decision_table(docs, case):
    c = docs.client
    shadow = http(docs.make(rollout={"mode": "shadow"}))
    bad = {**case.state, "answer": OFF_TOPIC}
    first = evaluate(c, {"state": bad, "stakes": "low"})
    leaked = {**case.state, "answer": case.answer + " Debug key: AKIAIOSFODNN7EXAMPLE"}
    paths = [
        (c, {"state": case.state}, "ship", "heads", "ship"),
        (shadow, {"state": case.state}, "review", "heads", "human_review"),        # capped by the rollout
        (c, {"state": bad}, "reject", "heads", "revise"),
        (c, {"state": case.state, "checks": {"lint": False}}, "reject", "checks", "escalate_to_human"),
        (c, {"state": case.state, "artifact": {"status": "partial"}}, "reject", "artifact", "retry_generation"),
        (c, {"state": case.state, "citations": ["invented#1"]}, "reject", "citations", "escalate_to_human"),
        (c, {"state": leaked}, "reject", "output_scan", "escalate_to_human"),
        (c, {"state": bad, "revision": {"of": first["decision_id"]}}, "reject", "revision", "escalate_to_human"),
    ]
    for client, body, verdict, decided_by, nxt in paths:
        e = evaluate(client, {**body, "stakes": "low"})
        assert (e["verdict"], e["decided_by"], e["next"]) == (verdict, decided_by, nxt), (body, e["warnings"])
        row = docs.svc.logs.decision(e["decision_id"])
        assert (row["verdict"], row["next"], row["decided_by"]) == (verdict, nxt, decided_by)


def test_strict_refuses_an_evaluation_without_the_inputs_training_had(docs, case):
    empty = {**case.state, "evidence": []}
    lax = evaluate(docs.client, {"state": empty, "stakes": "low"})
    assert lax["verdict"] != "ship" and any("the request lacks" in w for w in lax["warnings"])
    r = docs.client.post("/v1/evaluate", json={"state": empty, "stakes": "low", "strict": True})
    body = r.json()
    assert r.status_code == 422 and body["kind"] == "missing_evidence" and body["missing"]
    assert body["head"] in EVALUATE_HEADS and body["retryable"] is False
    assert evaluate(docs.client, {"state": case.state, "stakes": "low", "strict": True})["verdict"] == "ship"
    ok = docs.client.post("/v1/decide", json={"state": {"query": case.query}, "strict": True})
    assert ok.status_code == 200 and ok.json()["action"] == "act"


# ----------------------------------------------------------------------------- known defects (strict xfail)

def test_next_is_not_revise_when_no_revision_is_left(docs, case):
    c = http(docs.make(evaluate={"max_revisions": 0}))
    e = evaluate(c, {"state": {**case.state, "answer": OFF_TOPIC}, "stakes": "low"})
    assert e["verdict"] == "reject" and e["revise"]["attempts_left"] == 0
    follow = evaluate(c, {"state": case.state, "stakes": "low", "revision": {"of": e["decision_id"]}})
    assert follow["decided_by"] == "revision"                # following `next` can only be rejected
    assert e["next"] == "escalate_to_human"


def test_shadow_mode_reports_the_action_the_plan_would_have_taken(docs, case):
    shadow = docs.make(rollout={"mode": "shadow"})
    d = shadow.decide(DecideRequest(state={"query": case.query}))
    assert case.plan["action"] == "act" and d["action"] == "confirm" and d["rollout"]["capped"] is True
    assert d["rollout"]["proposed_action"] == case.plan["action"]
    e = evaluate(http(shadow), {"state": case.state, "stakes": "low"})
    assert case.review["action"] == "act" and e["action"] == "confirm"
    assert e["rollout"]["proposed_action"] == case.review["action"]
