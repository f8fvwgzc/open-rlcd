"""The documents pack end to end, offline: synthetic corpus + labels -> train all eight heads -> decide / evaluate /
rank / feedback over HTTP -> label tasks -> bundle reload. One trained service is shared by the module."""
from __future__ import annotations

import json
import random
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient

from open_rlcd.core.config import Project
from open_rlcd.core.contract import ContractError, DecideRequest, EvaluateRequest, TrainRequest
from open_rlcd.core.service import Service
from open_rlcd.packs.documents import DocumentsPack, write_synth
from open_rlcd.packs.documents.synth import corrupt_value
from open_rlcd.packs.documents.world import build_world
from open_rlcd.serve.api import create_app

ENC = {"kind": "hashing", "dim": 256}
HEADS = ("response_type", "route", "needs_graph", "chunk_relevant", "answerable", "grounded", "complete", "risk_tier")
LEARNABLE = ("response_type", "route", "chunk_relevant", "answerable", "grounded")


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    root = tmp_path_factory.mktemp("documents")
    home = root / "var"
    mp = pytest.MonkeyPatch()
    # packs read the label store and the corpus from project.home, which RLCD_HOME (set by conftest) overrides
    mp.setenv("RLCD_HOME", str(home))
    info = write_synth(home, ENC, seed=7, n_queries=900)
    project = Project(root=root, data={"pack": "documents", "pack_config": {"encoder": ENC}})
    svc = Service(project, home=home)
    job = svc.trainer.start(TrainRequest(), wait=True)
    yield SimpleNamespace(root=root, home=home, info=info, project=project, svc=svc, job=job, world=build_world(7),
                          client=TestClient(create_app(svc)))
    mp.undo()


def corpus(t):
    return t.svc.pack.corpus(t.project, t.svc.provider(t.svc.resolve(None)))


# ----------------------------------------------------------------------------- data and training

def test_every_head_has_300_trusted_rows_of_every_class(trained):
    assert trained.job["status"] == "succeeded", trained.job.get("error")
    specs = DocumentsPack().heads(trained.project)
    assert tuple(specs) == HEADS
    for head in HEADS:
        counts = trained.info["labels"][head]
        assert counts["rows"] >= 300, (head, counts)
        assert len(counts["classes"]) == len(specs[head].labels), (head, counts)
        data = trained.job["heads"][head]["data"]
        assert data["trusted"] >= 300 and data["by_source"] == {"synthetic": data["rows"]}
        assert data["ordering"] == "group"


def test_learnable_heads_show_skill_and_calibration(trained):
    heads = trained.job["heads"]
    for name in LEARNABLE:
        m = heads[name]["metrics"]
        assert m["has_skill"], (name, m)
        assert m["log_loss"] < m["baseline_log_loss"] and m["brier_skill_score"] > 0.1, (name, m)
        assert m["ece"] < (0.2 if name in ("response_type", "route") else 0.1), (name, m)
    for name in HEADS:
        h = heads[name]
        assert h["status"] == "trained"
        for v in h["verdict"].values():
            assert v["reason"]
            if not h["metrics"]["has_skill"]:          # the honest path: no skill -> abstain, saying why
                assert v["max_tier"] == "escalate" and "no skill" in v["reason"]
    # the gatekeeper either keeps its 5% conformal promise or is capped at confirm with the reason
    ans = heads["answerable"]
    if ans["metrics"].get("conformal_threshold") is None:
        assert ans["verdict"]["medium"]["max_tier"] != "act" and ans["verdict"]["medium"]["reason"]
    grounded = heads["grounded"]
    assert grounded["metrics"]["accuracy"] > 0.85
    if grounded["metrics"].get("conformal_threshold") is None:   # promise not establishable: capped, says why
        assert all(v["max_tier"] != "act" and v["reason"] for v in grounded["verdict"].values())


# ----------------------------------------------------------------------------- decide

def test_decide_returns_a_plan_and_logs_heads_that_accept_feedback(trained):
    r = trained.client.post("/v1/decide", json={"state": {"query": "What is the price of the Velobox 300?"}})
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["model"].startswith("rlcd-0.1.") and d["action"] in ("act", "confirm", "escalate")
    assert d["candidates_from"] == "corpus" and d["usage"]["input_tokens"] == 0 and d["usage"]["output_tokens"] == 0
    plan = d["plan"]
    assert plan["response_type"] == "direct_answer" and plan["route"] == "vector" and plan["needs_graph"] is False
    assert plan["use_chunks"][0] == "products/velobox-300#0" and len(plan["use_chunks"]) <= 5
    assert plan["next"] in ("answer", "retrieve_more", "escalate_to_human")
    assert plan["call_llm"] == (d["action"] in ("act", "confirm"))
    assert set(d["heads"]) == {"response_type", "route", "needs_graph", "chunk_relevant", "answerable"}
    for h in d["heads"].values():
        assert h["answer"]["calibrated"] is True and "decision" in h
    assert d["heads"]["response_type"]["reasons"]
    assert [x["rank"] for x in d["ranked"]] == list(range(1, len(d["ranked"]) + 1))
    assert d["ranked"][0]["id"] == "products/velobox-300#0"
    labels = {"response_type": "direct_answer", "route": "vector", "needs_graph": False, "answerable": True,
              "chunk_relevant": True}
    for head, label in labels.items():
        fb = trained.client.post("/v1/feedback", json={"decision_id": d["decision_id"], "head": head, "label": label,
                                                       "source": "test"})
        assert fb.status_code == 200, (head, fb.text)
        assert fb.json()["stored"] and fb.json()["bernoulli"]["updated"] in (True, False)


def test_out_of_scope_and_clarify_plans(trained):
    oos = trained.svc.decide(DecideRequest(state={"query": "What's the weather in Lisbon tomorrow?"}))
    assert oos["plan"]["response_type"] == "out_of_scope" and oos["plan"]["next"] == "refuse_out_of_scope"
    assert oos["plan"]["call_llm"] is False and "answerable" not in oos["plan"]["chain"]
    cl = trained.svc.decide(DecideRequest(state={"query": "What does it cost?"}))
    assert cl["plan"]["response_type"] == "clarify_question" and cl["plan"]["next"] == "ask_clarifying_question"
    hop = trained.svc.decide(DecideRequest(state={"query": "Who leads the team that owns the Velobox 300?"}))
    assert hop["plan"]["route"] == "graph" and hop["plan"]["needs_graph"] is True


def test_decide_ranks_gold_chunks_above_distractors(trained):
    gold_p, other_p, gold_rank = [], [], []
    for p in trained.world.products:
        team = next(t for t in trained.world.teams if t.subject == p.meta["team"])
        for q, gold in ((f"What is the warranty period of the {p.subject}?", {p.chunk_id}),
                        (f"Who leads the team that owns the {p.subject}?", {p.chunk_id, team.chunk_id})):
            d = trained.svc.decide(DecideRequest(state={"query": q}, heads=["chunk_relevant"], top_k=3))
            for x in d["ranked"]:
                (gold_p if x["id"] in gold else other_p).append(x["probability"])
                if x["id"] == p.chunk_id:
                    gold_rank.append(x["rank"])
    assert np.mean(gold_p) > np.mean(other_p) + 0.3, (np.mean(gold_p), np.mean(other_p))
    assert np.median(gold_rank) == 1 and np.mean(gold_rank) < 1.6


def test_decide_with_given_candidates_and_bad_requests(trained):
    c = corpus(trained)
    cands = [c.chunk("products/velobox-300#0"), c.chunk("products/arcternix-hub#0"), "A plain-text chunk."]
    d = trained.svc.decide(DecideRequest(state={"query": "What is the SKU of the Velobox 300?", "candidates": cands},
                                         min_relevance=0.0, top_k=2))
    assert d["candidates_from"] == "request" and len(d["plan"]["use_chunks"]) == 2
    assert {x["id"] for x in d["ranked"]} == {"products/velobox-300#0", "products/arcternix-hub#0", "c2"}
    for body, needle in (({"state": {"candidates": []}}, "query is required"),
                         ({"state": "hi"}, "state must be an object"),
                         ({"state": {"query": "q"}, "top_k": 0}, "top_k"),
                         ({"state": {"query": "q"}, "min_relevance": 2}, "min_relevance"),
                         ({"state": {"query": "q"}, "heads": ["grounded"]}, "subset"),
                         ({"state": {"query": "q", "candidates": [{"id": i} for i in range(201)]}}, "at most 200")):
        r = trained.client.post("/v1/decide", json=body)
        assert r.status_code == 422 and needle in r.json()["error"], (body, r.text)


def test_rank_without_a_question_uses_chunk_relevant(trained):
    c = corpus(trained)
    ids = ["news/spring-town-hall#0", "products/arcternix-hub#0", "products/velobox-300#0"]
    body = {"state": {"query": "How much does the Velobox 300 weigh?"},
            "candidates": [{"id": i, "state": {"chunk": c.chunk(i)}} for i in ids]}
    r = trained.client.post("/v1/rank", json=body)
    assert r.status_code == 200, r.text
    out = r.json()
    assert out["head"] == "chunk_relevant" and out["candidates"][0]["id"] == "products/velobox-300#0"
    ps = [x["probability"] for x in out["candidates"]]
    assert ps == sorted(ps, reverse=True)
    flat = {"state": {"query": body["state"]["query"]},
            "candidates": [{"id": i, "state": {k: v for k, v in c.chunk(i).items() if k in ("id", "title", "text")}}
                           for i in ids]}
    r2 = trained.client.post("/v1/rank", json=flat)
    assert r2.status_code == 200 and r2.json()["candidates"][0]["id"] == "products/velobox-300#0"


# ----------------------------------------------------------------------------- evaluate

def test_evaluate_flags_changed_numbers(trained):
    c = corpus(trained)
    rng = random.Random(1)
    good, bad, verdicts = [], [], {"good": [], "bad": []}
    for p in trained.world.products:
        ev = [c.chunk(p.chunk_id)]
        f = p.facts["price"]
        q = f"What is the price of the {p.subject}?"
        for kind, value in (("good", f.value["en"]), ("bad", corrupt_value(f, "en", rng, trained.world, p.text))):
            r = trained.svc.evaluate(EvaluateRequest(state={"query": q, "evidence": ev,
                                                            "answer": f"The price of {p.subject} is {value}."}))
            (good if kind == "good" else bad).append(r["heads"]["grounded"]["answer"]["noul"])
            verdicts[kind].append(r["verdict"])
            assert r["verdict"] in ("ship", "review", "reject") and set(r["heads"]) == {"grounded", "complete",
                                                                                       "risk_tier"}
            assert r["usage"]["output_tokens"] == 0 and r["decision_id"]
    assert np.mean(good) > np.mean(bad) + 0.4, (np.mean(good), np.mean(bad))
    assert sum(p < 0.5 for p in bad) > sum(p < 0.5 for p in good)
    assert verdicts["bad"].count("reject") > verdicts["good"].count("reject")
    r = trained.client.post("/v1/evaluate", json={"state": {"query": "q", "evidence": []}})
    assert r.status_code == 422 and "answer is required" in r.json()["error"]


def test_evaluate_stakes_follow_risk_and_feedback_names_its_heads(trained):
    c = corpus(trained)
    ev = [c.chunk("policies/hazardous-goods#0")]
    hz = trained.world.docs["policies/hazardous-goods"].facts["ppe"]
    r = trained.svc.evaluate(EvaluateRequest(state={
        "query": "What protective equipment is required when handling hazardous goods?", "evidence": ev,
        "answer": f"Staff must wear {hz.value['en']}."}))
    assert r["heads"]["risk_tier"]["answer"]["score"] == 3 and r["stakes"] == "high"
    assert r["stakes_from"] == "risk_tier" and r["action"] != "act"
    forced = trained.svc.evaluate(EvaluateRequest(state={"query": "q", "evidence": ev, "answer": "a"}, stakes="low"))
    assert forced["stakes"] == "low" and forced["stakes_from"] == "request"
    fb = trained.client.post("/v1/feedback", json={"decision_id": r["decision_id"], "head": "grounded", "label": True,
                                                   "source": "reviewer"})
    assert fb.status_code == 200


# ----------------------------------------------------------------------------- untrained, label tasks, reload

def test_untrained_heads_escalate_and_never_crash(tmp_path, monkeypatch):
    monkeypatch.setenv("RLCD_HOME", str(tmp_path / "var"))
    svc = Service(Project(root=tmp_path, data={"pack": "documents", "pack_config": {"encoder": ENC}}),
                  home=tmp_path / "var")
    d = svc.decide(DecideRequest(state={"query": "What is the price of the Velobox 300?",
                                        "candidates": ["The Velobox 300 sells for 1,250 USD.", "Other text."]}))
    assert d["action"] == "escalate" and d["plan"]["call_llm"] is False
    assert d["plan"]["next"] == "escalate_to_human" and len(d["plan"]["use_chunks"]) == 2
    assert any("not trained" in w for w in d["warnings"])
    assert all(h["answer"]["calibrated"] is False for h in d["heads"].values())
    nothing = svc.decide(DecideRequest(state={"query": "anything"}))
    assert nothing["candidates_from"] == "none" and any("no corpus" in w for w in nothing["warnings"])
    e = svc.evaluate(EvaluateRequest(state={"query": "q", "evidence": ["e"], "answer": "a"}))
    assert e["verdict"] == "reject" and e["stakes"] == "high" and e["stakes_from"] == "fail_safe"


def test_label_tasks_over_a_jsonl_query_log(trained, tmp_path):
    log = tmp_path / "queries.jsonl"
    rows = [{"query": "What is the price of the Velobox 300?", "time": "2025-05-01T10:00:00Z", "group": "g1"},
            {"query": "Who leads the team that owns the Arcternix Hub?", "answer": "Daniel Reyes leads it.",
             "evidence": ["products/arcternix-hub#0", {"id": "note", "text": "Team notes."}, "free text"]},
            {"not_a_query": 1}]
    log.write_text("\n".join(json.dumps(r) for r in rows) + "\nnot json\n", encoding="utf-8")
    project = Project(root=tmp_path, data={"pack": "documents", "pack_config": {"encoder": ENC,
                                                                                 "queries": str(log)}})
    pack = DocumentsPack()
    items = pack.label_tasks(project)
    views = [i.view for i in items]
    assert views.count("query") == 2 and views.count("query_evidence_answer") == 1
    assert views.count("query_evidence") == 3 and 2 <= views.count("query_chunk") <= 10
    first = [i for i in items if i.meta["row"] == 0]
    assert all(i.group == "g1" and i.t is not None for i in first)
    qea = next(i for i in items if i.view == "query_evidence_answer")
    assert [e["id"] for e in qea.state["evidence"]] == ["products/arcternix-hub#0", "note", "evidence:2"]
    assert "Drafted answer:" in qea.display and "Evidence:" in qea.display and qea.display.startswith("Request:")
    assert [i.id for i in pack.label_tasks(project)] == [i.id for i in items]          # stable ids
    provider = pack.provider(project)
    for view in ("query", "query_chunk", "query_evidence", "query_evidence_answer"):
        states = [i.state for i in items if i.view == view]
        assert provider.transform(view, states).shape[0] == len(states)
    with pytest.raises(ContractError, match="pack_config.queries"):
        pack.label_tasks(Project(root=tmp_path, data={"pack": "documents", "pack_config": {"encoder": ENC}}))


def test_reload_in_a_fresh_service_and_from_an_exported_bundle(trained, tmp_path):
    svc = trained.svc
    version = svc.resolve(None)
    c = corpus(trained)
    ev = [c.chunk("products/velobox-300#0")]
    states = {"response_type": [{"query": "List all products owned by the Cold Chain team."}],
              "chunk_relevant": [{"query": "Velobox 300 price?", "chunk": ev[0]}],
              "answerable": [{"query": "Velobox 300 price?", "evidence": ev}],
              "grounded": [{"query": "Velobox 300 price?", "evidence": ev, "answer": "It costs 99 USD."}]}
    before = {h: svc.probabilities(version, svc.spec(h, version), s)[0] for h, s in states.items()}
    fresh = Service(trained.project, home=trained.home)
    dest = svc.registry.export(version, tmp_path / "bundle")
    other = Service(Project(root=tmp_path, data={"pack": "documents"}), home=tmp_path / "other")
    vid = other.registry.import_(dest)
    for h, s in states.items():
        p1, cal = fresh.probabilities(version, fresh.spec(h, version), s)
        p2, _ = other.probabilities(vid, other.spec(h, vid), s)
        assert cal
        np.testing.assert_allclose(p1, before[h], atol=1e-9)
        np.testing.assert_allclose(p2, before[h], atol=1e-9)
    assert not list(dest.rglob("*.pkl")) and not list(dest.rglob("*.joblib"))


def test_ingest_a_folder_and_heads_view(trained, tmp_path, monkeypatch):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "a.md").write_text("# Returns\n\nItems can be returned within 30 days.", encoding="utf-8")
    (docs / "b.txt").write_text("The Zorvane 9 weighs 2 kg.", encoding="utf-8")
    monkeypatch.setenv("RLCD_HOME", str(tmp_path / "var"))
    project = Project(root=tmp_path, data={"pack": "documents", "pack_config": {"encoder": ENC}})
    out = DocumentsPack().ingest(project, {"path": str(docs)})
    assert out["docs_added"] == 2 and out["corpus"]["chunks"] >= 2
    with pytest.raises(ContractError, match="needs --collection"):
        DocumentsPack().ingest(project, {"path": str(docs), "qdrant": "http://localhost:6333"})
    monkeypatch.delenv("RLCD_ALLOW_REMOTE", raising=False)
    with pytest.raises(ContractError, match="refusing non-loopback host") as e:   # refused before any connection
        DocumentsPack().ingest(project, {"path": str(docs), "qdrant": "http://example.org:6333", "collection": "d"})
    assert e.value.kind == "connector_config"
    view = trained.svc.heads_view(None)
    by = {h["name"]: h for h in view["heads"]}
    assert by["answerable"]["decision"] == "gate" and by["grounded"]["stakes"] == "high"
    assert "evidence" in by["answerable"]["input"]["state"] and by["risk_tier"]["legend"]["3"].startswith("3:")


# ----------------------------------------------------------------------------- reviewed defects on the trained model

def test_decide_without_kept_evidence_retrieves_more_and_answerable_knows_the_state(trained):
    """F30 / F39: nothing passes min_relevance -> answerable judges an empty set it was trained and calibrated on,
    the plan does not answer, and no warning tells the caller to send fields it never had to send."""
    d = trained.svc.decide(DecideRequest(state={"query": "How much does standard shipping cost?"}, min_relevance=1.0))
    assert d["plan"]["use_chunks"] == [] and d["ranked"]
    assert d["action"] == "escalate" and d["plan"]["next"] == "retrieve_more" and d["plan"]["call_llm"] is False
    assert "min_relevance" in d["plan"]["reason"]
    assert d["heads"]["answerable"]["answer"]["noul"] < 0.3
    assert not any("Send the same fields" in w for w in d["warnings"]), d["warnings"]


def test_multi_hop_question_retrieves_the_team_page_in_every_language(trained):
    """F08 / F09 / F10: the team page the product sheet links to and names is among the candidates."""
    team = {t.subject: t.chunk_id for t in trained.world.teams}
    for p in trained.world.products[:6]:
        for q in (f"Who leads the team that owns the {p.subject}?", f"{p.subject}を担当するチームのリーダーは誰ですか？",
                  f"{p.subject}-г хариуцдаг багийн ахлагч хэн бэ?"):
            d = trained.svc.decide(DecideRequest(state={"query": q}, heads=["chunk_relevant"]))
            assert team[p.meta["team"]] in [x["id"] for x in d["ranked"]], q


def test_examples_readme_matches_what_the_demo_trains(trained):
    """F40: the 'What to expect' table of examples/documents/README.md is what build_demo.py prints (same seed, 900
    queries, hashing encoder): skill and verdicts exactly, accuracy within 0.03."""
    from pathlib import Path

    text = (Path(__file__).resolve().parents[1] / "examples" / "documents" / "README.md").read_text(encoding="utf-8")
    block = text.split("## What to expect", 1)[1].split("```", 2)[1]
    rows = [ln.split() for ln in block.strip().splitlines()[1:]]
    assert {r[0] for r in rows} == set(HEADS)
    for name, _status, skill, acc, _base, _ece, verdict in rows:
        h = trained.job["heads"][name]
        assert skill == ("yes" if h["metrics"]["has_skill"] else "NO"), name
        assert verdict == "/".join(h["verdict"][s]["max_tier"] for s in ("low", "medium", "high")), name
        assert abs(float(acc) - h["metrics"]["accuracy"]) < 0.03, (name, acc, h["metrics"]["accuracy"])
