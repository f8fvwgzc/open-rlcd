"""The service end to end over HTTP with the toy pack: train as a background job, answer, rank, decide, take
feedback (Bernoulli update now, a training row at the next train), and publish calibration."""
import threading
import time

import numpy as np
import pytest
from fastapi.testclient import TestClient

from open_rlcd.core.config import Project
from open_rlcd.core.dataset import Dataset
from open_rlcd.core.service import Service
from open_rlcd.serve.api import create_app
from tests.toypack import NAMES, ToyPack

# the toy generator's noul weights are fixed (seed 99 + 2): states along their direction are clear cases
_W = np.random.default_rng(99 + 2).normal(size=(6, 2))
_D = (_W[:, 1] - _W[:, 0]) / np.linalg.norm(_W[:, 1] - _W[:, 0])
POS = {"f": (2.5 * _D).round(3).tolist()}
NEG = {"f": (-2.5 * _D).round(3).tolist()}
MID = {"f": [0.0] * 6}


def make_client(path, pack=None) -> TestClient:
    svc = Service(Project(root=path), home=path, pack=pack or ToyPack())
    return TestClient(create_app(svc))


def train(client, body=None, timeout=60.0) -> dict:
    r = client.post("/v1/train", json=body or {})
    assert r.status_code == 202, r.text
    start = r.json()
    assert start["status"] in ("queued", "running") and start["poll"] == f"/v1/train/{start['job_id']}"
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = client.get(start["poll"]).json()
        if job["status"] not in ("queued", "running"):
            return job
        time.sleep(0.02)
    raise AssertionError("training did not finish in time")


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    c = make_client(tmp_path_factory.mktemp("service"))
    c.job = train(c)
    assert c.job["status"] == "succeeded", c.job
    return c


@pytest.fixture
def fresh(tmp_path):
    c = make_client(tmp_path)
    c.job = train(c)
    assert c.job["status"] == "succeeded", c.job
    return c


# ----------------------------------------------------------------------------- training jobs

def test_train_job_runs_to_completion(client):
    job = client.job
    assert job["model"] == "rlcd-0.1.1" and job["error"] is None
    assert job["progress"]["done"] == job["progress"]["total"] == 3 and job["progress"]["message"] == "done"
    assert job["started_at"] and job["finished_at"]
    for name in ("good", "color", "level"):
        h = job["heads"][name]
        assert h["status"] == "trained" and h["metrics"]["has_skill"] and set(h["verdict"]) == {"low", "medium",
                                                                                                "high"}
    heads = {h["name"]: h for h in client.get("/v1/heads").json()["heads"]}
    assert all(h["trained"] and h["model_version"] == "rlcd-0.1.1" for h in heads.values())
    assert heads["good"]["conformal_threshold"] is not None and heads["good"]["metrics"]["n_holdout"] > 0


def test_train_validation(client):
    r = client.post("/v1/train", json={"heads": ["nope"]})
    assert r.status_code == 422 and "unknown heads" in r.json()["error"]
    assert client.post("/v1/train", json={"holdout_fraction": 0.9}).json()["kind"] == "validation"
    r = client.get("/v1/train/missing")
    assert r.status_code == 404 and r.json()["kind"] == "unknown_job"


class BlockingPack(ToyPack):
    """Holds the training job inside `datasets` until the test releases it."""

    def __init__(self):
        super().__init__(n=1200)
        self.entered, self.release = threading.Event(), threading.Event()

    def datasets(self, *a, **kw):
        self.entered.set()
        assert self.release.wait(30)
        return super().datasets(*a, **kw)


def test_a_second_job_while_one_runs_is_409(tmp_path):
    pack = BlockingPack()
    c = make_client(tmp_path, pack)
    r = c.post("/v1/train", json={"heads": ["good"]})
    assert r.status_code == 202
    job_id = r.json()["job_id"]
    try:
        assert pack.entered.wait(10)
        busy = c.post("/v1/train", json={})
        assert busy.status_code == 409 and busy.json()["kind"] == "busy" and busy.json()["job_id"] == job_id
        assert c.get("/health").json()["training"] == job_id
        assert c.get(f"/v1/train/{job_id}").json()["status"] == "running"
    finally:
        pack.release.set()
    deadline = time.time() + 30
    while c.get(f"/v1/train/{job_id}").json()["status"] == "running" and time.time() < deadline:
        time.sleep(0.02)
    assert c.get(f"/v1/train/{job_id}").json()["status"] == "succeeded"
    assert c.get("/health").json()["training"] is None


class OneHeadStarved(ToyPack):
    def datasets(self, project, provider, heads, feedback=None, feedback_weight=5.0):
        out = super().datasets(project, provider, heads, feedback, feedback_weight)
        if "level" in out:
            out["level"] = out["level"].subset(np.arange(20))
        return out


def test_a_failing_head_never_costs_the_others(tmp_path):
    c = make_client(tmp_path, OneHeadStarved(n=1500))
    job = train(c)
    assert job["status"] == "partial" and job["model"] == "rlcd-0.1.1"
    assert job["heads"]["level"]["status"] == "failed" and "trusted" in job["heads"]["level"]["error"]
    assert job["heads"]["good"]["status"] == job["heads"]["color"]["status"] == "trained"
    assert "level" in job["error"]
    health = c.get("/health").json()["heads"]
    assert health == {"good": {"trained": True}, "color": {"trained": True}, "level": {"trained": False}}


# ----------------------------------------------------------------------------- answering

def test_trained_systemone(client):
    r = client.post("/v1/systemone", json={"state": POS, "questions": {
        "good": {"type": "noul"}, "c": {"type": "choice", "head": "color"},
        "lv": {"type": "score", "head": "level", "criteria": ["a", "b", "c", "d"]}}})
    assert r.status_code == 200, r.text
    body = r.json()
    a = body["answers"]
    assert body["model"] == "rlcd-0.1.1" and all(x["calibrated"] for x in a.values())
    assert a["lv"]["legend"] == {"0": "a", "1": "b", "2": "c", "3": "d"}
    pos = a["good"]["noul"]
    neg = client.post("/v1/systemone", json={"state": NEG, "questions": {"good": {"type": "noul"}}}).json()
    assert pos > 0.9 and neg["answers"]["good"]["noul"] < 0.1       # the features matter
    assert "warnings" not in body or all("not trained" not in w for w in body["warnings"])


def test_precomputed_state_and_feature_version_checks(client):
    vec = {"x": POS["f"], "feature_names": NAMES, "feature_version": "toy-v1"}
    q = {"good": {"type": "noul"}, "color": {"type": "choice"}}
    raw = client.post("/v1/systemone", json={"state": POS, "questions": q}).json()["answers"]
    pre = client.post("/v1/systemone", json={"state": vec, "questions": q})
    assert pre.status_code == 200 and pre.json()["answers"] == raw
    missing = client.post("/v1/systemone", json={"state": {**vec, "x": [None, *POS["f"][1:]]}, "questions": q})
    assert missing.status_code == 200
    assert sum(missing.json()["answers"]["color"]["probabilities"].values()) == pytest.approx(1, abs=1e-5)
    bad = client.post("/v1/systemone", json={"state": {**vec, "feature_version": "toy-v2"}, "questions": q})
    assert bad.status_code == 422 and bad.json()["kind"] == "feature_version_mismatch"
    assert bad.json()["expected"] == "toy-v1" and bad.json()["got"] == "toy-v2" and "toy-v2" in bad.json()["error"]
    short = client.post("/v1/systemone", json={"state": {**vec, "x": POS["f"][:5]}, "questions": q})
    assert short.status_code == 422 and short.json()["kind"] == "bad_feature_vector"
    names = client.post("/v1/systemone", json={"state": {**vec, "feature_names": NAMES[::-1]}, "questions": q})
    assert names.status_code == 422 and names.json()["kind"] == "feature_names_mismatch"
    nover = client.post("/v1/systemone", json={"state": {"x": POS["f"]}, "questions": q})
    assert nover.status_code == 422 and "feature_version is required" in nover.json()["error"]
    junk = client.post("/v1/systemone", json={"state": {**vec, "x": ["a"] * 6}, "questions": q})
    assert junk.status_code == 422 and junk.json()["kind"] == "bad_feature_vector"


def test_rank_sorts_and_defaults_to_the_packs_rank_head(client):
    cands = [{"id": "mid", "state": MID}, {"id": "neg", "state": NEG}, {"id": "pos", "state": POS}]
    r = client.post("/v1/rank", json={"candidates": cands})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["head"] == "good" and body["option"] is None and body["model"] == "rlcd-0.1.1"
    probs = [c["probability"] for c in body["candidates"]]
    assert probs == sorted(probs, reverse=True) and [c["rank"] for c in body["candidates"]] == [1, 2, 3]
    assert [c["id"] for c in body["candidates"]] == ["pos", "mid", "neg"]
    assert all(c["answer"]["noul"] == c["probability"] for c in body["candidates"])
    assert body["usage"]["input_tokens"] == 0 and body["usage"]["questions"] == 3
    # by one option of a choice head
    r = client.post("/v1/rank", json={"candidates": cands, "question": {"type": "choice", "head": "color"},
                                      "option": "green"})
    body = r.json()
    greens = [c["answer"]["probabilities"]["green"] for c in body["candidates"]]
    assert body["option"] == "green" and greens == sorted(greens, reverse=True)
    r = client.post("/v1/rank", json={"candidates": cands, "question": {"type": "choice", "head": "color"}})
    assert r.status_code == 422 and "needs `option`" in r.json()["error"]


def test_rank_merges_the_shared_state_and_checks_limits(client):
    shared = {"f": POS["f"]}
    r = client.post("/v1/rank", json={"state": shared, "candidates": [{"id": "a"}, {"id": "b", "state": NEG}]})
    out = {c["id"]: c["probability"] for c in r.json()["candidates"]}
    assert r.status_code == 200 and out["a"] > out["b"]           # "a" inherits the shared state
    assert client.post("/v1/rank", json={"candidates": []}).status_code == 422
    many = [{"id": str(i), "state": MID} for i in range(201)]
    r = client.post("/v1/rank", json={"candidates": many})
    assert r.status_code == 422 and r.json()["limit"] == 200


# ----------------------------------------------------------------------------- decisions

def test_decide_returns_a_decision_id_tiers_and_warnings(client):
    r = client.post("/v1/decide", json={"state": POS, "stakes": "low"})
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["decision_id"] and d["model"] == "rlcd-0.1.1" and d["explore"] is False
    assert isinstance(d["warnings"], list) and set(d["heads"]) == {"good", "color", "level"}
    tiers = [h["decision"]["tier"] for h in d["heads"].values()]
    assert d["action"] == d["tier"] == min(tiers, key=["escalate", "confirm", "act"].index)
    for h in d["heads"].values():
        dec = h["decision"]
        assert dec["stakes"] == "low" and dec["thresholds"]["act"] == 0.70 and dec["calibrated"] is True
        assert set(h["answer"]) >= {"type", "head", "calibrated", "confidence"}
        sizes = [abs(x["contribution"]) for x in h["reasons"]]
        assert sizes == sorted(sizes, reverse=True) and sum(h["reason_groups"].values()) == pytest.approx(1, abs=1e-3)
    assert d["policy"]["abstain"] == "escalate"
    assert "prediction_set" not in d["heads"]["good"]
    one = client.post("/v1/decide", json={"state": POS, "heads": ["color"], "explore": True}).json()
    assert list(one["heads"]) == ["color"] and one["explore"] is True
    assert one["heads"]["color"]["decision"]["gate"]["mode"] == "thompson_sampling"
    assert any("exploration mode" in w for w in one["warnings"])
    bad = client.post("/v1/decide", json={"state": POS, "stakes": "extreme"})
    assert bad.status_code == 422 and bad.json()["kind"] == "validation"
    assert client.post("/v1/decide", json={"state": POS, "heads": ["nope"]}).json()["kind"] == "unknown_head"


def test_evaluate_is_unsupported_without_an_evaluator_pack(client):
    r = client.post("/v1/evaluate", json={"state": POS})
    assert r.status_code == 422 and r.json()["kind"] == "unsupported"


# ----------------------------------------------------------------------------- reports

def test_calibration_view_shape(client):
    r = client.get("/v1/calibration", params={"head": "good"})
    assert r.status_code == 200, r.text
    c = r.json()
    assert c["head"] == "good" and c["type"] == "noul" and c["model"] == "rlcd-0.1.1"
    assert c["scope"] == "trusted holdout rows" and c["primary_metrics"] == ["log_loss", "brier"]
    assert set(c) >= {"n_holdout", "brier", "log_loss", "ece", "accuracy", "baseline", "brier_skill_score", "skill",
                      "reliability", "per_class", "policy", "split", "conformal", "calibration", "data", "notes"}
    assert len(c["reliability"]["top_label"]) == 10 and c["skill"]["has_skill"] is True
    assert c["calibration"]["fitted_on"] == "trusted rows of each fold's calibration block"
    assert c["split"]["ordering"] == "group" and c["data"]["holdout_rows"] == c["n_holdout"]
    assert c["policy"]["verdict"]["medium"]["max_tier"] in ("act", "confirm", "escalate")
    assert c["conformal"]["selective"]["target_error"] == 0.1
    r = client.get("/v1/calibration", params={"head": "good", "segment": "eu"})
    assert r.status_code == 422 and r.json()["segments"] == []


def test_bernoulli_view_shape(client):
    b = client.get("/v1/bernoulli", params={"head": "color", "stakes": "high"}).json()
    assert b["head"] == "color" and b["model"] == "rlcd-0.1.1" and b["stakes"] == "high"
    assert b["required_accuracy"] == 0.9 and b["table_is_live"] is True and b["model_version"] == "rlcd-0.1.1"
    assert len(b["bins"]) == 30 and b["config"]["min_p_above"] == 0.9
    assert sum(x["wins"] + x["losses"] for x in b["bins"]) == b["holdout_trials"] + b["online_trials"]
    assert set(b["policy_holdout"]) >= {"low", "medium", "high", "rows"}
    assert len(client.get("/v1/bernoulli", params={"head": "good"}).json()["bins"]) == 20


# ----------------------------------------------------------------------------- the reinforcement loop

def test_feedback_updates_the_ledger_once_and_trains_at_the_next_train(fresh):
    svc = fresh.app.state.svc
    d = fresh.post("/v1/decide", json={"state": POS, "heads": ["color"]}).json()
    did, chosen = d["decision_id"], d["heads"]["color"]["decision"]["class"]
    before = svc.ledger.table("color")
    fb = fresh.post("/v1/feedback", json={"decision_id": did, "label": chosen, "source": "crm"})
    assert fb.status_code == 200, fb.text
    out = fb.json()
    assert out["stored"] is True and out["head"] == "color" and out["label"] == chosen and out["feedback_rows"] == 1
    assert out["bernoulli"]["updated"] is True and out["bernoulli"]["correct"] is True
    again = fresh.post("/v1/feedback", json={"decision_id": did, "label": chosen, "source": "crm"}).json()
    assert again["bernoulli"] == {"updated": False, "reason": "this decision_id was already scored"}
    assert again["feedback_rows"] == 2                            # stored, but scored once
    after = svc.ledger.table("color")
    assert after["online_trials"] == before["online_trials"] + 1 and after["applied"] == [did]
    view = fresh.get("/v1/bernoulli", params={"head": "color"}).json()
    assert view["online_trials"] == 1 and view["table_is_live"] is True
    # a raw-state outcome without a decision: stored for training, no ledger update
    raw = fresh.post("/v1/feedback", json={"head": "color", "state": NEG, "label": "blue", "source": "audit"})
    assert raw.status_code == 200 and raw.json()["bernoulli"] is None
    heads = {h["name"]: h for h in fresh.get("/v1/heads").json()["heads"]}
    assert heads["color"]["feedback_rows"] == 3
    # the next train picks the rows up (one per decision_id) and restarts the ledger from the new holdout
    job = train(fresh, {"heads": ["color"]})
    assert job["status"] == "succeeded" and job["model"] == "rlcd-0.1.2"
    assert any("2 feedback rows added to training" in n for n in job["heads"]["color"]["notes"])
    assert job["heads"]["color"]["data"]["by_source"]["feedback"] == 2
    fresh_table = fresh.get("/v1/bernoulli", params={"head": "color"}).json()
    assert fresh_table["online_trials"] == 0 and fresh_table["model_version"] == "rlcd-0.1.2"
    # the heads that were not retrained are carried over, and their ledger stays live
    good = fresh.get("/v1/bernoulli", params={"head": "good"}).json()
    assert good["model"] == "rlcd-0.1.2" and good["table_is_live"] is True
    meta = svc.bundle("rlcd-0.1.2").meta
    assert meta["retrained"] == ["color"] and meta["heads"]["good"]["carried_over"] is True


def test_feedback_validation(client):
    r = client.post("/v1/feedback", json={"decision_id": "nope", "label": "red", "source": "t"})
    assert r.status_code == 422 and r.json()["kind"] == "unknown_decision"
    multi = client.post("/v1/decide", json={"state": POS}).json()["decision_id"]
    r = client.post("/v1/feedback", json={"decision_id": multi, "label": "red", "source": "t"})
    assert r.status_code == 422 and set(r.json()["heads"]) == {"good", "color", "level"}
    r = client.post("/v1/feedback", json={"head": "color", "state": POS, "label": "purple", "source": "t"})
    assert r.status_code == 422 and r.json()["classes"] == ["red", "green", "blue"]
    r = client.post("/v1/feedback", json={"head": "color", "label": "red", "source": "t"})
    assert r.status_code == 422 and "needs the state" in r.json()["error"]
    assert client.post("/v1/feedback", json={"head": "color", "state": POS, "label": "red"}).status_code == 422
    assert client.post("/v1/feedback", json={"label": "red", "state": POS, "source": "t"}).status_code == 422
    r = client.post("/v1/feedback", json={"decision_id": multi, "head": "good", "label": "yes", "source": "t"})
    assert r.status_code == 200 and r.json()["label"] == "true"     # people write yes/no


def test_dataset_from_labels_counts_a_decision_once_and_uses_precomputed_features():
    from open_rlcd.labeling.store import LabelItem
    from open_rlcd.packs.base import dataset_from_labels
    from tests.toypack import ToyProvider

    spec = ToyPack().heads(None)["color"]
    labelled = [{"item": LabelItem(id=f"i{i}", view="v", state={"f": [float(i)] * 6}, group=f"g{i}"),
                 "label": "red", "source": "human"} for i in range(3)]
    feedback = [
        {"id": "1", "decision_id": "d1", "head": "color", "state": POS, "label": "red", "weight": 1.0},
        {"id": "2", "decision_id": "d1", "head": "color", "state": POS, "label": "blue", "weight": 1.0},
        {"id": "3", "head": "color", "label": "green", "weight": 2.0,
         "state": {"x": NEG["f"], "feature_names": NAMES, "feature_version": "toy-v1"}},
        {"id": "4", "head": "color", "label": "green", "state": {"x": NEG["f"], "feature_version": "old"}},
        {"id": "5", "head": "good", "state": POS, "label": "true"},
    ]
    ds = dataset_from_labels(spec, ToyProvider(), labelled, feedback, feedback_weight=5.0)
    assert isinstance(ds, Dataset) and len(ds) == 5
    fb = ds.source == "feedback"
    assert fb.sum() == 2 and not ds.trusted[fb].any()
    assert ds.y[fb].tolist() == [2, 1]                            # the later outcome for d1 wins ("blue")
    assert ds.weight[fb].tolist() == [5.0, 10.0]
    assert np.allclose(ds.X[fb][1], NEG["f"])                     # precomputed features used as sent
    assert any("1 feedback rows replaced" in n for n in ds.notes)
    assert any("another feature version" in n for n in ds.notes)
    assert any("2 feedback rows added to training" in n for n in ds.notes)
