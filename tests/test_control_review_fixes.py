"""Regression tests for the defects an adversarial review of the control layer confirmed (one test per defect)."""
from __future__ import annotations

import json
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from open_rlcd.core import review as RV
from open_rlcd.core.config import Project
from open_rlcd.core.contract import TrainRequest
from open_rlcd.core.redact import redact, redact_text
from open_rlcd.core.service import Service
from open_rlcd.serve.api import create_app
from tests.toypack import ToyPack

NEG = {"f": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]}


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    root = tmp_path_factory.mktemp("review-fixes")
    svc = Service(Project(root=root), home=root / "home", pack=ToyPack(n=600))
    assert svc.trainer.start(TrainRequest(), wait=True)["status"] == "succeeded"
    return svc


def test_an_abstained_decision_is_never_a_current_approval(trained):
    c = TestClient(create_app(trained))
    d = c.post("/v1/decide", json={"state": NEG, "heads": ["good"], "stakes": "high"}).json()
    assert d["action"] == "escalate"
    chk = trained.check_receipt(d["decision_id"], state=NEG)
    assert chk["current"] is False and chk["checks"]["approval"] == "none"


def test_the_assignment_pattern_is_linear_on_adversarial_input():
    for text in ("secret-" * 2000 + "x", "a" * 50_000 + "token" + "b" * 50_000 + "=", "_" * 100_000):
        t0 = time.perf_counter()
        redact_text(text)
        assert time.perf_counter() - t0 < 1.0


def test_output_screening_rejects_secret_shapes_not_documentation():
    project = Project(root=Path("."))
    assert RV.output_scan("Password: minimum twelve characters, rotated yearly.", project) is None
    assert RV.output_scan("Set max_tokens=128000 in the request body.", project) is None
    hit = RV.output_scan("Use AKIAIOSFODNN7EXAMPLE to log in.", project)
    assert hit is not None and hit["rule"] == "output_scan" and "aws_key" in hit["found"]


def test_values_nested_under_a_secret_key_are_redacted():
    out, found = redact({"credentials": {"user": "svc", "keys": ["k1-abcdef", "k2-abcdef"]}, "note": "ok"})
    assert out["credentials"] == {"user": "[REDACTED:secret_field]",
                                  "keys": ["[REDACTED:secret_field]", "[REDACTED:secret_field]"]}
    assert out["note"] == "ok" and "secret_field" in found


def test_a_rolled_back_version_keeps_its_online_evidence(tmp_path):
    svc = Service(Project(root=tmp_path), home=tmp_path / "home", pack=ToyPack(n=600))
    assert svc.trainer.start(TrainRequest(), wait=True)["status"] == "succeeded"
    v1 = svc.registry.latest_id()
    d = svc.decide(__import__("open_rlcd.core.contract", fromlist=["DecideRequest"]).DecideRequest(
        state=NEG, heads=["good"]))
    from open_rlcd.core.contract import FeedbackRequest

    svc.feedback(FeedbackRequest(decision_id=d["decision_id"], label="true", source="t"))
    trials = svc.ledger.table("good").get("online_trials")
    assert trials == 1
    assert svc.trainer.start(TrainRequest(), wait=True)["status"] == "succeeded"     # v2 replaces the table
    assert svc.gate_table(v1, "good", svc.loaded(v1, "good")).get("online_trials") == 1   # archived, not lost
    svc.promote(v1, kind="rollback")
    assert svc.ledger.table("good").get("online_trials") == 1


def test_a_second_rollback_does_not_re_promote_the_rolled_back_version(tmp_path):
    svc = Service(Project(root=tmp_path), home=tmp_path / "home", pack=ToyPack(n=600))
    for _ in range(3):
        assert svc.trainer.start(TrainRequest(), wait=True)["status"] == "succeeded"
    v1, v2, v3 = (f"rlcd-0.1.{i}" for i in (1, 2, 3))
    assert svc.registry.previous_latest() == v2
    svc.promote(v2, kind="rollback")                       # v3 is bad
    assert svc.registry.previous_latest() == v1            # not v3 again
    svc.promote(v1, kind="rollback")
    assert svc.registry.latest_id() == v1 and svc.registry.previous_latest() is None


def test_the_body_limit_holds_with_content_length_and_chunked_together(tmp_path, monkeypatch):
    c = TestClient(create_app(Service(Project(root=tmp_path), home=tmp_path, pack=ToyPack())))
    monkeypatch.setenv("RLCD_MAX_BODY_BYTES", "1000")
    body = json.dumps({"state": NEG, "questions": {"good": {"type": "noul"}}, "pad": "x" * 5000}).encode()
    r = c.post("/v1/systemone", content=(body[i:i + 512] for i in range(0, len(body), 512)),
               headers={"content-type": "application/json", "content-length": "10"})
    assert r.status_code in (400, 413)


def test_confirming_needs_the_admin_key_when_one_is_set(trained, monkeypatch):
    c = TestClient(create_app(trained))
    d = c.post("/v1/decide", json={"state": NEG, "heads": ["good"]}).json()
    monkeypatch.setenv("RLCD_API_KEY", "caller")
    monkeypatch.setenv("RLCD_ADMIN_KEY", "operator")
    url = f"/v1/receipts/{d['decision_id']}/confirm"
    assert c.post(url, json={"by": "me"}, headers={"Authorization": "Bearer caller"}).status_code == 401
    assert c.post(url, json={"by": "me"}, headers={"Authorization": "Bearer operator"}).status_code == 200


def test_rank_reads_the_questions_inputs(trained):
    c = TestClient(create_app(trained))
    body = {"candidates": [{"id": "a", "state": {"x": NEG}}], "question": {"type": "noul", "head": "good",
                                                                          "path": "x"}}
    assert c.post("/v1/rank", json=body).status_code == 200
    body["question"]["path"] = "y"
    r = c.post("/v1/rank", json=body)
    assert r.status_code == 422 and r.json()["kind"] == "missing_evidence"


def test_a_logged_nan_does_not_break_the_views(trained):
    c = TestClient(create_app(trained))
    d = c.post("/v1/decide", content=b'{"state": {"f": [0, 0, 0, 0, 0, 0], "extra": NaN}, "heads": ["good"]}',
               headers={"content-type": "application/json"})
    if d.status_code != 200:     # a server that refuses NaN at the edge is fine too
        return
    assert c.get(f"/v1/receipts/{d.json()['decision_id']}").status_code == 200
    assert c.get("/v1/review").status_code == 200


def test_one_record_never_logs_more_than_the_state_budget(trained, monkeypatch):
    from open_rlcd.core import receipts as R

    monkeypatch.setattr(R, "MAX_LOGGED_STATE_BYTES", 2000)
    big = {"f": NEG["f"], "blob": "x" * 5000}
    row = trained.log_decision(trained.registry.latest_id(), "decide",
                               {"good": {"state": big, "class": "true", "p": 0.9, "tier": "confirm"}})
    h = row["heads"]["good"]
    assert "state" not in h and h["state_digest"].startswith("sha256:") and "state_omitted" in h


def test_check_fails_on_rollout_entries_for_heads_the_model_does_not_have(tmp_path):
    from open_rlcd.core.checklist import run

    svc = Service(Project(root=tmp_path, data={"rollout": {"heads": {"godo": "shadow"}}}), home=tmp_path,
                  pack=ToyPack(n=600))
    item = {c["id"]: c for c in run(svc)["checks"]}["rollout.mode"]
    assert item["status"] == "fail" and "godo" in item["detail"]
