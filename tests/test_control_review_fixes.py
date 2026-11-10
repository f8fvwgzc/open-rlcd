"""Regression tests for the defects an adversarial review of the control layer confirmed (one test per defect)."""
from __future__ import annotations

import json
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from open_rlcd.core import receipts as R
from open_rlcd.core import review as RV
from open_rlcd.core.checklist import _abstained_where
from open_rlcd.core.checklist import run as run_checks
from open_rlcd.core.config import Project
from open_rlcd.core.contract import TrainRequest
from open_rlcd.core.digest import digest
from open_rlcd.core.redact import redact, redact_text
from open_rlcd.core.service import Service
from open_rlcd.serve.api import create_app
from tests.test_control_service import BATTERY, first_with_action
from tests.test_control_service import client as toy_client
from tests.test_control_service import decide as decide_color
from tests.test_control_service import service as toy_service
from tests.test_control_service import train as toy_train
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


# ----------------------------------------------------------------------------- a second review (example output audit)

def _checks(svc) -> dict:
    return {c["id"]: c for c in run_checks(svc)["checks"]}


@pytest.fixture
def confirm_case(tmp_path):
    """A trained toy service with one confirm decision (the color head, low stakes) and its state."""
    svc = toy_service(tmp_path)
    toy_train(svc)
    c = toy_client(svc)
    state, d = first_with_action(c, "confirm")
    return svc, c, state, d


def test_a_grounded_failure_is_labelled_by_its_probability():
    from open_rlcd.packs.documents.decide import _revise

    # a gate head always names the class it would commit to ("true"): its lean is P(true), not the class
    heads = {"grounded": {"decision": {"class": "true", "p": 0.006, "tier": "escalate"}}}
    assert _revise(["grounded"], {}, heads, [], None, 2)["failed"][0]["why"] == "fails"
    heads["grounded"]["decision"]["p"] = 0.8
    assert _revise(["grounded"], {}, heads, [], None, 2)["failed"][0]["why"] == "uncertain"


def test_an_unchanged_revision_names_its_attempt_and_revision():
    stop = RV.revision_verdict({"attempt": 2, "unchanged": True, "of": "e1", "previously_failed": []}, [], 2)
    assert stop.startswith("attempt 2 (revision 1) sent exactly the state of evaluation e1")


def test_a_confirmation_counts_only_with_its_digest_and_the_judged_state(confirm_case):
    svc, c, state, d = confirm_case
    did = d["decision_id"]
    url = f"/v1/receipts/{did}"
    path = Path(svc.home) / "confirmations.jsonl"
    judged = svc.receipt_view(did)["receipt"]["state_digest"]
    # a line without a digest, even with the right state, confirms nothing
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps({"id": "x", "decision_id": did, "by": "mallory", "state_digest": judged}) + "\n")
    chk = c.post(f"{url}/check", json={"state": state}).json()
    assert chk["current"] is False and chk["checks"]["confirmation"] == "missing"
    assert any("1 confirmation line(s) ignored: it carries no digest" in s for s in chk["stale_reasons"])
    # the API's confirmation is chained, and counts
    ok = c.post(f"{url}/confirm", json={"by": "alice", "state_digest": digest(state)}).json()
    assert ok["digest"].startswith("sha256:")
    assert c.post(f"{url}/check", json={"state": state}).json()["current"] is True
    v = svc.verify_receipts()
    assert v["ok"] is True and v["confirmations"]["receipts"] == 1 and v["confirmations"]["head"] == ok["digest"]
    assert v["confirmations"]["legacy"] == 1             # the hand-written line came before the chain started
    detail = _checks(svc)["receipts.chain"]["detail"]
    assert v["head"] in detail and ok["digest"] in detail and "1 confirmation chained" in detail   # full digests
    # a copy that verifies by itself but names another state confirms nothing
    rows = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()]
    other = decide_color(c, state, stakes="low")
    assert other["action"] == "confirm"
    forged = {**rows[-1], "decision_id": other["decision_id"], "state_digest": digest({"f": [9.0] * 6})}
    forged["receipt"] = {k: x for k, x in rows[-1]["receipt"].items() if k != "digest"}
    forged["receipt"]["digest"] = R.row_digest(forged)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(forged) + "\n")
    chk = c.post(f"/v1/receipts/{other['decision_id']}/check", json={"state": state}).json()
    assert chk["current"] is False and any("names another state" in s for s in chk["stale_reasons"])
    # editing the real confirmation breaks its chain, and it stops counting
    rows[-1]["by"] = "mallory"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    v = svc.verify_receipts()
    assert v["ok"] is False and v["confirmations"]["breaks"] and not v["breaks"]
    chk = c.post(f"{url}/check", json={"state": state}).json()
    assert chk["current"] is False and any("does not match its digest" in s for s in chk["stale_reasons"])
    assert _checks(svc)["receipts.chain"]["status"] == "fail"


def test_who_and_why_are_redacted_like_every_other_logged_field(tmp_path):
    svc = toy_service(tmp_path, {"serve": {"redact": ["secrets", "pii"]}})
    v1 = toy_train(svc)
    toy_train(svc)
    c = toy_client(svc)
    state, d = first_with_action(c, "confirm")
    key = "AKIAIOSFODNN7EXAMPLE"
    ok = c.post(f"/v1/receipts/{d['decision_id']}/confirm", json={"by": "alice@example.com", "note": key}).json()
    assert ok["by"] == "[REDACTED:email]"
    assert c.post("/v1/feedback", json={"decision_id": d["decision_id"], "label": "red",
                                        "source": f"reviewer {key}"}).status_code == 200
    assert c.post("/v1/automation", json={"enabled": False, "reason": f"leaked {key}",
                                          "by": "bob@example.com"}).status_code == 200
    svc.promote(v1, reason=f"rehearsal {key}", by="carol@example.com", kind="rollback")
    for name in ("confirmations.jsonl", "feedback.jsonl", "automation.jsonl", "automation.json", "registry.json"):
        text = (Path(svc.home) / name).read_text(encoding="utf-8")
        assert key not in text and "@example.com" not in text, name
    assert "confirmations (who, note)" in _checks(svc)["secrets.redact"]["detail"]


def test_check_warns_when_a_caller_could_confirm_its_own_decisions(confirm_case, monkeypatch):
    svc = confirm_case[0]
    monkeypatch.delenv("RLCD_ADMIN_KEY", raising=False)
    item = _checks(svc)["security.admin_key"]
    assert item["status"] == "warn" and "/v1/receipts/{id}/confirm" in item["detail"]
    assert "a person's confirmation is not enforced" in item["detail"]
    monkeypatch.setenv("RLCD_ADMIN_KEY", "operator")
    item = _checks(svc)["security.admin_key"]
    assert item["status"] == "info" and "nor confirm its own confirm decisions" in item["detail"]


def test_battery_versions_say_when_no_receipt_can_be_compared(tmp_path):
    svc = toy_service(tmp_path, {"batteries": {"intake": BATTERY}})
    toy_train(svc)
    c = toy_client(svc)
    body = {"battery": "intake", "state": {"f": [0.0] * 6}}
    assert c.post("/v1/systemone", json=body).status_code == 200          # no receipt under serve.receipts: decisions
    item = _checks(svc)["batteries.versions"]
    assert item["status"] == "warn" and "goes unnoticed" in item["detail"] and '"receipt": true' in item["detail"]
    assert c.post("/v1/systemone", json={**body, "receipt": True}).status_code == 200
    item = _checks(svc)["batteries.versions"]
    assert item["status"] == "pass" and "only calls that asked for a receipt are compared" in item["detail"]
    every = toy_service(tmp_path / "all", {"batteries": {"intake": BATTERY}, "serve": {"receipts": "all"}})
    assert _checks(every)["batteries.versions"]["status"] == "info"
    plain = toy_service(tmp_path / "none")
    assert _checks(plain)["batteries.versions"]["status"] == "info"


def test_abstentions_are_counted_by_where_they_went():
    assert _abstained_where({"escalate_to_human": 1, "revise": 1}) == ("1 escalated to a person, 1 sent back to "
                                                                       "the generator")
    assert _abstained_where({"retry_generation": 1, "revise": 2, "": 1}) == ("3 sent back to the generator, 1 with "
                                                                             "no next step recorded")


def test_the_digest_is_not_a_signature_and_the_chain_says_so(confirm_case):
    """The documented limit, pinned: a line someone wrote into RLCD_HOME with a digest of its own counts at the
    check (the digest is unkeyed); its place in the chain is what `rlcd receipts verify` and receipts.chain judge."""
    svc, c, state, d = confirm_case
    row = svc.receipt_view(d["decision_id"])
    line = {"id": "forged", "received_at": row["created_at"], "decision_id": d["decision_id"], "by": "mallory",
            "state_digest": row["receipt"]["state_digest"], "action": "confirm",
            "receipt": {"format": R.CONFIRMATION_FORMAT, "prev": "sha256:" + "0" * 64}}
    line["receipt"]["digest"] = R.row_digest(line)
    with open(Path(svc.home) / "confirmations.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(line) + "\n")
    assert c.post(f"/v1/receipts/{d['decision_id']}/check", json={"state": state}).json()["current"] is True
    v = svc.verify_receipts()
    assert v["ok"] is False and "does not name the receipt before it" in v["confirmations"]["breaks"][0]["why"]
    assert _checks(svc)["receipts.chain"]["status"] == "fail"


@pytest.mark.parametrize("name", ["confirmations.jsonl", "decisions.jsonl"])
def test_one_malformed_receipt_line_does_not_stop_later_appends(confirm_case, name):
    svc, c, state, d = confirm_case
    assert c.post(f"/v1/receipts/{d['decision_id']}/confirm", json={"by": "alice"}).status_code == 200
    with open(Path(svc.home) / name, "a", encoding="utf-8") as f:
        f.write(json.dumps({"decision_id": "junk", "receipt": "x"}) + "\n")
    fresh = toy_client(toy_service(Path(svc.home)))      # a new process: no cached chain tip
    assert fresh.post(f"/v1/receipts/{d['decision_id']}/confirm", json={"by": "bob"}).status_code == 200
    assert decide_color(fresh, state, stakes="low")["decision_id"]
    v = svc.verify_receipts()
    chain = v["confirmations"] if name == "confirmations.jsonl" else v
    assert len(chain["breaks"]) == 1 and "without a receipt" in chain["breaks"][0]["why"]
    junk = fresh.post("/v1/receipts/junk/check", json={"state": state})
    assert junk.status_code in (404, 422)              # never a 500


def test_rollback_and_kill_switch_rehearsals_are_counted_from_operator_moves(tmp_path):
    svc = toy_service(tmp_path)
    toy_train(svc)
    v2 = toy_train(svc)
    c = _checks(svc)
    assert "never been moved by promote / rollback" in c["rollback.ready"]["detail"]       # two trains are no drill
    assert "never been switched off" in c["automation.switch"]["detail"]
    svc.promote("rlcd-0.1.1", kind="rollback")
    svc.promote(v2, kind="promote")
    svc.set_automation(False, reason="drill", by="ops")
    svc.set_automation(True, reason="drill over", by="ops")
    c = _checks(svc)
    assert "moved 2 time(s) by promote / rollback" in c["rollback.ready"]["detail"]
    assert "switched off 1 time(s) before" in c["automation.switch"]["detail"]


def test_the_admin_key_warning_covers_old_confirm_decisions(confirm_case, monkeypatch):
    from datetime import UTC, datetime, timedelta

    svc = confirm_case[0]
    monkeypatch.delenv("RLCD_ADMIN_KEY", raising=False)
    later = run_checks(svc, now=datetime.now(UTC) + timedelta(days=30))       # outside the review window
    item = {x["id"]: x for x in later["checks"]}["security.admin_key"]
    assert item["status"] == "warn" and "confirm decision(s)" in item["detail"]


def test_a_review_counts_as_capped_only_when_the_policy_alone_would_have_acted():
    from datetime import UTC, datetime, timedelta
    from types import SimpleNamespace

    from open_rlcd.core.checklist import _scan

    now = datetime.now(UTC)
    at = (now - timedelta(hours=1)).isoformat()
    rows = [{"action": "confirm", "created_at": at, "receipt": {"rollout": {"capped": True, "proposed_action": "act"}}},
            # a head was capped inside a plan that confirms anyway: not a review the cap created
            {"action": "confirm", "created_at": at,
             "receipt": {"rollout": {"capped": True, "proposed_action": "confirm"}}},
            {"action": "confirm", "created_at": (now - timedelta(days=40)).isoformat(), "receipt": {}}]
    s = _scan(SimpleNamespace(logs=SimpleNamespace(decisions=lambda: iter(rows))), now)
    assert s["capped"] == 1 and s["confirms"] == 3 and sum(v[0] for v in s["per_day"].values()) == 2


def test_receipts_chain_reports_confirmations_whose_decisions_are_gone(confirm_case):
    svc, c, state, d = confirm_case
    assert c.post(f"/v1/receipts/{d['decision_id']}/confirm", json={"by": "alice"}).status_code == 200
    (Path(svc.home) / "decisions.jsonl").rename(Path(svc.home) / "decisions.archived.jsonl")
    item = _checks(svc)["receipts.chain"]
    assert item["status"] == "warn" and "holds no receipt for the decisions they confirm" in item["detail"]
    head = svc.verify_receipts()["confirmations"]["head"]
    assert head in item["detail"]                       # the head to anchor is still printed in full
