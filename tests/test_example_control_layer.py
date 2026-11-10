"""examples/control-layer/run.py, the reference integration, run end to end on the shipped demo bundle.

What: the order of the request path, observed on the real calls (the stand-in LLM, the client's evaluations,
receipt checks and confirmations), not on labels the example writes: nothing reaches the LLM before a current check
of the plan's receipt, and nothing ships before a current check of the evaluation's; a stale receipt stops either;
a confirm plan is confirmed by a reviewer holding the admin key, never by the application, whose probe writes
nothing; every way out (a decide or evaluate fallback, a plan that does not generate, a server that lets the
decision key confirm) stops before the next side effect, and the plan's outcome (when there is a plan) says how the
request ended; the keys the example sets are restored.

Why: callers copy this file. An integration that generates before the approval holds, confirms its own decisions,
crashes on a fallback or reports an unverified request as a success teaches what the control layer exists to
prevent.
"""
from __future__ import annotations

import contextlib
import importlib.util
import json
import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def example():
    spec = importlib.util.spec_from_file_location("control_layer_example", ROOT / "examples" / "control-layer" / "run.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def calls(example, monkeypatch):
    """Every LLM call, receipt check and confirmation the example makes, in order, with keys set beforehand so the
    test sees whether the example restores them."""
    seen: list[tuple] = []
    llm, check, confirm = example.stand_in_llm, example.Client.check_receipt, example.Client.confirm_receipt
    evaluate = example.Client.evaluate

    def watched_llm(query, evidence):
        seen.append(("llm",))
        return llm(query, evidence)

    def watched_evaluate(self, state, **kw):
        r = evaluate(self, state, **kw)
        seen.append(("evaluate", kw.get("revision") is not None, r.decision_id))
        return r

    def watched_check(self, decision_id, **kw):
        r = check(self, decision_id, **kw)
        seen.append(("check", decision_id, r.current))
        return r

    def watched_confirm(self, decision_id, **kw):
        try:
            out = confirm(self, decision_id, **kw)
        except Exception:
            seen.append(("confirm", decision_id, kw.get("by"), False))
            raise
        seen.append(("confirm", decision_id, kw.get("by"), True))
        return out

    monkeypatch.setattr(example, "stand_in_llm", watched_llm)
    monkeypatch.setattr(example.Client, "check_receipt", watched_check)
    monkeypatch.setattr(example.Client, "evaluate", watched_evaluate)
    monkeypatch.setattr(example.Client, "confirm_receipt", watched_confirm)
    monkeypatch.delenv("RLCD_AUTOMATION", raising=False)
    for k in ("RLCD_API_KEY", "RLCD_ADMIN_KEY"):
        monkeypatch.setenv(k, "before")
    return seen


def _outcomes(home: Path) -> list[dict]:
    path = home / "var" / "outcomes.jsonl"
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()] if path.exists() else []


def _keys_restored() -> bool:
    return os.environ.get("RLCD_API_KEY") == os.environ.get("RLCD_ADMIN_KEY") == "before"


def test_the_llm_runs_only_after_a_current_check_of_the_plan(example, calls, tmp_path):
    out = example.walk("full", home=tmp_path)
    assert _keys_restored()
    plan_id, ev = out["plan_id"], out["events"]
    assert ("llm",) in calls, ev
    first_llm = calls.index(("llm",))
    assert ("check", plan_id, True) in calls[:first_llm]
    if "decide:confirm" in ev:
        # the application's own confirmation is refused; the reviewer's comes before the check that lets the LLM run
        confirms = [c for c in calls if c[0] == "confirm"]
        probe = confirms[0]                    # a made-up id: the probe never touches a real decision
        assert probe[1].startswith("probe-") and probe[2:] == ("the-application", False)
        assert ("confirm", plan_id, "reviewer@example.com", True) in calls[:calls.index(("check", plan_id, True))]
        assert out["verify"]["confirmations"]["receipts"] == 1
    assert out["verify"]["ok"] is True
    assert None not in [c[1] for c in calls if c[0] == "check"]
    # the outcome belongs to the plan, counts the revision the example sent, and claims no unverified success
    rows = _outcomes(tmp_path)
    assert len(rows) == 1 and rows[0]["decision_id"] == plan_id and "task_success" not in rows[0]
    revisions = sum(1 for c in calls if c[:2] == ("evaluate", True))         # the revisions really sent
    assert rows[0]["retries"]["workflow"] == revisions
    assert rows[0]["route_taken"] and rows[0]["provider"] == "stand-in"
    assert [c["by"] for c in out["svc"].logs.confirmations(plan_id)] in ([], ["reviewer@example.com"])
    for r in out["svc"].logs.decisions():
        if r.get("kind") == "evaluate":                     # every evaluation is bound to the plan's evidence
            assert ((r.get("receipt") or {}).get("audit") or {}).get("plan") == plan_id


def test_a_decide_fallback_generates_nothing(example, calls, tmp_path, monkeypatch):
    def down(self, req):
        raise RuntimeError("the decision service is down")

    monkeypatch.setattr(example.Service, "decide", down)
    out = example.walk("full", home=tmp_path)
    assert "fallback:decide" in out["events"] and out["plan_id"] is None
    assert not [c for c in calls if c[0] in ("llm", "check", "confirm")]
    assert _outcomes(tmp_path) == [] and _keys_restored()


def test_an_evaluate_fallback_ships_nothing_and_never_checks_a_missing_receipt(example, calls, tmp_path, monkeypatch):
    def down(self, request):
        raise RuntimeError("the evaluator is down")

    monkeypatch.setattr(example.Service, "evaluate", down)
    out = example.walk("full", home=tmp_path)
    ev = out["events"]
    assert "fallback:evaluate" in ev and "ship" not in ev
    assert None not in [c[1] for c in calls if c[0] == "check"]
    rows = _outcomes(tmp_path)
    assert len(rows) == 1 and rows[0]["decision_id"] == out["plan_id"] and "task_success" not in rows[0]
    assert "review was unavailable" in rows[0]["note"] and _keys_restored()


def test_a_plan_that_does_not_generate_stops_before_the_llm(example, calls, tmp_path, monkeypatch):
    monkeypatch.setattr(example, "QUERY", "hi")              # a greeting: the plan asks back instead of answering
    out = example.walk("full", home=tmp_path)
    if "stop" not in out["events"]:
        pytest.skip(f"the demo bundle planned {out['events']} for a greeting")
    assert ("llm",) not in calls
    rows = _outcomes(tmp_path)
    assert len(rows) == 1 and rows[0]["note"].startswith("no generation")
    assert "provider" not in rows[0] and "route_taken" not in rows[0]


def test_an_application_that_could_confirm_itself_stops(example, calls, tmp_path, monkeypatch):
    @contextlib.contextmanager
    def caller_key_only():
        os.environ["RLCD_API_KEY"] = "caller"
        os.environ.pop("RLCD_ADMIN_KEY", None)              # a server without the admin key
        yield "caller", "caller"

    monkeypatch.setattr(example, "_two_keys", caller_key_only)
    out = example.walk("full", home=tmp_path)
    if "decide:confirm" not in out["events"]:
        pytest.skip(f"the demo bundle planned {out['events']}")
    assert "confirm:self-accepted" in out["events"] and ("llm",) not in calls
    assert "set RLCD_ADMIN_KEY" in _outcomes(tmp_path)[0]["note"]
    # the probe wrote nothing: the plan holds no confirmation and its receipt approves nothing
    svc, pid = out["svc"], out["plan_id"]
    assert svc.logs.confirmations() == []
    judged = svc.receipt_view(pid)["receipt"]["state_digest"]
    assert svc.check_receipt(pid, state_digest=judged)["current"] is False


def test_the_keys_are_restored_when_the_walk_fails(example, calls, tmp_path, monkeypatch):
    def broken(self, **kw):
        raise RuntimeError("metrics are broken")

    monkeypatch.setattr(example.Service, "metrics_view", broken)
    with pytest.raises(RuntimeError, match="metrics are broken"):
        example.walk("full", home=tmp_path)
    assert _keys_restored()


def test_the_stand_in_cites_only_what_it_used(example):
    evidence = [{"id": "a", "text": "First. Second."}, {"id": "b", "text": "Other."}]
    assert example.stand_in_llm("q", evidence) == ("First.", ["a"])
    assert example.stand_in_llm("q", []) == ("", [])


def _force_act(monkeypatch):
    """Every head decision that commits to a class acts: the demo bundle's weak heads would never ship an answer."""
    from open_rlcd.core import policy as PL
    from open_rlcd.core.service import Service

    real = PL.decide

    def acting(*a, **kw):
        d = real(*a, **kw)
        if d.get("class") is not None:
            d["tier"] = d["uncapped_tier"] = d["max_tier"] = "act"
            d["commit"] = d["class"]
        return d

    monkeypatch.setattr(PL, "decide", acting)
    monkeypatch.setattr(Service, "_apply_verified", lambda self, d, *a, **kw: None)


def test_a_stale_plan_receipt_stops_before_the_llm(example, calls, tmp_path, monkeypatch):
    monkeypatch.setattr(example, "MAX_AGE_S", 1e-6)          # every receipt is older than this when it is checked
    out = example.walk("full", home=tmp_path)
    assert ("check", out["plan_id"], False) in calls and ("llm",) not in calls
    assert "stale:plan" in out["events"] and _outcomes(tmp_path)[0]["note"].startswith("no generation")


def test_an_answer_ships_only_after_a_current_check_of_its_evaluation(example, calls, tmp_path, monkeypatch):
    _force_act(monkeypatch)
    out = example.walk("full", home=tmp_path)
    ev = out["events"]
    assert ev[1] == "decide:act" and "ship" in ev, ev
    assert ("check", out["evaluation_id"], True) in calls
    assert calls.index(("check", out["evaluation_id"], True)) > calls.index(("llm",))
    rows = _outcomes(tmp_path)
    assert rows[0]["note"].startswith("shipped") and "task_success" not in rows[0]


def test_a_stale_evaluation_receipt_ships_nothing(example, calls, tmp_path, monkeypatch):
    _force_act(monkeypatch)
    check = example.Client.check_receipt                      # the watched one: the stale call is still recorded
    evaluations = set()

    def stale_for_evaluations(self, decision_id, **kw):
        if decision_id in evaluations:
            kw["max_age_s"] = 1e-6
        return check(self, decision_id, **kw)

    evaluate = example.Client.evaluate

    def remember(self, state, **kw):
        r = evaluate(self, state, **kw)
        evaluations.add(r.decision_id)
        return r

    monkeypatch.setattr(example.Client, "check_receipt", stale_for_evaluations)
    monkeypatch.setattr(example.Client, "evaluate", remember)
    out = example.walk("full", home=tmp_path)
    ev = out["events"]
    assert "evaluate:ship:ship" in ev and "stale:evaluation" in ev and "ship" not in ev, ev
    assert ("check", out["evaluation_id"], False) in calls
    assert "not shipped" in _outcomes(tmp_path)[0]["note"]
