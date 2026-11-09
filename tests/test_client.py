"""open_rlcd.client: the typed SDK over the open-rlcd HTTP API.

Two halves, both offline. The real app in process (fastapi's TestClient is an httpx.Client, passed as `http=`) for
the shapes the client reads: untrained answers, decisions, receipts, batteries, the kill switch. httpx.MockTransport
for the failure states the real app cannot produce on demand: a 503 then a 200, Retry-After, transport errors,
malformed bodies, missing answers, fail_closed. No sockets, no network, no training.
"""
from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from email.utils import format_datetime

import httpx
import numpy as np
import pytest
from fastapi.testclient import TestClient

from open_rlcd.client import (
    Answer,
    AuthError,
    Choice,
    Client,
    Conflict,
    Decision,
    Evaluation,
    MalformedResponse,
    MissingAnswer,
    NotAllowed,
    NotFound,
    Noul,
    RateLimited,
    RLCDError,
    Score,
    ServerError,
    Timeout,
    TransportError,
    Unavailable,
    ValidationError,
    digest,
)
from open_rlcd.core import digest as core_digest
from open_rlcd.core.config import Project
from open_rlcd.core.service import Service
from open_rlcd.serve.api import create_app
from tests.toypack import ToyPack

STATE = {"f": [0.5, -1.0, 0.2, 0.0, 1.0, -0.3]}
ALL = {"good": Noul(), "color": Choice(), "level": Score()}
BATTERIES = {"batteries": {"intake": {"version": "1", "owner": "tests", "changelog": ["1: ok, hue"],
                                      "questions": {"ok": {"type": "noul", "head": "good"},
                                                    "hue": {"type": "choice", "head": "color"}}}}}


def _no_retry_expected(seconds: float) -> None:
    raise AssertionError(f"the in-process app gave a retryable failure (sleep {seconds})")


@pytest.fixture
def server(tmp_path) -> TestClient:
    svc = Service(Project(root=tmp_path, data=json.loads(json.dumps(BATTERIES))), home=tmp_path, pack=ToyPack())
    return TestClient(create_app(svc))


@pytest.fixture
def rlcd(server) -> Client:
    return Client(http=server, sleep=_no_retry_expected)


# ----------------------------------------------------------------------------- scripted transport

class Script:
    """A MockTransport handler that plays its steps in order (the last one repeats): (status, body[, headers]) with
    a JSON body or raw bytes, or an httpx exception class raised as a transport failure."""

    def __init__(self, *steps):
        self.steps = steps
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        step = self.steps[min(len(self.requests), len(self.steps)) - 1]
        if isinstance(step, type) and issubclass(step, Exception):
            raise step(f"scripted {step.__name__}", request=request)
        status, body, *rest = step
        headers = rest[0] if rest else {}
        if isinstance(body, bytes):
            return httpx.Response(status, content=body, headers=headers)
        return httpx.Response(status, json=body, headers=headers)

    @property
    def calls(self) -> int:
        return len(self.requests)

    def body(self, i: int = -1) -> dict:
        return json.loads(self.requests[i].content)


def mocked(script: Script, **kw) -> tuple[Client, list[float]]:
    slept: list[float] = []
    http = httpx.Client(transport=httpx.MockTransport(script), base_url="http://rlcd.test")
    return Client(http=http, sleep=slept.append, **kw), slept


NOUL = {"type": "noul", "noul": 0.9, "confidence": 0.8, "head": "good", "calibrated": True, "skill": True}
S1 = {"model": "rlcd-0.1.3", "answers": {"a": NOUL}, "usage": {"input_tokens": 0, "output_tokens": 0, "questions": 1}}
DECIDE = {"decision_id": "d1", "model": "rlcd-0.1.3", "action": "act", "tier": "act",
          "heads": {"good": {"answer": NOUL, "decision": {"tier": "act"}}}, "rollout": {"mode": "full"},
          "receipt": {"id": "d1"}, "warnings": []}
EVALUATE = {"decision_id": "e1", "model": "rlcd-0.1.3", "verdict": "ship", "next": "ship", "action": "act",
            "decided_by": "heads", "stakes": "medium", "heads": {}, "receipt": {"id": "e1"}, "warnings": []}
ACK = {"id": "f1", "stored": True}


def err(status: int, kind: str | None, retryable: bool, **extra) -> dict:
    return {"error": f"scripted {kind}", "kind": kind, "retryable": retryable, **extra}


UNAVAILABLE = (503, err(503, "connector_unavailable", True), {"Retry-After": "2"})


# ----------------------------------------------------------------------------- question builders and answers

def test_question_builders_emit_only_what_is_set():
    assert Noul().to_dict() == {"type": "noul"}
    assert Noul("good", criteria={"true": "fine", "false": "not"}).to_dict() == {
        "type": "noul", "head": "good", "criteria": {"true": "fine", "false": "not"}}
    assert Choice(head="color", instructions="ignored by the head", allowed=("red", "blue"), option="red",
                  path="rag", inputs={"query": "request.text"}).to_dict() == {
        "type": "choice", "head": "color", "instructions": "ignored by the head", "allowed": ["red", "blue"],
        "option": "red", "path": "rag", "inputs": {"query": "request.text"}}
    assert Score(criteria=("none", "low", "mid", "high")).to_dict() == {
        "type": "score", "criteria": ["none", "low", "mid", "high"]}


def test_band_needs_calibration_and_skill_and_reads_the_committed_probability():
    def noul(p, **kw):
        return Answer({**NOUL, "noul": p, **kw})

    assert noul(0.9).band() == "high" and noul(0.7).band() == "medium" and noul(0.6).band() == "low"
    # a confident no is a confident answer: value False with p 0.95, not a "low" yes
    no = noul(0.05)
    assert no.value is False and no.p == pytest.approx(0.95) and no.band() == "high"
    # the contract: never gate on a prior or a base rate, whatever it looks like
    assert noul(0.99, calibrated=False).band() == "low"
    assert noul(0.99, skill=False).band() == "low"
    no_flags = {k: v for k, v in NOUL.items() if k not in ("calibrated", "skill")}
    assert Answer({**no_flags, "noul": 0.99}).band() == "low"
    assert noul(0.7).band(act=0.6, confirm=0.5) == "high"
    with pytest.raises(ValueError):
        noul(0.7).band(act=0.5, confirm=0.6)
    # choice with allowed options: the kept option's probability is NOT renormalised, so mass the head put on removed
    # options keeps the band honest
    c = Answer({"type": "choice", "choice": "red", "probabilities": {"red": 0.5, "blue": 0.1}, "confidence": 0.6,
                "head": "color", "calibrated": True, "skill": True, "allowed": ["red", "blue"],
                "excluded_mass": 0.4})
    assert c.value == "red" and c.p == 0.5 and c.band() == "low" and c.excluded_mass == 0.4
    assert c.allowed == ["red", "blue"] and c.noul is None and c.score is None
    s = Answer({"type": "score", "score": 2, "legend": {"0": "a", "1": "b", "2": "c"}, "calibrated": True,
                "skill": True, "probabilities": {"0": 0.05, "1": 0.1, "2": 0.85}, "confidence": 0.8, "head": "level"})
    assert s.value == 2 and s.band() == "high" and s.legend == {"0": "a", "1": "b", "2": "c"}


def test_decision_reads_allow_ask_deny_from_the_action_only():
    for action, expected in {"act": (True, False, False), "confirm": (False, True, False),
                             "escalate": (False, False, True), "hold": (False, False, True),
                             "ACT": (False, False, True)}.items():
        d = Decision.parse({**DECIDE, "action": action})
        assert (d.allow, d.ask, d.deny) == expected, action
    # every head acting does not make a confirm decision an allow: the overall action decides
    d = Decision.parse({**DECIDE, "action": "confirm", "tier": "act"})
    assert d.ask and not d.allow
    with pytest.raises(NotAllowed) as e:
        d.require_allow()
    assert e.value.result is d
    assert Decision.parse(DECIDE).require_allow().decision_id == "d1"
    assert Decision.parse(DECIDE).answer("good").noul == 0.9
    with pytest.raises(MissingAnswer):
        Decision.parse(DECIDE).answer("color")
    for broken in ({k: v for k, v in DECIDE.items() if k != "action"}, {**DECIDE, "decision_id": None},
                   {**DECIDE, "heads": []}):
        with pytest.raises(MalformedResponse):
            Decision.parse(broken)


def test_evaluation_ships_only_on_ship_and_act():
    assert Evaluation.parse(EVALUATE).ship and Evaluation.parse(EVALUATE).require_ship()
    for verdict, action in (("ship", "confirm"), ("review", "confirm"), ("reject", "escalate")):
        ev = Evaluation.parse({**EVALUATE, "verdict": verdict, "action": action})
        assert not ev.ship
        with pytest.raises(NotAllowed):
            ev.require_ship()
    with pytest.raises(MalformedResponse):
        Evaluation.parse({k: v for k, v in EVALUATE.items() if k != "verdict"})


def test_digest_is_the_servers_digest():
    assert digest is core_digest.digest
    assert digest({"b": 1, "a": (1, 2)}) == digest({"a": [1, 2], "b": 1})


# ----------------------------------------------------------------------------- the real app, in process

def test_system_one_untrained_answers_are_read_as_data_and_banded_low(rlcd):
    r = rlcd.system_one(STATE, ALL)
    assert r.model == "rlcd-0.1.0" and r.transport_retries == 0 and r.missing == []
    assert set(r.answers) == {"good", "color", "level"} and r.usage["questions"] == 3
    good, color, level = r.answers["good"], r.answers["color"], r.answers["level"]
    assert (good.type, good.noul, good.head, good.calibrated, good.skill) == ("noul", 0.5, "good", False, False)
    assert color.choice in ("red", "green", "blue") and set(color.probabilities) == {"red", "green", "blue"}
    assert level.score == 0 and level.legend == {"0": "none", "1": "low", "2": "mid", "3": "high"}
    # the uninformed prior is never eligible for anything, whatever thresholds the caller picks
    assert all(a.band() == "low" and a.band(act=0.2, confirm=0.1) == "low" for a in r.answers.values())
    assert any("not trained" in w for w in r.warnings) and r.battery is None and r.receipt is None
    assert r.raw["answers"]["good"] == good.raw


def test_system_one_receipt_binds_the_client_side_digest(rlcd):
    r = rlcd.system_one(STATE, {"ok": Noul("good"), "hue": Choice("color", allowed=["red", "blue"])}, receipt=True)
    assert r.receipt["id"] and r.receipt["state_digest"] == digest(STATE)
    hue = r.answers["hue"]
    assert hue.allowed == ["red", "blue"] and hue.excluded_mass == pytest.approx(1 / 3, abs=1e-5)
    # bodies go through the same normalisation as the digest: numpy arrays and tuples digest as the server sees them
    odd = {"f": np.array(STATE["f"]), "tag": ("a", "b")}
    r2 = rlcd.system_one(odd, {"ok": Noul("good")}, receipt=True)
    assert r2.receipt["state_digest"] == digest(odd) == digest({"f": STATE["f"], "tag": ["a", "b"]})


def test_missing_evidence_and_unknown_heads_are_validation_errors(rlcd):
    with pytest.raises(ValidationError) as e:
        rlcd.system_one({"request": {"x": 1}}, {"ok": Noul("good", path="request.text")})
    assert e.value.status == 422 and e.value.kind == "missing_evidence" and not e.value.retryable
    assert e.value.body["path"] == "request.text" and "request.text" in str(e.value)
    with pytest.raises(ValidationError) as e:
        rlcd.system_one(STATE, {"spam": Noul()})
    assert e.value.kind == "unknown_head" and e.value.body["available_heads"] == ["good", "color", "level"]
    with pytest.raises(ValidationError) as e:
        rlcd.system_one(STATE)                      # neither questions nor a battery
    assert e.value.kind == "validation"


def test_batteries_are_listed_and_answer_as_one_versioned_set(rlcd):
    listed = rlcd.batteries()
    (b,) = listed["batteries"]
    assert (b["name"], b["version"], b["owner"]) == ("intake", "1", "tests")
    assert all("is not trained" in p for p in b["problems"])          # the untrained service names each head
    r = rlcd.system_one(STATE, battery="intake@1")
    assert set(r.answers) == {"ok", "hue"} and r.battery == {"name": "intake", "version": "1", "digest": b["digest"]}
    with pytest.raises(ValidationError) as e:
        rlcd.system_one(STATE, battery="intake@2")
    assert e.value.kind == "battery_version_mismatch" and e.value.body["current"] == "1"


def test_decide_untrained_denies_and_carries_a_receipt(rlcd, server):
    d = rlcd.decide(STATE)
    assert d.action == "escalate" and d.deny and not d.allow and not d.ask and not d.fallback
    assert d.decision_id and d.receipt["id"] == d.decision_id and d.model == "rlcd-0.1.0"
    assert set(d.heads) == {"good", "color", "level"} and d.rollout["capped"] is False
    assert d.answer("good").calibrated is False and d.answer("color").band() == "low"
    with pytest.raises(NotAllowed):
        d.require_allow()
    one = rlcd.decide(STATE, stakes="low", heads=["color"], tags=["client-test"])
    assert list(one.heads) == ["color"] and one.heads["color"]["decision"]["stakes"] == "low"
    # a request error is not "could not answer": fail_closed never hides it
    strict = Client(http=server, fail_closed="abstain", sleep=_no_retry_expected)
    with pytest.raises(ValidationError) as e:
        strict.decide(STATE, stakes="extreme")
    assert e.value.kind == "validation" and strict.fallbacks == 0


def test_feedback_outcome_and_receipt_check_round_trip(rlcd):
    d = rlcd.decide(STATE)
    fb = rlcd.feedback(True, source="client-test", decision_id=d.decision_id, head="good", tags=["regression"],
                       note="checked by hand")
    assert fb["stored"] is True and fb["head"] == "good" and fb["label"] == "true"
    out = rlcd.outcome(d.decision_id, task_success=False, route_taken="human",
                       retries={"transport": d.transport_retries, "workflow": 1})
    assert out["stored"] is True and out["decision_id"] == d.decision_id
    # the untrained service abstains: the receipt still matches its state, but an abstention approves no action
    ok = rlcd.check_receipt(d.decision_id, state=STATE)
    assert not ok.current and ok.checks["state"] == "same" and ok.checks["approval"] == "none"
    assert rlcd.check_receipt(d.decision_id, state_digest=digest(STATE)).checks["state"] == "same"
    moved = rlcd.check_receipt(d.decision_id, state={**STATE, "f": [0.0] * 6})
    assert not moved.current and any("state differs" in r for r in moved.stale_reasons)
    with pytest.raises(NotAllowed) as e:
        moved.require_current()
    assert "state differs" in str(e.value)
    old = rlcd.check_receipt(d.decision_id, state=STATE, max_age_s=1e-9)
    assert not old.current and any("older than" in r for r in old.stale_reasons)
    row = rlcd.receipt(d.decision_id)
    assert row["decision_id"] == d.decision_id and row["receipt"]["state_digest"] == digest(STATE)
    with pytest.raises(NotFound) as nf:
        rlcd.receipt("0" * 32)
    assert nf.value.status == 404 and nf.value.kind == "unknown_decision"
    with pytest.raises(ValidationError) as unknown:
        rlcd.feedback(False, source="client-test", decision_id="0" * 32, head="good")
    assert unknown.value.kind == "unknown_decision"
    with pytest.raises(ValueError):
        rlcd.check_receipt(None, state=STATE)       # a fallback has no decision_id, so no receipt to check


def test_rank_reads_ranked_candidates(rlcd):
    r = rlcd.rank(None, [{"id": "a", "state": STATE}, ("b", STATE)], receipt=True)
    assert [c.id for c in r.candidates] == ["a", "b"] and [c.rank for c in r.candidates] == [1, 2]
    assert r.head == "good" and all(c.probability == 0.5 and c.answer.band() == "low" for c in r.candidates)
    assert r.receipt["id"] and r.model == "rlcd-0.1.0"


def test_automation_switch_and_its_environment_lock(rlcd, server, monkeypatch):
    view = rlcd.automation()
    assert view["automation"]["enabled"] is True and view["rollout"]["mode"] == "full" and view["history"] == []
    off = rlcd.set_automation(False, reason="incident 42", by="client-test")
    assert off["automation"]["enabled"] is False and off["history"][-1]["reason"] == "incident 42"
    assert rlcd.health()["automation"] is False
    assert rlcd.set_automation(True)["automation"]["enabled"] is True
    monkeypatch.setenv("RLCD_AUTOMATION", "off")
    sent: list[httpx.Request] = []
    server.event_hooks = {"request": [sent.append], "response": []}
    with pytest.raises(Conflict) as e:
        rlcd.set_automation(True, reason="try to re-enable")
    assert e.value.kind == "automation_locked" and not e.value.retryable and len(sent) == 1


def test_views_and_an_evaluator_the_pack_does_not_have(rlcd):
    h = rlcd.health()
    assert h["status"] == "ok" and h["model"] == "rlcd-0.1.0" and h["pack"] == "toy"
    assert rlcd.models()["data"][0] == {"id": "rlcd-latest", "alias_of": "rlcd-0.1.0"}
    assert [x["name"] for x in rlcd.heads()["heads"]] == ["good", "color", "level"]
    with pytest.raises(ValidationError) as e:
        rlcd.calibration("good")
    assert e.value.kind == "untrained"
    with pytest.raises(ValidationError) as e:
        rlcd.evaluate(STATE)
    assert e.value.kind == "unsupported"


def test_api_key_from_the_environment_or_the_constructor(server, monkeypatch):
    monkeypatch.setenv("RLCD_API_KEY", "s3cret")
    assert Client(http=server).heads()["model"] == "rlcd-0.1.0"          # RLCD_API_KEY is read by the client too
    wrong = Client(http=server, api_key="wrong")
    with pytest.raises(AuthError) as e:
        wrong.heads()
    assert e.value.status == 401 and e.value.kind == "unauthorized"
    assert wrong.health()["status"] == "ok"                               # /health stays open
    assert "s3cret" not in repr(Client(http=server)) and "wrong" not in repr(wrong)


# ----------------------------------------------------------------------------- retries (scripted transport)

def test_a_retryable_503_is_repeated_after_retry_after_then_answers():
    s = Script(UNAVAILABLE, (200, S1))
    c, slept = mocked(s)
    r = c.system_one(STATE, {"a": Noul("good")})
    assert s.calls == 2 and slept == [2.0] and r.transport_retries == 1 and c.transport_retries == 1
    assert r.answers["a"].noul == 0.9
    # a transport retry repeats one logical call unchanged; changing the evidence is the caller's workflow retry
    assert s.requests[0].content == s.requests[1].content


def test_retry_after_is_capped_by_max_backoff():
    later = format_datetime(datetime.now(UTC) + timedelta(seconds=60), usegmt=True)
    for header in ("30", later):
        s = Script((503, err(503, "encoder_unavailable", True), {"Retry-After": header}), (200, S1))
        c, slept = mocked(s, max_backoff=3.0)
        assert c.system_one(STATE, {"a": Noul()}).transport_retries == 1 and slept == [3.0], header


def test_backoff_without_retry_after_is_exponential_jittered_and_bounded():
    s = Script(httpx.ConnectError, httpx.ConnectError, httpx.ConnectError, (200, S1))
    c, slept = mocked(s, max_retries=3, backoff=0.2, max_backoff=0.5)
    r = c.system_one(STATE, {"a": Noul()})
    assert s.calls == 4 and r.transport_retries == 3 and c.transport_retries == 3
    for delay, cap in zip(slept, [0.2, 0.4, 0.5], strict=True):     # equal jitter: [cap / 2, cap]
        assert cap / 2 <= delay <= cap


def test_transport_failures_give_up_after_max_retries():
    s = Script(httpx.ConnectError)
    c, slept = mocked(s, max_retries=2)
    with pytest.raises(TransportError) as e:
        c.system_one(STATE, {"a": Noul()})
    assert s.calls == 3 and len(slept) == 2 and e.value.transport_retries == 2 and e.value.status is None
    assert isinstance(e.value.__cause__, httpx.ConnectError) and not isinstance(e.value, Timeout)
    s = Script(httpx.ReadTimeout)
    c, _ = mocked(s, max_retries=1)
    with pytest.raises(Timeout) as t:
        c.decide(STATE)
    assert isinstance(t.value, TransportError) and s.calls == 2 and t.value.transport_retries == 1


def test_a_non_retryable_422_is_raised_once_with_its_kind():
    s = Script((422, err(422, "missing_evidence", False, question="a", path="request.text")), (200, S1))
    c, slept = mocked(s)
    with pytest.raises(ValidationError) as e:
        c.system_one(STATE, {"a": Noul("good", path="request.text")})
    assert s.calls == 1 and slept == [] and c.transport_retries == 0
    assert (e.value.status, e.value.kind, e.value.retryable) == (422, "missing_evidence", False)
    assert e.value.body["path"] == "request.text"


@pytest.mark.parametrize("status, kind, cls", [
    (401, "unauthorized", AuthError), (404, "unknown_decision", NotFound), (409, "automation_locked", Conflict),
    (413, "too_large", ValidationError), (429, "rate_limited", RateLimited), (500, "internal", ServerError),
    (503, "encoder_unavailable", Unavailable)])
def test_each_status_is_its_own_error_and_not_retried_unless_marked(status, kind, cls):
    s = Script((status, err(status, kind, False)))
    c, slept = mocked(s)
    with pytest.raises(cls) as e:
        c.decide(STATE)
    assert isinstance(e.value, RLCDError) and e.value.kind == kind and e.value.status == status
    assert s.calls == 1 and slept == []
    assert issubclass(Unavailable, ServerError)


def test_a_5xx_without_a_json_body_is_a_server_error_with_a_snippet():
    s = Script((502, b"<html><body>Bad gateway</body></html>"))
    c, _ = mocked(s)
    with pytest.raises(ServerError) as e:
        c.health()
    assert e.value.kind is None and e.value.body is None and "Bad gateway" in str(e.value) and s.calls == 1


def test_retryable_429_and_busy_409_are_repeated_then_raised():
    s = Script((429, err(429, "rate_limited", True), {"Retry-After": "1"}))
    c, slept = mocked(s)
    with pytest.raises(RateLimited) as e:
        c.system_one(STATE, {"a": Noul()})
    assert s.calls == 3 and slept == [1.0, 1.0] and e.value.transport_retries == 2 and e.value.retry_after == 1.0
    s = Script((409, err(409, "busy", True), {"Retry-After": "5"}), (200, {"automation": {"enabled": False}}))
    c, slept = mocked(s)
    assert c.set_automation(False)["automation"]["enabled"] is False and slept == [5.0]


def test_the_bearer_key_is_sent_when_set():
    s = Script((200, {"status": "ok"}))
    mocked(s, api_key="k")[0].health()
    mocked(s, api_key="")[0].health()
    assert s.requests[0].headers["authorization"] == "Bearer k" and "authorization" not in s.requests[1].headers


# ----------------------------------------------------------------------------- malformed and missing answers

@pytest.mark.parametrize("body", [
    b"<html>proxy error page</html>",
    [1, 2],
    {"answers": {"a": NOUL}},
    {"model": "rlcd-0.1.3"},
    {"model": "rlcd-0.1.3", "answers": {"a": {"type": "noul"}}},
    {"model": "rlcd-0.1.3", "answers": {"a": {**NOUL, "noul": 1.5}}},
    {"model": "rlcd-0.1.3", "answers": {"a": {**NOUL, "calibrated": "yes"}}},
    {"model": "rlcd-0.1.3", "answers": {"a": {"type": "maybe"}}},
    {"model": "rlcd-0.1.3", "answers": {"a": {"type": "choice", "choice": "x", "probabilities": {"y": 1.0}}}},
    {"model": "rlcd-0.1.3", "answers": {"a": {"type": "score", "score": "2", "probabilities": {"2": 1.0}}}},
])
def test_malformed_responses_are_never_filled_in_or_retried(body):
    s = Script((200, body))
    c, slept = mocked(s)
    with pytest.raises(MalformedResponse):
        c.system_one(STATE, {"a": Noul()})
    assert s.calls == 1 and slept == []


def test_a_question_without_an_answer_raises_missing_answer():
    s = Script((200, S1))
    c, _ = mocked(s)
    r = c.system_one(STATE, {"a": Noul("good"), "b": Choice("color")})
    assert r.missing == ["b"] and len(r.answers) == 1 and "b" not in r.answers and "a" in r.answers
    with pytest.raises(MissingAnswer) as e:
        r.answers["b"]
    assert isinstance(e.value, KeyError) and e.value.kind == "missing_answer" and "'b'" in str(e.value)
    assert r.answers["a"].value is True


# ----------------------------------------------------------------------------- fail_closed

def test_fail_closed_abstain_escalates_and_rejects_never_acts_or_ships():
    s = Script(httpx.ConnectError)
    c, slept = mocked(s, fail_closed="abstain", max_retries=1)
    d = c.decide(STATE, stakes="high")
    assert d.fallback and d.action == "escalate" and d.deny and not d.allow and not d.ask
    assert d.decision_id is None and d.receipt is None and d.heads == {} and isinstance(d.error, TransportError)
    assert "ConnectError" in d.reason and d.warnings == [d.reason] and d.transport_retries == 1
    with pytest.raises(NotAllowed):
        d.require_allow()
    ev = c.evaluate(STATE, checks={"schema": True})
    assert ev.fallback and (ev.verdict, ev.next, ev.action, ev.decided_by) == (
        "reject", "escalate_to_human", "escalate", "fallback")
    assert not ev.ship and "ConnectError" in ev.reason
    with pytest.raises(NotAllowed):
        ev.require_ship()
    assert c.fallbacks == 2 and s.calls == 4 and c.transport_retries == 2 and len(slept) == 2


def test_fail_closed_confirm_asks_a_person():
    s = Script(UNAVAILABLE)
    c, slept = mocked(s, fail_closed="confirm")
    d = c.decide(STATE)
    assert d.fallback and d.action == "confirm" and d.ask and not d.allow and isinstance(d.error, Unavailable)
    assert d.transport_retries == 2 and slept == [2.0, 2.0]
    ev = c.evaluate(STATE)
    assert ev.fallback and (ev.verdict, ev.next, ev.action) == ("review", "human_review", "confirm") and not ev.ship


def test_fail_closed_covers_unusable_answers_but_never_request_errors():
    c, _ = mocked(Script((200, b"not json")), fail_closed="abstain")
    assert c.decide(STATE).fallback and isinstance(c.decide(STATE).error, MalformedResponse)
    c, _ = mocked(Script((200, {"decision_id": "d1"})), fail_closed="abstain")      # no action: not a decision
    assert c.decide(STATE).deny
    c, _ = mocked(Script((429, err(429, "rate_limited", True))), fail_closed="abstain")
    assert isinstance(c.decide(STATE).error, RateLimited)
    for status, kind, cls in ((422, "missing_evidence", ValidationError), (401, "unauthorized", AuthError),
                              (404, "not_found", NotFound)):
        c, _ = mocked(Script((status, err(status, kind, False))), fail_closed="abstain")
        with pytest.raises(cls):
            c.decide(STATE)
        with pytest.raises(cls):
            c.evaluate(STATE)
        assert c.fallbacks == 0


def test_system_one_never_falls_back():
    c, _ = mocked(Script(httpx.ConnectError), fail_closed="abstain", max_retries=0)
    with pytest.raises(TransportError):
        c.system_one(STATE, {"a": Noul()})
    c, _ = mocked(Script((200, {"model": "m", "answers": {}})), fail_closed="abstain")
    with pytest.raises(MissingAnswer):
        c.system_one(STATE, {"a": Noul()}).answers["a"]
    assert c.fallbacks == 0


# ----------------------------------------------------------------------------- replay safety, bodies, configuration

def test_feedback_without_a_decision_id_is_not_replayed_once_it_may_have_arrived():
    s = Script(httpx.ReadError, (200, ACK))
    c, _ = mocked(s)
    with pytest.raises(TransportError) as e:
        c.feedback(True, source="client-test", head="good", state=STATE)
    assert s.calls == 1 and "not repeated" in str(e.value) and not e.value.retryable
    s = Script(httpx.ConnectError, (200, ACK))       # never reached the server: safe to send again
    c, _ = mocked(s)
    assert c.feedback(True, source="client-test", head="good", state=STATE)["stored"] and s.calls == 2
    s = Script(httpx.ReadError, (200, ACK))          # with a decision_id a repeat is a no-op correction
    c, _ = mocked(s)
    assert c.feedback(True, source="client-test", decision_id="d1", head="good")["stored"] and s.calls == 2
    assert s.body() == {"label": True, "source": "client-test", "decision_id": "d1", "head": "good"}


def test_request_bodies_are_plain_json_and_nan_is_refused_before_sending():
    s = Script((200, S1))
    c, _ = mocked(s)
    c.system_one({"f": np.array([1.0, 2.0]), "t": (1, 2)}, {"a": Noul("good")}, model="rlcd-0.1.3", strict=True)
    assert s.body() == {"state": {"f": [1.0, 2.0], "t": [1, 2]}, "questions": {"a": {"type": "noul", "head": "good"}},
                        "model": "rlcd-0.1.3", "strict": True}
    assert s.requests[-1].headers["content-type"] == "application/json"
    with pytest.raises(ValueError, match="NaN"):
        c.system_one({"f": [float("nan")]}, {"a": Noul()})
    assert s.calls == 1
    s = Script((200, DECIDE))
    c, _ = mocked(s)
    c.decide(STATE, stakes="low", heads=["good"], capabilities=["vector"])
    assert s.body() == {"state": STATE, "heads": ["good"], "capabilities": ["vector"], "stakes": "low"}
    assert s.requests[-1].url.path == "/v1/decide"
    s = Script((200, {"decision_id": "d/1", "current": True, "stale_reasons": [], "checks": {"state": "same"}}))
    c, _ = mocked(s)
    assert c.check_receipt("d/1", state_digest=digest(STATE), max_age_s=60).current
    assert s.requests[-1].url.raw_path == b"/v1/receipts/d%2F1/check"     # the id is one path segment
    assert s.body() == {"state_digest": digest(STATE), "max_age_s": 60}


def test_client_configuration(monkeypatch):
    monkeypatch.delenv("RLCD_URL", raising=False)
    monkeypatch.delenv("RLCD_API_KEY", raising=False)
    with Client() as c:
        assert c.base_url == "http://127.0.0.1:8095" and "auth=none" in repr(c)
    assert c._http.is_closed                         # a client it created is closed with it
    monkeypatch.setenv("RLCD_URL", "http://rlcd.internal:9000/")
    with Client() as c, Client("https://example.test/rlcd/") as explicit:
        assert c.base_url == "http://rlcd.internal:9000" and explicit.base_url == "https://example.test/rlcd"
    with httpx.Client(base_url="http://injected.test") as injected:
        with Client(http=injected) as c:
            assert c.base_url == "http://injected.test"
        assert not injected.is_closed                # an injected client belongs to the caller
    for bad in ({"base_url": "ftp://x"}, {"fail_closed": "maybe"}, {"max_retries": -1}, {"max_retries": True},
                {"timeout": 0}, {"timeout": None}, {"backoff": -1}):
        with pytest.raises(ValueError):
            Client(**bad)
    monkeypatch.setenv("RLCD_API_KEY", "from-env")
    with Client() as c:
        assert "from-env" not in repr(c) and "auth=bearer" in repr(c)
    with pytest.raises(TypeError):
        Client(http=httpx.Client(base_url="http://x.test")).system_one(STATE, [Noul()])


# ----------------------------------------------------------------------------- server bugs found on the way (xfail)
# They assert the correct behaviour; strict xfail flips to a failure when api.py is fixed, so the marks get removed.

def test_operator_routes_need_the_admin_key_behind_a_path_prefix(tmp_path, monkeypatch):
    monkeypatch.setenv("RLCD_ADMIN_KEY", "adm")
    svc = Service(Project(root=tmp_path), home=tmp_path, pack=ToyPack())
    prefixed = TestClient(create_app(svc), root_path="/rlcd")
    anyone = Client("http://testserver/rlcd", http=prefixed, sleep=_no_retry_expected)    # no key at all
    assert anyone.health()["status"] == "ok"            # the prefix routes: this is the same app
    with pytest.raises(AuthError):
        anyone.set_automation(False, reason="switched without the admin key")


def test_the_body_size_limit_also_holds_for_a_chunked_body(server, monkeypatch):
    monkeypatch.setenv("RLCD_MAX_BODY_BYTES", "1000")
    body = json.dumps({"state": STATE, "questions": {"good": {"type": "noul"}}, "pad": "x" * 5000}).encode()
    headers = {"content-type": "application/json"}
    assert server.post("/v1/systemone", content=body, headers=headers).status_code == 413    # with Content-Length
    chunked = server.post("/v1/systemone", content=(body[i:i + 512] for i in range(0, len(body), 512)),
                          headers=headers)
    assert chunked.request.headers.get("transfer-encoding") == "chunked"
    assert chunked.status_code == 413
