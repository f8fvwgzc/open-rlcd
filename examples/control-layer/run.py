"""The control layer end to end, offline: open-rlcd around a (stand-in) LLM.

    uv run python examples/control-layer/run.py            # rollout mode full
    uv run python examples/control-layer/run.py assist     # every automatic action becomes a recommendation

It imports the shipped demo bundle (synthetic data, offline hashing encoder) into a throwaway home, serves it
in-process behind two keys (RLCD_API_KEY for decision callers, RLCD_ADMIN_KEY for operators and reviewers), and
walks one request through the path docs/control-layer.md describes:

    preflight battery -> /v1/decide (route contracts filter first) -> branch on the action: deny stops, a confirm
    plan waits for a person -> the plan's receipt is checked right before its context goes to the "LLM" -> the
    "LLM" writes from the kept evidence and cites it -> /v1/evaluate (deterministic checks, citations, plan
    binding, then the heads; at most one revision here) -> the evaluation's receipt is checked right before
    anything ships -> the reviewer's label and the request's outcome -> metrics and the pre-launch checklist

Every way out fails closed: a decision or an evaluation the service cannot give (the client's fail_closed
fallback), a plan that does not call the LLM, an approval that no longer holds — each stops before the next side
effect (generation, or shipping the answer). The plan's outcome says how the request ended; a decide fallback has
no decision_id, so there is no decision to post an outcome for: count those in your own metrics.

Two stand-ins, each labelled where it acts: the LLM copies the first sentence of the best chunk and cites that
chunk; the reviewer who confirms a confirm plan is a second client holding the admin key (in production a person
runs `rlcd receipts confirm ID --by NAME`, or your review tool posts the confirmation with that key). The point is
the control flow and what each step returns, not the prose. The demo bundle's heads are weak (hashing features,
synthetic data), so expect cautious decisions: that is the policy working, not failing.
"""
from __future__ import annotations

import contextlib
import json
import os
import secrets
import sys
import tempfile
import textwrap
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

from open_rlcd.client import AuthError, Choice, Client, NotFound, Noul, digest
from open_rlcd.core.config import Project
from open_rlcd.core.service import Service
from open_rlcd.serve.api import create_app

ROOT = Path(__file__).resolve().parents[2]
QUERY = "What is the warranty of the Velobox 300?"
CANDIDATES = [
    {"id": "velobox#0", "title": "Velobox 300", "text": "The Velobox 300 has a warranty of 24 months. It weighs 3.2 kg.",
     "updated_at": "2026-09-01T00:00:00Z"},
    {"id": "velobox#1", "title": "Velobox 300", "text": "The Velobox 300 costs 3,900 USD and ships in 5 days.",
     "updated_at": "2026-09-01T00:00:00Z"},
    {"id": "news#7", "title": "Company news", "text": "The robotics team moved to the new building in June.",
     "updated_at": "2026-06-20T00:00:00Z"},
]
MAX_AGE_S = 900          # how old a receipt may be when it is checked: long enough for a person's review turnaround
ARTIFACT = {"status": "complete", "model": "stand-in", "finish_reason": "stop"}
TO_A_PERSON = ("escalate_to_human", "human_review")


def show(title: str, obj: object, limit: int = 2400) -> None:
    text = json.dumps(obj, indent=1, ensure_ascii=False, default=str)
    if len(text) > limit:
        text = text[:limit] + f"\n... ({len(text) - limit} more characters not shown)"
    print(f"\n== {title}\n{text}")


def stand_in_llm(query: str, evidence: list[dict]) -> tuple[str, list[str]]:
    """Not an LLM: the first sentence of the best chunk, and the id of that chunk as its citation, so the example runs
    offline. A real generator returns the ids its claims rest on; never derive citations from the evidence list."""
    if not evidence:
        return "", []
    best = evidence[0]
    return str(best.get("text") or "").split(". ")[0].rstrip(".") + ".", [best["id"]]


@contextlib.contextmanager
def _two_keys() -> Iterator[tuple[str, str]]:
    """Two keys, as in production: the application holds the decision key (RLCD_API_KEY), operators and reviewers
    the admin key (RLCD_ADMIN_KEY). The in-process server reads them from this process's environment on every
    request; the environment is restored afterwards."""
    saved = {k: os.environ.get(k) for k in ("RLCD_API_KEY", "RLCD_ADMIN_KEY")}
    keys = ("demo-caller-" + secrets.token_hex(8), "demo-admin-" + secrets.token_hex(8))
    os.environ["RLCD_API_KEY"], os.environ["RLCD_ADMIN_KEY"] = keys
    try:
        yield keys
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def walk(mode: str = "full", *, home: Path | None = None) -> dict[str, Any]:
    """One request through the control layer. Returns what happened, in order (`events`), for the caller to read."""
    with _two_keys() as (api_key, admin_key):
        return _walk(mode, home or Path(tempfile.mkdtemp(prefix="rlcd-control-layer-")), api_key, admin_key)


def _walk(mode: str, home: Path, api_key: str, admin_key: str) -> dict[str, Any]:
    project = Project(root=home, data={
        "pack": "documents", "pack_config": {"encoder": {"kind": "hashing", "dim": 256}},
        "rollout": {"mode": mode},
        # what each retrieval route commits the application to; the request says what is available right now
        "routes": {"route": {"graph": {"requires": ["graph"], "fallback": "hybrid"},
                             "hybrid": {"requires": ["vector", "graph"], "fallback": "vector"},
                             "vector": {"requires": ["vector"], "context_budget_tokens": 2000}}},
        "owners": {"pack": "search-team", "heads": "search-team", "policy": "risk", "review": "support",
                   "incidents": "on-call"}})
    svc = Service(project, home=home / "var")
    version = svc.registry.import_(ROOT / "bundles" / "demo-documents")
    app = create_app(svc)
    # in-process HTTP that answers like a real server: an internal error is a 500 response, not an exception, so the
    # client's fail_closed fallback is what the application sees
    rlcd = Client(http=TestClient(app, raise_server_exceptions=False), api_key=api_key, fail_closed="abstain")
    reviewer = Client(http=TestClient(app, raise_server_exceptions=False), api_key=admin_key)  # a person's review tool
    out: dict[str, Any] = {"home": str(home), "svc": svc, "client": rlcd, "events": [], "workflow_retries": 0}
    events = out["events"]
    print(f"serving the demo bundle as {version} (synthetic data) in rollout mode {mode!r}")

    # One state for every call, with the request time: the query heads were trained with document ages measured
    # from the query time (/v1/decide fills `now` in when it is missing; /v1/systemone and /v1/rank do not).
    state = {"query": QUERY, "candidates": CANDIDATES, "now": datetime.now(UTC).isoformat(timespec="seconds")}

    # 1. preflight: typed questions on the request, one call. The battery is versioned; the receipt records its
    # version and digest, so `rlcd check` notices a battery changed without a version bump
    pre = rlcd.system_one(state, battery="preflight@1", receipt=True)
    show("preflight battery", {q: {"answer": a.choice if a.type == "choice" else a.noul, "band": a.band(),
                                   "skill": a.skill} for q, a in pre.answers.items()})
    kind = rlcd.system_one(state, {
        "kind": Choice(head="response_type", allowed=["direct_answer", "summarize", "escalate_human"]),
        "multi_hop": Noul(head="needs_graph")})
    show("the same request, with deterministic code allowing only three response types",
         {"kind": kind.answers["kind"].raw, "warnings": kind.warnings})
    events.append("preflight")

    # 2. decide: route contracts filter by what is available (no graph store today), then the plan
    d = rlcd.decide(state, capabilities=["vector"], baseline={"route": "vector", "next": "answer"},
                    return_evidence=True)
    out["plan_id"] = d.decision_id
    if d.fallback:
        # fail closed: the service could not answer, so nothing is generated and a person takes the request (a
        # fallback has no decision_id, so there is no decision to post an outcome for: count it in your metrics)
        print(f"\nthe decision is a fallback ({d.reason}): no generation, a person takes the request")
        events.append("fallback:decide")
        return _finish(out, svc)
    plan = d.plan or {}
    show("decide: the plan", {"action": d.action, "plan": plan, "rollout": d.rollout, "receipt": d.receipt})
    events.append(f"decide:{d.action}")

    # 3. branch on the action before anything leaves for the LLM
    if d.deny or not plan.get("call_llm"):
        print(f"\nno generation: action {d.action}, next {plan.get('next')} ({plan.get('reason')})")
        events.append("stop")
        return _finish(out, svc, note=f"no generation: action {d.action}, next {plan.get('next')}")
    if d.ask:
        # a confirm plan: a person reviews the route and the evidence before its context is sent anywhere.
        # Demonstration only, and it writes nothing: with the decision key, the application asks to confirm a
        # decision id that does not exist. A server that enforces the admin key on the confirm route refuses the
        # decision key before it looks (401). Any other answer (404) means a decision caller could confirm its
        # real decisions too, so a person's confirmation is not enforced here: stop.
        try:
            rlcd.confirm_receipt("probe-" + secrets.token_hex(8), by="the-application")
        except AuthError as e:
            print(f"\nthe application cannot confirm decisions: {e}")
            events.append("confirm:self-refused")
        except NotFound:
            print("\nthe decision key is accepted on the confirm route (the server has no RLCD_ADMIN_KEY, or the "
                  "application holds it): stopping")
            events.append("confirm:self-accepted")
            return _finish(out, svc, note="stopped: the decision key is accepted on the confirm route, so a "
                                          "person's confirmation is not enforced (set RLCD_ADMIN_KEY on the server)")
        c = reviewer.confirm_receipt(d.decision_id, by="reviewer@example.com", state_digest=digest(state),
                                     note="checked the route and the evidence")
        show("a reviewer (stand-in) confirmed the plan it saw", c)
        events.append("confirm:reviewer")

    # 4. right before the side effect (here: sending the plan's context to the LLM), does the approval still hold?
    chk = rlcd.check_receipt(d.decision_id, state=state, max_age_s=MAX_AGE_S)
    show("receipt check of the plan, right before the LLM call", {"current": chk.current,
                                                                   "stale_reasons": chk.stale_reasons,
                                                                   "checks": chk.checks})
    changed = rlcd.check_receipt(d.decision_id, state={**state, "query": QUERY + " And its price?"})
    show("the same receipt for a changed request", {"current": changed.current,
                                                     "stale_reasons": changed.stale_reasons})
    if not chk.current:
        print("\nthe plan's approval no longer holds: no generation")
        events.append("stale:plan")
        return _finish(out, svc, note="no generation: the plan's receipt was not current")
    events.append("check:plan")

    # 5. the LLM writes from the kept evidence only and cites it; your deterministic checks run first
    evidence = list(d.raw.get("evidence") or [])
    answer, cited = stand_in_llm(QUERY, evidence)
    events.append("llm")
    route = plan.get("route")
    ev_state = {"query": QUERY, "evidence": evidence, "answer": answer}
    checks = {"not_empty": bool(answer.strip()), "no_urls": "http" not in answer}
    e = rlcd.evaluate(ev_state, plan=d.decision_id, citations=cited, artifact=ARTIFACT, checks=checks)
    show("evaluate: the verdict", {"answer": answer, "citations": cited, "verdict": e.verdict, "next": e.next,
                                   "decided_by": e.decided_by, "revise": e.revise, "warnings": e.warnings})
    events.append(f"evaluate:{e.verdict}:{e.next}")
    final = e
    if not e.fallback and e.verdict == "reject" and e.next == "revise":
        # the generator would rewrite from e.revise; the stand-in cannot, so it sends the same answer again (a
        # workflow retry): the loop stops, because asking again over the same evidence is not a recovery
        out["workflow_retries"] += 1
        final = rlcd.evaluate(ev_state, revision={"of": e.decision_id}, plan=d.decision_id, citations=cited,
                              artifact=ARTIFACT, checks=checks)
        if final.fallback:
            print(f"\nthe revision could not be evaluated ({final.reason})")
        else:
            show("a revision that changed nothing", {"verdict": final.verdict, "decided_by": final.decided_by,
                                                     "next": final.next, "why": final.warnings[-1:]})
        events.append(f"revision:{final.verdict}:{final.next}")
    if final.fallback:
        # fail closed: the review is unavailable, so nothing ships and a person takes the request (a fallback has
        # no receipt to check)
        print(f"\nthe evaluation is a fallback ({final.reason}): nothing ships, a person takes the request")
        events.append("fallback:evaluate")
        return _finish(out, svc, route=route, generated=True,
                       note="the review was unavailable (fail closed): a person takes the request, nothing shipped")
    out["evaluation_id"] = final.decision_id

    # 6. right before anything ships: does the evaluation still approve exactly this answer?
    if final.ship:
        if not rlcd.check_receipt(final.decision_id, state=ev_state, max_age_s=MAX_AGE_S).current:
            print("\nthe evaluation's approval no longer holds: nothing ships")
            events.append("stale:evaluation")
            return _finish(out, svc, route=route, generated=True, note="not shipped: the evaluation's receipt was "
                                                                       "not current")
        print("\nthe answer ships")
        events.append("ship")
        note = "shipped: task success not known yet"
    else:
        held = rlcd.check_receipt(final.decision_id, state=ev_state, max_age_s=MAX_AGE_S)
        show("the evaluation's receipt: nothing ships", {"verdict": final.verdict, "next": final.next,
                                                         "current": held.current,
                                                         "stale_reasons": held.stale_reasons})
        events.append("held")
        note = ("escalated to a person: no answer shipped yet" if final.next in TO_A_PERSON
                else f"not shipped: next {final.next}")

    # 7. what happened. The person who takes an escalation labels the answer they read (the stand-in reviewer:
    # the warranty sentence is in velobox#0, so grounded is true)
    if not final.ship and final.next in TO_A_PERSON and "grounded" in final.heads:
        rlcd.feedback("true", source="reviewer", decision_id=final.decision_id, head="grounded",
                      note="the warranty sentence is in velobox#0")
        events.append("feedback")
    return _finish(out, svc, route=route, generated=True, note=note)


def _finish(out: dict, svc: Service, *, route: str | None = None, generated: bool = False,
            note: str | None = None) -> dict:
    """The request's outcome, then what the operators read: metrics, the checklist, the receipt chains.

    The outcome goes to the plan (it carries the route the plan chose) and says what is known now: the route taken,
    the provider and cost when the LLM ran, both retry counts, and a note on how the request ended. Task success is
    not known yet even for an answer that shipped: post the outcome again with `task_success` and `verified_by` once
    a downstream signal or a person's resolution says what happened (the last outcome of a decision counts). A decide
    fallback has no decision_id: nothing is posted for it."""
    rlcd = out["client"]
    if out.get("plan_id"):
        fields: dict[str, Any] = {"retries": {"transport": rlcd.transport_retries,
                                              "workflow": out["workflow_retries"]}}
        if generated:
            fields.update(route_taken=route, provider="stand-in", cost=0.0)
        if note:
            fields["note"] = note
        rlcd.outcome(out["plan_id"], **fields)
        out["events"].append("outcome")

    m = svc.metrics_view()
    show("metrics (excerpt)", {"actions": m["actions"], "coverage": m["coverage"]})
    print("notes:")
    for line in m["notes"]:
        print(textwrap.fill(line, 110, initial_indent="  - ", subsequent_indent="    "))
    from open_rlcd.core.checklist import format_table, run

    out["check"] = run(svc)
    print("\n== rlcd check\n" + format_table(out["check"]))
    v = svc.verify_receipts()
    conf = v.get("confirmations") or {}
    out["verify"] = v
    n = conf.get("receipts", 0)
    print(f"\nreceipt chains: {v['receipts']} receipts (head {v['head']}), {n} confirmation{'' if n == 1 else 's'} "
          f"(head {conf.get('head')}), ok={v['ok']}")
    print(f"everything lives in {out['home']} (delete it when done)")
    return out


def main(mode: str = "full") -> int:
    walk(mode)
    return 0


if __name__ == "__main__":
    sys.exit(main(*(sys.argv[1:2] or ["full"])))
