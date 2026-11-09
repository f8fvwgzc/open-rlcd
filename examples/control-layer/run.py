"""The control layer end to end, offline: open-rlcd around a (stand-in) LLM.

    uv run python examples/control-layer/run.py            # rollout mode full
    uv run python examples/control-layer/run.py assist     # every automatic action becomes a recommendation

It imports the shipped demo bundle (synthetic data, offline hashing encoder) into a throwaway home, serves it
in-process, and walks one request through the path docs/control-layer.md describes:

    preflight battery -> /v1/decide (route contracts filter first) -> the "LLM" writes from the kept evidence ->
    /v1/evaluate (deterministic checks, citations, plan binding, then the heads) -> receipt check (and a person's
    confirmation for a confirm decision) -> feedback and outcome -> metrics and the pre-launch checklist

The LLM here is a stand-in that copies the first sentence of the best chunk: the point is the control flow and
what each step returns, not the prose. The demo bundle's heads are weak (hashing features, synthetic data), so
expect cautious decisions: that is the policy working, not failing.
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

from fastapi.testclient import TestClient

from open_rlcd.client import Choice, Client, Noul
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


def show(title: str, obj: object) -> None:
    print(f"\n== {title}")
    print(json.dumps(obj, indent=1, ensure_ascii=False, default=str)[:1500])


def stand_in_llm(query: str, evidence: list[dict]) -> str:
    """Not an LLM: the first sentence of the best chunk, so the example runs offline."""
    best = evidence[0]["text"] if evidence else ""
    return best.split(". ")[0].rstrip(".") + "." if best else ""


def main(mode: str = "full") -> int:
    home = Path(tempfile.mkdtemp(prefix="rlcd-control-layer-"))
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
    rlcd = Client(http=TestClient(create_app(svc)), fail_closed="abstain")
    print(f"serving the demo bundle as {version} (synthetic data) in rollout mode {mode!r}")

    # 1. preflight: typed questions on the request, one call (the battery is versioned)
    pre = rlcd.system_one({"query": QUERY, "candidates": CANDIDATES}, battery="preflight@1")
    show("preflight battery", {q: {"answer": a.choice if a.type == "choice" else a.noul, "band": a.band(),
                                   "skill": a.skill} for q, a in pre.answers.items()})
    kind = rlcd.system_one({"query": QUERY, "candidates": CANDIDATES}, {
        "kind": Choice(head="response_type", allowed=["direct_answer", "summarize", "escalate_human"]),
        "multi_hop": Noul(head="needs_graph")})
    show("the same request, with deterministic code allowing only three response types",
         {"kind": kind.answers["kind"].raw, "warnings": kind.warnings})

    # 2. decide: route contracts filter by what is available (no graph store today), then the plan
    state = {"query": QUERY, "candidates": CANDIDATES}
    d = rlcd.decide(state, capabilities=["vector"], baseline={"route": "vector", "next": "answer"},
                    return_evidence=True)
    show("decide: the plan", {"action": d.action, "plan": d.plan, "rollout": d.rollout, "receipt": d.receipt})
    if not d.plan.get("call_llm"):
        print(f"\nthe plan does not call the LLM ({d.plan.get('next')}: {d.plan.get('reason')})")
        evidence = []
    else:
        evidence = [c for c in d.raw.get("evidence", [])]

    # 3. the LLM writes, and your own deterministic checks run first
    answer = stand_in_llm(QUERY, evidence or CANDIDATES[:1])
    ev_state = {"query": QUERY, "evidence": evidence or CANDIDATES[:1], "answer": answer}
    checks = {"not_empty": bool(answer.strip()), "no_urls": "http" not in answer}
    e = rlcd.evaluate(ev_state, plan=d.decision_id, citations=[c["id"] for c in (evidence or CANDIDATES[:1])],
                      artifact={"status": "complete", "model": "stand-in", "finish_reason": "stop"}, checks=checks)
    show("evaluate: the verdict", {"answer": answer, "verdict": e.verdict, "next": e.next,
                                   "decided_by": e.decided_by, "revise": e.revise, "warnings": e.warnings})

    if e.verdict == "reject" and e.next == "revise":
        # the generator would rewrite from e.revise; the stand-in cannot, so it sends the same answer again: the loop
        # stops (asking again over the same evidence is not a recovery)
        again = rlcd.evaluate(ev_state, revision={"of": e.decision_id}, plan=d.decision_id)
        show("a revision that changed nothing", {"verdict": again.verdict, "decided_by": again.decided_by,
                                                 "next": again.next, "why": again.warnings[-1:]})

    # 4. right before the side effect (here: sending the plan's context to the LLM), does the approval still hold?
    chk = rlcd.check_receipt(d.decision_id, state=state, max_age_s=300)
    show("receipt check of the plan", {"current": chk.current, "stale_reasons": chk.stale_reasons})
    if not chk.current and chk.checks.get("confirmation") == "missing":
        rlcd.confirm_receipt(d.decision_id, by="reviewer@example.com", note="checked the route and the evidence")
        chk = rlcd.check_receipt(d.decision_id, state=state, max_age_s=300)
        show("after a person confirmed it", {"current": chk.current, "stale_reasons": chk.stale_reasons})
    changed = rlcd.check_receipt(d.decision_id, state={**state, "query": QUERY + " And its price?"})
    show("the same receipt for a changed request", {"current": changed.current,
                                                     "stale_reasons": changed.stale_reasons})

    # 5. what happened: a label for a head, and a decision-level outcome
    rlcd.feedback("true", source="reviewer", decision_id=e.decision_id, head="grounded")
    rlcd.outcome(e.decision_id, task_success=True, route_taken="vector", provider="stand-in", cost=0.0,
                 retries={"transport": rlcd.transport_retries, "workflow": 0})

    # 6. measure, and run the pre-launch checklist
    m = svc.metrics_view()
    show("metrics (excerpt)", {"actions": m["actions"], "coverage": m["coverage"], "notes": m["notes"][:2]})
    from open_rlcd.core.checklist import format_table, run

    print("\n== rlcd check\n" + format_table(run(svc)))
    print(f"\nreceipt chain: {svc.verify_receipts()['receipts']} receipts, ok={svc.verify_receipts()['ok']}")
    print(f"everything lives in {home} (delete it when done)")
    return 0


if __name__ == "__main__":
    sys.exit(main(*(sys.argv[1:2] or ["full"])))
