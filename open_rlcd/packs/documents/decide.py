"""`/v1/decide` and `/v1/evaluate` for the documents pack.

decide(request) -> a plan for one request, before any LLM runs:

    1. response_type, route, needs_graph on the query state {query, candidates, now}
    2. candidates: the caller's, else the local corpus (`search.candidates`: Corpus.retrieve + linked documents),
       else none (with a warning)
    3. chunk_relevant re-ranks every candidate in one batch; evidence = P(relevant) >= min_relevance, at most top_k
    4. answerable on {query, evidence}
    5. plan {response_type, route, needs_graph, use_chunks, call_llm, next, chain, reason} and one overall action:
       the weakest of response_type, route and (when the plan answers) answerable. A chain of decisions is never
       stronger than its weakest link; needs_graph and chunk_relevant inform the plan but do not gate it.

The candidates are retrieved *before* the query heads run because those heads were trained with the retrieval
summary of the same retriever (`synth.py` and `tasks.py` call `Corpus.retrieve` with the same k): the state the
heads see at serve time matches the one they learned from.

`now`: the age features (r_age_*, c_age_days) measure document age against `now`. Every training row that carries
document timestamps also carries `now` (synth sends the query time, label_tasks the log's `time`), so when the
request has no `now`, decide() uses the time of the request: a missing `now` must not turn the column into ages
relative to the newest candidate, a distribution the heads never saw. The evaluator view has no age features.

answerable gates only plans that answer: for clarify / escalate / out-of-scope the right move does not depend on
the evidence, so it is reported but not part of the chain (otherwise every confident refusal would read
"escalate"). A plan that would answer from *no* evidence (no candidate reached min_relevance, or there were no
candidates) is never stronger than the abstain action and its next step is `retrieve_more`: an empty evidence set
cannot be sufficient, whatever P(answerable) says, and decide() built that state itself, so the input guard's
"send the same fields as in training" advice is replaced by a warning saying what happened.

`next`: the action of a confident non-answer response_type (clarify, escalate to a human, refuse out of scope);
`escalate_to_human` when response_type picked one of those but abstained (its argmax is a guess, not an
instruction); `retrieve_more` when there is no evidence or the answerable gate said no; `escalate_to_human` when
another link abstained; else `answer`. `reason` says why in words (None for `answer`); `call_llm` is true exactly
when next is `answer`.

Retrieval failures: when the local corpus cannot be opened or searched (a broken vector store, an unreadable
graph), decide() answers 503 `connector_unavailable` instead of deciding without evidence; the message names the
failure without absolute paths. A corpus refused for a configuration reason (encoder mismatch, unknown format)
still gives a plan without candidates and a warning, as before.

evaluate(request) -> ship / review / reject for an answer some LLM drafted: grounded (gate), complete (only level
3 can ship, level 2 at most review) and risk_tier, which sets the stakes when the caller does not (0 low,
1 medium, 2-3 high). The risk head chooses the stakes only when it is trained, shows skill and its own decision is
above the abstain action; otherwise the answer is judged at high stakes (fail safe), with a warning. A trained
risk head whose most probable level is 3 caps the answer at review, whether that decision abstained or not.

Both log the decision (`service.record`) so POST /v1/feedback can name any head that took part. Untrained heads
never fail a request: their answer is the uninformed prior and the policy turns it into the abstain action, with
a warning saying so.
"""
from __future__ import annotations

import re
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ...connectors.base import unavailable
from ...core import policy as PL
from ...core.contract import MAX_CANDIDATES, ContractError
from .provider import normalise_state
from .search import candidates
from .settings import retrieve_k

ANSWER_TYPES = ("direct_answer", "extract_fields", "summarize", "compare", "list")
NEXT_FOR = {"clarify_question": "ask_clarifying_question", "escalate_human": "escalate_to_human",
            "out_of_scope": "refuse_out_of_scope"}
DECIDE_HEADS = ("response_type", "route", "needs_graph", "chunk_relevant", "answerable")
EVALUATE_HEADS = ("grounded", "complete", "risk_tier")
RISK_STAKES = {0: "low", 1: "medium", 2: "high", 3: "high"}
VERDICT = {"act": "ship", "confirm": "review"}
_GUARD = "the request lacks"          # start of the input guard's warning (core.service.probabilities)
_ABS_PATH = re.compile(r"(?<![\w.~])(?:[A-Za-z]:[\\/]|/)[^\s:'\"(),;]+")


def _usage(t0: float, questions: int) -> dict:
    return {"input_tokens": 0, "output_tokens": 0, "questions": questions,
            "latency_ms": round((time.perf_counter() - t0) * 1000, 2)}


def _weakest(tiers: list[str], cfg: PL.PolicyConfig) -> str:
    return PL.weakest([{"tier": t} for t in tiers], cfg)


def _log(state: Any, r: dict) -> dict:
    d = r["decision"]
    return {"state": state, "class": d["class"], "p": d["p"], "tier": d["tier"]}


def _int(request: dict, key: str, default: int, lo: int, hi: int) -> int:
    v = request.get(key)
    if v is None:
        return default
    try:
        i = int(v)
    except (TypeError, ValueError):
        raise ContractError(f"{key} must be an integer", field=key) from None
    if not lo <= i <= hi:
        raise ContractError(f"{key} must be between {lo} and {hi}, got {i}", field=key)
    return i


def _float(request: dict, key: str, default: float) -> float:
    v = request.get(key)
    if v is None:
        return default
    try:
        f = float(v)
    except (TypeError, ValueError):
        raise ContractError(f"{key} must be a number between 0 and 1", field=key) from None
    if not 0.0 <= f <= 1.0:
        raise ContractError(f"{key} must be between 0 and 1, got {f}", field=key)
    return f


def _specs(service: Any, version: str, wanted: tuple[str, ...], warnings: list[str]) -> dict:
    specs = service.specs(version)
    out = {}
    for name in wanted:
        if name in specs:
            out[name] = specs[name]
        else:
            warnings.append(f"head '{name}' is disabled in this project: it takes no part in the decision")
    return out


def _no_paths(msg: str, home: Any) -> str:
    """`msg` without absolute paths (they reveal the server's file layout): the project home becomes
    `<RLCD_HOME>`, any other absolute path its last component."""
    homes = sorted({str(Path(home)), str(Path(home).resolve())}, key=len, reverse=True) if home else []

    def scrub(m: re.Match) -> str:
        p = m.group(0)
        for h in homes:
            if p == h or p.startswith(h.rstrip("\\/") + "/") or p.startswith(h.rstrip("\\/") + "\\"):
                return "<RLCD_HOME>" + p[len(h.rstrip("\\/")):]
        return re.split(r"[\\/]", p.rstrip("\\/"))[-1] or "<path>"

    return _ABS_PATH.sub(scrub, str(msg))


def _retrieve(pack: Any, service: Any, version: str, query: str, warnings: list[str]) -> tuple[list[dict], str]:
    home = getattr(service.project, "home", None)
    try:
        provider = service.provider(version)
        corpus = pack.corpus(service.project, provider)
    except ContractError as e:
        warnings.append(f"no candidates were given and retrieval is unavailable: {_no_paths(e.message, home)}")
        return [], "none"
    except Exception as e:  # a broken store: a 503, never a 500 with the store's path, never a silent empty plan
        raise unavailable(f"the local corpus cannot be opened: {type(e).__name__}: {_no_paths(str(e), home)}",
                          connector="local_corpus") from None
    if corpus is None:
        warnings.append("no candidates were given and no corpus is ingested (`rlcd ingest <folder>`): the plan has "
                        "no evidence")
        return [], "none"
    try:
        return candidates(corpus, query, retrieve_k(service.project.pack_config)), "corpus"
    except ContractError:
        raise
    except Exception as e:  # see above
        raise unavailable(f"retrieval from the local corpus failed: {type(e).__name__}: {_no_paths(str(e), home)}",
                          connector="local_corpus") from None


def _iso_now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


# ----------------------------------------------------------------------------- decide

def decide(pack: Any, service: Any, request: dict) -> dict:
    t0 = time.perf_counter()
    version = service.resolve(request.get("model"))
    raw = request.get("state")
    if not isinstance(raw, dict):
        raise ContractError("state must be an object {query, candidates?, now?}", field="state")
    norm = normalise_state("query", 0, raw)           # precise 422s for a bad query / candidates / now
    query = norm["query"]
    stakes, explore = request.get("stakes"), bool(request.get("explore"))
    top_k = _int(request, "top_k", 5, 1, MAX_CANDIDATES)
    min_rel = _float(request, "min_relevance", 0.5)
    wanted = request.get("heads") or list(DECIDE_HEADS)
    if not isinstance(wanted, list) or any(h not in DECIDE_HEADS for h in wanted):
        raise ContractError(f"heads must be a subset of {list(DECIDE_HEADS)}, got {wanted!r}", kind="unknown_head",
                            available_heads=list(DECIDE_HEADS))
    warnings: list[str] = []
    specs = _specs(service, version, tuple(h for h in DECIDE_HEADS if h in wanted), warnings)
    cfg = service.policy(version)
    abstain = cfg.tier_name(0)

    if norm["candidates"] is not None:
        cands, origin = norm["candidates"], "request"
    else:
        cands, origin = _retrieve(pack, service, version, query, warnings)
    # the training rows carry `now` (the query time): without one, age is measured from the time of the request
    extra = {"now": raw["now"] if raw.get("now") not in (None, "") else _iso_now()}
    qstate = {"query": query, "candidates": cands, **extra}

    def run(name: str, state: Any, sink: list[str] | None = None, **kw: Any) -> dict:
        return service.head_decision(version, specs[name], state, stakes=stakes, explore=explore,
                                     warnings=warnings if sink is None else sink, **kw)

    heads: dict[str, dict] = {}
    logged: dict[str, dict] = {}
    for name in ("response_type", "route", "needs_graph"):
        if name in specs:
            heads[name] = run(name, qstate)
            logged[name] = _log(qstate, heads[name])

    # ---- re-rank the candidates: one batched call, so the encoder runs once for all of them
    ranked: list[dict] = []
    best_p = None
    if cands and "chunk_relevant" in specs:
        spec = specs["chunk_relevant"]
        states = [{"query": query, "chunk": c, "candidates": cands, **extra} for c in cands]
        P, calibrated = service.probabilities(version, spec, states, warnings)
        p = [float(x) for x in P[:, 1]]
        order = sorted(range(len(cands)), key=lambda i: -p[i])      # stable: ties keep the retrieval order
        ranked = [{"id": cands[i]["id"], "probability": round(p[i], 6), "rank": r + 1} for r, i in enumerate(order)]
        evidence = [cands[i] for i in order if p[i] >= min_rel][:top_k]
        top = order[0]
        best_p = p[top]
        r = run("chunk_relevant", states[top], probs=P[top], calibrated=calibrated, explain=False)
        heads["chunk_relevant"] = {**r, "candidate": cands[top]["id"], "kept": len(evidence),
                                   "min_relevance": min_rel, "top_k": top_k}
        logged["chunk_relevant"] = _log(states[top], r)
        if not calibrated:
            warnings.append("chunk_relevant is not trained: candidates keep their retrieval order and the first "
                            f"{top_k} are used as evidence")
    else:
        ranked = [{"id": c["id"], "probability": None, "rank": i + 1} for i, c in enumerate(cands)]
        evidence = cands[:top_k]

    empty_why = (f"no candidate reached min_relevance {min_rel:g} (best P(relevant) {best_p:.2f})"
                 if best_p is not None else "there are no candidates")
    estate = {"query": query, "evidence": evidence, "candidates": cands, **extra}
    if "answerable" in specs:
        sink: list[str] = []
        heads["answerable"] = run("answerable", estate, sink)
        logged["answerable"] = _log(estate, heads["answerable"])
        for w in sink:
            if not evidence and w.startswith(f"head 'answerable': {_GUARD}"):
                # the empty evidence set is decide()'s own doing, not a field the caller forgot to send
                w = (f"head 'answerable': {empty_why}, so it judged an empty evidence set; a plan that answers "
                     f"needs evidence")
            if w not in warnings:
                warnings.append(w)

    # ---- the plan
    rt_dec = heads["response_type"]["decision"] if "response_type" in heads else None
    rt = rt_dec["class"] if rt_dec else None
    answers = rt is None or rt in ANSWER_TYPES
    no_evidence = answers and not evidence
    chain = [n for n in ("response_type", "route") if n in heads]
    tiers = [heads[n]["decision"]["tier"] for n in chain]
    if answers and "answerable" in heads:
        chain.append("answerable")
        tiers.append(abstain if no_evidence else heads["answerable"]["decision"]["tier"])
    action = abstain if no_evidence else _weakest(tiers, cfg)
    reason: str | None = None
    if rt in NEXT_FOR and rt_dec["tier"] != abstain:
        nxt, reason = NEXT_FOR[rt], f"response_type is {rt} (p {rt_dec['p']:.2f})"
    elif rt in NEXT_FOR:
        # an abstained argmax is a guess: refusing or asking back on a guess would contradict the escalate action
        nxt = "escalate_to_human"
        reason = (f"response_type abstained (most likely {rt}, p {rt_dec['p']:.2f}, below its confirm "
                  f"threshold): a person decides how to respond")
    elif no_evidence:
        nxt, reason = "retrieve_more", f"{empty_why}: there is no evidence to answer from"
    elif action == abstain:
        a = (heads.get("answerable") or {}).get("decision") or {}
        gate_said_no = a.get("calibrated") and a.get("uncapped_tier") == abstain and a.get("max_tier") != abstain
        nxt = "retrieve_more" if gate_said_no else "escalate_to_human"
        weak = [n for n, t in zip(chain, tiers, strict=True) if t == abstain]
        reason = (f"answerable: the evidence is not sufficient (P {a.get('p', 0):.2f})" if gate_said_no
                  else f"{', '.join(weak) or 'the plan'} abstained: a person decides")
    else:
        nxt = "answer"
    plan = {"response_type": rt,
            "route": heads["route"]["decision"]["class"] if "route" in heads else None,
            "needs_graph": (heads["needs_graph"]["decision"]["class"] == "true") if "needs_graph" in heads else None,
            "use_chunks": [c["id"] for c in evidence],
            "call_llm": nxt == "answer",
            "next": nxt, "chain": chain, "reason": reason}
    did = service.record(version, "decide", logged, {"plan": plan, "action": action})
    return {"decision_id": did, "model": version, "action": action, "tier": action, "explore": explore, "plan": plan,
            "heads": heads, "ranked": ranked, "candidates_from": origin, "policy": cfg.to_dict(),
            "warnings": warnings, "usage": _usage(t0, len(heads) + max(0, len(cands) - 1))}


# ----------------------------------------------------------------------------- evaluate

def evaluate(pack: Any, service: Any, request: dict) -> dict:
    t0 = time.perf_counter()
    version = service.resolve(request.get("model"))
    norm = normalise_state("query_evidence_answer", 0, request.get("state"))
    # the evaluator view has no age features, so `now` does not matter here (training rows carry none either)
    state = {"query": norm["query"], "evidence": norm["evidence"], "answer": norm["answer"]}
    explore = bool(request.get("explore"))
    warnings: list[str] = []
    specs = _specs(service, version, EVALUATE_HEADS, warnings)
    cfg = service.policy(version)
    abstain = cfg.tier_name(0)

    def run(name: str, stakes: str | None) -> dict:
        return service.head_decision(version, specs[name], state, stakes=stakes, explore=explore, warnings=warnings)

    heads: dict[str, dict] = {}
    risk_level = None              # the level the stakes follow: only from a confident, skilled risk decision
    risk_top = None                # the most probable level of a trained risk head (for the level-3 cap)
    risk_note = "risk_tier is not trained, disabled or shows no skill"
    if "risk_tier" in specs:
        heads["risk_tier"] = run("risk_tier", request.get("stakes"))
        rd = heads["risk_tier"]["decision"]
        if rd["calibrated"]:
            risk_top = int(heads["risk_tier"]["answer"]["score"])
            if rd["max_tier"] == abstain:
                risk_note = "risk_tier shows no skill"
            elif rd["tier"] == abstain:
                risk_note = (f"risk_tier abstained (most likely level {risk_top}, p {rd['p']:.2f}, below its "
                             f"confirm threshold)")
            else:
                risk_level = risk_top
    if request.get("stakes"):
        stakes, stakes_from = request["stakes"], "request"
    elif risk_level is not None:
        stakes, stakes_from = RISK_STAKES.get(risk_level, "high"), "risk_tier"
    else:
        stakes, stakes_from = "high", "fail_safe"
        warnings.append(f"{risk_note}: the answer is judged at high stakes")
    tiers = []
    if "grounded" in specs:
        heads["grounded"] = run("grounded", stakes)
        tiers.append(heads["grounded"]["decision"]["tier"])
    if "complete" in specs:
        heads["complete"] = run("complete", stakes)
        c = heads["complete"]
        level, t = int(c["answer"]["score"]), c["decision"]["tier"]
        # a confident "answers a small part" must not ship: only level 3 can act, level 2 can at most be reviewed
        tiers.append(t if level == 3 else _weakest([t, "confirm"], cfg) if level == 2 else abstain)
    action = _weakest(tiers, cfg) if tiers else abstain
    if risk_top == 3 and action == "act":
        action = "confirm"
        warnings.append("risk_tier 3 (legal, medical, safety, irreversible) is the most likely level: the answer is "
                        "capped at review")
    verdict = VERDICT.get(action, "reject")
    did = service.record(version, "evaluate", {n: _log(state, r) for n, r in heads.items()},
                         {"verdict": verdict, "action": action, "stakes": stakes})
    return {"decision_id": did, "model": version, "action": action, "tier": action, "verdict": verdict,
            "stakes": stakes, "stakes_from": stakes_from, "explore": explore, "heads": heads,
            "policy": cfg.to_dict(), "warnings": warnings, "usage": _usage(t0, len(heads))}
