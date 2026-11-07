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

Retrieval failures: when the corpus cannot be opened or searched (a broken local vector store, an unreadable
graph, a Qdrant / Postgres / Neo4j server that cannot be reached, a missing connector extra), decide() answers 503
`connector_unavailable` instead of deciding without evidence; the message names the failure without absolute
paths, URLs, DSNs or passwords. A corpus refused for a configuration reason (encoder mismatch, including an
attached collection whose vector size is not the encoder's; unknown format; a connector configuration that
cannot work, such as a refused remote host) still gives a plan without candidates and a warning, as before.

evaluate(request) -> ship / review / reject for an answer some LLM drafted: grounded (gate), complete (only level
3 can ship, level 2 at most review) and risk_tier, which sets the stakes when the caller does not (0 low,
1 medium, 2-3 high). The risk head chooses the stakes only when it is trained, shows skill and its own decision is
above the abstain action; otherwise the answer is judged at high stakes (fail safe), with a warning. A trained
risk head whose most probable level is 3 caps the answer at review, whether that decision abstained or not.

Both log the decision as a receipt (`service.log_decision`) so POST /v1/feedback can name any head that took
part and the caller can check the receipt before it acts (POST /v1/receipts/{id}/check). Untrained heads never fail
a request: their answer is the uninformed prior and the policy turns it into the abstain action, with a warning
saying so.

The control layer (docs/control-layer.md):

  * deterministic filters run before the semantic choices: `allowed` ({response_type | route: [options]}) and
    the route contracts of rlcd.yaml `routes:` filtered by the request's `capabilities`, `sensitivity` and
    `latency_budget_ms`. A head chooses only among what survives; the chosen option's contract (and its usable
    fallback) is returned and the plan carries it (`plan.contracts`). A head whose every option was removed
    commits to nothing and the plan escalates;
  * `strict: true`: a state missing inputs almost every training row had is a 422 missing_evidence instead of an
    answer capped at confirm;
  * stale evidence: with pack_config.max_evidence_age_days, a kept chunk older than that (by `updated_at` against
    `now`) caps `answerable` at confirm; with require_evidence_dates also a kept chunk without a date;
  * `return_evidence: true` returns the kept chunks (id, doc_id, title, text, updated_at, authority, score, rank)
    so the LLM context is assembled from exactly the evidence the plan accepted;
  * the receipt binds the evidence: each kept chunk's digest is logged, and /v1/evaluate with `plan: <decision_id>`
    checks that the answer is judged against the same chunks the plan selected (otherwise at most review);
  * evaluate's decision table (core/review.py): failed deterministic `checks` -> reject without asking the heads;
    a `partial` / `none` artifact -> reject; `citations` that name chunks outside the evidence -> reject; an answer
    that leaks a secret (evaluate.output_scan, default [secrets]; add pii for e-mails and card numbers) -> reject;
    then the heads; with `plan`, the chosen response_type contract's `max_retries` is the revision budget; evidence marked `generated: true` -> at most review (support by generated text is not support);
    a failed review returns `revise` (the failed properties, the evidence ids to stay within) and a revision
    (`revision: {of: <decision_id>}`) is stopped by the loop rules (budget, the same property failing twice,
    an unchanged state).
"""
from __future__ import annotations

import re
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ...connectors.base import unavailable
from ...core import policy as PL
from ...core import receipts as R
from ...core import review as RV
from ...core.contract import MAX_CANDIDATES, ContractError
from ...core.digest import digest
from ...features.retrieval import to_epoch
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
EVIDENCE_FIELDS = ("id", "doc_id", "title", "text", "updated_at", "authority", "score", "rank")
ROUTED_HEADS = ("response_type", "route")    # the choice heads deterministic filters and route contracts apply to
_ABS_PATH = re.compile(r"(?<![\w.~])(?:[A-Za-z]:[\\/]|/)[^\s:'\"(),;]+")


def _usage(t0: float, questions: int) -> dict:
    return {"input_tokens": 0, "output_tokens": 0, "questions": questions,
            "latency_ms": round((time.perf_counter() - t0) * 1000, 2)}


def _weakest(tiers: list[str], cfg: PL.PolicyConfig) -> str:
    return PL.weakest([{"tier": t} for t in tiers], cfg)


def _log(service: Any, state: Any, r: dict) -> dict:
    return service.head_log(state, r)


def _cap(r: dict, tier: str, reason: str, cfg: PL.PolicyConfig, warnings: list[str]) -> None:
    """Lower one head decision to at most `tier` (in place) for a deterministic reason the policy cannot see."""
    names = [cfg.tier_name(i) for i in range(3)]
    d = r["decision"]
    ro = d.get("rollout") or {}
    before = ro.get("proposed_tier") or d["tier"]          # the tier before any rollout cap
    if names.index(before) > names.index(tier):
        d.setdefault("caps", []).append({"from": before, "to": tier, "reason": reason})
        if ro.get("proposed_tier"):                         # the rule binds with or without the rollout
            ro["proposed_tier"] = tier
            ro["capped"] = names.index(tier) > names.index(d["tier"])
    if names.index(d["tier"]) > names.index(tier):
        d["tier"] = tier
    if reason not in warnings:
        warnings.append(reason)


def _chunk_digest(c: dict) -> str:
    return digest({"id": c.get("id"), "title": c.get("title") or "", "text": c.get("text") or ""})


def _evidence_age(evidence: list[dict], now: Any, project: Any) -> tuple[str | None, dict]:
    """(a reason to cap answerable, the age summary) under pack_config.max_evidence_age_days."""
    cfg = getattr(project, "pack_config", None) or {}
    limit = cfg.get("max_evidence_age_days")
    need_dates = bool(cfg.get("require_evidence_dates"))
    if limit is None and not need_dates:
        return None, {}
    ref = to_epoch(now) or time.time()
    ages, undated = {}, []
    for c in evidence:
        t = to_epoch(c.get("updated_at")) if c.get("updated_at") not in (None, "") else None
        if t is None:
            undated.append(c["id"])
        else:
            ages[c["id"]] = round((ref - t) / 86400.0, 2)
    summary = {"max_age_days": limit, "age_days": ages, "undated": undated}
    if limit is not None:
        old = [i for i, a in ages.items() if a > float(limit)]
        if old:
            return (f"evidence older than max_evidence_age_days={float(limit):g}: {', '.join(old[:5])}; answerable is "
                    f"capped at confirm (stale facts must be re-checked)"), summary
    if need_dates and undated:
        return (f"evidence without updated_at: {', '.join(undated[:5])}; with require_evidence_dates its freshness "
                f"cannot be checked, so answerable is capped at confirm"), summary
    return None, summary


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
        if e.kind == "connector_unavailable":   # a store that cannot be reached: 503, never a plan without evidence
            raise ContractError(_no_paths(e.message, home), kind=e.kind, status=e.status, **e.details) from None
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
    strict = bool(request.get("strict"))
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
    # deterministic filters before any semantic choice: the request's allowed options and the route contracts
    allowed, considered = service.filter_options(specs, request, [h for h in ROUTED_HEADS if h in specs])

    if norm["candidates"] is not None:
        cands, origin = norm["candidates"], "request"
    else:
        cands, origin = _retrieve(pack, service, version, query, warnings)
    # the training rows carry `now` (the query time): without one, age is measured from the time of the request
    extra = {"now": raw["now"] if raw.get("now") not in (None, "") else _iso_now()}
    qstate = {"query": query, "candidates": cands, **extra}

    def run(name: str, state: Any, sink: list[str] | None = None, **kw: Any) -> dict:
        return service.head_decision(version, specs[name], state, stakes=stakes, explore=explore,
                                     warnings=warnings if sink is None else sink, strict=strict, **kw)

    heads: dict[str, dict] = {}
    logged: dict[str, dict] = {}
    for name in ("response_type", "route", "needs_graph"):
        if name in specs:
            if name in allowed and not allowed[name]:
                heads[name] = service.no_option_decision(version, specs[name], qstate, stakes=stakes,
                                                         warnings=warnings, considered=considered.get(name))
            else:
                heads[name] = run(name, qstate, allowed=allowed.get(name))
            service.attach_contract(heads[name], name, allowed.get(name), considered.get(name))
            logged[name] = _log(service, qstate, heads[name])

    # ---- re-rank the candidates: one batched call, so the encoder runs once for all of them
    ranked: list[dict] = []
    best_p = None
    if cands and "chunk_relevant" in specs:
        spec = specs["chunk_relevant"]
        states = [{"query": query, "chunk": c, "candidates": cands, **extra} for c in cands]
        missing: list[str] = []
        P, calibrated = service.probabilities(version, spec, states, warnings, missing_out=missing) if strict \
            else service.probabilities(version, spec, states, warnings)
        if strict and missing and calibrated:
            raise service.missing_evidence(version, spec, missing)
        p = [float(x) for x in P[:, 1]]
        order = sorted(range(len(cands)), key=lambda i: -p[i])      # stable: ties keep the retrieval order
        ranked = [{"id": cands[i]["id"], "probability": round(p[i], 6), "rank": r + 1} for r, i in enumerate(order)]
        evidence = [cands[i] for i in order if p[i] >= min_rel][:top_k]
        top = order[0]
        best_p = p[top]
        r = run("chunk_relevant", states[top], probs=P[top], calibrated=calibrated, explain=False)
        heads["chunk_relevant"] = {**r, "candidate": cands[top]["id"], "kept": len(evidence),
                                   "min_relevance": min_rel, "top_k": top_k}
        logged["chunk_relevant"] = _log(service, states[top], r)
        if not calibrated:
            warnings.append("chunk_relevant is not trained: candidates keep their retrieval order and the first "
                            f"{top_k} are used as evidence")
    else:
        ranked = [{"id": c["id"], "probability": None, "rank": i + 1} for i, c in enumerate(cands)]
        evidence = cands[:top_k]

    empty_why = (f"no candidate reached min_relevance {min_rel:g} (best P(relevant) {best_p:.2f})"
                 if best_p is not None else "there are no candidates")
    estate = {"query": query, "evidence": evidence, "candidates": cands, **extra}
    age_note, age = _evidence_age(evidence, extra["now"], service.project)
    if "answerable" in specs:
        sink: list[str] = []
        heads["answerable"] = run("answerable", estate, sink)
        for w in sink:
            if not evidence and w.startswith(f"head 'answerable': {_GUARD}"):
                # the empty evidence set is decide()'s own doing, not a field the caller forgot to send
                w = (f"head 'answerable': {empty_why}, so it judged an empty evidence set; a plan that answers "
                     f"needs evidence")
            if w not in warnings:
                warnings.append(w)
        if age_note:
            _cap(heads["answerable"], "confirm", age_note, cfg, warnings)
        if age:
            heads["answerable"]["evidence_age"] = age
        logged["answerable"] = _log(service, estate, heads["answerable"])

    # ---- the plan
    rt_dec = heads["response_type"]["decision"] if "response_type" in heads else None
    rt = rt_dec["class"] if rt_dec else None
    no_option = [n for n in ROUTED_HEADS if n in heads and heads[n]["decision"]["class"] is None]
    answers = not no_option and (rt is None or rt in ANSWER_TYPES)
    no_evidence = answers and not evidence
    chain = [n for n in ("response_type", "route") if n in heads]
    if answers and "answerable" in heads:
        chain.append("answerable")

    def plan_action(tier_of: Any) -> tuple[str, list[str]]:
        t = [abstain if (n == "answerable" and no_evidence) else tier_of(heads[n]["decision"]) for n in chain]
        return (abstain if no_evidence or no_option else _weakest(t, cfg)), t

    action, tiers = plan_action(lambda d: d["tier"])
    # what the plan would have done without the rollout cap: the same rule on the heads' uncapped tiers
    proposed_action, _ = plan_action(lambda d: service.proposed_tier(d, cfg))
    reason: str | None = None
    if no_option:
        nxt = "escalate_to_human"
        reason = (f"no allowed option for {', '.join(no_option)} survives the deterministic filters: a person "
                  f"decides")
    elif rt in NEXT_FOR and rt_dec["tier"] != abstain:
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
    contracts = {n: heads[n]["contract"] for n in ROUTED_HEADS if (heads.get(n) or {}).get("contract")}
    if contracts:
        plan["contracts"] = contracts
    latency = round((time.perf_counter() - t0) * 1000, 2)
    rollout = service.rollout_summary(list(heads.values()), action, cfg, proposed_action=proposed_action)
    row = service.log_decision(
        version, "decide", logged,
        {"plan": plan, "action": action, "stakes": stakes,
         "evidence": [{"id": c["id"], "digest": _chunk_digest(c)} for c in evidence]},
        request_state=raw, requested_model=request.get("model"), latency_ms=latency, warnings=warnings,
        audit=_audit(request), rollout=rollout,
        questions=service.decision_contract("decide", list(specs), request, stakes=stakes, top_k=top_k,
                                            min_relevance=min_rel))
    out = {"decision_id": row["decision_id"], "model": version, "action": action, "tier": action,
           "explore": explore, "plan": plan, "heads": heads, "ranked": ranked, "candidates_from": origin,
           "policy": cfg.to_dict(), "rollout": rollout, "receipt": R.summary(row), "warnings": warnings,
           "usage": _usage(t0, len(heads) + max(0, len(cands) - 1))}
    if request.get("return_evidence"):
        out["evidence"] = [{k: c.get(k) for k in EVIDENCE_FIELDS if c.get(k) is not None} for c in evidence]
    return out


def _audit(request: dict) -> dict | None:
    from ...core.service import request_audit

    return request_audit(request)


# ----------------------------------------------------------------------------- evaluate

def _failed(heads: dict, abstain: str) -> list[str]:
    """The evaluator properties that failed: grounded abstained, complete below level 3 (or abstained)."""
    out = []
    g = (heads.get("grounded") or {}).get("decision")
    if g is not None and g["tier"] == abstain:
        out.append("grounded")
    c = heads.get("complete")
    if c is not None and (c["decision"]["tier"] == abstain or int(c["answer"]["score"]) < 3):
        out.append("complete")
    return out


def _revise(failed: list[str], specs: dict, heads: dict, evidence: list[dict], ctx: dict | None,
            max_rev: int) -> dict:
    """Structured reasons for the generator's next attempt: which property failed and what must not change."""
    items = []
    for name in failed:
        spec, r = specs.get(name), heads.get(name) or {}
        d = r.get("decision") or {}
        item = {"property": name, "meaning": getattr(spec, "positive", None) or getattr(spec, "description", ""),
                "p": d.get("p"), "tier": d.get("tier")}
        # grounded is a gate head: its decision always names the class it would commit to ("true"), so whether it
        # leans favourable is P(true), not the class
        favourable = float(d.get("p") or 0.0) >= 0.5 if name == "grounded" else False
        if name == "complete":
            item["level"] = int((r.get("answer") or {}).get("score", 0))
            item["needed"] = 3
            favourable = item["level"] == 3
        # "fails": the head judged the property false; "uncertain": it leans favourable but cannot vouch for it (a
        # better answer may not help — more or clearer evidence, or a person, may)
        item["why"] = "uncertain" if favourable else "fails"
        groups = r.get("reason_groups")
        if groups:
            item["reason_groups"] = groups
        items.append(item)
    attempt = (ctx or {}).get("attempt", 1)
    return {"failed": items, "evidence_ids": [c.get("id") for c in evidence], "attempt": attempt,
            "attempts_left": max(0, max_rev - (attempt - 1)),
            "keep": "answer only from the listed evidence ids; do not add claims that are not in them"}


def evaluate(pack: Any, service: Any, request: dict) -> dict:
    t0 = time.perf_counter()
    version = service.resolve(request.get("model"))
    artifact = RV.parse_artifact(request.get("artifact"))
    raw_state = request.get("state")
    if artifact is not None and artifact["status"] == "none" and isinstance(raw_state, dict) \
            and not (isinstance(raw_state.get("answer"), str) and raw_state["answer"].strip()):
        # no answer was produced: there is nothing to normalise, the artifact rule below rejects it
        raw_state = {**raw_state, "answer": "(no answer was produced)"}
    norm = normalise_state("query_evidence_answer", 0, raw_state)                    # 1. malformed state -> 422
    # the evaluator view has no age features, so `now` does not matter here (training rows carry none either)
    state = {"query": norm["query"], "evidence": norm["evidence"], "answer": norm["answer"]}
    explore = bool(request.get("explore"))
    warnings: list[str] = []
    cfg = service.policy(version)
    abstain = cfg.tier_name(0)
    checks = RV.parse_checks(request.get("checks"))
    of = RV.parse_revision(request.get("revision"))
    max_rev = RV.max_revisions(service.project)
    ctx = None
    if of is not None:
        ctx = RV.revision_context(service.logs.decision(of), of, digest(request.get("state")))
    evidence_ids = [c.get("id") for c in norm["evidence"]]
    pre = RV.precheck(artifact, checks)                                              # 2. checks, 3. artifact
    if pre is None and "citations" in request:
        cites = request.get("citations")
        if cites is None:
            warnings.append("the generator supplied no citations: which evidence supports which claim is unknown")
        elif not isinstance(cites, list) or not all(isinstance(c, str) for c in cites):
            raise ContractError("citations must be an array of evidence ids (or null when the generator gives none)",
                                field="citations")
        else:
            outside = [c for c in cites if c not in evidence_ids]
            if outside:
                pre = {"rule": "citations", "outside": outside,
                       "reason": f"the answer cites {outside}, which are not in the evidence it is judged against"}
    if pre is None:
        pre = RV.output_scan(norm["answer"], service.project)
    plan_note = None
    plan_id = request.get("plan")
    if plan_id is None and ctx is not None:
        # a revision inherits the plan of the evaluation it revises (and so its evidence binding and retry budget)
        prev_audit = (((service.logs.decision(ctx["of"]) or {}).get("receipt") or {}).get("audit") or {})
        plan_id = prev_audit.get("plan") if isinstance(prev_audit.get("plan"), str) else None
    if plan_id is not None:
        if not isinstance(plan_id, str):
            raise ContractError("plan must be the decision_id of the /v1/decide plan this answer follows",
                                field="plan")
        planned = service.logs.decision(plan_id)
        if planned is None or planned.get("kind") != "decide":
            raise ContractError(f"plan: {plan_id!r} is not a logged /v1/decide decision", kind="unknown_decision",
                                field="plan")
        bound = {e["id"]: e["digest"] for e in planned.get("evidence") or []}
        extra_ids = [c.get("id") for c in norm["evidence"] if c.get("id") not in bound]
        changed = [c.get("id") for c in norm["evidence"] if c.get("id") in bound and bound[c.get("id")] != _chunk_digest(c)]
        budget = (((planned.get("plan") or {}).get("contracts") or {}).get("response_type") or {}).get("contract")
        if isinstance(budget, dict) and isinstance(budget.get("max_retries"), int) and budget["max_retries"] >= 0:
            max_rev = int(budget["max_retries"])      # the route contract the plan chose binds the revision loop
        if extra_ids or changed:
            parts = []
            if extra_ids:
                parts.append(f"evidence the plan did not select: {extra_ids[:5]}")
            if changed:
                parts.append(f"evidence changed since the plan: {changed[:5]}")
            plan_note = (f"the answer is judged against other evidence than plan {plan_id} selected ("
                         f"{'; '.join(parts)}): the review is not of the same evidence, so it is at most review")

    specs = _specs(service, version, EVALUATE_HEADS, warnings) if pre is None else {}
    heads: dict[str, dict] = {}
    stakes, stakes_from, decided_by = request.get("stakes"), "request", "heads"
    proposed_action = None
    if pre is not None:
        action, decided_by = abstain, pre["rule"]
        warnings.append(pre["reason"])
        if not stakes:
            stakes, stakes_from = "high", "fail_safe"
    else:
        def run(name: str, st: str | None) -> dict:
            return service.head_decision(version, specs[name], state, stakes=st, explore=explore, warnings=warnings,
                                         strict=bool(request.get("strict")))

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
        if "grounded" in specs:
            heads["grounded"] = run("grounded", stakes)
        if "complete" in specs:
            heads["complete"] = run("complete", stakes)
        generated = [c.get("id") for c in norm["evidence"] if c.get("generated") is True]

        def verdict_action(tier_of: Any) -> tuple[str, str]:
            """(action, the action of the heads alone) under one tier reading."""
            t = []
            if "grounded" in heads:
                t.append(tier_of(heads["grounded"]["decision"]))
            if "complete" in heads:
                c = heads["complete"]
                level, ct = int(c["answer"]["score"]), tier_of(c["decision"])
                # a confident "answers a small part" must not ship: only level 3 can act, level 2 at most review
                t.append(ct if level == 3 else _weakest([ct, "confirm"], cfg) if level == 2 else abstain)
            alone = _weakest(t, cfg) if t else abstain
            # deterministic caps: the risk-3 level, generated evidence, evidence other than the plan selected
            return ("confirm" if alone == "act" and (risk_top == 3 or generated or plan_note) else alone), alone

        action, alone = verdict_action(lambda d: d["tier"])
        proposed_action, _ = verdict_action(lambda d: service.proposed_tier(d, cfg))
        if risk_top == 3 and alone == "act":
            warnings.append("risk_tier 3 (legal, medical, safety, irreversible) is the most likely level: the answer "
                            "is capped at review")
        if generated:
            warnings.append(f"evidence {generated[:5]} is marked generated: support by generated text is not "
                            f"support by a source, so the answer is at most reviewed")
        if plan_note:
            warnings.append(plan_note)
    failed = _failed(heads, abstain) if pre is None else []
    stop = RV.revision_verdict(ctx, failed, max_rev)                                  # 4. the revision loop
    if stop is not None:
        action, decided_by = abstain, "revision"
        proposed_action = abstain          # the loop stops whatever the rollout: nothing was capped here
        warnings.append(stop)
    verdict = VERDICT.get(action, "reject")
    out_extra: dict[str, Any] = {}
    if verdict != "ship" and failed and stop is None:
        out_extra["revise"] = _revise(failed, specs, heads, norm["evidence"], ctx, max_rev)
    # what the caller does next: ship; a person reviews; the generator revises (a property failed and the loop
    # may go on); generate again (partial / missing artifact); or escalate (failed checks, a stopped loop, no
    # verdict the model can vouch for)
    if verdict == "ship":
        nxt = "ship"
    elif verdict == "review":
        nxt = "human_review"
    elif decided_by == "artifact":
        nxt = "retry_generation"
    elif "revise" in out_extra and decided_by == "heads" and out_extra["revise"]["attempts_left"] > 0:
        nxt = "revise"
    else:
        nxt = "escalate_to_human"
    latency = round((time.perf_counter() - t0) * 1000, 2)
    rollout = service.rollout_summary(list(heads.values()), action, cfg, proposed_action=proposed_action)
    audit = _audit(request) or {}
    if ctx is not None:
        audit["revision"] = {"of": ctx["of"], "attempt": ctx["attempt"]}
    if plan_id is not None:
        audit["plan"] = plan_id
    # the request fields the decision table reads besides the state belong to the question contract, so two
    # evaluations of one state that differ in them never share a contract_digest (provenance such as the artifact's
    # model or latency is kept in `audit`, not here: it cannot change the verdict)
    contract: dict[str, Any] = {
        "stakes": request.get("stakes"), "plan": plan_id, "revision": of,
        "checks": {n: c["passed"] for n, c in checks.items()} if checks is not None else None,
        "artifact": artifact["status"] if artifact is not None else None}
    if "citations" in request:
        contract["citations"] = request.get("citations")
    row = service.log_decision(
        version, "evaluate", {n: _log(service, state, r) for n, r in heads.items()},
        {"verdict": verdict, "next": nxt, "action": action, "stakes": stakes, "decided_by": decided_by,
         "failed": failed},
        request_state=request.get("state"), requested_model=request.get("model"), latency_ms=latency,
        warnings=warnings, audit=audit or None, rollout=rollout,
        questions=service.decision_contract("evaluate", list(specs) or list(EVALUATE_HEADS), request, **contract))
    return {"decision_id": row["decision_id"], "model": version, "action": action, "tier": action, "verdict": verdict,
            "next": nxt, "decided_by": decided_by, "stakes": stakes, "stakes_from": stakes_from, "explore": explore,
            "heads": heads, "policy": cfg.to_dict(), "rollout": rollout, "receipt": R.summary(row),
            **({"revision": {"of": ctx["of"], "attempt": ctx["attempt"], "max_revisions": max_rev}} if ctx else {}),
            **out_extra, "warnings": warnings, "usage": _usage(t0, len(heads))}
