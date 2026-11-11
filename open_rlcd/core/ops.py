"""Operations metrics and the review queue: what to watch once decisions run in production.

The field guide this control layer follows (docs/control-layer.md) names the numbers a team should watch once a
decision model sits in a request path: coverage, automatic-action / human-review / abstention rates, false
automation, false rejection, calibration by probability band, latency percentiles, retries, cost and verified task
success, segmented by model version and risk level (X.E); answer entropy and the low-confidence rate per question
(VIII.F); the share that leaves each stage of a cascade (VII.D); the cost and latency of the complete path, with
transport and workflow retries counted apart (VIII.D, VIII.I); the review volume against the reviewers' capacity,
and a queue that shows a reviewer what drove each decision (X.I, XI.C). This module computes them from the rows of
the three append-only logs under RLCD_HOME (core/feedback.py):

    decisions.jsonl   one record per decision: kind, model, heads {class, p, tier, stakes, probabilities,
                      uncertainty, proposed_tier, state}, action / plan / verdict / next, and its receipt
                      (latency_ms, rollout, warnings, audit). Records written before receipts have fewer fields.
    feedback.jsonl    labels per head; a later row for the same (decision_id, head) corrects an earlier one
    outcomes.jsonl    decision-level outcomes: task success, route taken, fallback, cost, latency, retries

    metrics(decisions, feedback, outcomes, ...)   the report (its `definitions` repeat the words below)
    review_queue(decisions, feedback, ...)        confirm and abstain decisions that have no feedback yet
    holdout_summary(report, decision=...)         a head's holdout numbers, made comparable with the online ones

Definitions:

  tiers / action     act (automatic), confirm (a person confirms), the abstain action (`abstain_names`; "escalate",
                     "hold" in the tabular pack). A record's action is its logged `action`; a record written before
                     actions were logged takes the weakest of its head tiers (counted as `derived`). /v1/systemone
                     and /v1/rank records have neither tiers nor an action.
  shares             automatic / review / abstain / low-confidence rates and coverage count what happened to the
                     decisions in the window: they are not estimates, so they carry no interval.
  correct            the label posted for a head equals the class the decision committed to (a gate head commits to
                     `true`, as policy.evaluate scores the holdout). A decision that committed to no class (every
                     option removed) has no correctness and is left out of accuracy and calibration.
  accuracy           correct / labelled, with a Wilson 95% interval: an estimate from the labelled decisions.
  false automation   head decisions at `act` whose committed class was wrong, over labelled `act` head decisions (a
                     head at act inside a record whose action did not act was not automated: this is the act-tier
                     error rate of the head)
  false rejection    abstained decisions whose committed class was right, over labelled abstentions.
  review agreement   confirm decisions whose committed class was right, over labelled confirms.
  policy act         decisions the policy put at `act` before any rollout cap (`proposed_tier`): what full automation
                     would have done, so it can be measured in shadow mode. Per stakes level it is the number the
                     holdout's act-tier accuracy promised, hence the one compared with the holdout.
  calibration        on (p of the committed class, correct): the 10-bin reliability table, ECE and the binary Brier
                     score of core/metrics.py. For argmax heads this is top-label calibration (with `allowed`
                     options, of the most probable allowed option); for gate heads it is the calibration of P(true).
  entropy            Shannon entropy normalised to [0, 1] (core/uncertainty.py) of the logged distribution,
                     renormalised over the options it lists (only the allowed ones when deterministic code removed
                     some), or of [p, 1 - p] for a noul head; records with fewer than two options or no distribution
                     are left out (`entropy.n` says how many count).
  binding link       what held a decide / evaluate action below act. Candidate links are the plan's `chain` when the
                     record has one (documents decide), else every head with a tier. The heads at the action's tier
                     that got there on their own (not through the rollout cap) bind; failing that, the record's
                     `failed` heads (evaluate: a score level that cannot ship); failing that, `rollout_cap` when
                     heads reached the action's tier only through the rollout cap, else `pack_rule` (no evidence, a
                     plan mismatch, generated evidence, a revision stop). An evaluate record that a deterministic
                     rule decided binds to that rule (`decided_by:<rule>`). Tied heads bind together; `sole` counts
                     records one head bound alone. Without a chain, a head that does not gate but ties with the
                     action is counted too: read binding as "consistent with having set the action". A head that
                     never binds lowered no action in the window; it may still change what the plan does (its class
                     picks the next step, the route, the evidence) or guard a rare case.

Drift (`heads.<head>.drift_suspected`, only when `holdout` is given for the head; the numbers side by side are in
`heads.<head>.vs_holdout`): the online decisions of the holdout's own model version that the policy put at act
(before any rollout cap) and that have feedback are compared, per stakes level, with the holdout's act-tier rows at
that level. Drift is suspected only when at least MIN_OUTCOMES (30) such decisions are labelled AND the Wilson 95%
upper bound of their accuracy lies below the Wilson 95% lower bound of the holdout's act-tier accuracy. With equal
accuracies, two 95% Wilson intervals separate that way by chance in at most a few percent of comparisons: under 1%
when both samples are of similar size, and up to about 6-9% for accuracies near 1 with a very large holdout (the
Wilson bound's one-sided error near 1). When the holdout gives no row count, the bound
is the holdout accuracy itself: that ignores the holdout's own sampling error and flags more readily, and the check
says so (holdout_summary always passes the counts). A flag is evidence that acting on the labelled decisions is less
accurate than the holdout showed, not a measurement of how much less; it is a reason to label a random sample,
retrain or roll back. The test cannot see a drop the labelled decisions do not contain: feedback posted mostly for
disputed decisions biases it low, feedback only for easy ones biases it high.

Review load: confirm decisions per day over the window (from `since`, or the first decision, to `until`, or the last
one; at least one day), the busiest UTC day, the pending queue, and with `review_capacity_per_day`:
`review_over_capacity` = confirm decisions per day > capacity. Abstentions are counted apart: when they also go to
people (escalate_to_human), add them.

What it does not do: read or write any file, call the service, train, or change a decision; it computes from the
rows it is given, and the feedback and outcome rows join the decisions in the window whenever they were posted. It
cannot correct selection bias: feedback and outcomes describe the decisions they were posted for and stand for all
decisions only when they are posted for a random sample (the first note says how many decisions have any). Latency
is the decision service's own (receipt.latency_ms) and whatever the caller posted to /v1/outcome; nothing here
measures the rest of the path.
"""
from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import numpy as np

from . import metrics as M
from .bernoulli import binomial_test_below
from .contract import ContractError
from .uncertainty import entropy as normalised_entropy

ABSTAIN_NAMES = ("escalate", "hold")
MIN_OUTCOMES = 30     # labelled policy-act decisions before a drift test; labelled decisions before calibration reads
MIN_RECORDS = 30      # decisions before a head is listed as never binding or never varying
UNSET = "unset"

DEFINITIONS = {
    "action": "the record's logged action; records written before actions were logged take the weakest head tier "
              "(derived)",
    "shares": "automatic / review / abstain / low_confidence / coverage: shares of the decisions in the window "
              "(counts, no interval)",
    "correct": "the label posted for a head equals the class the decision committed to (gate heads commit to true)",
    "accuracy": "correct / labelled, Wilson 95% interval",
    "false_automation": "decisions at act whose committed class was wrong / labelled act decisions",
    "false_rejection": "abstained decisions whose committed class was right / labelled abstentions",
    "review_agreement": "confirm decisions whose committed class was right / labelled confirm decisions",
    "policy_act": "decisions the policy put at act before any rollout cap: what full automation would have done",
    "calibration": "on (p of the committed class, correct): 10-bin reliability, ECE, binary Brier (top-label for "
                   "argmax heads, P(true) for gate heads)",
    "entropy": "Shannon entropy of the logged distribution, normalised to [0, 1]",
    "low_confidence": "decisions whose policy tier (before any rollout cap) is not act / decisions with a tier",
    "binding": "what held a decide / evaluate action below act: heads at the action's tier on their own, else "
               "failed heads, else rollout_cap, else pack_rule; decided_by:<rule> for evaluate rules",
    "drift_suspected": f"at least {MIN_OUTCOMES} labelled policy-act decisions of the holdout's model at a stakes "
                       f"level, and their Wilson 95% upper bound below the holdout act-tier accuracy's Wilson 95% "
                       f"lower bound",
    "task_success": "outcomes.jsonl task_success (last row per decision), Wilson 95% interval",
    "review_over_capacity": "confirm decisions per day over the window > serve.review_capacity_per_day",
}


# ----------------------------------------------------------------------------- small helpers

def _names(abstain_names: Sequence[str] | str | None) -> tuple[str, ...]:
    if abstain_names is None:
        return ABSTAIN_NAMES
    if isinstance(abstain_names, str):
        return (abstain_names,)
    return tuple(dict.fromkeys(str(a) for a in abstain_names if a))


def _when(v: Any) -> datetime | None:
    """An ISO 8601 date or time (a trailing Z allowed; naive times are UTC), in UTC. None when unreadable."""
    if isinstance(v, datetime):
        t = v
    elif isinstance(v, str) and v.strip():
        try:
            t = datetime.fromisoformat(v.strip().replace("Z", "+00:00"))
        except ValueError:
            return None
    else:
        return None
    return (t if t.tzinfo is not None else t.replace(tzinfo=UTC)).astimezone(UTC)


def _bound(v: Any, name: str) -> datetime | None:
    if v is None or v == "":
        return None
    t = _when(v)
    if t is None:
        raise ContractError(f"{name} must be an ISO 8601 date or time (e.g. 2026-11-01 or 2026-11-01T12:00:00Z), "
                            f"got {v!r}", field=name)
    return t


def _iso(t: datetime | None) -> str | None:
    return None if t is None else t.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _num(v: Any) -> float | None:
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return None
    x = float(v)
    return x if math.isfinite(x) else None


def _count(v: Any) -> int | None:
    x = _num(v)
    return int(x) if x is not None and x >= 0 and x.is_integer() else None


def _nice(x: float) -> int | float | None:
    """A sum as an int when it is whole (retry and token counts), else rounded."""
    return int(x) if float(x).is_integer() else M._f(x)


def _label(v: Any) -> str | None:
    """A label or class as the logs compare them: booleans as true / false, whole floats as integers."""
    if v is None:
        return None
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, float) and v.is_integer():
        return str(int(v))
    return str(v)


def _share(k: int, n: int) -> dict:
    return {"count": int(k), "n": int(n), "rate": M._f(k / n) if n else None}


def _rate(k: int, n: int) -> dict:
    lo, hi = M.wilson(int(k), int(n))
    return {**_share(k, n), "wilson_95": [M._f(lo), M._f(hi)]}


def _dist(c: Mapping) -> dict[str, int]:
    return {str(k): int(v) for k, v in sorted(c.items(), key=lambda kv: (-kv[1], str(kv[0])))}


def _spread(values: Iterable[float | None]) -> dict:
    """n, sum, mean, p50 / p95 / p99 (numpy's linear interpolation, as `rlcd bench`) and max."""
    xs = np.asarray([v for v in values if v is not None], float)
    if not len(xs):
        return {"n": 0, "sum": None, "mean": None, "p50": None, "p95": None, "p99": None, "max": None}
    return {"n": int(len(xs)), "sum": M._f(xs.sum()), "mean": M._f(xs.mean()),
            "p50": M._f(np.percentile(xs, 50)), "p95": M._f(np.percentile(xs, 95)),
            "p99": M._f(np.percentile(xs, 99)), "max": M._f(xs.max())}


# ----------------------------------------------------------------------------- reading one decision record

def _receipt(row: Mapping) -> Mapping:
    r = row.get("receipt")
    return r if isinstance(r, Mapping) else {}


def _plan(row: Mapping) -> Mapping | None:
    p = row.get("plan")
    return p if isinstance(p, Mapping) else None


def _next(row: Mapping) -> Any:
    """The next step a record names: evaluate's `next`, else a documents plan's `next`."""
    if row.get("next") is not None:
        return row.get("next")
    plan = _plan(row)
    return plan.get("next") if plan else None


def _entries(row: Mapping) -> list[tuple[str, str, Mapping]]:
    """(key, head name, entry) per head entry: decide / evaluate key entries by head name, a /v1/systemone receipt by
    question id with the head name in the entry."""
    heads = row.get("heads")
    if not isinstance(heads, Mapping):
        return []
    out = []
    for key, e in heads.items():
        if isinstance(e, Mapping):
            name = e.get("head")
            out.append((str(key), name if isinstance(name, str) and name else str(key), e))
    return out


def _involves(row: Mapping, head: str) -> bool:
    return row.get("head") == head or any(head in (k, n) for k, n, _ in _entries(row))


def _strength(tier: Any, abstain: Sequence[str]) -> int | None:
    if tier == "act":
        return 2
    if tier == "confirm":
        return 1
    if isinstance(tier, str) and tier in abstain:
        return 0
    return None


def _policy_strength(e: Mapping, abstain: Sequence[str]) -> int | None:
    """The head's tier before any rollout cap (`proposed_tier` is logged only when the rollout lowered it)."""
    p = _strength(e.get("proposed_tier"), abstain)
    return p if p is not None else _strength(e.get("tier"), abstain)


def _was_capped(e: Mapping, abstain: Sequence[str]) -> bool:
    p, t = _strength(e.get("proposed_tier"), abstain), _strength(e.get("tier"), abstain)
    return p is not None and t is not None and p > t


def _row_action(row: Mapping, abstain: Sequence[str]) -> tuple[str | None, bool]:
    """(action, derived): the logged action, else the weakest head tier (records from before actions were logged)."""
    a = row.get("action")
    if isinstance(a, str) and a:
        return a, False
    tiers = [(s, str(e.get("tier"))) for _, _, e in _entries(row)
             if (s := _strength(e.get("tier"), abstain)) is not None]
    if not tiers:
        return None, False
    return min(tiers)[1], True


def _entropy(e: Mapping) -> float | None:
    probs = e.get("probabilities")
    if isinstance(probs, Mapping):
        ps = [x for x in (_num(v) for v in probs.values()) if x is not None and x >= 0]
        total = sum(ps)
        if len(ps) >= 2 and total > 0:
            return normalised_entropy([x / total for x in ps])
        return None
    p = _num(e.get("p"))
    if p is not None and _label(e.get("class")) in ("true", "false"):
        p = min(1.0, max(0.0, p))
        return normalised_entropy([p, 1.0 - p])
    return None


@dataclass
class _Rec:
    """One decision record in the window, with its action read once."""
    row: Mapping
    at: datetime | None
    action: str | None
    derived: bool
    strength: int | None

    @property
    def id(self) -> str | None:
        d = self.row.get("decision_id")
        return d if isinstance(d, str) and d else None


def _select(decisions: Iterable[Any], since: datetime | None, until: datetime | None, model: str | None,
            kind: str | None, head: str | None, abstain: Sequence[str]) -> tuple[list[_Rec], int, int]:
    """The records in the window (since inclusive, until exclusive) that pass the filters, plus (undated records
    left out because a bound was given, unreadable rows)."""
    recs: list[_Rec] = []
    undated = unreadable = 0
    bounded = since is not None or until is not None
    for row in decisions or ():
        if not isinstance(row, Mapping):
            unreadable += 1
            continue
        if model is not None and row.get("model") != model:
            continue
        if kind is not None and row.get("kind") != kind:
            continue
        if head is not None and not _involves(row, head):
            continue
        at = _when(row.get("created_at"))
        if bounded:
            if at is None:
                undated += 1
                continue
            if (since is not None and at < since) or (until is not None and at >= until):
                continue
        action, derived = _row_action(row, abstain)
        recs.append(_Rec(row, at, action, derived, _strength(action, abstain)))
    return recs, undated, unreadable


# ----------------------------------------------------------------------------- joining feedback and outcomes

@dataclass
class _Feedback:
    labels: dict[tuple[str, str], str] = field(default_factory=dict)   # (decision_id, entry key) -> last label
    decisions: set[str] = field(default_factory=set)                   # decisions in the window with any feedback
    rows: int = 0
    unlinked: int = 0          # no decision_id: training rows from elsewhere
    not_in_window: int = 0     # the decision is outside the window or the filters, or unknown
    unmatched: int = 0         # the decision has no entry for that head (or the row has no label)
    ambiguous: int = 0         # a /v1/systemone receipt asked that head in several questions
    corrected: int = 0         # rows that replaced an earlier row for the same (decision_id, head)


def _match(row: Mapping, name: Any) -> tuple[str | None, str]:
    """The entry key a feedback row's head refers to (Service.feedback stores the head name, never the question id)."""
    if not isinstance(name, str) or not name:
        return None, "unmatched"
    keys = []
    for k, n, _ in _entries(row):
        if k == name:
            return k, ""
        if n == name:
            keys.append(k)
    if len(keys) == 1:
        return keys[0], ""
    return None, "ambiguous" if keys else "unmatched"


def _join_feedback(feedback: Iterable[Any], by_id: Mapping[str, Mapping], head: str | None) -> _Feedback:
    fb = _Feedback()
    for r in feedback or ():
        if not isinstance(r, Mapping) or (head is not None and r.get("head") != head):
            continue
        fb.rows += 1
        did = r.get("decision_id")
        if not isinstance(did, str) or not did:
            fb.unlinked += 1
            continue
        row = by_id.get(did)
        if row is None:
            fb.not_in_window += 1
            continue
        fb.decisions.add(did)
        key, why = _match(row, r.get("head"))
        label = _label(r.get("label"))
        if key is None or label is None:
            if why == "ambiguous":
                fb.ambiguous += 1
            else:
                fb.unmatched += 1
            continue
        if (did, key) in fb.labels:
            fb.corrected += 1
        fb.labels[(did, key)] = label      # rows are in log order: the last one wins
    return fb


@dataclass
class _Outcomes:
    by_id: dict[str, Mapping] = field(default_factory=dict)
    rows: int = 0
    unlinked: int = 0
    not_in_window: int = 0
    corrected: int = 0


def _join_outcomes(outcomes: Iterable[Any], by_id: Mapping[str, Mapping]) -> _Outcomes:
    oc = _Outcomes()
    for r in outcomes or ():
        if not isinstance(r, Mapping):
            continue
        oc.rows += 1
        did = r.get("decision_id")
        if not isinstance(did, str) or not did:
            oc.unlinked += 1
            continue
        if did not in by_id:
            oc.not_in_window += 1
            continue
        if did in oc.by_id:
            oc.corrected += 1
        oc.by_id[did] = r
    return oc


# ----------------------------------------------------------------------------- per head

@dataclass(frozen=True)
class _Labelled:
    """One head decision with feedback."""
    model: str | None
    stakes: str
    tier: int | None        # effective tier (what happened)
    policy: int | None      # tier before any rollout cap
    p: float | None
    correct: bool | None    # None: the decision committed to no class


@dataclass
class _Head:
    decisions: int = 0
    tiers: Counter = field(default_factory=Counter)
    untiered: int = 0
    low: int = 0
    classes: Counter = field(default_factory=Counter)
    no_class: int = 0
    entropy: list[float] = field(default_factory=list)
    sources: Counter = field(default_factory=Counter)
    capped: int = 0
    labelled: list[_Labelled] = field(default_factory=list)


def _act_stats(xs: list[_Labelled]) -> dict:
    """Accuracy of policy-act decisions next to their mean predicted probability (as policy.evaluate on the holdout),
    with the exact one-sided binomial test of the accuracy against it (small = the probabilities overstate)."""
    n, k = len(xs), sum(bool(o.correct) for o in xs)
    lo, hi = M.wilson(k, n)
    mp = float(np.mean([o.p for o in xs])) if n else None
    return {"n": n, "correct": k, "accuracy": M._f(k / n) if n else None, "wilson_95": [M._f(lo), M._f(hi)],
            "mean_predicted": M._f(mp),
            "p_value_below_predicted": M._f(binomial_test_below(k, n, mp)) if mp is not None else None}


def _calibration(xs: list[_Labelled]) -> dict:
    pts = [(o.p, o.correct) for o in xs if o.p is not None and o.correct is not None]
    if not pts:
        return {"n": 0, "ece": None, "brier": None, "mean_predicted": None, "observed_rate": None,
                "reliability": None}
    p = np.clip(np.asarray([a for a, _ in pts], float), 0.0, 1.0)
    y = np.asarray([float(b) for _, b in pts])
    return {"n": len(pts), "ece": M._f(M.ece(p, y)), "brier": M._f(float(((p - y) ** 2).mean())),
            "mean_predicted": M._f(p.mean()), "observed_rate": M._f(y.mean()),
            "reliability": M.reliability_table(p, y)}


def _labelled_stats(xs: list[_Labelled]) -> dict:
    scored = [o for o in xs if o.correct is not None]

    def at(strength: int) -> list[_Labelled]:
        return [o for o in scored if o.tier == strength]

    act, confirm, abstained = at(2), at(1), at(0)
    policy: dict[str, list[_Labelled]] = defaultdict(list)
    for o in scored:
        if o.policy == 2 and o.p is not None:
            policy[o.stakes].append(o)
    return {"outcomes": len(xs), "no_class": len(xs) - len(scored),
            "accuracy": _rate(sum(bool(o.correct) for o in scored), len(scored)),
            "false_automation": _rate(sum(not o.correct for o in act), len(act)),
            "false_rejection": _rate(sum(bool(o.correct) for o in abstained), len(abstained)),
            "review_agreement": _rate(sum(bool(o.correct) for o in confirm), len(confirm)),
            "policy_act": {s: _act_stats(v) for s, v in sorted(policy.items())},
            "calibration": _calibration(scored)}


def _drift_check(stakes: str, online: Mapping, ha: Mapping) -> dict:
    n, k = int(online["n"]), int(online["correct"])
    lo, hi = M.wilson(k, n)
    enough = n >= MIN_OUTCOMES
    out: dict[str, Any] = {"stakes": stakes,
                           "online": {"n": n, "accuracy": online["accuracy"], "wilson_95": [M._f(lo), M._f(hi)]}}
    hn, hk, hacc = _count(ha.get("n")), _count(ha.get("correct")), _num(ha.get("accuracy"))
    if hn and hk is None and hacc is not None:
        hk = int(round(hacc * hn))
    if hn and hk is not None and hk <= hn:
        hlo, hhi = M.wilson(hk, hn)
        bound, against = hlo, "the holdout act-tier accuracy's Wilson 95% lower bound"
        out["holdout"] = {"n": hn, "accuracy": M._f(hk / hn), "wilson_95": [M._f(hlo), M._f(hhi)]}
    elif hacc is not None:
        bound = hacc
        against = "the holdout act-tier accuracy (no holdout row count given: its own uncertainty is ignored)"
        out["holdout"] = {"n": hn, "accuracy": M._f(hacc), "wilson_95": [None, None]}
    else:
        return {**out, "holdout": None, "compared_with": None, "enough_outcomes": enough, "suspected": False,
                "why": "the holdout has no act-tier rows at this stakes level"}
    suspected = bool(enough and hi is not None and bound is not None and hi < bound)
    if not enough:
        why = f"{n} labelled policy-act decisions at {stakes} stakes; the test needs at least {MIN_OUTCOMES}"
    elif suspected:
        why = (f"online act-tier accuracy {k / n:.3f} (Wilson 95% upper bound {hi:.3f}) is below {against} "
               f"({bound:.3f})")
    else:
        why = f"the online Wilson 95% upper bound ({hi:.3f}) is not below {against} ({bound:.3f}): no drop shown"
    return {**out, "compared_with": against, "enough_outcomes": enough, "suspected": suspected, "why": why}


def _versus(xs: list[_Labelled], h: Mapping) -> dict:
    """The holdout next to the online numbers of the same model version, and the drift test per stakes level."""
    model = h.get("model")
    same = [o for o in xs if model is None or o.model == model]
    on = _labelled_stats(same)
    act = h.get("act") if isinstance(h.get("act"), Mapping) else {}
    checks = []
    for stakes in sorted(act, key=str):
        ha, oa = act[stakes], on["policy_act"].get(str(stakes))
        if isinstance(ha, Mapping) and oa is not None:
            checks.append(_drift_check(str(stakes), oa, ha))
    return {"model": model,
            "holdout": {k: h.get(k) for k in ("n", "accuracy", "ece", "brier", "act") if k in h},
            "online": {"outcomes": on["outcomes"], "accuracy": on["accuracy"], "ece": on["calibration"]["ece"],
                       "brier": on["calibration"]["brier"], "calibration_n": on["calibration"]["n"],
                       "policy_act": on["policy_act"]},
            "drift": checks, "drift_suspected": any(c["suspected"] for c in checks)}


def _heads(recs: list[_Rec], fb: _Feedback, abstain: Sequence[str], head: str | None) -> dict[str, _Head]:
    acc: dict[str, _Head] = {}
    for r in recs:
        row_stakes = r.row.get("stakes")
        for key, name, e in _entries(r.row):
            if head is not None and name != head:
                continue
            h = acc.setdefault(name, _Head())
            h.decisions += 1
            tier = e.get("tier")
            if tier is None:
                h.untiered += 1
            else:
                h.tiers[str(tier)] += 1
                # the policy's tier before any rollout cap: in shadow / assist mode every tier reads confirm
                h.low += ((e.get("proposed_tier") or tier) if isinstance(e, dict) else tier) != "act"
            cls = _label(e.get("class"))
            if cls is None:
                h.no_class += 1
            else:
                h.classes[cls] += 1
            ent = _entropy(e)
            if ent is not None:
                h.entropy.append(ent)
            for src in e.get("uncertainty") or ():
                if isinstance(src, str):
                    h.sources[src] += 1
            h.capped += _was_capped(e, abstain)
            label = fb.labels.get((r.id, key)) if r.id else None
            if label is not None:
                stakes = e.get("stakes") or row_stakes
                h.labelled.append(_Labelled(
                    model=r.row.get("model"), stakes=str(stakes) if stakes else UNSET,
                    tier=_strength(tier, abstain), policy=_policy_strength(e, abstain), p=_num(e.get("p")),
                    correct=None if cls is None else label == cls))
    return acc


def _head_view(h: _Head) -> dict:
    tiered = h.decisions - h.untiered
    return {"decisions": h.decisions, "tiers": _dist(h.tiers), "untiered": h.untiered, "classes": _dist(h.classes),
            "no_class": h.no_class,
            "entropy": {"mean": M._f(float(np.mean(h.entropy))) if h.entropy else None, "n": len(h.entropy)},
            "low_confidence": _share(h.low, tiered), "uncertainty": _dist(h.sources), "rollout_capped": h.capped,
            "feedback": _labelled_stats(h.labelled)}


# ----------------------------------------------------------------------------- report sections

def _mix(recs: list[_Rec]) -> dict:
    acts = [r for r in recs if r.action is not None]
    s = Counter(r.strength for r in acts)
    n = len(acts)
    return {"decisions": len(recs), "kinds": _dist(Counter(str(r.row.get("kind")) for r in recs)),
            "actions": _dist(Counter(r.action for r in acts)),
            "automatic": _share(s[2], n), "review": _share(s[1], n), "abstain": _share(s[0], n)}


def _actions(recs: list[_Rec], abstain: Sequence[str]) -> dict:
    acts = [r for r in recs if r.action is not None]
    n = len(acts)
    s = Counter(r.strength for r in acts)
    by_kind: dict[str, list[_Rec]] = defaultdict(list)
    for r in acts:
        by_kind[str(r.row.get("kind"))].append(r)
    out: dict[str, Any] = {"n": n, "derived": sum(r.derived for r in acts),
                           "counts": _dist(Counter(r.action for r in acts)), "automatic": _share(s[2], n),
                           "review": _share(s[1], n), "abstain": _share(s[0], n), "abstain_names": list(abstain),
                           "by_kind": {k: _mix(v) for k, v in sorted(by_kind.items())}}
    other = Counter(r.action for r in acts if r.strength is None)
    if other:
        out["unrecognised"] = _dist(other)
    return out


def _rollout(recs: list[_Rec]) -> dict:
    """receipt.rollout of decide / evaluate records: the mode, the kill switch, and the action proposed before the
    rollout cap next to the one taken. The rollout only ever turns act into confirm, so any other pair (a proposal
    weaker than the action, or a proposal for an abstained action) is counted as `incoherent`."""
    receipts = capped = off = incoherent = 0
    modes: Counter = Counter()
    proposed: Counter = Counter()
    moves: Counter = Counter()
    for r in recs:
        if r.action is None:
            continue
        ro = _receipt(r.row).get("rollout")
        prop = r.action
        if isinstance(ro, Mapping):
            receipts += 1
            if ro.get("mode"):
                modes[str(ro["mode"])] += 1
            auto = ro.get("automation")
            if isinstance(auto, Mapping) and auto.get("enabled") is False:
                off += 1
            capped += bool(ro.get("capped"))
            pa = ro.get("proposed_action")
            if isinstance(pa, str) and pa:
                prop = pa
        proposed[prop] += 1
        if prop != r.action:
            moves[f"{prop} -> {r.action}"] += 1
            incoherent += not (prop == "act" and r.action == "confirm")
    return {"receipts": receipts, "capped": capped, "automation_off": off, "modes": _dist(modes),
            "proposed": _dist(proposed), "transitions": _dist(moves), "incoherent": incoherent}


def _binding(recs: list[_Rec], heads: Mapping[str, _Head], abstain: Sequence[str], head: str | None) -> dict:
    candidate: Counter = Counter()
    binds: Counter = Counter()
    sole: Counter = Counter()
    by: Counter = Counter()
    records = unbound = 0
    for r in recs:
        if r.strength is None:
            continue
        records += 1
        row = r.row
        plan = _plan(row)
        chain = plan.get("chain") if plan is not None and isinstance(plan.get("chain"), list) else None
        gating = [(k, n, e) for k, n, e in _entries(row) if _strength(e.get("tier"), abstain) is not None
                  and (chain is None or k in chain or n in chain)]
        for _, n, _ in gating:
            candidate[n] += 1
        if r.strength == 2:
            unbound += 1
            continue
        decided_by = row.get("decided_by")
        if row.get("kind") == "evaluate" and isinstance(decided_by, str) and decided_by not in ("", "heads"):
            by[f"decided_by:{decided_by}"] += 1
            continue
        at = [(n, e) for _, n, e in gating if _strength(e.get("tier"), abstain) == r.strength]
        failed = row.get("failed") if isinstance(row.get("failed"), list) else []
        binders = list(dict.fromkeys(n for n, e in at if not _was_capped(e, abstain)))
        if not binders:
            binders = list(dict.fromkeys(n for k, n, _ in gating if k in failed or n in failed))
        if binders:
            for n in binders:
                binds[n] += 1
                by[n] += 1
            if len(binders) == 1:
                sole[binders[0]] += 1
        else:
            by["rollout_cap" if at else "pack_rule"] += 1
    names = sorted(set(candidate) | set(heads))
    if head is not None:
        names = [n for n in names if n == head]
    never_bind = [n for n in names if candidate[n] >= MIN_RECORDS and binds[n] == 0]
    never_vary = [n for n in names if n in heads and heads[n].decisions >= MIN_RECORDS
                  and len(heads[n].classes) == 1 and heads[n].no_class == 0]
    return {"records": records, "unbound": unbound, "by": _dist(by),
            "heads": {n: {"candidate": candidate[n], "binds": binds[n], "sole": sole[n],
                          "rate": M._f(binds[n] / candidate[n]) if candidate[n] else None}
                      for n in names if candidate[n]},
            "never_bind": never_bind, "never_vary": never_vary,
            "latency_only_candidates": [n for n in never_vary if n in never_bind or candidate[n] == 0],
            "min_records": MIN_RECORDS}


def _plans(recs: list[_Rec]) -> dict | None:
    plans = [(r, p) for r in recs if (p := _plan(r.row)) is not None]
    if not plans:
        return None

    def dist(key: str) -> dict:
        return _dist(Counter(UNSET if p.get(key) is None else (_label(p.get(key)) or UNSET) for _, p in plans))

    return {"n": len(plans), "next": dist("next"), "call_llm": dist("call_llm"), "response_type": dist("response_type"),
            "route": dist("route")}


def _evaluations(recs: list[_Rec]) -> dict | None:
    evs = [r.row for r in recs if r.row.get("kind") == "evaluate"]
    if not evs:
        return None

    def dist(key: str) -> dict:
        return _dist(Counter(UNSET if row.get(key) is None else str(row.get(key)) for row in evs))

    failed: Counter = Counter()
    for row in evs:
        if isinstance(row.get("failed"), list):
            failed.update(str(f) for f in row["failed"])
    return {"n": len(evs), "verdict": dist("verdict"), "next": dist("next"), "decided_by": dist("decided_by"),
            "failed": _dist(failed)}


def _outcomes(recs: list[_Rec], oc: _Outcomes) -> dict:
    joined = [(r, oc.by_id[r.id]) for r in recs if r.id is not None and r.id in oc.by_id]
    known = [(r, o) for r, o in joined if isinstance(o.get("task_success"), bool)]

    def by(key: Callable[[_Rec, Mapping], Any]) -> dict:
        groups: dict[str, list[int]] = defaultdict(lambda: [0, 0])
        for r, o in known:
            k = key(r, o)
            if k is not None:
                g = groups[str(k)]
                g[0] += int(o["task_success"])
                g[1] += 1
        return {k: _rate(*v) for k, v in sorted(groups.items())}

    acted = [o for r, o in known if r.strength == 2]
    fallback = [bool(o["fallback"]) for _, o in joined if isinstance(o.get("fallback"), bool)]
    retries = {"transport": [0.0, 0], "workflow": [0.0, 0]}      # [sum, outcomes with at least one]
    reported = 0
    tokens: Counter = Counter()
    planned: Counter = Counter()
    for r, o in joined:
        rt = o.get("retries")
        if isinstance(rt, Mapping):
            reported += 1
            for kind, slot in retries.items():
                v = _num(rt.get(kind))
                if v is not None and v > 0:
                    slot[0] += v
                    slot[1] += 1
        tk = o.get("tokens")
        if isinstance(tk, Mapping):
            for k, v in tk.items():
                x = _num(v)
                if x is not None:
                    tokens[str(k)] += x
        plan = _plan(r.row)
        if plan is not None and plan.get("route") is not None and o.get("route_taken") is not None:
            planned[f"{plan.get('route')} -> {o.get('route_taken')}"] += 1
    return {
        "rows": oc.rows, "joined": len(joined), "not_in_window": oc.not_in_window, "unlinked": oc.unlinked,
        "corrected": oc.corrected,
        "task_success": _rate(sum(bool(o["task_success"]) for _, o in known), len(known)),
        "task_success_by_kind": by(lambda r, o: r.row.get("kind")),
        "task_success_by_action": by(lambda r, o: r.action),
        "task_success_by_next": by(lambda r, o: _next(r.row)),
        "task_success_by_route": by(lambda r, o: o.get("route_taken")),
        "task_success_by_provider": by(lambda r, o: o.get("provider")),
        "false_automation": _rate(sum(not o["task_success"] for o in acted), len(acted)),
        "routes_taken": _dist(Counter(str(o["route_taken"]) for _, o in joined if o.get("route_taken") is not None)),
        "planned_vs_taken": _dist(planned),
        "fallback": _rate(sum(fallback), len(fallback)),
        "cost": _spread(_num(o.get("cost")) for _, o in joined),
        "latency_ms": _spread(_num(o.get("latency_ms")) for _, o in joined),
        "retries": {"reported": reported,
                    **{k: {"sum": _nice(v[0]), "outcomes": int(v[1])} for k, v in retries.items()}},
        "tokens": {k: _nice(v) for k, v in sorted(tokens.items())},
    }


def _latency(recs: list[_Rec]) -> dict:
    groups: dict[str, list[float]] = defaultdict(list)
    for r in recs:
        v = _num(_receipt(r.row).get("latency_ms"))
        if v is not None:
            groups[str(r.row.get("kind"))].append(v)
    return {k: _spread(v) for k, v in sorted(groups.items())}


def _batteries(recs: list[_Rec]) -> dict:
    c: Counter = Counter()
    for r in recs:
        q = _receipt(r.row).get("questions")
        b = q.get("battery") if isinstance(q, Mapping) else None
        if isinstance(b, Mapping) and b.get("name"):
            c[f"{b.get('name')}@{b.get('version')}"] += 1
    return _dist(c)


def _review_load(recs: list[_Rec], reviewed: set[str], since: datetime | None, until: datetime | None,
                 capacity: float | None) -> dict:
    confirm = [r for r in recs if r.strength == 1]
    abstained = [r for r in recs if r.strength == 0]
    dated = [r.at for r in recs if r.at is not None]
    start = since or (min(dated) if dated else None)
    end = until or (max(dated) if dated else None)
    days = max(1.0, (end - start).total_seconds() / 86400) if start is not None and end is not None else None
    per_day: dict[str, dict[str, int]] = defaultdict(lambda: {"confirm": 0, "abstain": 0})
    for name, rs in (("confirm", confirm), ("abstain", abstained)):
        for r in rs:
            if r.at is not None:
                per_day[r.at.date().isoformat()][name] += 1
    busy = [(d, v) for d, v in per_day.items() if v["confirm"]]
    peak = max(busy, key=lambda kv: (kv[1]["confirm"], kv[0])) if busy else None
    cpd = len(confirm) / days if days else None
    out: dict[str, Any] = {
        "confirm": len(confirm), "abstain": len(abstained), "days": M._f(days), "confirm_per_day": M._f(cpd),
        "abstain_per_day": M._f(len(abstained) / days) if days else None,
        "per_day": {k: per_day[k] for k in sorted(per_day)},
        "peak_day": {"date": peak[0], **peak[1]} if peak else None,
        "pending": {"confirm": sum(1 for r in confirm if r.id not in reviewed),
                    "abstain": sum(1 for r in abstained if r.id not in reviewed)},
        "capacity_per_day": capacity, "review_over_capacity": None, "utilisation": None, "days_over_capacity": None}
    if capacity is not None and cpd is not None:
        out.update(review_over_capacity=cpd > capacity, utilisation=M._f(cpd / capacity),
                   days_over_capacity=sum(1 for d in per_day.values() if d["confirm"] > capacity))
    return out


def _capacity(v: Any) -> float | None:
    if v is None:
        return None
    x = _num(v)
    if x is None or x <= 0:
        raise ContractError(f"review_capacity_per_day must be a positive number, got {v!r}",
                            field="review_capacity_per_day")
    return x


def _notes(recs: list[_Rec], fb: _Feedback, oc: _Outcomes, actions: dict, rollout: dict, heads: dict,
           binding: dict, latency: dict, model: str | None) -> list[str]:
    n = len(recs)
    with_fb = sum(1 for r in recs if r.id in fb.decisions)
    with_oc = sum(1 for r in recs if r.id in oc.by_id)
    notes = [f"feedback and outcomes describe only the decisions they were posted for ({with_fb} of {n} decisions "
             f"have feedback, {with_oc} an outcome): they stand for all decisions only when they are posted for a "
             f"random sample. Confirm and abstain decisions usually reach a person and get feedback far more often "
             f"than automatic ones, so accuracy, false automation, calibration and drift here describe the decisions "
             f"that got feedback"]
    legacy = [r for r in recs if not _receipt(r.row)]
    if legacy:
        derived = sum(r.derived for r in legacy)
        notes.append(f"decision records written before receipts: {len(legacy)}. They have no latency, rollout, "
                     f"warnings or audit fields" + (f"; {derived} of them logged no action, so it is derived from "
                                                     f"their head tiers (the weakest)" if derived else ""))
    models = {r.row.get("model") for r in recs}
    if model is None and len(models) > 1:
        notes.append(f"decisions of {len(models)} model versions are pooled; filter by model to read one version "
                     f"(holdout comparisons use only the decisions of the holdout's own version)")
    thin = [h for h, v in heads.items() if 0 < v["feedback"]["calibration"]["n"] < MIN_OUTCOMES]
    if thin:
        notes.append(f"online calibration from fewer than {MIN_OUTCOMES} labelled decisions is noisy: "
                     f"{', '.join(thin)}")
    if fb.unmatched or fb.ambiguous:
        notes.append(f"feedback rows left out because they cannot be joined to one head of their decision: "
                     f"{fb.unmatched + fb.ambiguous} ({fb.ambiguous} name a head that a /v1/systemone receipt asked in "
                     f"several questions, {fb.unmatched} a head the decision does not have, or carry no label)")
    if actions.get("unrecognised"):
        notes.append(f"actions {sorted(actions['unrecognised'])} are neither act, confirm nor one of the abstain "
                     f"names {actions['abstain_names']}: pass the configured abstain name in abstain_names")
    if rollout["incoherent"]:
        notes.append(f"receipts whose proposed_action the rollout cap cannot explain (it only turns act into "
                     f"confirm): {rollout['incoherent']}. Such a proposal is not what the policy alone would have "
                     f"done: read rollout.proposed and rollout.transitions with care")
    if any(0 < v["n"] < 100 for v in latency.values()):
        notes.append("p99 of fewer than 100 values is close to the maximum and says little")
    if binding["never_bind"] or binding["latency_only_candidates"]:
        notes.append("a head that never binds lowered no action in this window; it may still change what the plan "
                     "does (its class sets the next step, the route or the evidence) or guard a rare case. A head "
                     "listed in latency_only_candidates also always gave the same answer: review it before removing")
    return notes


# ----------------------------------------------------------------------------- public API

def metrics(decisions: Iterable[Any], feedback: Iterable[Any], outcomes: Iterable[Any], *,
            since: str | datetime | None = None, until: str | datetime | None = None, model: str | None = None,
            kind: str | None = None, head: str | None = None, holdout: Mapping[str, Mapping] | None = None,
            review_capacity_per_day: float | None = None,
            abstain_names: Sequence[str] = ABSTAIN_NAMES) -> dict:
    """The operations report over decision records, feedback rows and outcome rows (see the module docstring).

    since / until: ISO 8601 dates or times (since inclusive, until exclusive) on the decisions' created_at; with
    either, records without a readable created_at are left out (`window.undated_excluded`). model / kind / head
    keep the decisions of one model version, one kind, or those a head took part in (per-head numbers then cover
    that head only). holdout: {head: holdout_summary(...)}, put next to the online numbers of its own model version
    (`heads.<head>.vs_holdout`) and tested for drift. review_capacity_per_day: the reviewers' capacity (rlcd.yaml
    serve.review_capacity_per_day).
    abstain_names: the names the abstain action goes by. Raises ContractError on an unreadable since / until, until
    not after since, or a capacity that is not a positive number."""
    abstain = _names(abstain_names)
    t0, t1 = _bound(since, "since"), _bound(until, "until")
    if t0 is not None and t1 is not None and t1 <= t0:
        raise ContractError(f"until ({_iso(t1)}) must be later than since ({_iso(t0)})", field="until")
    capacity = _capacity(review_capacity_per_day)
    recs, undated, unreadable = _select(decisions, t0, t1, model, kind, head, abstain)
    by_id = {r.id: r.row for r in recs if r.id is not None}
    fb = _join_feedback(feedback, by_id, head)
    oc = _join_outcomes(outcomes, by_id)
    acc = _heads(recs, fb, abstain, head)
    heads = {name: _head_view(acc[name]) for name in sorted(acc)}
    for name, h in (holdout or {}).items():
        if name in heads and name in acc and isinstance(h, Mapping):
            heads[name]["vs_holdout"] = _versus(acc[name].labelled, h)
            heads[name]["drift_suspected"] = heads[name]["vs_holdout"]["drift_suspected"]
    actions = _actions(recs, abstain)
    rollout = _rollout(recs)
    binding = _binding(recs, acc, abstain, head)
    latency = _latency(recs)
    dated = [r.at for r in recs if r.at is not None]
    n = len(recs)
    out: dict[str, Any] = {
        "window": {"since": _iso(t0), "until": _iso(t1), "decisions": n,
                   "kinds": _dist(Counter(str(r.row.get("kind")) for r in recs)),
                   "first": _iso(min(dated)) if dated else None, "last": _iso(max(dated)) if dated else None,
                   "undated_excluded": undated, "unreadable": unreadable,
                   "filters": {"model": model, "kind": kind, "head": head}},
        "coverage": {"feedback": _share(sum(1 for r in recs if r.id in fb.decisions), n),
                     "outcome": _share(sum(1 for r in recs if r.id in oc.by_id), n)},
        "actions": actions,
        "rollout": rollout,
        "review_load": _review_load(recs, fb.decisions, t0, t1, capacity),
        "heads": heads,
        "drift_suspected": [h for h, v in heads.items() if v.get("drift_suspected")],
        "binding": binding,
        "plans": _plans(recs),
        "evaluate": _evaluations(recs),
        "outcomes": _outcomes(recs, oc),
        "latency_ms": latency,
        "by_model": {m: _mix(rs) for m, rs in sorted(_group(recs, lambda r: r.row.get("model")).items())},
        "by_stakes": {s: _mix(rs) for s, rs in sorted(
            _group([r for r in recs if r.action is not None], lambda r: r.row.get("stakes")).items())},
        "batteries": _batteries(recs),
        "feedback": {"rows": fb.rows, "joined": len(fb.labels), "corrected": fb.corrected, "unlinked": fb.unlinked,
                     "not_in_window": fb.not_in_window, "unmatched": fb.unmatched, "ambiguous": fb.ambiguous},
        "abstain_names": list(abstain),
    }
    out["notes"] = _notes(recs, fb, oc, actions, rollout, heads, binding, latency, model)
    out["definitions"] = dict(DEFINITIONS)
    return out


def _group(recs: list[_Rec], key: Callable[[_Rec], Any]) -> dict[str, list[_Rec]]:
    out: dict[str, list[_Rec]] = defaultdict(list)
    for r in recs:
        k = key(r)
        out[UNSET if k is None else str(k)].append(r)
    return out


def holdout_summary(report: Mapping | None, *, decision: str = "argmax", model: str | None = None) -> dict:
    """The numbers of a head's holdout report (bundle report.json: core/metrics.calibration_report plus
    `policy.holdout`) that the online numbers of metrics() can be compared with, for heads that commit by `decision`
    ("argmax" or "gate", HeadSpec.decision) in model version `model`.

    accuracy: argmax heads, the top-label holdout accuracy; gate heads, the holdout rate of `true` (a gate head
    commits to true, so "the committed class came true" is that rate). ece: top-label for argmax heads, the binary
    ECE of P(true) for gate heads — each what metrics() computes online. brier: the binary Brier of P(true) for gate
    heads; None for argmax heads, whose holdout Brier is the multiclass score (not comparable with the online binary
    Brier of the committed class). act: {stakes: {n, correct, accuracy, wilson_95, mean_predicted}} of the holdout's
    act-tier rows, the reference of the drift test."""
    rep = report or {}
    gate = decision == "gate"
    out: dict[str, Any] = {"model": model, "decision": decision, "n": rep.get("n_holdout")}
    if gate:
        pos = (rep.get("per_class") or {}).get("true") or {}
        out.update(accuracy=(rep.get("class_rates_holdout") or {}).get("true"), ece=pos.get("ece"),
                   brier=pos.get("brier"))
    else:
        out.update(accuracy=rep.get("accuracy"), ece=rep.get("ece"), brier=None)
    act: dict[str, dict] = {}
    for stakes, ev in ((rep.get("policy") or {}).get("holdout") or {}).items():
        a = ev.get("act") if isinstance(ev, Mapping) else None
        if isinstance(a, Mapping):
            act[str(stakes)] = {k: a.get(k) for k in ("n", "correct", "accuracy", "wilson_95", "mean_predicted")}
    out["act"] = act
    return out


def _queue_reason(strength: int | None, include_abstain: bool) -> str | None:
    if strength == 1:
        return "confirm"
    if strength == 0 and include_abstain:
        return "abstain"
    return None


def _why(row: Mapping, action: str, reason: str, derived: bool, abstain: Sequence[str]) -> list[str]:
    out = [("action confirm: the policy asks a person to confirm before acting" if reason == "confirm" else
            f"action {action}: the decision abstained, the model cannot vouch for any action")]
    capped = [key for key, _, e in _entries(row) if _was_capped(e, abstain)]
    if capped and reason == "confirm":
        ro = _receipt(row).get("rollout")
        ro = ro if isinstance(ro, Mapping) else {}
        auto = ro.get("automation")
        off = isinstance(auto, Mapping) and auto.get("enabled") is False
        how = ", ".join(x for x in (f"mode {ro.get('mode')}" if ro.get("mode") else "",
                                    "automation switched off" if off else "") if x)
        out.append(f"the rollout{f' ({how})' if how else ''} capped {', '.join(capped)} from act at confirm")
    plan = _plan(row)
    if plan is not None and plan.get("reason"):
        out.append(f"plan: {plan.get('reason')}")
    if row.get("decided_by") not in (None, "", "heads"):
        out.append(f"decided by {row.get('decided_by')}")
    if isinstance(row.get("failed"), list) and row.get("failed"):
        out.append(f"failed: {', '.join(str(f) for f in row['failed'])}")
    strength = _strength(action, abstain)
    for key, _, e in _entries(row):
        if _strength(e.get("tier"), abstain) == strength and not _was_capped(e, abstain):
            p = _num(e.get("p"))
            sources = [s for s in e.get("uncertainty") or () if isinstance(s, str)]
            detail = ", ".join(([f"p {p:.2f}"] if p is not None else []) + sources)
            out.append(f"{key} at {e.get('tier')}" + (f" ({detail})" if detail else ""))
    if derived:
        out.append("action derived from the head tiers (a record written before actions were logged)")
    return out


def _queue_item(row: Mapping, action: str, reason: str, derived: bool, abstain: Sequence[str]) -> dict:
    item: dict[str, Any] = {"decision_id": row.get("decision_id"), "created_at": row.get("created_at"),
                            "kind": row.get("kind"), "model": row.get("model"), "action": action, "queue": reason}
    if derived:
        item["action_derived"] = True
    for k in ("stakes", "plan", "next", "verdict", "decided_by", "failed", "evidence"):
        if row.get(k) is not None:
            item[k] = row.get(k)
    heads: dict[str, dict] = {}
    for key, name, e in _entries(row):
        h: dict[str, Any] = {"class": e.get("class"), "p": e.get("p"), "tier": e.get("tier")}
        if name != key:
            h["head"] = name
        for k in ("proposed_tier", "stakes", "probabilities", "state_digest"):
            if e.get(k) is not None:
                h[k] = e.get(k)
        h["uncertainty"] = [s for s in e.get("uncertainty") or () if isinstance(s, str)]
        if "state" in e:
            h["state"] = e.get("state")
        heads[key] = h
    states = [h["state"] for h in heads.values() if "state" in h]
    if len(heads) > 1 and len(states) == len(heads) and all(s == states[0] for s in states[1:]):
        item["state"] = states[0]          # one shared state, shown once
        for h in heads.values():
            h.pop("state")
    item["heads"] = heads
    rec = _receipt(row)
    item["warnings"] = [str(w) for w in rec.get("warnings") or ()]
    ro = rec.get("rollout")
    if isinstance(ro, Mapping) and ro.get("capped"):
        item["rollout"] = {"mode": ro.get("mode"), "capped": True, "proposed_action": ro.get("proposed_action")}
    if rec.get("audit"):
        item["audit"] = rec.get("audit")
    if rec:
        item["receipt"] = {"digest": rec.get("digest"), "state_digest": rec.get("state_digest")}
    item["why"] = _why(row, action, reason, derived, abstain)
    return item


def review_queue(decisions: Iterable[Any], feedback: Iterable[Any], *, limit: int | None = 50,
                 head: str | None = None, kind: str | None = None, include_abstain: bool = True,
                 oldest_first: bool = True, abstain_names: Sequence[str] = ABSTAIN_NAMES) -> list[dict]:
    """Decisions waiting for a person: their action (or, for a record without one, its weakest head tier) is confirm
    or, with include_abstain, the abstain action, and no feedback names them yet (with `head`: no feedback for that
    head). /v1/systemone and /v1/rank records have no tier and never queue.

    Each item: decision_id, created_at, kind, model, action, `queue` (confirm | abstain), stakes / plan / next /
    verdict / decided_by / failed / evidence when the record has them, heads {class, p, tier, proposed_tier?, stakes?,
    probabilities?, uncertainty, state?} (a state every head shares is shown once as `state`), the receipt's
    warnings, rollout (when it capped the decision), audit (baseline, artifact, checks, revision, tags) and digests,
    and `why`: the reasons it is in the queue (the action, the rollout cap, the plan's reason, the heads at that tier
    with their probability and uncertainty sources). A reviewer resolves an item with POST /v1/feedback {decision_id,
    head, label}: correcting the route, naming `missing_evidence`, or tagging it `regression`.

    Items are ordered by created_at (oldest first by default; undated records last) and cut at `limit` (None: all).
    """
    abstain = _names(abstain_names)
    reviewed = {r.get("decision_id") for r in feedback or ()
                if isinstance(r, Mapping) and isinstance(r.get("decision_id"), str)
                and (head is None or r.get("head") == head)}
    picked: list[tuple[datetime | None, int, Mapping, str, bool, str]] = []
    for i, row in enumerate(decisions or ()):
        if not isinstance(row, Mapping) or not isinstance(row.get("decision_id"), str):
            continue
        if (kind is not None and row.get("kind") != kind) or (head is not None and not _involves(row, head)):
            continue
        if row["decision_id"] in reviewed:
            continue
        action, derived = _row_action(row, abstain)
        reason = _queue_reason(_strength(action, abstain), include_abstain)
        if action is None or reason is None:
            continue
        picked.append((_when(row.get("created_at")), i, row, action, derived, reason))
    dated = sorted((p for p in picked if p[0] is not None), key=lambda p: (p[0], p[1]), reverse=not oldest_first)
    ordered = dated + [p for p in picked if p[0] is None]
    if limit is not None:
        ordered = ordered[:max(0, int(limit))]
    return [_queue_item(row, action, reason, derived, abstain) for _, _, row, action, derived, reason in ordered]
