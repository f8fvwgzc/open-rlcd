"""Replay: re-run logged decisions under two model versions and compare what each would decide.

The guide (II.G, X.C offline stage, XI.C): before promoting a version, replay labelled and logged cases and compare
both final routes and probability bands. `replay(service, base, candidate, rows)` reads decision records (lines of
decisions.jsonl), recomputes every logged head state under both versions — the probabilities with
`service.probabilities`, the decision with `policy.decide` — and reports per head:

    classes               how often each version names each class (the distribution of answers), how often the
                          most probable class changed, and the most common transitions (with example decision ids).
                          For a gate head the decision always commits to `true`; its class here is still the most
                          probable one (which side of 0.5), and the gate itself is the tier
    tiers                 the transition matrix base tier -> candidate tier, and how often the candidate would
                          automate more (raised) or less (lowered)
    bands                 the probability band (`bernoulli.bin_of`, tenths) of the probability each version's policy
                          thresholds — its committed class, P(true) for a gate head — before and after
    mean_abs_delta_p      mean |p_candidate - p_base| of that probability; mean_tv_distance: the mean total-variation
                          distance between the two distributions (0 = identical, 1 = disjoint)
    labelled              joined with feedback labels (the last row per decision and head counts, as at training):
                          each version's accuracy (most probable class = label) and Brier score (sum_k (p_k - y_k)^2),
                          how often it would have acted wrongly, the decisions where the base was right and the
                          candidate wrong (regressions) and the reverse, and an exact paired test on those two counts

Each version is judged as trained, so neither is favoured:
  * the Bernoulli gate on act uses each version's own holdout table, not the live ledger: the live table belongs to
    whichever version is latest and moves with the outcomes posted since, so a version is neither penalised nor
    helped by whose turn it is to serve;
  * the operator's rollout mode and kill switch are not applied (they cap both versions alike), and exploration
    (Thompson sampling) is off;
  * the input guard applies as when serving: a state missing inputs almost every training row had is capped at
    confirm; the stakes are the logged ones (else the head's default in that version).

What it does not do:
  * it compares heads, not whole requests: pack logic above the heads (the documents plan, evidence caps), options
    removed by `allowed` or route contracts (not logged per head) and the rollout are not re-run. Request-level
    regression cases (core/regress.py) run whole requests;
  * it writes nothing: no decision is logged, no ledger moves;
  * a state is replayed as logged: a secret redacted at logging time is replayed as its [REDACTED:...] placeholder.
    Decisions logged without states (serve.log_states: false), pruned states, rank receipts (no head states), heads
    a version lacks or never trained, states a version's provider refuses and legacy or damaged rows are counted
    under `skipped` with the reason, never fatal;
  * labelled rows are few, and they are a random sample of traffic only if labels were posted for one. Feedback
    becomes training data at the next train, so a version trained after a label was posted may have trained on
    that row (`possibly_trained_on` counts such rows): its accuracy there is optimistic. The paired test only says
    whether the disagreements on these rows lean one way, not that a version is better on traffic.
"""
from __future__ import annotations

from collections import Counter, defaultdict, deque
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import numpy as np
from scipy import stats

from . import bernoulli as B
from . import policy as PL
from .contract import ContractError

CHUNK = 256                 # decisions per batch: each head's states go through its provider together
TOP_TRANSITIONS = 10        # class and band transitions listed per head
TRANSITION_EXAMPLES = 3     # decision ids kept per class transition
SKIP_EXAMPLES = 5           # examples kept per skip reason

SKIPS = {
    "malformed": "a row or head entry that is not a decision record object (a damaged or foreign line)",
    "undated": "a decision without a readable created_at while `since` was given",
    "no_heads": "a decision that logged no head entries (rank receipts log none)",
    "no_state": "the head's state was not logged (serve.log_states: false, or a row from before states were logged)",
    "state_pruned": "the state was dropped by `rlcd receipts prune`: only its digest is kept",
    "head_missing": "the version has no head of that name",
    "untrained": "the head is not trained in the version: its answer would be the uninformed prior",
    "stakes_unknown": "the logged stakes level is not defined by the version's policy",
    "state_refused": "the version's feature provider refuses the logged state (another shape or feature version)",
    "error": "the version failed on this state for another reason (see the examples)",
}
NOTES = [
    "both versions are judged as trained: each version's own holdout Bernoulli table gates act (not the live "
    "ledger), no rollout cap or kill switch, no exploration; the input guard applies as when serving",
    "heads only: pack logic above the heads (plans, evidence caps), `allowed` options and route contracts are not "
    "re-run (use request-level regression cases for those)",
    "class = the most probable class (also for gate heads, whose decision always commits to `true`); p and bands = "
    "the probability each version's policy thresholds (its committed class; P(true) for a gate head)",
    "labelled rows: feedback with a decision_id, the last row per decision and head; feedback becomes training data "
    "at the next train, so rows counted in `possibly_trained_on` may be in-sample for that version",
]


# ----------------------------------------------------------------------------- small helpers

def when(value: Any) -> datetime | None:
    """An ISO 8601 date or time (a trailing Z allowed) as an aware datetime (UTC when naive); None if unreadable."""
    if isinstance(value, datetime):
        t = value
    elif isinstance(value, str) and value.strip():
        try:
            t = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
        except ValueError:
            return None
    else:
        return None
    return t if t.tzinfo is not None else t.replace(tzinfo=UTC)


def band(k: int) -> str:
    """The name of probability bin k: "0.9-1.0"."""
    return f"{k / B.BINS:.1f}-{(k + 1) / B.BINS:.1f}"


def holdout_table(head: Any) -> dict | None:
    """A trained head's own holdout Bernoulli table (`LoadedHead.holdout_ledger`); None for an old bundle without
    one, which leaves the gate on act closed, as when serving."""
    if head is None:
        return None
    try:
        return head.holdout_ledger()
    except Exception:
        return None


def feedback_labels(rows: Iterable[Any]) -> dict[tuple[str, str], dict]:
    """{(decision_id, head): feedback row}: the last row for a decision and head wins (a later row corrects an
    earlier one, as at training). Rows without a decision_id label a state, not a decision: they are not joined."""
    out: dict[tuple[str, str], dict] = {}
    for r in rows:
        if isinstance(r, dict) and r.get("decision_id") and r.get("head") and r.get("label") is not None:
            out[(str(r["decision_id"]), str(r["head"]))] = r
    return out


def _r(x: float | None) -> float | None:
    return None if x is None else round(float(x), 6)


def _ratio(k: int, n: int) -> float | None:
    return _r(k / n) if n else None


def entry_head(key: str, h: Any) -> str:
    """The head of a logged entry: its key (decide / evaluate), or the `head` a systemone question entry names."""
    if isinstance(h, dict) and isinstance(h.get("head"), str) and h["head"]:
        return h["head"]
    return key


def _label_index(spec: Any, label: Any) -> int | None:
    labels = list(spec.labels)
    if label in labels:
        return labels.index(label)
    try:
        return int(spec.label_index(label))
    except Exception:
        return None


def _chunks(items: Iterable[Any], n: int) -> Iterator[list]:
    buf: list = []
    for it in items:
        buf.append(it)
        if len(buf) >= n:
            yield buf
            buf = []
    if buf:
        yield buf


# ----------------------------------------------------------------------------- one head under one version

@dataclass(frozen=True)
class _Skip:
    reason: str
    detail: str | None = None


@dataclass
class _Entry:
    decision_id: str
    head: str
    state: Any
    stakes: str | None
    label: dict | None


@dataclass
class _Ctx:
    """What one version needs to decide one head: spec, policy, verdict, conformal threshold, holdout table."""
    version: str
    spec: Any = None
    cfg: Any = None
    names: tuple[str, ...] = ()
    verdict: dict | None = None
    conformal: float | None = None
    ledger: dict | None = None
    entry: dict = field(default_factory=dict)
    problem: _Skip | None = None


@dataclass
class _Out:
    labels: tuple[str, ...]
    P: np.ndarray           # in labels order
    cls: str                # the class the decision commits to
    p: float                # its probability (what the policy thresholds)
    tier: str
    rank: int               # 0 abstain, 1 confirm, 2 act

    @property
    def top(self) -> int:
        return int(np.argmax(self.P))


def _context(service: Any, version: str, head: str) -> _Ctx:
    c = _Ctx(version)
    try:
        specs = service.specs(version)
        if head not in specs:
            c.problem = _Skip("head_missing", f"{version} has no head {head!r}")
            return c
        c.spec = specs[head]
        loaded = service.loaded(version, head)
        if loaded is None:
            c.problem = _Skip("untrained", f"head {head!r} is not trained in {version}")
            return c
        c.cfg = service.policy(version)
        c.names = tuple(c.cfg.tier_name(i) for i in range(3))
        c.verdict = loaded.verdict
        c.conformal = loaded.conformal_threshold
        c.ledger = holdout_table(loaded)
        c.entry = dict(loaded.entry)
    except ContractError as e:
        c.problem = _Skip("error", f"{version}: {e.message}")
    except Exception as e:   # a version whose bundle no longer loads: every entry is skipped, the rest goes on
        c.problem = _Skip("error", f"{version}: {type(e).__name__}: {e}")
    return c


def _probabilities(service: Any, c: _Ctx, states: list[Any]) -> list[tuple[np.ndarray, list[str], bool] | _Skip]:
    """Each state's (probabilities, missing inputs, calibrated) or _Skip. One batch per head and version; when a
    state is refused or some state lacks inputs, the states are taken one at a time so that the refusal and the
    input guard apply to the state they belong to."""
    if not states:
        return []
    try:
        missing: list[str] = []
        P, calibrated = service.probabilities(c.version, c.spec, list(states), missing_out=missing)
        if not missing:
            return [(np.asarray(P[i], float), [], bool(calibrated)) for i in range(len(states))]
    except Exception:
        pass   # one state the provider refuses fails the whole batch: find it below
    out: list[tuple[np.ndarray, list[str], bool] | _Skip] = []
    for s in states:
        try:
            m: list[str] = []
            P, calibrated = service.probabilities(c.version, c.spec, [s], missing_out=m)
            out.append((np.asarray(P[0], float), m, bool(calibrated)))
        except ContractError as e:
            out.append(_Skip("state_refused", e.message))
        except Exception as e:
            out.append(_Skip("error", f"{type(e).__name__}: {e}"))
    return out


def _decide(service: Any, c: _Ctx, entries: list[_Entry]) -> list[_Out | _Skip]:
    """Every entry's decision under one version (as trained: the version's own holdout table gates act)."""
    if c.problem is not None:
        return [c.problem] * len(entries)
    out: list[_Out | _Skip] = [_Skip("error", "not decided")] * len(entries)
    todo: list[int] = []
    for i, e in enumerate(entries):
        stakes = e.stakes or c.spec.stakes
        try:
            c.cfg.level(stakes)
        except Exception:
            out[i] = _Skip("stakes_unknown", f"{stakes!r} is not one of {list(c.cfg.stakes)} in {c.version}")
            continue
        todo.append(i)
    probs = _probabilities(service, c, [entries[i].state for i in todo])
    for i, pr in zip(todo, probs, strict=True):
        if isinstance(pr, _Skip):
            out[i] = pr
            continue
        P, missing, calibrated = pr
        try:
            d = PL.decide(c.spec, P, c.cfg, stakes=entries[i].stakes or c.spec.stakes, verdict=c.verdict,
                          conformal_threshold=c.conformal, ledger=c.ledger, calibrated=calibrated,
                          missing_inputs=missing or None)
            apply_verified = getattr(service, "_apply_verified", None)
            if callable(apply_verified):   # as served: no act below the act threshold the head was verified with
                apply_verified(d, c.version, c.spec, service.loaded(c.version, c.spec.name))
            labels = tuple(c.spec.labels)
            ci = labels.index(d["class"])
            out[i] = _Out(labels, P, d["class"], float(P[ci]), d["tier"], c.names.index(d["tier"]))
        except ContractError as e:
            out[i] = _Skip("error", e.message)
        except Exception as e:
            out[i] = _Skip("error", f"{type(e).__name__}: {e}")
    return out


# ----------------------------------------------------------------------------- accumulating one head

class _Scores:
    """One version's record on the labelled rows of one head."""

    def __init__(self) -> None:
        self.n = self.correct = self.act = self.act_wrong = self.trained_on = 0
        self.brier = 0.0

    def add(self, o: _Out, y: int, head_entry: dict, label_row: dict) -> bool:
        right = o.top == y
        self.n += 1
        self.correct += int(right)
        onehot = np.zeros(len(o.P))
        onehot[y] = 1.0
        self.brier += float(((o.P - onehot) ** 2).sum())
        if o.rank == 2:
            self.act += 1
            self.act_wrong += int(o.cls != o.labels[y])
        trained, received = when(head_entry.get("trained_at")), when(label_row.get("received_at"))
        if trained is not None and received is not None and received <= trained:
            self.trained_on += 1
        return right

    def view(self) -> dict:
        return {"n": self.n, "correct": self.correct, "accuracy": _ratio(self.correct, self.n),
                "brier": _r(self.brier / self.n) if self.n else None,
                "act": {"n": self.act, "wrong": self.act_wrong}, "possibly_trained_on": self.trained_on}


class _HeadStats:
    def __init__(self, max_examples: int) -> None:
        self.max_examples = max_examples
        self.n = 0
        self.names_b: tuple[str, ...] = ()
        self.names_c: tuple[str, ...] = ()
        self.same_model: bool | None = None
        self.class_changes = 0
        self.classes_b: Counter = Counter()
        self.classes_c: Counter = Counter()
        self.transitions: Counter = Counter()
        self.transition_ids: dict[tuple[str, str], list[str]] = defaultdict(list)
        self.tiers_b: Counter = Counter()
        self.tiers_c: Counter = Counter()
        self.matrix: dict[str, Counter] = defaultdict(Counter)
        self.raised = self.lowered = 0
        self.bands_b: Counter = Counter()
        self.bands_c: Counter = Counter()
        self.band_moves: Counter = Counter()
        self.band_up = self.band_down = 0
        self.sum_dp = self.max_dp = self.sum_tv = 0.0
        self.labelled = self.unmapped = 0
        self.scores = {"base": _Scores(), "candidate": _Scores()}
        self.regressions: list[dict] = []
        self.improvements: list[dict] = []
        self.n_regressions = self.n_improvements = 0

    def add(self, e: _Entry, rb: _Out, rc: _Out, cb: _Ctx, cc: _Ctx) -> None:
        if not self.n:
            self.names_b, self.names_c = cb.names, cc.names
            fb, fc = cb.entry.get("files"), cc.entry.get("files")
            self.same_model = (fb == fc) if fb and fc else None
        self.n += 1
        top_b, top_c = rb.labels[rb.top], rc.labels[rc.top]
        self.classes_b[top_b] += 1
        self.classes_c[top_c] += 1
        if top_b != top_c:
            self.class_changes += 1
            t = (top_b, top_c)
            self.transitions[t] += 1
            if len(self.transition_ids[t]) < TRANSITION_EXAMPLES:
                self.transition_ids[t].append(e.decision_id)
        self.tiers_b[rb.tier] += 1
        self.tiers_c[rc.tier] += 1
        self.matrix[rb.tier][rc.tier] += 1
        self.raised += int(rc.rank > rb.rank)
        self.lowered += int(rc.rank < rb.rank)
        kb, kc = B.bin_of(rb.p), B.bin_of(rc.p)
        self.bands_b[kb] += 1
        self.bands_c[kc] += 1
        if kb != kc:
            self.band_moves[(kb, kc)] += 1
            self.band_up += int(kc > kb)
            self.band_down += int(kc < kb)
        dp = abs(rc.p - rb.p)
        self.sum_dp += dp
        self.max_dp = max(self.max_dp, dp)
        pb, pc = dict(zip(rb.labels, rb.P, strict=True)), dict(zip(rc.labels, rc.P, strict=True))
        self.sum_tv += 0.5 * sum(abs(float(pb.get(k, 0.0)) - float(pc.get(k, 0.0))) for k in sorted(set(pb) | set(pc)))
        if e.label is None:
            return
        yb, yc = _label_index(cb.spec, e.label.get("label")), _label_index(cc.spec, e.label.get("label"))
        if yb is None or yc is None:
            self.unmapped += 1
            return
        self.labelled += 1
        right_b = self.scores["base"].add(rb, yb, cb.entry, e.label)
        right_c = self.scores["candidate"].add(rc, yc, cc.entry, e.label)
        if right_b == right_c:
            return
        ex = {"decision_id": e.decision_id, "label": rb.labels[yb], "base": top_b, "candidate": top_c}
        if right_b:
            self.n_regressions += 1
            if len(self.regressions) < self.max_examples:
                self.regressions.append(ex)
        else:
            self.n_improvements += 1
            if len(self.improvements) < self.max_examples:
                self.improvements.append(ex)

    def view(self) -> dict:
        n = self.n
        matrix = {b: {c: int(self.matrix[b][c]) for c in self.names_c} for b in self.names_b}
        out: dict[str, Any] = {
            "n": n, "same_model": self.same_model,
            "classes": {"base": dict(sorted(self.classes_b.items())),
                        "candidate": dict(sorted(self.classes_c.items()))},
            "class_changes": self.class_changes, "class_change_rate": _ratio(self.class_changes, n),
            "class_transitions": [{"from": a, "to": b, "n": k, "examples": self.transition_ids[(a, b)]}
                                  for (a, b), k in self.transitions.most_common(TOP_TRANSITIONS)],
            "tiers": {"base": {t: int(self.tiers_b[t]) for t in self.names_b},
                      "candidate": {t: int(self.tiers_c[t]) for t in self.names_c},
                      "matrix": matrix, "raised": self.raised, "lowered": self.lowered},
            "bands": {"base": {band(k): int(v) for k, v in sorted(self.bands_b.items())},
                      "candidate": {band(k): int(v) for k, v in sorted(self.bands_c.items())},
                      "changed": int(sum(self.band_moves.values())), "up": self.band_up, "down": self.band_down,
                      "transitions": [{"from": band(a), "to": band(b), "n": k}
                                      for (a, b), k in self.band_moves.most_common(TOP_TRANSITIONS)]},
            "mean_abs_delta_p": _ratio_f(self.sum_dp, n), "max_abs_delta_p": _r(self.max_dp) if n else None,
            "mean_tv_distance": _ratio_f(self.sum_tv, n),
        }
        lab: dict[str, Any] = {"n": self.labelled, "base": self.scores["base"].view(),
                               "candidate": self.scores["candidate"].view(),
                               "regressions": {"n": self.n_regressions, "examples": self.regressions},
                               "improvements": {"n": self.n_improvements, "examples": self.improvements},
                               "paired_test": _paired(self.n_regressions, self.n_improvements)}
        if self.unmapped:
            lab["unmapped"] = self.unmapped
        out["labelled"] = lab
        return out


def _ratio_f(total: float, n: int) -> float | None:
    return _r(total / n) if n else None


def _paired(regressions: int, improvements: int) -> dict | None:
    """Exact McNemar test: on the labelled rows exactly one version got right, are the two directions equally
    likely? (two-sided binomial test of the regressions against half of the discordant rows)."""
    m = regressions + improvements
    if m == 0:
        return None
    return {"discordant": m, "p_value": _r(stats.binomtest(regressions, m, 0.5).pvalue),
            "test": "exact McNemar (two-sided binomial on the rows exactly one version got right)"}


# ----------------------------------------------------------------------------- the replay

class _Replay:
    def __init__(self, service: Any, base: str, candidate: str, max_examples: int) -> None:
        self.service, self.base, self.candidate, self.max_examples = service, base, candidate, max_examples
        self.ctx: dict[tuple[str, str], _Ctx] = {}
        self.stats: dict[str, _HeadStats] = {}
        self.skipped: dict[str, dict] = {}
        self.scope = Counter()
        self.kinds: Counter = Counter()
        self.models: Counter = Counter()
        self.decisions = 0

    def skip(self, reason: str, decision_id: str | None, head: str | None, version: str | None = None,
             detail: str | None = None) -> None:
        s = self.skipped.setdefault(reason, {"count": 0, "why": SKIPS.get(reason, reason), "examples": []})
        s["count"] += 1
        if version is not None:
            by = s.setdefault("by_version", {})
            by[version] = by.get(version, 0) + 1
        if len(s["examples"]) < SKIP_EXAMPLES:
            ex = {"decision_id": decision_id, "head": head}
            if version is not None:
                ex["version"] = version
            if detail:
                ex["detail"] = detail[:300]
            s["examples"].append(ex)

    def context(self, version: str, head: str) -> _Ctx:
        key = (version, head)
        if key not in self.ctx:
            self.ctx[key] = _context(self.service, version, head)
        return self.ctx[key]

    def scoped(self, rows: Iterable[Any], wanted: frozenset[str] | None, cutoff: datetime | None,
               limit: int | None) -> Iterable[tuple[dict, list[tuple[str, Any]]]]:
        """The decisions in scope with their head entries in scope; the newest `limit` of them when limited."""
        def gen() -> Iterator[tuple[dict, list[tuple[str, Any]]]]:
            for row in rows:
                self.scope["rows_read"] += 1
                if not isinstance(row, dict):
                    self.skip("malformed", None, None, detail=f"a {type(row).__name__}, not an object")
                    continue
                did = str(row.get("decision_id") or "?")
                if cutoff is not None:
                    t = when(row.get("created_at"))
                    if t is None:
                        self.skip("undated", did, None)
                        continue
                    if t < cutoff:
                        self.scope["before_since"] += 1
                        continue
                hs = row.get("heads")
                if not isinstance(hs, dict) or not hs:
                    if wanted is None:
                        self.skip("no_heads", did, None, detail=f"kind {row.get('kind')!r}")
                    else:
                        self.scope["without_matching_heads"] += 1
                    continue
                items = [(str(k), h) for k, h in hs.items()
                         if wanted is None or str(k) in wanted or entry_head(str(k), h) in wanted]
                if not items:
                    self.scope["without_matching_heads"] += 1
                    continue
                self.scope["in_scope"] += 1
                yield row, items
        if limit is None:
            return gen()
        last = deque(gen(), maxlen=limit)
        self.scope["beyond_limit"] = self.scope["in_scope"] - len(last)
        return last

    def run_chunk(self, chunk: list[tuple[dict, list[tuple[str, Any]]]],
                  labels: dict[tuple[str, str], dict]) -> None:
        groups: dict[str, list[tuple[int, _Entry]]] = defaultdict(list)
        for i, (row, items) in enumerate(chunk):
            did = str(row.get("decision_id") or "?")
            self.kinds[str(row.get("kind") or "?")] += 1
            self.models[str(row.get("model") or "?")] += 1
            # a label names a head, not a question id: join it only when the decision asked that head once
            heads_of = Counter(entry_head(str(k), h) for k, h in (row.get("heads") or {}).items())
            for key, h in items:
                if not isinstance(h, dict):
                    self.skip("malformed", did, key, detail=f"a {type(h).__name__}, not an object")
                    continue
                name = entry_head(key, h)
                if "state" not in h:
                    self.skip("state_pruned" if h.get("state_digest") else "no_state", did, name)
                    continue
                stakes = h.get("stakes") if isinstance(h.get("stakes"), str) and h.get("stakes") else None
                label = labels.get((did, name)) if heads_of[name] == 1 else None
                groups[name].append((i, _Entry(did, name, h["state"], stakes, label)))
        replayed: set[int] = set()
        for name, group in groups.items():
            entries = [e for _, e in group]
            cb, cc = self.context(self.base, name), self.context(self.candidate, name)
            outs_b, outs_c = _decide(self.service, cb, entries), _decide(self.service, cc, entries)
            st = self.stats.setdefault(name, _HeadStats(self.max_examples))
            for (i, e), rb, rc in zip(group, outs_b, outs_c, strict=True):
                if isinstance(rb, _Skip):
                    self.skip(rb.reason, e.decision_id, name, self.base, rb.detail)
                elif isinstance(rc, _Skip):
                    self.skip(rc.reason, e.decision_id, name, self.candidate, rc.detail)
                else:
                    st.add(e, rb, rc, cb, cc)
                    replayed.add(i)
        self.decisions += len(replayed)

    def result(self, *, heads: frozenset[str] | None, since: Any, limit: int | None) -> dict:
        views = {name: st.view() for name, st in sorted(self.stats.items()) if st.n}
        tot: dict[str, int] = Counter()
        for v in views.values():
            tot["n"] += v["n"]
            tot["class_changes"] += v["class_changes"]
            tot["tier_raised"] += v["tiers"]["raised"]
            tot["tier_lowered"] += v["tiers"]["lowered"]
            tot["band_changes"] += v["bands"]["changed"]
            tot["labelled"] += v["labelled"]["n"]
            tot["regressions"] += v["labelled"]["regressions"]["n"]
            tot["improvements"] += v["labelled"]["improvements"]["n"]
        totals = {k: int(tot[k]) for k in ("n", "class_changes", "tier_raised", "tier_lowered", "band_changes",
                                           "labelled", "regressions", "improvements")}
        n_skipped = sum(s["count"] for s in self.skipped.values())
        try:
            policy_changed = self.service.policy_digest(self.base) != self.service.policy_digest(self.candidate)
        except Exception:
            policy_changed = None
        scope = {k: int(self.scope[k]) for k in ("rows_read", "in_scope", "before_since", "without_matching_heads",
                                                 "beyond_limit")}
        scope.update(since=since if since is None or isinstance(since, str) else str(since), limit=limit,
                     heads=sorted(heads) if heads else None)
        return {"base": self.base, "candidate": self.candidate, "policy_changed": policy_changed, "scope": scope,
                "decisions": self.decisions, "replayed": totals["n"], "kinds": dict(self.kinds),
                "models": dict(self.models), "totals": totals, "heads": views, "skipped": self.skipped,
                "n_skipped": n_skipped, "notes": list(NOTES), "summary": _summary(self, totals, views, n_skipped)}


def _summary(rp: _Replay, totals: dict, views: dict, n_skipped: int) -> str:
    head = f"{rp.base} -> {rp.candidate}: "
    if not totals["n"]:
        return head + f"nothing replayed ({n_skipped} head entries or rows skipped: see `skipped`)"
    out = (f"{totals['n']} head decisions from {rp.decisions} decisions replayed: {totals['class_changes']} class "
           f"changes, tiers raised {totals['tier_raised']} / lowered {totals['tier_lowered']}, "
           f"{totals['band_changes']} probability-band changes")
    if totals["labelled"]:
        acc = []
        for name, v in views.items():
            lab = v["labelled"]
            if lab["n"]:
                acc.append(f"{name} {lab['base']['accuracy']} -> {lab['candidate']['accuracy']}")
        out += (f"; {totals['labelled']} labelled: {totals['regressions']} regressions, {totals['improvements']} "
                f"improvements (accuracy {', '.join(acc)})")
    if n_skipped:
        out += f"; {n_skipped} skipped"
    return head + out


def replay(service: Any, base: str, candidate: str | None, rows: Iterable[Any] | None = None, *,
           heads: Sequence[str] | str | None = None, limit: int | None = None, since: Any = None,
           max_examples: int = 20) -> dict:
    """Recompute the logged head states of `rows` (decision records; None: the service's decisions.jsonl) under
    `base` and `candidate` (version ids, or `rlcd-latest`) and compare them per head (see the module docstring).

    `heads`: only these heads (head names, or question ids of systemone receipts); `since`: only decisions created
    at or after this ISO 8601 date or time; `limit`: only the newest `limit` decisions in scope; `max_examples`:
    decision ids listed per head where one version was right on a labelled row and the other wrong. Logs nothing."""
    if base is None:
        raise ContractError("name the base version to compare the candidate with", field="base")
    vb, vc = service.resolve(base), service.resolve(candidate)
    if isinstance(heads, str):
        heads = [h for h in heads.split(",") if h.strip()]
    wanted = frozenset(h.strip() for h in heads) if heads else None
    cutoff = None
    if since is not None:
        cutoff = when(since)
        if cutoff is None:
            raise ContractError(f"since must be an ISO 8601 date or time, got {since!r}", field="since")
    if limit is not None and (isinstance(limit, bool) or not isinstance(limit, int) or limit < 1):
        raise ContractError(f"limit must be a positive integer, got {limit!r}", field="limit")
    if isinstance(max_examples, bool) or not isinstance(max_examples, int) or max_examples < 0:
        raise ContractError(f"max_examples must be a non-negative integer, got {max_examples!r}",
                            field="max_examples")
    rp = _Replay(service, vb, vc, max_examples)
    labels = feedback_labels(service.logs.feedback())
    source = service.logs.decisions() if rows is None else rows
    for chunk in _chunks(rp.scoped(source, wanted, cutoff, limit), CHUNK):
        rp.run_chunk(chunk, labels)
    return rp.result(heads=wanted, since=since, limit=limit)
