"""`rlcd label audit`: a human checks labels in the terminal; their answers become trusted rows.

Picking what to show:
  * strategy "stratified" (default): for each head, the items of its view without a human label, stratified by
    (judge label, judge-confidence band) with **proportional** allocation, ordered so that every prefix of the
    session is close to proportional too (quitting early still leaves a representative sample). Equal allocation
    would over-sample rare judge labels; because calibrators are fitted and the holdout is scored on these rows,
    that would act like resampling and bias the published probabilities. Heads are interleaved, and the selected
    questions of one item are asked one after another so its text is read once.
  * strategy "queue": the active-learning queue (<home>/labels/queue.jsonl, written by `rlcd label next`) in its
    order. Those rows are chosen for being hard, not for being typical (see open_rlcd.labeling.active).

Keys at the prompt: Enter = accept the judge label, a number or a label name = choose, s = skip, b = back
(re-label the previous question), q = quit, ? = help. A noul takes y/n (also true/false, 1/0); a score takes the
level key shown (the level number 0..K-1, or the level's name when levels are named, e.g. "1".."5" for a rating);
a choice takes its option number (1..K) or name (a unique prefix is enough).

Every answer is written at once (LabelStore.add_human, source "human"), so the session is resumable: questions that
already have a human label are not asked again. The note on each label records the sampling strategy and whether
the human accepted, agreed with or overruled the judge.

At the end the judge-vs-human agreement per head (accuracy, Cohen's kappa, confusion) is printed and returned, with
how many trusted rows each head now has against the ~46 a head needs to train at all (at least train.min_trusted, 30
by default, and enough that 10 land in its holdout), the ~150 that give a 30-row holdout (fewer, and its decisions
stay capped at confirm), and the ~300 recommended.
Plain prompts (no curses); ANSI colour only when printing to a terminal.
"""
from __future__ import annotations

import math
import random
import sys
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..core.head import HeadSpec
from ..core.metrics import agreement
from .active import read_queue
from .bootstrap import select_heads
from .prompts import label_index
from .store import LabelItem, LabelStore

MIN_TRUSTED = 30            # train.min_trusted default: fewer trusted rows and a head cannot train
MIN_HOLDOUT = 10            # core/training.py: fewer trusted rows in the holdout and a head cannot train
RECOMMENDED_TRUSTED = 300   # enough for a holdout of ~60 rows and usable reliability bins
HOLDOUT_FRACTION = 0.2      # train.holdout_fraction default
MIN_VERIFY_ROWS = 30        # policy.min_verify_rows default: holdout (and act-tier) rows needed to verify `act`
MAX_DISPLAY = 6000          # characters of item text shown per question
STRATEGIES = ("stratified", "queue")
HELP = """keys:
  Enter      accept the judge label (when there is one)
  1..K       choose an option by its number (yes/no questions: y / n; scores: the key shown, lowest first)
  <name>     choose an option by name (a unique prefix is enough for choices)
  s          skip this question (it stays in the pool)
  b          back: re-label the previous question
  q          quit (everything answered so far is saved)
  ?          this help"""


@dataclass
class AuditTask:
    item: LabelItem
    spec: HeadSpec
    judge: dict | None = None
    why: str | None = None


def confidence_band(c: Any) -> str:
    if not isinstance(c, (int, float)) or isinstance(c, bool):
        return "none"
    c = float(c)
    return "low" if c < 0.6 else ("mid" if c < 0.85 else "high")


def _judge_key(spec: HeadSpec, j: dict | None) -> str:
    if j is None:
        return "(no judge label)"
    try:
        return spec.labels[label_index(spec, j.get("label"))]
    except (ValueError, TypeError):
        return "(invalid judge label)"


def proportional_order(tasks: list[AuditTask], rng: random.Random) -> list[AuditTask]:
    """Stratified by (judge label, confidence band) with proportional allocation: member j of a stratum of size N
    gets the key (j + u) / N (u uniform per stratum), and sorting by key interleaves the strata so that any prefix
    holds each stratum in proportion to its size."""
    strata: dict[tuple[str, str], list[AuditTask]] = {}
    for t in tasks:
        k = (_judge_key(t.spec, t.judge), confidence_band((t.judge or {}).get("confidence")))
        strata.setdefault(k, []).append(t)
    keyed: list[tuple[float, int, AuditTask]] = []
    serial = 0
    for k in sorted(strata):
        members = strata[k]
        rng.shuffle(members)
        u = rng.random()
        for j, t in enumerate(members):
            keyed.append(((j + u) / len(members), serial, t))
            serial += 1
    keyed.sort(key=lambda x: (x[0], x[1]))
    return [t for _, _, t in keyed]


def _round_robin(lists: list[list[AuditTask]], n: int) -> list[AuditTask]:
    out: list[AuditTask] = []
    pos = [0] * len(lists)
    while len(out) < n and any(p < len(lst) for p, lst in zip(pos, lists, strict=True)):
        for i, lst in enumerate(lists):
            if pos[i] < len(lst) and len(out) < n:
                out.append(lst[pos[i]])
                pos[i] += 1
    return out


def _group_by_item(tasks: list[AuditTask]) -> list[AuditTask]:
    """Keep the selection, but ask one item's questions back to back (first-appearance order)."""
    groups: dict[str, list[AuditTask]] = {}
    for t in tasks:
        groups.setdefault(str(t.item.id), []).append(t)
    return [t for g in groups.values() for t in g]


def plan_audit(store: LabelStore, specs: dict[str, HeadSpec], n: int = 300, strategy: str = "stratified",
               seed: int = 7) -> list[AuditTask]:
    """The questions (item, head) to ask, at most `n`, none of which already has a human label."""
    if strategy not in STRATEGIES:
        raise ValueError(f"strategy must be one of {STRATEGIES}")
    human, judge = store.human_labels(), store.judge_labels()
    if strategy == "queue":
        items = {str(it.id): it for it in store.items()}
        out, seen = [], set()
        for r in read_queue(store):
            head, iid = r.get("head"), str(r.get("item_id"))
            spec, it = specs.get(head), items.get(iid)
            if spec is None or it is None or it.view != spec.view or (iid, head) in human or (iid, head) in seen:
                continue
            seen.add((iid, head))
            out.append(AuditTask(it, spec, judge.get((iid, head)), r.get("why")))
            if len(out) >= n:
                break
        return out
    rng = random.Random(seed)
    per_head = []
    for name, spec in specs.items():
        pool = [AuditTask(it, spec, judge.get((str(it.id), name))) for it in store.items(spec.view)
                if (str(it.id), name) not in human]
        per_head.append(proportional_order(pool, rng))
    return _group_by_item(_round_robin(per_head, max(0, int(n))))


# ----------------------------------------------------------------------------- answers

def option_lines(spec: HeadSpec) -> list[str]:
    if spec.type == "noul":
        pos = f" - {spec.positive}" if spec.positive else ""
        return [f"  y) true{pos}", "  n) false"]
    if spec.type == "score":  # the key is the level name, which is the level number for "0".."K-1"
        return [f"  {c}) {d}" if d != c else f"  {c})" for c, d in zip(spec.classes, spec.legend, strict=True)]
    return [f"  {i}) {c}" for i, c in enumerate(spec.classes, start=1)]


def parse_answer(raw: str, spec: HeadSpec) -> int | None:
    """Class index for what the human typed, or None when it is not an option."""
    s = raw.strip()
    if not s:
        return None
    if spec.type == "choice":
        low = [c.lower() for c in spec.classes]
        if s.lower() in low:
            return low.index(s.lower())
        if s.isdigit() and 1 <= int(s) <= len(spec.classes):
            return int(s) - 1
        pref = [i for i, c in enumerate(low) if c.startswith(s.lower())]
        return pref[0] if len(pref) == 1 else None
    try:
        return label_index(spec, s)
    except (ValueError, TypeError):
        pass
    if spec.type == "score":  # a level's description ("great") also names it
        legend = [d.lower() for d in spec.legend]
        if legend.count(s.lower()) == 1:
            return legend.index(s.lower())
    return None


# ----------------------------------------------------------------------------- summary

def head_summary(store: LabelStore, spec: HeadSpec, holdout_fraction: float = HOLDOUT_FRACTION,
                 min_verify_rows: int = MIN_VERIFY_ROWS, min_trusted: int = MIN_TRUSTED) -> dict:
    """Trusted rows of one head vs the minimum / recommended counts, and the judge's agreement with humans.

    Both counts follow a split that holds out the first count of at least `holdout_fraction` * n rows (a hashed
    split, or groups of one row). A time split needs a few rows fewer. Groups move the counts either way, and not
    monotonically: whole groups overshoot the fraction, so small groups (such as the documents pack's chunk items,
    up to label_top_k per query) often train from fewer rows, while a few large groups can leave too few
    development rows for the calibration folds. So the counts are estimates.

    `minimum` is how many trusted rows let the head train: `min_trusted` (train.min_trusted, 30 by default), and
    enough that MIN_HOLDOUT of them land in the holdout, 46 at `holdout_fraction` (train.holdout_fraction) 0.2. `for_verified_holdout` is how many
    trusted rows give a holdout of `min_verify_rows` (policy.min_verify_rows, 30): 146 at 0.2. Below it the head
    can train, but every verdict stays at confirm at best; acting also needs that many act-tier rows on the
    holdout, so treat it as a floor. `more_for_verified_holdout` is what is still missing."""
    ids = {str(it.id) for it in store.items(spec.view)}
    human = {k: v for k, v in store.human_labels(spec.name).items() if k[0] in ids}
    judge = store.judge_labels(spec.name)
    by_source: dict[str, int] = {}
    jy, hy = [], []
    for (iid, _), h in human.items():
        src = h.get("source") or "human"
        by_source[src] = by_source.get(src, 0) + 1
        j = judge.get((iid, spec.name))
        if j is None:
            continue
        try:
            hi, ji = label_index(spec, h.get("label")), label_index(spec, j.get("label"))
        except (ValueError, TypeError):
            continue
        hy.append(hi)
        jy.append(ji)
    n = len(human)
    frac = min(max(float(holdout_fraction), 1e-6), 1.0)

    def rows_for(holdout_rows: int) -> int:    # the fewest rows whose first count >= frac * n is holdout_rows
        return math.floor((holdout_rows - 1) / frac + 1e-9) + 1

    trainable = max(int(min_trusted), rows_for(MIN_HOLDOUT))
    verified = max(trainable, rows_for(int(min_verify_rows)))
    more = max(0, verified - n)
    holdout = f"a {int(min_verify_rows)}-row holdout"
    if n < trainable:
        why = (f"{MIN_HOLDOUT} must land in its holdout" if rows_for(MIN_HOLDOUT) > int(min_trusted)
               else f"train.min_trusted is {int(min_trusted)}")
        status = (f"about {trainable - n} more trusted rows needed before this head can train ({why}), about {more} "
                  f"more for {holdout} (below that, decisions are capped at confirm)")
    elif more:
        status = (f"can train, but its holdout (~{math.ceil(n * frac - 1e-9)} rows) is under {int(min_verify_rows)}: "
                  f"decisions stay capped at confirm; about {more} more trusted rows for {holdout}")
    elif n < RECOMMENDED_TRUSTED:
        status = (f"can train with {holdout}; ~{RECOMMENDED_TRUSTED} trusted rows recommended for tight holdout "
                  f"metrics")
    else:
        status = "recommended number of trusted rows reached"
    agr = agreement(np.asarray(jy, int), np.asarray(hy, int), spec.labels)
    return {"trusted": n, "by_source": by_source, "minimum": trainable, "recommended": RECOMMENDED_TRUSTED,
            "for_verified_holdout": verified, "more_for_verified_holdout": more, "status": status,
            "judge_agreement": agr}


def _verify_settings(project: Any) -> tuple[float, int, int]:
    """(train.holdout_fraction, policy.min_verify_rows, train.min_trusted) from rlcd.yaml, defaults when unset or
    unreadable."""
    get = getattr(project, "get", None)
    try:
        frac = float(get("train.holdout_fraction", HOLDOUT_FRACTION)) if callable(get) else HOLDOUT_FRACTION
        k = int(get("policy.min_verify_rows", MIN_VERIFY_ROWS)) if callable(get) else MIN_VERIFY_ROWS
        mt = int(get("train.min_trusted", MIN_TRUSTED)) if callable(get) else MIN_TRUSTED
    except (TypeError, ValueError):
        return HOLDOUT_FRACTION, MIN_VERIFY_ROWS, MIN_TRUSTED
    return ((frac if 0.0 < frac <= 1.0 else HOLDOUT_FRACTION), (k if k > 0 else MIN_VERIFY_ROWS),
            (mt if mt > 0 else MIN_TRUSTED))


# ----------------------------------------------------------------------------- the session

class _Style:
    def __init__(self, on: bool):
        self.on = on

    def __call__(self, text: str, code: str) -> str:
        return f"\033[{code}m{text}\033[0m" if self.on else text


def audit(project: Any, pack: Any, heads: list[str] | None = None, n: int = 300, strategy: str = "stratified",
          annotator: str | None = None, input_fn: Callable[[str], str] = input,
          print_fn: Callable[..., None] = print, store: LabelStore | None = None, seed: int = 7,
          color: bool | None = None) -> dict:
    """Run an interactive audit session (see the module docstring for keys and sampling).

    Returns {strategy, planned, asked, labelled, accepted_judge, agreed, overruled, skipped, quit, remaining,
    heads: {head: head_summary}}."""
    store = store if store is not None else LabelStore(project.home)
    specs = select_heads(pack.heads(project), heads)
    tasks = plan_audit(store, specs, n, strategy, seed)
    style = _Style(bool(color) if color is not None else (print_fn is print and sys.stdout.isatty()))
    outcome: dict[int, str] = {}          # task index -> accepted | agreed | overruled | chosen | skipped
    quit_early = False
    i, shown_item = 0, None
    if not tasks:
        how = ("the queue is empty or fully labelled (fill it with `rlcd label next`)" if strategy == "queue" else
               "every selected question already has a human label, or there are no items yet (`rlcd label "
               "bootstrap` adds them)")
        print_fn(f"nothing to audit: {how}")
    else:
        print_fn(f"{len(tasks)} question(s) to review ({strategy}). ? = help, q = quit; answers are saved at once.")
    while i < len(tasks):
        t = tasks[i]
        spec, iid = t.spec, str(t.item.id)
        if shown_item != iid:
            print_fn("")
            print_fn(style(f"== item {iid} ==", "1"))
            text = (t.item.display or "").strip() or str(t.item.state)
            print_fn(text if len(text) <= MAX_DISPLAY else text[:MAX_DISPLAY] + " ...(truncated)")
            shown_item = iid
        print_fn(style(f"[{i + 1}/{len(tasks)}] {spec.name}: {spec.description or spec.name + '?'}", "36"))
        for line in option_lines(spec):
            print_fn(line)
        judge_idx = None
        if t.judge is not None:
            try:
                judge_idx = label_index(spec, t.judge.get("label"))
            except (ValueError, TypeError):
                judge_idx = None
        if judge_idx is not None:
            conf = t.judge.get("confidence")
            conf_s = f" (confidence {float(conf):.2f})" if isinstance(conf, (int, float)) else ""
            why = f" - {t.judge['rationale']}" if t.judge.get("rationale") else ""
            print_fn(style(f"judge: {spec.labels[judge_idx]}{conf_s}{why}", "33"))
        else:
            print_fn(style("judge: no label", "2"))
        if t.why:
            print_fn(style(f"queued because: {t.why}", "2"))
        cur = store.human_labels(spec.name).get((iid, spec.name))
        if cur is not None:
            print_fn(style(f"current human label: {cur.get('label')}", "35"))
        prompt = "label [Enter = accept judge, ? = help]: " if judge_idx is not None else "label [? = help]: "
        try:
            raw = input_fn(prompt)
        except (EOFError, KeyboardInterrupt):
            quit_early = True
            break
        cmd = raw.strip()
        if cmd.lower() == "q":
            quit_early = True
            break
        if cmd == "?":
            print_fn(HELP)
            continue
        if cmd.lower() == "s":
            outcome[i] = "skipped"
            i += 1
            continue
        if cmd.lower() == "b":
            if i == 0:
                print_fn("nothing to go back to")
            else:
                i -= 1
                shown_item = None
            continue
        if cmd == "":
            if judge_idx is None:
                print_fn("there is no judge label to accept: choose an option")
                continue
            idx, how = judge_idx, "accepted"
        else:
            idx = parse_answer(cmd, spec)
            if idx is None:
                print_fn(f"{cmd!r} is not an option (? = help)")
                continue
            how = "chosen" if judge_idx is None else ("agreed" if idx == judge_idx else "overruled")
        store.add_human(iid, spec.name, spec.labels[idx], source="human", annotator=annotator,
                        note=f"sample={strategy}; judge={how if how != 'chosen' else 'none'}")
        outcome[i] = how
        i += 1

    frac, k, mt = _verify_settings(project)
    summary = {name: head_summary(store, spec, frac, k, mt) for name, spec in specs.items()}
    print_fn("")
    print_fn(style("== trusted rows and judge agreement ==", "1"))
    for name, s in summary.items():
        a = s["judge_agreement"]
        agr = (f"judge agreement {a['accuracy']:.0%} on {a['n']} rows, kappa "
               f"{'n/a' if a.get('cohen_kappa') is None else format(a['cohen_kappa'], '.2f')}"
               if a.get("n") else "no rows with both a judge and a human label")
        print_fn(f"{name}: {s['trusted']} trusted ({s['status']}); {agr}")
    vals = list(outcome.values())
    labelled = [v for v in vals if v != "skipped"]
    return {"strategy": strategy, "planned": len(tasks), "asked": len(outcome), "labelled": len(labelled),
            "accepted_judge": vals.count("accepted"), "agreed": vals.count("agreed"),
            "overruled": vals.count("overruled"), "skipped": vals.count("skipped"), "quit": quit_early,
            "remaining": len(tasks) - len(outcome), "heads": summary}


__all__ = ["MIN_TRUSTED", "RECOMMENDED_TRUSTED", "AuditTask", "audit", "head_summary", "option_lines",
           "parse_answer", "plan_audit", "proportional_order"]
