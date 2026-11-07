"""`rlcd label bootstrap`: weak labels from an LLM judge for every item a pack wants labelled.

Generic over packs: items come from `pack.label_tasks(project, provider)` (LabelItems, or dicts of the same
shape) and from the label store; a head applies to an item when `spec.view == item.view`.

  * cheap: one judge call per batch of items of one view, covering every head of that view the batch still needs;
  * resumable: an (item, head) pair that already has a judge label is never sent again, so an interrupted or
    failed run continues where it stopped;
  * honest about provenance: judge labels go to judge.jsonl only. Human labels (human.jsonl) are never written or
    changed here, and they always outrank judge labels. Items that already have a human label are still judged by
    default, because the judge's agreement with humans on exactly those items is what training reports.

What this does not do: check that the judge is any good. That is measured afterwards, on audited items
(`rlcd label audit` prints the agreement; training reports it next to each head's holdout metrics).
"""
from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from typing import Any

from ..core.head import HeadSpec
from .judges import Judge, JudgeError
from .store import LabelItem, LabelStore

MAX_PROBLEMS = 500


def as_label_items(tasks: Iterable[Any]) -> list[LabelItem]:
    """LabelItems from what a pack yields (LabelItem or dict), with string ids."""
    out: list[LabelItem] = []
    for t in tasks or []:
        if isinstance(t, LabelItem):
            it = t
        elif isinstance(t, dict):
            it = LabelItem.from_dict(t)
        else:
            raise TypeError(f"label_tasks must yield LabelItem or dict, got {type(t).__name__}")
        if it.id is None or it.view is None:
            raise ValueError(f"a label item needs an id and a view, got {it.to_dict()!r:.200}")
        it.id = str(it.id)
        out.append(it)
    return out


def select_heads(all_specs: dict[str, HeadSpec], heads: Sequence[str] | None) -> dict[str, HeadSpec]:
    names = list(heads) if heads else list(all_specs)
    unknown = [h for h in names if h not in all_specs]
    if unknown:
        raise ValueError(f"unknown heads {unknown}; this pack defines {list(all_specs)}")
    return {h: all_specs[h] for h in names}


def bootstrap(project: Any, pack: Any, provider: Any, judge: Judge, heads: list[str] | None = None,
              limit: int | None = None, batch: int = 20,
              progress: Callable[[int, int, str], None] | None = None, store: LabelStore | None = None,
              skip_human: bool = False) -> dict:
    """Weak-label the pack's items with `judge` and store the labels.

    `limit` caps how many items are sent in this run (a cost cap; the rest wait for the next run), `batch` is the
    number of items per judge call, `progress(done_items, total_items, message)` is called after every batch,
    `skip_human=True` saves cost by not judging items a human already labelled (no agreement data for them).

    Returns {judge, items_added, items_total, items_sent, items_pending (still without a judge label, including
    those beyond `limit`), batches, batches_planned, labels: {head: new judge labels},
    already_judged: {head: n}, problems: [...], problems_total, stopped: None | reason, usage}."""
    if batch < 1:
        raise ValueError("batch must be >= 1")
    store = store if store is not None else LabelStore(project.home)
    specs = select_heads(pack.heads(project), heads)
    added = store.add_items(as_label_items(pack.label_tasks(project, provider)))
    items = store.items()
    judged = store.judge_labels()
    human = store.human_labels() if skip_human else {}

    already = dict.fromkeys(specs, 0)
    by_view: dict[str, list[tuple[LabelItem, list[str]]]] = {}
    for it in items:
        iid = str(it.id)
        need = []
        for name, spec in specs.items():
            if spec.view != it.view:
                continue
            if (iid, name) in judged:
                already[name] += 1
            elif not (skip_human and (iid, name) in human):
                need.append(name)
        if need:
            by_view.setdefault(it.view, []).append((it, need))

    queue = [(view, it, need) for view, rows in by_view.items() for it, need in rows]
    n_need = len(queue)
    if limit is not None:
        queue = queue[:max(0, int(limit))]
    chunks: list[tuple[str, list[tuple[LabelItem, list[str]]]]] = []
    for view in dict.fromkeys(v for v, _, _ in queue):
        rows = [(it, need) for v, it, need in queue if v == view]
        chunks += [(view, rows[i:i + batch]) for i in range(0, len(rows), batch)]

    counts = dict.fromkeys(specs, 0)
    problems: list[str] = []
    n_problems, done, n_batches, stopped = 0, 0, 0, None
    total = len(queue)
    for k, (view, rows) in enumerate(chunks, start=1):
        names = [h for h in specs if any(h in need for _, need in rows)]
        wanted = {(str(it.id), h) for it, need in rows for h in need}
        try:
            answers, probs = judge.label([it for it, _ in rows], [specs[h] for h in names])
        except JudgeError as e:
            stopped = str(e)
            break
        written: set[tuple[str, str]] = set()
        for a in answers:
            key = (str(a.item_id), a.head)
            if key not in wanted or key in written:
                continue
            store.add_judge(key[0], a.head, a.label, judge.id, a.confidence, a.rationale)
            written.add(key)
            counts[a.head] += 1
        n_problems += len(probs)
        room = MAX_PROBLEMS - len(problems)
        if room > 0:
            problems += [f"batch {k} ({view}): {p}" for p in probs[:room]]
        done += len(rows)
        n_batches += 1
        if progress is not None:
            progress(done, total, f"{view}: batch {k}/{len(chunks)}, {len(written)} labels")
    return {"judge": judge.id, "items_added": added, "items_total": len(items), "items_sent": done,
            "items_pending": n_need - done, "batches": n_batches, "batches_planned": len(chunks),
            "labels": counts, "already_judged": already, "problems": problems, "problems_total": n_problems,
            "stopped": stopped, "usage": getattr(judge, "usage", None)}
