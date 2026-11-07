"""The label store: items to label and the labels they received, as JSON lines under RLCD_HOME/labels/.

    items.jsonl   one row per item: {id, view, state, display, group, t, meta}
                  `view` names the feature view (e.g. "query", "query_chunk"); every head with that view can
                  be labelled on the item. `state` is exactly what the head sees at serve time.
    judge.jsonl   weak labels from an LLM judge: {item_id, head, label, judge, confidence, rationale, at}
    human.jsonl   trusted labels: {item_id, head, label, source, annotator, at, note}
                  source = human (audited), ground_truth (known outcome) or synthetic (generated with its answer)

The latest row per (item_id, head) wins in each file. A human label always outranks a judge label; the judge
label of a human-labelled item is kept to measure the judge's agreement with humans.
"""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ..core.feedback import JsonlLog
from ..core.training import now_iso

TRUSTED_LABEL_SOURCES = ("human", "ground_truth", "synthetic")


@dataclass
class LabelItem:
    id: str
    view: str
    state: Any
    display: str = ""
    group: str | None = None
    t: int | None = None
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    def __post_init__(self) -> None:
        self.id = str(self.id)   # labels are keyed by str(item_id): an int id would never match its labels

    @classmethod
    def from_dict(cls, d: dict) -> LabelItem:
        return cls(**{k: d.get(k) for k in ("id", "view", "state", "display", "group", "t")},
                   meta=d.get("meta") or {})


class LabelStore:
    def __init__(self, home: Path):
        self.dir = Path(home) / "labels"
        self.items_log = JsonlLog(self.dir / "items.jsonl")
        self.judge_log = JsonlLog(self.dir / "judge.jsonl")
        self.human_log = JsonlLog(self.dir / "human.jsonl")

    # ------------------------------------------------------------------ items
    def add_items(self, items: Iterable[LabelItem]) -> int:
        have = {i.id for i in self.items()}
        n = 0
        for it in items:
            if it.id in have:
                continue
            self.items_log.append(it.to_dict())
            have.add(it.id)
            n += 1
        return n

    def items(self, view: str | None = None) -> list[LabelItem]:
        seen: dict[str, LabelItem] = {}
        for r in self.items_log.rows():
            it = LabelItem.from_dict(r)
            if view is None or it.view == view:
                seen[it.id] = it
        return list(seen.values())

    # ------------------------------------------------------------------ labels
    def add_judge(self, item_id: str, head: str, label: Any, judge: str, confidence: float | None = None,
                  rationale: str | None = None) -> None:
        self.judge_log.append({"item_id": item_id, "head": head, "label": label, "judge": judge,
                               "confidence": confidence, "rationale": rationale, "at": now_iso()})

    def add_human(self, item_id: str, head: str, label: Any, source: str = "human", annotator: str | None = None,
                  note: str | None = None) -> None:
        if source not in TRUSTED_LABEL_SOURCES:
            raise ValueError(f"trusted label source must be one of {TRUSTED_LABEL_SOURCES}")
        self.human_log.append({"item_id": item_id, "head": head, "label": label, "source": source,
                               "annotator": annotator, "note": note, "at": now_iso()})

    @staticmethod
    def _latest(log: JsonlLog, head: str | None) -> dict[tuple[str, str], dict]:
        out: dict[tuple[str, str], dict] = {}
        for r in log.rows():
            if head is None or r.get("head") == head:
                out[(str(r["item_id"]), str(r["head"]))] = r
        return out

    def judge_labels(self, head: str | None = None) -> dict[tuple[str, str], dict]:
        return self._latest(self.judge_log, head)

    def human_labels(self, head: str | None = None) -> dict[tuple[str, str], dict]:
        return self._latest(self.human_log, head)

    def labelled(self, head: str, view: str) -> list[dict]:
        """Every item of `view` with a label for `head`: {item, label, source, judge_label}."""
        human, judge = self.human_labels(head), self.judge_labels(head)
        out = []
        for it in self.items(view):
            h, j = human.get((it.id, head)), judge.get((it.id, head))
            if h is not None:
                out.append({"item": it, "label": h["label"], "source": h.get("source") or "human",
                            "judge_label": None if j is None else j["label"], "note": h.get("note")})
            elif j is not None:
                out.append({"item": it, "label": j["label"], "source": "judge", "judge_label": j["label"],
                            "judge_confidence": j.get("confidence")})
        return out

    def counts(self) -> dict:
        items = self.items()
        human, judge = self.human_labels(), self.judge_labels()
        by_head: dict[str, dict] = {}
        for (_, head) in human:
            by_head.setdefault(head, {"human": 0, "judge": 0})
            by_head[head]["human"] += 1
        for (_, head) in judge:
            by_head.setdefault(head, {"human": 0, "judge": 0})
            by_head[head]["judge"] += 1
        return {"items": len(items), "by_view": _count([i.view for i in items]), "labels": by_head}


def _count(xs: list[str]) -> dict[str, int]:
    out: dict[str, int] = {}
    for x in xs:
        out[x] = out.get(x, 0) + 1
    return out
