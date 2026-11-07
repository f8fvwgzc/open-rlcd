"""Bootstrap with a scripted judge: writes judge labels, resumes without duplicating, never touches human labels."""
from __future__ import annotations

import pytest

from open_rlcd.core.config import Project
from open_rlcd.core.head import HeadSpec
from open_rlcd.labeling.bootstrap import bootstrap
from open_rlcd.labeling.judges import JudgeError, ScriptedJudge
from open_rlcd.labeling.prompts import JudgeAnswer
from open_rlcd.labeling.store import LabelItem, LabelStore
from open_rlcd.packs.base import BasePack

TEXTS = {"t1": "Could you please help?", "t2": "Close the door.", "t3": "Is it raining?", "t4": "Thanks a lot.",
         "t5": "Please sit down."}


class LabPack(BasePack):
    name = "labtoy"

    def __init__(self, extra: list | None = None):
        self.extra = list(extra or [])

    def heads(self, project):
        return {"kind": HeadSpec("kind", "choice", ("question", "statement", "command"), view="text"),
                "polite": HeadSpec("polite", "noul", view="text"),
                "quality": HeadSpec("quality", "score", ("0", "1", "2"), view="pair")}

    def provider(self, project, config=None):
        return None

    def datasets(self, project, provider, heads, feedback=None, feedback_weight=5.0):
        return {}

    def label_tasks(self, project, provider=None):
        items = [LabelItem(k, "text", {"text": v}, display=v) for k, v in TEXTS.items()]
        items += [{"id": f"p{i}", "view": "pair", "state": {"q": i}, "display": f"pair {i}"} for i in range(3)]
        return items + self.extra


def rule(item, spec):
    text = item.display
    if spec.name == "kind":
        return ("question" if text.endswith("?") else "statement"), 0.8
    if spec.name == "polite":
        return ("please" in text.lower() or "thanks" in text.lower()), 0.9
    return 2, 0.6


@pytest.fixture
def project(home, tmp_path):
    return Project(root=tmp_path)


def test_bootstrap_writes_judge_labels_in_view_batches(project, home):
    judge = ScriptedJudge(rule)
    seen: list[tuple[int, int, str]] = []
    out = bootstrap(project, LabPack(), None, judge, batch=2, progress=lambda d, t, m: seen.append((d, t, m)))
    store = LabelStore(home)
    assert out["items_added"] == 8 and out["items_total"] == 8
    assert out["labels"] == {"kind": 5, "polite": 5, "quality": 3}
    assert out["batches"] == 5 and out["stopped"] is None and out["problems"] == []
    # one call per batch, every item of a batch shares a view, and the call covers that view's heads only
    assert judge.calls == 5
    for ids, heads in judge.batches:
        views = {"text" if i.startswith("t") else "pair" for i in ids}
        assert len(views) == 1 and len(ids) <= 2
        assert heads == (["kind", "polite"] if views == {"text"} else ["quality"])
    j = store.judge_labels()
    assert j[("t3", "kind")]["label"] == "question" and j[("t3", "kind")]["confidence"] == 0.8
    assert j[("t1", "polite")]["label"] == "true" and j[("t2", "polite")]["label"] == "false"
    assert j[("p0", "quality")]["label"] == "2" and j[("p0", "quality")]["judge"] == "scripted"
    assert seen[-1][0] == seen[-1][1] == 8


def test_bootstrap_resumes_without_duplicating(project, home):
    bootstrap(project, LabPack(), None, ScriptedJudge(rule), batch=3)
    store = LabelStore(home)
    rows_before = len(list(store.judge_log.rows()))
    again = ScriptedJudge(rule)
    out = bootstrap(project, LabPack(), None, again, batch=3)
    assert again.calls == 0 and out["items_sent"] == 0
    assert out["labels"] == {"kind": 0, "polite": 0, "quality": 0}
    assert out["already_judged"] == {"kind": 5, "polite": 5, "quality": 3}
    assert len(list(store.judge_log.rows())) == rows_before
    # a new item from the pack is the only thing sent next time
    pack = LabPack(extra=[LabelItem("t6", "text", {}, display="Where is it?")])
    third = ScriptedJudge(rule)
    out = bootstrap(project, pack, None, third)
    assert third.batches == [(["t6"], ["kind", "polite"])] and out["items_added"] == 1


def test_bootstrap_never_overrides_human_labels(project, home):
    store = LabelStore(home)
    store.add_items([LabelItem("t1", "text", {"text": TEXTS["t1"]}, display=TEXTS["t1"])])
    store.add_human("t1", "kind", "command", annotator="ann")
    human_before = store.human_log.path.read_bytes()
    out = bootstrap(project, LabPack(), None, ScriptedJudge(rule), store=store)
    assert store.human_log.path.read_bytes() == human_before
    rows = {r["item"].id: r for r in store.labelled("kind", "text")}
    assert rows["t1"]["source"] == "human" and rows["t1"]["label"] == "command"
    assert rows["t1"]["judge_label"] == "question"  # kept for judge-vs-human agreement
    assert out["labels"]["kind"] == 5

    # skip_human: items a human already labelled are not sent for that head
    store2 = LabelStore(home / "other")
    store2.add_items([LabelItem("t1", "text", {}, display=TEXTS["t1"])])
    store2.add_human("t1", "kind", "command")
    judge = ScriptedJudge(rule)
    bootstrap(project, LabPack(), None, judge, store=store2, heads=["kind"], skip_human=True)
    sent = [i for ids, _ in judge.batches for i in ids]
    assert "t1" not in sent and len(sent) == 4


def test_bootstrap_heads_and_limit(project, home):
    judge = ScriptedJudge(rule)
    out = bootstrap(project, LabPack(), None, judge, heads=["polite"], limit=2)
    assert out["labels"] == {"polite": 2} and out["items_sent"] == 2 and out["items_pending"] == 3
    assert all(h == ["polite"] for _, h in judge.batches)
    with pytest.raises(ValueError, match="unknown heads"):
        bootstrap(project, LabPack(), None, judge, heads=["nope"])
    with pytest.raises(ValueError):
        bootstrap(project, LabPack(), None, judge, batch=0)


def test_bootstrap_stops_on_judge_error_and_resumes(project, home):
    class Flaky:
        id = "flaky"

        def __init__(self):
            self.calls = 0

        def label(self, items, heads):
            self.calls += 1
            if self.calls == 2:
                raise JudgeError("rate limited after retries")
            return ScriptedJudge(rule).label(items, heads)

    out = bootstrap(project, LabPack(), None, Flaky(), heads=["kind"], batch=2)
    assert out["stopped"] == "rate limited after retries"
    assert out["labels"]["kind"] == 2 and out["items_pending"] == 3
    out = bootstrap(project, LabPack(), None, ScriptedJudge(rule), heads=["kind"], batch=2)
    assert out["labels"]["kind"] == 3 and out["stopped"] is None


def test_bootstrap_ignores_answers_it_did_not_ask_for(project, home):
    class Chatty:
        id = "chatty"

        def label(self, items, heads):
            out = [JudgeAnswer(str(it.id), h.name, h.labels[0], 0.5, None) for it in items for h in heads]
            out += [JudgeAnswer("zzz", "kind", "question"), JudgeAnswer("t1", "quality", "1"),
                    JudgeAnswer(str(items[0].id), heads[0].name, heads[0].labels[-1])]  # duplicate
            return out, ["one odd thing"]

    out = bootstrap(project, LabPack(), None, Chatty(), heads=["kind"], batch=10)
    store = LabelStore(home)
    assert out["labels"]["kind"] == 5 and ("zzz", "kind") not in store.judge_labels()
    assert ("t1", "quality") not in store.judge_labels()
    assert store.judge_labels()[("t1", "kind")]["label"] == "question"  # the first answer, not the duplicate
    assert out["problems"] == ["batch 1 (text): one odd thing"] and out["problems_total"] == 1


def test_label_tasks_with_int_ids_are_normalised(project, home):
    class IntPack(LabPack):
        def label_tasks(self, project, provider=None):
            return [{"id": 7, "view": "text", "state": {}, "display": "Hello?"}]

    bootstrap(project, IntPack(), None, ScriptedJudge(rule), heads=["kind"])
    store = LabelStore(home)
    assert [it.id for it in store.items()] == ["7"]
    assert ("7", "kind") in store.judge_labels()
