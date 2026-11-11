"""The terminal audit, driven by a scripted input_fn: exactly the expected human labels, agreement, resumability."""
from __future__ import annotations

from collections.abc import Iterator

import pytest

from open_rlcd.core.config import Project
from open_rlcd.core.head import HeadSpec
from open_rlcd.labeling.active import write_queue
from open_rlcd.labeling.audit import audit, option_lines, parse_answer, plan_audit
from open_rlcd.labeling.store import LabelItem, LabelStore
from open_rlcd.packs.base import BasePack

KIND = HeadSpec("kind", "choice", ("question", "statement", "command"), view="text",
                description="What kind of sentence is this?")
POLITE = HeadSpec("polite", "noul", view="text", description="Is it polite?", positive="courteous")
QUALITY = HeadSpec("quality", "score", ("0", "1", "2"), view="text", levels=("bad", "ok", "great"))


class AuditPack(BasePack):
    name = "audittoy"

    def heads(self, project):
        return {"kind": KIND, "polite": POLITE, "quality": QUALITY}

    def provider(self, project, config=None):
        return None

    def datasets(self, project, provider, heads, feedback=None, feedback_weight=5.0):
        return {}


def scripted(answers: list[str]):
    it: Iterator[str] = iter(answers)
    prompts: list[str] = []

    def input_fn(prompt: str) -> str:
        prompts.append(prompt)
        try:
            return next(it)
        except StopIteration:
            raise EOFError from None

    input_fn.prompts = prompts  # type: ignore[attr-defined]
    return input_fn


@pytest.fixture
def project(home, tmp_path):
    return Project(root=tmp_path)


@pytest.fixture
def store(home):
    s = LabelStore(home)
    s.add_items([LabelItem(f"t{i}", "text", {"i": i}, display=f"sentence number {i}") for i in range(6)])
    for i in range(6):
        s.add_judge(f"t{i}", "kind", "question", "scripted", confidence=0.9, rationale="looks like one")
    return s


def test_scripted_session_writes_exactly_the_expected_labels(project, store):
    write_queue(store, "kind", [{"head": "kind", "item_id": f"t{i}", "score": 1.0 - i / 10, "why": "test"}
                                for i in range(6)])
    printed: list[str] = []
    inputs = scripted([
        "",          # t0: accept the judge label (question)
        "zzz",       # t1: not an option -> asked again
        "2",         # t1: choose by number -> statement (overrules the judge)
        "command",   # t2: choose by name
        "b",         # back to t2 ...
        "1",         # ... re-label it: question (agrees with the judge)
        "s",         # t3: skip
        "?",         # t4: help -> asked again
        "q",         # quit
    ])
    out = audit(project, AuditPack(), heads=["kind"], strategy="queue", annotator="ann", input_fn=inputs,
                print_fn=lambda *a: printed.append(" ".join(map(str, a))), store=store)
    human = store.human_labels("kind")
    assert {k[0]: v["label"] for k, v in human.items()} == {"t0": "question", "t1": "statement", "t2": "question"}
    assert len(list(store.human_log.rows())) == 4      # t2 twice: the latest row wins
    assert all(v["source"] == "human" and v["annotator"] == "ann" for v in human.values())
    assert human[("t0", "kind")]["note"] == "sample=queue; judge=accepted"
    assert human[("t1", "kind")]["note"] == "sample=queue; judge=overruled"
    assert human[("t2", "kind")]["note"] == "sample=queue; judge=agreed"
    assert out["labelled"] == 3 and out["skipped"] == 1 and out["quit"] is True
    assert out["accepted_judge"] == 1 and out["overruled"] == 1 and out["agreed"] == 1
    assert out["planned"] == 6 and out["remaining"] == 2
    kind = out["heads"]["kind"]
    assert kind["trusted"] == 3 and kind["minimum"] == 46 and kind["recommended"] == 300
    assert "43 more" in kind["status"]
    agr = kind["judge_agreement"]
    assert agr["n"] == 3 and agr["accuracy"] == pytest.approx(2 / 3, abs=1e-6)
    text = "\n".join(printed)
    assert "not an option" in text and "keys:" in text and "judge: question (confidence 0.90)" in text
    assert "queued because: test" in text
    assert "\033[" not in text  # no colour when not printing to a terminal
    assert "judge agreement 67% on 3 rows" in text


def test_audit_is_resumable_and_skips_human_labelled(project, store):
    audit(project, AuditPack(), heads=["kind"], n=2, input_fn=scripted(["", ""]), print_fn=lambda *a: None,
          store=store)
    assert len(store.human_labels("kind")) == 2
    done = {k[0] for k in store.human_labels("kind")}
    plan = plan_audit(store, {"kind": KIND}, n=100)
    assert len(plan) == 4 and not done & {str(t.item.id) for t in plan}
    out = audit(project, AuditPack(), heads=["kind"], input_fn=scripted([]), print_fn=lambda *a: None, store=store)
    assert out["quit"] is True and out["labelled"] == 0   # EOF on the first prompt ends the session cleanly


def test_enter_without_judge_label_and_other_head_types(project, store):
    printed: list[str] = []
    out = audit(project, AuditPack(), heads=["polite", "quality"], n=2, input_fn=scripted(["", "y", "great"]),
                print_fn=lambda *a: printed.append(" ".join(map(str, a))), store=store)
    assert "no judge label to accept" in "\n".join(printed)
    labels = {(k[1]): v["label"] for k, v in store.human_labels().items()}
    assert labels == {"polite": "true", "quality": "2"}
    assert out["labelled"] == 2 and out["remaining"] == 0


def test_stratified_order_is_proportional(project, home):
    s = LabelStore(home / "strat")
    s.add_items([LabelItem(f"x{i}", "text", {}, display=str(i)) for i in range(100)])
    for i in range(100):
        s.add_judge(f"x{i}", "kind", "statement" if i < 80 else "question", "j", confidence=0.95)
    plan = plan_audit(s, {"kind": KIND}, n=100, seed=3)
    for prefix in (10, 25, 50):
        n_statement = sum(t.judge["label"] == "statement" for t in plan[:prefix])
        assert abs(n_statement - 0.8 * prefix) <= 1.0
    assert len({t.item.id for t in plan}) == 100


def test_questions_of_one_item_are_asked_together(project, store):
    plan = plan_audit(store, {"kind": KIND, "polite": POLITE}, n=12)
    ids = [str(t.item.id) for t in plan]
    for iid in set(ids):
        pos = [i for i, x in enumerate(ids) if x == iid]
        assert pos == list(range(pos[0], pos[0] + len(pos)))


def test_parse_answer_per_head_type():
    assert parse_answer("2", KIND) == 1 and parse_answer("Command", KIND) == 2 and parse_answer("st", KIND) == 1
    assert parse_answer("9", KIND) is None and parse_answer("", KIND) is None
    assert parse_answer("y", POLITE) == 1 and parse_answer("n", POLITE) == 0 and parse_answer("true", POLITE) == 1
    assert parse_answer("0", QUALITY) == 0 and parse_answer("2", QUALITY) == 2 and parse_answer("OK", QUALITY) == 1
    assert parse_answer("3", QUALITY) is None
    rating = HeadSpec("rating", "score", ("1", "2", "3", "4", "5"), view="text",
                      levels=("awful", "poor", "fine", "good", "superb"))
    assert option_lines(rating)[0] == "  1) awful" and option_lines(QUALITY)[0] == "  0) bad"
    assert parse_answer("5", rating) == 4 and parse_answer("1", rating) == 0 and parse_answer("good", rating) == 3


def test_unknown_strategy_and_heads(project, store):
    with pytest.raises(ValueError):
        plan_audit(store, {"kind": KIND}, strategy="random")
    with pytest.raises(ValueError, match="unknown heads"):
        audit(project, AuditPack(), heads=["nope"], input_fn=scripted([]), print_fn=lambda *a: None, store=store)


def test_head_summary_says_how_many_rows_a_verified_holdout_needs(home):
    """F36: a head trains from about 46 trusted rows (at least 30, and enough that 10 land in its 20 % holdout), but
    that holdout is under the 30 rows that `verify` needs: the summary says so and how many more rows are missing
    (it follows train.holdout_fraction). Both counts are where a hashed split first holds out 10 and 30 rows."""
    from open_rlcd.labeling.audit import head_summary

    store = LabelStore(home)
    store.add_items([LabelItem(f"t{i}", "text", {"i": i}) for i in range(160)])
    for i in range(40):
        store.add_human(f"t{i}", "polite", "true" if i % 2 else "false")
    s = head_summary(store, POLITE)
    assert s["for_verified_holdout"] == 146 and s["more_for_verified_holdout"] == 106
    assert "capped at confirm" in s["status"] and "106 more" in s["status"]
    assert s["minimum"] == 46 and s["status"].startswith(
        "about 6 more trusted rows needed before this head can train (10 must land in its holdout)"), s["status"]
    s = head_summary(store, POLITE, holdout_fraction=0.25)
    assert s["for_verified_holdout"] == 117 and s["more_for_verified_holdout"] == 77 and s["minimum"] == 37
    for i in range(40, 45):
        store.add_human(f"t{i}", "polite", "true")
    assert head_summary(store, POLITE)["status"].startswith("about 1 more trusted rows needed before this head can train")
    store.add_human("t45", "polite", "true")                 # 46: a hashed split holds out 10 rows, enough to train
    assert head_summary(store, POLITE)["status"].startswith("can train, but its holdout (~10 rows) is under 30")
    for i in range(46, 145):
        store.add_human(f"t{i}", "polite", "true")
    assert "(~29 rows) is under 30" in head_summary(store, POLITE)["status"]
    store.add_human("t145", "polite", "true")                # 146: the hashed holdout has 30 rows
    assert "can train with a 30-row holdout" in head_summary(store, POLITE)["status"]
    for i in range(146, 155):
        store.add_human(f"t{i}", "polite", "true")
    s = head_summary(store, POLITE)
    assert s["more_for_verified_holdout"] == 0 and "can train with a 30-row holdout" in s["status"]


def test_the_summary_follows_train_min_trusted(home, tmp_path):
    """A project that raises train.min_trusted needs that many trusted rows before a head trains: the summary uses
    the project's value (it used to say "can train" from 30 rows whatever the setting) and names the rule that
    binds."""
    from open_rlcd.labeling.audit import head_summary

    store = LabelStore(home)
    store.add_items([LabelItem(f"t{i}", "text", {"i": i}) for i in range(60)])
    for i in range(50):
        store.add_human(f"t{i}", "polite", "true" if i % 2 else "false")
    s = head_summary(store, POLITE, min_trusted=60)
    assert s["minimum"] == 60, s
    assert s["status"].startswith("about 10 more trusted rows needed before this head can train (train.min_trusted "
                                  "is 60)"), s["status"]
    assert head_summary(store, POLITE)["status"].startswith("can train, but its holdout (~10 rows)")
    out = audit(Project(root=tmp_path, data={"train": {"min_trusted": 60}}), AuditPack(), heads=["polite"],
                input_fn=scripted([]), print_fn=lambda *a: None, store=store)
    assert out["heads"]["polite"]["minimum"] == 60
