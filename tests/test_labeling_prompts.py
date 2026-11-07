"""The judge prompt and its tolerant parser (offline)."""
from __future__ import annotations

import json

from open_rlcd.core.head import HeadSpec
from open_rlcd.labeling.prompts import JudgeAnswer, build_prompt, extract_json, parse_response
from open_rlcd.labeling.store import LabelItem

KIND = HeadSpec("kind", "choice", ("question", "statement", "command"), view="text",
                description="What kind of sentence is this?")
POLITE = HeadSpec("polite", "noul", view="text", description="Is the sentence polite?",
                  positive="the speaker is courteous")
QUALITY = HeadSpec("quality", "score", ("0", "1", "2"), view="text", levels=("unusable", "acceptable", "excellent"),
                   description="How well written is it?")
TIER = HeadSpec("tier", "score", ("low", "mid", "high"), view="text", description="Which tier?")
HEADS = [KIND, POLITE, QUALITY]
ITEMS = [LabelItem("s1", "text", {"text": "Could you help me?"}, display="Could you help me?"),
         LabelItem("s2", "text", {"text": "Close the door."}, display="Close the door."),
         LabelItem("s3", "text", {"text": "It rains."})]


def answer(entries: list[dict]) -> str:
    return json.dumps({"items": entries})


def test_prompt_lists_every_class_level_and_item():
    system, user = build_prompt(ITEMS, HEADS)
    for c in KIND.classes:
        assert f'"{c}"' in system
    for d in QUALITY.legend:
        assert d in system
    for i in range(len(QUALITY.classes)):
        assert f"{i} = {QUALITY.legend[i]}" in system
    assert "true means: the speaker is courteous" in system
    for spec in HEADS:
        assert spec.description in system
        assert f'"{spec.name}"' in system
    assert "unsure" in system and "confidence" in system and "strict JSON" in system
    assert "200 characters" in system
    for it in ITEMS:
        assert f"<id>{it.id}</id>" in user
    assert "Could you help me?" in user and "Close the door." in user
    assert '"It rains."' in user  # no display: the state is shown as compact JSON


def test_system_prompt_is_stable_across_batches():
    s1, u1 = build_prompt(ITEMS[:1], HEADS)
    s2, u2 = build_prompt(ITEMS[1:], HEADS)
    assert s1 == s2  # cacheable: the items live in the user part only
    assert u1 != u2
    assert "s1" not in s1 and "s2" not in s2


def test_parse_fenced_json_with_prose():
    body = answer([{"id": "s1", "labels": {"kind": {"label": "question", "confidence": 0.9, "rationale": "ends ?"},
                                          "polite": {"label": True, "confidence": 0.8},
                                          "quality": {"label": 2, "confidence": 0.7}}}])
    text = f"Sure, here are the labels:\n```json\n{body}\n```\nLet me know if you need more."
    problems: list[str] = []
    got = parse_response(text, ITEMS, HEADS, problems)
    by_head = {a.head: a for a in got}
    assert by_head["kind"] == JudgeAnswer("s1", "kind", "question", 0.9, "ends ?")
    assert by_head["polite"].label == "true"
    assert by_head["quality"].label == "2"
    # s2 and s3 were not answered: reported, not invented
    assert any("pairs got no valid label" in p for p in problems)


def test_parse_prose_around_unfenced_json_and_trailing_braces():
    body = answer([{"id": "s2", "labels": {"kind": {"label": "Command", "confidence": "85"}}}])
    text = f"Here you go {body} -- note {{this is not json}}"
    got = parse_response(text, ITEMS, [KIND])
    assert len(got) == 1
    assert got[0].label == "command"       # case-insensitive class match, canonical spelling stored
    assert got[0].confidence == 0.85       # a percentage is read as a probability


def test_parse_skips_invalid_labels_unknown_ids_and_declines():
    body = answer([
        {"id": "s1", "labels": {"kind": {"label": "exclamation"}, "polite": {"label": "unsure"},
                                "nonsense": {"label": "x"}}},
        {"id": "nope", "labels": {"kind": {"label": "question"}}},
        {"id": "s2", "labels": {"kind": {"label": "statement", "confidence": 2.5, "rationale": "x" * 500},
                                "polite": "no", "quality": {"label": 7}}},
        {"id": "s2", "labels": {"kind": {"label": "question"}}},
    ])
    problems: list[str] = []
    got = parse_response(body, ITEMS, HEADS, problems)
    assert [(a.item_id, a.head, a.label) for a in got] == [("s2", "kind", "statement"), ("s2", "polite", "false")]
    kind = got[0]
    assert kind.confidence is None              # out of range
    assert len(kind.rationale) == 200
    joined = "\n".join(problems)
    assert "'exclamation' is not an allowed label" in joined
    assert "declined" in joined
    assert "unknown question 'nonsense'" in joined
    assert "unknown item id 'nope'" in joined
    assert "7 is not an allowed label" in joined
    assert "answered twice" in joined


def test_parse_score_with_named_levels_and_bare_values():
    body = answer([{"id": "s3", "labels": {"tier": 1}}, {"id": "s1", "labels": {"tier": "HIGH"}}])
    got = parse_response(body, ITEMS, [TIER])
    assert {(a.item_id, a.label) for a in got} == {("s3", "mid"), ("s1", "high")}


def test_parse_heads_directly_on_item_and_list_form():
    text = json.dumps([{"id": "s1", "kind": {"label": "question"}}])
    got = parse_response(text, ITEMS, [KIND])
    assert [(a.item_id, a.label) for a in got] == [("s1", "question")]


def test_no_json_is_a_problem_not_a_crash():
    problems: list[str] = []
    assert parse_response("I cannot label these.", ITEMS, HEADS, problems) == []
    assert problems and "no JSON answer" in problems[0]
    assert extract_json("{broken") is None


def test_rating_scale_uses_level_names_not_indices():
    rating = HeadSpec("rating", "score", ("1", "2", "3", "4", "5"), view="text",
                      levels=("awful", "poor", "fine", "good", "superb"))
    system, _ = build_prompt(ITEMS, [rating])
    assert '"1" = awful' in system and '"5" = superb' in system and "level name" in system
    got = parse_response(answer([{"id": "s1", "labels": {"rating": 4}}, {"id": "s2", "labels": {"rating": "5"}}]),
                         ITEMS, [rating])
    assert {(a.item_id, a.label) for a in got} == {("s1", "4"), ("s2", "5")}
