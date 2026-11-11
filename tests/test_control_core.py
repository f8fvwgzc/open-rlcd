"""Adversarial unit tests for the M10 control layer, pure modules part 1: digests (core/digest.py), question inputs
(core/inputs.py), rollout and the automation kill switch (core/rollout.py), receipts: the chained decision log,
pruning and staleness (core/receipts.py).

No service, no training: every test builds its inputs by hand. Tests marked `xfail(strict=True)` document a bug
found while writing them: each asserts the correct behaviour and names the bug in its reason, so it fails loudly
(XPASS) once the bug is fixed and the marker must go. Part 2 (batteries, redaction, uncertainty, routes, review,
policy overrides, the contract) is tests/test_control_core_2.py.
"""
from __future__ import annotations

import json
import os
import random
import re
import subprocess
import sys
import threading
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

from open_rlcd.core import digest as D
from open_rlcd.core import receipts as R
from open_rlcd.core import rollout as RO
from open_rlcd.core.contract import ContractError, Question, questions_digest
from open_rlcd.core.feedback import DecisionLog
from open_rlcd.core.inputs import MAX_INPUTS, MAX_PATH_CHARS, MAX_PATH_DEPTH, InputSpec, parse_path, resolve

HEX = re.compile(r"^sha256:[0-9a-f]{64}$")
REPO = Path(__file__).resolve().parents[1]


# ============================================================================= digest

def test_digest_ignores_key_order_at_every_level():
    assert HEX.match(D.digest({"a": 1}))
    assert D.digest({"b": 1, "a": [1, 2]}) == D.digest({"a": [1, 2], "b": 1})
    assert D.digest({"x": {"z": [{"q": 1, "p": 2}], "y": 2}}) == D.digest({"x": {"y": 2, "z": [{"p": 2, "q": 1}]}})


def test_digest_containers_arrays_are_ordered_sets_are_not():
    assert D.digest([1, 2]) != D.digest([2, 1])
    assert D.digest((1, 2)) == D.digest([1, 2])                                   # a tuple is an array
    assert D.digest({3, 1, 2}) == D.digest(frozenset({2, 3, 1})) == D.digest([1, 2, 3])
    assert D.digest({1: "a"}) == D.digest({"1": "a"})                             # JSON keys are strings


def test_digest_tells_json_types_apart():
    values = [None, True, False, 0, 1, "1", "true", "null", "", [], {}, [None], {"": None}]
    assert len({D.digest(v) for v in values}) == len(values)


def test_digest_normalises_numpy_values():
    assert D.digest(np.array([1, 2, 3])) == D.digest([1, 2, 3])
    assert D.digest(np.zeros((2, 2))) == D.digest([[0.0, 0.0], [0.0, 0.0]])
    assert (D.digest({"p": np.float64(0.25), "n": np.int64(3), "b": np.bool_(True)})
            == D.digest({"p": 0.25, "n": 3, "b": True}))


def test_canonical_bytes_are_compact_sorted_and_utf8():
    expected = '{"a":[1,{"c":2.5,"d":null}],"b":"é漢"}'.encode()
    assert D.canonical({"b": "é漢", "a": [1, {"d": None, "c": 2.5}]}) == expected


def test_non_finite_numbers_have_a_stable_digest_that_survives_a_json_round_trip():
    state = {"x": float("nan"), "y": float("inf"), "z": float("-inf")}
    assert D.canonical(state) == b'{"x":NaN,"y":Infinity,"z":-Infinity}'
    assert D.digest(state) == D.digest(dict(state)) == D.digest(json.loads(json.dumps(D.plain(state))))


def test_plain_round_trip_keeps_the_digest():
    obj = {"t": (1, "a"), "s": {"b", "a"}, "np": np.arange(3), 7: Path("p/q"), "nested": [{"k": np.float32(0.5)}],
           "when": datetime(2026, 11, 9, 12, 0, tzinfo=UTC)}
    back = json.loads(json.dumps(D.plain(obj)))
    assert back["7"] == "p/q" and back["s"] == ["a", "b"] and back["when"] == "2026-11-09 12:00:00+00:00"
    assert D.digest(back) == D.digest(obj)


def _random_json(rnd: random.Random, depth: int = 0):
    r = rnd.random()
    if depth > 3 or r < 0.3:
        return rnd.choice([None, True, False, rnd.randint(-5, 5), rnd.uniform(-1e3, 1e3), f"s{rnd.randint(0, 99)}",
                           "é漢", "", float("nan")])
    if r < 0.65:
        return [_random_json(rnd, depth + 1) for _ in range(rnd.randint(0, 4))]
    return {f"k{rnd.randint(0, 20)}": _random_json(rnd, depth + 1) for _ in range(rnd.randint(0, 4))}


def _shuffled(obj, rnd: random.Random):
    if isinstance(obj, dict):
        items = list(obj.items())
        rnd.shuffle(items)
        return {k: _shuffled(v, rnd) for k, v in items}
    if isinstance(obj, list):
        return [_shuffled(v, rnd) for v in obj]
    return obj


def test_digest_is_invariant_to_key_order_and_json_round_trips_on_random_values():
    rnd = random.Random(7)
    for _ in range(300):
        obj = _random_json(rnd)
        d = D.digest(obj)
        assert D.digest(_shuffled(obj, rnd)) == d == D.digest(json.loads(json.dumps(obj)))
        assert D.digest([obj]) != d                                      # wrapping is a different value


def test_short():
    d = D.digest({"a": 1})
    assert D.short(d) == d[: len(D.PREFIX) + 12] and D.short(d, 4) == d[: len(D.PREFIX) + 4]
    assert D.short(None) is None and D.short("") == ""
    assert D.short("abcdef0123456789", 6) == "abcdef"                             # not a sha256: digest


def test_digest_of_a_state_with_a_lone_surrogate_does_not_fail():
    state = json.loads('{"text": "half an emoji \\ud83d"}')
    assert HEX.match(D.digest(state))


def test_digest_of_a_deeply_nested_state_does_not_fail():
    depth = 2000
    try:
        nested = json.loads("[" * depth + "]" * depth)
    except RecursionError:
        pytest.skip("this Python's JSON parser refuses the depth itself")
    assert HEX.match(D.digest({"query": "q", "extra": nested}))


# ============================================================================= inputs

STATE = {"request": {"text": "where is the invoice?", "lang": "en"},
         "retrieval": {"passages": [{"id": "p1", "text": "The invoice is in folder A."}, {"id": "p2", "text": ""}]},
         "llm_proposal": {"text": None}, "0": "key named zero", "név": {"ü": 1}, "a.b": "dotted key",
         "a": {"b": "nested b"}, "codes": {"007": "bond"}}


def test_parse_path_segments_and_the_whole_state():
    assert parse_path("a.b.0") == ("a", "b", 0)
    assert parse_path("") == () and parse_path(".") == () and parse_path("   ") == ()
    assert parse_path("  rag.passages ") == ("rag", "passages")
    assert parse_path("items.12") == ("items", 12)
    # only ASCII digit runs index arrays: other digits, signs and exponents stay keys (no accidental indexing)
    assert parse_path("a.٣") == ("a", "٣") and parse_path("a.²") == ("a", "²")
    assert parse_path("items.-1") == ("items", "-1") and parse_path("items.1e3") == ("items", "1e3")


@pytest.mark.parametrize("bad", [None, 3, 1.5, ["a"], {"a": 1}])
def test_parse_path_refuses_a_non_string(bad):
    with pytest.raises(ContractError, match="must be a string") as e:
        parse_path(bad, question="q")
    assert e.value.status == 422 and e.value.kind == "validation"
    assert e.value.details == {"question": "q", "field": "path"}


@pytest.mark.parametrize("bad", ["a..b", ".a", "a.", ".."])
def test_parse_path_refuses_an_empty_segment(bad):
    with pytest.raises(ContractError, match="empty segment"):
        parse_path(bad, question="q")


def test_parse_path_limits():
    assert len(parse_path(".".join(["k"] * MAX_PATH_DEPTH))) == MAX_PATH_DEPTH
    with pytest.raises(ContractError, match="deeper than"):
        parse_path(".".join(["k"] * (MAX_PATH_DEPTH + 1)))
    assert parse_path("k" * MAX_PATH_CHARS) == ("k" * MAX_PATH_CHARS,)
    with pytest.raises(ContractError, match="longer than"):
        parse_path("k" * (MAX_PATH_CHARS + 1))
    with pytest.raises(ContractError, match="longer than"):                     # checked before splitting
        parse_path(".".join(["k"] * 400))


def test_resolve_walks_objects_and_arrays_and_passes_values_through():
    assert resolve(STATE, parse_path("retrieval.passages.0.id")) == "p1"
    assert resolve(STATE, parse_path("retrieval.passages.1.text")) == ""          # present but empty: the head judges
    assert resolve(STATE, parse_path("llm_proposal.text")) is None                # present null: passed through
    assert resolve(STATE, parse_path("0")) == "key named zero"                    # an index segment on an object
    assert resolve(STATE, parse_path("név.ü")) == 1
    assert resolve(STATE, parse_path("a.b")) == "nested b"                        # never the dotted key "a.b"
    assert resolve(STATE, ()) is STATE
    assert resolve(STATE, parse_path("retrieval.passages")) is STATE["retrieval"]["passages"]
    assert resolve([[10, 11], [12]], parse_path("0.1")) == 11                       # a top-level array


@pytest.mark.parametrize("path,why", [
    ("request.missing", "request.missing is not in the state"),
    ("retrieval.passages.2", "out of range (the array has 2 items)"),
    ("retrieval.passages.99999999999999999999999", "out of range"),
    ("retrieval.passages.-1", "cannot be read"),          # no Python negative indexing
    ("retrieval.passages.first", "cannot be read"),
    ("request.text.0", "cannot be read"),                 # strings are not indexed
    ("llm_proposal.text.x", "cannot be read"),            # nothing below a null
    ("dotted.key", "is not in the state"),
])
def test_resolve_fails_closed_with_missing_evidence(path, why):
    with pytest.raises(ContractError, match=re.escape(why)) as e:
        resolve(STATE, parse_path(path), question="q1", path=path)
    assert e.value.kind == "missing_evidence" and e.value.status == 422
    assert e.value.details == {"question": "q1", "path": path}


def test_a_digit_key_with_leading_zeros_resolves_on_an_object():
    assert resolve(STATE, parse_path("codes.007")) == "bond"


def test_input_spec_without_path_or_inputs_is_the_request_state():
    spec = InputSpec()
    assert not spec.active and spec.to_dict() == {} and spec.project(STATE) is STATE


def test_input_spec_projects_one_shared_state():
    assert InputSpec("retrieval", question="q").project(STATE) is STATE["retrieval"]
    spec = InputSpec(inputs={"query": "request.text", "evidence": "retrieval.passages"}, question="q")
    out = spec.project(STATE)
    assert out == {"query": STATE["request"]["text"], "evidence": STATE["retrieval"]["passages"]}
    assert out["evidence"] is STATE["retrieval"]["passages"]                       # values are not copied or changed
    both = InputSpec(" retrieval ", {"first": "passages.0.text"}, question="q")
    assert both.project(STATE) == {"first": "The invoice is in folder A."}
    assert both.to_dict() == {"path": "retrieval", "inputs": {"first": "passages.0.text"}}


def test_input_spec_to_dict_is_normalised_and_sorted():
    # "01" is not a canonical array index: it stays an object key (a key with leading zeros must stay addressable)
    spec = InputSpec("  a.01 ", {"z": " x.y ", "a": "b", "M-1": "c.0"})
    assert spec.to_dict() == {"path": "a.01", "inputs": {"M-1": "c.0", "a": "b", "z": "x.y"}}
    assert list(spec.to_dict()["inputs"]) == ["M-1", "a", "z"]


def test_input_spec_names_the_question_and_the_path_that_is_missing():
    spec = InputSpec(inputs={"query": "request.text", "answer": "llm.answer"}, question="grounded")
    with pytest.raises(ContractError) as e:
        spec.project(STATE)
    assert e.value.kind == "missing_evidence" and e.value.details == {"question": "grounded", "path": "llm.answer"}
    with pytest.raises(ContractError) as e:
        InputSpec("nowhere", {"q": "request.text"}, question="q").project(STATE)
    assert e.value.details["path"] == "nowhere"                                    # the root is checked first
    for state in ("just a string", None, 42, [1, 2]):
        with pytest.raises(ContractError) as e:
            InputSpec(inputs={"q": "text"}, question="q").project(state)
        assert e.value.kind == "missing_evidence"


@pytest.mark.parametrize("inputs,match", [
    ({}, "non-empty object"), ([], "non-empty object"), ("request.text", "non-empty object"),
    ([("q", "a")], "non-empty object"),
    ({"1abc": "x"}, "identifier"), ({"has space": "x"}, "identifier"), ({"ü": "x"}, "identifier"),
    ({"a" * 65: "x"}, "identifier"), ({"": "x"}, "identifier"), ({3: "x"}, "identifier"),
    ({"q": 3}, "must be a string"), ({"q": None}, "must be a string"), ({"q": "a..b"}, "empty segment"),
    ({f"k{i}": "x" for i in range(MAX_INPUTS + 1)}, f"at most {MAX_INPUTS}"),
])
def test_input_spec_validation(inputs, match):
    with pytest.raises(ContractError, match=match) as e:
        InputSpec(inputs=inputs, question="q")
    assert e.value.status == 422 and e.value.details["question"] == "q"


def test_an_input_name_with_a_trailing_newline_is_not_an_identifier():
    with pytest.raises(ContractError, match="identifier"):
        InputSpec(inputs={"query\n": "request.text"}, question="q")


def test_input_spec_accepts_the_limits_exactly():
    spec = InputSpec(inputs={f"k{i}": f"s.{i}" for i in range(MAX_INPUTS)}, question="q")
    out = spec.project({"s": list(range(MAX_INPUTS))})
    assert out == {f"k{i}": i for i in range(MAX_INPUTS)}
    assert InputSpec(inputs={"a" * 64: "x"}).project({"x": 1}) == {"a" * 64: 1}


def test_question_paths_are_part_of_its_contract_and_digest():
    base = {"type": "noul", "head": "answerable", "instructions": "ignored"}
    io = {"query": "request.text", "evidence": "retrieval.passages"}
    q0 = Question("ans", base)
    q1 = Question("ans", {**base, "inputs": io})
    q2 = Question("ans", {**base, "path": "rag", "inputs": io})
    assert q1.contract() == {"type": "noul", "head": "answerable",
                             "inputs": {"evidence": "retrieval.passages", "query": "request.text"}}
    assert len({questions_digest([q]) for q in (q0, q1, q2)}) == 3
    assert q2.inputs.project({"rag": STATE}) == {"query": STATE["request"]["text"],
                                                 "evidence": STATE["retrieval"]["passages"]}
    with pytest.raises(ContractError, match="empty segment"):
        Question("ans", {**base, "path": "rag..x"})


# ============================================================================= rollout

GRID_STAKES = ("low", "medium", "high", None, "custom")
GRID_HEADS = ("route", "needs_graph", None)


@pytest.fixture
def no_env_switch(monkeypatch):
    monkeypatch.delenv(RO.ENV_SWITCH, raising=False)


def test_full_rollout_never_caps():
    ro = RO.Rollout.from_config(None)
    assert ro == RO.Rollout() == RO.Rollout.from_config({}) and ro.to_dict() == {"mode": "full"}
    for s in GRID_STAKES:
        for h in GRID_HEADS:
            assert ro.cap(stakes=s, head=h) == (2, None)


@pytest.mark.parametrize("mode", ["shadow", "assist"])
def test_shadow_and_assist_cap_every_decision_at_confirm(mode):
    ro = RO.Rollout.from_config({"mode": mode})
    assert ro.to_dict() == {"mode": mode}
    for s in GRID_STAKES:
        for h in GRID_HEADS:
            cap, why = ro.cap(stakes=s, head=h)
            assert cap == 1 and why.startswith(f"rollout mode {mode}:")


def test_limited_automates_only_the_configured_slice():
    ro = RO.Rollout.from_config({"mode": "limited"})            # default slice: low stakes, any head
    assert ro.limited_stakes == ("low",) and ro.limited_heads is None
    assert ro.cap(stakes="low", head="anything") == (2, None) and ro.cap(stakes="low", head=None) == (2, None)
    for s in ("medium", "high", None, "LOW"):
        cap, why = ro.cap(stakes=s, head="route")
        assert cap == 1 and "only stakes ['low'] are automated" in why and repr(s) in why
    ro = RO.Rollout.from_config({"mode": "limited", "limited": {"stakes": ["low", "medium"], "heads": ["route"]}})
    assert ro.cap(stakes="medium", head="route") == (2, None)
    assert ro.cap(stakes="high", head="route")[0] == 1
    cap, why = ro.cap(stakes="low", head="needs_graph")
    assert cap == 1 and "only heads ['route'] are automated" in why
    assert ro.cap(stakes="low", head=None)[0] == 1               # a decision of no named head is outside the slice
    assert ro.to_dict() == {"mode": "limited", "limited": {"stakes": ["low", "medium"], "heads": ["route"]}}


def test_limited_accepts_single_names_and_empty_or_null_stakes_automate_nothing():
    ro = RO.Rollout.from_config({"mode": "limited", "limited": {"stakes": "high", "heads": "route"}})
    assert ro.limited_stakes == ("high",) and ro.limited_heads == ("route",)
    assert ro.cap(stakes="high", head="route") == (2, None)
    for lim in ({"stakes": []}, {"stakes": None}):
        ro = RO.Rollout.from_config({"mode": "limited", "limited": lim})
        assert all(ro.cap(stakes=s, head="route")[0] == 1 for s in GRID_STAKES)


def test_per_head_modes_override_the_global_mode():
    ro = RO.Rollout.from_config({"mode": "full", "heads": {"needs_graph": "shadow", "route": "limited"}})
    assert ro.mode_for("needs_graph") == "shadow" and ro.mode_for("other") == "full" and ro.mode_for(None) == "full"
    cap, why = ro.cap(stakes="low", head="needs_graph")
    assert cap == 1 and why.startswith("head 'needs_graph' is in rollout mode shadow:")
    assert ro.cap(stakes="high", head="other") == (2, None)
    assert ro.cap(stakes="low", head="route") == (2, None)        # limited for this head only: low stakes act
    assert ro.cap(stakes="medium", head="route")[0] == 1
    assert ro.to_dict() == {"mode": "full", "limited": {"stakes": ["low"], "heads": None},
                            "heads": {"needs_graph": "shadow", "route": "limited"}}
    released = RO.Rollout.from_config({"mode": "shadow", "heads": {"route": "full"}})
    assert released.cap(stakes="high", head="route") == (2, None)
    assert released.cap(stakes="low", head="x")[0] == 1 and released.cap(stakes="low", head=None)[0] == 1


def test_rollout_caps_are_never_below_confirm_and_carry_a_reason_exactly_when_they_cap():
    configs = [{"mode": m} for m in RO.MODES] + [
        {"mode": "limited", "limited": {"stakes": [], "heads": ["nobody"]}},
        {"mode": "shadow", "heads": {"route": "assist", "needs_graph": "limited"}},
        {"mode": "limited", "limited": {"stakes": ["low", "custom"]}, "heads": {"route": "full"}}]
    for cfg in configs:
        ro = RO.Rollout.from_config(cfg)
        for s in GRID_STAKES:
            for h in GRID_HEADS:
                cap, why = ro.cap(stakes=s, head=h)
                assert cap in (1, 2) and (why is None) == (cap == 2), (cfg, s, h)


@pytest.mark.parametrize("cfg,match", [
    ({"mode": "off"}, "rollout.mode"), ({"mode": "Full"}, "rollout.mode"), ({"mode": "on"}, "rollout.mode"),
    ({"heads": {"route": "never"}}, "rollout.heads.route"), ({"heads": {"route": None}}, "rollout.heads.route"),
    ({"heads": ["route"]}, "maps a head name"), ({"heads": "route"}, "maps a head name"),
])
def test_rollout_validation(cfg, match):
    with pytest.raises(ValueError, match=match):
        RO.Rollout.from_config(cfg)
    with pytest.raises(ValueError):
        RO.Rollout(mode="automatic")


def test_a_boolean_rollout_mode_is_refused_not_read_as_full():
    import yaml

    cfg = yaml.safe_load("mode: off")
    assert cfg == {"mode": False}
    with pytest.raises(ValueError):
        RO.Rollout.from_config(cfg)


def test_an_empty_limited_heads_list_automates_no_head():
    ro = RO.Rollout.from_config({"mode": "limited", "limited": {"stakes": ["low"], "heads": []}})
    assert ro.cap(stakes="low", head="route")[0] == 1


def test_a_limited_section_that_is_not_a_mapping_is_refused():
    try:
        ro = RO.Rollout.from_config({"mode": "limited", "limited": ["high"]})
    except ValueError:
        return
    assert ro.cap(stakes="low", head="route")[0] == 1


@pytest.mark.parametrize("limited", [{"stakes": 5}, {"heads": 5}])
def test_wrong_typed_rollout_values_are_configuration_errors(limited):
    with pytest.raises(ValueError):
        RO.Rollout.from_config({"mode": "limited", "limited": limited})


def test_effective_cap_combines_the_rollout_and_the_switch(tmp_path, no_env_switch):
    sw = RO.AutomationSwitch(tmp_path)
    full, shadow = RO.Rollout(), RO.Rollout(mode="shadow")
    assert RO.effective_cap(full, None, stakes="high", head="h") == (2, [])
    assert RO.effective_cap(full, sw, stakes="high", head="h") == (2, [])
    cap, why = RO.effective_cap(shadow, sw, stakes="low", head="h")
    assert cap == 1 and len(why) == 1 and "shadow" in why[0]
    sw.set(False, reason="incident 42", by="ops", at="2026-11-09T12:00:00Z")
    cap, why = RO.effective_cap(full, sw, stakes="low", head="h")
    assert cap == 1 and len(why) == 1 and "incident 42" in why[0]
    cap, why = RO.effective_cap(shadow, sw, stakes="low", head="h")
    assert cap == 1 and len(why) == 2


def test_effective_cap_is_the_minimum_of_the_rollout_and_the_switch(tmp_path, no_env_switch):
    sw = RO.AutomationSwitch(tmp_path)
    configs = [{"mode": m} for m in RO.MODES] + [{"mode": "full", "heads": {"h": "assist"}}]
    for enabled in (True, False):
        sw.set(enabled, reason="r", by=None, at="t")
        for cfg in configs:
            ro = RO.Rollout.from_config(cfg)
            for s in GRID_STAKES:
                for h in GRID_HEADS:
                    cap, why = RO.effective_cap(ro, sw, stakes=s, head=h)
                    assert cap == min(ro.cap(stakes=s, head=h)[0], sw.cap()[0]) and (cap == 2) == (why == [])


# ============================================================================= automation switch

def test_switch_defaults_on_and_round_trips_through_its_file(tmp_path, no_env_switch):
    sw = RO.AutomationSwitch(tmp_path)
    st = sw.state()
    assert st.enabled and not st.forced_by_env and st.reason is None and sw.cap() == (2, None)
    assert sw.history() == [] and not (tmp_path / "automation.json").exists()
    out = sw.set(False, reason="incident 42 — rollback", by="ops", at="2026-11-09T12:00:00Z")
    assert out.to_dict() == {"enabled": False, "reason": "incident 42 — rollback", "by": "ops",
                             "at": "2026-11-09T12:00:00Z", "forced_by_env": False}
    assert json.loads((tmp_path / "automation.json").read_text(encoding="utf-8")) == {
        "enabled": False, "reason": "incident 42 — rollback", "by": "ops", "at": "2026-11-09T12:00:00Z"}
    assert not (tmp_path / "automation.json.tmp").exists()
    other = RO.AutomationSwitch(tmp_path)                    # another process on the same home reads the file
    assert other.state().enabled is False and other.state().reason == "incident 42 — rollback"
    cap, why = other.cap()
    assert cap == 1 and "incident 42" in why and "capped at confirm" in why
    sw.set(True, reason="fixed", by="ops", at="2026-11-09T13:00:00Z")
    assert other.state().enabled is True and other.cap() == (2, None)   # re-read when the file changes
    assert [h["enabled"] for h in other.history()] == [False, True]
    assert [h["reason"] for h in other.history(limit=1)] == ["fixed"]


@pytest.mark.parametrize("content", [b"{not json", b"[]", b'"off"', b"", b"null", b"42", b"\xff\xfe\x00garbage"])
def test_an_unreadable_switch_file_fails_safe(tmp_path, no_env_switch, content):
    (tmp_path / "automation.json").write_bytes(content)
    sw = RO.AutomationSwitch(tmp_path)
    st = sw.state()
    assert st.enabled is False and "unreadable" in st.reason
    assert sw.cap()[0] == 1
    sw.set(True, reason="rewritten", by="ops", at="2026-11-09T12:00:00Z")      # the documented way out
    assert sw.state().enabled is True


def test_a_switch_path_that_cannot_be_read_as_a_file_fails_safe(tmp_path, no_env_switch):
    (tmp_path / "automation.json").mkdir()
    st = RO.AutomationSwitch(tmp_path).state()
    assert st.enabled is False and "unreadable" in st.reason


def test_switch_history_skips_unparsable_lines(tmp_path, no_env_switch):
    sw = RO.AutomationSwitch(tmp_path)
    sw.set(False, reason="a", by=None, at="t1")
    with open(tmp_path / "automation.jsonl", "a", encoding="utf-8") as f:
        f.write("this is not json\n")
    sw.set(True, reason="b", by=None, at="t2")
    assert [h["reason"] for h in sw.history()] == ["a", "b"]


def test_concurrent_switch_changes_keep_a_readable_file_and_a_complete_audit(tmp_path, no_env_switch):
    sw = RO.AutomationSwitch(tmp_path)

    def flip(k: int) -> None:
        for j in range(25):
            sw.set((k + j) % 2 == 0, reason=f"{k}-{j}", by=f"t{k}", at=f"t{j}")

    threads = [threading.Thread(target=flip, args=(k,)) for k in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(sw.history(limit=1000)) == 150
    st = RO.AutomationSwitch(tmp_path).state()
    assert "unreadable" not in (st.reason or "") and st.reason == sw.history(limit=1)[0]["reason"]


@pytest.mark.parametrize("value", ["off", "OFF", " no ", "0", "false", "False", "disabled"])
def test_env_forces_automation_off_and_locks_it(tmp_path, monkeypatch, value):
    monkeypatch.delenv(RO.ENV_SWITCH, raising=False)
    sw = RO.AutomationSwitch(tmp_path)
    sw.set(True, reason="on", by="ops", at="t0")
    monkeypatch.setenv(RO.ENV_SWITCH, value)
    st = sw.state()
    assert st.enabled is False and st.forced_by_env and st.by == "environment" and RO.ENV_SWITCH in st.reason
    assert st.at == "t0" and sw.cap()[0] == 1
    with pytest.raises(ContractError) as e:
        sw.set(True, reason="please", by="api", at="t1")
    assert e.value.kind == "automation_locked" and e.value.status == 409
    assert e.value.body()["kind"] == "automation_locked"
    assert [h["reason"] for h in sw.history()] == ["on"]                   # the refused change is not recorded
    off = sw.set(False, reason="belt and braces", by="ops", at="t2")       # switching off stays possible
    assert off.enabled is False and off.forced_by_env
    assert json.loads((tmp_path / "automation.json").read_text())["enabled"] is False
    monkeypatch.delenv(RO.ENV_SWITCH)
    assert sw.state().enabled is False and sw.state().reason == "belt and braces"   # the file decides again


@pytest.mark.parametrize("value", ["", "on", "1", "true", "enabled", "offline"])
def test_env_values_that_do_not_force_automation_off(tmp_path, monkeypatch, value):
    monkeypatch.setenv(RO.ENV_SWITCH, value)
    sw = RO.AutomationSwitch(tmp_path)
    assert sw.state().enabled is True and not sw.state().forced_by_env
    assert sw.set(True, reason=None, by=None, at="t").enabled is True


@pytest.mark.parametrize("value", ["false", "off", "no", "0"])
def test_a_non_boolean_enabled_value_fails_safe(tmp_path, no_env_switch, value):
    (tmp_path / "automation.json").write_text(json.dumps({"enabled": value, "reason": "maintenance"}))
    assert RO.AutomationSwitch(tmp_path).state().enabled is False


def test_switch_history_limit_zero_returns_nothing(tmp_path, no_env_switch):
    sw = RO.AutomationSwitch(tmp_path)
    for i in range(3):
        sw.set(i % 2 == 1, reason=str(i), by=None, at=f"t{i}")
    assert sw.history(limit=0) == []


def test_switch_rereads_a_replaced_file_with_the_same_size_and_mtime(tmp_path, no_env_switch):
    path = tmp_path / "automation.json"
    off = json.dumps({"enabled": False, "reason": "abcd"})
    on = json.dumps({"enabled": True, "reason": "abcde"})
    assert len(off) == len(on)
    path.write_text(off)
    st = path.stat()
    sw = RO.AutomationSwitch(tmp_path)
    assert sw.state().enabled is False
    tmp = tmp_path / "other-process.tmp"
    tmp.write_text(on)
    tmp.replace(path)                                         # how `rlcd automation on` writes it
    os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns))      # a clock that has not ticked since the last write
    assert sw.state().enabled is True


# ============================================================================= receipts: building

MODEL = "rlcd-0.1.3"
PACK = {"name": "toy", "version": "0.1.0"}
NOW = datetime(2026, 11, 9, 12, 0, 10, tzinfo=UTC)


def receipt_row(i: int, *, state=None, created_at="2026-11-09T12:00:00Z", action="act", kind="decide",
                requested_model=None, battery=None, policy="sha256:policy", head_tier="act", log_state=True) -> dict:
    """A decision record shaped like the service's (heads, kind-specific fields, a receipt from receipts.make)."""
    state = {"q": f"question {i}", "evidence": ["e1"]} if state is None else state
    head = {"class": "true", "p": 0.97, "tier": head_tier, "stakes": "low",
            "probabilities": {"false": 0.03, "true": 0.97}, "uncertainty": []}
    if log_state:
        head["state"] = state
    row = {"decision_id": f"d{i:04d}", "created_at": created_at, "kind": kind, "model": MODEL,
           "heads": {"answerable": head}}
    if action is not None:
        row["action"], row["stakes"] = action, "low"
    row["receipt"] = R.make(requested_model=requested_model, model=MODEL, pack=PACK, feature_version="toy-v1",
                            state=state, questions={"digest": "sha256:questions", "battery": battery},
                            policy_digest=policy, rollout={"mode": "full", "automation": {"enabled": True},
                                                           "capped": False},
                            latency_ms=3.5, warnings=["w"])
    return row


def _make(**kw):
    base = dict(requested_model=None, model=MODEL, pack=PACK, feature_version="v1", state={},
                questions={"digest": "sha256:q", "battery": None}, policy_digest="sha256:p", rollout=None,
                latency_ms=None, warnings=[])
    return R.make(**{**base, **kw})


def test_make_binds_the_state_and_the_contract():
    state = {"query": "q", "evidence": [{"id": "e1", "text": "t"}]}
    r = _make(requested_model="rlcd-latest", state=state, latency_ms=1.25, warnings=None,
              audit={"baseline": "large_llm", "tags": None})
    assert r["format"] == R.FORMAT == "open-rlcd-receipt/1"
    assert r["state_digest"] == D.digest(state) == D.digest(json.loads(json.dumps(state)))
    assert r["contract_digest"] == R.contract_digest(model=MODEL, pack=PACK, feature_version="v1",
                                                     questions="sha256:q", policy="sha256:p")
    assert r["warnings"] == [] and r["audit"] == {"baseline": "large_llm"}          # None audit fields are dropped
    assert "prev" not in r and "digest" not in r                                    # set when appended
    assert r["requested_model"] == "rlcd-latest" and r["latency_ms"] == 1.25 and r["pack"] == PACK


def test_contract_digest_changes_with_every_part_of_the_contract_but_not_with_the_state():
    base = dict(model=MODEL, pack=PACK, feature_version="v1", questions="sha256:q", policy="sha256:p")
    d0 = R.contract_digest(**base)
    for key, value in (("model", "rlcd-0.1.4"), ("pack", {"name": "toy", "version": "0.1.1"}),
                       ("feature_version", "v2"), ("questions", "sha256:q2"), ("policy", "sha256:p2"),
                       ("policy", None), ("questions", None)):
        assert R.contract_digest(**{**base, key: value}) != d0, key
    a, b = _make(state={"x": 1}), _make(state={"x": 2})
    assert a["contract_digest"] == b["contract_digest"] and a["state_digest"] != b["state_digest"]
    assert _make(questions={"digest": "sha256:q", "battery": {"name": "b"}})["contract_digest"] == a["contract_digest"]


def test_make_bounds_warnings_and_audit():
    assert _make(warnings=[f"w{i}" for i in range(50)])["warnings"] == [f"w{i}" for i in range(R.MAX_WARNINGS)]
    assert "audit" not in _make(audit={"baseline": None}) and "audit" not in _make(audit={})
    head = len(json.dumps({"a": ""}))                         # the JSON around the value
    assert _make(audit={"a": "x" * (R.MAX_AUDIT_BYTES - head)})["audit"]
    with pytest.raises(ContractError) as e:
        _make(audit={"a": "x" * (R.MAX_AUDIT_BYTES - head + 1)})
    assert e.value.status == 422 and e.value.details == {"field": "audit", "limit": R.MAX_AUDIT_BYTES}
    with pytest.raises(ContractError):                      # bytes, not characters
        _make(audit={"a": "é" * (R.MAX_AUDIT_BYTES // 2)})


def test_summary_is_the_compact_view(tmp_path):
    rec = R.ChainedLog(tmp_path / "decisions.jsonl").append(receipt_row(1))
    r = rec["receipt"]
    assert R.summary(rec) == {"id": "d0001", "digest": r["digest"], "prev": None, "state_digest": r["state_digest"],
                              "contract_digest": r["contract_digest"], "questions_digest": "sha256:questions"}
    assert R.summary({"decision_id": "legacy"}) == {"id": "legacy", "digest": None, "prev": None,
                                                    "state_digest": None, "contract_digest": None,
                                                    "questions_digest": None}


# ============================================================================= receipts: the chain

def _lines(path: Path) -> list[bytes]:
    return path.read_bytes().splitlines()


def _rewrite(path: Path, lines: list[bytes]) -> None:
    path.write_bytes(b"".join(x + b"\n" for x in lines))


def append_rows(path: Path, n: int, **kw) -> list[dict]:
    log = R.ChainedLog(path)
    return [log.append(receipt_row(i, **kw)) for i in range(n)]


def test_chained_log_links_every_receipt_to_the_one_before(tmp_path):
    p = tmp_path / "sub" / "decisions.jsonl"                    # the directory is created on first append
    recs = append_rows(p, 5)
    assert recs[0]["receipt"]["prev"] is None
    for a, b in zip(recs, recs[1:], strict=False):
        assert b["receipt"]["prev"] == a["receipt"]["digest"]
    written = [json.loads(x) for x in _lines(p)]
    assert written == recs                                       # the record returned is the record written
    assert all(R.row_digest(r) == r["receipt"]["digest"] for r in written)
    assert R.verify_chain(p) == {"path": "decisions.jsonl", "lines": 5, "receipts": 5, "legacy": 0,
                                 "unreadable": [], "ok": True, "breaks": [], "head": recs[-1]["receipt"]["digest"]}


def test_a_restarted_writer_continues_the_chain_and_overwrites_caller_supplied_links(tmp_path):
    p = tmp_path / "decisions.jsonl"
    first = R.ChainedLog(p).append(receipt_row(1))
    row = receipt_row(2)
    row["receipt"]["prev"], row["receipt"]["digest"] = "sha256:forged", "sha256:forged"
    second = R.ChainedLog(p).append(row)                         # a new process: the tip is read from the file
    assert second["receipt"]["prev"] == first["receipt"]["digest"]
    assert second["receipt"]["digest"] == R.row_digest(second) != "sha256:forged"
    assert row["receipt"]["digest"] == "sha256:forged"           # the caller's dict is not modified
    assert R.verify_chain(p)["ok"]


def test_a_record_without_a_receipt_part_is_chained_too(tmp_path):
    p = tmp_path / "decisions.jsonl"
    rec = R.ChainedLog(p).append({"decision_id": "x", "kind": "decide", "heads": {}})
    assert set(rec["receipt"]) == {"prev", "digest"}
    assert R.verify_chain(p)["receipts"] == 1 and R.verify_chain(p)["ok"]


def test_append_normalises_values_json_cannot_carry_and_the_chain_still_verifies(tmp_path):
    p = tmp_path / "decisions.jsonl"
    row = receipt_row(1, state={"f": np.arange(3), "t": (1, 2), "nan": float("nan"), "inf": float("-inf"),
                                "text": "naïve café 漢字   line sep"})
    row["heads"]["answerable"]["probabilities"] = {"false": np.float32(0.25), "true": np.float64(0.75)}
    R.ChainedLog(p).append(row)
    back = json.loads(_lines(p)[0])
    assert back["heads"]["answerable"]["state"]["f"] == [0, 1, 2]
    assert back["heads"]["answerable"]["probabilities"] == {"false": 0.25, "true": 0.75}
    assert len(_lines(p)) == 1                                   # U+2028 does not split the JSON line
    assert R.verify_chain(p)["ok"] and R.verify_chain(p)["receipts"] == 1


def test_verify_detects_an_edited_line(tmp_path):
    p = tmp_path / "decisions.jsonl"
    append_rows(p, 4)
    lines = _lines(p)
    row = json.loads(lines[1])
    row["heads"]["answerable"]["tier"] = "confirm"               # a quiet edit: act -> confirm
    lines[1] = json.dumps(row).encode()
    _rewrite(p, lines)
    v = R.verify_chain(p)
    assert not v["ok"] and len(v["breaks"]) == 1
    assert v["breaks"][0]["line"] == 2 and v["breaks"][0]["decision_id"] == "d0001"
    assert "edited after it was written" in v["breaks"][0]["why"]


@pytest.mark.parametrize("edit", [
    lambda r: r["heads"]["answerable"]["state"].update(q="other question"),
    lambda r: r["receipt"].update(state_digest=D.digest({"q": "other"})),
    lambda r: r["receipt"].update(prev=None),
    lambda r: r.update(created_at="2026-11-09T12:00:01Z"),
    lambda r: r.pop("stakes"),
    lambda r: r.update(extra_field=1),
])
def test_verify_detects_any_edit_of_a_record(tmp_path, edit):
    p = tmp_path / "decisions.jsonl"
    append_rows(p, 3)
    lines = _lines(p)
    row = json.loads(lines[1])
    edit(row)
    lines[1] = json.dumps(row).encode()
    _rewrite(p, lines)
    v = R.verify_chain(p)
    assert not v["ok"] and 2 in [b["line"] for b in v["breaks"]]


def test_verify_detects_an_edit_whose_digest_was_recomputed_at_the_next_line(tmp_path):
    p = tmp_path / "decisions.jsonl"
    append_rows(p, 4)
    lines = _lines(p)
    row = json.loads(lines[1])
    row["action"] = "confirm"
    row["receipt"]["digest"] = R.row_digest(row)
    lines[1] = json.dumps(row).encode()
    _rewrite(p, lines)
    v = R.verify_chain(p)
    assert not v["ok"] and [b["line"] for b in v["breaks"]] == [3]
    assert "inserted, deleted or reordered" in v["breaks"][0]["why"]


def test_verify_detects_a_deleted_middle_line(tmp_path):
    p = tmp_path / "decisions.jsonl"
    append_rows(p, 5)
    lines = _lines(p)
    del lines[2]
    _rewrite(p, lines)
    v = R.verify_chain(p)
    assert not v["ok"] and [(b["line"], b["decision_id"]) for b in v["breaks"]] == [(3, "d0003")]


def test_verify_detects_deleting_the_oldest_lines(tmp_path):
    p = tmp_path / "decisions.jsonl"
    append_rows(p, 3)
    _rewrite(p, _lines(p)[1:])
    v = R.verify_chain(p)
    assert not v["ok"] and [b["line"] for b in v["breaks"]] == [1]


def test_verify_detects_an_inserted_line(tmp_path):
    p = tmp_path / "decisions.jsonl"
    append_rows(p, 4)
    lines = _lines(p)
    lines.insert(2, lines[0])                                    # a valid receipt replayed in the middle
    _rewrite(p, lines)
    v = R.verify_chain(p)
    assert not v["ok"] and [b["line"] for b in v["breaks"]] == [3, 4]


def test_verify_detects_a_reordered_pair(tmp_path):
    p = tmp_path / "decisions.jsonl"
    append_rows(p, 4)
    lines = _lines(p)
    lines[1], lines[2] = lines[2], lines[1]
    _rewrite(p, lines)
    v = R.verify_chain(p)
    assert not v["ok"] and [b["line"] for b in v["breaks"]] == [2, 3, 4]
    assert all("inserted, deleted or reordered" in b["why"] for b in v["breaks"])


def test_rewriting_or_dropping_the_newest_receipt_changes_the_head_digest(tmp_path):
    """Documented limit: the newest line can be rewritten or truncated without a break in the file; the head digest
    changes, so a head anchored elsewhere catches it."""
    p = tmp_path / "decisions.jsonl"
    append_rows(p, 3)
    anchored = R.verify_chain(p)["head"]
    lines = _lines(p)
    row = json.loads(lines[-1])
    row["action"] = "confirm"
    row["receipt"]["digest"] = R.row_digest(row)
    _rewrite(p, [*lines[:-1], json.dumps(row).encode()])
    v = R.verify_chain(p)
    assert v["ok"] and v["head"] != anchored
    _rewrite(p, lines[:-1])
    v = R.verify_chain(p)
    assert v["ok"] and v["head"] not in (None, anchored)


def test_a_torn_last_line_is_terminated_and_never_chained(tmp_path):
    p = tmp_path / "decisions.jsonl"
    log = R.ChainedLog(p)
    recs = [log.append(receipt_row(i)) for i in range(2)]
    whole = json.dumps(receipt_row(9)).encode()
    fragment = whole[: len(whole) // 2]
    with open(p, "ab") as f:
        f.write(fragment)                                        # a crash mid-write: no newline
    v = R.verify_chain(p)
    assert v["ok"] and v["unreadable"] == [3] and v["receipts"] == 2 and v["head"] == recs[-1]["receipt"]["digest"]
    nxt = log.append(receipt_row(3))                             # the same writer, whose cached tip is now stale
    assert nxt["receipt"]["prev"] == recs[-1]["receipt"]["digest"]
    lines = _lines(p)
    assert len(lines) == 4 and lines[2] == fragment and json.loads(lines[3])["decision_id"] == "d0003"
    v = R.verify_chain(p)
    assert v["ok"] and v["unreadable"] == [3] and v["receipts"] == 3 and v["head"] == nxt["receipt"]["digest"]
    again = R.ChainedLog(p).append(receipt_row(4))               # and a restarted writer skips it as well
    assert again["receipt"]["prev"] == nxt["receipt"]["digest"] and R.verify_chain(p)["ok"]


def test_a_torn_line_cut_inside_a_utf8_character_is_reported_not_fatal_to_verify(tmp_path):
    p = tmp_path / "decisions.jsonl"
    log = DecisionLog(p)
    first = log.append(receipt_row(1, state={"q": "héllo"}))
    whole = json.dumps(receipt_row(2, state={"q": "é" * 20}), ensure_ascii=False).encode("utf-8")
    with open(p, "ab") as f:
        f.write(whole[: whole.index("é".encode()) + 1])           # half of a two-byte character
    nxt = log.append(receipt_row(3))
    assert nxt["receipt"]["prev"] == first["receipt"]["digest"]
    v = R.verify_chain(p)
    assert v["ok"] and v["unreadable"] == [2] and v["receipts"] == 2

def test_reading_the_log_survives_a_torn_line_cut_inside_a_utf8_character(tmp_path):
    p = tmp_path / "decisions.jsonl"
    log = DecisionLog(p)
    log.append(receipt_row(1, state={"q": "héllo"}))
    whole = json.dumps(receipt_row(2, state={"q": "é" * 20}), ensure_ascii=False).encode("utf-8")
    with open(p, "ab") as f:
        f.write(whole[: whole.index("é".encode()) + 1])
    log.append(receipt_row(3))
    assert [r["decision_id"] for r in DecisionLog(p).rows()] == ["d0001", "d0003"]


def test_legacy_lines_are_counted_and_the_chain_starts_after_them(tmp_path):
    p = tmp_path / "decisions.jsonl"
    legacy = [{"decision_id": f"old{i}", "created_at": "2026-01-01T00:00:00Z", "kind": "decide", "heads": {}}
              for i in range(3)]
    legacy.append({"decision_id": "old3", "receipt": {"note": "a receipt part without a digest"}})
    p.write_text("".join(json.dumps(r) + "\n" for r in legacy) + "\n", encoding="utf-8")   # and a blank line
    recs = append_rows(p, 2)
    assert recs[0]["receipt"]["prev"] is None
    v = R.verify_chain(p)
    assert v["ok"] and v["legacy"] == 4 and v["receipts"] == 2 and v["lines"] == 6 and v["unreadable"] == []
    assert R.verify_chain(tmp_path / "missing.jsonl") == {"path": "missing.jsonl", "lines": 0, "receipts": 0,
                                                          "legacy": 0, "unreadable": [], "ok": True, "breaks": [],
                                                          "head": None}

def test_verify_flags_a_record_without_receipt_inserted_between_receipts(tmp_path):
    p = tmp_path / "decisions.jsonl"
    append_rows(p, 3)
    lines = _lines(p)
    forged = {"decision_id": "forged", "created_at": "2026-11-09T12:00:00Z", "kind": "decide", "model": MODEL,
              "action": "act", "heads": {"answerable": {"class": "true", "p": 0.99, "tier": "act",
                                                        "state": {"q": "poisoned training example"}}}}
    lines.insert(2, json.dumps(forged).encode())
    _rewrite(p, lines)
    assert not R.verify_chain(p)["ok"]

def test_verify_detects_a_replaced_state_with_a_planted_digest(tmp_path):
    p = tmp_path / "decisions.jsonl"
    append_rows(p, 3)
    lines = _lines(p)
    row = json.loads(lines[1])
    head = row["heads"]["answerable"]
    head["state_digest"] = D.digest(head["state"])
    head["state"] = {"q": "poisoned training example", "evidence": []}
    lines[1] = json.dumps(row).encode()
    _rewrite(p, lines)
    assert not R.verify_chain(p)["ok"]

def test_verify_reports_a_receipt_with_malformed_heads_instead_of_crashing(tmp_path):
    p = tmp_path / "decisions.jsonl"
    append_rows(p, 2)
    lines = _lines(p)
    row = json.loads(lines[0])
    row["heads"] = ["tampered"]
    lines[0] = json.dumps(row).encode()
    _rewrite(p, lines)
    v = R.verify_chain(p)
    assert not v["ok"] and 1 in [b["line"] for b in v["breaks"]]


@pytest.mark.parametrize("shared", [True, False], ids=["one-writer-object", "one-object-per-thread"])
def test_concurrent_appends_from_threads_keep_one_valid_chain(tmp_path, shared):
    p = tmp_path / "decisions.jsonl"
    one = R.ChainedLog(p)
    n_threads, per = 8, 40
    errors: list[BaseException] = []
    barrier = threading.Barrier(n_threads)

    def worker(k: int) -> None:
        log = one if shared else R.ChainedLog(p)
        barrier.wait()
        try:
            for j in range(per):       # some records are longer than the 64 KiB block the tip reader steps back by
                state = {"q": f"t{k}-{j}", "pad": "é" * ((j % 5) * 9000)}
                log.append(receipt_row(k * 1000 + j, state=state))
        except BaseException as e:       # collected and asserted empty below (a thread cannot fail the test)
            errors.append(e)

    threads = [threading.Thread(target=worker, args=(k,)) for k in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    v = R.verify_chain(p)
    assert v["ok"] and v["receipts"] == n_threads * per and v["unreadable"] == [] and v["legacy"] == 0
    assert len({json.loads(x)["decision_id"] for x in _lines(p)}) == n_threads * per


_WRITER = """
import os, sys, time
from open_rlcd.core import receipts as R
path, go, k, n = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
deadline = time.monotonic() + 60
while not os.path.exists(go) and time.monotonic() < deadline:   # start together
    time.sleep(0.001)
log = R.ChainedLog(path)
for j in range(n):
    log.append({"decision_id": f"p{k}-{j}", "kind": "decide", "receipt": {"format": R.FORMAT},
                "heads": {"h": {"tier": "act", "state": {"q": "x" * (j * 997 % 20000)}}}})
"""


@pytest.mark.skipif(R.fcntl is None, reason="appends from several processes are serialised by fcntl (POSIX)")
def test_concurrent_appends_from_processes_keep_one_valid_chain(tmp_path):
    """The documented multi-process case: several servers on one home append under the file lock."""
    p, go = tmp_path / "decisions.jsonl", tmp_path / "go"
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(REPO), os.environ.get("PYTHONPATH", "")])}
    procs = [subprocess.Popen([sys.executable, "-c", _WRITER, str(p), str(go), str(k), "150"], env=env,
                              cwd=str(REPO)) for k in range(4)]
    go.touch()
    assert [proc.wait(timeout=120) for proc in procs] == [0, 0, 0, 0]
    v = R.verify_chain(p)
    assert v["ok"] and v["receipts"] == 600 and v["unreadable"] == []


def test_a_restarted_writer_finds_the_tip_behind_lines_longer_than_a_read_block(tmp_path):
    p = tmp_path / "decisions.jsonl"
    block = getattr(R, "_BLOCK", 1 << 16)                     # the tip is read back from the end in blocks
    big = "x" * (3 * block)
    recs = [R.ChainedLog(p).append(receipt_row(i, state={"q": q}))
            for i, q in enumerate([big, "small", big + "y", "z", "é" * block])]
    for a, b in zip(recs, recs[1:], strict=False):
        assert b["receipt"]["prev"] == a["receipt"]["digest"]
    assert R.verify_chain(p)["ok"]

def test_cached_tip_is_not_trusted_after_the_file_shrank_and_regrew_to_the_same_size(tmp_path):
    p = tmp_path / "decisions.jsonl"
    a = R.ChainedLog(p)
    a.append(receipt_row(1, state={"q": "x" * 3000}, created_at="2026-11-01T00:00:00Z"))
    size = p.stat().st_size
    assert R.prune_states(p, "2026-11-05")["pruned"] == 1        # the old state goes: the file shrinks
    scratch = tmp_path / "scratch.jsonl"                         # measure the other writer's record first
    scratch.write_bytes(p.read_bytes())
    row = receipt_row(2)
    row["pad"] = ""
    R.ChainedLog(scratch).append(row)
    pad = size - scratch.stat().st_size
    assert pad >= 0
    row = receipt_row(2)
    row["pad"] = "p" * pad
    R.ChainedLog(p).append(row)                                  # another process
    assert p.stat().st_size == size
    a.append(receipt_row(3))
    assert R.verify_chain(p)["ok"]


# ============================================================================= receipts: pruning

def test_prune_drops_old_states_and_the_chain_still_verifies(tmp_path):
    p = tmp_path / "decisions.jsonl"
    legacy = {"decision_id": "legacy", "created_at": "2026-01-01T00:00:00Z", "heads": {"h": {"state": {"q": "old"}}}}
    p.write_text(json.dumps(legacy) + "\n", encoding="utf-8")
    log = R.ChainedLog(p)
    days = ["2026-11-01T12:00:00Z", "2026-11-03T23:59:59+00:00", "2026-11-05T00:00:00Z"]
    states = [{"q": f"s{i}", "document": "confidential " * 20} for i in range(3)]
    recs = [log.append(receipt_row(i, state=s, created_at=d)) for i, (s, d) in enumerate(zip(states, days,
                                                                                              strict=True))]
    before = p.stat().st_size
    out = R.prune_states(p, "2026-11-04")
    assert out == {"pruned": 3, "rows": 4, "before": "2026-11-04"}      # the legacy line's state goes too
    assert p.stat().st_size < before and not (tmp_path / "decisions.jsonl.bak").exists()
    rows = [json.loads(x) for x in _lines(p)]
    heads = [r["heads"]["answerable"] for r in rows[1:]]
    for h, s in zip(heads[:2], states[:2], strict=True):
        assert "state" not in h and h["state_digest"] == D.digest(s)
    assert heads[2]["state"] == states[2] and "state_digest" not in heads[2]
    # not a receipt (not chained): its state is dropped like the others, its digest kept
    assert "state" not in rows[0]["heads"]["h"] and rows[0]["heads"]["h"]["state_digest"] == D.digest({"q": "old"})
    assert [r["receipt"]["digest"] for r in rows[1:]] == [r["receipt"]["digest"] for r in recs]
    assert R.verify_chain(p)["ok"]
    assert R.prune_states(p, "2026-11-04") == {"pruned": 0, "rows": 4, "before": "2026-11-04"}   # idempotent
    nxt = log.append(receipt_row(7))                             # the writer from before the prune continues
    assert nxt["receipt"]["prev"] == recs[-1]["receipt"]["digest"] and R.verify_chain(p)["ok"]
    everything = R.prune_states(p, "2027-01-01")
    assert everything["pruned"] == 2 and R.verify_chain(p)["ok"]


def test_prune_cutoff_offsets_naive_times_and_bad_input(tmp_path):
    p = tmp_path / "decisions.jsonl"
    R.ChainedLog(p).append(receipt_row(1, created_at="2026-11-04T01:00:00Z"))
    assert R.prune_states(p, "2026-11-04T02:00:00+02:00")["pruned"] == 0      # 00:00Z: the record is younger
    assert R.prune_states(p, "2026-11-04T03:00:00+02:00")["pruned"] == 0      # 01:00Z: not strictly before
    assert R.prune_states(p, "2026-11-04T01:00:01")["pruned"] == 1            # naive: read as UTC
    assert R.verify_chain(p)["ok"]
    with pytest.raises(ValueError, match="ISO 8601"):
        R.prune_states(p, "last tuesday")
    assert R.prune_states(tmp_path / "none.jsonl", "2026-01-01") == {"pruned": 0, "rows": 0, "before": "2026-01-01"}


def test_prune_keeps_unreadable_lines_and_undated_or_unlogged_records(tmp_path):
    p = tmp_path / "decisions.jsonl"
    log = R.ChainedLog(p)
    a = log.append(receipt_row(1, created_at=None))                            # no date: never pruned
    with open(p, "ab") as f:
        f.write(b'{"torn": ')
    b = log.append(receipt_row(2, created_at="2026-01-01T00:00:00Z", log_state=False))   # nothing to prune
    out = R.prune_states(p, "2026-11-04")
    assert out == {"pruned": 0, "rows": 3, "before": "2026-11-04"}
    lines = _lines(p)
    assert lines[1] == b'{"torn": ' and json.loads(lines[0])["heads"]["answerable"]["state"] == a["heads"][
        "answerable"]["state"]
    v = R.verify_chain(p)
    assert v["ok"] and v["unreadable"] == [2] and v["head"] == b["receipt"]["digest"]


# ============================================================================= receipts: staleness

def check(row: dict, **kw) -> dict:
    base = dict(state_digest=row["receipt"]["state_digest"], latest=MODEL, known_models=[MODEL],
                policy_digest="sha256:policy", automation_cap=2, battery_version=None, now=NOW)
    return R.check(row, **{**base, **kw})


def test_check_same_state_is_current():
    out = check(receipt_row(1))
    assert out == {"decision_id": "d0001", "kind": "decide", "model": MODEL, "action": "act", "current": True,
                   "stale_reasons": [],
                   "checks": {"state": "same", "model": "same", "policy": "same", "automation": "allowed",
                              "age_s": 10.0}}


def test_check_a_changed_state_is_stale_and_key_order_is_not_a_change():
    row = receipt_row(1, state={"q": "a", "evidence": [1, 2]})
    out = check(row, state_digest=D.digest({"q": "b", "evidence": [1, 2]}))
    assert not out["current"] and out["checks"]["state"] == "changed" and len(out["stale_reasons"]) == 1
    assert check(row, state_digest=D.digest(json.loads('{"evidence": [1, 2], "q": "a"}')))["current"]
    assert not check(row, state_digest=D.digest({"q": "a", "evidence": [2, 1]}))["current"]
    assert not check(row, state_digest="")["current"]


@pytest.mark.parametrize("requested", [None, "", "rlcd-latest"])
def test_check_alias_moved_is_stale(requested):
    row = receipt_row(1, requested_model=requested)
    assert check(row)["checks"]["model"] == "same" and check(row)["current"]
    out = check(row, latest="rlcd-0.1.4", known_models=[MODEL, "rlcd-0.1.4"])
    assert not out["current"] and out["checks"]["model"] == "moved"
    assert out["stale_reasons"] == [f"rlcd-latest now points at rlcd-0.1.4, the decision was made by {MODEL}"]


def test_check_a_pinned_version_survives_a_promotion_but_not_its_removal():
    row = receipt_row(1, requested_model=MODEL)
    out = check(row, latest="rlcd-0.1.4", known_models=[MODEL, "rlcd-0.1.4"])
    assert out["current"] and out["checks"]["model"] == "pinned"
    out = check(row, latest="rlcd-0.1.4", known_models=["rlcd-0.1.4"], policy_digest=None)
    assert not out["current"] and out["checks"]["model"] == "gone"
    assert f"model {MODEL} is no longer served" in out["stale_reasons"]


def test_check_policy_changed_is_stale_and_a_receipt_without_a_policy_skips_the_check():
    out = check(receipt_row(1), policy_digest="sha256:other")
    assert not out["current"] and out["checks"]["policy"] == "changed"
    out = check(receipt_row(1, policy=None), policy_digest="sha256:anything")
    assert out["current"] and "policy" not in out["checks"]


def test_check_battery_version():
    row = receipt_row(1, battery={"name": "preflight", "version": "1", "digest": "sha256:b"})
    out = check(row, battery_version="1")
    assert out["current"] and out["checks"]["battery"] == "same"
    out = check(row, battery_version="2")
    assert not out["current"] and out["checks"]["battery"] == "changed"
    assert out["stale_reasons"] == ["battery preflight was version 1, now version 2"]
    out = check(row, battery_version=None)
    assert not out["current"] and out["stale_reasons"] == ["battery preflight was version 1, removed"]
    assert "battery" not in check(receipt_row(1), battery_version="9")["checks"]       # no battery, no check


def test_check_automation_cap_only_matters_for_act_decisions():
    acted = receipt_row(1, action="act")
    out = check(acted, automation_cap=1)
    assert not out["current"] and out["checks"]["automation"] == "capped"
    assert check(acted, automation_cap=2)["checks"]["automation"] == "allowed"
    # a head acted inside a plan that confirms: the plan authorised nothing automatic
    plan = receipt_row(2, action="confirm", head_tier="act")
    out = check(plan, automation_cap=1)
    assert out["current"] and "automation" not in out["checks"]
    esc = check(receipt_row(3, action="escalate", head_tier="escalate"), automation_cap=1)
    assert "automation" not in esc["checks"] and esc["checks"]["approval"] == "none" and not esc["current"]
    # a record without an overall action (/v1/rank, older lines): any head at act counts
    rank = receipt_row(4, action=None, kind="rank", head_tier="act")
    assert not check(rank, automation_cap=1)["current"] and check(rank, automation_cap=2)["current"]
    assert check(receipt_row(5, action=None, kind="rank", head_tier="confirm"), automation_cap=1)["current"]
    assert check(receipt_row(6, action=None, kind="systemone", head_tier=None), automation_cap=1)["current"]


def test_check_max_age():
    row = receipt_row(1, created_at="2026-11-09T12:00:00Z")
    assert check(row, max_age_s=30)["current"] and check(row, max_age_s=10)["current"]
    out = check(row, max_age_s=5)
    assert not out["current"] and out["stale_reasons"] == ["the decision is older than max_age_s=5"]
    assert out["checks"]["age_s"] == 10.0
    assert check(row, now=NOW + timedelta(days=365))["current"]          # without max_age_s, age never stales
    out = check(receipt_row(2, created_at=None), max_age_s=3600)
    assert not out["current"] and out["checks"]["age_s"] is None         # an unknown age fails closed
    out = check(receipt_row(3, created_at="not a time"), max_age_s=3600)
    assert not out["current"] and out["checks"]["age_s"] is None
    naive = receipt_row(4, created_at="2026-11-09T12:00:00")             # naive times are UTC
    assert check(naive)["checks"]["age_s"] == 10.0


def test_check_reports_every_reason_at_once():
    row = receipt_row(1, battery={"name": "b", "version": "1", "digest": "x"})
    out = check(row, state_digest="sha256:other", latest="rlcd-9", known_models=["rlcd-9"],
                policy_digest="sha256:new", automation_cap=1, battery_version="2", max_age_s=1)
    assert not out["current"] and len(out["stale_reasons"]) == 6
    assert out["checks"] == {"state": "changed", "model": "moved", "policy": "changed", "battery": "changed",
                             "automation": "capped", "age_s": 10.0}


def test_check_works_on_a_record_read_back_from_the_chain(tmp_path):
    p = tmp_path / "decisions.jsonl"
    state = {"q": "refund order 7", "amount": 12.5, "nan": float("nan")}
    R.ChainedLog(p).append(receipt_row(1, state=state))
    row = json.loads(_lines(p)[0])
    assert check(row, state_digest=D.digest(state))["current"]
    assert not check(row, state_digest=D.digest({**state, "amount": 125}))["current"]
