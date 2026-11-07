"""The wire contract and the confidence formulas, without HTTP (ported from the reference test_contract /
test_confidence). The answer shapes are what View Engine clients parse: keys must match exactly."""
import pytest

from open_rlcd.core.confidence import choice_confidence, noul_confidence, score_confidence
from open_rlcd.core.contract import (
    MAX_CHOICE_OPTIONS,
    MAX_QUESTIONS,
    ContractError,
    Question,
    build_answer,
    choice_answer,
    noul_answer,
    parse_questions,
    positive_probability,
    score_answer,
)
from open_rlcd.core.head import HeadSpec

NOUL_KEYS = {"type", "noul", "confidence", "head", "calibrated"}
CHOICE_KEYS = {"type", "choice", "probabilities", "confidence", "head", "calibrated"}
SCORE_KEYS = {"type", "score", "legend", "probabilities", "confidence", "head", "calibrated"}


# ----------------------------------------------------------------------------- confidence

@pytest.mark.parametrize("p,expected", [(0.5, 0.0), (1.0, 1.0), (0.0, 1.0), (0.8, 0.6), (0.2, 0.6)])
def test_noul_confidence_is_distance_from_a_coin_flip(p, expected):
    assert noul_confidence(p) == pytest.approx(expected)


def test_choice_confidence_worked_examples():
    assert choice_confidence([0.6, 0.3, 0.1]) == pytest.approx(0.4)        # (0.6 - 1/3) / (2/3)
    assert choice_confidence([1 / 3] * 3) == pytest.approx(0.0)
    assert choice_confidence([1.0, 0.0]) == pytest.approx(1.0)
    assert choice_confidence([0.5, 0.5]) == pytest.approx(0.0)
    assert choice_confidence([0.25, 0.25, 0.5, 0.0]) == pytest.approx((0.5 - 0.25) / 0.75)
    assert choice_confidence([1.0]) == 1.0 and choice_confidence([]) == 0.0


def test_score_confidence_worked_examples():
    assert score_confidence([0, 0.5, 0.5]) == pytest.approx(0.25)          # split between adjacent levels
    assert score_confidence([0.5, 0, 0.5]) == pytest.approx(0.0)           # split between opposite ends
    assert score_confidence([0, 0.57, 0.43]) == pytest.approx(0.355)
    assert score_confidence([0, 0, 1]) == pytest.approx(1.0)
    assert score_confidence([0.25] * 4) < 0.3
    # by hand: n = 4, MAD_unif = (1.5 + 0.5 + 0.5 + 1.5) / 4 = 1; m = 1; spread = 0.2*1 + 0.2*1 + 0.1*2 = 0.6
    assert score_confidence([0.2, 0.5, 0.2, 0.1]) == pytest.approx(0.4)


def test_score_is_distance_aware_where_choice_is_not():
    near, far = [0.6, 0.4, 0.0, 0.0, 0.0], [0.6, 0.0, 0.0, 0.0, 0.4]
    assert choice_confidence(near) == pytest.approx(choice_confidence(far))
    assert score_confidence(near) > score_confidence(far)


# ----------------------------------------------------------------------------- answers

def test_noul_answer_shape():
    a = noul_answer(0.8, head="good", calibrated=True)
    assert a == {"type": "noul", "noul": 0.8, "confidence": pytest.approx(0.6), "head": "good", "calibrated": True}
    assert set(a) == NOUL_KEYS
    assert noul_answer(1.7, head="h", calibrated=False)["noul"] == 1.0      # clipped to a probability


def test_choice_answer_shape():
    a = choice_answer(["red", "green", "blue"], [0.6, 0.3, 0.1], head="color", calibrated=True)
    assert set(a) == CHOICE_KEYS and a["type"] == "choice" and a["choice"] == "red"
    assert a["probabilities"] == {"red": 0.6, "green": 0.3, "blue": 0.1}
    assert a["confidence"] == pytest.approx(0.4) and a["head"] == "color" and a["calibrated"] is True
    tie = choice_answer(["a", "b"], [0.5, 0.5], head="h", calibrated=False)
    assert tie["choice"] == "a" and tie["confidence"] == 0.0              # ties go to the first option


def test_score_answer_shape():
    a = score_answer(["weak", "fair", "strong"], [0, 0.5, 0.5], head="h", calibrated=True)
    assert set(a) == SCORE_KEYS and a["type"] == "score" and a["score"] == 1
    assert a["legend"] == {"0": "weak", "1": "fair", "2": "strong"}
    assert a["probabilities"] == {"0": 0.0, "1": 0.5, "2": 0.5}
    assert a["confidence"] == pytest.approx(0.25)


def test_build_answer_follows_the_head_and_its_label_order():
    noul = HeadSpec("good", "noul")
    assert build_answer(None, noul, [0.3, 0.7], True)["noul"] == 0.7      # [P(false), P(true)]
    level = HeadSpec("level", "score", ("0", "1", "2"), levels=("none", "some", "all"))
    a = build_answer(None, level, [0.1, 0.2, 0.7], False)
    assert a["legend"] == {"0": "none", "1": "some", "2": "all"} and a["score"] == 2
    q = Question("q", {"type": "score", "head": "level", "criteria": ["lo", "mid", "hi"]})
    assert build_answer(q, level, [0.1, 0.2, 0.7], False)["legend"] == {"0": "lo", "1": "mid", "2": "hi"}


def test_positive_probability_for_ranking():
    assert positive_probability(noul_answer(0.3, head="h", calibrated=True), None) == 0.3
    c = choice_answer(["a", "b"], [0.2, 0.8], head="h", calibrated=True)
    assert positive_probability(c, "b") == 0.8
    with pytest.raises(ContractError, match="needs `option`"):
        positive_probability(c, None)
    with pytest.raises(ContractError, match="not one of"):
        positive_probability(c, "z")
    with pytest.raises(ContractError, match="noul or a choice"):
        positive_probability(score_answer(["x", "y"], [0.5, 0.5], head="h", calibrated=True), None)


# ----------------------------------------------------------------------------- questions

def test_choice_limit_is_255_options():
    Question("q", {"type": "choice", "criteria": {f"o{i}": "x" for i in range(MAX_CHOICE_OPTIONS)}})
    with pytest.raises(ContractError, match="255") as e:
        Question("q", {"type": "choice", "criteria": {f"o{i}": "x" for i in range(256)}})
    assert e.value.kind == "validation" and e.value.details["limit"] == 255


@pytest.mark.parametrize("levels", [2, 10])
def test_score_accepts_2_to_10_levels(levels):
    assert len(Question("q", {"type": "score", "criteria": [str(i) for i in range(levels)]}).criteria) == levels


@pytest.mark.parametrize("levels", [0, 1, 11])
def test_score_rejects_other_level_counts(levels):
    with pytest.raises(ContractError, match="2 to 10"):
        Question("q", {"type": "score", "criteria": [str(i) for i in range(levels)]})


@pytest.mark.parametrize("raw,match", [
    ("noul", "must be an object"),
    ({"type": "essay"}, "type must be one of"),
    ({"type": "noul", "instructions": 3}, "instructions must be a string"),
    ({"type": "noul", "head": " "}, "head must be a non-empty string"),
    ({"type": "noul", "criteria": {"maybe": "?"}}, "keys 'true' and"),
    ({"type": "choice", "criteria": {}}, "non-empty object"),
    ({"type": "choice", "criteria": ["a", "b"]}, "non-empty object"),
    ({"type": "score", "criteria": {"0": "low"}}, "ordered array"),
])
def test_malformed_questions(raw, match):
    with pytest.raises(ContractError, match=match) as e:
        Question("q", raw)
    assert e.value.kind == "validation" and e.value.status == 422 and e.value.body()["question"] == "q"


def test_parse_questions_limits():
    with pytest.raises(ContractError, match="non-empty object"):
        parse_questions({})
    with pytest.raises(ContractError, match="non-empty object"):
        parse_questions([{"type": "noul"}])
    with pytest.raises(ContractError, match=f"at most {MAX_QUESTIONS}"):
        parse_questions({f"q{i}": {"type": "noul"} for i in range(MAX_QUESTIONS + 1)})
    qs = parse_questions({"a": {"type": "noul"}, "b": {"type": "choice", "head": "color"}})
    assert [(q.id, q.head) for q in qs] == [("a", "a"), ("b", "color")]


def test_bind_checks_type_classes_and_levels():
    color = HeadSpec("color", "choice", ("red", "green", "blue"))
    level = HeadSpec("level", "score", ("0", "1", "2", "3"))
    with pytest.raises(ContractError, match="answers a choice question") as e:
        Question("q", {"type": "score", "head": "color", "criteria": ["a", "b", "c"]}).bind(color)
    assert e.value.details["head_type"] == "choice"
    with pytest.raises(ContractError, match="do not match") as e:
        Question("q", {"type": "choice", "head": "color", "criteria": {"red": "", "purple": ""}}).bind(color)
    assert e.value.details["classes"] == ["red", "green", "blue"]
    with pytest.raises(ContractError, match="has 4 levels") as e:
        Question("q", {"type": "score", "head": "level", "criteria": ["a", "b", "c"]}).bind(level)
    assert e.value.details["levels"] == 4
    # matching criteria (any order) and no criteria at all are both fine
    Question("q", {"type": "choice", "head": "color", "criteria": {"blue": "", "red": "", "green": ""}}).bind(color)
    Question("q", {"type": "choice", "head": "color"}).bind(color)
    Question("q", {"type": "score", "head": "level", "criteria": ["a", "b", "c", "d"]}).bind(level)


def test_contract_error_body():
    e = ContractError("nope", kind="unknown_head", available_heads=["a"])
    assert e.body() == {"error": "nope", "kind": "unknown_head", "available_heads": ["a"]} and e.status == 422
