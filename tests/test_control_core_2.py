"""Adversarial unit tests for the M10 control layer, pure modules part 2: batteries (core/battery.py), redaction
(core/redact.py), uncertainty sources (core/uncertainty.py), route contracts (core/routes.py), the review decision
table (core/review.py), asymmetric policy overrides (core/policy.py) and the contract changes (core/contract.py).

No service, no training: decisions come from policy.decide on hand-made probabilities. Tests marked
`xfail(strict=True)` document a bug found while writing them: each asserts the correct behaviour and names the bug in
its reason, so it fails loudly (XPASS) once the bug is fixed and the marker must go. Part 1 (digests, inputs,
rollout, receipts) is tests/test_control_core.py.
"""
from __future__ import annotations

import copy
import json
import random

import numpy as np
import pytest
import yaml
from pydantic import ValidationError

from open_rlcd.core import battery as BA
from open_rlcd.core import bernoulli as B
from open_rlcd.core import policy as P
from open_rlcd.core import redact as RD
from open_rlcd.core import review as RV
from open_rlcd.core import routes as RT
from open_rlcd.core import uncertainty as U
from open_rlcd.core.confidence import choice_confidence
from open_rlcd.core.config import Project
from open_rlcd.core.contract import (
    MAX_CHOICE_OPTIONS,
    AutomationRequest,
    ContractError,
    FeedbackRequest,
    OutcomeRequest,
    Question,
    ReceiptCheckRequest,
    SystemOneRequest,
    build_answer,
    choice_answer,
    questions_digest,
    restrict,
)
from open_rlcd.core.head import HeadSpec

COLOR = HeadSpec("color", "choice", ("red", "green", "blue"))
GATE = HeadSpec("answerable", "noul", decision="gate")
SYM = HeadSpec("needs_graph", "noul")
LEVEL = HeadSpec("level", "score", ("0", "1", "2", "3"), levels=("none", "low", "mid", "high"))
STARS = HeadSpec("stars", "score", ("one", "two", "three"))


# ============================================================================= batteries

Q = {"kind": {"type": "choice", "head": "response_type"},
     "multi_hop": {"type": "noul", "head": "needs_graph", "instructions": "Does answering need several hops?"}}


def bat(**kw) -> dict:
    return {"version": "1", "questions": Q, **kw}


def one(d: dict) -> BA.Battery:
    return BA.load({"b": d}, None)["b"]


def test_load_overlays_the_projects_batteries_on_the_packs():
    pack = {"preflight": bat(owner="search", changelog="1: first"), "postflight": bat()}
    project = {"preflight": {"version": "2", "questions": {"kind": Q["kind"]}}, "intake": bat()}
    out = BA.load(pack, project)
    assert sorted(out) == ["intake", "postflight", "preflight"]
    pre = out["preflight"]
    assert pre.source == "project" and pre.version == "2" and pre.owner is None and pre.changelog == ()
    assert [q.id for q in pre.questions()] == ["kind"]                          # replaced, not merged
    assert out["postflight"].source == "pack" and out["intake"].source == "project"
    assert BA.load(pack, None)["preflight"].changelog == ("1: first",)
    assert BA.load(pack, None)["preflight"].owner == "search"
    assert BA.load(None, None) == {} and BA.load({}, {}) == {}


@pytest.mark.parametrize("section,match", [
    ({"bad name": bat()}, "battery name"),
    ({"x" * 65: bat()}, "battery name"),
    ({"pré": bat()}, "battery name"),
    ({"b": ["not", "a", "mapping"]}, "must be a mapping"),
    ({"b": None}, "must be a mapping"),
    ({"b": {**bat(), "owners": "typo"}}, "unknown fields"),
    ({"b": {"questions": Q}}, "needs a version"),
    ({"b": bat(version=None)}, "needs a version"),
    ({"b": bat(version="1 0")}, "needs a version"),
    ({"b": bat(version="v@1")}, "needs a version"),
    ({"b": bat(version="")}, "needs a version"),
    ({"b": bat(version="1" * 33)}, "needs a version"),
    ({"b": {"version": "1"}}, "needs questions"),
    ({"b": {"version": "1", "questions": {}}}, "needs questions"),
    ({"b": {"version": "1", "questions": ["kind"]}}, "needs questions"),
    ({"b": {"version": "1", "questions": {"q": {"type": "maybe"}}}}, "battery 'b': question 'q'"),
    ({"b": {"version": "1", "questions": {"q": {"type": "noul", "allowed": ["x"]}}}}, "allowed"),
    ({"b": {"version": "1", "questions": {"q": {"type": "noul", "path": "a..b"}}}}, "empty segment"),
    ({"b": bat(consumers=["kind"])}, "consumers maps"),
    ({"b": bat(consumers={"kind": "router", "ghost": "x"})}, r"does not ask: \['ghost'\]"),
])
def test_battery_validation(section, match):
    with pytest.raises(ValueError, match=match):
        BA.load(section, None)
    with pytest.raises(ValueError, match=match):          # the project's section follows the same rules
        BA.load(None, section)


def test_a_batteries_section_must_be_a_mapping():
    for bad in (["preflight"], "preflight"):
        with pytest.raises(ValueError, match="mapping of name"):
            BA.load(bad, None)


def test_parse_ref():
    assert BA.parse_ref("preflight") == ("preflight", None)
    assert BA.parse_ref(" preflight@2.1-rc_1 ") == ("preflight", "2.1-rc_1")
    assert BA.parse_ref("a" * 64 + "@" + "1" * 32) == ("a" * 64, "1" * 32)


@pytest.mark.parametrize("bad", [None, 3, "", "   ", "@1", "pre flight", "a@b@c", "pre@1 0", "x" * 65, "pré",
                                 "pre@" + "1" * 33, ["preflight"]])
def test_parse_ref_refuses_what_is_not_a_battery_reference(bad):
    with pytest.raises(ContractError) as e:
        BA.parse_ref(bad)
    assert e.value.status == 422 and e.value.details == {"field": "battery"}


def test_resolve_unknown_and_version_mismatch():
    bs = BA.load({"preflight": bat(version="2", changelog=["1: a", "2: b"])}, None)
    assert BA.resolve("preflight", bs) is bs["preflight"] and BA.resolve("preflight@2", bs) is bs["preflight"]
    with pytest.raises(ContractError) as e:
        BA.resolve("postflight", bs)
    assert e.value.kind == "unknown_battery" and e.value.status == 422
    assert e.value.details == {"available": ["preflight"]}
    with pytest.raises(ContractError) as e:
        BA.resolve("preflight@1", bs)
    assert e.value.kind == "battery_version_mismatch" and e.value.status == 422
    assert e.value.details == {"battery": "preflight", "expected": "1", "current": "2", "changelog": ["1: a", "2: b"]}
    assert "pin preflight@2" in e.value.message
    with pytest.raises(ContractError) as e:
        BA.resolve("anything", {})
    assert e.value.kind == "unknown_battery" and e.value.details == {"available": []}


def test_battery_digest_follows_question_meaning_not_wording_or_metadata():
    base = one(bat())
    assert base.digest.startswith("sha256:")
    same = [
        bat(questions={**Q, "multi_hop": {**Q["multi_hop"], "instructions": "Reworded entirely."}}),
        bat(questions={"multi_hop": Q["multi_hop"], "kind": Q["kind"]}),            # declaration order
        bat(questions={"kind": {"type": "choice", "head": "response_type", "instructions": None}, "multi_hop": {
            "type": "noul", "head": "needs_graph"}}),
        bat(version="2", owner="someone", description="d", changelog=["2: words"], consumers={"kind": "router"}),
    ]
    for d in same:
        assert one(d).digest == base.digest, d
    changed = [
        {"kind": {**Q["kind"], "head": "route"}},
        {"kind": {**Q["kind"], "allowed": ["a", "b"]}},
        {"kind": {**Q["kind"], "criteria": {"a": "A", "b": "B"}}},
        {"kind": {**Q["kind"], "option": "a"}},
        {"kind": {**Q["kind"], "path": "request"}},
        {"kind": {**Q["kind"], "inputs": {"query": "request.text"}}},
        {"kind": {**Q["kind"], "type": "noul"}},
        {"multi_hop": {**Q["multi_hop"], "criteria": {"true": "several hops"}}},
    ]
    for q in changed:
        assert one(bat(questions={**Q, **q})).digest != base.digest, q
    assert one(bat(questions={"renamed": Q["kind"], "multi_hop": Q["multi_hop"]})).digest != base.digest
    assert one(bat(questions={"kind": Q["kind"]})).digest != base.digest


def test_battery_ref_and_to_dict():
    b = one(bat(consumers={"kind": "router"}, description="before the LLM"))
    assert b.ref() == {"name": "b", "version": "1", "digest": b.digest}
    d = b.to_dict()
    assert d["questions"]["multi_hop"] == {"type": "noul", "head": "needs_graph"}
    assert d["consumers"] == {"kind": "router"} and d["source"] == "pack" and d["description"] == "before the LLM"
    assert json.loads(json.dumps(d)) == d


def test_a_battery_name_with_a_trailing_newline_is_refused():
    with pytest.raises(ContractError):
        BA.parse_ref("pre\n@1")
    with pytest.raises(ValueError):
        BA.load({"pre\n": bat()}, None)


def test_a_float_battery_version_is_refused():
    section = yaml.safe_load("pre:\n  version: 1.10\n  questions: {q: {type: noul}}\n")
    assert section["pre"]["version"] == 1.1
    with pytest.raises(ValueError, match="version"):
        BA.load(section, None)


# ============================================================================= redaction

PEM = {
    "RSA": "-----BEGIN RSA PRIVATE KEY-----\nMIIEowIBAAKCAQEAu1SU1L\n-----END RSA PRIVATE KEY-----",
    "PKCS8": "-----BEGIN PRIVATE KEY-----\nMIIEvQIBADANBgkqhkiG9w0BAQEF\n-----END PRIVATE KEY-----",
    "OPENSSH": "-----BEGIN OPENSSH PRIVATE KEY-----\nb3BlbnNzaC1rZXktdjEAAAAA\n-----END OPENSSH PRIVATE KEY-----",
    "EC": "-----BEGIN EC PRIVATE KEY-----\nMHcCAQEEIBkg\n-----END EC PRIVATE KEY-----",
    "ENCRYPTED": "-----BEGIN ENCRYPTED PRIVATE KEY-----\nMIIFHDBOBgkq\n-----END ENCRYPTED PRIVATE KEY-----",
}
SECRETS = [
    *[("private_key", v) for v in PEM.values()],
    ("aws_key", "AKIAIOSFODNN7EXAMPLE"), ("aws_key", "ASIAY34FZKBOKMUTVV7A"),
    ("github_token", "ghp_" + "a1B2" * 9), ("github_token", "gho_" + "Z" * 36), ("github_token", "ghu_" + "9" * 40),
    ("github_token", "ghs_" + "x" * 36), ("github_token", "ghr_" + "y" * 36),
    ("github_token", "github_pat_11ABCDEFG0123456789_abcdefghijklmnopqrstuvwxyz"),
    ("slack_token", "xoxb-1234567890-0987654321-abcdEFGHijkl"), ("slack_token", "xoxp-12345678901"),
    ("slack_token", "xoxa-2-abcdefghij"), ("slack_token", "xoxr-abcdefghijk"), ("slack_token", "xoxs-abcdefghijk"),
    ("api_key", "sk-" + "A" * 48), ("api_key", "sk-ant-api03-" + "b" * 40),
    ("api_key", "sk-proj-" + "c_d-x" * 6 + "9"), ("api_key", "sk_live_" + "4eC39HqLyjWDarjtT1zdp7dc"),
    ("api_key", "rk_live_" + "0" * 24),
    ("jwt", "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.dozjgNryP4J3jVmNHl0w5N_XgL0n3I9PlFUP0THsR8U"),
]


@pytest.mark.parametrize("kind,secret", SECRETS, ids=[f"{k}-{i}" for i, (k, _) in enumerate(SECRETS)])
def test_every_secret_kind_is_redacted_in_place(kind, secret):
    text = f"config: {secret} (rotated weekly)"
    out, found = RD.redact_text(text)
    assert out == f"config: [REDACTED:{kind}] (rotated weekly)" and found == [kind]
    assert RD.redact({"nested": [{"v": text}]}) == ({"nested": [{"v": out}]}, [kind])


@pytest.mark.parametrize("text,expected", [
    ("Authorization: Bearer abcdefghijklmnop1234", "Authorization: Bearer [REDACTED:bearer]"),
    ("authorization: BEARER abcdefghijklmnop1234==", "authorization: BEARER [REDACTED:bearer]"),
    ("postgres://admin:s3cr3t@db.internal:5432/x", "postgres://admin:[REDACTED:url_password]@db.internal:5432/x"),
    ("https://user:pa55word@example.com/x", "https://user:[REDACTED:url_password]@example.com/x"),
    ("password=hunter2xyz", "password=[REDACTED:assignment]"),
    ("PASSWORD: hunter2xyz;", "PASSWORD: [REDACTED:assignment];"),
    ("API_KEY = 'abcdefgh'", "API_KEY = [REDACTED:assignment]"),
    ('secret="s3cr3tvalue"', "secret=[REDACTED:assignment]"),
    ("api-key: abcdef123", "api-key: [REDACTED:assignment]"),
    ("apikey=abcdef123", "apikey=[REDACTED:assignment]"),
    ("token=abcdef, user=ann", "token=[REDACTED:assignment], user=ann"),
    ("passwd: abcdefg", "passwd: [REDACTED:assignment]"),
    ("pwd=abcdef", "pwd=[REDACTED:assignment]"),
])
def test_contextual_secrets_keep_their_context(text, expected):
    out, found = RD.redact_text(text)
    assert out == expected and len(found) == 1


def test_several_secrets_in_one_string_are_all_redacted_and_listed():
    text = "AKIAIOSFODNN7EXAMPLE then ghp_" + "a" * 36 + " and password=hunter2xyz in Zürich"
    out, found = RD.redact_text(text)
    assert out == "[REDACTED:aws_key] then [REDACTED:github_token] and password=[REDACTED:assignment] in Zürich"
    assert found == ["aws_key", "github_token", "assignment"]


PROSE = [
    "The token was valid until midnight.",
    "Our secret sauce is patience and good data.",
    "Password reset links expire after 24 hours.",
    "See https://example.com:8080/docs?page=2#auth for the bearer token flow.",
    "The commit 9fceb02d0ae598e95dc970b74767f19372d61af8 fixed it.",
    "Request 123e4567-e89b-12d3-a456-426614174000, order 1234 5678 9012, total 4111.11 EUR.",
    "Bearer of bad news: the API key rotation is on Friday.",
    "sk-short and AKIA123 are not keys; neither is eyJhbGc.",
    "Write to jane@example.com about card 4111 1111 1111 1111.",       # pii is opt-in
    "Ünïcödé text — with dashes, «quotes», emoji 🙂 and 漢字.",
    "",
]


@pytest.mark.parametrize("text", PROSE)
def test_normal_prose_is_untouched(text):
    assert RD.redact_text(text) == (text, [])


def test_redact_walks_nested_values_and_never_touches_keys_numbers_or_the_input():
    state = {"query": "deploy with AKIAIOSFODNN7EXAMPLE", "n": 3, "p": 0.5, "ok": True, "none": None,
             "evidence": [{"text": "token=abcdef123456"}, ("tuple", "ghp_" + "a" * 36)],
             "AKIAIOSFODNN7EXAMPLE": "a key named like a secret", "nan": float("nan")}
    original = copy.deepcopy(state)
    out, kinds = RD.redact(state)
    assert json.dumps(state, sort_keys=True) == json.dumps(original, sort_keys=True)   # the request is not modified
    assert out["query"] == "deploy with [REDACTED:aws_key]"
    assert out["n"] == 3 and out["p"] == 0.5 and out["ok"] is True and out["none"] is None and out["nan"] != out["nan"]
    assert out["evidence"][0] == {"text": "token=[REDACTED:assignment]"}
    assert out["evidence"][1] == ["tuple", "[REDACTED:github_token]"]              # tuples come back as lists
    assert out["AKIAIOSFODNN7EXAMPLE"] == "a key named like a secret"               # keys are never touched
    assert kinds == ["assignment", "aws_key", "github_token"]                        # sorted, once each
    assert RD.redact("plain") == ("plain", []) and RD.redact(None) == (None, []) and RD.redact(7) == (7, [])


def test_redact_replaces_nesting_deeper_than_max_depth_without_recursing_further():
    deep: object = "AKIAIOSFODNN7EXAMPLE"
    for _ in range(RD.MAX_DEPTH + 5):
        deep = [deep]
    out, kinds = RD.redact({"d": deep, "shallow": 1})
    assert kinds == ["too_deep"] and out["shallow"] == 1
    assert "AKIAIOSFODNN7EXAMPLE" not in json.dumps(out) and "[REDACTED:too_deep]" in json.dumps(out)
    very: object = "x"
    for _ in range(20_000):                                   # far beyond the interpreter's recursion limit
        very = {"k": very}
    assert RD.redact(very)[1] == ["too_deep"]
    ok: object = "fine"
    for _ in range(RD.MAX_DEPTH):
        ok = [ok]
    assert RD.redact(ok) == (ok, [])                          # MAX_DEPTH levels are still read


@pytest.mark.parametrize("card", ["4111 1111 1111 1111", "4111-1111-1111-1111", "4111111111111111",
                                  "378282246310005", "6011 1111 1111 1117", "4012888888881881"])
def test_pii_masks_luhn_valid_card_numbers_only_when_asked(card):
    text = f"paid with {card} yesterday"
    assert RD.redact_text(text) == (text, [])                                       # secrets only: untouched
    assert RD.redact_text(text, ("pii",)) == ("paid with [REDACTED:card] yesterday", ["card"])
    assert RD.redact({"t": text}, ["secrets", "pii"]) == ({"t": "paid with [REDACTED:card] yesterday"}, ["card"])


@pytest.mark.parametrize("number", ["4111 1111 1111 1112", "1234 5678 9012", "12345678901234567890", "2026-11-09",
                                    "+1 415 555 2671", "0000"])
def test_pii_keeps_numbers_that_are_not_card_numbers(number):
    text = f"ref {number} ok"
    assert RD.redact_text(text, ("pii", "secrets")) == (text, [])


def test_pii_masks_email_addresses_only_when_asked():
    text = "mail jane.doe+x@example.co.uk now"
    assert RD.redact_text(text, ("pii",)) == ("mail [REDACTED:email] now", ["email"])
    assert RD.redact_text(text) == (text, [])
    assert RD.redact_text(text, ()) == (text, [])
    # with pii only, secrets stay (the kinds are independent)
    assert RD.redact_text("AKIAIOSFODNN7EXAMPLE", ("pii",)) == ("AKIAIOSFODNN7EXAMPLE", [])


def _random_state(rnd: random.Random, leaves: list, depth: int = 0):
    r = rnd.random()
    if depth > 3 or r < 0.35:
        return rnd.choice(leaves)
    if r < 0.65:
        return [_random_state(rnd, leaves, depth + 1) for _ in range(rnd.randint(0, 4))]
    return {f"k{rnd.randint(0, 20)}": _random_state(rnd, leaves, depth + 1) for _ in range(rnd.randint(0, 4))}


def _shape(v):
    if isinstance(v, dict):
        return {k: _shape(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_shape(x) for x in v]
    return "str" if isinstance(v, str) else repr(v)


def test_redaction_keeps_the_shape_of_random_states_and_removes_every_planted_secret():
    rnd = random.Random(11)
    leaves = [None, True, 0, 2.5, "plain text", "", "AKIAIOSFODNN7EXAMPLE", "password=hunter2xyz",
              "ghp_" + "a" * 36, "mail ann@example.com", "ok 漢字"]
    for _ in range(300):
        state = _random_state(rnd, leaves)
        out, kinds = RD.redact(state)
        assert _shape(out) == _shape(state) and kinds == sorted(set(kinds))
        dumped = json.dumps(out)
        assert "AKIAIOSFODNN7EXAMPLE" not in dumped and "hunter2xyz" not in dumped and "a" * 36 not in dumped
        assert ("ann@example.com" in dumped) == ("ann@example.com" in json.dumps(state))   # pii is opt-in
        again, _ = RD.redact(out)
        assert again == out                                                                # a fixed point


def test_kinds_from_config():
    assert RD.kinds_from_config(None) == RD.kinds_from_config(True) == ("secrets",)
    assert RD.kinds_from_config(False) == () and RD.kinds_from_config([]) == ()
    assert RD.kinds_from_config("pii") == ("pii",)
    assert RD.kinds_from_config(["secrets", "pii"]) == ("secrets", "pii")
    for bad in (["secrets", "emails"], "all", ["Secrets"]):
        with pytest.raises(ValueError, match="unknown kinds"):
            RD.kinds_from_config(bad)


def test_values_under_secret_named_keys_are_redacted():
    out, _ = RD.redact({"user": "ann", "password": "hunter2!xyz", "db": {"api_key": "abc123def456", "token": "z" * 12}})
    dumped = json.dumps(out)
    assert "hunter2!xyz" not in dumped and "abc123def456" not in dumped and "z" * 12 not in dumped
    assert out["user"] == "ann"


@pytest.mark.parametrize("text,secret", [
    ("DB_PASSWORD=hunter2xyz", "hunter2xyz"),
    ("AWS_SECRET_ACCESS_KEY=wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY", "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"),
    ("client_secret=abcdefgh123", "abcdefgh123"),
    ("export API_TOKEN=abcdef123456", "abcdef123456"),
    ('{"password": "hunter2xyz"}', "hunter2xyz"),
])
def test_common_assignment_forms_are_redacted(text, secret):
    out, _ = RD.redact_text(text)
    assert secret not in out


def test_a_private_key_block_without_its_end_line_is_redacted():
    text = ("-----BEGIN RSA PRIVATE KEY-----\nMIIEowIBAAKCAQEAu1SU1LfVLPHCozMxH2Mo4lgOEePzNm0tRgeLezV6ffAt0gun\n"
            "VTLw7onLRnrq0/IzW7yWR7QkrmBL7jTKEn5u")
    out, _ = RD.redact_text(text)
    assert "MIIEowIBAAKCAQEA" not in out and "VTLw7onLRnrq0" not in out


# ============================================================================= uncertainty

CFG = P.PolicyConfig()
ACT = {s: {"tier": 2, "reason": "verified"} for s in CFG.stakes}
CONFIRM = {s: {"tier": 1, "reason": "too few holdout rows"} for s in CFG.stakes}
NO_SKILL = {s: {"tier": 0, "reason": "no skill"} for s in CFG.stakes}


def strong_ledger(classes) -> dict:
    t = B.Ledger.empty(classes, "v1")
    for c in classes:
        t["bins"][c] = [[400, 4] for _ in range(10)]
    return t


def decide(spec, probs, *, cfg=CFG, verdict=ACT, ledger=True, calibrated=True, missing=None, stakes="medium"):
    return P.decide(spec, np.array(probs, float), cfg, stakes=stakes, verdict=verdict,
                    ledger=strong_ledger(spec.labels) if ledger else None, calibrated=calibrated,
                    missing_inputs=missing)


def sources(spec, probs, *, removed=0.0, verdict_tier=None, **kw) -> list[str]:
    d = decide(spec, probs, **kw)
    return U.classify(probs, spec.labels, d, missing_inputs=kw.get("missing"), removed_mass=removed,
                      verdict_tier=verdict_tier)["sources"]


def test_entropy_and_margin():
    assert U.entropy([1 / 3] * 3) == pytest.approx(1.0) and U.entropy([1, 0, 0]) == 0.0
    assert U.entropy([1.0]) == 0.0 and U.entropy([]) == 0.0
    assert U.entropy([0.5, 0.5, 0, 0]) == pytest.approx(0.5)            # removed options count in n: log 2 / log 4
    assert U.margin([0.2, 0.5, 0.3]) == pytest.approx(0.2) and U.margin([1.0]) == 1.0 and U.margin([]) == 1.0


def test_a_clear_verified_answer_has_no_uncertainty_source():
    d = decide(COLOR, [0.94, 0.04, 0.02])
    assert d["tier"] == "act"
    out = U.classify([0.94, 0.04, 0.02], COLOR.labels, d)
    assert out == {"sources": [], "margin": pytest.approx(0.9), "entropy": pytest.approx(0.241, abs=1e-3),
                   "top2": ["red", "green"]}
    assert sources(GATE, [0.03, 0.97]) == [] and sources(SYM, [0.9, 0.1]) == []


def test_untrained_replaces_the_verdict_sources():
    s = sources(COLOR, [1 / 3, 1 / 3, 1 / 3], calibrated=False, verdict=None, ledger=False)
    assert s == ["untrained", "ambiguous", "diffuse", "below_threshold"]
    assert sources(COLOR, [0.94, 0.04, 0.02], calibrated=False, verdict=NO_SKILL) == ["untrained"]


def test_missing_evidence_and_the_input_guard_cap_is_not_read_as_unverified():
    assert sources(COLOR, [0.94, 0.04, 0.02], missing=["f3", "f4"]) == ["missing_evidence"]
    # the holdout verdict said so explicitly: both apply
    assert sources(COLOR, [0.94, 0.04, 0.02], missing=["f3"], verdict_tier=1) == ["missing_evidence", "unverified"]
    assert sources(COLOR, [0.94, 0.04, 0.02], missing=["f3"], verdict=CONFIRM, verdict_tier=1) == [
        "missing_evidence", "unverified"]


@pytest.mark.parametrize("removed,expected", [(0.0, False), (0.3, False), (0.4999, False), (0.5, True), (0.9, True)])
def test_options_removed_when_they_held_at_least_half_the_mass(removed, expected):
    assert ("options_removed" in sources(COLOR, [0.94, 0.04, 0.02], removed=removed)) is expected


@pytest.mark.parametrize("p_true,expected", [(0.5, True), (0.52, True), (0.549, True), (0.451, True),
                                             (0.551, False), (0.449, False), (0.7, False), (0.05, False)])
def test_a_noul_is_ambiguous_near_a_coin_flip(p_true, expected):
    for spec in (SYM, GATE):
        assert ("ambiguous" in sources(spec, [1 - p_true, p_true])) is expected


def test_a_choice_is_ambiguous_when_the_top_two_are_close_and_diffuse_when_flat():
    assert "ambiguous" in sources(COLOR, [0.45, 0.40, 0.15])
    assert "ambiguous" not in sources(COLOR, [0.6, 0.3, 0.1])
    assert "diffuse" in sources(COLOR, [0.34, 0.33, 0.33])
    assert "diffuse" not in sources(COLOR, [0.6, 0.3, 0.1])               # normalised entropy 0.82
    assert "diffuse" not in sources(SYM, [0.5, 0.5])                      # two options are never "diffuse"


def test_below_threshold_follows_the_stakes_level():
    assert "below_threshold" in sources(COLOR, [0.72, 0.18, 0.10], stakes="medium")      # act at 0.85
    assert "below_threshold" not in sources(COLOR, [0.72, 0.18, 0.10], stakes="low")     # act at 0.70


def test_verdict_sources_no_skill_and_unverified():
    assert sources(COLOR, [0.94, 0.04, 0.02], verdict=NO_SKILL) == ["no_skill"]
    assert sources(COLOR, [0.94, 0.04, 0.02], verdict=CONFIRM) == ["unverified"]
    assert sources(COLOR, [0.94, 0.04, 0.02], verdict_tier=0) == ["no_skill"]   # an explicit verdict wins
    hold = P.PolicyConfig(abstain="hold")                                       # the abstain action may be renamed
    assert sources(COLOR, [0.94, 0.04, 0.02], cfg=hold, verdict=NO_SKILL) == ["no_skill"]


def test_insufficient_track_record_when_the_bernoulli_gate_holds_act_back():
    d = decide(COLOR, [0.94, 0.04, 0.02], ledger=False)
    assert d["uncapped_tier"] == "act" and d["max_tier"] == "act" and d["tier"] == "confirm"
    assert U.classify([0.94, 0.04, 0.02], COLOR.labels, d)["sources"] == ["insufficient_track_record"]
    # a decision capped below act for another reason does not blame the track record
    assert sources(COLOR, [0.94, 0.04, 0.02], ledger=False, verdict=CONFIRM) == ["unverified"]


KNOWN_SOURCES = {"untrained", "missing_evidence", "options_removed", "ambiguous", "diffuse", "below_threshold",
                 "no_skill", "unverified", "insufficient_track_record"}


def test_classify_stays_in_range_on_random_distributions():
    rng = np.random.default_rng(3)
    specs, verdicts = [COLOR, SYM, GATE], [ACT, CONFIRM, NO_SKILL, None]
    for _ in range(300):
        spec = specs[int(rng.integers(len(specs)))]
        ps = rng.dirichlet(np.ones(len(spec.labels)) * rng.uniform(0.2, 3))
        missing = ["f1"] if rng.random() < 0.2 else None
        d = decide(spec, ps, ledger=bool(rng.integers(2)), verdict=verdicts[int(rng.integers(len(verdicts)))],
                   calibrated=bool(rng.random() < 0.9), missing=missing, stakes=str(rng.choice(["low", "high"])))
        out = U.classify(ps, spec.labels, d, missing_inputs=missing, removed_mass=float(rng.random()))
        assert set(out["sources"]) <= KNOWN_SOURCES and len(out["sources"]) == len(set(out["sources"]))
        assert 0.0 <= out["margin"] <= 1.0 and 0.0 <= out["entropy"] <= 1.0
        assert len(out["top2"]) == 2 and out["top2"][0] == spec.labels[int(np.argmax(ps))]
        assert ("untrained" in out["sources"]) is (not d["calibrated"])
        assert not ({"no_skill", "unverified"} <= set(out["sources"]))           # one verdict at most


def test_top2_breaks_ties_in_option_order():
    d = decide(COLOR, [0.4, 0.4, 0.2])
    assert U.classify([0.4, 0.4, 0.2], COLOR.labels, d)["top2"] == ["red", "green"]
    assert U.classify([0.2, 0.4, 0.4], COLOR.labels, d)["top2"] == ["green", "blue"]


# ============================================================================= route contracts

ROUTES = {
    "response_type": {
        "direct_answer": {"llm": "small", "context_budget_tokens": 2000, "max_retries": 1, "verify": "postflight",
                          "fallback": "escalate_human"},
        "summarize": {"llm": "large", "requires": ["llm_large"], "allowed_data": ["public", "internal"],
                      "latency_ms": 4000, "verify": "postflight", "fallback": "escalate_human"},
        "escalate_human": {"human": True},
        "experimental": {"enabled": False},
    },
    "route": {
        "graph": {"requires": ["graph"], "fallback": "hybrid"},
        "hybrid": {"requires": ["vector", "graph"], "fallback": "vector"},
        "vector": {"requires": "vector"},
    },
}


def test_constraints_from_request():
    c = RT.Constraints.from_request(None)
    assert not c.active and c == RT.Constraints() == RT.Constraints.from_request({})
    c = RT.Constraints.from_request({"capabilities": ["vector"], "sensitivity": "internal", "latency_budget_ms": 2500})
    assert c.active and c.capabilities == frozenset({"vector"}) and c.sensitivity == "internal"
    assert c.latency_budget_ms == 2500.0
    assert RT.Constraints.from_request({"capabilities": []}).active          # "nothing is available" constrains


@pytest.mark.parametrize("request_,field", [
    ({"capabilities": "vector"}, "capabilities"), ({"capabilities": ["vector", 3]}, "capabilities"),
    ({"capabilities": {"vector": True}}, "capabilities"),
    ({"sensitivity": ["internal"]}, "sensitivity"), ({"sensitivity": 1}, "sensitivity"),
    ({"latency_budget_ms": "fast"}, "latency_budget_ms"), ({"latency_budget_ms": [1]}, "latency_budget_ms"),
    ({"latency_budget_ms": 0}, "latency_budget_ms"), ({"latency_budget_ms": -5}, "latency_budget_ms"),
])
def test_constraints_validation(request_, field):
    with pytest.raises(ContractError) as e:
        RT.Constraints.from_request(request_)
    assert e.value.status == 422 and e.value.details == {"field": field}


@pytest.mark.parametrize("section,match", [
    (["route"], "maps a choice head"),
    ({"route": {}}, "maps each option"),
    ({"route": ["graph"]}, "maps each option"),
    ({"route": None}, "maps each option"),
    ({"route": {"graph": ["requires"]}}, "must be a mapping"),
    ({"route": {"graph": {"requires": [1]}}}, "requires must be a list"),
    ({"route": {"graph": {"requires": {"a": 1}}}}, "requires must be a list"),
    ({"route": {"graph": {"allowed_data": [None]}}}, "allowed_data must be a list"),
    ({"route": {"graph": {"latency_ms": "fast"}}}, "latency_ms must be a number"),
    ({"route": {"graph": {"enabled": "no"}}}, "enabled must be true or false"),
    ({"route": {"graph": {"fallback": "graph"}}}, "must name another option"),
    ({"route": {"graph": {"fallback": "ghost"}}}, "must name another option"),
    ({"route": {"graph": {"fallback": 1}}}, "must name another option"),
    ({"route": {"graph": {"fallback": "vector"}, "vector": {"fallback": "graph"}}},
     "fallback cycle graph -> vector -> graph"),
    ({"route": {"a": {"fallback": "b"}, "b": {"fallback": "c"}, "c": {"fallback": "a"}}}, "fallback cycle"),
    ({"route": {"a": {}, "b": {"fallback": "c"}, "c": {"fallback": "b"}}}, "fallback cycle"),
])
def test_route_book_validation(section, match):
    with pytest.raises(ValueError, match=match):
        RT.RouteBook.from_config(section)


def test_route_book_parses_contracts_and_hands_out_copies():
    book = RT.RouteBook.from_config(ROUTES)
    assert RT.RouteBook.from_config(None).routes == {} and RT.RouteBook.from_config({}).routes == {}
    assert book.has("route") and not book.has("color")
    assert book.contract("route", "vector") == {"requires": "vector"}
    assert book.contract("route", None) is None and book.contract("route", "ghost") is None
    assert book.contract("color", "red") is None
    assert RT.RouteBook.from_config({"h": {"a": None, "b": {}}}).routes == {"h": {"a": {}, "b": {}}}
    c = book.contract("response_type", "direct_answer")
    c["llm"] = "huge"
    assert book.contract("response_type", "direct_answer")["llm"] == "small"
    chain = RT.RouteBook.from_config({"r": {"a": {"fallback": "b"}, "b": {"fallback": "c"}, "c": {}}})
    assert chain.contract("r", "a") == {"fallback": "b"}


def test_filter_reports_why_each_option_was_removed():
    book = RT.RouteBook.from_config(ROUTES)
    opts = ("direct_answer", "summarize", "escalate_human", "experimental", "no_contract")
    cons = RT.Constraints.from_request({"capabilities": ["vector"], "sensitivity": "restricted",
                                        "latency_budget_ms": 1000})
    kept, considered = book.filter("response_type", opts, cons)
    assert kept == ["direct_answer", "escalate_human", "no_contract"]
    assert [c["option"] for c in considered] == list(opts)
    assert all(c["kept"] == (c["reason"] is None) for c in considered)
    why = {c["option"]: c["reason"] for c in considered}
    assert why["experimental"] == "disabled by configuration"
    assert why["summarize"].startswith("needs llm_large")   # capabilities are checked before data class and latency


@pytest.mark.parametrize("request_,reason", [
    ({"capabilities": ["vector"]}, "needs llm_large, which the request does not list as available"),
    ({"capabilities": ["llm_large"], "sensitivity": "restricted"},
     "may not receive restricted data (allowed: public, internal)"),
    ({"sensitivity": ""}, "may not receive  data (allowed: public, internal)"),
    ({"latency_budget_ms": 1000}, "expected latency 4000 ms exceeds the budget 1000 ms"),
    ({"latency_budget_ms": 3999.5}, "expected latency 4000 ms exceeds the budget 3999.5 ms"),
])
def test_filter_reason_per_constraint(request_, reason):
    book = RT.RouteBook.from_config(ROUTES)
    kept, considered = book.filter("response_type", ["summarize"], RT.Constraints.from_request(request_))
    assert kept == [] and considered == [{"option": "summarize", "kept": False, "reason": reason}]


def test_filter_without_constraints_removes_only_disabled_options():
    book = RT.RouteBook.from_config(ROUTES)
    kept, _ = book.filter("response_type", ["direct_answer", "summarize", "experimental"], RT.Constraints())
    assert kept == ["direct_answer", "summarize"]
    kept, _ = book.filter("response_type", ["summarize"], RT.Constraints.from_request({"latency_budget_ms": 4000}))
    assert kept == ["summarize"]                                     # the budget is inclusive
    kept, _ = book.filter("response_type", ["summarize"], RT.Constraints.from_request({"sensitivity": "public"}))
    assert kept == ["summarize"]
    kept, _ = book.filter("route", ["graph", "hybrid", "vector"], RT.Constraints.from_request({"capabilities": []}))
    assert kept == []
    kept, _ = book.filter("route", ("graph", "hybrid"), RT.Constraints.from_request({"capabilities": ["graph",
                                                                                                     "vector"]}))
    assert kept == ["graph", "hybrid"]
    kept, considered = book.filter("unknown_head", ["a", "b"], RT.Constraints.from_request({"capabilities": []}))
    assert kept == ["a", "b"] and all(c["reason"] is None for c in considered)


def test_resolve_walks_the_fallback_chain_to_the_first_surviving_option():
    book = RT.RouteBook.from_config(ROUTES)
    kept, _ = book.filter("route", ["graph", "hybrid", "vector"], RT.Constraints.from_request(
        {"capabilities": ["vector"]}))
    assert kept == ["vector"]
    assert book.resolve("route", "graph", kept) == {               # graph -> hybrid (removed) -> vector
        "option": "graph", "contract": {"requires": ["graph"], "fallback": "hybrid"}, "fallback": "vector",
        "fallback_contract": {"requires": "vector"}}
    full = ["graph", "hybrid", "vector"]
    assert book.resolve("route", "graph", full)["fallback"] == "hybrid"
    assert book.resolve("route", "hybrid", full)["fallback"] == "vector"
    gone = book.resolve("route", "graph", [])
    assert gone["fallback"] is None and gone["fallback_contract"] is None and gone["contract"] is not None
    assert book.resolve("route", "vector", full)["fallback"] is None              # the end of the chain
    assert book.resolve("route", None, full) is None and book.resolve("color", "red", ["red"]) is None
    assert book.resolve("route", "no_contract", full) == {"option": "no_contract", "contract": None,
                                                          "fallback": None, "fallback_contract": None}
    rt = RT.RouteBook.from_config(ROUTES)
    assert rt.resolve("response_type", "summarize", ["direct_answer"])["fallback"] is None   # escalate_human removed


def test_check_head_against_the_trained_head():
    book = RT.RouteBook.from_config(ROUTES)
    book.check_head("route", HeadSpec("route", "choice", ("graph", "hybrid", "vector", "none")))
    with pytest.raises(ContractError) as e:
        book.check_head("route", HeadSpec("route", "choice", ("graph", "vector")))
    assert e.value.kind == "bad_config" and e.value.status == 500 and "['hybrid']" in e.value.message
    with pytest.raises(ContractError) as e:
        book.check_head("route", HeadSpec("route", "noul"))
    assert e.value.kind == "bad_config" and "noul head" in e.value.message


# ============================================================================= review (evaluate decision table)

def test_parse_artifact():
    assert RV.parse_artifact(None) is None
    assert RV.parse_artifact({}) == {"status": "complete"}
    a = {"status": "partial", "finish_reason": "length", "model": "m", "provider": "p", "request_digest": "sha256:x",
         "latency_ms": 12.5, "usage": {"in": 3}, "anything": ["kept"]}
    assert RV.parse_artifact(a) == a


@pytest.mark.parametrize("raw,field", [
    ("complete", "artifact"), (["complete"], "artifact"), (1, "artifact"),
    ({"status": "done"}, "artifact.status"), ({"status": None}, "artifact.status"),
    ({"status": "COMPLETE"}, "artifact.status"), ({"status": ["complete"]}, "artifact.status"),
    ({"finish_reason": 3}, "artifact.finish_reason"), ({"model": ["m"]}, "artifact.model"),
    ({"provider": {}}, "artifact.provider"), ({"request_digest": 1}, "artifact.request_digest"),
    ({"latency_ms": "12"}, "artifact.latency_ms"),
])
def test_parse_artifact_validation(raw, field):
    with pytest.raises(ContractError) as e:
        RV.parse_artifact(raw)
    assert e.value.status == 422 and e.value.details["field"] == field


def test_parse_checks():
    assert RV.parse_checks(None) is None and RV.parse_checks({}) == {}
    out = RV.parse_checks({"schema": True, "tests": False, "lint": {"passed": False, "detail": "E501 line 3"},
                           "x": {"passed": True, "detail": ""}, 7: True})
    assert out == {"schema": {"passed": True}, "tests": {"passed": False},
                   "lint": {"passed": False, "detail": "E501 line 3"}, "x": {"passed": True}, "7": {"passed": True}}


@pytest.mark.parametrize("raw", ["ok", [True], {"a": 1}, {"a": 0}, {"a": "true"}, {"a": None},
                                 {"a": {"passed": 1}}, {"a": {"passed": "yes"}}, {"a": {"detail": "x"}},
                                 {"a": {"passed": True, "detail": 3}}])
def test_parse_checks_validation(raw):
    with pytest.raises(ContractError) as e:
        RV.parse_checks(raw)
    assert e.value.status == 422


def test_parse_checks_limit():
    assert len(RV.parse_checks({f"c{i}": True for i in range(RV.MAX_CHECKS)})) == RV.MAX_CHECKS
    with pytest.raises(ContractError) as e:
        RV.parse_checks({f"c{i}": True for i in range(RV.MAX_CHECKS + 1)})
    assert e.value.details == {"field": "checks", "limit": RV.MAX_CHECKS}


def test_precheck_precedence_checks_before_artifact():
    partial = RV.parse_artifact({"status": "partial", "finish_reason": "length"})
    failed = RV.parse_checks({"schema": True, "tests": {"passed": False, "detail": "2 failing"}, "lint": False})
    out = RV.precheck(partial, failed)
    assert out["rule"] == "checks" and out["failed"] == ["tests", "lint"]
    assert out["reason"].startswith("deterministic checks failed (tests: 2 failing; lint)")
    out = RV.precheck(partial, RV.parse_checks({"schema": True}))
    assert out == {"rule": "artifact", "status": "partial",
                   "reason": "the answer is partial (finish_reason length): a partial or missing artifact never "
                             "ships; retry the generation or fall back"}
    out = RV.precheck(RV.parse_artifact({"status": "none"}), None)
    assert out["rule"] == "artifact" and out["status"] == "none" and out["reason"].startswith("no answer was produced:")
    assert RV.precheck(RV.parse_artifact({}), RV.parse_checks({"schema": True})) is None
    assert RV.precheck(RV.parse_artifact({"status": "complete"}), {}) is None and RV.precheck(None, None) is None


def test_parse_revision():
    assert RV.parse_revision(None) is None and RV.parse_revision({"of": "abc"}) == "abc"
    for bad in ({}, {"of": ""}, {"of": 3}, {"of": None}, "abc", ["abc"]):
        with pytest.raises(ContractError) as e:
            RV.parse_revision(bad)
        assert e.value.details == {"field": "revision"}


def evaluation(did="e1", *, attempt=None, failed=(), state_digest="sha256:s1", kind="evaluate") -> dict:
    receipt: dict = {"state_digest": state_digest}
    if attempt is not None:
        receipt["audit"] = {"revision": {"of": "e0", "attempt": attempt}}
    return {"decision_id": did, "kind": kind, "failed": list(failed), "receipt": receipt}


def test_revision_context():
    assert RV.revision_context(evaluation(failed=["grounded"]), "e1", "sha256:s2") == {
        "of": "e1", "attempt": 2, "previously_failed": ["grounded"], "unchanged": False}
    assert RV.revision_context(evaluation(attempt=3), "e1", "sha256:s2")["attempt"] == 4
    assert RV.revision_context(evaluation(), "e1", "sha256:s1")["unchanged"] is True
    with pytest.raises(ContractError) as e:
        RV.revision_context(None, "ghost", "sha256:s")
    assert e.value.kind == "unknown_decision" and e.value.status == 422 and e.value.details == {"field": "revision"}
    with pytest.raises(ContractError, match="is a decide, not an evaluation"):
        RV.revision_context(evaluation(kind="decide"), "e1", "sha256:s")
    # an evaluation logged before receipts: attempt 2, and never "unchanged" (nothing to compare with)
    assert RV.revision_context({"kind": "evaluate"}, "old", "sha256:s") == {
        "of": "old", "attempt": 2, "previously_failed": [], "unchanged": False}


@pytest.mark.parametrize("max_revisions,attempt,stops", [
    (2, 2, False), (2, 3, False), (2, 4, True), (0, 2, True), (1, 2, False), (1, 3, True), (5, 6, False),
    (5, 7, True)])
def test_revision_budget_arithmetic(max_revisions, attempt, stops):
    ctx = {"of": "e", "attempt": attempt, "previously_failed": [], "unchanged": False}
    why = RV.revision_verdict(ctx, ["grounded"], max_revisions)
    assert (why is not None) is stops
    if stops:
        assert f"attempt {attempt}, after {attempt - 1} revisions" in why
        assert f"evaluate.max_revisions = {max_revisions}" in why


def test_revision_budget_along_a_chain_of_evaluations():
    """Attempt 1 is the first evaluation; max_revisions = 2 allows attempts 2 and 3 and refuses 4 and later."""
    prev = evaluation("e1", failed=["p1"], state_digest="sha256:s1")
    verdicts = {}
    for n in range(2, 6):
        ctx = RV.revision_context(prev, prev["decision_id"], f"sha256:s{n}")
        verdicts[ctx["attempt"]] = RV.revision_verdict(ctx, [f"p{n}"], 2)
        prev = evaluation(f"e{n}", attempt=ctx["attempt"], failed=[f"p{n}"], state_digest=f"sha256:s{n}")
    assert list(verdicts) == [2, 3, 4, 5]
    assert verdicts[2] is None and verdicts[3] is None
    assert "budget exhausted" in verdicts[4] and "budget exhausted" in verdicts[5]


def test_revision_verdict_rules_and_their_precedence():
    assert RV.revision_verdict(None, ["x"], 2) is None
    ctx = {"of": "e1", "attempt": 2, "previously_failed": ["grounded", "complete"], "unchanged": False}
    why = RV.revision_verdict(ctx, ["safe", "grounded", "complete"], 2)
    assert why.startswith("complete, grounded failed in two consecutive attempts")
    assert RV.revision_verdict(ctx, ["safe"], 2) is None and RV.revision_verdict(ctx, [], 2) is None
    # an unchanged state stops the loop first, even within budget and with nothing failing
    assert "sent exactly the state of evaluation e1" in RV.revision_verdict({**ctx, "unchanged": True}, [], 2)
    assert "sent exactly the state" in RV.revision_verdict({**ctx, "unchanged": True, "attempt": 9}, ["grounded"], 2)
    # then the budget, then a repeated failure
    assert "budget exhausted" in RV.revision_verdict({**ctx, "attempt": 4}, ["grounded"], 2)


def test_max_revisions_from_the_project(tmp_path):
    def project(v):
        return Project(root=tmp_path, data={"evaluate": {"max_revisions": v}})

    assert RV.max_revisions(Project(root=tmp_path)) == RV.DEFAULT_MAX_REVISIONS == 2
    assert RV.max_revisions(project(0)) == 0 and RV.max_revisions(project("3")) == 3
    for bad in (-1, "many", None, [2]):
        with pytest.raises(ContractError) as e:
            RV.max_revisions(project(bad))
        assert e.value.kind == "bad_config" and e.value.status == 500


# ============================================================================= policy overrides

OV = {"needs_graph": {"true": {"*": {"act": 0.9}, "high": {"act": 0.99, "confirm": 0.9}}},
      "color": {"blue": {"low": {"act": 0.95, "min_accuracy": 0.8}}}}


def test_level_for_merges_star_then_the_stakes_level():
    cfg = P.PolicyConfig(overrides=OV)
    base = cfg.level("medium")
    assert cfg.level_for("needs_graph", "true", "low") == P.StakesLevel(0.9, 0.5, 0.6)
    assert cfg.level_for("needs_graph", "true", "medium") == P.StakesLevel(0.9, 0.65, 0.75)
    assert cfg.level_for("needs_graph", "true", "high") == P.StakesLevel(0.99, 0.9, 0.9)
    assert cfg.level_for("needs_graph", "true", None) == P.StakesLevel(0.9, 0.65, 0.75)     # default: medium
    assert cfg.level_for("needs_graph", "false", "medium") == base                          # another class
    assert cfg.level_for("color", "blue", "low") == P.StakesLevel(0.95, 0.5, 0.8)
    assert cfg.level_for("color", "blue", "medium") == base                                 # only at low
    assert cfg.level_for("other", "true", "medium") == base and cfg.level_for(None, None, "medium") == base
    assert cfg.level_for("needs_graph", None, "medium") == base
    with pytest.raises(P.UnknownStakes) as e:
        cfg.level_for("needs_graph", "true", "extreme")
    assert e.value.status == 422 and isinstance(e.value, ValueError) and e.value.details == {
        "stakes": ["low", "medium", "high"]}


@pytest.mark.parametrize("ov,match", [
    (["needs_graph"], "maps head"),
    ({"needs_graph": ["true"]}, "maps a class"),
    ({"needs_graph": {"true": 0.9}}, "maps a stakes level"),
    ({"needs_graph": {"true": {"extreme": {"act": 0.9}}}}, "unknown stakes level 'extreme'"),
    ({"needs_graph": {"true": {"*": {}}}}, "sets some of"),
    ({"needs_graph": {"true": {"*": 0.9}}}, "sets some of"),
    ({"needs_graph": {"true": {"*": {"act": 0.9, "acts": 0.95}}}}, "sets some of"),
    ({"needs_graph": {"true": {"*": {"act": 0.6}}}}, "at medium stakes"),        # below medium's confirm (0.65)
    ({"needs_graph": {"true": {"high": {"confirm": 0.97}}}}, "at high stakes"),   # above high's act (0.95)
    ({"needs_graph": {"true": {"*": {"act": 1.2}}}}, "confirm <= act <= 1"),
    ({"needs_graph": {"true": {"*": {"confirm": -0.1}}}}, "confirm <= act <= 1"),
    ({"needs_graph": {"true": {"*": {"min_accuracy": 1.5}}}}, "min_accuracy"),
    ({"needs_graph": {"true": {"*": {"act": float("nan")}}}}, "confirm <= act <= 1"),
])
def test_override_validation(ov, match):
    with pytest.raises(ValueError, match=match):
        P.PolicyConfig(overrides=ov)


def test_overrides_reach_custom_stakes_levels_and_round_trip():
    cfg = P.PolicyConfig.from_dict({"stakes": {"vip": {"act": 0.97, "confirm": 0.9, "min_accuracy": 0.95}},
                                    "overrides": {"color": {"red": {"vip": {"act": 0.99}}}}})
    assert cfg.level_for("color", "red", "vip") == P.StakesLevel(0.99, 0.9, 0.95)
    assert P.PolicyConfig.from_dict(cfg.to_dict()) == cfg
    assert P.PolicyConfig.from_dict(json.loads(json.dumps(cfg.to_dict()))) == cfg
    with pytest.raises(ValueError, match="unknown stakes level 'vip'"):
        P.PolicyConfig(overrides={"color": {"red": {"vip": {"act": 0.99}}}})
    assert P.PolicyConfig(overrides=None).overrides == {} and P.PolicyConfig().overrides == {}


def test_class_tiers_use_the_committed_class_thresholds():
    cfg = P.PolicyConfig(overrides={"needs_graph": {"true": {"*": {"act": 0.95}}}})
    probs = np.array([[0.1, 0.9], [0.9, 0.1], [0.02, 0.98], [0.3, 0.7]])
    c, p = P.chosen(SYM, probs)
    assert c.tolist() == [1, 0, 1, 1]
    assert P.class_tiers(SYM, c, p, cfg, "medium").tolist() == [1, 2, 2, 1]      # 0.9 on "true" needs 0.95
    assert P.class_tiers(SYM, c, p, P.PolicyConfig(), "medium").tolist() == [2, 2, 2, 1]
    assert P.class_tiers(SYM, np.array([], int), np.array([]), cfg, "medium").tolist() == []
    # a conformal threshold still only restricts act
    assert P.class_tiers(SYM, c, p, cfg, "medium", conformal_threshold=0.99).tolist() == [1, 1, 1, 1]


def test_evaluate_counts_tiers_with_the_class_thresholds():
    cfg = P.PolicyConfig(overrides={"needs_graph": {"true": {"*": {"act": 0.95}}}})
    probs = np.array([[0.1, 0.9]] * 10 + [[0.9, 0.1]] * 10)
    y = np.array([1] * 10 + [0] * 10)
    ev = P.evaluate(SYM, probs, y, cfg)
    for stakes in ("low", "medium"):
        assert ev[stakes]["tiers"] == {"escalate": 0, "confirm": 10, "act": 10}
        assert ev[stakes]["act"]["n"] == 10 and ev[stakes]["act_or_confirm"]["n"] == 20
    assert ev["high"]["tiers"] == {"escalate": 0, "confirm": 20, "act": 0}
    assert P.evaluate(SYM, probs, y, P.PolicyConfig())["medium"]["tiers"]["act"] == 20


def test_decide_returns_and_applies_the_overridden_thresholds():
    cfg = P.PolicyConfig(overrides={"needs_graph": {"true": {"*": {"act": 0.95}}}})
    led = strong_ledger(SYM.labels)
    d = P.decide(SYM, np.array([0.1, 0.9]), cfg, stakes="medium", ledger=led)
    assert d["class"] == "true" and d["tier"] == "confirm" and d["uncapped_tier"] == "confirm"
    assert d["thresholds"] == {"act": 0.95, "confirm": 0.65, "min_accuracy": 0.75, "conformal": None}
    d = P.decide(SYM, np.array([0.9, 0.1]), cfg, stakes="medium", ledger=led)
    assert d["class"] == "false" and d["tier"] == "act" and d["thresholds"]["act"] == 0.85
    # a gate head always commits to `true`: its override applies to every decision
    gate_cfg = P.PolicyConfig(overrides={"answerable": {"true": {"*": {"act": 0.99}}}})
    d = P.decide(GATE, np.array([0.03, 0.97]), gate_cfg, stakes="low", ledger=strong_ledger(GATE.labels))
    assert d["tier"] == "confirm" and d["thresholds"]["act"] == 0.99
    # min_accuracy reaches the Bernoulli gate: 400 of 404 right is not surely above 99.5%
    strict = P.PolicyConfig(overrides={"color": {"red": {"*": {"min_accuracy": 0.995}}}})
    led = strong_ledger(COLOR.labels)
    d = P.decide(COLOR, np.array([0.95, 0.03, 0.02]), strict, stakes="medium", ledger=led)
    assert d["gate"]["required_accuracy"] == 0.995 and not d["gate"]["passed"] and d["tier"] == "confirm"
    d = P.decide(COLOR, np.array([0.03, 0.95, 0.02]), strict, stakes="medium", ledger=led)
    assert d["gate"]["required_accuracy"] == 0.75 and d["tier"] == "act"
    with pytest.raises(P.UnknownStakes):
        P.decide(COLOR, np.array([0.95, 0.03, 0.02]), strict, stakes="extreme", ledger=led)


@pytest.mark.parametrize("parse", [
    lambda: RD.kinds_from_config(0),
    lambda: BA.load({"b": {"version": "1", "questions": {"q": {"type": "noul"}}, "changelog": 5}}, None),
    lambda: P.PolicyConfig(overrides={"h": {"c": {"*": {"act": None}}}}),
    lambda: P.PolicyConfig(overrides={"h": {"c": {"*": {"act": [0.9]}}}}),
], ids=["serve.redact", "battery.changelog", "override-null", "override-list"])
def test_wrong_typed_configuration_values_are_value_errors(parse):
    with pytest.raises(ValueError):
        parse()


def test_a_boolean_class_key_reaches_the_noul_class_or_is_refused():
    ov = yaml.safe_load("needs_graph: {true: {'*': {act: 0.95}}}")
    assert list(ov["needs_graph"]) == [True]
    try:
        cfg = P.PolicyConfig(overrides=ov)
    except ValueError:
        return                                                    # refusing it is a fix as well
    assert cfg.level_for("needs_graph", "true", "medium").act == 0.95


# ============================================================================= contract: allowed, criteria, digest

def test_allowed_is_parsed_for_choice_questions():
    assert Question("c", {"type": "choice"}).allowed is None
    assert Question("c", {"type": "choice", "allowed": ["red", "blue"]}).allowed == ("red", "blue")
    many = [f"o{i}" for i in range(MAX_CHOICE_OPTIONS)]
    assert len(Question("c", {"type": "choice", "allowed": many}).allowed) == MAX_CHOICE_OPTIONS


@pytest.mark.parametrize("raw,match", [
    ({"type": "noul", "allowed": ["true"]}, "a noul question has none"),
    ({"type": "score", "allowed": ["0"]}, "a score question has none"),
    ({"type": "choice", "allowed": []}, "non-empty array"),
    ({"type": "choice", "allowed": "red"}, "non-empty array"),
    ({"type": "choice", "allowed": ["red", ""]}, "non-empty array"),
    ({"type": "choice", "allowed": ["red", 1]}, "non-empty array"),
    ({"type": "choice", "allowed": ["red", None]}, "non-empty array"),
    ({"type": "choice", "allowed": {"red": True}}, "non-empty array"),
    ({"type": "choice", "allowed": ("red",)}, "non-empty array"),
    ({"type": "choice", "allowed": ["red", "red"]}, "twice"),
    ({"type": "choice", "allowed": [f"o{i}" for i in range(MAX_CHOICE_OPTIONS + 1)]}, "more than"),
])
def test_allowed_refusals(raw, match):
    with pytest.raises(ContractError, match=match) as e:
        Question("c", raw)
    assert e.value.status == 422 and e.value.details["field"] == "allowed" and e.value.details["question"] == "c"


def test_bind_checks_allowed_against_the_head_classes():
    assert Question("color", {"type": "choice", "allowed": ["blue", "red"]}).bind(COLOR) == []
    with pytest.raises(ContractError) as e:
        Question("color", {"type": "choice", "allowed": ["red", "purple", "Red"]}).bind(COLOR)
    assert "['purple', 'Red']" in e.value.message and e.value.details["classes"] == ["red", "green", "blue"]


def test_restrict_zeroes_removed_options_without_renormalising():
    labels = ["red", "green", "blue"]
    ps, removed, mass = restrict(labels, [0.5, 0.3, 0.2], ["green", "blue"])
    assert ps == [0.0, 0.3, 0.2] and removed == {"red": 0.5} and mass == pytest.approx(0.5)
    assert sum(ps) + mass == pytest.approx(1.0)
    assert restrict(labels, [0.5, 0.3, 0.2], None) == ([0.5, 0.3, 0.2], {}, 0.0)
    assert restrict(labels, [0.5, 0.3, 0.2], labels) == ([0.5, 0.3, 0.2], {}, 0.0)
    assert restrict(labels, np.array([0.5, 0.3, 0.2]), ("red",)) == ([0.5, 0.0, 0.0], {"green": 0.3, "blue": 0.2},
                                                                     pytest.approx(0.5))
    assert restrict(labels, [0.5, 0.3, 0.2], ["purple"])[2] == pytest.approx(1.0)    # bind refuses this earlier
    with pytest.raises(ValueError):
        restrict(labels, [0.5, 0.5], ["red"])


def test_choice_answer_with_allowed_keeps_the_calibrated_probabilities():
    a = choice_answer(["red", "green", "blue"], [0.6, 0.3, 0.1], head="color", calibrated=True,
                      allowed=["blue", "green"])
    assert a["choice"] == "green"                                       # the removed argmax is never chosen
    assert a["probabilities"] == {"green": 0.3, "blue": 0.1}            # not renormalised
    assert a["allowed"] == ["green", "blue"]                            # in the head's option order
    assert a["excluded_mass"] == pytest.approx(0.6)
    assert a["confidence"] == pytest.approx(choice_confidence([0.75, 0.25]))
    assert set(a) == {"type", "choice", "probabilities", "confidence", "head", "calibrated", "allowed",
                      "excluded_mass"}


def test_restrict_and_choice_answer_invariants_on_random_inputs():
    rng = np.random.default_rng(5)
    labels = [f"o{i}" for i in range(8)]
    for _ in range(400):
        k = int(rng.integers(2, 9))
        opts = labels[:k]
        ps = rng.dirichlet(np.ones(k) * rng.uniform(0.1, 3))
        allowed = [o for o in opts if rng.random() < 0.5] or [opts[int(rng.integers(k))]]
        kept, removed, mass = restrict(opts, ps, allowed)
        assert kept == [float(p) if o in allowed else 0.0 for o, p in zip(opts, ps, strict=True)]
        assert set(removed) == set(opts) - set(allowed) and mass == pytest.approx(sum(removed.values()))
        assert sum(kept) + mass == pytest.approx(1.0)
        a = choice_answer(opts, ps, head="h", calibrated=True, allowed=allowed)
        assert a["choice"] in allowed and a["allowed"] == list(a["probabilities"]) == [o for o in opts
                                                                                     if o in allowed]
        assert float(ps[opts.index(a["choice"])]) == max(float(ps[opts.index(o)]) for o in allowed)
        assert a["excluded_mass"] == pytest.approx(mass, abs=2e-6) and 0.0 <= a["confidence"] <= 1.0
        for o in allowed:                                               # each kept value is the calibrated one
            assert a["probabilities"][o] == pytest.approx(float(ps[opts.index(o)]), abs=1e-6)


def test_choice_answer_with_allowed_edge_cases():
    zero = choice_answer(["a", "b", "c"], [1.0, 0.0, 0.0], head="h", calibrated=True, allowed=["c", "b"])
    assert zero["choice"] == "b" and zero["confidence"] == 0.0 and zero["excluded_mass"] == 1.0
    assert zero["probabilities"] == {"b": 0.0, "c": 0.0}
    tie = choice_answer(["a", "b", "c"], [0.2, 0.4, 0.4], head="h", calibrated=True, allowed=["c", "b"])
    assert tie["choice"] == "b"                                         # ties go to the first option in head order
    full = choice_answer(["a", "b"], [0.7, 0.3], head="h", calibrated=True, allowed=["a", "b"])
    assert full["excluded_mass"] == 0.0 and full["choice"] == "a" and full["probabilities"] == {"a": 0.7, "b": 0.3}
    pre = choice_answer(["a", "b", "c"], [0.0, 0.3, 0.2], head="h", calibrated=True, allowed=["b", "c"])
    assert pre["excluded_mass"] == pytest.approx(0.5)                   # already restricted probabilities: the same


def test_build_answer_uses_the_questions_allowed_unless_told_otherwise():
    q = Question("color", {"type": "choice", "allowed": ["green", "blue"]})
    a = build_answer(q, COLOR, [0.6, 0.3, 0.1], True)
    assert a["choice"] == "green" and a["allowed"] == ["green", "blue"]
    b = build_answer(q, COLOR, [0.6, 0.3, 0.1], True, allowed=["blue"])
    assert b["choice"] == "blue" and b["excluded_mass"] == pytest.approx(0.9)
    c = build_answer(None, COLOR, [0.6, 0.3, 0.1], True)
    assert "allowed" not in c and c["choice"] == "red"


def test_score_criteria_reordered_is_a_criteria_order_error():
    with pytest.raises(ContractError) as e:
        Question("level", {"type": "score", "criteria": ["high", "mid", "low", "none"]}).bind(LEVEL)
    assert e.value.kind == "criteria_order" and e.value.status == 422
    assert e.value.details["levels"] == ["none", "low", "mid", "high"] and e.value.details["head"] == "level"
    # one swapped pair counts, and case or whitespace do not hide it
    with pytest.raises(ContractError) as e:
        Question("level", {"type": "score", "criteria": ["None", "  low", "HIGH", "mid "]}).bind(LEVEL)
    assert e.value.kind == "criteria_order"
    with pytest.raises(ContractError) as e:
        Question("stars", {"type": "score", "criteria": ["three", "two", "one"]}).bind(STARS)
    assert e.value.kind == "criteria_order"


def test_score_criteria_wording_differences_warn():
    assert Question("level", {"type": "score", "criteria": ["None", "low ", " MID", "high"]}).bind(LEVEL) == []
    notes = Question("level", {"type": "score", "criteria": ["nothing", "a bit", "medium", "a lot"]}).bind(LEVEL)
    assert len(notes) == 1 and "criteria differ" in notes[0]
    assert "{0: 'none', 1: 'low', 2: 'mid', 3: 'high'}" in notes[0]
    assert Question("stars", {"type": "score", "criteria": ["one", "two", "three"]}).bind(STARS) == []
    with pytest.raises(ContractError, match="has 4 levels, criteria has 3"):
        Question("level", {"type": "score", "criteria": ["a", "b", "c"]}).bind(LEVEL)


def test_score_criteria_listing_the_level_names_in_another_order_is_refused():
    q = Question("level", {"type": "score", "criteria": ["3", "2", "1", "0"]})
    with pytest.raises(ContractError) as e:
        q.bind(LEVEL)
    assert e.value.kind == "criteria_order"


def _qd(**raw) -> str:
    return questions_digest([Question("q", {"type": "choice", "head": "color", **raw})])


def test_questions_digest_tracks_meaning_not_instructions():
    d0 = _qd()
    assert _qd(instructions="Pick the colour, carefully.") == d0 == _qd(instructions="") == _qd(instructions=None)
    for kw in ({"criteria": {"red": "R", "green": "G", "blue": "B"}}, {"allowed": ["red"]}, {"path": "request"},
               {"inputs": {"q": "request.text"}}, {"head": "colour"}, {"option": "red"}):
        assert _qd(**kw) != d0, kw
    assert _qd(allowed=["red", "blue"]) == _qd(allowed=["blue", "red"])          # a set of options
    assert _qd(criteria={"red": "R", "green": "G"}) == _qd(criteria={"green": "G", "red": "R"})
    assert _qd(criteria={"red": "R", "green": "G"}) != _qd(criteria={"red": "R", "green": "Green"})
    assert _qd(inputs={"a": "x", "b": "y"}) == _qd(inputs={"b": "y", "a": "x"})
    assert _qd(inputs={"a": "x", "b": "y"}) != _qd(inputs={"a": "y", "b": "x"})
    assert _qd(path="rag.0") == _qd(path=" rag.0 ") and _qd(path="rag.0") != _qd(path="rag.1")


def test_questions_digest_of_scores_and_of_question_sets():
    def s(criteria):
        return questions_digest([Question("s", {"type": "score", "criteria": criteria})])

    assert s(["low", "high"]) != s(["high", "low"])                             # score levels are ordered
    a, b = Question("a", {"type": "noul"}), Question("b", {"type": "noul"})
    assert questions_digest([a, b]) == questions_digest([b, a])                 # declaration order is not meaning
    assert questions_digest([a]) != questions_digest([Question("a2", {"type": "noul", "head": "a"})])   # ids are
    assert questions_digest([a]) == questions_digest([Question("a", {"type": "noul", "head": "a"})])
    assert questions_digest([a]) != questions_digest([a, b])


# ============================================================================= contract: request bodies

def test_systemone_request_needs_questions_or_a_battery_but_not_both():
    with pytest.raises(ValidationError, match="questions is required"):
        SystemOneRequest(state={})
    with pytest.raises(ValidationError, match="not both"):
        SystemOneRequest(state={}, questions={"q": {"type": "noul"}}, battery="preflight")
    with pytest.raises(ValidationError, match="not both"):
        SystemOneRequest(questions={}, battery="preflight")
    r = SystemOneRequest(state={}, battery="preflight@1")
    assert r.questions is None and r.strict is False and r.receipt is False
    r = SystemOneRequest.model_validate({"state": {}, "questions": {"q": {"type": "noul"}}, "strict": True,
                                         "receipt": True, "unknown": 1})
    assert r.battery is None and r.strict and r.receipt
    with pytest.raises(ValidationError):
        SystemOneRequest(battery="x" * 101)
    assert SystemOneRequest(battery="x" * 100).battery == "x" * 100


def test_new_request_bodies_validate_their_bounds():
    ok = OutcomeRequest(decision_id="d1", task_success=True, latency_ms=0, cost=0.0,
                        retries={"transport": 1, "workflow": 0}, tokens={"input": 10, "output": 2.5})
    assert ok.fallback is None and ok.retries == {"transport": 1, "workflow": 0}
    for bad in ({"decision_id": ""}, {"decision_id": "d", "latency_ms": -1}, {"decision_id": "d", "cost": -0.01},
                {}, {"decision_id": "d", "tags": ["t"] * 21}, {"decision_id": "d", "retries": {"transport": "x"}}):
        with pytest.raises(ValidationError):
            OutcomeRequest(**bad)
    assert ReceiptCheckRequest().state is None and ReceiptCheckRequest(state_digest="sha256:x").max_age_s is None
    for bad in ({"max_age_s": 0}, {"max_age_s": -1}, {"state_digest": "x" * 101}):
        with pytest.raises(ValidationError):
            ReceiptCheckRequest(**bad)
    with pytest.raises(ValidationError):
        AutomationRequest()
    assert AutomationRequest(enabled=False, reason="incident").enabled is False
    with pytest.raises(ValidationError):
        AutomationRequest(enabled=True, reason="x" * 501)
    fb = FeedbackRequest(source="review", label="true", tags=["regression"], note="n", missing_evidence=["f1"])
    assert fb.tags == ["regression"] and fb.missing_evidence == ["f1"]
    for bad in ({"tags": ["t"] * 21}, {"note": "x" * 2001}, {"missing_evidence": ["f"] * 51}):
        with pytest.raises(ValidationError):
            FeedbackRequest(source="review", **bad)
