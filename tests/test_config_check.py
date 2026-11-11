"""rlcd.yaml validation (core/config_check.py, `rlcd config check | show`): a wrong key or value is reported at
its path, in plain words, before anything runs; a valid file is never refused. Offline: no file but rlcd.yaml is
read, the documents pack is only asked for its head specs."""
from __future__ import annotations

import copy
import dataclasses
import importlib.util
import json
import re
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from open_rlcd.cli import _config_ok, main
from open_rlcd.core import battery as BT
from open_rlcd.core import checklist as CL
from open_rlcd.core import config_check as CC
from open_rlcd.core import policy as PL
from open_rlcd.core.config import Project, load_project
from open_rlcd.core.contract import ContractError, TrainRequest
from open_rlcd.core.head import HeadSpec
from open_rlcd.core.routes import RESERVED, RouteBook
from open_rlcd.core.service import RECEIPT_MODES, Service
from open_rlcd.packs.base import load_pack
from open_rlcd.packs.tabular.data import BASE_KEYS, SPEC_KEYS, TARGET_KEYS, TOLERATED_KEYS
from tests.toypack import ToyPack

REPO = Path(__file__).resolve().parent.parent
DOC = REPO / "docs" / "configuration.md"
TABULAR = {"pack": "tabular", "pack_config": {"data": "data.csv", "targets": {"churned": {"label": "churned"}}}}


def check(tmp_path, data: dict, pack=None) -> list[dict]:
    return CC.check_project(Project(root=tmp_path, data=data), pack)


def at(findings: list[dict], path: str, level: str = "error") -> str:
    """The message of the one finding at `path` (asserting it is there, once, at that level)."""
    hits = [f for f in findings if f["path"] == path]
    assert len(hits) == 1, (path, findings)
    assert hits[0]["level"] == level, hits[0]
    return hits[0]["message"]


def write(tmp_path, data: dict | str, name: str = "p") -> Path:
    root = tmp_path / name
    root.mkdir(parents=True, exist_ok=True)
    text = data if isinstance(data, str) else yaml.safe_dump(data, sort_keys=False)
    (root / "rlcd.yaml").write_text(text, encoding="utf-8")
    return root


# ----------------------------------------------------------------------------- valid files are never refused

@pytest.mark.parametrize("pack", ["documents", "tabular"])
def test_what_rlcd_init_writes_has_no_findings(tmp_path, pack, capsys):
    assert main(["init", str(tmp_path / pack), "--pack", pack]) == 0
    project, findings = CC.check_file(tmp_path / pack)
    assert project is not None and project.pack == pack
    assert findings == []
    capsys.readouterr()
    assert main(["--project", str(tmp_path / pack), "config", "check"]) == 0
    assert "no problems found" in capsys.readouterr().out


def test_the_default_project_and_an_empty_file_have_no_findings(tmp_path):
    assert check(tmp_path, {}) == []
    root = write(tmp_path, "# nothing yet\n")
    assert CC.check_file(root)[1] == []


def test_values_the_code_accepts_are_accepted(tmp_path):
    data = {
        "name": 2026, "home": None, "pack": "documents",
        "train": {"holdout_fraction": "0.25", "n_splits": 4.0, "seed": 11, "feedback_weight": 1000},
        "labeling": {"judge": "ollama", "model": "qwen2.5", "timeout": None, "batch": "10"},
        "serve": {"port": 0, "redact": ["secrets", "pii"], "receipts": None, "log_states": None,
                  "review_capacity_per_day": 12.5},
        "evaluate": {"output_scan": False, "max_revisions": 0},
        "policy": {"stakes": {"low": {"act": 0.6}, "vip": {"act": 0.99, "confirm": 0.9, "min_accuracy": 0.95}},
                   "overrides": {"needs_graph": {True: {"*": {"act": 0.9}}}}, "abstain": "hold"},
        "heads": {"answerable": {"stakes": "vip", "conformal": 0.05, "model": "auto"}, "grounded": None,
                  "needs_graph": {"enabled": False}},
        "rollout": {"mode": "limited", "limited": {"stakes": "low", "heads": ["route"]},
                    "heads": {"needs_graph": "shadow"}},
        "routes": {"route": {"graph": {"requires": "graph", "fallback": "vector", "llm": "small"}, "vector": None}},
        "batteries": {"mine": {"version": "2", "owner": "qa", "changelog": "2: first",
                               "questions": {"enough": {"type": "noul", "head": "answerable"}}}},
        "owners": {"pack": "a", "heads": ["b", "c"], "policy": "d", "review": "e", "incidents": "f"},
        "regressions": "cases/regressions.jsonl", "security": {"allow_remote": False},
        "pack_config": {"encoder": {"kind": "hashing", "dim": 256}, "graph": {"kind": "local", "terms": ["Velobox"]},
                        "vector": {"backend": "numpy"}, "retrieve_k": 8, "max_evidence_age_days": 30},
    }
    assert check(tmp_path, data) == []


def test_the_example_projects_have_no_errors(tmp_path):
    spec = importlib.util.spec_from_file_location("tabular_example", REPO / "examples" / "tabular" / "run.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    for ordering in ("time", "row"):
        cfg = mod.write_project(tmp_path, ordering)
        assert CC.errors(CC.check_file(cfg)[1]) == []
    control_layer = {
        "pack": "documents", "pack_config": {"encoder": {"kind": "hashing", "dim": 256}}, "rollout": {"mode": "full"},
        "routes": {"route": {"graph": {"requires": ["graph"], "fallback": "hybrid"},
                             "hybrid": {"requires": ["vector", "graph"], "fallback": "vector"},
                             "vector": {"requires": ["vector"], "context_budget_tokens": 2000}}},
        "owners": {"pack": "search-team", "heads": "search-team", "policy": "risk", "review": "support",
                   "incidents": "on-call"}}
    assert check(tmp_path, control_layer) == []


# ----------------------------------------------------------------------------- unknown keys

def test_a_misspelled_key_is_an_error_with_a_suggestion(tmp_path):
    f = check(tmp_path, {"polcy": {}, "train": {"holdout_fracton": 0.3}, "serve": {"prot": 1}, "n_splits": 3,
                         "zzz": 1})
    assert "Did you mean 'policy'?" in at(f, "polcy")
    assert "Did you mean 'holdout_fraction'?" in at(f, "train.holdout_fracton")
    assert "Did you mean 'port'?" in at(f, "serve.prot")
    assert "Did you mean 'train.n_splits'?" in at(f, "n_splits")          # a section's key written at the top
    assert "the keys here are: name, pack, home" in at(f, "zzz")
    assert len(CC.errors(f)) == 5


def test_a_section_that_is_not_a_mapping_is_an_error(tmp_path):
    f = check(tmp_path, {"train": 5, "heads": ["answerable"], "rollout": "shadow"})
    assert "must be a mapping" in at(f, "train") and "must be a mapping" in at(f, "heads")
    assert "must be a mapping" in at(f, "rollout")


# ----------------------------------------------------------------------------- types, enums, ranges

BAD_VALUES = [
    ({"rollout": {"mode": "shaddow"}}, "rollout.mode", "Did you mean 'shadow'?"),
    ({"rollout": {"mode": False}}, "rollout.mode", "quote it"),
    ({"rollout": {"mode": None}}, "rollout.mode", "is empty"),
    ({"rollout": {"heads": {"route": "ful"}}}, "rollout.heads.route", "Did you mean 'full'?"),
    ({"rollout": {"limited": {"stakes": {"low": True}}}}, "rollout.limited.stakes", "a name or a list of names"),
    ({"policy": {"abstain": "act"}}, "policy.abstain", "cannot be 'act'"),
    ({"policy": {"abstain": True}}, "policy.abstain", "must be text"),
    ({"policy": {"stakes": {"medium": {"act": 1.2}}}}, "policy.stakes.medium.act", "must be in [0, 1]"),
    ({"policy": {"stakes": {"medium": {"act": "0.9"}}}}, "policy.stakes.medium.act", "without quotes"),
    ({"policy": {"stakes": {"medium": {"confirm": 0.9}}}}, "policy.stakes.medium", "must not be above act"),
    ({"policy": {"stakes": {"medium": {"act": 0.5, "confirm": 0.6}}}}, "policy.stakes.medium", "confirm (0.6)"),
    ({"policy": {"stakes": {"vip": {"act": 0.99}}}}, "policy.stakes.vip", "missing: confirm, min_accuracy"),
    ({"policy": {"stakes": {"medium": {"actt": 0.9}}}}, "policy.stakes.medium.actt", "Did you mean 'act'?"),
    ({"policy": {"min_verify_rows": -1}}, "policy.min_verify_rows", "must be >= 0"),
    ({"policy": {"gate_min_p_above": 1.5}}, "policy.gate_min_p_above", "must be in [0, 1]"),
    ({"policy": {"overrides": {"route": {"graph": {"vip": {"act": 0.9}}}}}}, "policy.overrides",
     "unknown stakes level 'vip'"),
    ({"train": {"holdout_fraction": 0}}, "train.holdout_fraction", "must be in [0.05, 0.5], got 0"),
    ({"train": {"holdout_fraction": 1}}, "train.holdout_fraction", "must be in [0.05, 0.5], got 1"),
    ({"train": {"holdout_fraction": "a fifth"}}, "train.holdout_fraction", "must be a number"),
    ({"train": {"holdout_fraction": None}}, "train.holdout_fraction", "is empty"),
    ({"train": {"n_splits": 1}}, "train.n_splits", "must be in [2, 20], got 1"),
    ({"train": {"n_splits": 2.5}}, "train.n_splits", "whole number"),
    ({"train": {"n_splits": True}}, "train.n_splits", "whole number"),
    ({"train": {"ordering": "tme"}}, "train.ordering", "Did you mean 'time'?"),
    ({"train": {"calibration": "isotonik"}}, "train.calibration", "Did you mean 'isotonic'?"),
    ({"train": {"feedback_weight": 0}}, "train.feedback_weight", "must be in (0, 1000]"),
    ({"train": {"gap": -1}}, "train.gap", "must be >= 0"),
    ({"serve": {"port": 70000}}, "serve.port", "must be in [0, 65535]"),
    ({"serve": {"port": "http"}}, "serve.port", "whole number"),
    ({"serve": {"receipts": "al"}}, "serve.receipts", "Did you mean 'all'?"),
    ({"serve": {"redact": ["secret"]}}, "serve.redact", "Did you mean 'secrets'?"),
    ({"serve": {"redact": 3}}, "serve.redact", "true, false or a list of secrets / pii"),
    ({"serve": {"log_states": "false"}}, "serve.log_states", "must be true or false"),
    ({"serve": {"review_capacity_per_day": 0}}, "serve.review_capacity_per_day", "must be > 0"),
    ({"serve": {"host": None}}, "serve.host", "is empty"),
    ({"security": {"allow_remote": "no"}}, "security.allow_remote", "must be true or false"),
    ({"evaluate": {"max_revisions": -1}}, "evaluate.max_revisions", "must be >= 0"),
    ({"evaluate": {"output_scan": ["pi"]}}, "evaluate.output_scan", "Did you mean 'pii'?"),
    ({"labeling": {"judge": "antropic"}}, "labeling.judge", "Did you mean 'anthropic'?"),
    ({"labeling": {"judge": "ollama"}}, "labeling.judge", "ollama needs a model"),
    ({"labeling": {"judge": "openai:gpt"}}, "labeling.judge", "a base URL and a model"),
    ({"labeling": {"batch": 0}}, "labeling.batch", "must be >= 1"),
    ({"regressions": ["a.jsonl"]}, "regressions", "must be text"),
    ({"owners": {"review": {"name": "x"}}}, "owners.review", "a name or a list of names"),
    ({"owners": "everyone"}, "owners", "must be a mapping"),
    ({"routes": {"route": {"vector": {"latency_ms": "fast"}}}}, "routes.route.vector.latency_ms", "must be a number"),
    ({"routes": {"route": {"vector": {"fallback": "nowhere"}}}}, "routes.route.vector.fallback",
     "must name another option"),
    ({"routes": {"route": {}}}, "routes.route", "maps each option"),
    ({"batteries": {"b": {"versoin": "1"}}}, "batteries.b.versoin", "Did you mean 'version'?"),
    ({"batteries": {"b": {"version": 1.1, "questions": {"q": {"type": "noul", "head": "answerable"}}}}},
     "batteries", "quote the version"),
    ({"batteries": {"b": {"version": "1", "questions": {"q": {"type": "maybe"}}}}}, "batteries", "type must be"),
    ({"pack": "documnts"}, "pack", "Did you mean 'documents'?"),
    ({"pack": ["documents"]}, "pack", "must be text"),
    ({"home": {"path": "var"}}, "home", "must be text"),
]


@pytest.mark.parametrize("data, path, fragment", BAD_VALUES, ids=[f"{p}:{i}" for i, (_, p, _) in enumerate(BAD_VALUES)])
def test_a_wrong_value_is_one_error_at_its_path(tmp_path, data, path, fragment):
    f = check(tmp_path, data)
    assert fragment in at(f, path)
    assert len(CC.errors(f)) == 1, f                      # one mistake, one error: no echo from a second validator


def test_names_that_match_nothing_are_warnings(tmp_path):
    f = check(tmp_path, {"rollout": {"mode": "limited", "limited": {"stakes": ["lo"], "heads": ["rout"]},
                                     "heads": {"answerble": "shadow"}},
                         "routes": {"nope": {"a": {}}, "response_type": {"summarize": {"requries": ["llm_large"]}}},
                         "owners": {"incident": "on-call"}})
    assert CC.errors(f) == []
    assert "Did you mean 'low'?" in at(f, "rollout.limited.stakes", "warn")
    assert "Did you mean 'route'?" in at(f, "rollout.limited.heads", "warn")
    assert "Did you mean 'answerable'?" in at(f, "rollout.heads.answerble", "warn")
    assert "not a head of this project" in at(f, "routes.nope", "warn")    # never consulted by a decision
    assert "Did you mean 'requires'?" in at(f, "routes.response_type.summarize.requries", "warn")
    assert "Did you mean 'incidents'?" in at(f, "owners.incident", "warn")
    f = check(tmp_path, {"rollout": {"mode": "full", "limited": {"stakes": ["low"]}}})
    assert "only read in mode limited" in at(f, "rollout.limited", "warn")


def test_a_route_contract_the_service_would_refuse_is_an_error(tmp_path):
    """A contract under an option the head does not have (a misspelt option), or on a head that is not a choice
    head, fails `rlcd check` (routes.contracts), and RouteBook.check_head refuses every /v1/decide that applies route
    contracts to the head with 500 bad_config. The documents decide applies them to response_type and route only:
    for routes.route.vectr `rlcd decide` exits 1, while a contract on answerable never reaches a decide. It used
    to be only a warning, so `rlcd train` and `rlcd serve` started on a file `rlcd check` fails."""
    specs = load_pack("documents").heads(Project(root=tmp_path, data={}))
    cases = (({"route": {"vectr": {}, "vector": {}}}, "routes.route.vectr", "is not an option of head 'route'"),
             ({"answerable": {"true": {}}}, "routes.answerable", "is a noul head"))
    for routes, path, fragment in cases:
        f = check(tmp_path, {"routes": routes})
        msg = at(f, path)                                   # an error, once
        assert fragment in msg and "500 bad_config" in msg and "`rlcd check` fails routes.contracts" in msg, msg
        assert len(CC.errors(f)) == 1, f
        head = next(iter(routes))
        with pytest.raises(ContractError) as e:             # what a decide applying the head's contracts runs into
            RouteBook.from_config(routes).check_head(head, specs[head])
        assert e.value.kind == "bad_config" and e.value.status == 500
    assert "Did you mean 'vector'?" in at(check(tmp_path, {"routes": cases[0][0]}), "routes.route.vectr")
    # a head switched off is not served, so its contracts are never consulted
    off = check(tmp_path, {"heads": {"answerable": {"enabled": False}}, "routes": {"answerable": {"true": {}}}})
    assert CC.errors(off) == []


# ----------------------------------------------------------------------------- heads

def test_head_override_fields_are_the_headspec_fields(tmp_path):
    assert set(CC.head_keys()) == (set(HeadSpec.__dataclass_fields__) - {"name"}) | {"enabled"}
    f = check(tmp_path, {"heads": {"answerable": {"stakess": "high", "model": "xgboost", "conformal": 0.7,
                                                 "decision": "gat", "enabled": "no", "classes": "a,b"},
                                   "grounded": {"stakes": "hihg"}, "route": "high"}})
    assert "Did you mean 'stakes'?" in at(f, "heads.answerable.stakess")
    assert "logreg, lightgbm, auto" in at(f, "heads.answerable.model")
    assert "must be in (0, 0.5)" in at(f, "heads.answerable.conformal")
    assert "Did you mean 'gate'?" in at(f, "heads.answerable.decision")
    assert "must be true or false" in at(f, "heads.answerable.enabled")
    assert "must be a list" in at(f, "heads.answerable.classes")
    assert "Did you mean 'high'?" in at(f, "heads.grounded.stakes")
    assert "must be a mapping" in at(f, "heads.route")


def test_what_only_the_merged_head_shows_is_reported_once(tmp_path):
    f = check(tmp_path, {"heads": {"answerble": {"stakes": "low"}, "brand_new": None}})
    assert "Did you mean 'answerable'?" in at(f, "heads.answerble")           # a misspelled head is not a new head
    assert "needs at least `type`" in at(f, "heads.brand_new") and len(CC.errors(f)) == 2
    f = check(tmp_path, {"heads": {"brand_new": {"type": "choice"}}})         # what HeadSpec itself refuses
    assert "a choice head needs 2 to 255 classes" in at(f, "heads") and len(CC.errors(f)) == 1
    f = check(tmp_path, {"heads": {"route": {"decision": "gate"}}})           # gate on a choice head
    assert "'gate' is for noul heads" in at(f, "heads")
    f = check(tmp_path, {"heads": {"brand_new": {"type": "choice", "classes": ["a", "b"], "view": "query"},
                                   "route": {"name": "router"}}})
    assert "is ignored" in at(f, "heads.route.name", "warn") and CC.errors(f) == []


def test_a_stakes_level_must_exist_in_the_policy(tmp_path):
    f = check(tmp_path, {"heads": {"answerable": {"stakes": "critical"}}})
    assert "low, medium, high" in at(f, "heads.answerable.stakes")
    ok = {"heads": {"answerable": {"stakes": "critical"}},
          "policy": {"stakes": {"critical": {"act": 0.99, "confirm": 0.95, "min_accuracy": 0.98}}}}
    assert check(tmp_path, ok) == []

    class Typo(ToyPack):                                  # the pack itself names a level nobody defined
        def heads(self, project):
            return {"good": HeadSpec("good", "noul", view="v", stakes="hihg")}

    assert "Did you mean 'high'?" in at(check(tmp_path, {}, Typo()), "heads.good")


# ----------------------------------------------------------------------------- pack and pack_config

def test_documents_pack_config_keys(tmp_path):
    f = check(tmp_path, {"pack_config": {"encodr": {"kind": "hashing"}, "encoder": {"kind": "onx", "dims": 256},
                                         "chunking": {"max_chars": 0}, "vector": {"kind": "qdrant"},
                                         "retrieve_k": "many", "on_encoder_change": "reembd"}})
    assert "Did you mean 'encoder'?" in at(f, "pack_config.encodr")
    assert "Did you mean 'onnx'?" in at(f, "pack_config.encoder.kind")
    assert "Did you mean 'dim'?" in at(f, "pack_config.encoder.dims")
    assert "must be >= 1" in at(f, "pack_config.chunking.max_chars")
    assert "must be one of local" in at(f, "pack_config.vector.kind")
    assert "whole number" in at(f, "pack_config.retrieve_k")
    assert "Did you mean 'reembed'?" in at(f, "pack_config.on_encoder_change")


def test_tabular_pack_config_keys(tmp_path):
    assert check(tmp_path, TABULAR) == []
    pc = {"data": "data.csv", "featurs": ["a"], "max_categories": 0, "label_source": "feedback",
          "hash_categories": "yes", "targets": {"churned": {"lable": "churned", "label": "churned"}}}
    f = check(tmp_path, {"pack": "tabular", "pack_config": pc})
    assert "Did you mean 'features'?" in at(f, "pack_config.featurs")
    assert "must be in [1, 10000]" in at(f, "pack_config.max_categories")
    assert "human, ground_truth, synthetic, judge" in at(f, "pack_config.label_source")
    assert "must be true or false" in at(f, "pack_config.hash_categories")
    assert "Did you mean 'label'?" in at(f, "pack_config.targets.churned.lable")
    assert len(CC.errors(f)) == 5
    # what only parse_config can say comes from parse_config, once
    f = check(tmp_path, {"pack": "tabular", "pack_config": {"label": "y"}})
    assert "pack_config.data is required" in at(f, "pack_config")
    f = check(tmp_path, {"pack": "tabular", "pack_config": {**TABULAR["pack_config"], "label": "y"}})
    assert "use one" in at(f, "pack_config")
    f = check(tmp_path, {"pack": "tabular", "pack_config": {"data": "d.csv", "positive": "yes",
                                                            "targets": {"a": "a"}}})
    assert "unknown keys ['positive']" in at(f, "pack_config")
    # `heads:` may add a head with its own label column; without one it is an error
    assert check(tmp_path, {**TABULAR, "heads": {"renewed": {"label": "renewed", "stakes": "low"}}}) == []
    f = check(tmp_path, {**TABULAR, "heads": {"renewed": {"stakes": "low"}}})
    assert "has no label column" in at(f, "pack_config")
    assert "Did you mean 'label'?" in at(check(tmp_path, {**TABULAR, "heads": {"churned": {"labl": "x"}}}),
                                         "heads.churned.labl")


def test_the_tabular_key_list_is_the_one_parse_config_uses():
    keys = load_pack("tabular").config_keys()
    assert set(keys) == set(BASE_KEYS) | set(TOLERATED_KEYS) | set(TARGET_KEYS) | set(SPEC_KEYS)
    assert set(keys["targets"].each.keys) == set(TARGET_KEYS) | set(SPEC_KEYS)


def test_a_tabular_target_switched_off_in_pack_config_is_kept_like_a_switched_off_head(tmp_path):
    """`pack_config.targets.<head>.enabled: false` switches a head off as `heads.<head>.enabled: false` does, and
    config check reads the list the service reads (`switched_off`): the head's route contracts, rollout modes and
    policy overrides are kept, not compared with the head. (It used to know only the `heads:` switch, so it warned
    "not a head of this project" about a switched-off target's contract.) Switched on again under `heads:`, the
    override winning field by field, they are compared with the head; a target removed from `targets` is not a head
    of the project. `rlcd config show` lists the switched-off head as `enabled: false`."""
    (tmp_path / "data.csv").write_text(TABLE, encoding="utf-8")
    targets = {"churned": {"label": "churned"}, "upsell": {"label": "plan", "enabled": False}}
    data = {"pack": "tabular", "pack_config": {"data": "data.csv", "targets": targets},
            "routes": {"upsell": {"true": {}}},                           # a contract on a noul head
            "rollout": {"heads": {"upsell": "shadow"}},
            "policy": {"overrides": {"upsell": {"maybe": {"*": {"act": 0.99}}}}}}   # a class it does not have
    assert check(tmp_path, data) == []
    view = CC.effective(Project(root=tmp_path, data=data))
    assert set(view["heads"]) == {"churned", "upsell"} and view["heads"]["upsell"] == {"enabled": False}
    f = check(tmp_path, {**data, "heads": {"upsell": {"enabled": True}}})
    assert "is a noul head" in at(f, "routes.upsell")
    assert "is not a class of head 'upsell'" in at(f, "policy.overrides.upsell.maybe")
    assert len(CC.errors(f)) == 2 and CC.warnings(f) == []
    gone = copy.deepcopy(data)
    del gone["pack_config"]["targets"]["upsell"]
    f = check(tmp_path, gone)
    assert "not a head of this project" in at(f, "routes.upsell", "warn")
    assert "not a head of this project" in at(f, "rollout.heads.upsell", "warn")
    assert "is not a head of this project" in at(f, "policy.overrides.upsell")


def test_a_custom_pack_is_checked_only_for_the_keys_it_declares(tmp_path):
    data = {"pack_config": {"anything": 1, "windw_days": 3}, "heads": {"good": {"extra": 1}}}
    f = check(tmp_path, data, ToyPack())                                   # declares nothing: pack_config is its own
    assert [x["path"] for x in f] == ["heads.good.extra"]

    class Declares(ToyPack):
        def config_keys(self):
            return {"window_days": CC.Key("int", default=30, min=1), "anything": None}

        def head_override_keys(self):
            return {"extra"}

        def check_config(self, project):
            return [{"level": "warn", "path": "pack_config.anything", "message": "is deprecated"}]

    f = check(tmp_path, data, Declares())
    assert "Did you mean 'window_days'?" in at(f, "pack_config.windw_days") and len(f) == 1
    f = check(tmp_path, {"pack_config": {"window_days": 0, "anything": {"x": 1}}}, Declares())
    assert "must be >= 1" in at(f, "pack_config.window_days")
    f = check(tmp_path, {"pack_config": {"anything": 1}}, Declares())
    assert at(f, "pack_config.anything", "warn") == "is deprecated"
    assert CC.known_keys(Declares()) == ["pack_config.anything", "pack_config.window_days"]
    assert CC.known_keys(ToyPack()) == []


def test_a_pack_that_cannot_be_imported_is_a_warning_not_a_config_error(tmp_path):
    f = check(tmp_path, {"pack": "no_such_pkg_xyz.pack:Pack", "pack_config": {"whatever": 1}})
    assert "could not be loaded" in at(f, "pack", "warn") and CC.errors(f) == []


# ----------------------------------------------------------------------------- secrets

def test_secrets_written_into_the_file_are_warnings_that_never_quote_the_value(tmp_path):
    key = "sk-ant-" + "a1B2c3D4e5" * 4
    f = check(tmp_path, {"pack_config": {"queries": "postgres://rlcd:hunter2secret@db.internal/q"},
                         "labeling": {"base_url": "http://127.0.0.1:11434", "executable": key},
                         "routes": {"route": {"vector": {"api_key": "abc123def456", "token": "${VECTOR_TOKEN}",
                                                         "context_budget_tokens": 2000}}}})
    assert CC.errors(f) == []
    assert "url_password" in at(f, "pack_config.queries", "warn")
    assert "api_key" in at(f, "labeling.executable", "warn")
    assert "the key names a secret" in at(f, "routes.route.vector.api_key", "warn")
    assert len(f) == 3                                     # an environment reference and a token budget are fine
    text = json.dumps(f)
    assert key not in text and "hunter2secret" not in text and "abc123def456" not in text
    assert "environment" in text


# ----------------------------------------------------------------------------- the file itself

def test_yaml_that_does_not_parse_is_one_finding_with_the_line(tmp_path):
    root = write(tmp_path, "train:\n  n_splits: 5\n holdout_fraction: [0.2\n")
    project, f = CC.check_file(root)
    assert project is None and "not valid YAML: line" in at(f, "(file)")
    root = write(tmp_path, "- a\n- b\n", "list")
    project, f = CC.check_file(root)
    assert project is None and "top level must be a mapping" in at(f, "(file)")
    with pytest.raises(FileNotFoundError):
        CC.check_file(tmp_path / "missing")


def test_a_key_written_twice_is_an_error(tmp_path):
    root = write(tmp_path, "heads:\n  answerable: {stakes: high}\ntrain: {seed: 1}\nheads:\n  grounded: {stakes: "
                           "high}\nserve:\n  port: 1\n  port: 2\n")
    f = CC.check_file(root)[1]
    assert "written twice" in at(f, "heads") and "written twice" in at(f, "serve.port")
    assert CC.duplicate_keys("a: {b: 1}\nc: {b: 2}\n") == []


# ----------------------------------------------------------------------------- the schema follows the code

def test_the_schema_names_the_same_things_as_the_modules_it_describes(tmp_path):
    s = CC.schema()
    assert set(s["routes"].each.each.keys) == set(RESERVED)
    assert tuple(s["owners"].keys) == tuple(CL.OWNER_ROLES) == CC.OWNER_ROLES
    assert s["serve"].keys["receipts"].choices == RECEIPT_MODES == CC.RECEIPT_MODES
    ok = {"version": "1", "owner": "o", "description": "d", "changelog": ["1"], "consumers": {"q": "x"},
          "questions": {"q": {"type": "noul", "head": "h"}}}
    assert set(ok) == set(CC.BATTERY_FIELDS) and BT.load(None, {"b": ok})
    with pytest.raises(ValueError, match="unknown fields"):
        BT.load(None, {"b": {**ok, "extra": 1}})
    # defaults: what the trainer and the policy use with an empty rlcd.yaml
    svc = Service(Project(root=tmp_path), home=tmp_path / "home", pack=ToyPack())
    cfg, train = svc.trainer.config(TrainRequest()), s["train"].keys
    for name, key in train.items():
        got = svc.trainer.feedback_weight(TrainRequest()) if name == "feedback_weight" else getattr(cfg, name)
        assert key.default == got, name
    policy, base = s["policy"].keys, PL.PolicyConfig()
    for name in ("abstain", "min_verify_rows", "gate_min_p_above", "miscalibration_p"):
        assert policy[name].default == getattr(base, name), name
    assert svc.receipts_mode() == s["serve"].keys["receipts"].default
    assert Project(root=tmp_path).pack == s["pack"].default


# ----------------------------------------------------------------------------- docs/configuration.md

def _documented() -> tuple[set[str], dict[str, set[str]]]:
    """Keys in the first column of the reference tables: (core keys, {pack: its pack_config keys})."""
    core: set[str] = set()
    packs: dict[str, set[str]] = {}
    section = ""
    for line in DOC.read_text(encoding="utf-8").splitlines():
        if line.startswith("## "):
            section = line[3:]
        m = re.match(r"^\| `([a-z0-9_]+(?:\.[a-z0-9_<>]+)*)` \|", line)
        if not m:
            continue
        pack = next((p for p in ("documents", "tabular") if f"`pack_config` of the {p} pack" in section), None)
        if pack:
            packs.setdefault(pack, set()).add(m.group(1))
        else:
            core.add(m.group(1))
    return core, packs


def test_the_reference_page_lists_exactly_the_keys_the_check_knows():
    core, packs = _documented()
    assert core == set(CC.known_keys())
    assert set(packs) == {"documents", "tabular"}
    for name, keys in packs.items():
        assert keys == set(CC.known_keys(load_pack(name))), name
    text = DOC.read_text(encoding="utf-8")
    for variable, _ in CC.ENVIRONMENT:
        assert f"| `{variable}` |" in text, variable


def _not_nullable(keys: dict, prefix: str = "") -> set[str]:
    """Every path whose explicit empty value (`key:`) the check refuses: Key.nullable False, maps included."""
    out: set[str] = set()
    for name, key in keys.items():
        path = f"{prefix}{name}"
        if not key.nullable:
            out.add(path)
        if key.keys is not None:
            out |= _not_nullable(key.keys, path + ".")
        elif key.each is not None:
            out |= _not_nullable({f"<{key.each.label}>": key.each}, path + ".")
    return out


def _paths(keys: dict, prefix: str = "") -> list[str]:
    """Every path of a key tree, placeholders kept (`heads.<head>.stakes`)."""
    out: list[str] = []
    for name, key in keys.items():
        path = f"{prefix}{name}"
        out.append(path)
        if key.keys is not None:
            out += _paths(key.keys, path + ".")
        elif key.each is not None:
            out += _paths({f"<{key.each.label}>": key.each}, path + ".")
    return out


# a readable table: the tabular pack resolves its heads (and so checks their fields) only from one
TABLE = "x,plan,churned\n" + "".join(f"{i},{'ab'[i % 2]},{'yes' if i % 3 else 'no'}\n" for i in range(60))
SHORTHAND = (*TARGET_KEYS, *SPEC_KEYS, "question")


def _required() -> set[str]:
    """The keys the reference tables mark "required" in their default column: written empty, such a key is refused
    just as it is when missing."""
    return set(re.findall(r"^\| `([^`]+)` \|[^|]*\| required \|", DOC.read_text(encoding="utf-8"), re.M))


def _empty(tmp_path, key: str, pack: str) -> tuple[dict, list[dict]]:
    """A valid project of `pack` with `key` written empty (placeholders filled with names that exist), and the
    errors config check reports on it."""
    parts = key.split(".")
    head = "churned" if pack == "tabular" else ("response_type" if parts[0] == "routes" else "answerable")
    fill = {"<head>": head, "<level>": "low", "<battery>": "pre", "<option>": "direct_answer"}
    path = [fill.get(p, p) for p in parts]
    if pack == "tabular":
        (tmp_path / "data.csv").write_text(TABLE, encoding="utf-8")
        target = {"label": "churned"} if path[1] in SHORTHAND else {"targets": {"churned": {"label": "churned"}}}
        data: dict = {"pack": "tabular", "pack_config": {"data": "data.csv", **target}}
    elif parts[0] == "batteries":
        data = {"batteries": {"pre": {"version": "1", "questions": {"q": {"type": "noul", "head": "answerable"}}}}}
    else:
        data = {}
    assert CC.errors(check(tmp_path, data)) == [], data
    node = data
    for p in path[:-1]:
        node = node.setdefault(p, {})
    node[path[-1]] = None
    return data, CC.errors(check(tmp_path, data))


def test_the_reference_page_names_every_key_whose_empty_value_is_an_error(tmp_path):
    """The reference page lists the keys that refuse an empty value (`key:`); every other key accepts one, as its
    default except where the page's other bullets say otherwise (`heads.<head>.conformal:` removes the pack's
    promise, `rollout.limited.stakes:` means no level, `routes.<head>.<option>:` is an empty contract, which a
    `fallback` may name, unlike a missing option). Checked both ways on the check itself: each listed key,
    written empty, is an error, and every other key of the schema and of both packs' pack_config is accepted,
    except one the tables mark required. (The check's own `Key.nullable` flags miss the errors that its validators
    raise.) Head fields, which this sweep fills with one noul head, are swept for every head type by the next
    test."""
    bullet = DOC.read_text(encoding="utf-8").split("* **An empty value is an error**", 1)[1].split("\n* **", 1)[0]
    named: set[str] = set()
    parent = ""
    for word in re.findall(r"`([a-z_.<>*]+)`", bullet):
        if "." in word:
            # a bare name after it is a field of a key ending in a placeholder (`policy.stakes.<level>` and its
            # `act`), else a sibling (`routes.<head>.<option>.enabled`, `requires`)
            parent = word if word.endswith(">") else word.rsplit(".", 1)[0]
        else:
            word = f"{parent}.{word}"
        named |= {f"train.{k}" for k in CC.schema()["train"].keys} if word == "train.*" else {word}
    want = _not_nullable(CC.schema())
    for pack in ("documents", "tabular"):
        want |= _not_nullable(CC._pack_keys(load_pack(pack)) or {}, "pack_config.")
    assert want <= named, want - named
    swept = [("documents", k) for k in _paths(CC.schema()) if k != "pack_config"]
    swept += [(p, k) for p in ("documents", "tabular") for k in _paths(CC._pack_keys(load_pack(p)) or {}, "pack_config.")]
    assert named <= {k for _, k in swept}, named - {k for _, k in swept}
    required = _required()
    assert {"batteries.<battery>.version", "pack_config.targets.<head>.label"} <= required <= {k for _, k in swept}
    wrong = []
    for pack, key in swept:
        data, errs = _empty(tmp_path, key, pack)
        if bool(errs) != (key in named or key in required):
            wrong.append((pack, key, errs[:1]))
    assert wrong == []
    # next to the one-head shorthand an empty `targets:` is ignored, as the bullet says
    (tmp_path / "data.csv").write_text(TABLE, encoding="utf-8")
    shorthand = {"pack": "tabular", "pack_config": {"data": "data.csv", "label": "churned", "targets": None}}
    assert CC.errors(check(tmp_path, shorthand)) == []
    # the example the paragraph opens with is a key whose empty value means the default
    assert check(tmp_path, {"labeling": {"timeout": None}}) == []
    assert at(check(tmp_path, {"serve": {"port": None}}), "serve.port").startswith("is empty: give it a value")


def test_an_empty_head_field_keeps_the_packs_value(tmp_path):
    """Under heads.<head>, a field written with no value keeps the pack's value (the target's, in the tabular pack),
    as if the line were not there, for a noul, a choice and a score head in both packs; `conformal:` alone removes
    the pack's promise. A tabular target field written with no value is its default, as when it is missing, but a
    target's `label` is required. (An empty `type:` used to be an error or turn a declared score head into a choice
    head; an empty `params:` passed the check and crashed `rlcd train`.)"""
    def heads(data: dict) -> dict:
        return load_pack(data.get("pack", "documents")).heads(Project(root=tmp_path, data=data))

    (tmp_path / "data.csv").write_text("x,churned,plan,tier\n" + "".join(
        f"{i},{('yes', 'no')[i % 2]},{('up', 'same', 'down')[i % 3]},{('low', 'mid', 'high')[i % 3]}\n"
        for i in range(60)), encoding="utf-8")
    targets = {"churned": {"label": "churned", "positive": "no", "stakes": "high", "params": {"C": 0.5}},
               "plan": {"label": "plan", "classes": ["up", "same", "down"], "model": "logreg"},
               "tier": {"label": "tier", "type": "score", "classes": ["low", "mid", "high"], "decision": "argmax",
                        "conformal": 0.1}}
    tabular = {"pack": "tabular", "pack_config": {"data": "data.csv", "targets": targets}}
    for base, names, fields in (({}, ("answerable", "response_type", "complete"), CC.head_keys()),
                                (tabular, tuple(targets), (*CC.head_keys(), "label"))):
        plain = heads(base)
        assert {plain[n].type for n in names} == {"noul", "choice", "score"}
        for name in names:
            for field in fields:
                data = {**copy.deepcopy(base), "heads": {name: {field: None}}}
                assert CC.errors(check(tmp_path, data)) == [], (name, field)
                want = dataclasses.replace(plain[name], conformal=None) if field == "conformal" else plain[name]
                assert heads(data)[name] == want, (name, field)
    for name in targets:
        for field in (*TARGET_KEYS, *SPEC_KEYS):
            data, missing = copy.deepcopy(tabular), copy.deepcopy(tabular)
            data["pack_config"]["targets"][name][field] = None
            missing["pack_config"]["targets"][name].pop(field, None)
            found = check(tmp_path, data)
            assert found == check(tmp_path, missing), (name, field)    # the same findings as without the line
            if field == "label":                                       # required
                assert "has no label column" in at(found, "pack_config"), name
            elif not CC.errors(found):
                assert heads(data)[name] == heads(missing)[name], (name, field)


def test_an_empty_feature_cache_is_the_default_cache(tmp_path):
    """`pack_config.feature_cache:` with no value keeps the default 1024 featurised states (it used to turn the
    decide-time cache off)."""
    def size(**pc):
        project = Project(root=tmp_path, data={"pack_config": {"encoder": {"kind": "hashing"}, **pc}})
        return load_pack("documents").provider(project)._cache_size

    assert size() == size(feature_cache=None) == 1024 and size(feature_cache=0) == 0


def test_the_examples_on_the_reference_page_have_no_findings(tmp_path):
    text = DOC.read_text(encoding="utf-8").split("## Worked examples")[1]
    blocks = re.findall(r"```yaml\n(.*?)```", text, re.S)
    assert len(blocks) == 3
    for i, block in enumerate(blocks):
        root = write(tmp_path, block, f"example{i}")
        assert CC.check_file(root)[1] == [], block


# ----------------------------------------------------------------------------- the commands

def test_config_check_prints_path_and_message_and_exits_1_on_an_error(tmp_path, capsys):
    root = write(tmp_path, {"pack": "documents", "rollout": {"mode": "shaddow"}, "train": {"holdout_fracton": 0.3},
                            "owners": {"incident": "x"}})
    assert main(["--project", str(root), "config", "check"]) == 1
    out = capsys.readouterr().out.splitlines()
    assert out[0].startswith("error  rollout.mode: must be one of shadow, assist, limited, full")
    assert out[1] == "error  train.holdout_fracton: unknown key. Did you mean 'holdout_fraction'?"
    assert out[2].startswith("warn   owners.incident: ")
    assert out[3].endswith("rlcd.yaml: 2 errors, 1 warning")
    assert main(["--project", str(root), "config", "check", "--json"]) == 1
    res = json.loads(capsys.readouterr().out)
    assert res["ok"] is False and res["errors"] == 2 and res["warnings"] == 1 and res["config"].endswith("rlcd.yaml")
    assert {f["path"] for f in res["findings"]} == {"rollout.mode", "train.holdout_fracton", "owners.incident"}
    assert all(set(f) == {"level", "path", "message"} for f in res["findings"])


def test_config_check_exits_0_with_warnings_only_and_without_a_file(tmp_path, capsys, monkeypatch):
    root = write(tmp_path, {"owners": {"incident": "x"}})
    assert main(["--project", str(root), "config", "check"]) == 0
    assert "0 errors, 1 warning" in capsys.readouterr().out
    bad = write(tmp_path, "train: [\n", "broken")
    assert main(["--project", str(bad), "config", "check"]) == 1
    assert "not valid YAML" in capsys.readouterr().out
    empty = tmp_path / "empty"
    empty.mkdir()
    monkeypatch.chdir(empty)
    assert main(["config", "check"]) == 0
    assert "no rlcd.yaml" in capsys.readouterr().out
    assert main(["config", "check", "--json"]) == 0
    assert json.loads(capsys.readouterr().out) == {"config": None, "ok": True, "errors": 0, "warnings": 0,
                                                   "findings": []}


def test_config_show_prints_the_effective_configuration_without_secrets(tmp_path, capsys, monkeypatch):
    monkeypatch.delenv("RLCD_HOME", raising=False)
    monkeypatch.setenv("RLCD_API_KEY", "super-secret-value")
    monkeypatch.setenv("RLCD_ALLOW_REMOTE", "1")
    root = write(tmp_path, {"name": "shop", "pack": "documents", "home": "./artifacts",
                            "pack_config": {"encoder": {"kind": "hashing", "dim": 64}, "password": "hunter2secret"},
                            "heads": {"answerable": {"stakes": "high", "conformal": 0.05},
                                      "needs_graph": {"enabled": False}},
                            "policy": {"stakes": {"high": {"act": 0.97}}}, "rollout": {"mode": "assist"},
                            "train": {"n_splits": 3}})
    assert main(["--project", str(root), "config", "show", "--json"]) == 0
    io = capsys.readouterr()
    view = json.loads(io.out)
    assert view["project"] == {"name": "shop", "config": str((root / "rlcd.yaml").resolve()),
                               "root": str(root.resolve()), "home": str((root / "artifacts").resolve()),
                               "home_from": "rlcd.yaml"}
    assert view["pack"]["name"] == "documents" and view["pack"]["from"] == "rlcd.yaml"
    a = view["heads"]["answerable"]                       # the pack's head with the file's two fields on top
    assert a["stakes"] == "high" and a["conformal"] == 0.05 and a["type"] == "noul" and a["decision"] == "gate"
    assert a["from_rlcd_yaml"] == ["conformal", "stakes"] and view["heads"]["route"]["from_rlcd_yaml"] == []
    assert view["heads"]["needs_graph"] == {"enabled": False}
    assert view["policy"]["abstain"] == "escalate"
    assert view["policy"]["stakes"]["high"] == {"act": 0.97, "confirm": 0.8, "min_accuracy": 0.9}
    assert view["policy"]["stakes"]["low"] == {"act": 0.7, "confirm": 0.5, "min_accuracy": 0.6}
    assert view["rollout"] == {"mode": "assist"}
    assert view["train"]["n_splits"] == 3 and view["train"]["holdout_fraction"] == 0.2
    assert view["serve"]["port"] == 8095 and view["evaluate"]["max_revisions"] == 2
    assert set(view["batteries"]) == {"preflight", "evidence", "postflight"}
    assert view["environment"]["RLCD_API_KEY"] == "set" and view["environment"]["RLCD_ADMIN_KEY"] == "not set"
    assert view["environment"]["RLCD_ALLOW_REMOTE"] == "1"
    assert view["pack_config"]["password"] == "[REDACTED:secret_field]"
    assert "super-secret-value" not in io.out + io.err and "hunter2secret" not in io.out + io.err
    assert "warn   pack_config.password" in io.err        # findings go to stderr, the view stays parseable
    assert main(["--project", str(root), "config", "show"]) == 0
    io = capsys.readouterr()
    assert yaml.safe_load(io.out) == view and "super-secret-value" not in io.out


def test_config_show_marks_what_cannot_be_resolved_instead_of_failing(tmp_path, capsys):
    root = write(tmp_path, {"rollout": {"mode": "yolo"}, "policy": {"stakes": {"low": {"act": 0.1}}}})
    assert main(["--project", str(root), "config", "show", "--json"]) == 0
    view = json.loads(capsys.readouterr().out)
    assert "rollout.mode must be one of" in view["rollout"]["error"] and "error" in view["policy"]
    assert "answerable" in view["heads"]


def test_train_and_serve_refuse_to_start_on_a_config_error(tmp_path, capsys):
    root = write(tmp_path, {**TABULAR, "train": {"holdout_fracton": 0.3}, "owners": {"incident": "x"}})
    for cmd in ("train", "serve"):
        assert main(["--project", str(root), "--home", str(tmp_path / "home"), cmd]) == 2
        err = capsys.readouterr().err
        assert "rlcd.yaml: error  train.holdout_fracton: unknown key. Did you mean 'holdout_fraction'?" in err
        assert "rlcd.yaml: warn   owners.incident" in err
        assert "has 1 error, so nothing was started" in err and "rlcd config check" in err


def test_warnings_do_not_stop_a_command_and_no_file_means_no_check(tmp_path, capsys):
    root = write(tmp_path, {**TABULAR, "owners": {"incident": "x"}})
    svc = Service(load_project(root), home=tmp_path / "home")
    assert _config_ok(svc, block=True) is True
    assert "rlcd.yaml: warn   owners.incident" in capsys.readouterr().err
    # the default project (no rlcd.yaml) always runs, whatever a caller put into Project.data
    fake = SimpleNamespace(project=Project(root=tmp_path, data={"polcy": 1}), pack=None)
    assert _config_ok(fake, block=True) is True
    assert capsys.readouterr().err == ""


def test_rlcd_check_prints_the_findings_and_fails_the_config_item(tmp_path, capsys):
    root = write(tmp_path, {**TABULAR, "serve": {"prot": 1}})
    assert main(["--project", str(root), "--home", str(tmp_path / "home"), "check", "--json"]) == 1
    io = capsys.readouterr()
    assert "rlcd.yaml: error  serve.prot: unknown key. Did you mean 'port'?" in io.err
    item = {c["id"]: c for c in json.loads(io.out)["checks"]}["config.valid"]
    assert item["status"] == "fail" and "serve.prot: unknown key" in item["detail"]


# ----------------------------------------------------------------------------- the checklist item

def test_the_checklist_item_follows_the_findings(tmp_path):
    def item(data, **kw):
        svc = Service(Project(root=tmp_path, data=data), home=tmp_path / "home", pack=ToyPack())
        return {c["id"]: c for c in CL.run(svc, **kw)["checks"]}["config.valid"]

    assert item({})["status"] == "info"
    assert item({"rollout": {"mode": "shadow"}})["status"] == "pass"
    warned = item({"rollout": {"heads": {"godo": "shadow"}}})
    assert warned["status"] == "warn" and "rollout.heads.godo" in warned["detail"]
    failed = item({"rollout": {"mode": "yolo"}, "serve": {"prot": 1}})
    assert failed["status"] == "fail" and "2 errors" in failed["detail"] and "got 'yolo'" in failed["detail"]
    quiet = item({"rollout": {"mode": "yolo"}}, error_text=False)          # over HTTP: the key, not the message
    assert quiet["status"] == "fail" and "rollout.mode" in quiet["detail"] and "yolo" not in quiet["detail"]
    assert Project(root=tmp_path, data={"polcy": 1}).check(ToyPack())[0]["path"] == "polcy"
