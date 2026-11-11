"""The pre-launch checklist (core/checklist.py, `rlcd check`): one line per launch item, `ok` false only when a check
fails, every check isolated from the others. Offline: the toy pack, and the untrained documents pack's specs."""
from __future__ import annotations

import copy
import json
import shutil
import sys
from datetime import UTC, datetime, timedelta

import pytest

from open_rlcd.core import checklist as CL
from open_rlcd.core.config import Project
from open_rlcd.core.config_check import check_project
from open_rlcd.core.contract import ContractError, DecideRequest, SystemOneRequest, TrainRequest
from open_rlcd.core.head import HeadSpec
from open_rlcd.core.service import Service
from tests.toypack import ToyPack

ENV = ("RLCD_API_KEY", "RLCD_ADMIN_KEY", "RLCD_ALLOW_REMOTE", "RLCD_AUTOMATION")
STATES = [{"f": [0.1] * 6}, {"f": [2.0, -1.0, 0.5, 1.0, 0.0, -0.5]}, {"f": [-2.0, 1.0, 0.0, 0.0, 1.5, 0.3]}]
BATTERY = {"pre": {"version": "1", "owner": "qa", "questions": {"g": {"type": "noul", "head": "good"}}}}
NOW = datetime(2026, 11, 9, 12, 0, tzinfo=UTC)
CONTRACT_CHECKS = ("options.no_match", "score.anchors", "heads.described", "noul.positive", "options.overlap")


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for k in ENV:
        monkeypatch.delenv(k, raising=False)


@pytest.fixture(scope="module")
def trained_home(tmp_path_factory):
    """A toy home with every head trained once (rlcd-0.1.1). Tests work on copies of it."""
    root = tmp_path_factory.mktemp("checklist")
    svc = Service(Project(root=root), home=root / "home", pack=ToyPack())
    assert svc.trainer.start(TrainRequest(), wait=True)["status"] == "succeeded"
    return root / "home"


def make(tmp_path, home=None, data=None, pack=None) -> Service:
    """A service rooted at tmp_path on tmp_path/home: a copy of `home` (made once), else a fresh untrained one."""
    dest = tmp_path / "home"
    if home is not None and not dest.exists():
        shutil.copytree(home, dest)
    return Service(Project(root=tmp_path, data=copy.deepcopy(data or {})), home=dest, pack=pack or ToyPack())


def by_id(res: dict) -> dict:
    return {c["id"]: c for c in res["checks"]}


def checks(svc: Service, **kw) -> dict:
    return by_id(CL.run(svc, **kw))


class Described(ToyPack):
    """The toy heads with the question contract the guide asks for (descriptions, positive, anchors, a no-match)."""

    def __init__(self, color_levels=("a red item", "a green item", "no listed color fits"),
                 score_levels=("0: misses the question", "1: answers a small part", "2: answers most of it",
                               "3: answers every part"), **kw):
        super().__init__(**kw)
        self.color_levels, self.score_levels = tuple(color_levels), tuple(score_levels)

    def heads(self, project):
        return {"good": HeadSpec("good", "noul", view="v", decision="gate", conformal=0.1,
                                 description="Does the item meet the spec?", positive="every listed requirement holds"),
                "color": HeadSpec("color", "choice", ("red", "green", "none"), view="v",
                                  description="Which color is the item?", levels=self.color_levels),
                "level": HeadSpec("level", "score", ("0", "1", "2", "3"), view="v",
                                  description="How much of the request is answered?", levels=self.score_levels)}


# ----------------------------------------------------------------------------- the whole run

def test_an_untrained_service_fails_on_its_heads_and_still_runs_every_check(tmp_path):
    res = CL.run(make(tmp_path))
    c = by_id(res)
    assert res["ok"] is False and res["model"] == "rlcd-0.1.0"
    assert [i for i, x in c.items() if x["status"] == "fail"] == ["heads.trained"]
    assert "no head is trained in rlcd-0.1.0" in c["heads.trained"]["detail"]
    assert not any(i.startswith("skill.") for i in c)
    assert c["thresholds.holdout"]["status"] == "info" and c["model.bundle"]["status"] == "info"
    assert c["rollback.ready"]["status"] == "warn" and c["receipts.chain"]["status"] == "info"
    assert len(c) == len(res["checks"])                       # ids are unique
    for x in res["checks"]:
        assert set(x) == {"id", "status", "title", "detail"} and x["status"] in CL.STATUSES and x["title"]
        assert x["detail"]
    assert set(res["counts"]) == set(CL.STATUSES) and sum(res["counts"].values()) == len(res["checks"])
    assert {"config.valid", "heads.trained", "thresholds.holdout", "thresholds.config", *CONTRACT_CHECKS, "inputs.required",
            "deterministic.first", "routes.contracts", "security.api_key", "security.allow_remote",
            "security.admin_key", "secrets.redact", "secrets.log_states", "receipts.chain", "receipts.mode",
            "model.bundle", "rollback.ready", "rollout.mode", "automation.switch", "batteries.bound",
            "batteries.versions", "regressions.cases", "review.capacity", "owners", "failure.behaviour"} <= set(c)


def test_a_trained_service_passes_with_warnings_about_its_bare_question_contract(trained_home, tmp_path):
    res = CL.run(make(tmp_path, trained_home))
    c = by_id(res)
    assert res["ok"] is True and res["model"] == "rlcd-0.1.1" and res["counts"]["fail"] == 0
    assert c["heads.trained"]["status"] == "pass"
    for head in ("good", "color", "level"):
        s = c[f"skill.{head}"]
        assert s["status"] == "pass" and "skill over the base rate on 600 trusted holdout rows" in s["detail"]
        assert all(f"{lv} " in s["detail"].split("verdict: ")[1] for lv in ("low", "medium", "high"))
    assert c["thresholds.holdout"]["status"] == "pass" and c["thresholds.config"]["status"] == "pass"
    assert c["model.bundle"]["status"] == "pass" and c["security.api_key"]["status"] == "pass"
    # the toy pack's question contract is bare: each gap is a warning, none is a failure
    assert c["options.no_match"]["status"] == "warn" and "color" in c["options.no_match"]["detail"]
    assert c["score.anchors"]["status"] == "warn" and "'mid'" in c["score.anchors"]["detail"]
    assert c["noul.positive"]["status"] == "warn" and c["noul.positive"]["detail"].startswith("good:")
    assert c["heads.described"]["status"] == "warn" and c["options.overlap"]["status"] == "pass"
    assert c["rollback.ready"]["status"] == "warn" and "no earlier version" in c["rollback.ready"]["detail"]
    assert c["owners"]["status"] == "warn"
    assert c["failure.behaviour"]["status"] == "info" and "docs/contract.md" in c["failure.behaviour"]["detail"]


def test_a_head_without_skill_is_a_warning_that_shows_its_verdict(tmp_path):
    svc = make(tmp_path, pack=ToyPack(signal=0.0, n=1500))
    assert svc.trainer.start(TrainRequest(), wait=True)["status"] == "succeeded"
    res = CL.run(svc)
    c = by_id(res)
    for head in ("good", "color", "level"):
        s = c[f"skill.{head}"]
        assert s["status"] == "warn" and "no skill over the base rate" in s["detail"]
        assert "verdict: low escalate, medium escalate, high escalate" in s["detail"]
    assert c["heads.trained"]["status"] == "pass" and res["ok"] is True      # every decision abstains: safe


def test_heads_left_untrained_are_a_warning(tmp_path):
    svc = make(tmp_path)
    assert svc.trainer.start(TrainRequest(heads=["good"]), wait=True)["status"] == "succeeded"
    c = checks(svc)
    assert c["heads.trained"]["status"] == "warn"
    assert c["heads.trained"]["detail"].startswith("1 of 3 heads are trained in rlcd-0.1.1; not trained: color, level")
    assert {i for i in c if i.startswith("skill.")} == {"skill.good"}


def test_a_corrupted_served_bundle_fails_and_only_its_head_is_affected(trained_home, tmp_path):
    svc = make(tmp_path, trained_home)
    head_json = svc.home / "models" / "rlcd-0.1.1" / "heads" / "level" / "head.json"
    head_json.write_text(head_json.read_text(encoding="utf-8") + " ", encoding="utf-8")   # no longer its sha256
    res = CL.run(svc)
    c = by_id(res)
    assert c["model.bundle"]["status"] == "fail" and "head 'level'" in c["model.bundle"]["detail"]
    assert c["skill.level"]["status"] == "fail" and "BundleError" in c["skill.level"]["detail"]
    assert c["skill.good"]["status"] == c["skill.color"]["status"] == "pass"
    assert c["thresholds.holdout"]["status"] == "fail" and c["thresholds.holdout"]["detail"].startswith("level: ")


def test_an_unknown_model_is_a_contract_error(trained_home, tmp_path):
    with pytest.raises(ContractError) as e:
        CL.run(make(tmp_path, trained_home), model="rlcd-0.1.99")
    assert e.value.kind == "unknown_model"


def test_a_check_that_raises_fails_alone_and_hides_its_text_on_request(trained_home, tmp_path):
    svc = make(tmp_path, trained_home)

    def boom():
        raise RuntimeError("disk on fire under /very/secret/path")

    svc.verify_receipts = boom
    res = CL.run(svc)
    c = by_id(res)
    assert c["receipts.chain"]["status"] == "fail" and res["ok"] is False
    assert "RuntimeError: disk on fire" in c["receipts.chain"]["detail"]
    assert c["heads.trained"]["status"] == "pass" and c["receipts.mode"]["status"] == "info"   # the others ran
    quiet = checks(svc, error_text=False)["receipts.chain"]
    assert quiet["status"] == "fail" and "RuntimeError" in quiet["detail"] and "secret" not in quiet["detail"]


def test_the_table_lists_every_check_and_ends_with_the_verdict(trained_home, tmp_path):
    res = CL.run(make(tmp_path / "untrained"))
    lines = CL.format_table(res).splitlines()
    assert lines[0] == "rlcd check: model rlcd-0.1.0"
    assert all(any(f"  {c['id']}  " in ln for ln in lines) for c in res["checks"])
    assert any(ln.startswith("FAIL  heads.trained") for ln in lines)
    assert lines[-1].startswith("NOT READY") and lines[-1].endswith("fix heads.trained before launch")
    ok = CL.format_table(CL.run(make(tmp_path / "trained", trained_home))).splitlines()
    assert ok[-1].startswith("no check fails")


# ----------------------------------------------------------------------------- thresholds

def test_thresholds_served_other_than_the_verified_ones_fail(trained_home, tmp_path):
    """model.json carries the policy and no hash covers it: an edited (or imported) act threshold validates, so the
    checklist compares it with the policy each head's holdout report was computed with."""
    home = tmp_path / "home"
    shutil.copytree(trained_home, home)
    mj = home / "models" / "rlcd-0.1.1" / "model.json"
    meta = json.loads(mj.read_text(encoding="utf-8"))
    meta["policy"]["stakes"]["medium"].update(act=0.6, confirm=0.5)      # lowered after the holdout verified 0.85
    mj.write_text(json.dumps(meta), encoding="utf-8")
    res = CL.run(make(tmp_path))
    c = by_id(res)
    t = c["thresholds.holdout"]
    assert t["status"] == "fail" and res["ok"] is False
    assert "act at medium stakes: verified 0.85, served 0.6" in t["detail"]
    assert all(f"{h} was verified under other thresholds" in t["detail"] for h in ("good", "color", "level"))
    assert c["thresholds.config"]["status"] == "warn"     # rlcd.yaml (the defaults) is not what the version serves
    assert c["model.bundle"]["status"] == "pass"          # the bundle validates: only the comparison sees the edit


def test_rlcd_yaml_thresholds_apply_only_after_a_retrain(trained_home, tmp_path):
    c = checks(make(tmp_path, trained_home, {"policy": {"stakes": {"medium": {"act": 0.9}}}}))
    assert c["thresholds.config"]["status"] == "warn"
    assert "stakes.medium.act 0.85 -> 0.9" in c["thresholds.config"]["detail"]
    assert c["thresholds.holdout"]["status"] == "pass"    # what is served is still what was verified


def test_the_holdout_check_fails_exactly_when_a_carried_over_head_was_verified_under_other_thresholds(tmp_path):
    project = Project(root=tmp_path)
    svc = Service(project, home=tmp_path / "home", pack=ToyPack(n=600))
    assert svc.trainer.start(TrainRequest(), wait=True)["status"] == "succeeded"
    project.data["policy"] = {"stakes": {"medium": {"act": 0.6, "confirm": 0.4}}}
    assert svc.trainer.start(TrainRequest(heads=["good"]), wait=True)["status"] == "succeeded"
    v2 = svc.registry.latest_id()
    served = svc.policy(v2).level("medium").act
    drifted = [n for n in svc.registry.bundle(v2).names()
               if svc.loaded(v2, n).report["policy"]["config"]["stakes"]["medium"]["act"] != served]
    c = checks(svc)["thresholds.holdout"]
    assert (c["status"] == "fail") == bool(drifted)
    assert all(f"{n} was verified under other thresholds" in c["detail"] for n in drifted)


def test_a_stakes_level_without_a_holdout_verdict_is_never_automated(tmp_path):
    project = Project(root=tmp_path)
    svc = Service(project, home=tmp_path / "home", pack=ToyPack(n=600))
    assert svc.trainer.start(TrainRequest(), wait=True)["status"] == "succeeded"
    project.data["policy"] = {"stakes": {"critical": {"act": 0.6, "confirm": 0.4, "min_accuracy": 0.5}}}
    assert svc.trainer.start(TrainRequest(heads=["good"]), wait=True)["status"] == "succeeded"
    v2 = svc.registry.latest_id()
    assert "critical" not in svc.loaded(v2, "color").verdict          # carried over from rlcd-0.1.1
    d = svc.head_decision(v2, svc.spec("color", v2), STATES[0], stakes="critical", explain=False)["decision"]
    assert d["max_tier"] != "act"            # no holdout evidence at this level: at most confirm


def test_a_head_is_never_served_act_below_the_act_threshold_it_was_verified_with(tmp_path):
    """A partial retrain under a looser policy carries the other heads over with the verdict of their own training;
    the version serves its new thresholds, so acting below a head's verified act threshold is capped at confirm."""
    import numpy as np

    project = Project(root=tmp_path)
    svc = Service(project, home=tmp_path / "home", pack=ToyPack(n=600))
    assert svc.trainer.start(TrainRequest(), wait=True)["status"] == "succeeded"
    project.data["policy"] = {"stakes": {"medium": {"act": 0.6, "confirm": 0.4}}}
    assert svc.trainer.start(TrainRequest(heads=["good"]), wait=True)["status"] == "succeeded"
    v2 = svc.registry.latest_id()
    assert svc.policy(v2).level("medium").act == 0.6
    verified = svc.verified_policy(v2, "color", svc.loaded(v2, "color")).level("medium").act
    assert verified == 0.85                                  # color was carried over, verified at the old threshold
    r = svc.head_decision(v2, svc.spec("color", v2), {"f": [0.0] * 6}, stakes="medium",
                          probs=np.array([0.1, 0.75, 0.15]), calibrated=True, explain=False)
    assert r["decision"]["p"] == 0.75 and r["decision"]["tier"] != "act"
    d = {"class": "green", "calibrated": True, "tier": "act", "stakes": "medium", "thresholds": {"act": 0.6},
         "warnings": []}
    svc._apply_verified(d, v2, svc.spec("color", v2), svc.loaded(v2, "color"))
    assert d["tier"] == "confirm" and "verified with act >= 0.85" in d["caps"][0]["reason"]


# ----------------------------------------------------------------------------- the question contract

def test_a_described_question_contract_passes(tmp_path):
    c = checks(make(tmp_path, pack=Described()))
    for cid in CONTRACT_CHECKS:
        assert c[cid]["status"] == "pass", c[cid]
    assert "color (none)" in c["options.no_match"]["detail"]


def test_the_documents_pack_follows_the_question_contract_rules(tmp_path):
    data = {"pack": "documents", "pack_config": {"encoder": {"kind": "hashing", "dim": 256}}}
    svc = Service(Project(root=tmp_path, data=data), home=tmp_path / "home")
    c = checks(svc)
    for cid in CONTRACT_CHECKS:
        assert c[cid]["status"] == "pass", c[cid]
    b = c["batteries.bound"]                      # its batteries bind, but nothing is trained yet
    assert b["status"] == "warn" and "not trained" in b["detail"]
    assert all(f"{n}@1" in b["detail"] for n in ("preflight", "evidence", "postflight"))


@pytest.mark.parametrize("levels", [("low", "medium", "high", "very high"), ("0: bad", "1: ok", "2: good", "3: great"),
                                    ("0", "1", "2", "3")])
def test_score_levels_without_observable_anchors_warn(tmp_path, levels):
    c = checks(make(tmp_path, pack=Described(score_levels=levels)))["score.anchors"]
    assert c["status"] == "warn" and c["detail"].split(": ", 1)[1].startswith("level:")


def test_option_descriptions_that_only_repeat_the_names_do_not_count(tmp_path):
    c = checks(make(tmp_path, pack=Described(color_levels=("red", "Green", "none"))))["heads.described"]
    assert c["status"] == "warn" and "without option descriptions (`levels`): color" in c["detail"]


def test_options_with_the_same_description_warn(tmp_path):
    pack = Described(color_levels=("a red item", "A red  item.", "no listed color fits"))
    c = checks(make(tmp_path, pack=pack))["options.overlap"]
    assert c["status"] == "warn" and "color: red / green" in c["detail"]


def test_route_contracts_must_match_the_heads(trained_home, tmp_path):
    routes = {"color": {"red": {"llm": "small", "allowed_data": ["public"]}, "green": {"llm": "large"}}}
    c = checks(make(tmp_path, trained_home, {"routes": routes}))["routes.contracts"]
    assert c["status"] == "pass"
    assert c["detail"] == ("contracts for color; accept every data class (no allowed_data): color.green; options "
                           "without a contract (never filtered): color.blue")
    for routes, why, refused in (({"color": {"purple": {"llm": "small"}}}, "purple", True),
                                 ({"good": {"true": {"llm": "small"}}}, "is a noul head", True),
                                 ({"nope": {"a": {}}}, "has no head 'nope'", False)):
        res = CL.run(make(tmp_path, trained_home, {"routes": routes}))
        c = by_id(res)["routes.contracts"]
        assert c["status"] == "fail" and why in c["detail"] and res["ok"] is False
        # the 500 is what a decide applying the contract runs into; a head the model lacks is never decided
        assert ("500 bad_config" in c["detail"]) is refused, c["detail"]


class Switchable(ToyPack):
    """The toy pack, honouring `heads.<name>.enabled: false` the way every BasePack does."""

    def heads(self, project):
        off = {n for n, o in project.head_overrides.items() if isinstance(o, dict) and o.get("enabled") is False}
        return {n: s for n, s in super().heads(project).items() if n not in off}


def test_route_contracts_of_a_disabled_head_are_kept_not_failed(trained_home, tmp_path):
    """A head switched off in rlcd.yaml is not served, so no decision consults its contracts: config check passes
    them, and so does `rlcd check` (it used to fail, saying the trained model had no such head)."""
    routes = {"color": {"red": {"llm": "small", "allowed_data": ["public"]}}, "good": {"true": {"llm": "small"}}}
    svc = make(tmp_path, trained_home, {"heads": {"color": {"enabled": False}, "good": {"enabled": False}},
                                        "routes": routes}, pack=Switchable())
    assert "color" not in svc.specs(svc.registry.latest_id()) and "good" not in svc.specs(svc.registry.latest_id())
    c = checks(svc)["routes.contracts"]
    assert c["status"] == "pass", c
    assert c["detail"] == ("no contracts for a served head; kept but not consulted while the head is disabled in "
                           "rlcd.yaml: color, good")
    assert not [f for f in check_project(svc.project, svc.pack) if f["path"].startswith("routes")]
    # with one head served again, its contracts are checked and summarised as before
    svc = make(tmp_path, trained_home, {"heads": {"good": {"enabled": False}}, "routes": routes}, pack=Switchable())
    c = checks(svc)["routes.contracts"]
    assert c["status"] == "pass" and c["detail"] == (
        "contracts for color; options without a contract (never filtered): color.green, color.blue; kept but not "
        "consulted while the head is disabled in rlcd.yaml: good")
    # a pack that serves the head anyway (it ignores `enabled`) has its contracts checked like any other
    c = checks(make(tmp_path, trained_home, {"heads": {"good": {"enabled": False}}, "routes": routes}))
    assert c["routes.contracts"]["status"] == "fail" and "is a noul head" in c["routes.contracts"]["detail"]


@pytest.mark.parametrize(("option", "escape"), [
    ("other", True), ("Out-of-scope", True), ("escalate_human", True), ("clarify_question", True),
    ("none_of_these", True), ("unknown", True), ("needs review", True), ("no_match", True),
    ("blue", False), ("nonetheless", False), ("others_team", False)])
def test_what_counts_as_an_escape_option(option, escape):
    assert CL._is_escape(option) is escape


# ----------------------------------------------------------------------------- security and secrets

@pytest.mark.parametrize("host", ["0.0.0.0", "10.1.2.3", "rlcd.example.com"])
def test_serving_off_loopback_needs_an_api_key(trained_home, tmp_path, monkeypatch, host):
    svc = make(tmp_path, trained_home, {"serve": {"host": host}})
    res = CL.run(svc)
    c = by_id(res)["security.api_key"]
    assert c["status"] == "fail" and f"serve.host is {host}" in c["detail"] and res["ok"] is False
    monkeypatch.setenv("RLCD_API_KEY", "k-s3cret-value")
    res = CL.run(svc)
    assert by_id(res)["security.api_key"]["status"] == "pass" and res["ok"] is True
    assert "k-s3cret-value" not in json.dumps(res)          # only whether a key is set, never its value


@pytest.mark.parametrize("host", ["127.0.0.1", "localhost", "::1"])
def test_loopback_serving_passes_without_a_key(trained_home, tmp_path, host):
    c = checks(make(tmp_path, trained_home, {"serve": {"host": host}}))["security.api_key"]
    assert c["status"] == "pass" and "loopback" in c["detail"]


def test_remote_hosts_warn_and_the_admin_key_is_reported(trained_home, tmp_path, monkeypatch):
    c = checks(make(tmp_path, trained_home, {"security": {"allow_remote": True}}))
    assert c["security.allow_remote"]["status"] == "warn" and "security.allow_remote" in c["security.allow_remote"][
        "detail"]
    assert c["security.admin_key"]["status"] == "info" and "not set" in c["security.admin_key"]["detail"]
    monkeypatch.setenv("RLCD_ALLOW_REMOTE", "1")
    monkeypatch.setenv("RLCD_ADMIN_KEY", "zz-admin-value-zz")
    res = CL.run(make(tmp_path, trained_home))
    c = by_id(res)
    assert c["security.allow_remote"]["status"] == "warn" and "RLCD_ALLOW_REMOTE=1" in c["security.allow_remote"][
        "detail"]
    assert c["security.admin_key"]["status"] == "info" and "RLCD_ADMIN_KEY is set" in c["security.admin_key"]["detail"]
    assert "zz-admin-value-zz" not in json.dumps(res)


@pytest.mark.parametrize(("redact", "status"), [(False, "warn"), (True, "pass"), (["secrets", "pii"], "pass"),
                                                ("everything", "fail")])
def test_redaction_setting(trained_home, tmp_path, redact, status):
    c = checks(make(tmp_path, trained_home, {"serve": {"redact": redact}}))["secrets.redact"]
    assert c["status"] == status
    if status == "fail":                          # an invalid setting is the service's own 500 bad_config
        assert "bad_config" in c["detail"]


def test_log_states_and_receipts_mode_are_reported(trained_home, tmp_path):
    c = checks(make(tmp_path, trained_home))
    off = checks(make(tmp_path, trained_home, {"serve": {"log_states": False, "receipts": "all"}}))
    assert c["secrets.log_states"]["status"] == off["secrets.log_states"]["status"] == "info"
    assert "is on" in c["secrets.log_states"]["detail"] and "is off" in off["secrets.log_states"]["detail"]
    assert c["receipts.mode"]["detail"].startswith("serve.receipts: decisions")
    assert off["receipts.mode"]["status"] == "info" and off["receipts.mode"]["detail"].startswith("serve.receipts: all")
    bad = checks(make(tmp_path, trained_home, {"serve": {"receipts": "sometimes"}}))["receipts.mode"]
    assert bad["status"] == "fail" and "bad_config" in bad["detail"]


# ----------------------------------------------------------------------------- receipts, bundles, rollback

def test_a_tampered_decision_line_fails_the_receipt_chain(trained_home, tmp_path):
    svc = make(tmp_path, trained_home)
    for s in STATES:
        svc.decide(DecideRequest(state=s))
    c = checks(svc)["receipts.chain"]
    assert c["status"] == "pass" and c["detail"].startswith("3 receipts chained")
    path = svc.logs.decision_log.path
    lines = path.read_text(encoding="utf-8").splitlines()
    row = json.loads(lines[1])
    row["action"] = "confirm" if row["action"] == "act" else "act"        # an approval rewritten after the fact
    lines[1] = json.dumps(row, separators=(",", ":"))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    res = CL.run(svc)
    c = by_id(res)["receipts.chain"]
    assert c["status"] == "fail" and "first at line 2" in c["detail"] and res["ok"] is False


def test_the_rollback_target_must_load(trained_home, tmp_path):
    svc = make(tmp_path, trained_home)
    assert svc.trainer.start(TrainRequest(heads=["good"]), wait=True)["status"] == "succeeded"
    c = checks(svc)
    assert c["rollback.ready"]["status"] == "pass" and "at rlcd-0.1.1" in c["rollback.ready"]["detail"]
    assert checks(svc, model="rlcd-0.1.1")["model.bundle"]["detail"].startswith("rlcd-0.1.1:")
    head_json = svc.home / "models" / "rlcd-0.1.1" / "heads" / "color" / "head.json"
    head_json.write_text(head_json.read_text(encoding="utf-8") + " ", encoding="utf-8")   # no longer its sha256
    res = CL.run(svc)
    c = by_id(res)
    assert c["rollback.ready"]["status"] == "fail" and "rlcd-0.1.1, which does not load" in c["rollback.ready"][
        "detail"]
    assert "sha256" in c["rollback.ready"]["detail"]
    assert c["model.bundle"]["status"] == "pass"          # the served version keeps its own copy of the head
    assert str(svc.home) not in json.dumps(res)           # absolute paths are written as RLCD_HOME


# ----------------------------------------------------------------------------- rollout and kill switch

def test_rollout_mode_and_the_kill_switch(trained_home, tmp_path, monkeypatch):
    svc = make(tmp_path, trained_home, {"rollout": {"mode": "limited", "limited": {"stakes": ["low"],
                                                                                   "heads": ["color"]}}})
    c = checks(svc)
    assert c["rollout.mode"]["status"] == "info" and c["rollout.mode"]["detail"].startswith("limited: act only at")
    assert "color" in c["rollout.mode"]["detail"]
    assert c["automation.switch"]["status"] == "info" and "automation is on" in c["automation.switch"]["detail"]
    svc.set_automation(False, reason="incident 42", by="ops")
    c = checks(svc)["automation.switch"]
    assert c["status"] == "warn" and "incident 42" in c["detail"] and "by ops" in c["detail"]
    svc.set_automation(True, reason="resolved", by="ops")
    monkeypatch.setenv("RLCD_AUTOMATION", "off")
    c = checks(svc)["automation.switch"]
    assert c["status"] == "warn" and "RLCD_AUTOMATION=off" in c["detail"]
    bad = checks(make(tmp_path, trained_home, {"rollout": {"mode": "yolo"}}))["rollout.mode"]
    assert bad["status"] == "fail" and "bad_config" in bad["detail"]


# ----------------------------------------------------------------------------- batteries

def test_battery_questions_must_bind_to_trained_heads(trained_home, tmp_path):
    c = checks(make(tmp_path, trained_home, {"batteries": BATTERY}))["batteries.bound"]
    assert c["status"] == "pass" and c["detail"].startswith("pre@1:")
    for q, why in (({"type": "noul", "head": "nope"}, "no head 'nope'"),
                   ({"type": "choice", "head": "good"}, "answers a noul question")):
        bats = {"pre": {**BATTERY["pre"], "questions": {"g": q}}}
        res = CL.run(make(tmp_path, trained_home, {"batteries": bats}))
        c = by_id(res)["batteries.bound"]
        assert c["status"] == "fail" and why in c["detail"] and res["ok"] is False
    unowned = {"pre": {k: v for k, v in BATTERY["pre"].items() if k != "owner"}}
    c = checks(make(tmp_path, trained_home, {"batteries": unowned}))["batteries.bound"]
    assert c["status"] == "warn" and "pre@1 has no owner" in c["detail"]


def test_battery_questions_on_untrained_heads_warn(tmp_path):
    c = checks(make(tmp_path, data={"batteries": BATTERY}))["batteries.bound"]
    assert c["status"] == "warn" and "pre@1 g (good)" in c["detail"]


def test_a_battery_changed_without_a_version_bump_fails(trained_home, tmp_path):
    svc = make(tmp_path, trained_home, {"batteries": BATTERY})
    unseen = checks(svc)["batteries.versions"]          # a battery no receipt names: a change would go unnoticed
    assert unseen["status"] == "warn" and "goes unnoticed" in unseen["detail"]
    r = svc.systemone(SystemOneRequest(battery="pre", state=STATES[0], receipt=True))
    assert r["battery"]["version"] == "1" and r["receipt"]["id"]
    assert checks(svc)["batteries.versions"]["status"] == "pass"
    svc.project.data["batteries"]["pre"]["questions"]["c"] = {"type": "choice", "head": "color"}   # a new meaning
    res = CL.run(svc)
    c = by_id(res)["batteries.versions"]
    assert c["status"] == "fail" and c["detail"].startswith("changed without a version bump: pre@1: 1 receipt(s)")
    assert res["ok"] is False
    svc.project.data["batteries"]["pre"]["version"] = "2"
    c = checks(svc)["batteries.versions"]
    assert c["status"] == "pass" and "receipts of earlier versions: pre@1" in c["detail"]


# ----------------------------------------------------------------------------- regression cases

def _write_cases(path, cases, extra: str = "") -> None:
    path.write_text("".join(json.dumps(x) + "\n" for x in cases) + extra, encoding="utf-8")


def test_regression_cases_run_through_core_regress(trained_home, tmp_path):
    svc = make(tmp_path, trained_home)
    c = checks(svc)["regressions.cases"]
    assert c["status"] == "info" and c["detail"].startswith("no regression cases")
    got = svc.head_decision("rlcd-0.1.1", svc.spec("color", "rlcd-0.1.1"), STATES[1], explain=False)["decision"]
    other = next(x for x in ("red", "green", "blue") if x != got["class"])
    cases = [{"id": "c1", "kind": "head", "head": "color", "state": STATES[1], "expect": {"class": got["class"]}}]
    path = tmp_path / "regressions.jsonl"
    _write_cases(path, cases)
    c = checks(svc)["regressions.cases"]
    assert c["status"] == "pass" and c["detail"].startswith("all 1 case(s) in regressions.jsonl pass on rlcd-0.1.1")
    cases.append({"id": "c2", "kind": "head", "head": "color", "state": STATES[1], "expect": {"class": other}})
    _write_cases(path, cases)
    res = CL.run(svc)
    c = by_id(res)["regressions.cases"]
    assert c["status"] == "fail" and res["ok"] is False
    assert c["detail"].startswith("1 of 2 case(s) in regressions.jsonl fail on rlcd-0.1.1: c2: ")
    _write_cases(path, cases, "{not json\n")                     # a line that is not a case cannot pass
    c = checks(svc)["regressions.cases"]
    assert c["status"] == "fail" and c["detail"].startswith("2 of 3 case(s)") and "could not be judged" in c["detail"]
    assert not svc.logs.decision_log.path.exists()                 # cases run in a dry run: nothing was logged
    c = checks(make(tmp_path, trained_home, {"regressions": "cases/none.jsonl"}))["regressions.cases"]
    assert c["status"] == "warn" and "cases/none.jsonl" in c["detail"]


def test_regression_cases_are_counted_when_the_runner_is_missing(trained_home, tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, CL.REGRESS_MODULE, None)        # as if open_rlcd.core.regress were missing
    svc = make(tmp_path, trained_home)
    _write_cases(tmp_path / "regressions.jsonl", [{"id": "c1"}, {"id": "c2"}])
    c = checks(svc)["regressions.cases"]
    assert c["status"] == "info" and c["detail"].startswith("2 regression case(s) in regressions.jsonl")
    assert "not run" in c["detail"]


# ----------------------------------------------------------------------------- review load and owners

def _log(svc: Service, action: str, hours_ago: float, **receipt) -> None:
    t = (NOW - timedelta(hours=hours_ago)).strftime("%Y-%m-%dT%H:%M:%SZ")
    svc.logs.add_receipt({"created_at": t, "kind": "decide", "model": "rlcd-0.1.1", "heads": {}, "action": action,
                          "receipt": dict(receipt)})


def test_review_capacity_against_the_last_seven_days_of_decisions(trained_home, tmp_path):
    svc = make(tmp_path, trained_home, {"serve": {"review_capacity_per_day": 3}})
    c = checks(svc, now=NOW)["review.capacity"]
    assert c["status"] == "info" and c["detail"].startswith("no decisions in the last 7 days")
    for h in (1, 2, 3, 4, 5):
        _log(svc, "confirm", h)                          # five reviews in the last 24 hours
    _log(svc, "confirm", 30, rollout={"capped": True})   # one the day before, capped by the rollout
    _log(svc, "escalate", 2)
    _log(svc, "act", 3)
    _log(svc, "confirm", 24 * 10)                         # outside the window
    c = checks(svc, now=NOW)["review.capacity"]
    assert c["status"] == "warn"
    assert c["detail"].startswith("6 confirm decision(s) over the last 1.25 day(s): 4.8 a day on average, 5 on the "
                                  "busiest UTC day (2026-11-09); 1 day(s) above the review capacity of 3 a day "
                                  "(2026-11-09); so is the average")
    assert "1 more abstained" in c["detail"] and "1 of the reviews were capped" in c["detail"]
    assert checks(make(tmp_path, trained_home, {"serve": {"review_capacity_per_day": 10}}), now=NOW)[
        "review.capacity"]["status"] == "pass"
    c = checks(make(tmp_path, trained_home), now=NOW)["review.capacity"]
    assert c["status"] == "warn" and "not declared" in c["detail"]
    c = checks(make(tmp_path, trained_home, {"serve": {"review_capacity_per_day": "plenty"}}), now=NOW)
    assert c["review.capacity"]["status"] == "fail"


def test_owners(trained_home, tmp_path):
    full = {"pack": "search-platform", "heads": "ml-team", "policy": "risk", "review": ["ops-a", "ops-b"],
            "incidents": "sre"}
    c = checks(make(tmp_path, trained_home, {"owners": full}))["owners"]
    assert c["status"] == "pass" and "review: ops-a, ops-b" in c["detail"]
    partial = {k: v for k, v in full.items() if k not in ("review", "incidents")}
    c = checks(make(tmp_path, trained_home, {"owners": partial}))["owners"]
    assert c["status"] == "warn" and c["detail"].startswith("no owner named for review (")
    assert "incidents (" in c["detail"] and "pack (" not in c["detail"]
    c = checks(make(tmp_path, trained_home, {"owners": {**full, "policy": "  "}}))["owners"]
    assert c["status"] == "warn" and "policy (" in c["detail"]
