"""Tabular pack end to end through core.service.Service: train on a synthetic CSV, answer JSON rows, decide,
reload the bundle, take feedback, abstain on noise. Offline; logistic regression unless a test says otherwise."""
from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pytest

from open_rlcd.core import checklist as CL
from open_rlcd.core.config import Project
from open_rlcd.core.contract import ContractError, DecideRequest, FeedbackRequest, SystemOneRequest, TrainRequest
from open_rlcd.core.head import lightgbm_available
from open_rlcd.core.service import Service
from open_rlcd.packs.tabular import generate, synthetic_pack_config, true_probabilities, write_synthetic_csv

HEADS = ("churned", "plan_tier_next", "satisfaction")
QUESTIONS = {"churned": {"type": "noul"}, "plan_tier_next": {"type": "choice"}, "satisfaction": {"type": "score"}}


def make(tmp: Path, csv: Path, model: str | None = "logreg", **extra) -> tuple[Project, Service]:
    data = {"pack": "tabular", "pack_config": synthetic_pack_config(csv), **extra}
    if model:
        data["heads"] = {h: {"model": model} for h in HEADS}
    project = Project(root=tmp, data=data)
    return project, Service(project, home=tmp / "var")


def train(svc: Service, **kw) -> dict:
    job = svc.trainer.start(TrainRequest(**kw), wait=True)
    assert job["status"] == "succeeded", job.get("error")
    return job


def features(row: dict) -> dict:
    """A table row as a client would send it: the feature columns, numbers as JSON numbers."""
    out = {}
    for c in ("tenure_months", "monthly_charges", "support_calls"):
        out[c] = None if row[c] == "" else float(row[c])
    for c in ("plan", "contract", "region", "paperless"):
        out[c] = row[c] or None
    return out


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("tabular")
    csv = tmp / "churn.csv"
    write_synthetic_csv(csv, n=4000, seed=7)
    project, svc = make(tmp, csv)
    job = train(svc)
    return {"tmp": tmp, "csv": csv, "project": project, "svc": svc, "job": job}


def test_signal_heads_have_skill_and_calibration(trained):
    heads = trained["job"]["heads"]
    for name in HEADS:
        m = heads[name]["metrics"]
        assert m["has_skill"], (name, m)
        assert m["log_loss"] < m["baseline_log_loss"] and m["brier_skill_score"] > 0.05
        assert m["ece"] < 0.07, (name, m)
        assert heads[name]["data"]["ordering"] == "group"
    churn = heads["churned"]
    assert churn["metrics"]["ece"] < 0.04 and churn["verdict"]["medium"]["max_tier"] == "act"
    assert churn["data"]["by_source"] == {"ground_truth": 4000}


def test_probabilities_track_the_generating_model(trained):
    svc = trained["svc"]
    rows, truth = generate(1500, seed=99)                    # rows the model never saw
    version = svc.resolve(None)
    P, calibrated = svc.probabilities(version, svc.spec("churned", version), [features(r) for r in rows])
    assert calibrated
    assert np.abs(P[:, 1] - truth["churned"]).mean() < 0.04
    P, _ = svc.probabilities(version, svc.spec("plan_tier_next", version), [features(r) for r in rows])
    assert np.abs(P - truth["plan_tier_next"]).sum(1).mean() < 0.15


def test_systemone_answers_a_json_row(trained):
    svc = trained["svc"]
    body = json.loads('{"state": {"tenure_months": 3, "monthly_charges": 99.5, "support_calls": 5, '
                      '"plan": "premium", "contract": "month-to-month", "region": "north", "paperless": "yes", '
                      '"customer_id": "ignored"}, "questions": ' + json.dumps(QUESTIONS) + '}')
    out = svc.systemone(SystemOneRequest(**body))
    a = out["answers"]
    assert out["model"] == svc.registry.latest_id() and "warnings" not in out
    assert a["churned"]["head"] == "churned" and a["churned"]["calibrated"] is True
    assert 0.5 < a["churned"]["noul"] < 1 and 0 <= a["churned"]["confidence"] <= 1
    truth = true_probabilities([{k: (str(v) if v is not None else None) for k, v in body["state"].items()}])
    assert abs(a["churned"]["noul"] - truth["churned"][0]) < 0.12
    assert abs(sum(a["plan_tier_next"]["probabilities"].values()) - 1) < 1e-5
    assert a["satisfaction"]["legend"]["0"] == "very unhappy" and a["satisfaction"]["score"] in range(4)


def test_decide_holds_policy_and_gives_reasons(trained):
    svc = trained["svc"]
    row = features(generate(5, seed=1)[0][0])
    d = svc.generic_decide(DecideRequest(state=row, heads=["churned"]))
    assert d["action"] in ("act", "confirm", "hold") and d["policy"]["abstain"] == "hold"
    r = d["heads"]["churned"]
    assert r["decision"]["head"] == "churned" and r["answer"]["calibrated"] is True
    assert r["reasons"] and {x["feature"] for x in r["reasons"]} <= set(svc.provider(d["model"]).names("row"))
    assert set(r["reason_groups"]) <= {"tenure_months", "monthly_charges", "support_calls", "plan", "contract",
                                       "region", "paperless"}


def test_unseen_category_is_answered(trained):
    svc = trained["svc"]
    row = features(generate(5, seed=1)[0][0])
    seen = svc.systemone(SystemOneRequest(state=row, questions=QUESTIONS))
    unseen = svc.systemone(SystemOneRequest(state={**row, "plan": "enterprise", "region": "mars"},
                                            questions=QUESTIONS))
    assert unseen["answers"]["churned"]["calibrated"] is True
    assert 0 < unseen["answers"]["churned"]["noul"] < 1
    assert seen["answers"]["churned"]["noul"] != unseen["answers"]["churned"]["noul"]


def test_missing_columns_are_a_422(trained):
    svc = trained["svc"]
    row = features(generate(5, seed=1)[0][0])
    del row["support_calls"]
    with pytest.raises(ContractError) as e:
        svc.systemone(SystemOneRequest(state=row, questions={"churned": {"type": "noul"}}))
    assert e.value.status == 422 and e.value.details["missing_columns"] == ["support_calls"]
    from fastapi.testclient import TestClient

    from open_rlcd.serve.api import create_app

    r = TestClient(create_app(service=svc)).post("/v1/systemone", json={"state": row, "questions": QUESTIONS})
    assert r.status_code == 422 and r.json()["missing_columns"] == ["support_calls"]


def test_bundle_reload_gives_identical_probabilities(trained):
    svc = trained["svc"]
    rows = [features(r) for r in generate(300, seed=5)[0]]
    version = svc.resolve(None)
    before = {h: svc.probabilities(version, svc.spec(h, version), rows)[0] for h in HEADS}
    # a fresh service whose project no longer has the training file: the provider must come from the bundle
    gone = Project(root=trained["tmp"], data={"pack": "tabular", "pack_config": {
        **synthetic_pack_config(trained["tmp"] / "moved-away.csv")}})
    fresh = Service(gone, home=trained["tmp"] / "var")
    assert fresh.resolve(None) == version
    for h in HEADS:
        after, calibrated = fresh.probabilities(version, fresh.spec(h, version), rows)
        assert calibrated and np.array_equal(before[h], after), h
    meta = fresh.bundle(version).meta
    assert fresh.provider(version).to_config() == meta["pack"]["config"]
    assert meta["feature_version"] == fresh.provider(version).feature_version


def test_feedback_rows_join_the_next_train(tmp_path):
    csv = tmp_path / "churn.csv"
    write_synthetic_csv(csv, n=1500, seed=4)
    project, svc = make(tmp_path, csv)
    train(svc, heads=["churned"])
    rows, _ = generate(6, seed=21)
    for r in rows[:4]:
        svc.feedback(FeedbackRequest(head="churned", state={**features(r), "customer_id": r["customer_id"]},
                                     label="yes", source="crm"))
    # a decision's logged state is reused when feedback names its decision_id
    d = svc.generic_decide(DecideRequest(state=features(rows[4]), heads=["churned"]))
    fb = svc.feedback(FeedbackRequest(decision_id=d["decision_id"], label="no", source="crm"))
    assert fb["stored"] and fb["head"] == "churned" and fb["label"] == "false"
    # precomputed features of the current schema are used; a state of another schema is skipped, not fatal
    version = svc.resolve(None)
    prov = svc.provider(version)
    x = prov.transform("row", [features(rows[5])])[0].tolist()
    svc.feedback(FeedbackRequest(head="churned", x=[None if np.isnan(v) else v for v in x],
                                 feature_names=prov.names("row"), feature_version=prov.feature_version,
                                 label="no", source="crm"))
    svc.feedback(FeedbackRequest(head="churned", state={"tenure_months": 1}, label="yes", source="crm"))
    job = train(svc, heads=["churned"])
    h = job["heads"]["churned"]
    assert h["data"]["by_source"]["feedback"] == 6
    assert any("6 feedback rows added to training (weight x 5)" in n for n in h["notes"])
    assert any("1 feedback rows" in n and "skipped" in n for n in h["notes"])
    assert h["data"]["holdout_rows"] <= h["data"]["by_source"]["ground_truth"]     # feedback is never scored


def test_a_decisions_outcome_counts_once(tmp_path):
    """Two POSTs of an outcome for one decision_id (a retry, or a correction) are one training row: the later
    one. As in packs.base.dataset_from_labels and the Bernoulli ledger."""
    csv = tmp_path / "churn.csv"
    write_synthetic_csv(csv, n=1500, seed=4)
    project, svc = make(tmp_path, csv)
    train(svc, heads=["churned"])
    rows, _ = generate(3, seed=22)
    d = svc.generic_decide(DecideRequest(state=features(rows[0]), heads=["churned"]))
    svc.feedback(FeedbackRequest(decision_id=d["decision_id"], label="yes", source="crm"))
    svc.feedback(FeedbackRequest(decision_id=d["decision_id"], label="yes", source="crm"))     # the retry
    e = svc.generic_decide(DecideRequest(state=features(rows[1]), heads=["churned"]))
    svc.feedback(FeedbackRequest(decision_id=e["decision_id"], label="yes", source="crm"))
    svc.feedback(FeedbackRequest(decision_id=e["decision_id"], label="no", source="crm"))      # the correction
    # outcomes without a decision_id are separate cases, even with identical states
    for _ in range(2):
        svc.feedback(FeedbackRequest(head="churned", state=features(rows[2]), label="no", source="crm"))
    pack, prov = svc.pack, svc.pack.provider(project)
    ds = pack.datasets(project, prov, ["churned"], feedback=svc.logs.feedback())["churned"]
    fb = ds.source == "feedback"
    assert int(fb.sum()) == 4
    assert sorted(ds.y[fb].tolist()) == [0, 0, 0, 1]            # the correction ("no") replaced the first "yes"
    assert any("2 feedback rows replaced by a later outcome posted for the same decision_id" in n
               for n in ds.notes)
    h = train(svc, heads=["churned"])["heads"]["churned"]
    assert h["data"]["by_source"]["feedback"] == 4


def test_noise_has_no_skill_and_holds(tmp_path):
    csv = tmp_path / "noise.csv"
    # labels independent of the features. Seed 3: see the note on the core skill test's false-positive rate in
    # this pack's report (a near-constant forecast slightly closer to the holdout's own base rate can pass it)
    write_synthetic_csv(csv, n=4000, seed=3, noise=True)
    project, svc = make(tmp_path, csv)
    job = train(svc)
    for name in HEADS:
        h = job["heads"][name]
        assert h["metrics"]["has_skill"] is False, (name, h["metrics"])
        assert h["verdict"]["medium"]["max_tier"] == "hold"
    row = features(generate(5, seed=1)[0][0])
    d = svc.generic_decide(DecideRequest(state=row))
    assert d["action"] == "hold" and d["tier"] == "hold"
    assert all(r["decision"]["tier"] == "hold" for r in d["heads"].values())
    assert any("no skill over the base rate" in w for w in d["warnings"])
    version = svc.resolve(None)
    warnings: list[str] = []
    r = svc.head_decision(version, svc.spec("churned", version), row, stakes="low", warnings=warnings)
    assert r["decision"]["tier"] == "hold" and r["decision"]["max_tier"] == "hold"
    assert any("no skill" in w for w in warnings)
    answers = svc.systemone(SystemOneRequest(state=row, questions=QUESTIONS))
    assert any("no skill" in w for w in answers["warnings"])


def test_default_model_auto(tmp_path):
    csv = tmp_path / "churn.csv"
    write_synthetic_csv(csv, n=2500, seed=8)
    project, svc = make(tmp_path, csv, model=None, train={"n_estimators": 60})
    assert svc.pack.heads(project)["churned"].model == "auto"
    job = train(svc, heads=["churned"])
    entry = svc.registry.entry(job["model"])["heads"]["churned"]
    assert entry["model"] == ("lightgbm" if lightgbm_available() else "logreg")
    assert job["heads"]["churned"]["metrics"]["has_skill"]
    row = features(generate(5, seed=1)[0][0])
    d = svc.generic_decide(DecideRequest(state=row, heads=["churned"]))
    assert d["heads"]["churned"]["reasons"]


def test_hashed_categories_train_serve_and_carry_over(tmp_path):
    """F23: with hash_categories the bundle holds no raw category value, answers JSON rows like raw mode, and a
    retrain of one head carries the others over (one salt per project home keeps the feature schema)."""
    csv = tmp_path / "churn.csv"
    write_synthetic_csv(csv, n=1500, seed=4)
    data = {"pack": "tabular", "pack_config": {**synthetic_pack_config(csv), "hash_categories": True},
            "heads": {h: {"model": "logreg"} for h in HEADS}}
    project = Project(root=tmp_path, data=data)
    svc = Service(project, home=tmp_path / "var")
    v1 = train(svc)["model"]
    b = svc.bundle(v1)
    blob = (b.path / "model.json").read_text() + "".join(p.read_text() for p in b.path.rglob("ensemble.json"))
    for value in ("basic", "premium", "north", "month-to-month", "one-year"):
        assert f"={value}" not in blob and f'"{value}"' not in blob, value
    row = {"tenure_months": 3, "monthly_charges": 99.5, "support_calls": 5, "plan": "Premium",
           "contract": "month-to-month", "region": "north", "paperless": "yes"}
    a = svc.systemone(SystemOneRequest(state=row, questions=QUESTIONS))["answers"]
    truth = true_probabilities([{k: str(v).lower() for k, v in row.items()}])   # "Premium": case-only match
    assert a["churned"]["calibrated"] is True and abs(a["churned"]["noul"] - truth["churned"][0]) < 0.15
    v2 = train(svc, heads=["churned"])["model"]
    meta = svc.bundle(v2).meta
    assert set(meta["heads"]) == set(HEADS) and meta["feature_version"] == b.meta["feature_version"]
    assert not any("not carried over" in n for n in meta.get("notes") or [])


# ----------------------------------------------------------------------------- targets the project switches off

ROUTES = {"next_plan": {"up": {"requires": ["upsell_api"]}}}
ROW = {"x": 0.9, "y": 0.5, "plan": "a"}


def small_table(path: Path, n: int = 600) -> Path:
    """Two label columns: churned (noul, from x) and next_plan (choice up / same / down, from y)."""
    rnd = random.Random(0)
    lines = ["x,y,plan,churned,next_plan"]
    for i in range(n):
        x, y = rnd.random(), rnd.random()
        churned = "yes" if x + rnd.random() * 0.3 > 0.7 else "no"
        lines.append(f"{x:.3f},{y:.3f},{'ab'[i % 2]},{churned},{('up', 'same', 'down')[int(y * 3)]}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def small(tmp: Path, next_plan: dict | None, pack_config: dict | None = None, **extra) -> Service:
    """A service on the small table with the target `churned`, and `next_plan` (with these fields) unless None."""
    targets = {"churned": {"label": "churned", "model": "logreg"}}
    if next_plan is not None:
        targets["next_plan"] = {"label": "next_plan", "model": "logreg", **next_plan}
    pc = {"data": "small.csv", "targets": targets, **(pack_config or {})}
    return Service(Project(root=tmp, data={"pack": "tabular", "pack_config": pc, **extra}), home=tmp / "var")


def checked(svc: Service) -> dict:
    """`rlcd check` on rlcd-latest: {check id: {status, detail, ...}}."""
    return {c["id"]: c for c in CL.run(svc)["checks"]}


def kept(svc: Service, version: str) -> None:
    """next_plan is trained in `version` and switched off: the service does not serve it, and `rlcd check` keeps its
    route contract (and any rollout mode) as a switched-off head's."""
    assert set(svc.specs(version)) == {"churned"} and svc.switched_off(version) == {"next_plan"}
    assert "next_plan" in svc.bundle(version).names()
    with pytest.raises(ContractError) as e:
        svc.systemone(SystemOneRequest(state=ROW, questions={"next_plan": {"type": "choice"}}, model=version))
    assert e.value.kind == "unknown_head"
    assert set(svc.generic_decide(DecideRequest(state=ROW, model=version))["heads"]) == {"churned"}
    c = checked(svc)
    assert c["routes.contracts"]["status"] == "pass" and c["rollout.mode"]["status"] == "info"
    assert c["routes.contracts"]["detail"] == ("no contracts for a served head; kept but not consulted while the "
                                               "head is switched off in rlcd.yaml: next_plan")


def test_a_target_switched_off_in_pack_config_is_not_served_before_or_after_a_retrain(tmp_path):
    """`pack_config.targets.<head>.enabled: false` switches the head off as `heads.<head>.enabled: false` does: not
    served, at once and without a retrain, nor after a retrain, although each train carries the trained head over
    (its label column stays out of the features, so the feature version does not change). A partial train still
    carries the heads it does not retrain. `rlcd check` and config check read the list the service reads
    (`switched_off`): the head's route contract and rollout mode are kept, compared with nothing, and fail
    nothing. Switched on again, the trained head is served at once."""
    small_table(tmp_path / "small.csv")
    svc = small(tmp_path, {})
    v1 = train(svc)["model"]
    job = train(svc, heads=["churned"])
    v2 = job["model"]
    assert set(job["heads"]) == {"churned"} and set(svc.specs(v2)) == {"churned", "next_plan"}
    assert svc.bundle(v2).meta["heads"]["next_plan"]["trained_in"] == v1
    off = small(tmp_path, {"enabled": False}, routes=ROUTES, rollout={"heads": {"next_plan": "shadow"}})
    assert off.project.check(off.pack) == []
    kept(off, v2)                                            # before a retrain
    job = train(off)
    v3 = job["model"]
    assert set(job["heads"]) == {"churned"}
    assert off.bundle(v3).meta["heads"]["next_plan"]["trained_in"] == v1      # carried over, never retrained
    kept(off, v3)                                            # after it
    assert checked(off)["config.valid"]["status"] == "pass"
    on = small(tmp_path, {}, routes=ROUTES)
    assert on.specs(v3)["next_plan"] == on.bundle(v3).head("next_plan").spec
    assert checked(on)["routes.contracts"]["detail"].startswith("contracts for next_plan")


@pytest.mark.parametrize("keep_out", [{"exclude": ["next_plan"]}, {"features": ["x", "y", "plan"]}, None],
                         ids=["exclude", "features", "auto"])
def test_a_target_removed_from_pack_config_is_not_served_either(tmp_path, keep_out):
    """Removing a target switches its head off the same way, at once. While its column stays out of the features
    (under `exclude`, outside an explicit `features` list) the feature version does not change, so each train
    carries the trained head over, `rlcd check` keeps its contract as a switched-off head's, and listing the target
    again serves it again without a retrain. Under `features: auto` the column becomes a feature: the next train
    drops the head, and its contract fails `rlcd check` (no model has that head). Config check, which sees no
    model, warns either way that the name is not a head of this project."""
    small_table(tmp_path / "small.csv")
    v1 = train(small(tmp_path, {}, keep_out))["model"]
    gone = small(tmp_path, None, keep_out, routes=ROUTES)
    findings = gone.project.check(gone.pack)
    assert [(f["level"], f["path"]) for f in findings] == [("warn", "routes.next_plan")]
    assert "not a head of this project" in findings[0]["message"]
    kept(gone, v1)
    v2 = train(gone)["model"]
    assert set(gone.specs(v2)) == {"churned"}
    if keep_out is not None:
        kept(gone, v2)
        back = small(tmp_path, {}, keep_out)
        assert back.specs(v2)["next_plan"] == back.bundle(v2).head("next_plan").spec
    else:
        assert gone.bundle(v2).names() == ["churned"] and gone.switched_off(v2) == set()
        c = checked(gone)["routes.contracts"]
        assert c["status"] == "fail" and f"{v2} has no head 'next_plan'" in c["detail"]
