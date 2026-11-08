"""Guarded weight edits: an edit is a claim and the holdout decides.

  (a) a harmful edit (a flipped coefficient) is recalibrated, scored, rejected, kept for inspection, never served;
  (b) an edit that removes a planted spurious feature is promoted when the holdout supports it, and only then;
  (c) provenance (sha256, author, parameters touched, base version, paired test) is in model.json, the
      hash-checked head.json and MODEL_CARD.md;
  (d) edited bundles still pass `bundle.validate`, tampered ones still fail, and nothing is ever a pickle;
  (e) the holdout only judges: edits tuned on holdout rows are refused before anything is scored, and every
      further candidate scored on the same holdout faces a tighter bound;
  (f) an edit is judged against its base: changed rows, a moved rlcd-latest, malformed deltas and LightGBM members
      are refused or not promoted.
"""
import json
import shutil
from pathlib import Path

import numpy as np
import pytest
from safetensors.numpy import save_file

from open_rlcd import LATEST
from open_rlcd.cli import main
from open_rlcd.core import bundle as BD
from open_rlcd.core import metrics as M
from open_rlcd.core import patch as PT
from open_rlcd.core.config import Project, write_config
from open_rlcd.core.contract import ContractError, SystemOneRequest, TrainRequest
from open_rlcd.core.head import CalibratedEnsemble, HeadSpec, lightgbm_available
from open_rlcd.core.registry import Registry
from open_rlcd.core.service import Service
from open_rlcd.core.training import TrainConfig, train_head
from open_rlcd.report.model_card import model_card
from tests.leakpack import HoldingLeakPack, LeakPack, clean_rows, leak_rows


def trained_service(tmp_path: Path, pack=None) -> Service:
    svc = Service(Project(root=tmp_path), home=tmp_path / "home", pack=pack or LeakPack())
    job = svc.trainer.start(TrainRequest(), wait=True)
    assert job["status"] == "succeeded", job.get("error")
    return svc


def members(svc: Service, head: str, version: str | None = None):
    return svc.bundle(version or svc.registry.latest_id()).head(head).ensemble.members


def no_leak_delta(svc: Service, path: Path, version: str | None = None, **meta) -> Path:
    """Deltas that zero the `leak` coefficient in every member of `good`."""
    ms = members(svc, "good", version)
    li = svc.bundle(version or svc.registry.latest_id()).head("good").ensemble.feature_names.index("leak")
    delta = {}
    for k, (base, _) in enumerate(ms):
        d = np.zeros_like(base.coef)
        d[:, li] = -base.coef[:, li]
        delta[f"member_{k}.coef"] = d
    save_file(delta, str(path), metadata={k: str(v) for k, v in meta.items()} or None)
    return path


def flip_delta(svc: Service, path: Path) -> Path:
    """Deltas that flip the sign of the strongest coefficient of `clean` in every member."""
    ms = members(svc, "clean")
    j = int(np.argmax(np.abs(ms[0][0].coef[0])))
    delta = {}
    for k, (base, _) in enumerate(ms):
        d = np.zeros_like(base.coef)
        d[:, j] = -2 * base.coef[:, j]
        delta[f"member_{k}.coef"] = d
    save_file(delta, str(path))
    return path


def manual(**kw):
    return {"author": "manual", "tuned_on": "development", **kw}


# ----------------------------------------------------------------------------- (a) harmful edits are rejected

def test_a_flipped_coefficient_is_rejected_kept_for_inspection_and_never_served(tmp_path):
    svc = trained_service(tmp_path)
    base = svc.registry.latest_id()
    res = PT.edit_head(svc, head="clean", delta_path=flip_delta(svc, tmp_path / "flip.safetensors"),
                       author="LLM: test-model", tuned_on="none")
    vid, j = res["version"], res["heads"]["clean"]
    assert res["passed"] is False and res["promoted"] is False and res["servable"] is False
    assert svc.registry.latest_id() == base != vid                        # rlcd-latest did not move
    t = j["paired_test"]
    assert t["brier"]["mean_gain"] < 0 and t["log_loss"]["mean_gain"] < 0 and t["passed"] is False
    assert j["weaker_verdicts"] and "not reliably positive" in res["reason"] and "weaker" in res["reason"]
    assert res["parameters_touched"] == len(members(svc, "clean"))       # one coefficient per member
    # kept as a version, for inspection only
    assert vid in svc.registry.registry()["versions"] and not svc.registry.servable(vid)
    with pytest.raises(ContractError) as e:
        svc.systemone(SystemOneRequest(model=vid, state={"f": [0.1] * 6}, questions={"clean": {"type": "noul"}}))
    assert e.value.kind == "unknown_model" and "never serves" in e.value.message
    assert vid not in e.value.details.get("available", []) and LATEST in e.value.details.get("available", [])
    assert svc.calibration_view("clean", vid)["model"] == vid             # inspection still works
    listed = {m["id"]: m for m in svc.models_view()["data"]}
    assert listed[vid]["edit"]["servable"] is False and "edit" not in listed[base]
    assert svc.systemone(SystemOneRequest(state={"f": [0.1] * 6},          # latest still answers
                                          questions={"clean": {"type": "noul"}}))["model"] == base
    # applying the delta changed copies: the base head, in memory and on disk, is untouched
    on_disk = CalibratedEnsemble.load(svc.bundle(base).head("clean").dir)
    assert np.array_equal(on_disk.members[0][0].coef, members(svc, "clean", base)[0][0].coef)
    assert not np.array_equal(on_disk.members[0][0].coef, members(svc, "clean", vid)[0][0].coef)
    # it stays unservable across export / import, and never becomes latest there
    other = Registry(tmp_path / "other")
    svc.registry.export(vid, tmp_path / "exported")
    imported = other.import_(tmp_path / "exported")
    assert not other.servable(imported) and other.registry()["latest"] is None


# ----------------------------------------------------------------------------- (b) a planted leak

def test_b_removing_a_planted_leak_is_promoted_when_the_holdout_supports_it(tmp_path):
    svc = trained_service(tmp_path)
    base = svc.registry.latest_id()
    base_good = svc.bundle(base).head("good")
    assert base_good.summary["metrics"]["has_skill"] is False   # the shortcut broke in the holdout: no skill
    res = PT.edit_head(svc, head="good", delta_path=no_leak_delta(svc, tmp_path / "noleak.safetensors"), **manual())
    vid, j = res["version"], res["heads"]["good"]
    assert res["passed"] and res["promoted"] and res["servable"] and svc.registry.latest_id() == vid
    t = j["paired_test"]
    assert t["brier"]["lower"] > 0 and t["log_loss"]["lower"] > 0 and t["alpha"] == 0.05 and j["attempt"] == 1
    assert not j["weaker_verdicts"]
    new = svc.bundle(vid).head("good")
    assert new.summary["metrics"]["has_skill"] is True
    # the leak's weights are gone from every member, and every member was recalibrated
    li = new.ensemble.feature_names.index("leak")
    for (b_new, c_new), (_, c_old) in zip(new.ensemble.members, base_good.ensemble.members, strict=True):
        assert np.all(b_new.coef[:, li] == 0)
        assert c_new.to_dict() != c_old.to_dict()
    # the other head is carried over unchanged; the Bernoulli ledger restarts from the edited head's holdout
    assert svc.bundle(vid).meta["heads"]["clean"].get("carried_over") is True
    assert svc.ledger.table("good")["model_version"] == vid
    assert vid in svc.ledger.table("clean")["valid_for"] and base in svc.ledger.table("clean")["valid_for"]


def test_b_removing_a_feature_the_holdout_still_uses_is_not_promoted(tmp_path):
    svc = trained_service(tmp_path, HoldingLeakPack())
    base = svc.registry.latest_id()
    res = PT.edit_head(svc, head="good", delta_path=no_leak_delta(svc, tmp_path / "noleak.safetensors"), **manual())
    assert res["promoted"] is False and res["servable"] is False and svc.registry.latest_id() == base
    assert res["heads"]["good"]["paired_test"]["brier"]["mean_gain"] < 0


# ----------------------------------------------------------------------------- (c) provenance

def test_c_provenance_in_model_json_head_json_and_model_card(tmp_path):
    svc = trained_service(tmp_path)
    base = svc.registry.latest_id()
    path = no_leak_delta(svc, tmp_path / "noleak.safetensors", author="LLM: weight-programmer-x",
                         tuned_on="development", description="drop the leak feature")
    res = PT.edit_head(svc, head="good", delta_path=path)
    b = svc.bundle(res["version"])
    e = b.meta["edit"]
    assert e["sha256"] == BD.sha256(path) and e["kind"] == "head" and e["base_version"] == base
    assert e["provenance"]["author"] == "LLM: weight-programmer-x" and e["provenance"]["tuned_on"] == "development"
    assert e["provenance"]["description"] == "drop the leak feature"
    assert e["parameters_touched"] == len(b.head("good").ensemble.members) and e["parameters_total"] > 0
    assert e["heads"]["good"]["paired_test"]["passed"] is True and e["promoted"] is True
    he = b.head("good").summary["edit"]                     # head.json: covered by the sha256 in model.json
    assert he["sha256"] == e["sha256"] and he["base_version"] == base and he["provenance"]["author"] == e["author"]
    assert he["parameters_touched"] == e["parameters_touched"] and he["paired_test"] == e["heads"]["good"]["paired_test"]
    assert he["holdout_ids_sha256"] == e["holdouts"]["good"]
    assert "edit" not in b.head("clean").summary            # the carried head is not an edit
    card = model_card(b)
    assert "## Weight edit (experimental)" in card and e["sha256"] in card and "LLM: weight-programmer-x" in card
    assert f"edit of `{base}`" in card and "promoted to rlcd-latest" in card and "paired Brier gain" in card
    reg = svc.registry.entry(res["version"])["edit"]
    assert reg["sha256"] == e["sha256"] and reg["promoted"] is True and reg["author"] == "LLM: weight-programmer-x"


# ----------------------------------------------------------------------------- (d) bundles stay plain and checked

def test_d_edited_bundles_validate_tampered_ones_fail_and_nothing_is_a_pickle(tmp_path):
    svc = trained_service(tmp_path)
    res = PT.edit_head(svc, head="good", delta_path=no_leak_delta(svc, tmp_path / "noleak.safetensors"), **manual())
    path = svc.bundle(res["version"]).path
    BD.validate(path, check_models=True)
    for p in path.rglob("*"):
        if p.is_file():
            assert p.suffix in BD.ALLOWED_SUFFIXES and not p.read_bytes().startswith(b"\x80")   # no pickle header
    # a tampered member and a tampered verdict both fail the hash check
    for victim, change in (("heads/good/member_0.safetensors", lambda b: b[:-8] + bytes(8)),
                           ("heads/good/head.json", lambda b: b.replace(b'"passed": true', b'"passed": false', 1))):
        bad = tmp_path / f"tampered-{Path(victim).name}"
        shutil.copytree(path, bad)
        target = bad / victim
        before = target.read_bytes()
        target.write_bytes(change(before))
        assert target.read_bytes() != before
        with pytest.raises(BD.BundleError, match="sha256"):
            BD.validate(bad)
    # an edit is read as safetensors only: a pickle under that name is refused, and the module never unpickles
    fake = tmp_path / "evil.safetensors"
    fake.write_bytes(b"\x80\x04\x95" + bytes(64))
    with pytest.raises(PT.EditRefused, match="not a readable safetensors"):
        PT.read_delta(fake)
    src = Path(PT.__file__).read_text()
    assert "import pickle" not in src and "joblib" not in src and "torch.load" not in src


# ----------------------------------------------------------------------------- (e) the holdout only judges

def test_e_edits_tuned_on_the_holdout_are_refused_before_scoring(tmp_path):
    svc = trained_service(tmp_path)
    versions = set(svc.registry.registry()["versions"])
    path = no_leak_delta(svc, tmp_path / "noleak.safetensors")
    with pytest.raises(PT.EditRefused, match="tuned on the holdout"):
        PT.edit_head(svc, head="good", delta_path=path, author="manual", tuned_on="holdout")
    # rows listed as "looked at" that are holdout rows (the latest 20 % in time) are refused too
    ids = tmp_path / "ids.txt"
    ids.write_text("10\n2999\n")
    from open_rlcd.cli import _read_ids

    with pytest.raises(PT.EditRefused, match="holdout rows"):
        PT.edit_head(svc, head="good", delta_path=path, author="manual", tuned_on="development",
                     tuned_on_ids=_read_ids(str(ids)))
    tagged = no_leak_delta(svc, tmp_path / "tagged.safetensors", author="LLM: x", tuned_on="holdout")
    with pytest.raises(PT.EditRefused, match="tuned on the holdout"):
        PT.edit_head(svc, head="good", delta_path=tagged)
    with pytest.raises(PT.EditRefused, match="says tuned_on"):         # the file and the command disagree
        PT.edit_head(svc, head="good", delta_path=tagged, tuned_on="development")
    with pytest.raises(PT.EditRefused, match="author"):
        PT.edit_head(svc, head="good", delta_path=path, tuned_on="development")
    with pytest.raises(PT.EditRefused, match="tuned_on must be one of"):
        PT.edit_head(svc, head="good", delta_path=path, author="manual")
    with pytest.raises(PT.EditRefused, match="lists 1 row ids"):
        PT.edit_head(svc, head="good", delta_path=path, author="manual", tuned_on="none", tuned_on_ids=["5"])
    assert set(svc.registry.registry()["versions"]) == versions          # nothing was scored or saved
    # development rows are fine
    res = PT.edit_head(svc, head="good", delta_path=path, author="manual", tuned_on="development",
                       tuned_on_ids=["10", "11"])
    assert res["promoted"] and res["provenance"]["tuned_on_ids"]["n"] == 2


def test_e_every_further_attempt_on_a_holdout_faces_a_tighter_bound_and_base_must_be_latest(tmp_path):
    svc = trained_service(tmp_path)
    base = svc.registry.latest_id()
    path = no_leak_delta(svc, tmp_path / "noleak.safetensors")
    first = PT.edit_head(svc, head="good", delta_path=path, **manual())
    assert first["promoted"] and first["heads"]["good"]["attempt"] == 1
    again = PT.edit_head(svc, head="good", delta_path=path, base=base, **manual())
    j = again["heads"]["good"]
    assert j["attempt"] == 2 and j["alpha"] == 0.025 and j["paired_test"]["confidence"] == 0.975
    # it passes its gate, but the base is no longer rlcd-latest: kept, servable by name, not promoted
    assert again["passed"] and again["servable"] and not again["promoted"]
    assert "no longer rlcd-latest" in again["reason"] and svc.registry.latest_id() == first["version"]
    assert svc.resolve(again["version"]) == again["version"]
    # the other head's holdout has not judged any edit yet
    assert svc.registry.edit_attempts("clean", first["holdouts"]["good"]) == 0


# ----------------------------------------------------------------------------- (f) judged against its base

def test_f_malformed_deltas_are_refused(tmp_path):
    svc = trained_service(tmp_path)
    ms = members(svc, "good")
    shape = ms[0][0].coef.shape

    def refused(arrays, match):
        p = tmp_path / "bad.safetensors"
        save_file(arrays, str(p))
        with pytest.raises(PT.EditRefused, match=match):
            PT.edit_head(svc, head="good", delta_path=p, **manual())

    refused({"member_0.weights": np.ones(shape)}, "unknown array")
    refused({"good.coef": np.ones(shape)}, "unknown array")
    refused({f"member_{len(ms)}.coef": np.ones(shape)}, "members")
    refused({"member_0.coef": np.ones((shape[0], shape[1] + 1))}, "shape")
    refused({"member_0.coef": np.full(shape, np.nan)}, "NaN")
    refused({"member_0.scale": -2 * ms[0][0].scale}, "positive")
    refused({"member_0.coef": np.zeros(shape)}, "changes no parameter")
    with pytest.raises(PT.EditRefused, match="no head"):
        PT.edit_head(svc, head="nope", delta_path=tmp_path / "bad.safetensors", **manual())
    (tmp_path / "edit.npz").write_bytes(b"PK\x03\x04")
    with pytest.raises(PT.EditRefused, match="safetensors"):
        PT.read_delta(tmp_path / "edit.npz")


def test_f_rows_that_changed_since_training_are_refused(tmp_path, monkeypatch):
    svc = trained_service(tmp_path)
    path = no_leak_delta(svc, tmp_path / "noleak.safetensors")
    import tests.leakpack as LP

    monkeypatch.setattr(LP, "leak_rows", lambda holds=False: leak_rows(n=3300, holds=holds))   # more labels now
    with pytest.raises(PT.EditRefused, match="rows changed"):
        PT.edit_head(svc, head="good", delta_path=path, **manual())


@pytest.mark.skipif(not lightgbm_available(), reason="LightGBM not importable (macOS: brew install libomp)")
def test_f_lightgbm_members_are_not_editable():
    th = train_head(HeadSpec("trees", "noul", model="lightgbm"), clean_rows(1500), TrainConfig(n_estimators=20))
    with pytest.raises(PT.EditRefused, match="LightGBM"):
        PT.apply_delta(th.ensemble, {"member_0.coef": np.zeros((1, 6))})


def test_paired_gain_test_needs_a_reliable_gain_on_both_scores():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 400)
    good = np.where(y[:, None] == np.arange(2), 0.8, 0.2)
    flat = np.full((400, 2), 0.5)
    assert M.paired_gain_test(flat, good, y)["passed"] is True
    assert M.paired_gain_test(good, flat, y)["passed"] is False
    same = M.paired_gain_test(good, good, y)
    assert same["passed"] is False and same["brier"]["mean_gain"] == 0
    assert M.paired_gain_test(flat[:8], good[:8], y[:8])["passed"] is False      # too few rows to tell
    tight = M.paired_gain_test(flat, good, y, alpha=0.005)
    assert tight["confidence"] == 0.995 and tight["n_boot"] >= 40000 and tight["brier"]["lower"] <= \
        M.paired_gain_test(flat, good, y)["brier"]["lower"]


# ----------------------------------------------------------------------------- CLI

def test_cli_patch_exit_status_says_promoted_kept_or_refused(tmp_path, capsys):
    root = tmp_path / "proj"
    root.mkdir()
    write_config(root / "rlcd.yaml", {"name": "leak", "pack": "tests.leakpack:LeakPack", "home": "./var"})
    rl = ["--project", str(root), "--home", str(root / "var")]
    assert main([*rl, "train"]) == 0
    svc = Service(Project(root=root, data={"pack": "tests.leakpack:LeakPack"}), home=root / "var")
    good = no_leak_delta(svc, tmp_path / "noleak.safetensors")
    flip = flip_delta(svc, tmp_path / "flip.safetensors")
    capsys.readouterr()
    assert main([*rl, "patch", "--delta", str(flip), "--head", "clean", "--author", "manual", "--tuned-on", "none"]) == 3
    out = capsys.readouterr().out
    assert "NOT promoted" in out and "never served" in out and "FAIL" in out
    assert main([*rl, "patch", "--delta", str(good), "--head", "good", "--author", "LLM: x", "--tuned-on",
                 "holdout"]) == 1
    assert "edit refused" in capsys.readouterr().err
    assert main([*rl, "patch", "--delta", str(good), "--head", "good", "--author", "LLM: x", "--tuned-on",
                 "development", "--json"]) == 0
    res = json.loads(capsys.readouterr().out)
    assert res["promoted"] and res["version"] == "rlcd-0.1.3"
    assert main([*rl, "report", "--no-open", "--model", "rlcd-0.1.2"]) == 0     # a rejected candidate can be inspected
    assert main([*rl, "patch", "--delta", str(good), "--author", "manual", "--tuned-on", "none"]) == 2
