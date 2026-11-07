"""The open-weight bundle and the registry: plain files only (JSON, safetensors, LightGBM text), validated before
they are trusted, identical predictions after a round trip, versions under RLCD_HOME."""
import json
import os
import shutil

import numpy as np
import pytest

from open_rlcd import LATEST
from open_rlcd.core import bundle as BD
from open_rlcd.core.dataset import Dataset
from open_rlcd.core.head import HeadSpec, lightgbm_available
from open_rlcd.core.registry import Registry, UnknownModel
from open_rlcd.core.training import TrainConfig, train_head
from tests.toypack import NAMES, generate

PACK = {"name": "toy", "version": "0.1.0", "config": {"kind": "toy"}}
POLICY = {"abstain": "escalate"}
Xq = np.random.default_rng(42).normal(size=(64, 6)).astype(np.float32)


def _ds(k, seed, n=1500):
    X, y, _ = generate(n, k, 1.5, seed=seed)
    return Dataset(X=X, y=y, feature_names=NAMES, feature_version="toy-v1", source=["synthetic"] * n)


@pytest.fixture(scope="module")
def heads():
    color_model = "lightgbm" if lightgbm_available() else "logreg"
    good = HeadSpec("good", "noul", view="v", decision="gate", conformal=0.1)
    color = HeadSpec("color", "choice", ("red", "green", "blue"), view="v", model=color_model, conformal=0.2)
    cfg = TrainConfig(n_estimators=40)
    return {"good": train_head(good, _ds(2, 0, 3000), cfg), "color": train_head(color, _ds(3, 1), cfg)}


@pytest.fixture(scope="module")
def bundle_dir(heads, tmp_path_factory):
    dest = tmp_path_factory.mktemp("bundles") / "b1"
    BD.write_bundle(dest, version="rlcd-0.1.1", created_at="2026-11-07T00:00:00Z", pack=PACK,
                    feature_version="toy-v1", encoder=None, policy=POLICY, trained=heads,
                    data_card={"pack": "toy"}, notes=["test"])
    return dest


def corrupt_copy(src, tmp_path, name="bad"):
    dest = tmp_path / name
    shutil.copytree(src, dest)
    return dest


# ----------------------------------------------------------------------------- bundle round trip

def test_save_load_gives_identical_predictions(heads, bundle_dir):
    b = BD.Bundle(bundle_dir)
    assert b.version == "rlcd-0.1.1" and sorted(b.names()) == ["color", "good"] and b.head("nope") is None
    for name, th in heads.items():
        h = b.head(name)
        assert h.spec == th.spec
        assert np.array_equal(h.ensemble.predict_proba(Xq), th.ensemble.predict_proba(Xq))
        assert h.verdict == th.verdict and h.report["n_holdout"] == th.report["n_holdout"]
        assert h.holdout_ledger()["holdout_trials"] == th.ledger["holdout_trials"]
    assert b.head("good").conformal_threshold == heads["good"].conformal["selective"]["threshold"]
    assert b.head("color").set_quantile == heads["color"].conformal["sets"]["q"]
    assert b.head("good").set_quantile is None


def test_model_json_contents(heads, bundle_dir):
    meta = json.loads((bundle_dir / "model.json").read_text())
    assert meta["format"] == BD.FORMAT == "open-rlcd-bundle/1"
    assert meta["version"] == "rlcd-0.1.1" and meta["created_at"] == "2026-11-07T00:00:00Z"
    assert meta["pack"] == PACK and meta["feature_version"] == "toy-v1" and meta["policy"] == POLICY
    assert meta["retrained"] == ["color", "good"] and meta["data_card"] == {"pack": "toy"} and meta["notes"] == ["test"]
    for name, th in heads.items():
        e = meta["heads"][name]
        assert e["dir"] == f"heads/{name}" and HeadSpec.from_dict(e["spec"]) == th.spec
        assert e["metrics"]["n_holdout"] == th.metrics["n_holdout"] and e["model"] == th.model
        assert {"ensemble.json", "head.json", "report.json", "bernoulli.json"} <= set(e["files"])
        for f, digest in e["files"].items():
            assert BD.sha256(bundle_dir / e["dir"] / f) == digest and len(digest) == 64
    assert BD.validate(bundle_dir)["version"] == "rlcd-0.1.1"


def _is_valid_safetensors(data: bytes) -> bool:
    n = int.from_bytes(data[:8], "little")
    header = json.loads(data[8:8 + n])
    end = max(v["data_offsets"][1] for k, v in header.items() if k != "__metadata__")
    return len(data) == 8 + n + end


def test_no_file_in_a_trained_bundle_is_a_pickle(bundle_dir):
    files = [p for p in bundle_dir.rglob("*") if p.is_file()]
    assert files and {p.suffix for p in files} <= BD.ALLOWED_SUFFIXES
    for p in files:
        head = p.read_bytes()[:2]
        if p.suffix == ".safetensors":                            # a length-prefixed JSON header, then raw tensors
            assert _is_valid_safetensors(p.read_bytes()), p
        else:
            assert not (head[:1] == b"\x80" and head[1:2] in (b"\x02", b"\x03", b"\x04", b"\x05")), p
            if p.suffix == ".json":
                json.loads(p.read_text(encoding="utf-8"))
            else:
                p.read_text(encoding="utf-8")                     # LightGBM models are text


# ----------------------------------------------------------------------------- validate

def test_validate_refuses_a_symlink(bundle_dir, tmp_path):
    bad = corrupt_copy(bundle_dir, tmp_path)
    os.symlink("/etc/hosts", bad / "heads" / "good" / "notes.txt")
    with pytest.raises(BD.BundleError, match="symlink"):
        BD.validate(bad)
    with pytest.raises(BD.BundleError, match="not a directory"):
        link = tmp_path / "link"
        os.symlink(bundle_dir, link)
        BD.validate(link)


@pytest.mark.parametrize("name", ["model.pkl", "heads/good/member_0.joblib", "heads/color/x.npy", "run.py"])
def test_validate_refuses_pickles_and_code(bundle_dir, tmp_path, name):
    bad = corrupt_copy(bundle_dir, tmp_path)
    (bad / name).write_bytes(b"\x80\x04\x95 not really a pickle")
    with pytest.raises(BD.BundleError, match="unexpected file type"):
        BD.validate(bad)


def test_validate_refuses_a_sha256_mismatch(bundle_dir, tmp_path):
    bad = corrupt_copy(bundle_dir, tmp_path)
    ens = bad / "heads" / "good" / "ensemble.json"
    meta = json.loads(ens.read_text())
    meta["members"][0]["calibrator"]["params"]["a"] = 99.0         # a tampered calibrator
    ens.write_text(json.dumps(meta))
    with pytest.raises(BD.BundleError, match="does not match its sha256"):
        BD.validate(bad)
    assert BD.validate(bad, check_hashes=False)


def test_validate_refuses_a_head_dir_escaping_the_bundle(bundle_dir, tmp_path):
    bad = corrupt_copy(bundle_dir, tmp_path)
    meta = json.loads((bad / "model.json").read_text())
    meta["heads"]["good"]["dir"] = "../b1/heads/good"
    (bad / "model.json").write_text(json.dumps(meta))
    with pytest.raises(BD.BundleError, match="escapes the bundle"):
        BD.validate(bad)
    meta["heads"]["good"]["dir"] = "heads/good"
    meta["heads"]["good"]["files"] = {"../color/ensemble.json": "0" * 64}
    (bad / "model.json").write_text(json.dumps(meta))
    with pytest.raises(BD.BundleError, match="missing or invalid file"):
        BD.validate(bad)


def test_validate_refuses_bad_names_formats_and_missing_files(bundle_dir, tmp_path):
    bad = corrupt_copy(bundle_dir, tmp_path)
    meta = json.loads((bad / "model.json").read_text())
    meta["heads"]["../evil"] = meta["heads"].pop("good")
    (bad / "model.json").write_text(json.dumps(meta))
    with pytest.raises(BD.BundleError, match="must match"):
        BD.validate(bad)
    meta["format"] = "pickle-bundle/9"
    (bad / "model.json").write_text(json.dumps(meta))
    with pytest.raises(BD.BundleError, match="unknown bundle format"):
        BD.validate(bad)
    (bad / "model.json").unlink()
    with pytest.raises(BD.BundleError, match="no model.json"):
        BD.validate(bad)
    gone = corrupt_copy(bundle_dir, tmp_path, "gone")
    (gone / "heads" / "color" / "report.json").unlink()
    with pytest.raises(BD.BundleError, match="missing or invalid file"):
        BD.validate(gone)


def test_member_files_are_checked_at_load_time(bundle_dir, tmp_path):
    bad = corrupt_copy(bundle_dir, tmp_path)
    ens = bad / "heads" / "good" / "ensemble.json"
    meta = json.loads(ens.read_text())
    meta["members"][0]["file"] = "../../model.json"
    ens.write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="bad member file name"):
        _ = BD.Bundle(bad).head("good").ensemble


def test_write_bundle_never_overwrites(bundle_dir, heads):
    with pytest.raises(BD.BundleError, match="already exists"):
        BD.write_bundle(bundle_dir, version="x", created_at="", pack=PACK, feature_version="toy-v1", encoder=None,
                        policy=POLICY, trained=heads)


# ----------------------------------------------------------------------------- registry

def commit(reg, trained, feature_version="toy-v1"):
    return reg.commit(trained, pack=PACK, feature_version=feature_version, encoder=None, policy=POLICY)


def test_registry_versions_and_alias(heads, tmp_path):
    reg = Registry(tmp_path / "home")
    assert reg.latest_id() == "rlcd-0.1.0" and reg.bundle("rlcd-0.1.0") is None and reg.available() == [LATEST]
    assert reg.resolve(None) == reg.resolve(LATEST) == "rlcd-0.1.0"
    with pytest.raises(UnknownModel):
        reg.resolve("rlcd-0.1.7")
    v1 = commit(reg, heads)
    assert v1 == "rlcd-0.1.1" and reg.latest_id() == v1 and reg.resolve(LATEST) == v1
    assert reg.available() == [LATEST, v1]
    with pytest.raises(UnknownModel):
        reg.resolve("rlcd-0.1.0")                                  # the untrained id is gone once trained
    data = json.loads((tmp_path / "home" / "registry.json").read_text())
    assert data["latest"] == v1 and data["next"] == 2 and data["versions"][v1]["path"] == "models/rlcd-0.1.1"
    for name in heads:                                            # the ledger starts from the new holdout
        t = reg.ledger.table(name)
        assert t["model_version"] == v1 and t["holdout_trials"] == heads[name].ledger["holdout_trials"]


def test_partial_retrain_carries_the_other_heads_over(heads, tmp_path):
    reg = Registry(tmp_path / "home")
    v1 = commit(reg, heads)
    before = reg.bundle(v1).head("color").ensemble.predict_proba(Xq)
    led = reg.ledger
    led.update("color", "d1", "red", 0.97, "red")                 # an outcome scored against v1
    v2 = commit(reg, {"good": heads["good"]})
    b2 = reg.bundle(v2)
    meta = b2.meta
    assert meta["retrained"] == ["good"] and meta["heads"]["color"]["carried_over"] is True
    assert "carried_over" not in meta["heads"]["good"]
    assert np.array_equal(b2.head("color").ensemble.predict_proba(Xq), before)
    assert BD.validate(b2.path)
    # the carried head is the same model: its evidence carries over and stays live for the new version
    t = led.table("color")
    assert t["model_version"] == v2 and t["online_trials"] == 1 and t["applied"] == ["d1"]
    assert led.table("good")["model_version"] == v2 and led.table("good")["online_trials"] == 0
    assert reg.registry()["versions"][v2]["retrained"] == ["good"]


def test_a_new_feature_version_does_not_carry_heads_over(heads, tmp_path):
    reg = Registry(tmp_path / "home")
    commit(reg, heads)
    v2 = commit(reg, {"good": heads["good"]}, feature_version="toy-v2")
    meta = reg.bundle(v2).meta
    assert list(meta["heads"]) == ["good"] and any("not carried over" in n and "color" in n for n in meta["notes"])


def test_export_and_import_register_a_new_version(heads, tmp_path):
    src = Registry(tmp_path / "a")
    v1 = commit(src, heads)
    out = src.export(v1, tmp_path / "exported")
    assert BD.validate(out)["version"] == v1
    with pytest.raises(BD.BundleError, match="not empty"):
        src.export(v1, tmp_path / "exported")
    dst = Registry(tmp_path / "b")
    commit(dst, {"good": heads["good"]})                         # dst already has rlcd-0.1.1
    vid = dst.import_(out)
    assert vid == "rlcd-0.1.2" and dst.latest_id() == vid
    entry = dst.registry()["versions"][vid]
    assert entry["imported_from"] == v1 and entry["imported_at"]
    b = dst.bundle(vid)
    assert b.meta["version"] == vid and b.meta["imported_from"] == v1
    for name, th in heads.items():
        assert np.array_equal(b.head(name).ensemble.predict_proba(Xq), th.ensemble.predict_proba(Xq))
        assert dst.ledger.table(name)["model_version"] == vid
    quiet = dst.import_(out, make_latest=False)
    assert quiet == "rlcd-0.1.3" and dst.latest_id() == vid


def test_import_refuses_an_invalid_bundle(heads, bundle_dir, tmp_path):
    reg = Registry(tmp_path / "home")
    bad = corrupt_copy(bundle_dir, tmp_path)
    (bad / "weights.pkl").write_bytes(b"\x80\x05")
    with pytest.raises(BD.BundleError):
        reg.import_(bad)
    assert reg.registry()["versions"] == {} and not any((tmp_path / "home" / "models").iterdir())


def test_registry_refuses_a_path_escaping_rlcd_home(heads, tmp_path):
    reg = Registry(tmp_path / "home")
    v1 = commit(reg, heads)
    path = tmp_path / "home" / "registry.json"
    data = json.loads(path.read_text())
    data["versions"][v1]["path"] = "../elsewhere"
    path.write_text(json.dumps(data))
    fresh = Registry(tmp_path / "home")                            # no cached bundle
    with pytest.raises(RuntimeError, match="escapes RLCD_HOME"):
        fresh.bundle(v1)
