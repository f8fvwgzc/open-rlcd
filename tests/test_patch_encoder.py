"""Encoder edits, offline: a replacement ONNX text encoder is a new feature version, so every head of the base is
retrained on it (never reused), each is judged against its base head on the same holdout rows, and the candidate
serves only if every head passes. The encoder is a tiny Gather-over-embeddings ONNX graph (tests/tinyonnx.py)
run by the real ONNX Runtime through the documents pack."""
from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("onnxruntime")
pytest.importorskip("tokenizers")

from open_rlcd.core import bundle as BD  # noqa: E402
from open_rlcd.core import patch as PT  # noqa: E402
from open_rlcd.core.config import Project  # noqa: E402
from open_rlcd.core.contract import TrainRequest  # noqa: E402
from open_rlcd.core.service import Service  # noqa: E402
from open_rlcd.features.text import EncoderError, make_encoder  # noqa: E402
from open_rlcd.packs.documents import write_synth  # noqa: E402
from open_rlcd.packs.documents.world import build_world  # noqa: E402
from open_rlcd.report.model_card import model_card  # noqa: E402
from tests.leakpack import LeakPack  # noqa: E402
from tests.tinyonnx import tiny_onnx, tiny_tokenizer  # noqa: E402

HEADS = ["answerable", "grounded"]
D = 32


@pytest.fixture(scope="module")
def base(tmp_path_factory):
    root = tmp_path_factory.mktemp("encoder-edit")
    home = root / "var"
    mp = pytest.MonkeyPatch()
    mp.setenv("RLCD_HOME", str(home))        # the documents pack reads its label store from project.home
    V = tiny_tokenizer([d.record()["text"] for d in build_world(7).docs.values()],
                       home / "encoders" / "base" / "tokenizer.json")
    emb = np.random.default_rng(0).normal(size=(V, D)).astype(np.float32)
    (home / "encoders" / "base" / "tiny.onnx").write_bytes(tiny_onnx(emb))
    enc = {"kind": "onnx", "model_id": "test/tiny", "path": "encoders/base/tiny.onnx",
           "tokenizer": "encoders/base/tokenizer.json"}
    write_synth(home, enc, seed=7, n_queries=400)
    svc = Service(Project(root=root, data={"pack": "documents", "pack_config": {"encoder": enc}}), home=home)
    job = svc.trainer.start(TrainRequest(heads=HEADS), wait=True)
    assert job["status"] == "succeeded", job.get("error")
    yield SimpleNamespace(root=root, home=home, svc=svc, emb=emb, V=V, version=job["model"])
    mp.undo()


def edited_file(t, path: Path, rows: int = 10) -> Path:
    emb = t.emb.copy()
    emb[:rows] += 1.0
    path.write_bytes(tiny_onnx(emb))
    return path


def test_an_encoder_edit_retrains_every_head_and_is_judged_on_the_same_holdout(base, tmp_path):
    svc, b = base.svc, base.svc.bundle(base.version)
    src = edited_file(base, tmp_path / "edited.onnx")
    res = PT.edit_encoder(svc, onnx_path=src, author="script:perturb", tuned_on="none",
                          description="nudge ten token embeddings")
    vid = res["version"]
    assert res["kind"] == "encoder" and res["base_version"] == base.version and res["sha256"] == BD.sha256(src)
    assert res["retrained"] == sorted(HEADS) and res["feature_version"] != b.meta["feature_version"]
    assert any("none was reused" in n for n in res["notes"]) and not any("int8" in n for n in res["notes"])
    assert res["parameters_touched"] == 10 * D and res["parameters_total"] == base.V * D
    # each head: retrained (not carried), scored on the base's holdout rows, compared row by row
    cand = svc.bundle(vid)
    for name in HEADS:
        j = res["heads"][name]
        assert j["paired_test"]["n"] == b.head(name).report["n_holdout"] == cand.head(name).report["n_holdout"]
        assert res["holdouts"][name] == b.head(name).report["split"]["holdout_ids_sha256"]
        assert cand.meta["heads"][name].get("carried_over") is None and cand.meta["heads"][name]["trained_in"] == vid
        assert cand.head(name).ensemble.feature_version == res["feature_version"]
        he = cand.head(name).summary["edit"]
        assert he["kind"] == "encoder" and he["provenance"]["author"] == "script:perturb"
    # the decision follows the rule: every head must pass, and the base must still be rlcd-latest
    passed = all(j["passed"] for j in res["heads"].values())
    assert res["passed"] is passed and res["promoted"] is passed and res["servable"] is passed
    assert svc.registry.latest_id() == (vid if passed else base.version)
    # the edited file lives in the home; the bundle's encoder config points at it by path and sha256
    assert (base.home / res["encoder_path"]).is_file() and BD.sha256(base.home / res["encoder_path"]) == res["sha256"]
    enc = cand.meta["encoder"]
    assert enc["path"] == res["encoder_path"] == f"encoders/edited/{res['sha256']}.onnx" and enc["sha256"] == res["sha256"]
    assert svc.provider(vid).feature_version == res["feature_version"]       # it can be rebuilt for serving
    BD.validate(cand.path, check_models=True)
    assert not any(p.suffix == ".onnx" for p in cand.path.rglob("*"))         # bundles never carry the encoder
    card = model_card(cand)
    assert "**encoder** edit" in card and "parameters touched: 320" in card


def test_encoder_edits_are_refused_when_they_cannot_be_judged(base, tmp_path):
    svc = base.svc
    versions = set(svc.registry.registry()["versions"])
    same = tmp_path / "same.onnx"
    same.write_bytes((base.home / "encoders" / "base" / "tiny.onnx").read_bytes())
    with pytest.raises(PT.EditRefused, match="byte-identical"):
        PT.edit_encoder(svc, onnx_path=same, base=base.version, author="manual", tuned_on="none")
    src = edited_file(base, tmp_path / "other.onnx", rows=3)
    with pytest.raises(PT.EditRefused, match="tuned on the holdout"):
        PT.edit_encoder(svc, onnx_path=src, base=base.version, author="LLM: x", tuned_on="holdout")
    assert not (base.home / "encoders" / "edited" / f"{BD.sha256(src)}.onnx").exists()   # refused before copying
    with pytest.raises(PT.EditRefused, match="regular .onnx"):
        PT.edit_encoder(svc, onnx_path=tmp_path / "missing.onnx", base=base.version, author="manual", tuned_on="none")
    link = tmp_path / "link.onnx"
    os.symlink(src, link)
    with pytest.raises(PT.EditRefused, match="regular .onnx"):
        PT.edit_encoder(svc, onnx_path=link, base=base.version, author="manual", tuned_on="none")
    assert set(svc.registry.registry()["versions"]) == versions
    # a version without an ONNX encoder has nothing to replace
    other = Service(Project(root=tmp_path), home=tmp_path / "leak", pack=LeakPack())
    other.trainer.start(TrainRequest(), wait=True)
    with pytest.raises(PT.EditRefused, match="no encoder"):
        PT.edit_encoder(other, onnx_path=src, author="manual", tuned_on="none")


def test_parameters_touched_are_counted_from_the_onnx_initializers(base, tmp_path):
    a = base.home / "encoders" / "base" / "tiny.onnx"
    assert PT.count_touched(a, a)[:2] == (0, base.V * D)
    assert PT.count_touched(a, edited_file(base, tmp_path / "e.onnx", rows=4))[:2] == (4 * D, base.V * D)
    wider = tmp_path / "wider.onnx"
    wider.write_bytes(tiny_onnx(np.zeros((base.V, D + 1), np.float32)))
    assert PT.count_touched(a, wider)[:2] == (base.V * (D + 1), base.V * (D + 1))   # another shape: all touched
    junk = tmp_path / "junk.onnx"
    junk.write_bytes(b"\xff" * 64)
    touched, total, why = PT.count_touched(a, junk)
    assert touched is None and total is None and why


def test_local_encoder_files_must_stay_inside_the_home(base):
    for bad in ("../tiny.onnx", "/etc/passwd.onnx", "encoders/base/../../../x.onnx"):
        with pytest.raises(EncoderError, match="inside"):
            make_encoder({"kind": "onnx", "model_id": "test/tiny", "path": bad,
                          "tokenizer": "encoders/base/tokenizer.json", "cache": False}, base.home)
    with pytest.raises(EncoderError, match="not a regular file"):
        make_encoder({"kind": "onnx", "model_id": "test/tiny", "path": "encoders/none.onnx",
                      "tokenizer": "encoders/base/tokenizer.json", "cache": False}, base.home)
