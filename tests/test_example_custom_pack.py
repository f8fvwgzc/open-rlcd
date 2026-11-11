"""examples/custom-pack: the ticket-triage pack trains, explains, takes feedback and round-trips as a bundle."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from importlib import import_module
from pathlib import Path

import numpy as np
import pytest

from open_rlcd.core.config import load_project
from open_rlcd.core.contract import DecideRequest, FeedbackRequest, TrainRequest
from open_rlcd.core.service import Service
from open_rlcd.features import HashedNgrams, TextEmbeddings, text_block, tokens

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "custom-pack"
TICKET = {"subject": "URGENT: charged twice",
          "body": "I was charged twice for my subscription. Please answer today, our customers are waiting."}
TICKETS = [TICKET, {"subject": "注文した商品が届きません", "body": "荷物が破損していました。至急対応をお願いします。"},
           {"subject": "Нууц үгээ мартсан", "body": "Бүртгэлдээ нэвтэрч чадахгүй байна. Яарах зүйлгүй."},
           {"subject": "Hello", "body": ""}]


@pytest.fixture(scope="module")
def triage(tmp_path_factory):
    """The example copied to a temp folder (as a user would have it: rlcd.yaml with ticketpack/ next to it),
    entered with `cd`, with 900 synthetic tickets, trained once."""
    root = tmp_path_factory.mktemp("triage") / "project"
    shutil.copytree(EXAMPLE, root, ignore=shutil.ignore_patterns("var", "data", "bundle", "__pycache__"))
    mp = pytest.MonkeyPatch()
    mp.chdir(root)
    mp.delenv("PYTHONPATH", raising=False)
    try:
        project = load_project(root, overrides={"pack_config": {"dim": 512}})
        svc = Service(project, home=root / "var")
        pack = svc.pack                                       # imports ticketpack from the project folder
        import_module("ticketpack.synth").write(root / "data" / "tickets.jsonl", n=900, seed=3)
        job = svc.trainer.start(TrainRequest(), wait=True)
        yield {"root": root, "project": project, "svc": svc, "pack": pack, "job": job}
    finally:
        mp.undo()
        for name in [m for m in sys.modules if m == "ticketpack" or m.startswith("ticketpack.")]:
            del sys.modules[name]


def test_the_pack_loads_from_the_project_folder_without_path_tricks(triage):
    assert triage["pack"].name == "tickets"
    assert str(triage["root"]) not in sys.path and not os.environ.get("PYTHONPATH")
    assert sys.modules["ticketpack.pack"].__file__.startswith(str(triage["root"]))
    assert triage["project"].check(triage["pack"]) == []      # config_keys / check_config: nothing to report


def test_the_cli_finds_the_pack_too(triage):
    # -P: the current folder is not put on sys.path, as for the installed `rlcd` command
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "RLCD_PROJECT")}
    r = subprocess.run([sys.executable, "-P", "-c", "import sys; from open_rlcd.cli import main; sys.exit(main())",
                        "config", "check"], cwd=triage["root"], env=env, capture_output=True, text=True, timeout=60)
    assert r.returncode == 0 and "no problems found" in r.stdout, r.stdout + r.stderr


def test_both_heads_find_skill(triage):
    job = triage["job"]
    assert job["status"] == "succeeded", job.get("error")
    for name in ("queue", "urgent"):
        metrics = job["heads"][name]["metrics"]
        assert metrics["has_skill"] is True, name
        assert job["heads"][name]["data"]["by_source"] == {"synthetic": 900}


def test_a_decision_carries_readable_reasons(triage):
    out = triage["svc"].decide(DecideRequest(state=TICKET))
    assert out["heads"]["queue"]["answer"]["choice"] == "billing"
    assert out["heads"]["urgent"]["answer"]["noul"] > 0.5
    words = set(tokens(f"{TICKET['subject']}\n{TICKET['body']}"))
    for head in ("queue", "urgent"):
        reasons = [r["feature"] for r in out["heads"][head]["reasons"]]
        assert reasons and all(r.startswith("text:") or r in ("length_chars", "exclamation_marks") for r in reasons)
        named = [r.removeprefix("text:").split(" (+")[0] for r in reasons if r.startswith("text:")]
        assert named and not any(n.startswith("#") for n in named)
        # most reasons name an n-gram of this very ticket (a shared column may be named after another one)
        assert sum(set(n.split()) <= words for n in named) >= len(named) / 2, named
        assert set(out["heads"][head]["reason_groups"]) <= {"wording", "shape"}


def test_feedback_reaches_the_next_training_set(triage):
    svc, pack, project = triage["svc"], triage["pack"], triage["project"]
    out = svc.decide(DecideRequest(state=TICKETS[1]))
    for head, label in (("queue", "shipping"), ("urgent", True)):
        saved = svc.feedback(FeedbackRequest(decision_id=out["decision_id"], head=head, label=label, source="test"))
        assert saved["stored"] is True
    datasets = pack.datasets(project, pack.provider(project), ["queue", "urgent"], svc.logs.feedback(), 5.0)
    for ds in datasets.values():
        assert ds.counts()["by_source"] == {"synthetic": 900, "feedback": 1}
        assert ds.weight[ds.source == "feedback"].tolist() == [5.0]
        assert "1 feedback rows added to training (weight x 5)" in ds.notes


def test_the_bundle_round_trips_with_identical_probabilities(triage, tmp_path):
    svc = triage["svc"]
    version = svc.resolve(None)
    dest = svc.registry.export(version, tmp_path / "bundle")
    assert not [p for p in dest.rglob("*") if p.suffix in (".pkl", ".pickle", ".joblib")]
    other = Service(load_project(triage["root"]), home=tmp_path / "other-home")      # same pack code, empty home
    imported = other.registry.import_(dest)
    assert other.pack_for(imported).name == "tickets"
    for state in TICKETS:
        a = svc.decide(DecideRequest(state=state, model=version))["heads"]
        b = other.decide(DecideRequest(state=state, model=imported))["heads"]
        assert a["queue"]["answer"]["probabilities"] == b["queue"]["answer"]["probabilities"]
        assert a["urgent"]["answer"]["noul"] == b["urgent"]["answer"]["noul"]
        assert [r["feature"] for r in a["queue"]["reasons"]] == [r["feature"] for r in b["queue"]["reasons"]]


# ----------------------------------------------------------------------------- the text blocks on their own

TEXTS = ["Refund the invoice please", "The invoice total is wrong", "請求書の金額が間違っています", "Нууц үгээ мартсан",
         "Refund please", ""]


def test_hashed_ngrams_have_readable_names_and_round_trip_as_json():
    block = HashedNgrams(dim=64, prefix="body").fit(TEXTS)
    names = block.names()
    assert len(names) == len(set(names)) == 64 and all(n.startswith("body:") for n in names)
    assert {"refund", "invoice", "please"} <= set(block.vocabulary)          # in two texts each; "wrong" in one
    assert "wrong" not in block.vocabulary
    assert {"請求", "нууц", "wrong"} <= set(HashedNgrams(dim=64).fit(TEXTS, min_df=1).vocabulary)   # ja, mn tokens
    X = block.transform(TEXTS)
    assert X.shape == (6, 64) and X.dtype == np.float32 and not X[5].any()
    assert np.allclose(np.linalg.norm(X[[0, 1, 4]], axis=1), 1.0, atol=1e-5)
    hit = {names[i].removeprefix("body:").split(" (+")[0] for i in np.flatnonzero(X[4])}
    assert {"refund", "please"} & hit
    assert not block.transform(["wholly unseen words"]).any()                # outside the vocabulary: no column
    again = text_block(json.loads(json.dumps(block.to_config())))
    assert again.names() == names and np.array_equal(again.transform(TEXTS), X)
    assert again.signature == block.signature != HashedNgrams(dim=64, prefix="body").fit(TEXTS[:2]).signature
    assert block.groups("wording") == dict.fromkeys(names, "wording")
    unfitted = HashedNgrams(dim=64)
    assert unfitted.names()[3] == "text:#3" and unfitted.transform(["wholly unseen words"]).any()


def test_text_embeddings_use_the_projects_encoder_cache_and_round_trip(tmp_path):
    project = load_project(overrides={"pack_config": {"encoder": {"kind": "hashing", "dim": 32}}})
    project.home_override = tmp_path / "home"
    block = TextEmbeddings.for_project(project, prefix="text")
    X = block.transform(TEXTS)
    assert X.shape == (6, 32) and block.names()[:2] == ["text_000", "text_001"]
    assert block.encoder.misses == 6 and list((tmp_path / "home" / "cache" / "embeddings").glob("*.sqlite"))
    block.transform(TEXTS)
    assert block.encoder.misses == 6 and block.encoder.hits == 6             # second pass: all from the cache
    config = json.loads(json.dumps(block.to_config()))
    again = text_block(config, home=tmp_path / "home")
    assert np.array_equal(again.transform(TEXTS), X) and again.signature == block.signature
    with pytest.raises(ValueError, match="needs home"):
        text_block(config)
    config["encoder"]["sha256"] = "0" * 64
    with pytest.raises(Exception, match="sha256"):
        text_block(config, home=tmp_path / "home")
    default = TextEmbeddings.for_project(load_project(overrides={"home": str(tmp_path / "h2")}))
    assert default.to_config()["encoder"]["kind"] == "hashing"               # nothing configured: offline encoder


def test_the_example_can_train_on_embeddings(triage, tmp_path):
    project = load_project(triage["root"], overrides={"pack_config": {"text": "embedding",
                                                                      "encoder": {"kind": "hashing", "dim": 64}}})
    project.home_override = tmp_path / "home"
    pack = triage["pack"]
    assert project.check(pack) == []
    provider = pack.provider(project)
    assert provider.feature_version.startswith("tickets-v1:") and len(provider.names("ticket")) == 66
    rebuilt = pack.provider(project, json.loads(json.dumps(provider.to_config())))
    assert rebuilt.feature_version == provider.feature_version
    assert np.array_equal(rebuilt.transform("ticket", TICKETS), provider.transform("ticket", TICKETS))
