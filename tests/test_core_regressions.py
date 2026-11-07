"""Regression tests for issues found while integrating the packs (each fails if its fix is reverted)."""
from __future__ import annotations

import numpy as np
import pytest

from open_rlcd.core import metrics as M
from open_rlcd.core import splits as S
from open_rlcd.core.config import Project
from open_rlcd.core.dataset import Dataset
from open_rlcd.core.head import HeadSpec
from open_rlcd.core.service import Service
from open_rlcd.core.training import TrainConfig, train_head
from open_rlcd.labeling.store import LabelItem
from open_rlcd.packs.base import epoch
from tests.toypack import NAMES, ToyPack, generate


def test_near_constant_forecast_closer_to_the_holdout_base_rate_is_not_skill():
    """A forecaster that knows nothing but happens to sit nearer this holdout's base rate than the training base
    rate shows a consistent Brier gain over the training base rate. That is not skill."""
    rng = np.random.default_rng(0)
    n = 2000
    y = (rng.random(n) < 0.30).astype(int)              # holdout base rate 0.30
    P1 = np.full(n, 0.31) + rng.normal(0, 0.002, n)       # knows nothing, sits near 0.30
    P = np.column_stack([1 - P1, P1])
    prior = np.array([0.6, 0.4])                          # training base rate 0.40
    rep = M.calibration_report(P, y, ["false", "true"], prior)
    assert rep["skill"]["lower_95"] > 0                   # the training-baseline test alone would claim skill
    assert rep["skill"]["has_skill"] is False
    assert rep["skill"]["vs_holdout_base_rate"]["mean_brier_gain"] <= 0
    assert "why_not" in rep["skill"]


def test_real_signal_passes_both_skill_tests():
    X, y, P = generate(3000, 2, 1.0, seed=1)
    rep = M.calibration_report(P, y, ["false", "true"], np.array([0.5, 0.5]))
    assert rep["skill"]["has_skill"] is True
    assert rep["skill"]["vs_holdout_base_rate"]["lower_95"] > 0


@pytest.mark.parametrize("seed", range(12))
def test_pure_noise_never_claims_skill_through_training(seed):
    X, y, _ = generate(1500, 2, 0.0, seed=seed)
    ds = Dataset(X=X, y=y, feature_names=NAMES, feature_version="t", group=[f"g{j // 3}" for j in range(1500)],
                 t=np.arange(1500) * 60)
    th = train_head(HeadSpec("h", "noul"), ds, TrainConfig(seed=seed))
    assert th.report["skill"]["has_skill"] is False
    assert all(v["max_tier"] == "escalate" for v in th.verdict.values())


def test_noul_label_index_accepts_the_heads_positive_value_and_score_descriptions():
    churn = HeadSpec("churned", "noul", positive="Churn")
    assert churn.label_index("churn") == 1 and churn.label_index("no") == 0
    with pytest.raises(ValueError):
        churn.label_index("maybe")
    sat = HeadSpec("sat", "score", ("0", "1", "2"), levels=("unhappy", "neutral", "happy"))
    assert sat.label_index("happy") == 2 and sat.label_index("1") == 1 and sat.label_index(0) == 0


def test_epoch_reads_naive_iso_as_utc_and_detects_milliseconds():
    assert epoch("2025-01-01T00:00:00") == 1735689600
    assert epoch("2025-01-01T00:00:00Z") == 1735689600
    assert epoch("2025-01-01T09:00:00+09:00") == 1735689600
    assert epoch(1735689600000) == 1735689600 and epoch("1735689600") == 1735689600
    assert epoch(None) is None and epoch("not a date") is None


def test_label_item_ids_are_strings():
    assert LabelItem(id=7, view="v", state={}).id == "7"


def test_service_home_is_the_home_the_pack_sees(tmp_path, monkeypatch):
    monkeypatch.setenv("RLCD_HOME", str(tmp_path / "elsewhere"))
    svc = Service(Project(root=tmp_path), home=tmp_path / "var", pack=ToyPack())
    assert svc.project.home == (tmp_path / "var").resolve() == svc.home


def test_actively_sampled_rows_train_and_calibrate_but_are_never_in_the_holdout():
    X, y, _ = generate(1200, 2, 1.0, seed=3)
    ok = np.ones(1200, bool)
    ok[::4] = False                                        # every 4th row came from the active-learning queue
    groups = np.array([f"g{j}" for j in range(1200)], dtype=object)
    ds = Dataset(X=X, y=y, feature_names=NAMES, feature_version="t", group=groups, holdout_ok=ok)
    sp = S.holdout_split(ds, 0.2, "group")
    assert not (sp.holdout & ~ok).any()
    assert (sp.dev & ~ok).sum() == (~ok).sum()            # they all stay in development
    folds, _ = S.calibration_folds(ds, sp.dev, 5, "group", n_classes=2)
    cal = np.concatenate([c for _, c in folds])
    assert (~ok[cal]).any()                                # ... and calibrate there
    assert any("active learning" in n for n in sp.notes)


def test_input_guard_warns_when_a_request_lacks_features_training_always_had(tmp_path):
    from open_rlcd.core.contract import SystemOneRequest, TrainRequest

    svc = Service(Project(root=tmp_path), home=tmp_path / "var", pack=ToyPack(n=1500))
    assert svc.trainer.start(TrainRequest(heads=["good"]), wait=True)["status"] == "succeeded"
    q = {"g": {"type": "noul", "head": "good"}}
    full = svc.systemone(SystemOneRequest(state={"f": [0.1] * 6}, questions=q))
    assert not any("lacks" in w for w in full.get("warnings", []))
    x = [0.1, None, 0.1, 0.1, 0.1, 0.1]
    gap = svc.systemone(SystemOneRequest(state={"x": x, "feature_version": "toy-v1"}, questions=q))
    assert any("lacks 1 input feature" in w and "f1" in w for w in gap["warnings"])
