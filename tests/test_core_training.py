"""Training principles, on generators whose true probabilities are known.

  (a) a calibrated generator comes out calibrated, with skill, and may act;
  (b) pure noise shows no skill and every decision abstains, saying why;
  (c) weak (judge) labels train the base model but never decide what a probability means;
  (d) feedback rows are up-weighted in training and never scored;
  (e) too little trusted data is an actionable error, not a silent model;
  (f) both base models save and load to identical predictions (no pickle);
  (g) explanations are sorted by size and their group shares add up.
"""
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from open_rlcd.core import calibration as C
from open_rlcd.core import training as T
from open_rlcd.core.config import Project
from open_rlcd.core.contract import DecideRequest, SystemOneRequest, TrainRequest
from open_rlcd.core.dataset import Dataset
from open_rlcd.core.head import CalibratedEnsemble, HeadSpec, lightgbm_available
from open_rlcd.core.service import Service
from tests.toypack import NAMES, ToyPack, ToyProvider, generate

NOUL = HeadSpec("good", "noul", view="v")
COLOR = HeadSpec("color", "choice", ("red", "green", "blue"), view="v")
needs_lgb = pytest.mark.skipif(not lightgbm_available(), reason="LightGBM not importable (macOS: brew install libomp)")


def make(n, k, signal, seed=0, source="synthetic", **kw) -> tuple[Dataset, np.ndarray]:
    X, y, P = generate(n, k, signal, seed=seed)
    ds = Dataset(X=X, y=y, feature_names=NAMES, feature_version="toy-v1", source=[source] * n,
                 feature_groups=ToyProvider().groups("v"), **kw)
    return ds, P


def capture_folds(monkeypatch) -> list:
    seen = []
    real = T.S.calibration_folds

    def spy(*a, **kw):
        folds, notes = real(*a, **kw)
        seen.extend(folds)
        return folds, notes

    monkeypatch.setattr(T.S, "calibration_folds", spy)
    return seen


# ----------------------------------------------------------------------------- (a) calibrated in, calibrated out

def test_a_calibrated_generator_comes_out_calibrated():
    ds, P = make(6000, 2, 1.5)
    cfg = T.TrainConfig()
    th = T.train_head(NOUL, ds, cfg)
    m = th.metrics
    assert m["has_skill"] is True and m["brier_gain_lower_95"] > 0 and m["ece"] < 0.04
    assert m["brier"] < m["baseline_brier"] and m["log_loss"] < m["baseline_log_loss"]
    assert th.verdict["medium"]["max_tier"] == "act" and th.verdict["medium"]["tier"] == 2
    assert th.calibration["method"] == "sigmoid" and th.calibration["ensemble_members"] == 5
    for b in th.report["reliability"]["per_class"]["true"]:
        if b["count"] >= 60:
            assert abs(b["mean_predicted"] - b["observed_rate"]) < 0.1, b
    # the generator's true probabilities are known: the served ones are close to them on unseen rows
    h = T.holdout_predictions(th, ds, cfg)
    assert np.abs(h["P"][:, 1] - P[h["idx"], 1]).mean() < 0.03
    assert len(h["idx"]) == th.report["n_holdout"] == th.data["holdout_rows"]


def test_a_calibrated_multiclass_generator_comes_out_calibrated():
    """One-vs-rest sigmoid on the log-softmax overstates the top class by about a point on a well-specified
    multinomial generator (act tier: 0.975 predicted vs 0.964 true), which `verify` may catch and cap at confirm.
    Temperature scaling is exact for this generator and is allowed to act."""
    ds, P = make(6000, 3, 1.5, seed=1)
    for method in ("auto", "temperature"):
        cfg = T.TrainConfig(calibration=method)
        th = T.train_head(COLOR, ds, cfg)
        assert th.metrics["has_skill"] and th.metrics["ece"] < 0.05
        tier = th.verdict["medium"]["max_tier"]
        assert tier == "act" if method == "temperature" else tier in ("act", "confirm")
        h = T.holdout_predictions(th, ds, cfg)
        assert np.allclose(h["P"].sum(1), 1) and np.abs(h["P"] - P[h["idx"]]).mean() < 0.04
        assert set(th.report["per_class"]) == {"red", "green", "blue"}


def test_the_holdout_is_never_fitted_or_calibrated_on(monkeypatch):
    folds = capture_folds(monkeypatch)
    ds, _ = make(3000, 2, 1.5, t=np.arange(3000) * 60)
    cfg = T.TrainConfig()
    th = T.train_head(NOUL, ds, cfg)
    hold = set(T.holdout_predictions(th, ds, cfg)["idx"].tolist())
    assert folds and th.data["ordering"] == "time"
    for tr, ca in folds:
        assert not hold & set(tr.tolist()) and not hold & set(ca.tolist())


# ----------------------------------------------------------------------------- (b) noise abstains

def test_pure_noise_shows_no_skill_and_escalates_at_every_stakes():
    ds, _ = make(4000, 2, 0.0, seed=3)
    th = T.train_head(NOUL, ds)
    assert th.metrics["has_skill"] is False and abs(th.metrics["brier_skill_score"]) < 0.02
    for stakes in ("low", "medium", "high"):
        v = th.verdict[stakes]
        assert v["max_tier"] == "escalate" and v["tier"] == 0 and "no skill over the base rate" in v["reason"]
    h = T.holdout_predictions(th, ds, T.TrainConfig())
    assert np.abs(h["P"][:, 1] - 0.5).max() < 0.15               # calibrated noise stays near the base rate


def test_a_service_decision_on_a_noise_head_abstains_and_says_why(tmp_path):
    svc = Service(Project(root=tmp_path), home=tmp_path, pack=ToyPack(signal=0.0, n=2400))
    job = svc.trainer.start(TrainRequest(), wait=True)
    assert job["status"] == "succeeded", job
    assert all(not h["metrics"]["has_skill"] for h in job["heads"].values())
    state = {"f": [2.0, -2.0, 1.5, 0.0, -1.0, 2.5]}
    d = svc.generic_decide(DecideRequest(state=state, stakes="low"))
    assert d["action"] == "escalate" and d["model"] == "rlcd-0.1.1"
    for name, r in d["heads"].items():
        assert r["decision"]["max_tier"] == "escalate" and r["decision"]["tier"] == "escalate", name
        assert r["answer"]["calibrated"] is True
    assert any("no skill over the base rate" in w for w in d["warnings"])
    out = svc.systemone(SystemOneRequest(state=state, questions={"good": {"type": "noul"}}))
    assert any("no skill" in w for w in out["warnings"]) and out["answers"]["good"]["confidence"] < 0.3


# ----------------------------------------------------------------------------- (c) weak labels

def test_flipped_judge_labels_train_but_calibration_stays_honest():
    X, y, P = generate(7000, 2, 1.5, seed=4)
    rng = np.random.default_rng(4)
    flip = rng.random(7000) < 0.3
    noisy = np.where(flip, 1 - y, y)
    trusted = np.arange(7000) < 2500
    source = np.where(trusted, "human", "judge")
    labels = np.where(trusted, y, noisy)                          # judge rows carry the flipped labels
    judge_y = np.where(trusted, noisy, -1)                        # the judge also labelled the human rows
    ds = Dataset(X=X, y=labels, feature_names=NAMES, feature_version="toy-v1", source=source, judge_y=judge_y)
    cfg = T.TrainConfig()
    th = T.train_head(NOUL, ds, cfg)
    assert th.metrics["has_skill"] and th.metrics["ece"] < 0.05
    h = T.holdout_predictions(th, ds, cfg)
    assert trusted[h["idx"]].all()                                # only human rows are scored
    assert np.abs(h["P"][:, 1] - P[h["idx"], 1]).mean() < 0.04    # close to the truth despite the judge
    ja = th.report["judge_agreement"]
    assert ja["n"] == 2500 and ja["accuracy"] == pytest.approx(0.7, abs=0.03)
    assert ja["cohen_kappa"] == pytest.approx(0.4, abs=0.07) and ja["confusion"]["rows"] == "human label"
    assert th.data["by_source"] == {"human": 2500, "judge": 4500} and th.data["trusted"] == 2500
    assert th.report["n_holdout"] <= 2500


def test_judge_rows_are_never_trusted_even_if_flagged():
    ds = Dataset(X=np.zeros((4, 1)), y=[0, 1, 0, 1], feature_names=["a"], feature_version="v",
                 source=["judge", "feedback", "human", "synthetic"], trusted=[True, True, True, True])
    assert ds.trusted.tolist() == [False, False, True, True]


# ----------------------------------------------------------------------------- (d) feedback rows

def test_feedback_rows_are_weighted_in_training_and_never_scored(monkeypatch):
    folds = capture_folds(monkeypatch)
    fits = []
    real_make = T.make_base

    def spy_make(*a, **kw):
        base = real_make(*a, **kw)
        real_fit = base.fit

        def fit(X, y, w=None):
            fits.append(np.asarray(w).copy())
            return real_fit(X, y, w)

        base.fit = fit
        return base

    monkeypatch.setattr(T, "make_base", spy_make)
    n = 3000
    X, y, _ = generate(n, 2, 1.5, seed=5)
    fb = np.arange(n) % 10 == 0
    ds = Dataset(X=X, y=y, feature_names=NAMES, feature_version="toy-v1",
                 source=np.where(fb, "feedback", "ground_truth"), weight=np.where(fb, 5.0, 1.0),
                 group=[f"g{i // 3}" if not f else f"feedback:{i}" for i, f in enumerate(fb)])
    cfg = T.TrainConfig()
    th = T.train_head(NOUL, ds, cfg)
    h = T.holdout_predictions(th, ds, cfg)
    assert not fb[h["idx"]].any() and th.report["n_holdout"] == len(h["idx"])
    assert len(fits) == len(folds) > 0
    for (tr, ca), w in zip(folds, fits, strict=True):
        assert fb[tr].any() and not fb[ca].any()                 # train side only
        assert np.array_equal(w, ds.weight[tr])                  # with their weight
        assert set(np.unique(w[fb[tr]])) == {5.0} and set(np.unique(w[~fb[tr]])) == {1.0}


# ----------------------------------------------------------------------------- (e) actionable failures

def test_too_few_trusted_rows_is_an_actionable_error():
    ds, _ = make(200, 2, 1.5)
    src = np.array(["judge"] * 200, dtype=object)
    src[:20] = "human"
    weak = Dataset(X=ds.X, y=ds.y, feature_names=NAMES, feature_version="toy-v1", source=src)
    with pytest.raises(T.TrainError, match="20 trusted") as e:
        T.train_head(NOUL, weak)
    assert "at least 30" in str(e.value) and "rlcd label audit" in str(e.value)
    only_judge = Dataset(X=ds.X, y=ds.y, feature_names=NAMES, feature_version="toy-v1", source=["judge"] * 200)
    with pytest.raises(T.TrainError, match="0 trusted"):
        T.train_head(NOUL, only_judge)


def test_single_class_trusted_data_is_an_actionable_error():
    X, _, _ = generate(500, 2, 1.0)
    y = np.r_[np.ones(300, int), np.zeros(200, int)]
    src = np.r_[["human"] * 300, ["judge"] * 200]                 # every trusted row says "true"
    ds = Dataset(X=X, y=y, feature_names=NAMES, feature_version="toy-v1", source=src)
    with pytest.raises(T.TrainError, match="carry the label 'true'") as e:
        T.train_head(NOUL, ds)
    assert "at least two classes" in str(e.value)


def test_a_class_without_any_rows_is_named():
    X, y, _ = generate(600, 3, 1.5)
    y = np.where(y == 1, 0, y)                                    # nobody ever labelled "green"
    ds = Dataset(X=X, y=y, feature_names=NAMES, feature_version="toy-v1", source=["human"] * 600)
    with pytest.raises(T.TrainError, match=r"no labelled rows for \['green'\]"):
        T.train_head(COLOR, ds)


def test_bad_labels_and_empty_data_are_errors():
    with pytest.raises(T.TrainError, match="no labelled rows"):
        T.train_head(NOUL, Dataset(X=np.zeros((0, 6)), y=[], feature_names=NAMES, feature_version="v"))
    ds, _ = make(300, 3, 1.0)
    with pytest.raises(T.TrainError, match="class indices 0..1"):
        T.train_head(NOUL, ds)                                    # 3-class labels on a noul head


def test_a_holdout_too_small_to_measure_is_an_error():
    ds, _ = make(60, 2, 1.5)
    with pytest.raises(T.TrainError, match="trusted holdout rows"):
        T.train_head(NOUL, ds, T.TrainConfig(holdout_fraction=0.1))


# ----------------------------------------------------------------------------- (f) save / load

@pytest.mark.parametrize("model", ["logreg", pytest.param("lightgbm", marks=needs_lgb)])
def test_heads_save_and_load_to_identical_predictions(tmp_path, model):
    spec = HeadSpec("color", "choice", ("red", "green", "blue"), view="v", model=model)
    ds, _ = make(2500, 3, 1.5, seed=6)
    th = T.train_head(spec, ds, T.TrainConfig(n_estimators=40))
    assert th.model == model and th.ensemble.kind == model
    files = th.ensemble.save(tmp_path / "h")
    suffix = ".safetensors" if model == "logreg" else ".txt"
    assert files[-1] == "ensemble.json" and all(f.endswith(suffix) for f in files[:-1])
    back = CalibratedEnsemble.load(tmp_path / "h")
    Xq = np.random.default_rng(0).normal(size=(200, 6)) * 2
    Xq[::7, 2] = np.nan                                           # missing values too
    assert np.array_equal(th.ensemble.predict_proba(Xq), back.predict_proba(Xq))
    assert back.spec == spec and back.feature_names == NAMES and back.feature_version == "toy-v1"


FAISS_THEN_LIGHTGBM = """
import numpy as np, faiss
index = faiss.IndexFlatIP(16)
index.add(np.random.rand(20000, 16).astype("float32"))
index.search(np.random.rand(500, 16).astype("float32"), 5)          # faiss's own OpenMP runtime, multithreaded
from open_rlcd.core.dataset import Dataset
from open_rlcd.core.head import HeadSpec, openmp_conflict
from open_rlcd.core.training import TrainConfig, train_head
from tests.toypack import NAMES, generate
assert openmp_conflict()
X, y, _ = generate(1500, 3, 1.5)
ds = Dataset(X=X, y=y, feature_names=NAMES, feature_version="v", source=["synthetic"] * 1500)
th = train_head(HeadSpec("c", "choice", ("a", "b", "c"), model="lightgbm"), ds, TrainConfig(n_estimators=30))
th.ensemble.predict_proba(X[:100]); th.ensemble.explain(X[:1], 0)
print("survived")
"""


@needs_lgb
@pytest.mark.skipif(sys.platform != "darwin", reason="the two-OpenMP-runtimes crash is specific to macOS wheels")
def test_lightgbm_survives_faiss_in_the_same_process():
    """faiss-cpu and LightGBM each bring an OpenMP runtime on macOS; multithreaded LightGBM after a FAISS search
    segfaults the interpreter. Run in a subprocess so a regression fails this test instead of killing pytest."""
    pytest.importorskip("faiss")
    root = Path(__file__).resolve().parents[1]
    r = subprocess.run([sys.executable, "-c", FAISS_THEN_LIGHTGBM], cwd=root, capture_output=True, text=True,
                       timeout=120, env={**os.environ, "PYTHONPATH": str(root)})
    assert r.returncode == 0 and "survived" in r.stdout, (r.returncode, r.stderr[-2000:])


def test_model_auto_is_recorded(tmp_path):
    spec = HeadSpec("good", "noul", view="v", model="auto")
    th = T.train_head(spec, make(1500, 2, 1.5)[0], T.TrainConfig(n_estimators=30))
    assert th.model == ("lightgbm" if lightgbm_available() else "logreg")
    if th.model == "logreg":
        assert any("model 'auto'" in n for n in th.notes)


# ----------------------------------------------------------------------------- (g) explanations

@pytest.mark.parametrize("model", ["logreg", pytest.param("lightgbm", marks=needs_lgb)])
def test_explain_sorts_reasons_and_group_shares_sum_to_one(model):
    spec = HeadSpec("color", "choice", ("red", "green", "blue"), view="v", model=model)
    th = T.train_head(spec, make(2500, 3, 1.5, seed=7)[0], T.TrainConfig(n_estimators=40))
    x = np.array([[1.5, -0.5, 0.2, 2.0, -1.0, 0.1]], np.float32)
    for ci in range(3):
        reasons, groups = th.ensemble.explain(x, ci, top=4, groups=ToyProvider().groups("v"))
        assert 0 < len(reasons) <= 4
        sizes = [abs(r["contribution"]) for r in reasons]
        assert sizes == sorted(sizes, reverse=True)
        assert all(r["feature"] in NAMES and r["effect"].endswith(f"P({spec.labels[ci]})") for r in reasons)
        assert all(("raises" in r["effect"]) == (r["contribution"] > 0) for r in reasons)
        assert set(groups) <= {"a", "b"} and sum(groups.values()) == pytest.approx(1.0, abs=1e-3)
        assert list(groups.values()) == sorted(groups.values(), reverse=True)


def test_logreg_explanations_are_exact_for_a_binary_head():
    th = T.train_head(NOUL, make(2000, 2, 1.5, seed=8)[0])
    x = np.array([0.3, -1.2, 0.8, 0.0, 1.1, -0.4], np.float32)
    c_true = th.ensemble.contributions(x, 1)
    assert np.allclose(th.ensemble.contributions(x, 0), -c_true)   # P(false) moves opposite to P(true)
    base = th.ensemble.members[0][0]
    z = (x - base.mean) / base.scale
    assert np.allclose(base.contributions(x, 1), z * base.coef[0])


# ----------------------------------------------------------------------------- more edge cases

def test_missing_and_infinite_values_train_and_predict():
    ds, _ = make(2500, 2, 1.5, seed=9)
    X = ds.X.copy()
    rng = np.random.default_rng(9)
    X[rng.random(X.shape) < 0.1] = np.nan
    X[rng.random(X.shape) < 0.01] = np.inf
    X[:, 5] = np.nan                                              # one feature never observed
    bad = Dataset(X=X, y=ds.y, feature_names=NAMES, feature_version="toy-v1", source=ds.source)
    th = T.train_head(NOUL, bad)
    P = th.ensemble.predict_proba(np.array([[np.nan, np.inf, -np.inf, 0, 1, np.nan]]))
    assert np.isfinite(P).all() and P.sum() == pytest.approx(1) and th.metrics["has_skill"]


def test_a_rare_class_missing_from_some_calibration_blocks():
    X, y, _ = generate(3000, 3, 1.5, seed=10)
    y = y.copy()
    rare = np.flatnonzero(y == 2)
    y[rare[12:]] = 0                                              # class "blue" keeps 12 rows only
    ds = Dataset(X=X, y=y, feature_names=NAMES, feature_version="toy-v1", source=["human"] * 3000)
    th = T.train_head(COLOR, ds)
    P = th.ensemble.predict_proba(np.random.default_rng(0).normal(size=(50, 6)))
    assert np.allclose(P.sum(1), 1) and (P >= 0).all()
    assert th.report["base_rates_train"]["blue"] < 0.01


def test_gate_head_with_a_conformal_target_reports_threshold_and_holdout_check():
    spec = HeadSpec("good", "noul", view="v", decision="gate", conformal=0.1)
    th = T.train_head(spec, make(6000, 2, 2.0, seed=11)[0])
    sel, chk = th.conformal["selective"], th.conformal["holdout"]
    assert sel["target_error"] == 0.1 and sel["delta"] == 0.1 and sel["threshold"] is not None
    assert chk["n_selected"] > 0 and chk["consistent"] is True
    assert th.metrics["conformal_threshold"] == sel["threshold"]
    s = th.summary()["conformal"]
    assert s["threshold"] == sel["threshold"] and s["holdout"] == chk
    pol = th.report["policy"]["holdout"]["medium"]
    assert all(p >= sel["threshold"] for p in [pol["act"]["mean_predicted"]] if p is not None)


def test_choice_head_with_a_conformal_target_reports_prediction_sets():
    spec = HeadSpec("color", "choice", ("red", "green", "blue"), view="v", conformal=0.1)
    th = T.train_head(spec, make(5000, 3, 1.5, seed=12)[0])
    sets = th.conformal["sets"]
    assert 0 < sets["q"] <= 1 and th.summary()["conformal"]["set_quantile"] == sets["q"]
    assert th.conformal["holdout"]["coverage"] == pytest.approx(0.9, abs=0.05)


@pytest.mark.parametrize("method", ["temperature", "isotonic"])
def test_requested_calibration_methods(method):
    th = T.train_head(COLOR, make(3000, 3, 1.5, seed=13)[0], T.TrainConfig(calibration=method))
    assert th.calibration["method"] == method and th.calibration["why"] == "requested"
    assert th.metrics["has_skill"]


def test_auto_calibration_switches_to_isotonic_with_enough_rows():
    th = T.train_head(NOUL, make(4000, 2, 1.5, seed=14)[0], T.TrainConfig(isotonic_min_per_class=100))
    assert th.calibration["method"] == "isotonic" and th.metrics["ece"] < 0.06


def test_segments_get_their_own_baseline_and_slice():
    n = 6000
    seg = np.where(np.arange(n) % 2 == 0, "eu", "us")
    ds, _ = make(n, 2, 1.5, seed=15, segment=seg)
    th = T.train_head(NOUL, ds)
    by = th.report["by_segment"]
    assert set(by) == {"eu", "us"} and all(r["n_holdout"] >= 50 and r["n_train_trusted"] > 0 for r in by.values())
    assert sum(r["n_holdout"] for r in by.values()) == th.report["n_holdout"]
    assert "segment" in th.report["definitions"]["baseline"]


def test_trained_head_summary_shape():
    th = T.train_head(NOUL, make(1500, 2, 1.5, seed=16)[0])
    s = th.summary()
    assert set(s) == {"head", "type", "model", "trained_at", "metrics", "verdict", "calibration", "conformal",
                      "data", "notes"}
    assert s["head"] == "good" and s["conformal"] is None and s["trained_at"].endswith("Z")
    assert set(s["verdict"]) == {"low", "medium", "high"}
    assert s["metrics"]["verdict"] == {k: v["max_tier"] for k, v in s["verdict"].items()}
    assert s["data"]["rows"] == 1500 and s["data"]["ordering"] == "row"
    assert T.TrainConfig().isotonic_min_per_class == C.ISOTONIC_MIN_PER_CLASS
