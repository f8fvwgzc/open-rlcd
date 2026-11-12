"""Regression tests for the verified review findings fixed in core, the service, bundles and security (group A).

Each test names its finding (F01 ... F45) and fails on the code before the fix."""
from __future__ import annotations

import importlib
import json
import os
import re
import shutil

import numpy as np
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from open_rlcd.connectors.base import ConnectorError
from open_rlcd.connectors.security import check_host
from open_rlcd.core import bundle as BD
from open_rlcd.core import calibration as C
from open_rlcd.core import policy as PL
from open_rlcd.core import splits as S
from open_rlcd.core import training as T
from open_rlcd.core.config import Project
from open_rlcd.core.contract import ContractError, DecideRequest, FeedbackRequest, SystemOneRequest, TrainRequest
from open_rlcd.core.dataset import Dataset
from open_rlcd.core.feedback import FeedbackLog
from open_rlcd.core.head import HeadSpec, lightgbm_available
from open_rlcd.core.registry import Registry
from open_rlcd.core.service import Service
from open_rlcd.labeling.store import LabelItem
from open_rlcd.packs.base import BUILTIN, dataset_from_labels, load_pack
from open_rlcd.serve.api import create_app
from tests.toypack import NAMES, ToyPack, ToyProvider, generate

_W = np.random.default_rng(99 + 2).normal(size=(6, 2))
_D = (_W[:, 1] - _W[:, 0]) / np.linalg.norm(_W[:, 1] - _W[:, 0])
POS = {"f": (2.5 * _D).round(3).tolist()}
PACK = {"name": "toy", "version": "0.1.0", "config": {"kind": "toy"}}


def make_service(tmp_path, pack=None, data=None, name="home") -> Service:
    root = tmp_path / f"{name}-root"
    root.mkdir(parents=True, exist_ok=True)
    return Service(Project(root=root, data=dict(data or {})), home=tmp_path / name, pack=pack or ToyPack(n=3000))


def train(svc: Service, **kw) -> dict:
    job = svc.trainer.start(TrainRequest(**kw), wait=True)
    assert job["status"] in ("succeeded", "partial"), job
    return job


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    """A service with every toy head trained (shared, read-only use)."""
    svc = make_service(tmp_path_factory.mktemp("trained"))
    train(svc)
    return svc


# ============================================================================ F01 / F17 / F18: groups and splits

class EmbedProvider:
    """A state's feature row is its `x` (an embedding of the query, say)."""

    feature_version = "emb-v1"

    def __init__(self, d=32):
        self.d = d

    def names(self, view):
        return [f"e{i}" for i in range(self.d)]

    def transform(self, view, states):
        return np.asarray([s["x"] for s in states], np.float32).reshape(len(states), self.d)

    def groups(self, view):
        return {}


def _noise_labels(seed=0, n=400, d=32, share=0.5, with_groups=True, with_time=True):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    y = rng.integers(0, 2, n)
    labelled, feedback = [], []
    for i in range(n):
        st = {"q": f"query {i}", "x": X[i].tolist()}
        it = LabelItem(id=f"i{i}", view="v", state=st, group=f"query:{i}" if with_groups else None,
                       t=1_700_000_000 + 3600 * i if with_time else None)
        labelled.append({"item": it, "label": bool(y[i]), "source": "human", "judge_label": None, "note": None})
        if rng.random() < share:   # the same query was decided in production and its outcome posted
            feedback.append({"head": "h", "state": st, "label": bool(y[i]), "decision_id": f"d{i}", "id": f"f{i}",
                             "time": 1_700_000_000 + 3600 * i + 60})
    return labelled, feedback


def _holdout_twins(ds, split):
    fb_dev = ds.X[split.dev & (ds.source == "feedback")]
    keys = {r.tobytes() for r in np.ascontiguousarray(fb_dev, np.float32)}
    return sum(np.ascontiguousarray(ds.X[i], np.float32).tobytes() in keys for i in np.flatnonzero(split.holdout))


def test_f01_feedback_rows_carry_their_items_group_and_never_twin_a_holdout_row():
    spec = HeadSpec("h", "noul")
    labelled, feedback = _noise_labels()
    ds = dataset_from_labels(spec, EmbedProvider(), labelled, feedback)
    fb = ds.source == "feedback"
    # linked by identical state: the feedback row's group is its labelled item's group
    assert all(g.startswith("query:") for g in ds.group[fb])
    assert any("feedback rows are about a labelled item" in n for n in ds.notes)
    split = S.holdout_split(ds, 0.2, "auto", 7)
    assert split.mode == "group"
    assert _holdout_twins(ds, split) == 0                      # was 39: exact twins of training rows were scored
    assert split.holdout.sum() >= 75                           # the holdout still holds ~20 % (80 rows)
    assert not set(ds.group[split.holdout]) & set(ds.group[fb])
    note = next(n for n in split.notes if "holdout groups have feedback rows" in n)
    moved = int(note.split()[0])
    assert 0 < moved <= len(set(ds.group[~fb]))                # real item groups, not feedback pseudo-groups


def test_f01_group_of_links_feedback_without_an_identical_state():
    spec = HeadSpec("h", "noul")
    labelled, _ = _noise_labels(n=60, share=0)
    fb = [{"head": "h", "state": {"q": "query 3", "x": [0.0] * 32}, "label": True, "decision_id": "d", "id": "1"}]
    ds = dataset_from_labels(spec, EmbedProvider(), labelled, fb,
                             group_of=lambda s: "query:" + s["q"].split()[-1])
    assert ds.group[ds.source == "feedback"].tolist() == ["query:3"]


def test_f01_split_backstop_removes_exact_twins_in_group_and_time_mode():
    rng = np.random.default_rng(3)
    n = 300
    X = rng.normal(size=(n, 4))
    twins = np.arange(n - 60, n, 2)                             # late rows, which the holdout would score
    Xf = X[twins]
    y = rng.integers(0, 2, n)
    base = dict(feature_names=list("abcd"), feature_version="v")
    t = np.arange(n) * 60
    # group mode: the feedback rows got pseudo-groups (not linked), only their features match
    ds = Dataset(X=np.vstack([X, Xf]), y=np.concatenate([y, y[twins]]),
                 group=[f"g{i}" for i in range(n)] + [f"feedback:d{i}" for i in twins],
                 t=np.concatenate([t, t[twins] + 30]),
                 source=["human"] * n + ["feedback"] * len(twins), **base)
    sp = S.holdout_split(ds, 0.2, "group")
    assert _holdout_twins(ds, sp) == 0 and sp.holdout.sum() >= 55
    assert any("holdout groups have feedback rows" in n_ for n_ in sp.notes)
    # time mode, no groups: the twins are removed from the holdout
    ds_t = Dataset(X=np.vstack([X, Xf]), y=np.concatenate([y, y[twins]]), t=np.concatenate([t, t[twins] + 30]),
                   source=["human"] * n + ["feedback"] * len(twins), **base)
    sp_t = S.holdout_split(ds_t, 0.2, "time")
    assert _holdout_twins(ds_t, sp_t) == 0
    assert any("exactly the features of a feedback row" in n_ for n_ in sp_t.notes)


def test_f01_a_common_feature_vector_is_not_taken_for_one_item():
    # low-cardinality features: many groups share each vector, so a feedback row does not knock them out
    rng = np.random.default_rng(1)
    n = 400
    X = rng.integers(0, 2, size=(n, 2)).astype(float)
    ds = Dataset(X=np.vstack([X, [[1.0, 1.0]]]), y=np.concatenate([rng.integers(0, 2, n), [1]]),
                 feature_names=["a", "b"], feature_version="v",
                 group=[f"g{i}" for i in range(n)] + ["feedback:d"], source=["human"] * n + ["feedback"])
    sp = S.holdout_split(ds, 0.2, "group")
    assert 75 <= sp.holdout.sum() <= 85
    assert any("several groups" in n_ for n_ in sp.notes)


def test_f01_noise_with_feedback_duplicates_does_not_show_skill():
    spec = HeadSpec("h", "noul")
    for seed in range(3):
        labelled, feedback = _noise_labels(seed=100 + seed, d=96, share=0.6)
        ds = dataset_from_labels(spec, EmbedProvider(96), labelled, feedback)
        assert _holdout_twins(ds, S.holdout_split(ds, 0.2, "auto", seed)) == 0
        th = T.train_head(spec, ds, T.TrainConfig(seed=seed))
        assert th.report["n_holdout"] >= 75 and not th.report["skill"]["has_skill"]


def test_f17_items_without_groups_get_time_ordering():
    spec = HeadSpec("h", "noul")
    labelled, feedback = _noise_labels(n=120, with_groups=False)
    ds = dataset_from_labels(spec, EmbedProvider(), labelled, feedback)
    assert ds.group is None and S.resolve_mode(ds) == "time"
    grouped, _ = _noise_labels(n=120)
    assert S.resolve_mode(dataset_from_labels(spec, EmbedProvider(), grouped)) == "group"


def test_f18_group_mode_with_timestamps_says_there_is_no_time_cut():
    spec = HeadSpec("h", "noul")
    labelled, _ = _noise_labels(n=200, share=0)
    sp = S.holdout_split(dataset_from_labels(spec, EmbedProvider(), labelled), 0.2)
    assert any("ordered by their first timestamp" in n and "ordering: time" in n for n in sp.notes)


# ============================================================================ F03 / F42: conformal

def test_f03_cross_fit_scores_never_depend_on_the_rows_own_label():
    rng = np.random.default_rng(0)
    raw = rng.normal(size=60)
    y = (rng.random(60) < 1 / (1 + np.exp(-raw))).astype(int)
    a = T.cross_fit_scores("sigmoid", 2, raw, y)
    for i in (0, 7, 31):
        y2 = y.copy()
        y2[i] = 1 - y2[i]
        b = T.cross_fit_scores("sigmoid", 2, raw, y2)
        assert np.allclose(a[i], b[i])            # an in-sample calibrator would move this row's own score
    # the served calibrator (fitted on the whole block) does see the row: its score moves
    full = C.Calibrator("sigmoid", 2).fit(raw, y).predict(raw)
    y2 = y.copy()
    y2[0] = 1 - y2[0]
    assert not np.allclose(full[0], C.Calibrator("sigmoid", 2).fit(raw, y2).predict(raw)[0])


def _gen(rng, W, n):
    X = rng.normal(size=(n, W.shape[0])).astype(np.float32)
    L = X @ W
    P = np.exp(L - L.max(1, keepdims=True))
    P /= P.sum(1, keepdims=True)
    return X, (rng.random(n)[:, None] > np.cumsum(P, 1)).sum(1)


def test_f03_prediction_sets_of_the_served_model_cover_on_fresh_rows():
    """Through training: 8 classes, 200 rows, alpha = 0.1. Before the fix the quantile came from calibrators fitted
    to the rows they scored and fresh coverage was about 0.85."""
    k, alpha = 8, 0.1
    W = np.random.default_rng(3).normal(size=(6, k))
    spec = HeadSpec("c", "choice", tuple("abcdefgh"), conformal=alpha)
    covs = []
    for s in range(15):
        X, y = _gen(np.random.default_rng(55 + s), W, 200)
        th = T.train_head(spec, Dataset(X=X, y=y, feature_names=[f"f{i}" for i in range(6)], feature_version="v"),
                          T.TrainConfig(seed=s))
        q = th.conformal["sets"]["q"]
        Xf, yf = _gen(np.random.default_rng(10**7 + s), W, 20000)
        Pf = th.ensemble.predict_proba(Xf)
        covs.append(np.mean(1 - Pf[np.arange(len(yf)), yf] <= q + 1e-12))
    assert np.mean(covs) >= 1 - alpha


def test_f42_set_quantile_is_the_standard_order_statistic():
    s = np.linspace(0.01, 1.0, 100)                    # scores 1 - p, sorted: the m-th smallest is s[m - 1]
    q = C.set_quantile(1 - s, 0.1)["q"]                # m = ceil(101 * 0.9) = 91
    assert q == pytest.approx(s[90])
    s19 = np.linspace(0.05, 0.95, 19)
    assert C.set_quantile(1 - s19, 0.1)["q"] == pytest.approx(s19[17])     # m = 18
    odd = 1 - np.array([0.1234567891, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 0.999] * 3)
    q = C.set_quantile(odd, 0.2)["q"]
    assert q in set((1 - odd).tolist())                # never rounded away from an actual score


# ============================================================================ F02 / F14 / F34: packs

def test_f02_a_bundle_cannot_name_code_to_import(trained, tmp_path, monkeypatch):
    version = trained.registry.latest_id()
    src = trained.bundle(version).path
    bad = tmp_path / "evil"
    shutil.copytree(src, bad)
    meta = json.loads((bad / "model.json").read_text())
    meta["pack"]["name"] = "os:abort"
    (bad / "model.json").write_text(json.dumps(meta))
    with pytest.raises(BD.BundleError, match="never names code"):
        BD.validate(bad)
    # a bundle already inside RLCD_HOME whose pack name was edited: refused without importing anything
    svc = make_service(tmp_path, name="h2")
    train(svc, heads=["good"])
    v = svc.registry.latest_id()
    mj = svc.bundle(v).path / "model.json"
    m = json.loads(mj.read_text())
    m["pack"]["name"] = "faulthandler:dump_traceback"
    mj.write_text(json.dumps(m))
    imported = []
    real = importlib.import_module
    monkeypatch.setattr(importlib, "import_module", lambda name, *a: imported.append(name) or real(name, *a))
    fresh = make_service(tmp_path, name="h2")
    with pytest.raises(ContractError) as e:
        fresh.pack_for(v)
    assert e.value.kind == "bad_bundle" and "never names code" in e.value.message
    assert "error" in fresh.health()["heads"]
    assert "faulthandler" not in imported


def test_f14_a_custom_pack_from_rlcd_yaml_serves_after_training(tmp_path, monkeypatch):
    (tmp_path / "fix14_custom_pack.py").write_text(
        "from tests.toypack import ToyPack\n\n\nclass ApplicationsPack(ToyPack):\n    name = 'applications'\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    data = {"pack": "fix14_custom_pack:ApplicationsPack"}
    svc = Service(Project(root=tmp_path, data=data), home=tmp_path / "home")
    job = train(svc, heads=["good"])
    version = job["model"]
    assert svc.bundle(version).meta["pack"]["name"] == "applications"
    again = Service(Project(root=tmp_path, data=data), home=tmp_path / "home")     # a fresh process
    assert again.pack_for(version).name == "applications"
    assert {h["name"] for h in again.heads_view(None)["heads"]} >= {"good"}
    d = again.generic_decide(DecideRequest(state=POS, heads=["good"]))
    assert d["heads"]["good"]["answer"]["calibrated"] is True
    assert again.health()["heads"]["good"] == {"trained": True}


def test_f34_the_trading_pack_ships_with_its_branch():
    assert "trading" in BUILTIN
    assert load_pack("trading").name == "trading"


# ============================================================================ F04 / F20 / F21: bundles

def _toy_heads(model="logreg"):
    good = HeadSpec("good", "noul", view="v", decision="gate", conformal=0.1, model=model)
    X, y, _ = generate(1500, 2, 1.5, seed=0)
    ds = Dataset(X=X, y=y, feature_names=NAMES, feature_version="toy-v1", source=["synthetic"] * 1500)
    return {"good": T.train_head(good, ds, T.TrainConfig(n_estimators=20))}


def _write(dest, heads):
    BD.write_bundle(dest, version="rlcd-0.1.1", created_at="2026-11-07T00:00:00Z", pack=PACK,
                    feature_version="toy-v1", encoder=None, policy={"abstain": "escalate"}, trained=heads)
    return dest


@pytest.mark.skipif(not lightgbm_available(), reason="LightGBM not importable")
def test_f04_a_malformed_lightgbm_member_fails_the_import_not_the_server(tmp_path):
    src = _write(tmp_path / "b", _toy_heads("lightgbm"))
    reg = Registry(tmp_path / "home")
    assert reg.import_(src) == "rlcd-0.1.1"                  # a well-formed LightGBM bundle passes the check
    member = src / "heads" / "good" / "member_0.txt"
    member.write_text(re.sub(r"split_feature=(\d+)", "split_feature=99999999", member.read_text(), count=1))
    meta = json.loads((src / "model.json").read_text())
    meta["heads"]["good"]["files"]["member_0.txt"] = BD.sha256(member)
    (src / "model.json").write_text(json.dumps(meta))
    assert BD.validate(src)                                  # hashes are self-consistent: only parsing catches it
    with pytest.raises(BD.BundleError, match="LightGBM member"):
        reg.import_(src)
    assert reg.latest_id() == "rlcd-0.1.1" and not (tmp_path / "home" / "models" / "rlcd-0.1.2").exists()


def test_f20_policy_caps_come_from_the_hashed_head_json(tmp_path):
    heads = _toy_heads()
    src = _write(tmp_path / "b", heads)
    honest = heads["good"].verdict
    meta = json.loads((src / "model.json").read_text())
    meta["heads"]["good"]["verdict"] = {s: {"max_tier": "act", "tier": 2, "reason": "x"} for s in honest}
    meta["heads"]["good"]["conformal"] = {"threshold": 0.0}
    (src / "model.json").write_text(json.dumps(meta))
    h = BD.Bundle(src).head("good")
    assert h.verdict == honest                               # model.json's edited entry is not what is enforced
    assert h.conformal_threshold == heads["good"].conformal["selective"]["threshold"]
    hj = src / "heads" / "good" / "head.json"
    summary = json.loads(hj.read_text())
    summary["verdict"] = meta["heads"]["good"]["verdict"]
    hj.write_text(json.dumps(summary))
    with pytest.raises(BD.BundleError, match="head.json"):
        _ = BD.Bundle(src).head("good").verdict


def test_f21_a_symlink_planted_after_validation_is_refused(tmp_path, monkeypatch):
    src = _write(tmp_path / "b", _toy_heads())
    secret = tmp_path / "outside-secret.txt"
    secret.write_text("PRIVATE")
    real = BD.validate
    calls = []

    def racing(p, *a, **kw):
        m = real(p, *a, **kw)
        calls.append(p)
        if len(calls) == 1:          # the attacker wins the race right after the source was validated
            os.symlink(secret, src / "heads" / "notes.txt")
        return m

    monkeypatch.setattr(BD, "validate", racing)
    reg = Registry(tmp_path / "home")
    with pytest.raises(BD.BundleError, match="symlink"):
        reg.import_(src)
    assert reg.registry()["versions"] == {} and not any((tmp_path / "home" / "models").iterdir())


# ============================================================================ F05: loopback only

@pytest.mark.parametrize("target", ["http://evil.example/x.sock", "https://evil.example/v1.sock",
                                    "unix+http://evil.com/", "x.sock", "evil.example:80/x.sock"])
def test_f05_a_sock_suffix_is_not_a_unix_socket(target):
    with pytest.raises(ValueError, match="refusing non-loopback"):
        check_host(target)


@pytest.mark.parametrize("target", ["unix:///var/run/qdrant.sock", "http+unix://%2Ftmp%2Fs.sock/", "/run/pg.sock",
                                    "./pg.sock", "postgresql:///docs?host=/var/run/postgresql"])
def test_f05_real_unix_sockets_stay_local(target):
    assert check_host(target) == "unix"


# ============================================================================ F06 / F41: the ledger

def test_f06_feedback_for_a_carried_over_head_still_updates_the_ledger(tmp_path):
    svc = make_service(tmp_path)
    train(svc, heads=["good"])
    before = svc.generic_decide(DecideRequest(state=POS, heads=["good", "color"]))["decision_id"]
    train(svc, heads=["level"])                              # 'good' is carried over unchanged
    v2 = svc.registry.latest_id()
    assert svc.bundle(v2).meta["heads"]["good"]["carried_over"] is True
    r = svc.feedback(FeedbackRequest(decision_id=before, head="good", label="true", source="crm"))
    assert r["bernoulli"]["updated"] is True                 # was: "made by a model version other than ..."
    assert svc.ledger.table("good")["model_version"] == v2
    # an untrained head has no table at all: a distinct reason
    r = svc.feedback(FeedbackRequest(decision_id=before, head="color", label="red", source="crm"))
    assert r["bernoulli"]["updated"] is False and "not trained" in r["bernoulli"]["reason"]
    # a retrained head starts a new table: decisions of the old head no longer count
    train(svc, heads=["good"])
    r = svc.feedback(FeedbackRequest(decision_id=before, head="good", label="false", source="crm"))
    assert r["bernoulli"]["updated"] is False and "other than" in r["bernoulli"]["reason"]


# ============================================================================ F19 / F45: the input guard

def test_f19_a_request_missing_always_present_inputs_never_acts(tmp_path):
    trained = make_service(tmp_path, pack=ToyPack(n=4000))
    train(trained, heads=["good"])
    w = _W[:, 1] - _W[:, 0]
    keep = np.argsort(-np.abs(w))[:2]
    x = [None] * 6
    for j in keep:
        x[j] = float(3.0 * np.sign(w[j]))
    verdict = trained.loaded(trained.registry.latest_id(), "good").verdict
    assert verdict["low"]["max_tier"] == "act"
    for stakes in ("low", "medium"):
        d = trained.generic_decide(DecideRequest(state={"x": x, "feature_version": "toy-v1"}, stakes=stakes,
                                                 heads=["good"]))
        r = d["heads"]["good"]
        assert r["answer"]["calibrated"] is True and r["decision"]["p"] > 0.99
        assert r["decision"]["tier"] != "act" and r["decision"]["max_tier"] == "confirm"
        if verdict[stakes]["max_tier"] == "act":     # otherwise the holdout verdict already capped it
            assert any("capped at confirm" in m and "lacks 4 input" in m for m in d["warnings"])
    full = trained.generic_decide(DecideRequest(state=POS, heads=["good"], stakes="low"))
    assert full["heads"]["good"]["decision"]["max_tier"] == "act"


def test_f45_non_finite_inputs_count_as_missing(trained):
    ens = trained.loaded(trained.registry.latest_id(), "good").ensemble
    assert ens.missing_inputs(np.array([[np.inf, 0, 0, 0, 0, 0]])) == ["f0"]
    assert ens.missing_inputs(np.array([[0, -np.inf, 0, 0, 0, np.nan]])) == ["f1", "f5"]
    warnings: list[str] = []
    state = {"x": [1e999, 0, 0, 0, 0, 0], "feature_version": "toy-v1"}
    trained.probabilities(trained.registry.latest_id(), trained.spec("good", trained.registry.latest_id()),
                          [state], warnings)
    assert any("lacks 1 input" in m for m in warnings)


# ============================================================================ F25 / F26: precomputed states

def test_f25_feedback_with_x_must_match_the_current_feature_schema(trained):
    def post(**kw):
        return trained.feedback(FeedbackRequest(head="good", label="true", source="crm", **kw))

    x = [0.1] * 6
    with pytest.raises(ContractError, match="feature_version is required"):
        post(x=x)
    with pytest.raises(ContractError) as e:
        post(x=x, feature_version="toy-v0")
    assert e.value.kind == "feature_version_mismatch"
    with pytest.raises(ContractError) as e:
        post(x=x[:3], feature_version="toy-v1")
    assert e.value.kind == "bad_feature_vector"
    with pytest.raises(ContractError) as e:
        post(x=x, feature_version="toy-v1", feature_names=list("abcdef"))
    assert e.value.kind == "feature_names_mismatch"
    assert post(x=x, feature_version="toy-v1", feature_names=NAMES)["stored"] is True


class XYProvider(ToyProvider):
    feature_version = "xy-v1"

    def names(self, view):
        return ["x", "y"]

    def transform(self, view, states):
        rows = []
        for s in states:
            if not isinstance(s, dict) or not all(isinstance(s.get(k), (int, float)) for k in ("x", "y")):
                raise ContractError("xy heads need state {x: number, y: number}")
            rows.append([float(s["x"]), float(s["y"])])
        return np.asarray(rows, np.float32).reshape(len(rows), 2)

    def groups(self, view):
        return {"x": "x", "y": "y"}


class XYPack(ToyPack):
    name = "xy"

    def heads(self, project):
        return {"inside": HeadSpec("inside", "noul", view="v")}

    def provider(self, project, config=None):
        return XYProvider()

    def datasets(self, project, provider, heads, feedback=None, feedback_weight=5.0):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(1500, 2)).astype(np.float32)
        y = (rng.random(1500) < 1 / (1 + np.exp(-(2 * X[:, 0] - X[:, 1])))).astype(int)
        return {"inside": Dataset(X=X, y=y, feature_names=["x", "y"], feature_version="xy-v1",
                                  source=["synthetic"] * 1500)}


def test_f26_a_feature_column_called_x_is_a_raw_state(tmp_path):
    from open_rlcd.core.service import is_precomputed

    assert not is_precomputed({"x": 1.2, "y": -0.3}) and is_precomputed({"x": [1.0], "feature_version": "v"})
    svc = make_service(tmp_path, pack=XYPack())
    train(svc)
    q = {"inside": {"type": "noul"}}
    out = svc.systemone(SystemOneRequest(state={"x": 1.2, "y": -0.3}, questions=q))
    assert out["answers"]["inside"]["calibrated"] is True
    assert svc.generic_decide(DecideRequest(state={"x": 1.2, "y": -0.3}))["heads"]["inside"]["reasons"]
    # a vector `x` without feature_version still gets the precomputed-features hint
    with pytest.raises(ContractError, match="feature_version is required with state.x"):
        svc.systemone(SystemOneRequest(state={"x": [1.2, -0.3]}, questions=q))


# ============================================================================ F27: errors at the API boundary

def test_f27_internal_errors_do_not_leak_paths_or_exception_text(trained, monkeypatch):
    c = TestClient(create_app(trained), raise_server_exceptions=False)
    monkeypatch.setattr(trained, "models_view", lambda: (_ for _ in ()).throw(
        RuntimeError("/Users/someone/secret/home/registry.json: boom")))
    r = c.get("/v1/models")
    assert r.status_code == 500 and r.json()["kind"] == "internal"
    assert "RuntimeError" in r.json()["error"] and "/Users" not in r.text and "boom" not in r.text
    monkeypatch.setattr(trained, "models_view", lambda: (_ for _ in ()).throw(
        ConnectorError("postgresql://rlcd:hunter2@db.internal/docs: refused")))
    r = c.get("/v1/models")
    assert r.status_code == 503 and r.json()["kind"] == "connector_unavailable"
    assert "hunter2" not in r.text and "db.internal" not in r.text


# ============================================================================ F28: stakes

def test_f28_unknown_stakes_are_422_listing_the_levels(trained, tmp_path):
    c = TestClient(create_app(trained), raise_server_exceptions=False)
    r = c.get("/v1/bernoulli", params={"head": "good", "stakes": "extreme"})
    assert r.status_code == 422 and r.json()["kind"] == "validation"
    assert r.json()["stakes"] == ["low", "medium", "high"]
    r = c.post("/v1/decide", json={"state": POS, "stakes": "extreme"})
    assert r.status_code == 422 and r.json()["stakes"] == ["low", "medium", "high"]
    spec = HeadSpec("good", "noul", stakes="hihg")
    with pytest.raises(ContractError, match="unknown stakes level 'hihg'"):
        PL.decide(spec, np.array([0.1, 0.9]), PL.PolicyConfig())
    # a custom level the policy defines is accepted by /v1/decide
    vip = make_service(tmp_path, data={"policy": {"stakes": {"vip": {"act": 0.99, "confirm": 0.9,
                                                                         "min_accuracy": 0.95}}}})
    train(vip, heads=["good"])
    d = vip.decide(DecideRequest(state=POS, heads=["good"], stakes="vip"))
    assert d["heads"]["good"]["decision"]["stakes"] == "vip"


class TypoPack(ToyPack):
    name = "toy"

    def heads(self, project):
        specs = super().heads(project)
        g = specs["good"]
        specs["good"] = HeadSpec(g.name, g.type, view=g.view, decision=g.decision, conformal=g.conformal,
                                 stakes="hihg")
        return specs


def test_f28_a_stakes_typo_fails_that_head_at_train_time(tmp_path):
    svc = make_service(tmp_path, pack=TypoPack(n=1500))
    job = svc.trainer.start(TrainRequest(heads=["good", "color"]), wait=True)
    assert job["status"] == "partial"
    assert "stakes 'hihg'" in job["heads"]["good"]["error"] and job["heads"]["color"]["status"] == "trained"


# ============================================================================ F35: train.feedback_weight

def test_f35_feedback_weight_comes_from_rlcd_yaml_unless_the_request_sets_it(tmp_path):
    svc = make_service(tmp_path, pack=ToyPack(n=1500), data={"train": {"feedback_weight": 50}})
    for _ in range(3):
        svc.feedback(FeedbackRequest(head="good", state=POS, label="true", source="crm"))
    assert svc.trainer.feedback_weight(TrainRequest()) == 50.0
    assert svc.trainer.feedback_weight(TrainRequest(feedback_weight=2)) == 2.0
    job = train(svc, heads=["good"])
    assert any("weight x 50" in n for n in job["heads"]["good"]["notes"])


# ============================================================================ F43: bounded feedback

def test_f43_feedback_size_is_capped_and_counts_are_incremental(tmp_path, monkeypatch):
    with pytest.raises(ValidationError, match="at most"):
        FeedbackRequest(head="good", label="true", source="x", state={"pad": "A" * (3 << 20)})
    svc = make_service(tmp_path, pack=ToyPack(n=300))
    c = TestClient(create_app(svc), raise_server_exceptions=False)
    monkeypatch.setenv("RLCD_MAX_BODY_BYTES", "2000")
    r = c.post("/v1/feedback", json={"head": "good", "label": True, "source": "x", "state": {"pad": "A" * 4000}})
    assert r.status_code == 413 and r.json()["kind"] == "too_large"
    monkeypatch.delenv("RLCD_MAX_BODY_BYTES")
    log = FeedbackLog(tmp_path / "logs")
    assert log.feedback_counts() == {}
    log.add_feedback({"head": "good", "label": "true"})
    assert log.feedback_counts() == {"good": 1}
    with open(log.feedback_log.path, "a", encoding="utf-8") as f:   # appended by someone else
        f.write(json.dumps({"head": "color"}) + "\n" + '{"head": "tor')        # ... and a torn last line
    assert log.feedback_counts() == {"good": 1, "color": 1}
    log.feedback_log.path.write_text(json.dumps({"head": "level"}) + "\n")     # replaced: counted again
    assert log.feedback_counts() == {"level": 1}


# ============================================================================ F44: views

def test_f44_views_list_every_head_with_trained_and_n_train(tmp_path):
    svc = make_service(tmp_path, pack=ToyPack(n=1500))
    untrained = {h["name"]: h for h in svc.heads_view(None)["heads"]}
    assert all("n_train" in h and h["n_train"] is None and "positive" in h for h in untrained.values())
    train(svc, heads=["good"])
    heads = {h["name"]: h for h in svc.heads_view(None)["heads"]}
    assert isinstance(heads["good"]["n_train"], int) and heads["color"]["n_train"] is None
    v = svc.models_view()["data"][1]
    assert v["heads"]["good"]["trained"] is True and v["heads"]["good"]["n_train"] == heads["good"]["n_train"]
    assert v["heads"]["good"]["model_version"] == v["id"]
    assert v["heads"]["color"] == {"trained": False, "n_train": None, "trained_at": None, "model_version": None,
                                   "metrics": None, "verdict": None, "model": None}
