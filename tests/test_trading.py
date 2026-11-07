"""Trading adapter pack against a fake quant sidecar (httpx.MockTransport): no network, no trading logic.

The fake sidecar serves generic features x0..x5 and labels drawn from a known multinomial logistic model of
two of them (signal) or independent of them (noise), in both wire formats (JSON and npz).
"""
from __future__ import annotations

import io
import json
import zlib
from datetime import UTC, datetime

import httpx
import numpy as np
import pytest

from open_rlcd.core.config import Project
from open_rlcd.core.contract import ContractError, DecideRequest, SystemOneRequest, TrainRequest
from open_rlcd.core.service import Service
from open_rlcd.packs.trading import TradingPack

NAMES = [f"x{i}" for i in range(6)]
T0 = int(datetime(2025, 1, 1, tzinfo=UTC).timestamp())


def iso(ts: int) -> str:
    return datetime.fromtimestamp(int(ts), UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


class FakeSidecar:
    def __init__(self, signal: float = 2.0, rows: int = 2400, npz: bool = False, up: bool = True):
        self.signal, self.rows, self.npz, self.up = signal, rows, npz, up
        self.calls: list[str] = []

    def build(self, symbol: str, interval: str, horizon: int):
        rng = np.random.default_rng(zlib.crc32(f"{symbol}|{interval}".encode()))
        n = self.rows
        X = rng.normal(size=(n, len(NAMES))).astype(np.float32)
        s = self.signal * (X[:, 0] + 0.5 * X[:, 1])
        logits = np.stack([s, -s, np.zeros(n)], axis=1)
        P = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
        u = rng.random(n)
        cls = np.where(u < P[:, 0], 0, np.where(u < P[:, 0] + P[:, 1], 1, 2))   # 0 buy, 1 sell, 2 hold
        labelled = np.arange(n) < n - horizon
        times = T0 + np.arange(n) * 900
        meta = {"symbol": symbol, "interval": interval, "pip": 0.1, "params": {"sl_pips": 20, "tp_pips": 50,
                "horizon": horizon, "spread_pips": 1.0}, "feature_version": "fake-v1", "feature_names": NAMES}
        return X, cls, labelled, times, meta

    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self.handle)

    def handle(self, request: httpx.Request) -> httpx.Response:
        q = dict(request.url.params)
        self.calls.append(request.url.path)
        if not self.up:
            raise httpx.ConnectError("refused", request=request)
        symbol, interval = q.get("symbol", ""), q.get("interval", "15m")
        if symbol == "NOPE":
            return httpx.Response(400, json={"error": "unknown symbol"})
        X, cls, labelled, times, meta = self.build(symbol, interval, int(q.get("horizon", 48)))
        if request.url.path == "/features":
            return httpx.Response(200, json={"symbol": symbol, "interval": interval, "time": iso(times[-1]),
                                             "feature_version": "fake-v1", "feature_names": NAMES,
                                             "x": [float(v) for v in X[-1]]})
        if request.url.path == "/dataset":
            if self.npz:
                wire = np.where(labelled, np.select([cls == 0, cls == 1], [1, 2], 0), -1).astype(np.int8)
                buf = io.BytesIO()
                np.savez(buf, X=X, times=times.astype(np.int64), best=wire, meta=np.array(json.dumps(meta)))
                return httpx.Response(200, content=buf.getvalue(), headers={"content-type": "application/octet-stream"})
            best = [("buy", "sell", "hold")[c] if ok else None for c, ok in zip(cls, labelled, strict=True)]
            return httpx.Response(200, json={**meta, "times": [iso(t) for t in times],
                                             "X": [[float(v) for v in r] for r in X], "y": {"best": best}})
        return httpx.Response(404, json={"detail": "Not Found"})


def make(tmp_path, sidecar: FakeSidecar, **cfg) -> Service:
    project = Project(root=tmp_path, data={"pack": "trading", "pack_config": {
        "quant_url": "http://127.0.0.1:8090", "symbols": ["XAUUSD", "EURUSD"], "intervals": ["15m"], "horizon": 8,
        **cfg}, "heads": {"setup:15m": {"model": "logreg"}}})
    return Service(project, home=tmp_path / "var", pack=TradingPack(transport=sidecar.transport()))


@pytest.fixture(scope="module")
def planted(tmp_path_factory):
    svc = make(tmp_path_factory.mktemp("planted"), FakeSidecar(signal=2.0))
    job = svc.trainer.start(TrainRequest(), wait=True)
    assert job["status"] == "succeeded", job
    return svc


def test_planted_signal_is_calibrated_and_skilled(planted):
    rep = planted.calibration_view("setup:15m", None)
    assert rep["skill"]["has_skill"] is True
    assert rep["ece"] < 0.06
    assert rep["brier_skill_score"] > 0.1
    assert rep["data"]["ordering"] == "time" and rep["data"]["by_source"] == {"ground_truth": rep["data"]["rows"]}


def test_serve_fetches_features_from_the_sidecar(planted):
    r = planted.systemone(SystemOneRequest(state={"symbol": "XAUUSD"}, questions={"s": {"type": "choice",
                                                                                       "head": "setup:15m"}}))
    a = r["answers"]["s"]
    assert a["calibrated"] is True and set(a["probabilities"]) == {"buy", "sell", "hold"}


def test_decision_uses_hold_as_the_abstain_action(planted):
    d = planted.decide(DecideRequest(state={"symbol": "EURUSD"}, heads=["setup:15m"]))
    assert d["action"] in ("act", "confirm", "hold")
    assert planted.policy(d["model"]).abstain == "hold"


def test_noise_shows_no_skill_and_holds(tmp_path):
    svc = make(tmp_path, FakeSidecar(signal=0.0))
    job = svc.trainer.start(TrainRequest(), wait=True)
    assert job["status"] == "succeeded", job
    assert job["heads"]["setup:15m"]["metrics"]["has_skill"] is False
    d = svc.decide(DecideRequest(state={"symbol": "XAUUSD"}, heads=["setup:15m"]))
    assert d["action"] == "hold"
    assert any("no skill" in w for w in d["warnings"])


def test_npz_and_json_exports_train_the_same_rows(tmp_path):
    a = make(tmp_path / "a", FakeSidecar(rows=600))
    b = make(tmp_path / "b", FakeSidecar(rows=600, npz=True))
    pa, pb = a.pack.provider(a.project), b.pack.provider(b.project)
    da = a.pack.datasets(a.project, pa, ["setup:15m"])["setup:15m"]
    db = b.pack.datasets(b.project, pb, ["setup:15m"])["setup:15m"]
    np.testing.assert_array_equal(da.y, db.y)
    np.testing.assert_array_equal(da.t, db.t)
    np.testing.assert_allclose(da.X, db.X, rtol=1e-6)
    assert (da.t_end >= da.t).all()


def test_label_window_follows_the_candles(tmp_path):
    svc = make(tmp_path, FakeSidecar(rows=200))
    ds = svc.pack.datasets(svc.project, svc.pack.provider(svc.project), ["setup:15m"])["setup:15m"]
    first = ds.segment == ds.segment[0]
    np.testing.assert_array_equal(ds.t_end[first], ds.t[first] + 8 * 900)


def test_remote_sidecar_refused_unless_allowed(tmp_path, monkeypatch):
    svc = make(tmp_path, FakeSidecar(), quant_url="http://quant.example.com:8090")
    with pytest.raises(ContractError, match="not local|allow"):
        svc.pack.provider(svc.project)
    monkeypatch.setenv("RLCD_ALLOW_REMOTE", "1")
    assert svc.pack.provider(svc.project) is not None


def test_sidecar_down_is_503(tmp_path):
    svc = make(tmp_path, FakeSidecar(up=False))
    job = svc.trainer.start(TrainRequest(), wait=True)
    assert job["status"] == "failed" and "unreachable" in job["error"]


def test_unknown_symbol_is_skipped_not_fatal(tmp_path):
    svc = make(tmp_path, FakeSidecar(rows=800), symbols=["XAUUSD", "NOPE", "EURUSD"])
    ds = svc.pack.datasets(svc.project, svc.pack.provider(svc.project), ["setup:15m"])["setup:15m"]
    assert any("NOPE" in n for n in ds.notes)
    assert len(set(ds.segment.tolist())) == 2
