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
import yaml

from open_rlcd.cli import main
from open_rlcd.core import config_check as CC
from open_rlcd.core.config import Project
from open_rlcd.core.contract import ContractError, DecideRequest, SystemOneRequest, TrainRequest
from open_rlcd.core.service import Service
from open_rlcd.packs.trading import DEFAULTS, TradingPack

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


# ----------------------------------------------------------------------------- rlcd.yaml (config check, heads)

def offline() -> TradingPack:
    """A pack whose sidecar must not be called: config check and head building need no network."""
    return TradingPack(transport=httpx.MockTransport(lambda r: pytest.fail(f"network call to {r.url}")))


def findings(tmp_path, pack_config: dict | None = None, **data) -> list[dict]:
    return CC.check_project(Project(root=tmp_path, data={"pack": "trading", "pack_config": pack_config or {}, **data}),
                            offline())


def at(found: list[dict], path: str, level: str = "error") -> str:
    hits = [f for f in found if f["path"] == path]
    assert len(hits) == 1 and hits[0]["level"] == level, (path, found)
    return hits[0]["message"]


def test_init_prints_the_trading_steps(tmp_path, capsys):
    assert main(["init", str(tmp_path / "t"), "--pack", "trading"]) == 0
    out = capsys.readouterr().out
    assert "quant sidecar" in out and "rlcd config check" in out and "data.csv" not in out


def test_config_check_knows_the_trading_keys(tmp_path):
    assert findings(tmp_path, {**DEFAULTS, "pips": {"XAUUSD": [0.1, 1]}}) == []
    assert "Did you mean 'intervals'?" in at(findings(tmp_path, {"interval": ["15m"]}), "pack_config.interval")
    assert "whole number" in at(findings(tmp_path, {"horizon": "48h"}), "pack_config.horizon")
    assert "> 0" in at(findings(tmp_path, {"sl_pips": 0}), "pack_config.sl_pips")
    assert "must be a list" in at(findings(tmp_path, {"symbols": "XAUUSD"}), "pack_config.symbols")
    assert "must be a list" in at(findings(tmp_path, {"pips": {"XAUUSD": 0.1}}), "pack_config.pips.XAUUSD")
    # what only the pack can say (check_config)
    msg = at(findings(tmp_path, {"intervals": ["1h", "15min"]}), "pack_config.intervals")
    assert "'15min' is not an interval" in msg and "Did you mean '15m'?" in msg
    assert "is empty" in at(findings(tmp_path, {"intervals": []}), "pack_config.intervals")
    assert "is empty" in at(findings(tmp_path, {"symbols": []}), "pack_config.symbols")
    assert "http://" in at(findings(tmp_path, {"quant_url": "127.0.0.1:8090"}), "pack_config.quant_url")
    assert "numbers > 0" in at(findings(tmp_path, {"pips": {"XAUUSD": [0]}}), "pack_config.pips.XAUUSD")
    assert "never used" in at(findings(tmp_path, {"pips": {"GBPUSD": [0.1]}}), "pack_config.pips.GBPUSD", "warn")


def test_train_and_serve_refuse_a_trading_file_config_check_rejects(tmp_path, capsys):
    root = tmp_path / "p"
    root.mkdir()
    (root / "rlcd.yaml").write_text(yaml.safe_dump({"pack": "trading", "pack_config": {"interval": ["15m"]}}))
    assert main(["--project", str(root), "config", "check"]) == 1
    assert main(["--project", str(root), "train"]) == 2            # before any sidecar call
    assert main(["--project", str(root), "serve"]) == 2
    assert "pack_config.interval: unknown key" in capsys.readouterr().err


def test_an_empty_trading_key_is_its_default(tmp_path):
    empty = dict.fromkeys(DEFAULTS)
    project = Project(root=tmp_path, data={"pack": "trading", "pack_config": empty})
    assert findings(tmp_path, empty) == []
    pack = offline()
    assert pack._cfg(project) == DEFAULTS
    assert list(pack.heads(project)) == ["setup:15m"]
    provider = pack.provider(project)
    assert provider.client.url == DEFAULTS["quant_url"] and provider.cfg["params"]["horizon"] == 48


def test_heads_overrides_change_the_fields_every_pack_has_but_not_the_question(tmp_path):
    pack = offline()
    over = {"setup:15m": {"stakes": "high", "conformal": 0.1, "model": "logreg", "description": "d",
                          "params": {"C": 0.5}, "type": "choice", "view": "15m"}}
    spec = pack.heads(Project(root=tmp_path, data={"pack": "trading", "heads": over}))["setup:15m"]
    assert (spec.stakes, spec.conformal, spec.model, spec.description, spec.params) == ("high", 0.1, "logreg", "d",
                                                                                        {"C": 0.5})
    assert spec.classes == ("buy", "sell", "hold") and spec.view == "15m"
    assert findings(tmp_path, heads=over) == []
    for field, value in (("classes", ["long", "short", "flat"]), ("type", "score"), ("view", "1h")):
        heads = {"setup:15m": {field: value}}
        assert "cannot be changed" in at(findings(tmp_path, heads=heads), f"heads.setup:15m.{field}")
        with pytest.raises(ValueError, match="cannot be changed"):
            pack.heads(Project(root=tmp_path, data={"pack": "trading", "heads": heads}))
    msg = at(findings(tmp_path, heads={"setup:1h": {"stakes": "high"}}), "heads.setup:1h")
    assert "not a head of the trading pack" in msg and "Did you mean 'setup:15m'?" in msg
    assert findings(tmp_path, heads={"setup:1h": {"enabled": False}}) == []


@pytest.fixture(scope="module")
def two_heads(tmp_path_factory):
    """setup:15m and setup:1h trained once; each test rebuilds the service on that home with another rlcd.yaml."""
    root, sidecar = tmp_path_factory.mktemp("two"), FakeSidecar(rows=600)
    svc = make(root, sidecar, intervals=["15m", "1h"])
    svc.project.data["heads"]["setup:1h"] = {"model": "logreg"}
    job = svc.trainer.start(TrainRequest(), wait=True)
    assert job["status"] == "succeeded" and set(job["heads"]) == {"setup:15m", "setup:1h"}, job
    return root, sidecar, job["model"]


def test_a_switched_off_trading_head_is_not_trained_or_served(two_heads):
    """`heads.<head>.enabled: false` and an interval removed from pack_config.intervals switch a head off: not
    trained, and not served although the model has it (`rlcd config show` lists it as switched off). Listed again,
    it is served again without a retrain. (The pack's own heads() used to ignore `enabled`.)"""
    root, sidecar, version = two_heads

    def service(intervals: list[str], heads: dict | None = None) -> Service:
        data = {"pack": "trading", "heads": heads or {},
                "pack_config": {"quant_url": "http://127.0.0.1:8090", "symbols": ["XAUUSD", "EURUSD"],
                                "intervals": intervals, "horizon": 8}}
        return Service(Project(root=root, data=data), home=root / "var", pack=TradingPack(sidecar.transport()))

    for svc in (service(["15m", "1h"], {"setup:1h": {"enabled": False}}), service(["15m"])):
        assert svc.trainer.plan(TrainRequest()) == ["setup:15m"]
        assert set(svc.specs(version)) == {"setup:15m"} and svc.switched_off(version) == {"setup:1h"}
        with pytest.raises(ContractError) as e:
            svc.decide(DecideRequest(state={"symbol": "XAUUSD"}, heads=["setup:1h"]))
        assert e.value.kind == "unknown_head"
        assert svc.decide(DecideRequest(state={"symbol": "XAUUSD"}, heads=["setup:15m"]))["model"] == version
    view = CC.effective(service(["15m", "1h"], {"setup:1h": {"enabled": False}}).project, offline())
    assert view["heads"]["setup:1h"] == {"enabled": False}
    again = service(["15m", "1h"])
    assert set(again.specs(version)) == {"setup:15m", "setup:1h"}
    assert again.decide(DecideRequest(state={"symbol": "XAUUSD"}, heads=["setup:1h"]))["model"] == version
