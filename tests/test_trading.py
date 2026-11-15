"""Trading adapter pack against a fake quant sidecar (httpx.MockTransport): no network, no trading logic.

The fake sidecar serves generic features x0..x5 and labels drawn from a known multinomial logistic model of
two of them (signal) or independent of them (noise), in both wire formats (JSON and npz).
"""
from __future__ import annotations

import base64
import io
import itertools
import json
import logging
import traceback
import zlib
from datetime import UTC, datetime
from urllib.parse import quote

import httpx
import numpy as np
import pytest
import yaml
from fastapi.testclient import TestClient

from open_rlcd.cli import main
from open_rlcd.connectors.security import is_loopback
from open_rlcd.core import config_check as CC
from open_rlcd.core.config import Project
from open_rlcd.core.contract import ContractError, DecideRequest, SystemOneRequest, TrainRequest
from open_rlcd.core.service import Service
from open_rlcd.packs.base import load_pack
from open_rlcd.packs.trading import DEFAULTS, INVALID_URL, NOT_A_URL, TradingPack
from open_rlcd.serve.api import create_app

NAMES = [f"x{i}" for i in range(6)]
T0 = int(datetime(2025, 1, 1, tzinfo=UTC).timestamp())


def iso(ts: int) -> str:
    return datetime.fromtimestamp(int(ts), UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def basic(user: str, password: str) -> str:
    """The Authorization header of HTTP basic auth."""
    return "Basic " + base64.b64encode(f"{user}:{password}".encode()).decode()


def echo(request: httpx.Request) -> str:
    """What an error that echoes the request shows: its URL, and its Authorization header as sent and decoded."""
    header = request.headers.get("authorization")
    decoded = base64.b64decode(header.removeprefix("Basic ")).decode() if header else None
    return f"{request.url} ({header}; {decoded})"


class FakeSidecar:
    def __init__(self, signal: float = 2.0, rows: int = 2400, npz: bool = False, up: bool = True,
                 auth: tuple[str, str] | None = None):
        self.signal, self.rows, self.npz, self.up = signal, rows, npz, up
        self.auth = auth                                  # (user, password) it answers 401 without
        self.fail: int | None = None                      # a status (4xx, 5xx) it answers every request with
        self.rejected: object = {"error": "unknown symbol"}   # the JSON of its 400 for symbol NOPE
        self.raw = False                                  # `rejected` is that answer's text, sent as written
        self.calls: list[tuple[str, dict]] = []      # (path, query params) of every request

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
        self.calls.append((request.url.path, q))
        if not self.up:                 # an error that echoes the request
            raise httpx.ConnectError(f"refused at {echo(request)}", request=request)
        if self.auth is not None and request.headers.get("authorization") != basic(*self.auth):
            return httpx.Response(401, json={"error": "unauthorized"})
        if self.fail:                   # an error page that echoes the request
            return httpx.Response(self.fail, text=f"boom at {echo(request)}")
        symbol, interval = q.get("symbol", ""), q.get("interval", "15m")
        if symbol == "NOPE":
            body = self.rejected if self.raw else json.dumps(self.rejected)
            return httpx.Response(400, content=body, headers={"content-type": "application/json"})
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


def hidden(text: str) -> bool:
    """No part of the test credentials (S3cret...Passw0rd, whatever stands between the two halves, in a password, a
    user name or a token) is in `text`, in any case: a host is lowercased."""
    text = text.lower()
    return "s3cret" not in text and "passw0rd" not in text


# quant_urls with user:password@ that the client accepts, and the basic auth the sidecar then gets: the password
# percent-encoded, and with an unencoded `@` or space, which httpx reads as part of it (scrub's user:password@
# pattern alone does not cover those: it stops at the `@` or the space); a user name with an unencoded `@`, and a
# token alone (a user name and no password) with a space; and two whose scrubbing depends on its order: a password
# that starts with the user name, and a one-letter user name that is also a letter of its own basic-auth header
# (scrubbed first, the short one splits the longer one and leaves the rest of it in the text)
ACCEPTED = [("http://quant:S3cret%40Passw0rd@127.0.0.1:8090", ("quant", "S3cret@Passw0rd")),
            ("http://quant:S3cret@Passw0rd@127.0.0.1:8090", ("quant", "S3cret@Passw0rd")),
            ("http://quant:S3cret Passw0rd@127.0.0.1:8090", ("quant", "S3cret Passw0rd")),
            ("http://ann@S3cret:Passw0rd@127.0.0.1:8090", ("ann@S3cret", "Passw0rd")),
            ("http://S3cret Passw0rd@127.0.0.1:8090", ("S3cret Passw0rd", "")),
            ("http://quant:quantS3cretPassw0rd@127.0.0.1:8090", ("quant", "quantS3cretPassw0rd")),
            ("http://z:S3cretPassw0rd@127.0.0.1:8090", ("z", "S3cretPassw0rd"))]


@pytest.mark.parametrize(("url", "auth"), ACCEPTED)
@pytest.mark.parametrize("failure", ["down", "500"])
def test_a_failing_sidecar_names_no_url_host_or_password(tmp_path, caplog, failure, url, auth):
    """quant_url carries user:password@. The sidecar gets them as basic auth, percent-decoded (it answers 401
    otherwise), and no log record at any level holds the user name or the password in any form: httpx's own line for
    each request (INFO) names the URL without credentials. (It used to name it with user:password@ in clear.) When
    the sidecar is down, or answers 500, with an error that echoes the request (its URL, and its Authorization header
    as sent and decoded), the 503 (through the API and through Service) and the failed train job name the sidecar,
    the request and the exception type or status, never the URL, its host:port or the password (docs/contract.md);
    the server log names the URL as httpx read it, with `***@` in place of its user info, and the error with the user
    name and password scrubbed."""
    sidecar = FakeSidecar(rows=600, auth=auth)
    svc = make(tmp_path, sidecar, quant_url=url)
    client = TestClient(create_app(svc))
    with caplog.at_level(logging.DEBUG):      # the root logger: httpx logs each request at INFO, httpcore at DEBUG
        job = svc.trainer.start(TrainRequest(), wait=True)
        assert job["status"] == "succeeded", job
        if failure == "down":
            sidecar.up = False
        else:
            sidecar.fail = 500
        answers = [client.post("/v1/decide", json={"state": {"symbol": "XAUUSD"}, "heads": ["setup:15m"]}),
                   client.post("/v1/systemone", json={"state": {"symbol": "XAUUSD"},
                                                      "questions": {"s": {"type": "choice", "head": "setup:15m"}}})]
        with pytest.raises(ContractError) as e:
            svc.decide(DecideRequest(state={"symbol": "XAUUSD"}, heads=["setup:15m"]))
        job = svc.trainer.start(TrainRequest(), wait=True)
    said = {"down": "quant sidecar unreachable (GET {}): ConnectError; details are in the server log",
            "500": "quant sidecar GET {} failed with 500; details are in the server log"}[failure]
    for r in answers:
        assert r.status_code == 503 and r.json()["kind"] == "connector_unavailable"
        assert r.json()["error"] == said.format("/features")
    assert e.value.status == 503 and e.value.kind == "connector_unavailable"
    assert e.value.message == said.format("/features")
    assert job["status"] == "failed" and said.format("/dataset") in job["error"]
    for text in [*(r.text for r in answers), json.dumps(e.value.body()), json.dumps(job, default=str)]:
        assert hidden(text) and not any(s in text for s in ("http://", "127.0.0.1", ":8090")), text
    logged = [r.getMessage() for r in caplog.records if r.name == "open_rlcd.packs.trading"]
    echoed = f" (Basic ***; ***:{'***' if auth[1] else ''})"      # the credentials it echoes, scrubbed
    for path in ("/features", "/dataset"):
        line = {"down": f"quant sidecar unreachable: GET http://***@127.0.0.1:8090{path}: ConnectError: refused at ",
                "500": f"quant sidecar GET http://***@127.0.0.1:8090{path} failed with 500: boom at "}[failure]
        assert any(m.startswith(f"{line}http://127.0.0.1:8090{path}?") and m.endswith(echoed) for m in logged), logged
    requests = [r.getMessage() for r in caplog.records if r.name == "httpx"]
    assert any("GET http://127.0.0.1:8090/dataset?" in m for m in requests), requests
    assert hidden(caplog.text) and basic(*auth)[6:] not in caplog.text


NO_HOST = "cannot tell which host 'http://:8090' points at"
# (quant_url, remote hosts allowed, what its 422 says): user names and passwords with a character they should have
# percent-encoded. httpx reads quant_url first: what it cannot read ("Invalid port: 'S3cret'"), or reads with an `@`
# outside its user info (in the path, query or fragment), is INVALID_URL, with remote hosts allowed or not. The host
# check then reads what httpx read, without the user info. (check_host used to read quant_url as written: it took
# what follows an `@` in the password as the host and quoted it lowercased, 'passw0rd', and quoted the URL with
# repr(), which escapes a quote in its own way.)
REFUSED = [("http://quant:S3cretPassw0rd@:8090", False, NO_HOST),
           ("http://quant:S3cret@Passw0rd@:8090", False, NO_HOST),
           ("http://quant:S3cret Passw0rd@:8090", False, NO_HOST),
           ("http://quant:S3cret\\ Passw0rd@:8090", False, NO_HOST),
           ("http://qu\"ant:S3cret' Passw0rd@:8090", False, NO_HOST),
           ("http://quant:S3cret@Passw0rd@", False, "cannot tell which host 'http://' points at"),
           ("http://quant:S3cret@Passw0rd@quant.example.com:8090", False, "refusing non-loopback host "
                                                                          "'quant.example.com'"),
           ("http://quant:S3cret@Passw0rd@xn--zz.com", False, "refusing non-loopback host 'xn--zz.com'"),  # not IDNA
           ("http://quant:S3cret@Passw0rd/x@127.0.0.1:8090", False, INVALID_URL),   # httpx: the host 'passw0rd'
           ("http://quant:S3cret@Passw0rd?x@127.0.0.1:8090", False, INVALID_URL),
           ("http://quant:S3cret@Passw0rd#x@127.0.0.1:8090", False, INVALID_URL),
           ("http://:S3cret/Passw0rd@127.0.0.1:8090", False, INVALID_URL),
           ("quant:S3cret Passw0rd@127.0.0.1:8090", False, INVALID_URL),
           ("http://quant:S3cret/Passw0rd@127.0.0.1:8090", True, INVALID_URL),     # httpx: "Invalid port: 'S3cret'"
           ("http://quant:S3cret?Passw0rd@127.0.0.1:8090", True, INVALID_URL),
           ("http://quant:S3cret#Passw0rd@127.0.0.1:8090", True, INVALID_URL),
           ("http://quant:8090/S3cretPassw0rd@127.0.0.1:8090", True, INVALID_URL),  # httpx: port 8090, the rest a path
           ("http://quant:8090?S3cretPassw0rd@127.0.0.1:8090", True, INVALID_URL),  # ... a query
           ("http://quant:8090#S3cretPassw0rd@127.0.0.1:8090", True, INVALID_URL),  # ... a fragment
           ("quant:S3cret Passw0rd@127.0.0.1:8090", True, INVALID_URL),            # no http://: the `@` in the path
           ("http://quant:S3cretPassw0rd@[::1%ä]:8090", False, INVALID_URL),     # a zone id httpx cannot send
           ("http://quant:S3cretPassw0rd@[fe80::1%25ä]:8090", True, INVALID_URL)]


@pytest.mark.parametrize(("url", "remote", "said"), REFUSED)
def test_a_refused_quant_url_is_quoted_without_its_password(tmp_path, monkeypatch, url, remote, said):
    """However malformed its password, a quant_url the pack refuses (a URL httpx cannot read or reads with an `@`
    outside its user info, no host, a host the check refuses) is a 422 that quotes no part of the password, and so is
    the failed train job, its trace included. (The 422 used to show part of the password, as the host check read
    quant_url as written, and a train with remote hosts allowed failed with httpx's own message: "InvalidURL: Invalid
    port: 'S3cret'". An IPv6 zone id that is not ASCII, which httpx reads but cannot send, raised UnicodeEncodeError:
    a 500, or a failed train.)"""
    if remote:
        monkeypatch.setenv("RLCD_ALLOW_REMOTE", "1")
    svc = make(tmp_path, FakeSidecar(), quant_url=url)
    with pytest.raises(ContractError) as e:
        svc.pack.provider(svc.project)
    assert e.value.status == 422 and e.value.kind == "validation" and said in e.value.message, e.value.message
    assert hidden(e.value.message), e.value.message
    job = svc.trainer.start(TrainRequest(), wait=True)
    assert job["status"] == "failed" and said in job["error"] and hidden(json.dumps(job, default=str)), job


def test_unknown_symbol_is_skipped_not_fatal(tmp_path):
    svc = make(tmp_path, FakeSidecar(rows=800), symbols=["XAUUSD", "NOPE", "EURUSD"])
    ds = svc.pack.datasets(svc.project, svc.pack.provider(svc.project), ["setup:15m"])["setup:15m"]
    assert any("NOPE" in n for n in ds.notes)
    assert len(set(ds.segment.tolist())) == 2


@pytest.mark.parametrize("body", [["unknown symbol"], "unknown symbol", 7, None])
def test_a_4xx_whose_json_is_not_an_object_is_a_quant_rejected_422(tmp_path, body):
    """A 4xx is the sidecar refusing the request: a 422 `quant_rejected` that quotes the `error` of a JSON object, or
    else the body as text. With JSON that is not an object (a list, a string, a number, null) the client used to
    raise AttributeError: a 500 for a decision, and a failed train instead of a skipped symbol."""
    sidecar = FakeSidecar(rows=600)
    sidecar.rejected = body
    svc = make(tmp_path, sidecar, symbols=["XAUUSD", "NOPE"])
    job = svc.trainer.start(TrainRequest(), wait=True)
    assert job["status"] == "succeeded", job
    said = "quant sidecar {}: " + json.dumps(body)
    assert f"skipped NOPE: {said.format('/dataset')}" in job["heads"]["setup:15m"]["notes"], job
    r = TestClient(create_app(svc)).post("/v1/decide", json={"state": {"symbol": "NOPE"}, "heads": ["setup:15m"]})
    answer = r.json()
    assert r.status_code == 422 and answer["kind"] == "quant_rejected" and answer["quant_status"] == 400, answer
    assert answer["error"] == said.format("/features")


@pytest.mark.parametrize(("url", "auth"), [ACCEPTED[0], ACCEPTED[-1]], ids=["quant", "z"])
def test_a_4xx_quotes_the_sidecar_without_the_credentials_it_echoes(tmp_path, url, auth):
    """The 422 for a 4xx quotes what the sidecar said with the user name and the password scrubbed from it, the
    basic-auth header included: here an error page that echoes the request, its Authorization header as sent and
    decoded. The header is scrubbed whole also when the user name is one of its letters (`z`)."""
    sidecar = FakeSidecar(rows=600, auth=auth)
    svc = make(tmp_path, sidecar, quant_url=url)
    assert svc.trainer.start(TrainRequest(), wait=True)["status"] == "succeeded"
    sidecar.fail = 403
    with pytest.raises(ContractError) as e:
        svc.decide(DecideRequest(state={"symbol": "XAUUSD"}, heads=["setup:15m"]))
    assert e.value.status == 422 and e.value.kind == "quant_rejected", e.value.body()
    assert e.value.message.startswith("quant sidecar /features: boom at http://127.0.0.1:8090/features?symbol=XAUUSD")
    assert e.value.message.endswith(" (Basic ***; ***:***)") and hidden(e.value.message), e.value.message


@pytest.mark.parametrize("password", ['S3cret"Passw0rd', "S3cret\u00e4Passw0rd", 'S3cret\u00e4"Passw0rd'],
                         ids=["quote", "not ASCII", "both"])
@pytest.mark.parametrize("ascii_only", [True, False])
def test_a_4xx_json_answer_that_echoes_the_credentials_is_quoted_without_them(tmp_path, password, ascii_only):
    """A sidecar's JSON answer writes a `"` in the password as `\\"`, and a letter that is not ASCII as `\\u00e4`
    unless it writes it as it is: the 422 that quotes such an answer has the password scrubbed in those forms too."""
    sidecar = FakeSidecar(rows=600, auth=("quant", password))
    svc = make(tmp_path, sidecar, quant_url=f"http://quant:{quote(password)}@127.0.0.1:8090")
    assert svc.trainer.start(TrainRequest(), wait=True)["status"] == "succeeded"
    sidecar.rejected = json.dumps({"detail": f"bad credentials quant:{password}"}, ensure_ascii=ascii_only)
    sidecar.raw = True
    with pytest.raises(ContractError) as e:
        svc.decide(DecideRequest(state={"symbol": "NOPE"}, heads=["setup:15m"]))
    assert e.value.status == 422 and e.value.kind == "quant_rejected", e.value.body()
    assert e.value.message == 'quant sidecar /features: {"detail": "bad credentials ***:***"}', e.value.message


def test_a_4xx_error_object_that_echoes_the_credentials_is_quoted_without_them(tmp_path):
    """A 4xx whose JSON `error` is an object is quoted as Python writes it: a string in it that holds both quote kinds
    writes a `'` in the password as `\\'`, a form the 422 scrubs too."""
    password = "S3cret'Passw0rd"
    sidecar = FakeSidecar(rows=600, auth=("quant", password))
    svc = make(tmp_path, sidecar, quant_url=f"http://quant:{quote(password)}@127.0.0.1:8090")
    assert svc.trainer.start(TrainRequest(), wait=True)["status"] == "succeeded"
    sidecar.rejected = {"error": {"detail": f'bad credentials quant:{password} for "basic"'}}
    with pytest.raises(ContractError) as e:
        svc.decide(DecideRequest(state={"symbol": "NOPE"}, heads=["setup:15m"]))
    assert e.value.status == 422 and e.value.kind == "quant_rejected", e.value.body()
    said = """quant sidecar /features: {'detail': 'bad credentials ***:*** for "basic"'}"""
    assert e.value.message == said, e.value.message


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
    # a user info without http:// (httpx reads no host, and the `@` in the path): the error names the scheme
    assert quant_url_findings(findings(tmp_path, {"quant_url": "quant@127.0.0.1:8090"}), "error") == [NOT_A_URL]
    assert NOT_A_URL.startswith("must be an http:// or https:// URL") and "check its scheme" in NOT_A_URL
    assert "numbers > 0" in at(findings(tmp_path, {"pips": {"XAUUSD": [0]}}), "pack_config.pips.XAUUSD")
    assert "never used" in at(findings(tmp_path, {"pips": {"GBPUSD": [0.1]}}), "pack_config.pips.GBPUSD", "warn")


def quant_url_findings(found: list[dict], level: str) -> list[str]:
    return [f["message"] for f in found if f["path"] == "pack_config.quant_url" and f["level"] == level]


@pytest.mark.parametrize(("url", "said"), sorted({(url, said) for url, _, said in REFUSED}))
def test_config_check_and_show_quote_no_part_of_the_quant_url_password(tmp_path, url, said):
    """`rlcd config check`, and `rlcd train` / `rlcd serve`, which print its findings on stderr before they start,
    quote no part of the quant_url password, however malformed. A quant_url the client refuses with INVALID_URL is an
    error there (NOT_A_URL), so train and serve do not start with it. quant_url is declared one URL (`Key.url`): the
    url_password warning names any user info in it, and `rlcd config show` redacts the whole user info, the user name
    included, with or without `http://`. (The config check passed a URL with a `/`, `?` or `#` in the password; the
    warning missed a password with a space or a `/`, and `rlcd config show` showed it, or the part after an `@`.)"""
    found = findings(tmp_path, {"quant_url": url})
    assert all(hidden(f["message"]) and "own check failed" not in f["message"] for f in found), found
    if said == INVALID_URL:
        assert quant_url_findings(found, "error") == [NOT_A_URL], found
    assert any("(url_password)" in m for m in quant_url_findings(found, "warn")), found
    shown = CC.effective(Project(root=tmp_path, data={"pack": "trading", "pack_config": {"quant_url": url}}),
                         offline())["pack_config"]["quant_url"]
    before, marker, after = shown.partition("[REDACTED:url_password]@")
    assert marker and before in ("", "http://") and after == url.rpartition("@")[2], shown


# logins shaped like a known key or token: the scan finds that kind too, and the advice stays the same
TOKEN_SHAPED = ["http://quant:sk-ant-abcdefghijklmnopqrstuvwxyz0123@127.0.0.1:8090",
                "http://ghp_" + "a1B2c3D4e5" * 3 + "a1B2c3@127.0.0.1:8090",
                "http://eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJxdWFudCJ9.c2lnbmF0dXJlMTIz@127.0.0.1:8090"]


@pytest.mark.parametrize("url", TOKEN_SHAPED, ids=["api key", "github token", "jwt"])
def test_a_token_shaped_quant_url_login_gets_the_same_advice(tmp_path, url):
    """A login shaped like a known key or token is found as that kind and as url_password. The pack still reads it
    from the URL, so the warning still says to protect the file, not to move it to the environment."""
    found = findings(tmp_path, {"quant_url": url})
    assert [(f["level"], f["path"]) for f in found] == [("warn", "pack_config.quant_url")], found
    kinds, _, why = found[0]["message"].partition("): ")
    assert kinds.startswith("contains what looks like a secret (") and "url_password" in kinds.split("(")[1], found
    assert why.startswith("the user info of this URL is a login written into rlcd.yaml"), found
    project = Project(root=tmp_path, data={"pack": "trading", "pack_config": {"quant_url": url}})
    shown = CC.effective(project, offline())["pack_config"]["quant_url"]
    assert shown == "http://[REDACTED:url_password]@127.0.0.1:8090", shown


@pytest.mark.parametrize(("url", "auth"), ACCEPTED)
def test_config_check_and_show_of_a_quant_url_the_client_accepts(tmp_path, url, auth):
    """quant_url is declared one URL (`Key.url`): `rlcd config check` gives the url_password warning for any user
    info, a token alone included, and `rlcd config show` redacts the whole user info, the user name included. The
    pack reads the login from the URL, so the warning does not say to move it to the environment, which nothing
    would read, but to protect the file. (The user name was shown, as were a token alone, with no warning, a
    password with a space, and the part of a password after an `@`.)"""
    found = findings(tmp_path, {"quant_url": url})
    assert [(f["level"], f["path"]) for f in found] == [("warn", "pack_config.quant_url")], found
    assert found[0]["message"] == (
        "contains what looks like a secret (url_password): the user info of this URL is a login written into "
        "rlcd.yaml, where its pack reads it: keep the file out of version control and readable only by the service "
        "(see docs/configuration.md)"), found
    project = Project(root=tmp_path, data={"pack": "trading", "pack_config": {"quant_url": url}})
    shown = CC.effective(project, offline())["pack_config"]["quant_url"]
    assert shown == "http://[REDACTED:url_password]@127.0.0.1:8090", shown


def test_config_check_quotes_a_quant_url_it_rejects_without_its_password(tmp_path):
    """A quant_url httpx reads, but not as an http(s) URL with a host, is quoted as httpx read it, with `***@` in
    place of its user info, so no part of the user name or password is quoted, whatever it holds. (It used to be
    quoted as written; and a netloc urlsplit refuses failed the pack's whole check with urlsplit's message, which
    quotes the netloc.)"""
    for url, shown in [("http://quant:S3cretPassw0rd@:8090", "http://***@:8090"),
                       ("http://quant:S3cret Passw0rd@:8090", "http://***@:8090"),
                       ("http://qu\"ant:S3cret' Passw0rd@:8090", "http://***@:8090"),
                       ("ftp://quant:S3cret@Passw0rd@127.0.0.1:8090", "ftp://***@127.0.0.1:8090"),
                       ("ftp://ann@S3cret:Passw0rd@127.0.0.1:8090", "ftp://***@127.0.0.1:8090"),
                       ("ftp://S3cret Passw0rd@127.0.0.1:8090", "ftp://***@127.0.0.1:8090"),
                       ("//S3cret Passw0rd@127.0.0.1:8090", "//***@127.0.0.1:8090")]:
        found = findings(tmp_path, {"quant_url": url})
        errors = quant_url_findings(found, "error")
        assert errors == [f"must be an http:// or https:// URL with a host, got {shown!r}"], found


def test_the_client_reads_quant_url_as_the_config_check_does(tmp_path):
    """Both strip quant_url (YAML keeps the spaces of a quoted value), and a quant_url that is not text is the client's
    INVALID_URL. `rlcd config show` redacts the user info after the leading space as well. (The client used to keep the
    spaces, so httpx read a relative URL, which it sends no request to, while the config check passed it.)"""
    url = " http://quant:S3cret Passw0rd@127.0.0.1:8090 "
    project = Project(root=tmp_path, data={"pack": "trading", "pack_config": {"quant_url": url}})
    found = CC.check_project(project, offline())
    assert [(f["level"], f["path"]) for f in found] == [("warn", "pack_config.quant_url")], found
    assert "(url_password)" in found[0]["message"]
    shown = CC.effective(project, offline())["pack_config"]["quant_url"]
    assert shown == " http://[REDACTED:url_password]@127.0.0.1:8090 ", shown
    client = offline().client(project)
    assert str(client._http.base_url) == "http://127.0.0.1:8090" and client.url == url.strip()
    with pytest.raises(ContractError) as e:
        offline().client(Project(root=tmp_path, data={"pack": "trading", "pack_config": {"quant_url": 8090}}))
    assert e.value.message == INVALID_URL


def test_a_password_urlsplit_refuses_is_read_as_httpx_reads_it(tmp_path):
    """The host check and `rlcd config check` read quant_url as httpx does: a password with U+FF0F, the fullwidth
    solidus, which urlsplit refused (NFKC normalizes it to `/`, so it took the netloc for a forged one), is part of the
    user info for httpx, before the loopback host it connects to. The client is built and authenticates with it, and
    the config check gives only the url_password warning. (The client used to refuse it, and the pack's config check
    to fail on it, both with urlsplit's message, which quotes the netloc, password included.)"""
    url = "http://quant:S3cret\uff0fPassw0rd@127.0.0.1:8090"
    svc = make(tmp_path, FakeSidecar(rows=600, auth=("quant", "S3cret\uff0fPassw0rd")), quant_url=url)
    assert svc.trainer.start(TrainRequest(), wait=True)["status"] == "succeeded"
    found = findings(tmp_path, {"quant_url": url})
    assert CC.errors(found) == [] and all(hidden(f["message"]) for f in found), found
    assert any("(url_password)" in m for m in quant_url_findings(found, "warn")), found


# a fuzz of quant_urls with a password: S3cret, one of FUZZ_SEPS, Passw0rd, behind each scheme form and user name,
# before each host
FUZZ_SCHEMES = ["http://", "HTTP://", "https://", "//", "http:/", "http:\\\\", "ftp://", ""]
FUZZ_USERS = ["quant", "", "qu\"ant"]
FUZZ_SEPS = ["", "@", " ", "/", "?", "#", ":", "\\", "'", "%40", "%ZZ", " %21", "\t", "ä", "\uff0f", "\uff20",
             "\ud800"]
FUZZ_HOSTS = ["127.0.0.1:8090", ":8090", "", "[::1", "127.0.0.1:8090/base", "quant.example.com:8090"]


@pytest.mark.parametrize("sep", FUZZ_SEPS, ids=[ascii(s)[1:-1] or "none" for s in FUZZ_SEPS])
def test_no_message_quotes_the_quant_url_password_however_malformed(tmp_path, sep):
    """For each quant_url of the fuzz, with remote hosts allowed or not, the client is built (with remote hosts not
    allowed, only for a loopback host or none) or refuses it with a 422 whose message and traceback quote no part of
    the password, and neither its base URL nor the URL it logs holds any. No `rlcd config check` finding quotes any
    part of it, and every one, with a scheme or not, gets the url_password warning and is redacted by
    `rlcd config show`."""
    for scheme, user, host in itertools.product(FUZZ_SCHEMES, FUZZ_USERS, FUZZ_HOSTS):
        url = f"{scheme}{user}:S3cret{sep}Passw0rd@{host}"
        for remote in (False, True):
            project = Project(root=tmp_path, data={"pack": "trading", "pack_config": {"quant_url": url},
                                                   "security": {"allow_remote": remote}})
            try:
                client = offline().client(project)
            except ContractError as e:
                assert e.status == 422 and e.kind == "validation", (url, e.body())
                assert hidden("".join(traceback.format_exception(e))), (url, e.message)
            else:
                base = client._http.base_url
                assert hidden(f"{base} {client._logged_url}"), (url, str(base))
                assert remote or not base.raw_host or is_loopback(base.raw_host.decode()), (url, str(base))
        project = Project(root=tmp_path, data={"pack": "trading", "pack_config": {"quant_url": url}})
        found = CC.check_project(project, offline())
        assert all(hidden(f["message"]) for f in found), (url, found)
        assert any("(url_password)" in m for m in quant_url_findings(found, "warn")), (url, found)
        assert hidden(CC.effective(project, offline())["pack_config"]["quant_url"]), url


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


# ----------------------------------------------------------------------------- carry-over and the trading parameters

TRAINED_WITH = {"sl_pips": 20, "tp_pips": 50, "tp2_pips": 100, "horizon": 8}
# the three ways a train leaves setup:1h out: switched off, its interval removed, a partial train (`--heads`)
UNTRAINED = ["switched off", "interval removed", "partial train"]


def service_at(root, sidecar: FakeSidecar, intervals: list[str], heads: dict | None = None, **cfg) -> Service:
    """A trading service on root/var with these intervals, both heads logreg unless `heads` says otherwise."""
    data = {"pack": "trading",
            "heads": {"setup:15m": {"model": "logreg"}, "setup:1h": {"model": "logreg"}, **(heads or {})},
            "pack_config": {"quant_url": "http://127.0.0.1:8090", "symbols": ["XAUUSD", "EURUSD"],
                            "intervals": intervals, "horizon": 8, **cfg}}
    return Service(Project(root=root, data=data), home=root / "var", pack=TradingPack(sidecar.transport()))


def train_twice(root, sidecar: FakeSidecar, how: str, sl_pips: int) -> tuple[str, Service, str]:
    """v1: setup:15m and setup:1h with sl_pips 20. v2: sl_pips set to `sl_pips`, and a train that retrains only
    setup:15m (`how`). Returns v1, the service of the second train and v2."""
    first = service_at(root, sidecar, ["15m", "1h"], sl_pips=20).trainer.start(TrainRequest(), wait=True)
    assert first["status"] == "succeeded" and set(first["heads"]) == {"setup:15m", "setup:1h"}, first
    req = TrainRequest(heads=["setup:15m"]) if how == "partial train" else TrainRequest()
    if how == "switched off":
        svc = service_at(root, sidecar, ["15m", "1h"], {"setup:1h": {"enabled": False}}, sl_pips=sl_pips)
    else:
        svc = service_at(root, sidecar, ["15m"] if how == "interval removed" else ["15m", "1h"], sl_pips=sl_pips)
    job = svc.trainer.start(req, wait=True)
    assert job["status"] == "succeeded" and list(job["heads"]) == ["setup:15m"], job
    return first["model"], svc, job["model"]


def features_asked(sidecar: FakeSidecar) -> set[tuple[str, str]]:
    """(interval, sl_pips) of each /features call since `sidecar.calls` was last cleared."""
    return {(q.get("interval"), q.get("sl_pips")) for path, q in sidecar.calls if path == "/features"}


@pytest.mark.parametrize("how", UNTRAINED)
def test_a_head_trained_with_other_trading_params_is_not_carried_over(tmp_path, how):
    """sl_pips edited from 20 to 30, then a train that does not retrain setup:1h. Its labels were made with a 20-pip
    stop, and the new version sends 30 with every /features call, so it is not carried over: v2 holds setup:15m
    only, and its notes name setup:1h. Switched on (or listed) again, setup:1h is answered as any head the model
    lacks: the uninformed prior, calibrated false, hold, and no /features call goes out for it. (It used to be carried
    over and answered by v2 with sl_pips 30.)"""
    sidecar = FakeSidecar(rows=600)
    _, svc, v2 = train_twice(tmp_path, sidecar, how, sl_pips=30)
    meta = svc.bundle(v2).meta
    assert list(meta["heads"]) == ["setup:15m"] and meta["pack"]["carry_key"] == {**TRAINED_WITH, "sl_pips": 30}
    assert any("not carried over" in n and "setup:1h" in n for n in meta["notes"]), meta["notes"]
    on = service_at(tmp_path, sidecar, ["15m", "1h"], sl_pips=30)
    sidecar.calls.clear()
    d = on.decide(DecideRequest(state={"symbol": "XAUUSD"}, heads=["setup:1h"]))
    answer = d["heads"]["setup:1h"]["answer"]
    assert d["model"] == v2 and d["action"] == "hold" and answer["calibrated"] is False
    assert answer["probabilities"] == pytest.approx(dict.fromkeys(("buy", "sell", "hold"), 1 / 3))
    assert any(f"head 'setup:1h' is not trained in {v2}" in w for w in d["warnings"]), d["warnings"]
    assert features_asked(sidecar) == set()
    on.decide(DecideRequest(state={"symbol": "XAUUSD"}, heads=["setup:15m"]))
    assert features_asked(sidecar) == {("15m", "30")}           # what v2 asks with: the values it was trained with


@pytest.mark.parametrize("how", UNTRAINED)
def test_with_the_same_trading_params_a_train_carries_the_other_head_over(tmp_path, how):
    """The control: sl_pips, tp_pips, tp2_pips and horizon unchanged. setup:1h is carried over from v1 and, switched
    on (or listed) again, served by v2 without a retrain, asking the sidecar with the values it was trained with.
    (test_a_switched_off_trading_head_is_not_trained_or_served covers serving it again on the version that trained
    it; this is the same across a train.)"""
    sidecar = FakeSidecar(rows=600)
    v1, svc, v2 = train_twice(tmp_path, sidecar, how, sl_pips=20)
    meta = svc.bundle(v2).meta
    assert set(meta["heads"]) == {"setup:15m", "setup:1h"} and meta["heads"]["setup:1h"]["trained_in"] == v1
    assert meta["pack"]["carry_key"] == svc.bundle(v1).meta["pack"]["carry_key"] == TRAINED_WITH
    on = service_at(tmp_path, sidecar, ["15m", "1h"], sl_pips=20)
    sidecar.calls.clear()
    d = on.decide(DecideRequest(state={"symbol": "XAUUSD"}, heads=["setup:1h"]))
    assert d["model"] == v2 and d["heads"]["setup:1h"]["answer"]["calibrated"] is True
    assert features_asked(sidecar) == {("1h", "20")}


def test_a_trading_version_trained_before_carry_key_carries_no_head_over(tmp_path):
    """A version trained before carry_key existed stores the four values (pack.config.params) but no pack.carry_key,
    and Registry.commit compares only carry_key: the next train carries none of its heads over, even with the same
    parameters."""
    sidecar = FakeSidecar(rows=600)
    v1 = service_at(tmp_path, sidecar, ["15m", "1h"], sl_pips=20).trainer.start(TrainRequest(), wait=True)["model"]
    path = tmp_path / "var" / "models" / v1 / "model.json"
    meta = json.loads(path.read_text(encoding="utf-8"))
    meta["pack"].pop("carry_key", None)
    path.write_text(json.dumps(meta), encoding="utf-8")
    svc = service_at(tmp_path, sidecar, ["15m", "1h"], sl_pips=20)
    v2 = svc.trainer.start(TrainRequest(heads=["setup:15m"]), wait=True)["model"]
    meta = svc.bundle(v2).meta
    assert list(meta["heads"]) == ["setup:15m"] and meta["pack"]["carry_key"] == TRAINED_WITH
    assert any("not carried over" in n and "setup:1h" in n for n in meta["notes"]), meta["notes"]


def test_the_pack_info_has_a_carry_key_only_when_the_pack_gives_one(tmp_path):
    """BasePack.info stores `carry_key` only when it is not None: the trading pack's is its four parameters, the
    tabular and documents packs give none, so their pack info is what it was."""
    svc = make(tmp_path, FakeSidecar(), sl_pips=25)
    info = svc.pack.info(svc.pack.provider(svc.project))
    assert info["carry_key"] == {**TRAINED_WITH, "sl_pips": 25} and "carry_key" not in svc.pack.info()
    for name in ("tabular", "documents"):
        assert set(load_pack(name).info()) == {"name", "version", "config"}
