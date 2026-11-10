"""What a brand-new business and a containerised deployment ran into, pinned.

A. commands outside a project stop with one line (nothing read, written or downloaded); RLCD_PROJECT pointing at
   nothing is an error, never "no project";
B. `rlcd serve` refuses a non-loopback host without RLCD_API_KEY, says when it serves no trained model, and
   GET /ready tells "process up" from "ready to decide"; promote / rollback record who did it;
C. the Bernoulli gate is explained with numbers computed by the gate; a policy edit that is not served yet, and
   a config typo, are said on stderr by `rlcd decide` / `rlcd ask`; `rlcd decide --heads`; a stored policy with
   `abstain: act` refuses to decide;
D. `policy.overrides` names are checked, `train.*` ranges equal the API's, dataset notes are printed, and the
   small messages (head name twice, labelling advice for a pack without labelling, an empty table) are fixed.
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml
from fastapi.testclient import TestClient

from open_rlcd import cli
from open_rlcd.cli import _head_error, _notes_block, is_loopback, main
from open_rlcd.core import bernoulli as B
from open_rlcd.core import config_check as CC
from open_rlcd.core import policy as PL
from open_rlcd.core.checklist import gate_lines, policy_drift
from open_rlcd.core.config import Project, find_config, load_project
from open_rlcd.core.contract import ContractError, DecideRequest, FeedbackRequest, TrainRequest
from open_rlcd.core.service import Service
from open_rlcd.serve.api import create_app
from tests.toypack import NAMES, ToyPack

REPO = Path(__file__).resolve().parent.parent
STATE = {"f": [0.3, -0.2, 0.1, 0.0, 0.5, -0.4]}


# ----------------------------------------------------------------------------- helpers

def write_table(path: Path, n: int, seed: int = 1) -> None:
    rng = np.random.default_rng(seed)
    a, b = rng.normal(size=n), rng.normal(size=n)
    plan = rng.choice(["x", "y", "z"], size=n)
    p = 1 / (1 + np.exp(-(2.5 * a - 1.5 * b + (plan == "x"))))
    y = np.where(rng.random(n) < p, "yes", "no")
    rows = ["a,b,plan,outcome"] + [f"{a[i]:.3f},{b[i]:.3f},{plan[i]},{y[i]}" for i in range(n)]
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def tabular_project(root: Path, n: int | None = 900) -> Path:
    assert main(["init", str(root), "--pack", "tabular"]) == 0
    if n is not None:
        write_table(root / "data.csv", n)
    return root


def edit_yaml(root: Path, fn) -> None:
    path = root / "rlcd.yaml"
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    fn(data)
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def run(root: Path, home: Path, *argv: str) -> int:
    return main(["--project", str(root), "--home", str(home), *argv])


@pytest.fixture(scope="module")
def biz(tmp_path_factory):
    """A tabular project trained once through the CLI: (root, home, what `rlcd train` printed)."""
    base = tmp_path_factory.mktemp("hardening")
    root, home = tabular_project(base / "biz"), base / "home"
    with pytest.MonkeyPatch.context() as mp:
        capture = _Capture(mp)
        assert run(root, home, "train") == 0
    return SimpleNamespace(root=root, home=home, out=capture.out(), err=capture.err())


class _Capture:
    """stdout / stderr of a `main(...)` call made outside a test function (a module fixture has no capsys)."""

    def __init__(self, mp: pytest.MonkeyPatch):
        import io
        import sys

        self._out, self._err = io.StringIO(), io.StringIO()
        mp.setattr(sys, "stdout", self._out)
        mp.setattr(sys, "stderr", self._err)

    def out(self) -> str:
        return self._out.getvalue()

    def err(self) -> str:
        return self._err.getvalue()


def toy_service(home: Path, data: dict | None = None, pack: ToyPack | None = None) -> Service:
    return Service(Project(root=home, data=dict(data or {})), home=home, pack=pack or ToyPack())


def toy_train(svc: Service) -> str:
    job = svc.trainer.start(TrainRequest(), wait=True)
    assert job["status"] == "succeeded", job
    return job["model"]


@pytest.fixture(scope="module")
def toy(tmp_path_factory):
    """One trained toy service for tests that change nothing in its home."""
    home = tmp_path_factory.mktemp("hardening-toy")
    svc = toy_service(home)
    return SimpleNamespace(svc=svc, model=toy_train(svc), home=home, client=TestClient(create_app(svc)))


@pytest.fixture
def served(monkeypatch):
    """`uvicorn.run` replaced by a recorder: `rlcd serve` returns instead of listening."""
    import uvicorn

    calls: list[dict] = []
    monkeypatch.setattr(uvicorn, "run", lambda app, **kw: calls.append({"app": app, **kw}))
    monkeypatch.delenv("RLCD_API_KEY", raising=False)
    monkeypatch.delenv("RLCD_ADMIN_KEY", raising=False)
    return calls


@pytest.fixture
def nowhere(tmp_path, monkeypatch):
    """An empty directory with no rlcd.yaml above it, no RLCD_HOME and no RLCD_PROJECT."""
    assert find_config(tmp_path) is None, "the test directory sits under an rlcd.yaml"
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("RLCD_HOME", raising=False)
    monkeypatch.delenv("RLCD_PROJECT", raising=False)
    monkeypatch.delenv("RLCD_PACK", raising=False)
    return tmp_path


# ----------------------------------------------------------------------------- A. commands outside a project

@pytest.mark.parametrize("argv", [["train"], ["ingest", "./data"], ["label", "status"], ["label", "audit"],
                                  ["label", "bootstrap"], ["label", "next", "--head", "x"]])
def test_project_commands_stop_with_one_line_outside_a_project(nowhere, monkeypatch, capsys, argv):
    def boom(*a, **k):
        raise AssertionError("a service was built (a pack or an encoder could be loaded) outside a project")
    monkeypatch.setattr(Service, "__init__", boom)
    assert main(argv) == 2
    io = capsys.readouterr()
    assert io.out == "" and len(io.err.strip().splitlines()) == 1
    assert io.err.startswith("rlcd: no rlcd.yaml here or in any parent folder")
    assert f"`rlcd {argv[0]}`" in io.err and "rlcd init <name> --pack" in io.err
    assert list(nowhere.iterdir()) == []                         # no ./var, nothing downloaded into it


@pytest.mark.parametrize("argv", [["import", "bundle"], ["export", "out"], ["decide", "--query", "hi"],
                                  ["ask", "--query", "hi"], ["check"], ["report"], ["bench"], ["calibration", "h"],
                                  ["batteries"], ["receipts", "verify"], ["automation", "off"], ["promote", "v"],
                                  ["rollback"], ["metrics"], ["review"], ["replay"], ["regress"],
                                  ["push", "--hf", "a/b"],
                                  ["case", "add", "--from-decision", "d", "--expect", "class=x"]])
def test_home_commands_do_not_invent_a_home_outside_a_project(nowhere, capsys, argv):
    assert main(argv) == 2
    io = capsys.readouterr()
    assert io.out == "" and len(io.err.strip().splitlines()) == 1
    assert "no rlcd.yaml here or in any parent folder, and no model home" in io.err
    assert "--home DIR or RLCD_HOME" in io.err and f"`rlcd {argv[0]}`" in io.err
    assert list(nowhere.iterdir()) == []                         # `rlcd import` used to create ./var here


def test_home_commands_run_against_a_named_or_existing_home(nowhere, tmp_path_factory, monkeypatch, capsys):
    home = tmp_path_factory.mktemp("named-home")
    assert main(["--home", str(home), "automation", "status"]) == 0          # --home names it
    assert json.loads(capsys.readouterr().out)["automation"]["enabled"] is True
    monkeypatch.setenv("RLCD_HOME", str(home))
    assert main(["automation", "status"]) == 0                               # so does RLCD_HOME (the image sets it)
    capsys.readouterr()
    monkeypatch.delenv("RLCD_HOME")
    assert main(["automation", "status"]) == 2
    (nowhere / "var").mkdir()
    (nowhere / "var" / "registry.json").write_text('{"latest": null, "next": 1, "versions": {}}')
    capsys.readouterr()
    assert main(["automation", "status"]) == 0                               # ./var already holds a registry
    assert "automation" in json.loads(capsys.readouterr().out)


def test_commands_that_need_no_project_still_run_anywhere(nowhere, capsys):
    assert main(["config", "check"]) == 0
    assert "no rlcd.yaml here" in capsys.readouterr().out
    assert main(["config", "show", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["project"]["config"] is None
    assert main(["init", "shop", "--pack", "tabular"]) == 0
    assert (nowhere / "shop" / "rlcd.yaml").is_file()
    assert [p.name for p in nowhere.iterdir()] == ["shop"]


def test_serve_without_a_project_still_serves_the_demo_bundle(nowhere, served, capsys):
    if not (REPO / "bundles" / "demo-documents" / "model.json").is_file():
        pytest.skip("demo bundle not built")
    assert main(["serve"]) == 0
    io = capsys.readouterr()
    assert "imported the demo bundle (synthetic data only)" in io.out
    assert len(served) == 1 and served[0]["host"] == "127.0.0.1"
    assert "cannot decide yet" not in io.err                     # the demo model is trained: no readiness warning


def test_every_command_has_a_decision_about_where_it_runs():
    """A new subcommand must be put into NEEDS_PROJECT, NEEDS_HOME or the list of commands that run anywhere."""
    sub = next(a for a in cli.build_parser()._actions if a.dest == "cmd")
    anywhere = {"init", "config", "serve", "demo"}
    assert set(sub.choices) == cli.NEEDS_PROJECT | cli.NEEDS_HOME | anywhere
    assert not cli.NEEDS_PROJECT & cli.NEEDS_HOME and not anywhere & (cli.NEEDS_PROJECT | cli.NEEDS_HOME)


def test_rlcd_project_pointing_at_nothing_is_an_error_not_no_project(tmp_path, monkeypatch, served, capsys):
    monkeypatch.setenv("RLCD_PROJECT", str(tmp_path / "typo"))
    with pytest.raises(FileNotFoundError, match="RLCD_PROJECT=.*typo.*no rlcd.yaml"):
        find_config()
    with pytest.raises(FileNotFoundError):
        load_project()
    with pytest.raises(FileNotFoundError):
        Service()                                                # what `uvicorn open_rlcd.serve.api:app` builds
    (tmp_path / "empty").mkdir()
    monkeypatch.setenv("RLCD_PROJECT", str(tmp_path / "empty"))  # a directory that exists, without rlcd.yaml
    for argv in (["serve"], ["train"], ["decide", "--query", "hi"], ["config", "check"]):
        assert main(argv) == 1
        err = capsys.readouterr().err
        assert err.startswith("rlcd: ") and "RLCD_PROJECT=" in err and "rlcd.yaml" in err
    assert served == []                                          # the demo model was never served
    root = tabular_project(tmp_path / "ok")
    monkeypatch.setenv("RLCD_PROJECT", str(root))
    assert find_config() == root / "rlcd.yaml"
    monkeypatch.setenv("RLCD_PROJECT", str(root / "rlcd.yaml"))
    assert load_project().path == root / "rlcd.yaml"


# ----------------------------------------------------------------------------- B. serving

def test_loopback_hosts():
    assert all(is_loopback(h) for h in ("127.0.0.1", "localhost", "::1", "[::1]", "127.0.0.2", "LOCALHOST"))
    assert not any(is_loopback(h) for h in ("0.0.0.0", "::", "10.0.0.5", "rlcd.example.com", ""))


def test_serve_refuses_a_non_loopback_host_without_an_api_key(tmp_path, served, monkeypatch, capsys):
    root, home = tabular_project(tmp_path / "p"), tmp_path / "home"
    assert run(root, home, "serve", "--host", "0.0.0.0") == 2
    err = capsys.readouterr().err
    assert "refusing to serve on 0.0.0.0 without RLCD_API_KEY" in err and "--insecure-no-auth" in err
    assert "every route, training and the kill switch included" in err and served == []

    monkeypatch.setenv("RLCD_ADMIN_KEY", "admin")                # an admin key alone leaves the other routes open
    assert run(root, home, "serve", "--host", "0.0.0.0") == 2
    err = capsys.readouterr().err
    assert "RLCD_ADMIN_KEY is set, but it only protects training, the kill switch and confirmations" in err
    assert served == []
    monkeypatch.delenv("RLCD_ADMIN_KEY")

    edit_yaml(root, lambda d: d["serve"].update(host="10.1.2.3"))   # the host from rlcd.yaml counts too
    assert run(root, home, "serve") == 2 and served == []
    assert "refusing to serve on 10.1.2.3" in capsys.readouterr().err

    assert run(root, home, "serve", "--insecure-no-auth") == 0      # the explicit opt-out: a warning, and it starts
    assert "warning: --insecure-no-auth: serving on 10.1.2.3 without RLCD_API_KEY" in capsys.readouterr().err
    assert served.pop()["host"] == "10.1.2.3"

    monkeypatch.setenv("RLCD_API_KEY", "key")                       # with the key: no flag needed, no warning
    assert run(root, home, "serve", "--host", "0.0.0.0") == 0
    assert "RLCD_API_KEY" not in capsys.readouterr().err and served.pop()["host"] == "0.0.0.0"
    monkeypatch.delenv("RLCD_API_KEY")

    assert run(root, home, "serve", "--host", "127.0.0.1") == 0     # loopback never needed a key
    assert "refusing" not in capsys.readouterr().err and served.pop()["host"] == "127.0.0.1"


def test_the_serve_help_says_what_each_key_protects():
    sub = next(a for a in cli.build_parser()._actions if a.dest == "cmd")
    flag = next(a for a in sub.choices["serve"]._actions if "--insecure-no-auth" in a.option_strings)
    for part in ("RLCD_API_KEY protects every route except /health and /ready", "RLCD_ADMIN_KEY protects only",
                 "/v1/train", "/v1/automation", "/v1/receipts/{id}/confirm", "leaves every other route open"):
        assert part in " ".join(flag.help.split())


def test_an_admin_key_alone_leaves_the_decision_routes_open(toy, monkeypatch):
    """Why `rlcd serve` treats "admin key set, API key unset" like "no key": only the operator routes are guarded."""
    monkeypatch.delenv("RLCD_API_KEY", raising=False)
    monkeypatch.setenv("RLCD_ADMIN_KEY", "admin")
    assert toy.client.get("/v1/heads").status_code == 200
    assert toy.client.post("/v1/systemone", json={"state": STATE, "questions": {
        "q": {"type": "noul", "head": "good"}}}).status_code == 200
    assert toy.client.post("/v1/automation", json={"enabled": True}).status_code == 401
    assert toy.client.post("/v1/train", json={}).status_code == 401


def test_serve_says_at_startup_that_no_model_is_trained(tmp_path, served, capsys):
    root, home = tabular_project(tmp_path / "p"), tmp_path / "home"
    assert run(root, home, "serve") == 0
    err = capsys.readouterr().err
    assert "warning: serving rlcd-0.1.0, which cannot decide yet: no model is trained" in err
    assert "rlcd train" in err and "GET /ready answers 503" in err and len(served) == 1


def test_a_trained_project_serves_without_startup_warnings(biz, served, capsys):
    assert run(biz.root, biz.home, "serve") == 0
    assert capsys.readouterr().err == "" and len(served) == 1


class NoHeads(ToyPack):
    """A pack that has no head yet (the tabular pack before its data file exists)."""

    def heads(self, project):
        return {}


def test_requests_to_an_untrained_service_say_to_train(tmp_path, toy):
    c = TestClient(create_app(toy_service(tmp_path, pack=NoHeads())))
    r = c.post("/v1/decide", json={"state": STATE})
    assert r.status_code == 422 and r.json()["kind"] == "validation"          # status and kind are unchanged
    assert r.json()["error"].startswith("no heads to decide with") and "run `rlcd train`" in r.json()["error"]
    assert "No model is trained yet" in r.json()["error"]
    r = c.post("/v1/systemone", json={"state": STATE, "questions": {"q": {"type": "noul", "head": "outcome"}}})
    body = r.json()
    assert r.status_code == 422 and body["kind"] == "unknown_head" and body["available_heads"] == []
    assert body["error"].startswith("no head named 'outcome'") and "run `rlcd train`" in body["error"]
    # a trained service with heads keeps the plain message: the caller misspelled a head
    r = toy.client.post("/v1/systemone", json={"state": STATE, "questions": {"q": {"type": "noul", "head": "nope"}}})
    assert r.json()["kind"] == "unknown_head" and "rlcd train" not in r.json()["error"]
    assert r.json()["available_heads"] == ["good", "color", "level"]


def test_ready_tells_process_up_from_ready_to_decide(tmp_path, monkeypatch):
    svc = toy_service(tmp_path)
    c = TestClient(create_app(svc))
    h = c.get("/health")
    assert h.status_code == 200 and h.json()["status"] == "ok" and h.json()["ready"] is False
    r = c.get("/ready")
    body = r.json()
    assert r.status_code == 503 and body["kind"] == "not_ready" and body["retryable"] is True
    assert body["ready"] is False and body["status"] == "not_ready" and body["model"] == "rlcd-0.1.0"
    assert "no model is trained" in body["reasons"][0] and "rlcd train" in body["error"]
    assert r.headers["retry-after"]
    toy_train(svc)
    r = c.get("/ready")
    assert r.status_code == 200
    assert r.json() == {"status": "ready", "ready": True, "model": "rlcd-0.1.1", "reasons": []}
    h = c.get("/health").json()
    assert h["status"] == "ok" and h["ready"] is True and h["heads"]["good"] == {"trained": True}
    monkeypatch.setenv("RLCD_API_KEY", "s3cret")                 # probes carry no key
    assert c.get("/ready").status_code == 200 and c.get("/health").status_code == 200
    assert c.get("/v1/heads").status_code == 401
    assert str(tmp_path) not in c.get("/ready").text


class Broken(ToyPack):
    def heads(self, project):
        raise RuntimeError("cannot open /secret/path/corpus.db")


def test_ready_is_503_when_the_pack_cannot_load_or_every_head_is_disabled(tmp_path, toy):
    c = TestClient(create_app(toy_service(tmp_path / "a", pack=Broken())))
    assert c.get("/health").status_code == 200 and c.get("/health").json()["ready"] is False
    r = c.get("/ready")
    assert r.status_code == 503 and "the pack cannot load (RuntimeError" in r.json()["reasons"][0]
    assert "/secret/path" not in r.text                          # exception text stays in the server log
    off = Service(Project(root=toy.home, data={"heads": {n: {"enabled": False} for n in ("good", "color", "level")}}),
                  home=toy.home, pack=NoHeads())
    r = TestClient(create_app(off)).get("/ready")
    assert r.status_code == 503 and "serves no trained head" in r.json()["reasons"][0]


def test_promote_and_rollback_record_who_did_it(tmp_path, monkeypatch, capsys):
    home = tmp_path / "home"
    svc = toy_service(home)
    v1, v2 = toy_train(svc), toy_train(svc)
    monkeypatch.delenv("USER", raising=False)                    # a container has no $USER

    def history() -> list[dict]:
        return json.loads((home / "registry.json").read_text())["history"]

    assert main(["--home", str(home), "promote", v1, "--by", "ana", "--reason", "CHG-1"]) == 0
    assert history()[-1]["by"] == "ana" and history()[-1]["kind"] == "promote" and history()[-1]["to"] == v1
    assert main(["--home", str(home), "rollback"]) == 0          # no --by, no $USER: a literal, never null
    assert history()[-1]["by"] == "unknown" and history()[-1]["kind"] == "rollback" and history()[-1]["to"] == v2
    monkeypatch.setenv("USER", "root")
    assert main(["--home", str(home), "rollback", "--to", v1]) == 0
    assert history()[-1]["by"] == "root"
    assert main(["--home", str(home), "rollback", "--to", v2, "--by", "oncall"]) == 0
    assert history()[-1]["by"] == "oncall"
    moves = [h for h in history() if h.get("kind") in ("promote", "rollback")]
    assert [h["by"] for h in moves] == ["ana", "unknown", "root", "oncall"]
    capsys.readouterr()


# ----------------------------------------------------------------------------- C. the gate, said in numbers

def _needed(wins: int, losses: int, required: float, cfg: B.BernoulliConfig) -> int:
    n = 0
    while (B.posterior(wins + n, losses, required, cfg)["p_above_required"] or 0) < cfg.min_p_above:
        n += 1
    return n


def test_the_gate_outlook_is_computed_by_the_gate_itself(toy):
    cfg = toy.svc.policy(toy.model)
    bcfg = cfg.bernoulli()
    out = toy.svc.gate_outlook(toy.model, "color")
    verdict = toy.svc.loaded(toy.model, "color").verdict
    assert set(out) == {lv for lv, v in verdict.items() if v["tier"] == 2} and out   # only levels whose verdict is act
    table = toy.svc.ledger.table("color")
    for level, o in out.items():
        lv = cfg.level(level)
        assert (o["act"], o["min_accuracy"], o["threshold"]) == (lv.act, lv.min_accuracy, cfg.gate_min_p_above)
        assert o["from_empty"] == _needed(0, 0, lv.min_accuracy, bcfg)
        assert {c["class"] for c in o["cells"]} == {"red", "green", "blue"}
        assert {c["bin"] for c in o["cells"]} == set(range(B.bin_of(lv.act), B.BINS))
        for c in o["cells"]:
            w, l_ = table["bins"][c["class"]][c["bin"]]
            assert (c["wins"], c["losses"]) == (w, l_)
            assert c["needed"] == _needed(w, l_, lv.min_accuracy, bcfg) and c["passes"] == (c["needed"] == 0)
    # the defaults, as docs/configuration.md states them: 4 / 8 / 21 correct outcomes for a cell without evidence
    default = PL.PolicyConfig()
    assert [_needed(0, 0, default.level(lv).min_accuracy, default.bernoulli()) for lv in ("low", "medium", "high")] \
        == [4, 8, 21]
    assert toy.svc.gate_outlook(toy.model, "nope") == {}
    gate = toy.svc.gate_outlook(toy.model, "good")               # a gate head only ever commits to `true`
    assert all(c["class"] == "true" for o in gate.values() for c in o["cells"])


def test_a_cell_without_evidence_starts_at_confirm_and_acts_after_the_stated_number_of_outcomes(tmp_path):
    svc = toy_service(tmp_path)
    model = toy_train(svc)
    w = np.random.default_rng(99 + 3).normal(size=(len(NAMES), 3))          # the toy generator's weights
    d = w[:, 0] - w[:, 1:].mean(axis=1)
    red = {"f": (3.0 * d / np.linalg.norm(d)).round(3).tolist()}            # red with p close to 1

    def decide() -> dict:
        return svc.decide(DecideRequest(state=red, heads=["color"], stakes="medium"))

    first = decide()["heads"]["color"]["decision"]
    assert first["class"] == "red" and first["tier"] == "act" and first["bernoulli"]["bin"] == 9
    table = svc.ledger.table("color")
    table["bins"]["red"][9] = [0, 0]                             # as if the holdout never reached this cell
    svc.ledger.reset("color", table)
    cell = next(c for c in svc.gate_outlook(model, "color")["medium"]["cells"]
                if (c["class"], c["bin"]) == ("red", 9))
    assert cell == {"class": "red", "bin": 9, "lo": 0.9, "hi": 1.0, "wins": 0, "losses": 0, "passes": False,
                    "needed": 8}
    line = next(x for x in gate_lines(svc.gate_outlook(model, "color")) if x.startswith("medium:"))
    assert "red at p 0.9-1 needs 8 correct (0 of 0 right)" in line and "POST /v1/feedback" in line
    assert "a cell with no evidence yet needs 8 correct outcomes in a row" in line
    for i in range(cell["needed"]):
        out = decide()
        dec = out["heads"]["color"]["decision"]
        assert dec["tier"] == "confirm", i                       # the tester's "0 of 0 times ... act becomes confirm"
        assert any("Bernoulli gate" in x and "act becomes confirm" in x for x in out["warnings"])
        svc.feedback(FeedbackRequest(decision_id=out["decision_id"], label="red", source="test"))
    assert decide()["heads"]["color"]["decision"]["tier"] == "act"
    assert next(c for c in svc.gate_outlook(model, "color")["medium"]["cells"]
                if (c["class"], c["bin"]) == ("red", 9))["passes"] is True


def test_train_and_check_say_what_the_gate_will_do(biz, capsys):
    assert "outcome           trained" in biz.out and "verdict(medium)" in biz.out
    assert "Live decisions (verdict act): `act` also needs the Bernoulli gate" in biz.out
    svc = Service(load_project(biz.root), home=biz.home)
    lines = gate_lines(svc.gate_outlook(svc.registry.latest_id(), "outcome"))
    assert lines and all(f"  outcome  {line}" in biz.out for line in lines)      # the numbers are the gate's own
    assert any("cells that can act (p >= 0.85)" in x for x in lines)
    assert run(biz.root, biz.home, "check", "--json") in (0, 1)
    item = {c["id"]: c for c in json.loads(capsys.readouterr().out)["checks"]}["skill.outcome"]
    assert item["status"] == "pass" and "Live decisions also pass the Bernoulli gate" in item["detail"]
    assert all(line in item["detail"] for line in lines)


def test_gate_lines_say_when_everything_passes_and_when_a_cell_cannot():
    cell = {"class": "true", "bin": 9, "lo": 0.9, "hi": 1.0, "wins": 40, "losses": 1, "passes": True, "needed": 0}
    level = {"act": 0.85, "min_accuracy": 0.75, "threshold": 0.9, "from_empty": 8}
    assert gate_lines({"medium": {**level, "cells": [cell]}}) == [
        "medium: all 1 cells that can act (p >= 0.85) pass on holdout evidence"]
    bad = {**cell, "wins": 1, "losses": 400, "passes": False, "needed": None}
    line = gate_lines({"medium": {**level, "cells": [cell, bad]}})[0]
    assert line.startswith("medium: 1 of 2 cells") and "cannot pass on correct outcomes alone (1 of 401 right)" in line
    assert gate_lines({}) == []


# ----------------------------------------------------------------------------- C. policy drift, findings, --heads

def test_decide_ask_and_serve_note_a_policy_that_is_not_served_yet(biz, tmp_path, served, capsys):
    import shutil

    root = tmp_path / "biz"
    shutil.copytree(biz.root, root)
    home = tmp_path / "home"
    shutil.copytree(biz.home, home)
    state = json.dumps({"a": 0.9, "b": 0.0, "plan": "y"})
    assert run(root, home, "decide", "--state", state) == 0
    assert capsys.readouterr().err == ""                         # rlcd.yaml is what is served: nothing to say

    edit_yaml(root, lambda d: d["policy"].update(stakes={"medium": {"act": 0.99}}))
    assert run(root, home, "decide", "--state", state) == 0
    io = capsys.readouterr()
    d = json.loads(io.out)                                       # stdout is still exactly the decision
    assert d["policy"]["stakes"]["medium"]["act"] == 0.85        # the served threshold, not the edited one
    note = [x for x in io.err.splitlines() if x.startswith("note: ")]
    assert len(note) == 1 and "rlcd.yaml differs from what rlcd-0.1.1 serves" in note[0]
    assert "stakes.medium.act 0.85 -> 0.99" in note[0] and "`rlcd train`" in note[0]
    assert run(root, home, "ask", "--state", state, "--heads", "outcome") == 0
    io = capsys.readouterr()
    assert "answers" in json.loads(io.out) and "stakes.medium.act 0.85 -> 0.99" in io.err
    assert run(root, home, "serve") == 0
    assert "note: rlcd.yaml differs from what rlcd-0.1.1 serves" in capsys.readouterr().err and len(served) == 1
    # `rlcd check` reports the same text (thresholds.config)
    svc = Service(load_project(root), home=home)
    drift = policy_drift(svc, "rlcd-0.1.1")
    assert drift == "stakes.medium.act 0.85 -> 0.99" and drift in note[0]
    assert run(root, home, "check", "--json") in (0, 1)
    item = {c["id"]: c for c in json.loads(capsys.readouterr().out)["checks"]}["thresholds.config"]
    assert item["status"] == "warn" and drift in item["detail"]

    # a head's stakes is served as trained too
    edit_yaml(root, lambda d: (d["policy"].pop("stakes"), d["pack_config"].update(stakes="low")))
    assert policy_drift(Service(load_project(root), home=home), "rlcd-0.1.1") == \
        "heads.outcome.stakes 'medium' -> 'low'"
    assert run(root, home, "decide", "--state", state) == 0
    io = capsys.readouterr()
    assert "heads.outcome.stakes 'medium' -> 'low'" in io.err
    assert json.loads(io.out)["heads"]["outcome"]["decision"]["stakes"] == "medium"

    # and so is its conformal promise: an edit is served only after a retrain, so it is reported like stakes
    # (docs/adoption.md "How to change them" lists it with the thresholds; it used to pass without a word)
    edit_yaml(root, lambda d: (d["pack_config"].pop("stakes"), d["pack_config"].update(conformal=0.1)))
    svc = Service(load_project(root), home=home)
    assert svc.specs("rlcd-0.1.1")["outcome"].conformal is None          # what is served
    assert policy_drift(svc, "rlcd-0.1.1") == "heads.outcome.conformal None -> 0.1"
    assert run(root, home, "check", "--json") in (0, 1)
    item = {c["id"]: c for c in json.loads(capsys.readouterr().out)["checks"]}["thresholds.config"]
    assert item["status"] == "warn" and "heads.outcome.conformal None -> 0.1" in item["detail"]
    assert run(root, home, "decide", "--state", state) == 0
    assert "heads.outcome.conformal None -> 0.1" in capsys.readouterr().err


def test_policy_drift_is_none_without_a_stored_policy_or_a_difference(toy, tmp_path):
    assert policy_drift(toy.svc, toy.model) is None
    assert policy_drift(toy_service(tmp_path), "rlcd-0.1.0") is None           # untrained: rlcd.yaml is served


def test_decide_and_ask_print_config_findings_and_still_answer(biz, tmp_path, capsys):
    import shutil

    root = tmp_path / "biz"
    shutil.copytree(biz.root, root)
    edit_yaml(root, lambda d: d.update(rolout={"mode": "shadow"}))
    state = json.dumps({"a": 0.9, "b": 0.0, "plan": "y"})
    for argv in (["decide", "--state", state], ["ask", "--state", state, "--heads", "outcome"]):
        assert run(root, biz.home, *argv) == 0                   # read-only commands are not blocked
        io = capsys.readouterr()
        assert "rlcd.yaml: error  rolout: unknown key. Did you mean 'rollout'?" in io.err
        assert json.loads(io.out)["model"] == "rlcd-0.1.1"


def test_decide_takes_heads_like_the_api(tmp_path, monkeypatch, capsys):
    seen: list = []

    class Spy:
        project = SimpleNamespace(path=None)

        def resolve(self, model):
            return "rlcd-0.1.0"

        def bundle(self, version):
            return None

        def decide(self, req):
            seen.append(req)
            return {"ok": True}

    monkeypatch.setattr(cli, "_service", lambda args: Spy())
    assert main(["--home", str(tmp_path), "decide", "--state", "{}", "--heads", "a, b"]) == 0
    assert main(["--home", str(tmp_path), "decide", "--state", "{}"]) == 0
    assert [r.heads for r in seen] == [["a", "b"], None]
    capsys.readouterr()


def test_decide_heads_selects_and_rejects_heads(biz, capsys):
    state = json.dumps({"a": 0.9, "b": 0.0, "plan": "y"})
    assert run(biz.root, biz.home, "decide", "--state", state, "--heads", "outcome") == 0
    assert list(json.loads(capsys.readouterr().out)["heads"]) == ["outcome"]
    assert run(biz.root, biz.home, "decide", "--state", state, "--heads", "nope") == 1
    assert "no head named 'nope'" in capsys.readouterr().err


# ----------------------------------------------------------------------------- C. a stored policy that fails open

@pytest.mark.parametrize("name", ["act", "confirm", "", "  ", None, 3])
def test_a_policy_cannot_name_its_abstention_act_or_confirm(name):
    with pytest.raises(ValueError, match="policy.abstain"):
        PL.PolicyConfig(abstain=name)
    with pytest.raises(ValueError, match="policy.abstain"):
        PL.PolicyConfig.from_dict({"abstain": name, "stakes": {"medium": {"act": 0.9}}})


def test_the_usual_abstain_names_and_the_defaults_are_unchanged():
    assert PL.PolicyConfig().abstain == "escalate" and PL.PolicyConfig(abstain="hold").tier_name(0) == "hold"
    d = PL.PolicyConfig().to_dict()
    assert (d["min_verify_rows"], d["gate_min_p_above"], d["miscalibration_p"]) == (30, 0.9, 0.05)
    assert d["stakes"]["medium"] == {"act": 0.85, "confirm": 0.65, "min_accuracy": 0.75}


def test_a_version_whose_stored_policy_abstains_as_act_refuses_to_decide(tmp_path):
    svc = toy_service(tmp_path)
    model = toy_train(svc)
    c = TestClient(create_app(svc))
    assert c.post("/v1/decide", json={"state": STATE, "heads": ["color"]}).status_code == 200
    svc.bundle(model).meta["policy"]["abstain"] = "act"          # a bundle trained before the validator existed
    with pytest.raises(ContractError) as e:
        svc.policy(model)
    assert e.value.kind == "bad_bundle" and e.value.status == 422 and e.value.details["version"] == model
    r = c.post("/v1/decide", json={"state": STATE, "heads": ["color"]})
    body = r.json()
    assert r.status_code == 422 and body["kind"] == "bad_bundle" and "action" not in body
    assert f"model {model} cannot decide" in body["error"] and "policy.abstain cannot be 'act'" in body["error"]
    assert "`rlcd train`" in body["error"] and "`rlcd rollback`" in body["error"]
    r = c.post("/v1/decide", json={"state": STATE, "heads": ["color"], "stakes": "low"})
    assert r.status_code == 422 and r.json()["kind"] == "bad_bundle"
    ready = c.get("/ready")
    assert ready.status_code == 503 and "cannot decide" in ready.json()["reasons"][0]
    assert c.get("/health").json()["ready"] is False
    # answers (no tier, no abstain name) are still given: only decisions are refused
    assert c.post("/v1/systemone", json={"state": STATE, "questions": {
        "q": {"type": "choice", "head": "color"}}}).status_code == 200
    # a retrain under a sound rlcd.yaml serves again
    assert toy_train(svc) != model
    assert c.post("/v1/decide", json={"state": STATE, "heads": ["color"]}).status_code == 200
    assert c.get("/ready").status_code == 200


# ----------------------------------------------------------------------------- D. the validator

def check(tmp_path, data: dict, pack=None) -> list[dict]:
    return CC.check_project(Project(root=tmp_path, data=data), pack)


def at(findings: list[dict], path: str, level: str = "error") -> str:
    hits = [f for f in findings if f["path"] == path]
    assert len(hits) == 1 and hits[0]["level"] == level, (path, findings)
    return hits[0]["message"]


def test_overrides_must_name_a_head_and_a_class_of_that_head(tmp_path):
    th = {"*": {"act": 0.9}}
    f = check(tmp_path, {"policy": {"overrides": {"colour": {"red": th}, "color": {"redd": th, "blue": th},
                                                  "good": {True: th}}}}, ToyPack())
    assert len(f) == 2
    msg = at(f, "policy.overrides.colour")
    assert "is not a head of this project (good, color, level)" in msg and "Did you mean 'color'?" in msg
    msg = at(f, "policy.overrides.color.redd")
    assert "is not a class of head 'color' (red, green, blue)" in msg and "Did you mean 'red'?" in msg
    f = check(tmp_path, {"policy": {"overrides": {"good": {"yes": th}}}}, ToyPack())
    assert "written \"true\" and \"false\"" in at(f, "policy.overrides.good.yes")
    f = check(tmp_path, {"policy": {"overrides": {"good": {False: th}}}}, ToyPack())   # a gate head commits to true
    assert "only ever commit to 'true'" in at(f, "policy.overrides.good.false", "warn")
    # what is valid stays valid, and a head that is switched off keeps its overrides
    assert check(tmp_path, {"policy": {"overrides": {"color": {"red": th, "blue": {"high": {"act": 0.99}}},
                                                     "good": {"true": th}, "level": {"3": th}}}}, ToyPack()) == []
    off = {"pack": "documents", "pack_config": {"encoder": {"kind": "hashing"}},
           "heads": {"needs_graph": {"enabled": False}},
           "policy": {"overrides": {"needs_graph": {True: th}, "answerable": {True: th}, "rout": {"vector": th}}}}
    f = check(tmp_path, off)
    assert [x["path"] for x in CC.errors(f)] == ["policy.overrides.rout"]
    assert "Did you mean 'route'?" in at(f, "policy.overrides.rout")


def test_overrides_are_not_judged_when_the_heads_are_unknown(tmp_path):
    th = {"*": {"act": 0.9}}
    data = {"pack": "tabular", "pack_config": {"data": "not-there-yet.csv", "label": "outcome"},
            "policy": {"overrides": {"outcome": {"true": th}}}}
    assert CC.errors(check(tmp_path, data)) == []                # no table yet: its heads cannot be known
    assert CC.errors(check(tmp_path, {"policy": {"overrides": {"x": {"y": th}}}}, NoHeads())) == []


def test_cli_config_check_reports_override_typos(tmp_path, capsys):
    root = tabular_project(tmp_path / "p")
    th = {"*": {"act": 0.9}}
    edit_yaml(root, lambda d: d["policy"].update(overrides={"outcomee": {"true": th}, "outcome": {"maybe": th}}))
    assert main(["--project", str(root), "config", "check"]) == 1
    out = capsys.readouterr().out
    assert "error  policy.overrides.outcomee: is not a head of this project (outcome)" in out
    assert "Did you mean 'outcome'?" in out and "error  policy.overrides.outcome.maybe: is not a class" in out
    assert "2 errors" in out


def test_a_string_threshold_and_a_holdout_fraction_out_of_range_are_caught_by_the_check(tmp_path, capsys):
    root = tabular_project(tmp_path / "p")
    edit_yaml(root, lambda d: (d["policy"].update(stakes={"medium": {"act": "0.9"}}),
                               d["train"].update(holdout_fraction=1.5)))
    assert main(["--project", str(root), "config", "check"]) == 1
    out = capsys.readouterr().out
    assert "error  policy.stakes.medium.act: must be a number, got '0.9' (write it without quotes)" in out
    assert "error  train.holdout_fraction: must be in [0.05, 0.5], got 1.5" in out
    assert run(root, tmp_path / "home", "train") == 2            # and training refuses to start on it
    err = capsys.readouterr().err
    assert "'<=' not supported" not in err and "development groups" not in err
    assert "has 2 errors, so nothing was started" in err


def test_train_ranges_in_the_file_are_the_ones_the_api_enforces():
    keys = CC.schema()["train"].keys
    for name in ("holdout_fraction", "n_splits", "n_estimators", "feedback_weight"):
        key, bounds = keys[name], {}
        for m in TrainRequest.model_fields[name].metadata:
            for attr in ("ge", "gt", "le", "lt"):
                if getattr(m, attr, None) is not None:
                    bounds[attr] = getattr(m, attr)
        assert key.min == bounds.get("ge", bounds.get("gt")), name
        assert key.max == bounds.get("le", bounds.get("lt")), name
        assert key.open_min == ("gt" in bounds) and key.open_max == ("lt" in bounds), name
        assert key.default == TrainRequest.model_fields[name].default, name
    pattern = next(m.pattern for m in TrainRequest.model_fields["calibration"].metadata if getattr(m, "pattern", None))
    assert set(keys["calibration"].choices) == set(pattern.strip("^$()").split("|"))


@pytest.mark.parametrize("train, ok", [
    ({"holdout_fraction": 0.05}, True), ({"holdout_fraction": 0.5}, True), ({"holdout_fraction": 0.04}, False),
    ({"holdout_fraction": 0.51}, False), ({"n_splits": 20}, True), ({"n_splits": 21}, False),
    ({"n_estimators": 10}, True), ({"n_estimators": 9}, False), ({"n_estimators": 3001}, False),
    ({"feedback_weight": 1000}, True), ({"feedback_weight": 0}, False)])
def test_the_file_and_the_api_accept_the_same_training_values(tmp_path, train, ok):
    from pydantic import ValidationError

    assert (CC.errors(check(tmp_path, {"train": train}, ToyPack())) == []) is ok
    try:
        TrainRequest(**train)
        api_ok = True
    except ValidationError:
        api_ok = False
    assert api_ok is ok


# ----------------------------------------------------------------------------- D. what `rlcd train` prints

def test_train_prints_each_dataset_note_once(biz):
    assert "\nData notes:\n" in biz.out
    block = biz.out.split("Data notes:\n")[1].split("\n\n")[0].splitlines()
    assert all(line.startswith("  outcome: ") for line in block) and len(block) == len(set(block))
    assert any("categorical columns ['plan']" in line for line in block)
    job = {"heads": {"outcome": {"status": "trained"}}}
    assert _notes_block(job) == []                               # no notes: no block


def test_notes_are_deduplicated_grouped_and_capped():
    shared, long = "rows have neither a time nor a group key", "x" * 500
    job = {"heads": {"a": {"notes": [shared, "column  plan was\none-hot encoded", shared]},
                     "b": {"notes": [shared, long]}, "c": {"status": "failed"}}}
    lines = _notes_block(job)
    assert lines[0] == f"  all heads: {shared}"                  # one line for a note both heads carry
    assert lines[1] == "  a: column plan was one-hot encoded"    # whitespace folded
    assert lines[2].startswith("  b: xxx") and lines[2].endswith("…") and len(lines[2]) < 440
    many = {"heads": {"a": {"notes": [f"note {i}" for i in range(30)]}}}
    capped = _notes_block(many, limit=5)
    assert len(capped) == 6 and capped[-1].startswith("  … and 25 more") and "rlcd report" in capped[-1]


def test_a_failed_heads_error_is_said_once_and_fits_the_pack():
    err = ("outcome: 20 trusted (human-verified or ground-truth) rows; at least 30 are needed to calibrate and to "
           "measure anything. Audit more rows (`rlcd label audit`).")
    assert _head_error("outcome", f"outcome: {err}") == err[len("outcome: "):]   # "outcome: outcome: ..." no more
    assert "`rlcd label audit`" in _head_error("outcome", err, labelling=True)   # a pack with a labelling step
    plain = _head_error("outcome", err, labelling=False)
    assert "rlcd label" not in plain and "Audit" not in plain
    assert plain.startswith("20 trusted (human-verified or ground-truth) rows; at least 30 are needed")
    assert plain.endswith("This pack has no labelling step: its labels come from the training data, so add rows "
                          "(or fix the label column) there.")
    two = ("h: all 40 trusted rows carry the label 'no'; calibrating and measuring a head needs trusted examples "
           "of at least two classes. Audit rows of the other classes (`rlcd label audit`).")
    assert "rlcd label" not in _head_error("h", two, False) and "at least two classes." in _head_error("h", two, False)
    none = "h: no labelled rows (label some first: `rlcd label bootstrap` / `rlcd label audit`)"
    assert "rlcd label" not in _head_error("h", none, False)
    assert _head_error("h", none, False).startswith("no labelled rows. This pack has no labelling step")
    assert _head_error("h", "h: something else", False) == "something else"


def test_a_small_tabular_table_fails_with_one_clear_line_per_head(tmp_path, capsys):
    root = tabular_project(tmp_path / "small", n=20)
    capsys.readouterr()
    assert run(root, tmp_path / "home", "train") == 1
    io = capsys.readouterr()
    assert "outcome: outcome:" not in io.out + io.err
    assert "rlcd label" not in io.out + io.err                   # the tabular pack has no labelling step
    row = next(line for line in io.out.splitlines() if line.startswith("outcome"))
    assert "failed" in row and "20 trusted" in row and "This pack has no labelling step" in row
    assert "rlcd: no head could be trained (outcome): nothing was saved." in io.err


def test_a_missing_data_file_prints_its_error_and_no_empty_table(tmp_path, capsys):
    root = tabular_project(tmp_path / "nodata", n=None)
    capsys.readouterr()
    assert run(root, tmp_path / "home", "train") == 1
    io = capsys.readouterr()
    assert io.out == ""                                          # no "model (none saved)" line, no table header
    assert io.err.startswith("rlcd: training did not start: ") and "data.csv" in io.err
    assert len(io.err.strip().splitlines()) == 1
