"""The `rlcd` command end to end on a small synthetic documents project (offline, hashing encoder)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from open_rlcd.cli import _maybe_demo, main
from open_rlcd.core.bundle import validate
from open_rlcd.core.config import Project
from open_rlcd.core.service import Service

REPO = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def demo(tmp_path_factory):
    root = tmp_path_factory.mktemp("cli") / "demo"
    mp = pytest.MonkeyPatch()
    mp.delenv("RLCD_HOME", raising=False)
    assert main(["demo", "--dir", str(root), "--queries", "400", "--seed", "3"]) == 0
    yield root, mp
    mp.undo()


def run(root: Path, *argv: str) -> int:
    return main(["--project", str(root), *argv])


def test_init_writes_a_project_per_pack(tmp_path):
    for pack in ("documents", "tabular"):
        assert main(["init", str(tmp_path / pack), "--pack", pack]) == 0
        cfg = yaml.safe_load((tmp_path / pack / "rlcd.yaml").read_text())
        assert cfg["pack"] == pack and cfg["home"] == "./var" and cfg["security"]["allow_remote"] is False
        assert "var/" in (tmp_path / pack / ".gitignore").read_text()
    assert main(["init", str(tmp_path / "documents")]) == 1          # refuses to overwrite without --force
    with pytest.raises(SystemExit):                                   # F34: no trading pack on this branch
        main(["init", str(tmp_path / "trading"), "--pack", "trading"])
    assert not (tmp_path / "trading" / "rlcd.yaml").exists()


def test_demo_trains_and_report_writes_card_and_page(demo, capsys):
    root, _ = demo
    assert run(root, "report", "--no-open") == 0
    out = capsys.readouterr().out
    card = Path(out.split("wrote ")[1].split("\n")[0])
    page = Path(out.split("wrote ")[2].strip())
    text = card.read_text()
    assert "synthetic data only" in text.lower() and "What it is not" in text and "## Heads" in text
    html = page.read_text()
    assert html.startswith("<!doctype html>") and "<svg" in html and "http" not in html.split("<style>")[0]
    assert any(p.name.startswith("calibration_") for p in card.parent.iterdir())


def test_decide_and_calibration_commands_print_json(demo, capsys):
    root, _ = demo
    capsys.readouterr()
    assert run(root, "decide", "--query", "What is the warranty of the Velobox 300?") == 0
    d = json.loads(capsys.readouterr().out)
    assert d["decision_id"] and d["action"] in ("act", "confirm", "escalate") and "plan" in d
    assert run(root, "calibration", "chunk_relevant") == 0
    c = json.loads(capsys.readouterr().out)
    assert "skill" in c and "verdict" in c


def test_bench_reports_every_stage(demo, capsys):
    root, _ = demo
    capsys.readouterr()
    assert run(root, "bench", "--n", "20", "--json") == 0
    res = json.loads(capsys.readouterr().out)
    stages = res["stages"]
    assert "encoder.query" in stages and "decide.end_to_end" in stages
    assert any(k.startswith("head.") for k in stages) and "ann.search" in stages
    for v in stages.values():
        assert 0 <= v["p50_ms"] <= v["p95_ms"] <= v["p99_ms"]


def test_export_and_import_round_trip(demo, tmp_path, capsys):
    root, _ = demo
    dest = tmp_path / "bundle"
    assert run(root, "export", str(dest)) == 0
    meta = validate(dest)
    assert (dest / "MODEL_CARD.md").is_file() and meta["data_card"]["synthetic_only"] is True
    other = tmp_path / "other"
    assert main(["--home", str(other), "import", str(dest)]) == 0
    assert "imported" in capsys.readouterr().out


def test_label_status_counts(demo, capsys):
    root, _ = demo
    capsys.readouterr()
    assert run(root, "label", "status") == 0
    counts = json.loads(capsys.readouterr().out)
    assert counts["items"] > 0 and "response_type" in counts["labels"]


def test_serve_right_after_clone_imports_the_demo_bundle(tmp_path, monkeypatch):
    """No rlcd.yaml and no trained model: `rlcd serve` imports the shipped demo bundle (synthetic data only)."""
    if not (REPO / "bundles" / "demo-documents" / "model.json").is_file():
        pytest.skip("demo bundle not built")
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("RLCD_HOME", raising=False)
    svc = Service(Project(root=tmp_path), home=tmp_path / "var")
    _maybe_demo(svc)
    assert svc.registry.registry()["versions"]
    v = svc.registry.latest_id()
    assert svc.bundle(v).meta["data_card"]["synthetic_only"] is True
    _maybe_demo(svc)                                       # idempotent: a second start imports nothing
    assert len(svc.registry.registry()["versions"]) == 1


def test_errors_are_one_readable_line(tmp_path, capsys, monkeypatch):
    monkeypatch.delenv("RLCD_HOME", raising=False)
    assert main(["--project", str(tmp_path / "missing"), "train"]) == 1
    assert capsys.readouterr().err.startswith("rlcd: ")
