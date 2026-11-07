"""Regression tests for review findings in the CLI, labeling and bench (offline: fake `claude`, stubbed Hub).

F13  `rlcd label bootstrap` end to end (several judge batches) on a tiny documents project, plus `label audit`
     (scripted stdin) and `label next`.
F16  through the CLI: the judge's "none" for the documents `route` head is stored, not dropped as declined.
F22  `rlcd push --private` checks the visibility of an existing repo and refuses a public one.
F23  `rlcd export` names the categorical columns and warns when raw values are embedded.
F34  `rlcd init` offers only the packs that exist on this branch.
F36  `rlcd init` prints steps that run as written; `label audit` says how many trusted rows a head still needs.
F37  `rlcd bench` times distinct queries with cold caches and expands a real graph entity.
"""
from __future__ import annotations

import io
import json
import stat
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from open_rlcd.cli import main
from open_rlcd.core.config import load_project
from open_rlcd.core.service import Service
from open_rlcd.labeling.store import LabelStore

REPO = Path(__file__).resolve().parent.parent

# A judge for every documents head: reads the questions and the allowed labels from the prompt and answers the
# LAST allowed label of a choice (route -> "none", which F16 used to drop), `true` for yes/no questions and the
# highest level of a score. Logs argv and cwd per call.
FAKE_CLAUDE = r'''#!{python}
import json, os, re, sys
prompt = sys.stdin.read()
with open(os.environ["FAKE_CLAUDE_LOG"], "a") as f:
    f.write(json.dumps({{"argv": sys.argv[1:], "cwd": os.getcwd()}}) + "\n")
system, _, user = prompt.partition("Label the following")
heads = {{}}
for block in system.split('### question id "')[1:]:
    name, body = block.split('"', 1)
    if "yes/no" in body:
        heads[name] = True
    elif "level number" in body:
        heads[name] = len(re.findall(r"^  (\d+) = ", body, re.M)) - 1
    else:
        heads[name] = (re.findall(r'^  - "([^"]+)"', body, re.M) or re.findall(r'^  "([^"]+)"', body, re.M))[-1]
ids = re.findall(r"<id>(.*?)</id>", user)
items = [{{"id": i, "labels": {{h: {{"label": v, "confidence": 0.7}} for h, v in heads.items()}}}} for i in ids]
print(json.dumps({{"type": "result", "subtype": "success", "is_error": False, "total_cost_usd": 0.0,
                  "result": json.dumps({{"items": items}})}}))
'''

DOCS = {
    "warranty.md": "# Warranty\n\nThe Velobox 300 has a two year warranty. Returns are accepted within 30 days of "
                   "purchase. The Imaging team owns the Velobox product line.\n",
    "team.md": "# Imaging team\n\nThe Imaging team is led by Dana Smith. Dana Smith reports to the Hardware group. "
               "The team also maintains the K-200 scanner.\n",
    "scanner.md": "# K-200 scanner\n\nThe K-200 scanner costs 450 euros. Its warranty is one year. Firmware updates "
                  "ship every quarter.\n",
}
QUERIES = ["What is the warranty of the Velobox 300?", "Who leads the Imaging team?",
           "How much does the K-200 scanner cost?", "What's the weather tomorrow?"]


@pytest.fixture
def fake_claude(tmp_path, monkeypatch):
    bindir = tmp_path / "bin"
    bindir.mkdir()
    exe = bindir / "claude"
    exe.write_text(FAKE_CLAUDE.format(python=sys.executable), encoding="utf-8")
    exe.chmod(exe.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    log = tmp_path / "claude-calls.jsonl"
    monkeypatch.setenv("PATH", f"{bindir}:{__import__('os').environ.get('PATH', '')}")
    monkeypatch.setenv("FAKE_CLAUDE_LOG", str(log))
    return log


@pytest.fixture
def docs_project(tmp_path, monkeypatch):
    """A tiny documents project as `rlcd init` writes it, switched to the offline hashing encoder."""
    monkeypatch.delenv("RLCD_HOME", raising=False)
    root = tmp_path / "biz"
    assert main(["init", str(root), "--pack", "documents"]) == 0
    cfg = yaml.safe_load((root / "rlcd.yaml").read_text())
    cfg["pack_config"]["encoder"] = {"kind": "hashing", "dim": 64}
    (root / "rlcd.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")
    (root / "data").mkdir()
    for name, text in DOCS.items():
        (root / "data" / name).write_text(text, encoding="utf-8")
    (root / "queries.jsonl").write_text("".join(json.dumps({"query": q}) + "\n" for q in QUERIES), encoding="utf-8")
    return root


def run(root: Path, *argv: str) -> int:
    return main(["--project", str(root), *argv])


# ----------------------------------------------------------------------------- F13 / F16 / F36: the labeling flow

def test_label_bootstrap_runs_every_batch_end_to_end(docs_project, fake_claude, capsys):
    root = docs_project
    assert run(root, "ingest", "./data") == 0                              # relative to the project
    capsys.readouterr()
    assert run(root, "label", "bootstrap", "--batch", "3") == 0          # used to die after the first batch
    out = capsys.readouterr().out
    res = json.loads(out[out.index("{"):])
    calls = [json.loads(line) for line in fake_claude.read_text().splitlines()]
    assert res["stopped"] is None and res["items_pending"] == 0 and res["batches"] == res["batches_planned"] >= 3
    assert len(calls) == res["batches"]
    assert f"[{res['items_sent']}/{res['items_sent']}]" in out           # the progress line of the last batch
    # every documents head with items got a judge label for every one of its items
    store = LabelStore(load_project(root).home)
    judged = store.judge_labels()
    route = {k: v for k, v in judged.items() if k[1] == "route"}
    assert len(route) == len(QUERIES) and {v["label"] for v in route.values()} == {"none"}   # F16
    assert res["labels"]["route"] == len(QUERIES) and res["problems"] == []
    # F24 through the CLI: the judge ran locked down, outside the project
    for c in calls:
        assert c["argv"][:7] == ["-p", "--output-format", "json", "--tools", "", "--strict-mcp-config",
                                 "--no-session-persistence"]
        assert Path(c["cwd"]).resolve() != root.resolve()
    # resumable: a second run sends nothing
    assert run(root, "label", "bootstrap", "--batch", "3") == 0
    assert len(fake_claude.read_text().splitlines()) == len(calls)


def test_label_audit_and_next_through_the_cli(docs_project, fake_claude, monkeypatch, capsys):
    root = docs_project
    assert run(root, "ingest", "./data") == 0
    assert run(root, "label", "bootstrap", "--heads", "answerable,route") == 0
    capsys.readouterr()
    # Enter accepts the judge, "n" overrules it, then quit
    monkeypatch.setattr("sys.stdin", io.StringIO("\nn\nq\n"))
    assert run(root, "label", "audit", "--heads", "answerable", "--n", "5") == 0
    out = capsys.readouterr().out
    res = json.loads(out[out.index('{\n  "strategy"'):])
    assert res["labelled"] == 2 and res["accepted_judge"] == 1 and res["overruled"] == 1 and res["quit"] is True
    head = res["heads"]["answerable"]
    # F36: how many trusted rows the head still needs, to train and for a 30-row holdout
    assert head["trusted"] == 2 and head["for_verified_holdout"] == 150
    assert head["more_for_verified_holdout"] == 148 and "28 more trusted rows needed" in head["status"]
    assert run(root, "label", "next", "--head", "answerable", "--n", "3") == 0
    out = capsys.readouterr().out
    left = len(LabelStore(load_project(root).home).items("query_evidence")) - 2
    assert left >= 1 and f"queued {min(3, left)} items for `rlcd label audit --queue`" in out


def test_init_next_steps_name_what_each_step_needs(tmp_path, capsys):
    assert main(["init", str(tmp_path / "d"), "--pack", "documents"]) == 0
    out = capsys.readouterr().out
    steps = out[out.index("Next:"):]
    for needed in ("./data", "queries.jsonl", "--extra documents", "hashing", "rlcd ingest ./data",
                   "rlcd label bootstrap", "rlcd label audit", "rlcd train"):
        assert needed in steps, needed
    # the order runs as written: documents and the query log before ingest, ingest before bootstrap
    assert steps.index("queries.jsonl") < steps.index("rlcd ingest") < steps.index("rlcd label bootstrap") \
        < steps.index("rlcd label audit") < steps.index("rlcd train")
    assert main(["init", str(tmp_path / "t"), "--pack", "tabular"]) == 0
    steps = capsys.readouterr().out
    assert "data.csv" in steps and "pack_config.label" in steps and "rlcd train" in steps


def test_ingest_help_does_not_advertise_unimplemented_connectors(capsys):
    with pytest.raises(SystemExit):
        main(["ingest", "--help"])
    out = capsys.readouterr().out
    assert "--qdrant" not in out and "--neo4j" not in out and "--csv" not in out


# ----------------------------------------------------------------------------- F22: push visibility

class FakeHub:
    """Stands in for huggingface_hub.HfApi: `repos` maps repo id -> private flag (absent = no such repo)."""

    def __init__(self, repos: dict[str, bool], create_makes: bool | None = None):
        self.repos, self.create_makes = dict(repos), create_makes
        self.created: list[tuple[str, bool]] = []
        self.uploaded: list[str] = []

    def repo_info(self, repo_id, **kw):
        if repo_id not in self.repos:
            import httpx2
            from huggingface_hub.errors import RepositoryNotFoundError

            req = httpx2.Request("GET", f"http://127.0.0.1/api/models/{repo_id}")
            raise RepositoryNotFoundError("404", response=httpx2.Response(404, request=req))
        return SimpleNamespace(id=repo_id, private=self.repos[repo_id])

    def create_repo(self, repo_id, private=False, exist_ok=False, **kw):
        self.created.append((repo_id, private))
        if repo_id not in self.repos:
            self.repos[repo_id] = private if self.create_makes is None else self.create_makes

    def upload_folder(self, folder_path, repo_id, **kw):
        self.uploaded.append(repo_id)


@pytest.fixture
def pushable(tmp_path, monkeypatch):
    pytest.importorskip("huggingface_hub")
    bundle = REPO / "bundles" / "demo-documents"
    if not (bundle / "model.json").is_file():
        pytest.skip("demo bundle not built")
    monkeypatch.delenv("RLCD_HOME", raising=False)
    home = tmp_path / "home"
    assert main(["--home", str(home), "import", str(bundle)]) == 0
    return home


def _hub(monkeypatch, hub: FakeHub) -> None:
    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "HfApi", lambda *a, **k: hub)


def test_push_private_refuses_an_existing_public_repo(pushable, monkeypatch, capsys):
    hub = FakeHub({"someone/public-repo": False})
    _hub(monkeypatch, hub)
    assert main(["--home", str(pushable), "push", "--hf", "someone/public-repo", "--private", "--yes"]) == 1
    assert hub.uploaded == [] and hub.created == []
    assert "already exists and is PUBLIC" in capsys.readouterr().err


def test_push_private_to_a_new_repo_creates_it_private_and_verifies(pushable, monkeypatch, capsys):
    hub = FakeHub({})
    _hub(monkeypatch, hub)
    asked: list[str] = []
    monkeypatch.setattr("builtins.input", lambda prompt="": asked.append(prompt) or "y")
    assert main(["--home", str(pushable), "push", "--hf", "me/new-repo", "--private"]) == 0
    assert hub.created == [("me/new-repo", True)] and hub.uploaded == ["me/new-repo"]
    assert "new repo, created private" in asked[0]
    # the Hub ignored `private` (or someone created it public in between): nothing is uploaded
    hub = FakeHub({}, create_makes=False)
    _hub(monkeypatch, hub)
    assert main(["--home", str(pushable), "push", "--hf", "me/racy", "--private", "--yes"]) == 1
    assert hub.uploaded == [] and "nothing was uploaded" in capsys.readouterr().err


def test_push_prompt_shows_the_actual_visibility(pushable, monkeypatch):
    hub = FakeHub({"me/open": False, "me/closed": True})
    _hub(monkeypatch, hub)
    asked: list[str] = []
    monkeypatch.setattr("builtins.input", lambda prompt="": asked.append(prompt) or "n")
    assert main(["--home", str(pushable), "push", "--hf", "me/open"]) == 1
    assert main(["--home", str(pushable), "push", "--hf", "me/closed"]) == 1
    assert "existing PUBLIC repo" in asked[0] and "existing private repo" in asked[1]
    assert hub.uploaded == []


def test_push_private_refuses_when_visibility_cannot_be_checked(pushable, monkeypatch, capsys):
    hub = FakeHub({})

    def broken(repo_id, **kw):
        raise ConnectionError("hub unreachable")

    hub.repo_info = broken
    _hub(monkeypatch, hub)
    assert main(["--home", str(pushable), "push", "--hf", "me/x", "--private", "--yes"]) == 1
    assert hub.uploaded == [] and "cannot verify" in capsys.readouterr().err


# ----------------------------------------------------------------------------- F23: export message

def _tabular_project(tmp_path: Path, hashed: bool) -> Path:
    from open_rlcd.packs.tabular import synthetic_pack_config, write_synthetic_csv

    root = tmp_path / ("hashed" if hashed else "raw")
    root.mkdir()
    data = root / "churn.csv"
    write_synthetic_csv(data, n=500, seed=2)
    pc = {**synthetic_pack_config(data), "hash_categories": hashed}
    cfg = {"name": root.name, "pack": "tabular", "home": "./var", "pack_config": pc,
           "heads": {"plan_tier_next": {"enabled": False}, "satisfaction": {"enabled": False},
                     "churned": {"model": "logreg"}}}
    (root / "rlcd.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return root


@pytest.mark.parametrize("hashed", [False, True])
def test_export_names_categorical_columns_and_warns_about_raw_values(tmp_path, monkeypatch, capsys, hashed):
    monkeypatch.delenv("RLCD_HOME", raising=False)
    root = _tabular_project(tmp_path, hashed)
    assert run(root, "train") == 0
    capsys.readouterr()
    assert run(root, "export", str(tmp_path / "bundle")) == 0
    cap = capsys.readouterr()
    assert "contains no training rows" in cap.out and "contains no training data" not in cap.out
    assert "categorical columns stored in the bundle: plan (" in cap.out
    if hashed:
        assert "salted sha256" in cap.out and "raw" not in cap.out and cap.err == ""
    else:
        assert "plan (3 values, raw)" in cap.out
        assert "warning:" in cap.err and "hash_categories: true" in cap.err
    card = (tmp_path / "bundle" / "MODEL_CARD.md").read_text()
    assert "Categorical feature columns" in card and ("salted SHA-256" in card) == hashed
    assert ("Raw training values inside" in card) == (not hashed)


# ----------------------------------------------------------------------------- F37: bench

def test_bench_times_distinct_cold_queries_and_a_real_graph_entity(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("RLCD_HOME", raising=False)
    root = tmp_path / "demo"
    assert main(["demo", "--dir", str(root), "--queries", "300", "--seed", "5"]) == 0
    svc = Service(load_project(root))
    prov = svc.provider(svc.resolve(None))
    cache_rows = len(prov.encoder)
    capsys.readouterr()
    assert run(root, "bench", "--n", "40", "--json") == 0
    res = json.loads(capsys.readouterr().out)
    st = res["stages"]
    e2e = st["decide.end_to_end"]
    assert e2e["distinct_states"] > 1 and e2e["cache"].startswith("cold")
    assert "embedding cache" in e2e["cache"] and "feature cache" in e2e["cache"]
    assert st["decide.repeated_query"]["cache"].startswith("warm")
    g = st["graph.expand"]
    assert g["results"] > 0 and g["seed"]                     # a real entity, not a miss
    svc2 = Service(load_project(root))
    prov2 = svc2.provider(svc2.resolve(None))
    assert len(prov2.encoder) == cache_rows                    # bench queries never land in the user's cache
    assert run(root, "bench", "--n", "20") == 0
    table = capsys.readouterr().out
    assert "decide.end_to_end: cold" in table and "graph.expand: from the entity" in table


def test_tabular_template_trains_as_written(tmp_path, monkeypatch, capsys):
    """F36: `rlcd init --pack tabular`, a table at ./data.csv with an `outcome` column, `rlcd train`: no edits."""
    from open_rlcd.packs.tabular import write_synthetic_csv

    monkeypatch.delenv("RLCD_HOME", raising=False)
    root = tmp_path / "t"
    assert main(["init", str(root), "--pack", "tabular"]) == 0
    assert "hash_categories" in (root / "rlcd.yaml").read_text()
    src = tmp_path / "churn.csv"
    write_synthetic_csv(src, n=600, seed=1)
    lines = src.read_text().splitlines()
    header = lines[0].split(",")
    keep = [i for i, h in enumerate(header) if h not in ("plan_tier_next", "satisfaction", "row_id", "customer_id",
                                                         "month", "channel")]
    rows = [",".join(("outcome" if header[i] == "churned" else c) if n == 0 else c
                     for i, c in enumerate(line.split(",")) if i in keep) for n, line in enumerate(lines)]
    (root / "data.csv").write_text("\n".join(rows) + "\n")
    assert run(root, "train") == 0
    assert "outcome" in capsys.readouterr().out
