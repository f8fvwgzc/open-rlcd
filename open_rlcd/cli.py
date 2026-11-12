"""`rlcd` — the command teammates use.

    rlcd init my-business --pack documents     write my-business/rlcd.yaml (packs: documents, tabular, trading)
    rlcd config check [--json]                 rlcd.yaml checked key by key (exit 1 on an error); docs/configuration.md
    rlcd config show [--json]                  what the project resolves to: home, pack, heads, thresholds, rollout
    rlcd ingest ./data                         build the corpus (documents pack; tabular has no ingest step)
    rlcd ingest ./data --qdrant http://localhost:6333 --collection docs --neo4j bolt://localhost:7687
    rlcd label bootstrap --judge claude-cli    weak labels from an LLM judge (documents: needs queries.jsonl)
    rlcd label audit --heads answerable        human review in the terminal (~150+ trusted rows per head)
    rlcd label next --head answerable          the most informative rows to audit next
    rlcd train                                 every head; prints the calibration summary
    rlcd report                                MODEL_CARD.md + report.html (+ per-head JSON)
    rlcd serve --port 8095                     the HTTP API
    rlcd bench                                 p50 / p95 / p99 per stage on this machine
    rlcd export ./bundle | rlcd import ./bundle | rlcd push --hf user/repo
    rlcd demo                                  synthetic documents project, trained, in ./rlcd-demo

  the control layer (docs/control-layer.md)
    rlcd ask --query "..." --heads a,b         one System One call against the local model (or --battery NAME)
    rlcd batteries                             the versioned question sets and whether their heads are trained
    rlcd receipts verify | show ID | check ID --state s.json | confirm ID --by NAME | prune --before DATE
    rlcd automation status | off --reason ... | on   the run-time kill switch (caps every decision at confirm)
    rlcd promote VERSION | rlcd rollback [--to VERSION]   point rlcd-latest at another version, no retrain
    rlcd metrics [--since DATE] [--model V]    the operations report: action mix, outcomes, drift, review load
    rlcd review [--limit 50]                   decisions waiting for a person (confirm / abstain without feedback)
    rlcd replay [--base A] [--candidate B]     logged decisions re-run under two versions: classes, tiers, bands
    rlcd regress [--model V]                   run the regression cases (exit 1 when one fails or cannot be judged)
    rlcd case add --from-decision ID --head H --expect class=X   turn a logged decision into a regression case
    rlcd check [--model V] [--json]            the pre-launch checklist (exit 1 when an item fails)
"""
from __future__ import annotations

import argparse
import ipaddress
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

from . import VERSION

TEMPLATES: dict[str, dict] = {
    "documents": {
        "pack_config": {
            "encoder": {"kind": "onnx", "model_id": "intfloat/multilingual-e5-small"},
            "corpus": "./data",
            "queries": "./queries.jsonl",
            "chunking": {"max_chars": 1200, "overlap": 150},
            "vector": {"kind": "local"},
            "graph": {"kind": "local"},
        },
        "heads": {"answerable": {"stakes": "medium", "conformal": 0.05}, "grounded": {"stakes": "high"}},
    },
    "tabular": {
        "pack_config": {"data": "./data.csv", "label": "outcome", "id": None, "group": None, "time": None,
                        "features": "auto", "exclude": [], "text": [], "categorical": "auto",
                        "hash_categories": False},
        "policy": {"abstain": "hold"},
    },
    "trading": {
        "pack_config": {"quant_url": "http://127.0.0.1:8090", "symbols": ["XAUUSD", "EURUSD"], "intervals": ["15m"],
                        "sl_pips": 20, "tp_pips": 50, "tp2_pips": 100, "horizon": 48},
        "policy": {"abstain": "hold"},
    },
}

HEADER = """# Open RLCD project configuration. Every key: docs/configuration.md. Check it with `rlcd config check`.
# Docs: docs/teammate-quickstart.md (documents), docs/tabular-quickstart.md (tables), docs/control-layer.md
# Secrets (ANTHROPIC_API_KEY, RLCD_API_KEY, database passwords) come from the environment, never from this file.
# rollout.mode: shadow | assist | limited | full. Before automating anything, run `rlcd check`.
"""


# Commands that work on a project's own files (its data, its labels, the heads rlcd.yaml configures): without an
# rlcd.yaml they used to fall back to a default documents project in the current folder, create ./var there and
# download an encoder before failing. They stop before anything is read, written or downloaded.
NEEDS_PROJECT = frozenset({"ingest", "label", "train"})
# Commands that read or write a model home (registry, logs, ledger). They run in a project, or against a home
# that was named (--home, RLCD_HOME) or already exists in ./var (what `rlcd serve` leaves after a clone); they
# no longer invent an empty ./var in whatever folder they are started from.
NEEDS_HOME = frozenset({"report", "bench", "export", "import", "push", "decide", "calibration", "ask", "batteries",
                        "receipts", "automation", "promote", "rollback", "metrics", "review", "replay", "regress",
                        "case", "check"})
# Everything else runs anywhere: `init` and `demo` create their project, `config` says there is no file, and
# `serve` without a project deliberately serves the bundled demo model (`_maybe_demo`; the Docker image relies on it).


def _misplaced(args: argparse.Namespace) -> str | None:
    """Why `args.cmd` cannot run from here (one line for stderr, exit 2), or None. Reads nothing but the
    environment and whether files exist; `--project` pointing at a missing file is `load_project`'s error."""
    cmd = args.cmd
    if (cmd not in NEEDS_PROJECT and cmd not in NEEDS_HOME) or args.project:
        return None
    from .core.config import find_config

    if find_config() is not None:      # raises when RLCD_PROJECT points at nothing
        return None
    if cmd in NEEDS_PROJECT:
        return (f"no rlcd.yaml here or in any parent folder, and `rlcd {cmd}` works on a project. Create one with "
                f"`rlcd init <name> --pack documents|tabular|trading` and run it there (or pass --project DIR, or "
                f"set RLCD_PROJECT). Nothing was read, written or downloaded.")
    if getattr(args, "home", None) or os.environ.get("RLCD_HOME") or (Path.cwd() / "var" / "registry.json").is_file():
        return None
    return (f"no rlcd.yaml here or in any parent folder, and no model home to work on: run `rlcd {cmd}` inside a "
            f"project (`rlcd init <name> --pack documents|tabular|trading`), or name the home with --home DIR or "
            f"RLCD_HOME. Nothing was written.")


def _project(args: argparse.Namespace):
    from .core.config import load_project

    project = load_project(args.project)
    if getattr(args, "home", None):   # explicit for this command only (never written into the environment)
        project.home_override = Path(args.home).expanduser().resolve()
    return project


def _service(args: argparse.Namespace):
    from .core.service import Service

    return Service(_project(args))


def _print(obj: Any) -> None:
    print(json.dumps(obj, indent=2, ensure_ascii=False, default=str))


def _policy_note(svc: Any, model: str | None = None) -> None:
    """One stderr line when rlcd.yaml's policy (or a head's stakes) is not what the served version uses: a version
    serves the policy stored in its bundle until `rlcd train` verifies the new one (`rlcd check`,
    thresholds.config, reports the same text). Never fails the command."""
    try:
        from .core.checklist import policy_drift

        version = svc.resolve(model)
        drift = policy_drift(svc, version)
    except Exception:       # an unknown model, a pack that cannot load: the command itself says so
        return
    if drift:
        print(f"note: rlcd.yaml differs from what {version} serves ({drift}): {version} keeps the policy it was "
              f"trained and verified with; the edit takes effect at the next `rlcd train`.", file=sys.stderr)


def _config_ok(svc: Any, block: bool) -> bool:
    """Print what `rlcd config check` finds in the project's rlcd.yaml to stderr before a command runs on it.
    False when `block` and the file has an error (the caller exits 2). Without a rlcd.yaml there is nothing to
    check: the default project always runs."""
    project = svc.project
    if project.path is None:
        return True
    from .core import config_check as CC

    try:
        pack = svc.pack
    except Exception:      # the check reports a pack that cannot be loaded as a warning; the command says the rest
        pack = None
    findings = CC.check_project(project, pack)
    for line in CC.format_findings(findings):
        print(f"rlcd.yaml: {line}", file=sys.stderr)
    n = len(CC.errors(findings))
    if block and n:
        print(f"rlcd: {project.path} has {n} error{'' if n == 1 else 's'}, so nothing was started. Fix "
              f"{'it' if n == 1 else 'them'} and run `rlcd config check`.", file=sys.stderr)
        return False
    return True


# ----------------------------------------------------------------------------- commands

def cmd_init(args: argparse.Namespace) -> int:
    from .core.config import write_config

    root = Path(args.name).expanduser()
    if (root / "rlcd.yaml").exists() and not args.force:
        print(f"{root / 'rlcd.yaml'} already exists (use --force to overwrite)", file=sys.stderr)
        return 1
    root.mkdir(parents=True, exist_ok=True)
    data = {"name": root.name, "pack": args.pack, "home": "./var", **json.loads(json.dumps(TEMPLATES[args.pack])),
            "train": {"holdout_fraction": 0.2, "n_splits": 5, "ordering": "auto", "calibration": "auto",
                      "feedback_weight": 5},
            "labeling": {"judge": "claude-cli", "batch": 20},
            "serve": {"host": "127.0.0.1", "port": 8095, "log_states": True, "redact": True,
                      "receipts": "decisions"},
            # shadow -> assist -> limited -> full as the evidence earns it (docs/control-layer.md)
            "rollout": {"mode": "full"},
            "security": {"allow_remote": False}}
    write_config(root / "rlcd.yaml", data, HEADER)
    if args.pack == "tabular":   # comments the YAML writer cannot produce: outcomes, `exclude`, `targets`
        from .packs.tabular.scaffold import annotate

        (root / "rlcd.yaml").write_text(annotate((root / "rlcd.yaml").read_text(encoding="utf-8")), encoding="utf-8")
    gi = root / ".gitignore"
    if not gi.exists():
        gi.write_text("var/\n.env\n*.npz\n", encoding="utf-8")
    print(f"wrote {root / 'rlcd.yaml'} (pack: {args.pack}). Next:\n" + next_steps(args.pack, root))
    return 0


def next_steps(pack: str, root: Path) -> str:
    """The steps after `rlcd init`, in an order that runs as written (each step has what it needs)."""
    if pack == "documents":
        steps = [
            f"cd {root}",
            "put your documents in ./data (.md .txt .html .pdf .json .jsonl)",
            "write ./queries.jsonl: real questions, one {\"query\": \"...\", \"answer\": \"...\", \"evidence\": [...]} "
            "per line (bootstrap labels these; grounded needs the answer your LLM drafted, evidence what it was shown)",
            "the encoder in rlcd.yaml is ONNX: it needs the documents extra (`uv sync --extra documents`) and "
            "downloads ~135 MB on the first ingest; offline, set pack_config.encoder to {kind: hashing, dim: 256}",
            "rlcd ingest ./data",
            "rlcd label bootstrap        # the default judge (claude-cli) sends item text to Anthropic: opt in with "
            "RLCD_ALLOW_REMOTE=1, or stay local with --judge ollama:<model>",
            "rlcd label audit --heads answerable,grounded --n 600   # your labels calibrate the model",
            "rlcd train",
        ]
    elif pack == "trading":
        steps = [
            f"cd {root}",
            "start the View Engine quant sidecar at pack_config.quant_url (http://127.0.0.1:8090); a sidecar on "
            "another host needs security.allow_remote: true or RLCD_ALLOW_REMOTE=1",
            "edit rlcd.yaml: pack_config.symbols and intervals (one setup:<interval> head per interval); sl_pips, "
            "tp_pips, tp2_pips and horizon go to the sidecar, which computes the labels",
            "rlcd config check",
            "rlcd train                  # reads /dataset from the sidecar; each decision reads /features",
        ]
    else:
        from .packs.tabular.scaffold import next_steps as tabular_steps

        steps = tabular_steps(root)
    return "\n".join(f"  {i}. {st}" for i, st in enumerate(steps, 1))


def cmd_ingest(args: argparse.Namespace) -> int:
    svc = _service(args)
    opts = {"path": args.path, "csv": args.csv, "qdrant": args.qdrant, "collection": args.collection,
            "neo4j": args.neo4j, "max_chars": args.max_chars}
    res = svc.pack.ingest(svc.project, {k: v for k, v in opts.items() if v is not None})
    note = res.pop("note", None) if isinstance(res, dict) else None
    _print(res)
    if note:
        print(f"\nnote: {note}", file=sys.stderr)
    return 0


def _bootstrap_progress(done: int, total: int, message: str) -> None:
    """`bootstrap(progress=...)` calls this after every judge batch: (items done, items in this run, message)."""
    print(f"[{done}/{total}] {message}", flush=True)


def cmd_label(args: argparse.Namespace) -> int:
    svc = _service(args)
    project, pack = svc.project, svc.pack
    heads = args.heads.split(",") if getattr(args, "heads", None) else None
    if args.label_cmd == "bootstrap":
        from .labeling.bootstrap import bootstrap
        from .labeling.judges import make_judge

        # no --judge: make_judge reads labeling.judge and labeling.model itself (an explicit spec would hide the model)
        judge = make_judge(args.judge, project)
        provider = pack.provider(project, None)
        res = bootstrap(project, pack, provider, judge, heads=heads, limit=args.limit,
                        batch=args.batch or int(project.get("labeling.batch", 20)), progress=_bootstrap_progress)
        _print(res)
    elif args.label_cmd == "audit":
        from .labeling.audit import audit

        res = audit(project, pack, heads=heads, n=args.n, strategy="queue" if args.queue else "stratified",
                    annotator=args.annotator)
        _print(res)
    elif args.label_cmd == "next":
        from .labeling.active import next_batch

        if not args.head:
            print("--head is required", file=sys.stderr)
            return 2
        rows = next_batch(project, svc, args.head, n=args.n)
        print(f"queued {len(rows)} items for `rlcd label audit --queue`")
        for r in rows[:10]:
            print(f"  {r.get('item_id')}  score {r.get('score', 0):.3f}  {r.get('why', '')}")
    elif args.label_cmd == "status":
        from .labeling.store import LabelStore

        _print(LabelStore(svc.home).counts())
    return 0


def _summary_table(job: dict, labelling: bool = True) -> str:
    rows = [f"{'head':<18}{'status':<10}{'holdout':>8}{'logloss':>9}{'base':>8}{'BSS':>8}{'ECE':>7}  "
            f"{'skill':<6}verdict(medium)"]
    for name, h in job["heads"].items():
        if h.get("status") != "trained":
            rows.append(f"{name:<18}{h.get('status', '?'):<10}  {_head_error(name, h.get('error') or '', labelling)}")
            continue
        m = h["metrics"]

        def f(x, nd=3):
            return "–" if x is None else f"{x:.{nd}f}"
        v = (h.get("verdict") or {}).get("medium", {})
        rows.append(f"{name:<18}{'trained':<10}{m.get('n_holdout') or 0:>8}{f(m.get('log_loss')):>9}"
                    f"{f(m.get('baseline_log_loss')):>8}{f(m.get('brier_skill_score')):>8}{f(m.get('ece')):>7}  "
                    f"{'yes' if m.get('has_skill') else 'NO':<6}{v.get('max_tier', '–')}")
    return "\n".join(rows)


def _head_error(name: str, error: str, labelling: bool = True) -> str:
    """A failed head's error for the terminal: without the head's name in front (the line already starts with
    it), and, for a pack with no labelling step, without advice to run `rlcd label ...` (the labels of such a
    pack are a column of its data)."""
    text = str(error)
    while text.startswith(f"{name}: "):
        text = text[len(name) + 2:]
    if not labelling and "`rlcd label" in text:
        text = re.sub(r"(?:(?<=\. )|^)[^.]*\(`rlcd label[^)]*\)\.?", "", text)       # "Audit more rows (`...`)."
        text = re.sub(r"\s*\([^()]*`rlcd label[^()]*\)", "", text)                   # "(label some first: ...)"
        text = text.rstrip()
        text += ("" if text.endswith((".", "!", "?")) else ".") + (" This pack has no labelling step: its labels come from the training data, so add "
                                "rows (or fix the label column) there.")
    return text


def _notes_block(job: dict, limit: int = 12, width: int = 420) -> list[str]:
    """What the datasets said while they were built (the same notes MODEL_CARD.md lists), once each: rows that
    only train, columns that were encoded or left out, how a class was chosen. A note several heads share is
    printed once with their names; long ones are cut, and past `limit` the rest is counted."""
    by_note: dict[str, list[str]] = {}
    for name, h in job["heads"].items():
        for note in h.get("notes") or []:
            heads = by_note.setdefault(" ".join(str(note).split()), [])
            if name not in heads:
                heads.append(name)
    lines = []
    everyone = len([h for h in job["heads"].values() if h.get("notes")])
    for note, heads in list(by_note.items())[:limit]:
        who = "all heads" if everyone > 1 and len(heads) == everyone else ", ".join(heads)
        lines.append(f"  {who}: {note if len(note) <= width else note[:width - 1].rstrip() + '…'}")
    if len(by_note) > limit:
        lines.append(f"  … and {len(by_note) - limit} more (MODEL_CARD.md lists every note: `rlcd report`)")
    return lines


def _gate_block(svc: Any, job: dict) -> list[str]:
    """Per head whose holdout verdict allows act: what the Bernoulli gate will do to live decisions (which
    cells already pass on holdout evidence, how many correct outcomes the others need), computed by the gate."""
    from .core.checklist import gate_lines

    lines: list[str] = []
    if not job.get("model"):
        return lines
    for name, h in job["heads"].items():
        if h.get("status") != "trained":
            continue
        try:
            got = gate_lines(svc.gate_outlook(job["model"], name))
        except Exception:      # a reading aid: never fail a finished training over it
            continue
        lines += [f"  {name}  {line}" for line in got]
    return lines


def cmd_train(args: argparse.Namespace) -> int:
    from .core.contract import TrainRequest

    svc = _service(args)
    if not _config_ok(svc, block=True):
        return 2
    body: dict = {}
    if args.heads:
        body["heads"] = args.heads.split(",")
    for k in ("calibration", "seed", "holdout_fraction", "n_splits", "feedback_weight"):
        v = getattr(args, k, None)
        if v is not None:
            body[k] = v
    job = svc.trainer.start(TrainRequest(**body), wait=True)
    attempted = [h for h in job["heads"].values() if h.get("status") in ("trained", "failed")]
    if not attempted:      # it stopped before any head (a data file that is missing, a provider that cannot start)
        print(f"rlcd: training did not start: {job.get('error') or 'there is no head to train'}", file=sys.stderr)
        return 1
    try:
        labelling = bool(svc.pack.supports("label_tasks"))     # no labelling step: no advice to run `rlcd label`
    except Exception:
        labelling = True
    print(f"model {job.get('model') or '(none saved)'} — job {job['status']}")
    print(_summary_table(job, labelling))
    for name, h in job["heads"].items():
        if h.get("status") == "trained" and not (h.get("metrics") or {}).get("has_skill"):
            print(f"\n{name}: no skill over the base rate on the holdout — it will abstain. "
                  f"{(h.get('verdict') or {}).get('medium', {}).get('reason', '')}")
    notes = _notes_block(job)
    if notes:
        print("\nData notes:\n" + "\n".join(notes))
    gate = _gate_block(svc, job)
    if gate:
        print("\nLive decisions (verdict act): `act` also needs the Bernoulli gate, which counts right and wrong per "
              "(class, probability bin), starting from the holdout rows.\n" + "\n".join(gate))
    failed = [n for n, h in job["heads"].items() if h.get("status") == "failed"]
    if failed and not job.get("model"):     # the table above says why, head by head
        print(f"\nrlcd: no head could be trained ({', '.join(failed)}): nothing was saved.", file=sys.stderr)
        return 1
    if failed:
        print(f"\nnot trained: {', '.join(failed)} (see the table); the other heads were saved.", file=sys.stderr)
    print("\nNext: `rlcd report` for the reliability page, `rlcd serve` for the API.")
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    from .report.html import write_report
    from .report.model_card import write_model_card

    svc = _service(args)
    version = svc.resolve(args.model)
    b = svc.bundle(version)
    if b is None:
        print("no trained model yet: run `rlcd train` first", file=sys.stderr)
        return 1
    out = Path(args.out).expanduser() if args.out else svc.home / "reports" / version
    out.mkdir(parents=True, exist_ok=True)
    for name in b.names():
        (out / f"calibration_{name.replace(':', '_')}.json").write_text(
            json.dumps(svc.calibration_view(name, version), indent=1, default=str), encoding="utf-8")
    card = write_model_card(b, out / "MODEL_CARD.md")
    write_model_card(b)            # also inside the bundle, so `rlcd export` carries it
    page = write_report(b, out / "report.html")
    print(f"wrote {card}\nwrote {page}")
    if not args.no_open:
        import webbrowser

        webbrowser.open(page.resolve().as_uri())
    return 0


def _maybe_demo(svc: Any) -> None:
    """`rlcd serve` right after clone: with no trained model and no project, import the shipped demo bundle."""
    if svc.registry.registry()["versions"] or svc.project.path is not None:
        return
    for cand in (Path(__file__).resolve().parent / "_demo" / "demo-documents",
                 Path(__file__).resolve().parent.parent / "bundles" / "demo-documents"):
        if (cand / "model.json").is_file():
            vid = svc.registry.import_(cand)
            print(f"no project and no trained model here: imported the demo bundle (synthetic data only) as {vid}")
            return


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    from .serve.api import create_app

    svc = _service(args)
    if not _config_ok(svc, block=True):
        return 2
    _maybe_demo(svc)
    host = args.host or svc.project.get("serve.host", "127.0.0.1")
    port = int(args.port or svc.project.get("serve.port", 8095))
    if not is_loopback(host) and not os.environ.get("RLCD_API_KEY"):
        admin = bool(os.environ.get("RLCD_ADMIN_KEY"))
        what = ("RLCD_ADMIN_KEY is set, but it only protects training, the kill switch and confirmations: every "
                "other route (decide, systemone, feedback, receipts, metrics, ...) would be open to" if admin else
                "every route, training and the kill switch included, would be open to")
        if not args.insecure_no_auth:
            print(f"rlcd: refusing to serve on {host} without RLCD_API_KEY: {what} anyone who can reach port "
                  f"{port}. Set RLCD_API_KEY (and RLCD_ADMIN_KEY for the operator's routes), serve on 127.0.0.1 "
                  f"behind a proxy, or pass --insecure-no-auth if this network is trusted.", file=sys.stderr)
            return 2
        print(f"warning: --insecure-no-auth: serving on {host} without RLCD_API_KEY: {what} anyone who can reach "
              f"port {port}.", file=sys.stderr)
    for line in _startup_notes(svc):
        print(line, file=sys.stderr)
    _policy_note(svc)
    uvicorn.run(create_app(svc), host=host, port=port, log_level=args.log_level)
    return 0


def is_loopback(host: str) -> bool:
    """Is `host` a loopback bind address (localhost, 127.0.0.0/8, ::1)? Anything else can be reached by others."""
    h = str(host).strip().strip("[]").lower()
    if h == "localhost":
        return True
    try:
        return ipaddress.ip_address(h).is_loopback
    except ValueError:
        return False


def _startup_notes(svc: Any) -> list[str]:
    """What `rlcd serve` says before it listens when no trained model can decide (GET /ready answers 503 then)."""
    try:
        r = svc.readiness()
    except Exception as e:      # never keep the server from starting: /health and /ready report it
        return [f"warning: could not check readiness ({type(e).__name__})"]
    if r["ready"]:
        return []
    return [f"warning: serving {r['model']}, which cannot decide yet: {'; '.join(r['reasons'])}. Until then every "
            f"answer is the uninformed prior, every decision abstains, and GET /ready answers 503 (GET /health "
            f"stays 200)."]


def cmd_bench(args: argparse.Namespace) -> int:
    from .bench import format_table, run

    svc = _service(args)
    res = run(svc, n=args.n, model=args.model, progress=lambda m: print(f"  timing {m}…", file=sys.stderr))
    if args.json:
        _print(res)
    else:
        print(format_table(res))
    return 0


def _print_data_exposure(meta: dict) -> None:
    """What of the training data travels inside a bundle: categorical columns (raw or hashed) on stdout, and a
    warning on stderr when raw categorical values from the training rows are embedded."""
    from .report.model_card import categorical_summary, raw_categorical_warning

    summary = categorical_summary(meta)
    if summary:
        print(summary)
    warning = raw_categorical_warning(meta)
    if warning:
        print(f"warning: {warning}", file=sys.stderr)


def cmd_export(args: argparse.Namespace) -> int:
    from .core.bundle import Bundle
    from .report.model_card import write_model_card

    svc = _service(args)
    version = svc.resolve(args.model)
    if svc.bundle(version) is None:
        print("no trained model to export", file=sys.stderr)
        return 1
    dest = svc.registry.export(version, Path(args.dest))
    b = Bundle(dest)
    write_model_card(b)
    print(f"exported {version} to {dest} (JSON + safetensors + LightGBM text; no pickles). It contains no training "
          f"rows, but heads learned from your data: share it only if that is allowed.")
    _print_data_exposure(b.meta)
    return 0


def cmd_import(args: argparse.Namespace) -> int:
    svc = _service(args)
    vid = svc.registry.import_(Path(args.src), make_latest=not args.keep_latest)
    print(f"imported {args.src} as {vid}")
    return 0


def cmd_push(args: argparse.Namespace) -> int:
    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("`rlcd push` needs huggingface-hub: install the 'documents' extra", file=sys.stderr)
        return 1
    from .core.bundle import validate

    svc = _service(args)
    version = svc.resolve(args.model)
    b = svc.bundle(version)
    if b is None:
        print("no trained model to push", file=sys.stderr)
        return 1
    validate(b.path)
    api = HfApi()
    # create_repo(exist_ok=True) never changes (or reports) the visibility of a repo that already exists, so it is
    # checked here: --private must never end with the bundle in a public repo
    state, detail = _hub_visibility(api, args.hf)
    if args.private and state == "public":
        print(f"refusing to push: {args.hf} already exists and is PUBLIC, so --private would not keep this bundle "
              f"private. Make the repo private on the Hub first, or push to a new repo id.", file=sys.stderr)
        return 1
    if args.private and state == "unknown":
        print(f"refusing to push: cannot verify that {args.hf} is private ({detail}). Check the repo and your "
              f"Hugging Face login, then retry.", file=sys.stderr)
        return 1
    where = {"absent": f"new repo, created {'private' if args.private else 'PUBLIC'}",
             "private": "existing private repo", "public": "existing PUBLIC repo: anyone can download it",
             "unknown": f"visibility unknown: {detail}"}[state]
    _print_data_exposure(b.meta)
    if not args.yes:
        ans = input(f"Upload {version} to the Hugging Face Hub repo {args.hf} ({where})? [y/N] ")
        if ans.strip().lower() not in ("y", "yes"):
            print("cancelled")
            return 1
    if state != "private" and state != "public":
        api.create_repo(args.hf, private=args.private, exist_ok=True)
        if args.private:
            after, detail = _hub_visibility(api, args.hf)
            if after != "private":
                print(f"refusing to upload: {args.hf} is not a private repo you can see after creating it "
                      f"({after}{': ' + detail if detail else ''}); nothing was uploaded.", file=sys.stderr)
                return 1
    api.upload_folder(folder_path=str(b.path), repo_id=args.hf, commit_message=f"open-rlcd bundle {version}")
    final = "private" if state == "private" or args.private else "PUBLIC" if state != "unknown" else where
    print(f"pushed {version} to https://huggingface.co/{args.hf} ({final})")
    return 0


def _hub_visibility(api: Any, repo_id: str) -> tuple[str, str]:
    """('private' | 'public' | 'absent' | 'unknown', detail) for a model repo on the Hugging Face Hub. 'absent'
    also covers a private repo this login cannot see (the Hub answers 'not found' for both)."""
    try:
        from huggingface_hub.errors import RepositoryNotFoundError
    except ImportError:  # older huggingface_hub
        from huggingface_hub.utils import RepositoryNotFoundError
    try:
        info = api.repo_info(repo_id)
    except RepositoryNotFoundError:
        return "absent", ""
    except Exception as e:  # network, auth, a gated repo: visibility cannot be verified
        return "unknown", f"{type(e).__name__}: {str(e)[:200]}"
    private = getattr(info, "private", None)
    if private is True:
        return "private", ""
    if private is False:
        return "public", ""
    return "unknown", "the Hub did not say whether the repo is private"


def cmd_demo(args: argparse.Namespace) -> int:
    from .core.config import load_project, write_config
    from .core.contract import TrainRequest
    from .core.service import Service

    root = Path(args.dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    enc = {"kind": "hashing", "dim": 256} if args.encoder == "hashing" else \
        {"kind": "onnx", "model_id": "intfloat/multilingual-e5-small"}
    write_config(root / "rlcd.yaml", {"name": "rlcd-demo", "pack": "documents", "home": "./var",
                                      "pack_config": {"encoder": enc}}, HEADER + "# synthetic demo project\n")
    project = load_project(root)
    project.home_override = root / "var"   # the demo's own home, whatever RLCD_HOME says
    from .packs.documents.synth import write_synth

    print("writing the synthetic corpus and labels…")
    info = write_synth(project.home, enc, seed=args.seed, n_queries=args.queries)
    print(json.dumps(info, default=str)[:400])
    svc = Service(project)
    print("training…")
    job = svc.trainer.start(TrainRequest(seed=args.seed), wait=True)
    print(_summary_table(job))
    if args.export and job.get("model"):
        from .core.bundle import Bundle
        from .report.model_card import write_model_card

        dest = svc.registry.export(job["model"], Path(args.export))
        write_model_card(Bundle(dest))
        print(f"exported the demo bundle to {dest}")
    print(f"\ndemo project in {root} (synthetic data only). Try:\n  cd {root} && rlcd serve")
    return 0


def cmd_decide(args: argparse.Namespace) -> int:
    from .core.contract import DecideRequest

    svc = _service(args)
    _config_ok(svc, block=False)       # read-only: the findings are printed, the decision is still made
    _policy_note(svc, args.model)
    state: Any = json.loads(args.state) if args.state else {"query": args.query}
    heads = [h.strip() for h in args.heads.split(",") if h.strip()] if args.heads else None
    _print(svc.decide(DecideRequest(model=args.model, state=state, stakes=args.stakes, heads=heads)))
    return 0


def cmd_calibration(args: argparse.Namespace) -> int:
    svc = _service(args)
    rep = svc.calibration_view(args.head, args.model)
    keep = ("head", "model", "n_holdout", "log_loss", "brier", "brier_skill_score", "ece", "accuracy", "skill",
            "baseline", "judge_agreement", "conformal", "data")
    _print({k: rep.get(k) for k in keep if k in rep} | {"verdict": rep.get("policy", {}).get("verdict")})
    return 0


def _read_json_arg(value: str) -> Any:
    """A JSON value given inline, or `@path` / a path to a JSON file."""
    if value.startswith("@"):
        return json.loads(Path(value[1:]).expanduser().read_text(encoding="utf-8"))
    p = Path(value).expanduser()
    if not value.lstrip().startswith(("{", "[", '"')) and p.is_file():
        return json.loads(p.read_text(encoding="utf-8"))
    return json.loads(value)


def cmd_ask(args: argparse.Namespace) -> int:
    from .core.contract import SystemOneRequest

    svc = _service(args)
    _config_ok(svc, block=False)       # read-only: the findings are printed, the questions are still answered
    _policy_note(svc, args.model)
    state: Any = _read_json_arg(args.state) if args.state else {"query": args.query}
    battery = args.battery
    if not battery and not args.heads and args.query and "preflight" in svc.batteries(svc.resolve(args.model)):
        battery = "preflight"      # a bare query: the questions that read one (every head would need evidence too)
    if battery:
        req = SystemOneRequest(model=args.model, state=state, battery=battery, strict=args.strict)
    else:
        version = svc.resolve(args.model)
        specs = svc.specs(version)
        names = args.heads.split(",") if args.heads else list(specs)
        unknown = [n for n in names if n not in specs]
        if unknown:
            print(f"unknown heads {unknown}; heads: {list(specs)}", file=sys.stderr)
            return 2
        req = SystemOneRequest(model=args.model, state=state, strict=args.strict,
                               questions={n: {"type": specs[n].type, "head": n} for n in names})
    _print(svc.systemone(req))
    return 0


def cmd_batteries(args: argparse.Namespace) -> int:
    _print(_service(args).batteries_view(args.model))
    return 0


def cmd_receipts(args: argparse.Namespace) -> int:
    svc = _service(args)
    if args.receipts_cmd == "verify":
        res = svc.verify_receipts()
        _print(res)
        conf = res.get("confirmations") or {}
        if not res["ok"]:
            print(f"receipt chain broken at {len(res['breaks']) + len(conf.get('breaks') or [])} place(s): see "
                  f"`breaks` (decisions.jsonl) and `confirmations.breaks`", file=sys.stderr)
            return 1
        n = conf.get("receipts") or 0
        tail = f"; {n} confirmation{'' if n == 1 else 's'} chained, head {conf['head']}" if n else ""
        print(f"{res['receipts']} receipts chained, head {res['head']}{tail} (anchor "
              f"{'these digests' if tail else 'this digest'} elsewhere to detect truncation)", file=sys.stderr)
        return 0
    if args.receipts_cmd == "show":
        _print(svc.receipt_view(args.decision_id))
        return 0
    if args.receipts_cmd == "check":
        kw: dict[str, Any] = {"max_age_s": args.max_age}
        if args.state is not None:
            kw["state"] = _read_json_arg(args.state)
        else:
            kw["state_digest"] = args.state_digest
        res = svc.check_receipt(args.decision_id, **kw)
        _print(res)
        return 0 if res["current"] else 1
    if args.receipts_cmd == "confirm":
        _print(svc.confirm_receipt(args.decision_id, by=args.by or os.environ.get("USER") or "unknown",
                                   note=args.note))
        return 0
    if args.receipts_cmd == "prune":
        from .core.receipts import prune_states

        _print(prune_states(svc.logs.decision_log.path, args.before))
        return 0
    return 2


def cmd_automation(args: argparse.Namespace) -> int:
    svc = _service(args)
    if args.state in (None, "status"):
        _print(svc.automation_view())
        return 0
    view = svc.set_automation(args.state == "on", reason=args.reason, by=args.by or os.environ.get("USER"))
    _print(view)
    return 0


def cmd_promote(args: argparse.Namespace) -> int:
    svc = _service(args)
    target = args.version if args.cmd == "promote" else (args.to or svc.registry.previous_latest())
    if not target:
        print("no earlier version to roll back to", file=sys.stderr)
        return 1
    _print(svc.promote(target, reason=args.reason, by=args.by or os.environ.get("USER") or "unknown",
                       kind="promote" if args.cmd == "promote" else "rollback"))
    return 0


def cmd_metrics(args: argparse.Namespace) -> int:
    svc = _service(args)
    _print(svc.metrics_view(since=args.since, until=args.until, model=args.model, kind=args.kind, head=args.head))
    return 0


def cmd_review(args: argparse.Namespace) -> int:
    svc = _service(args)
    view = svc.review_view(limit=args.limit, head=args.head, kind=args.kind, include_abstain=not args.confirm_only,
                           oldest_first=not args.newest_first)
    _print(view)
    print(f"{view['pending']} decisions wait for review; resolve one with POST /v1/feedback", file=sys.stderr)
    return 0


def cmd_replay(args: argparse.Namespace) -> int:
    from .core.replay import replay

    svc = _service(args)
    candidate = svc.resolve(args.candidate)
    base = args.base
    if base is None:   # what served before the candidate (when it is rlcd-latest), else what serves now
        base = svc.registry.previous_latest() if candidate == svc.registry.latest_id() else svc.registry.latest_id()
    if not base:
        print("no earlier version to compare with: pass --base VERSION", file=sys.stderr)
        return 1
    heads = args.heads.split(",") if args.heads else None
    rep = replay(svc, base, candidate, None, heads=heads, limit=args.limit, since=args.since,
                 max_examples=args.examples)
    _print(rep)
    print(rep["summary"], file=sys.stderr)
    return 0


def cmd_regress(args: argparse.Namespace) -> int:
    from .core import regress as RG

    svc = _service(args)
    path = Path(args.cases).expanduser() if args.cases else RG.cases_path(svc.project)
    cases = RG.load_cases(path)
    if not cases:
        print(f"no regression cases in {path}: add one with `rlcd case add --from-decision ID --head H --expect "
              f"class=X`, or post feedback tagged \"regression\"", file=sys.stderr)
        return 0
    res = RG.run_cases(svc, cases, model=args.model)
    _print(res if args.all else {**res, "results": [r for r in res["results"] if r["status"] != "passed"]})
    print(f"{res['model']}: {res['passed']} passed, {res['failed']} failed, {res['errors']} errors ({path})",
          file=sys.stderr)
    return 1 if res["failed"] or res["errors"] else 0


def cmd_case(args: argparse.Namespace) -> int:
    from .core import regress as RG

    svc = _service(args)
    path = Path(args.cases).expanduser() if args.cases else RG.cases_path(svc.project)
    case = RG.case_from_decision(svc, args.from_decision, head=args.head, expect=RG.parse_expect(args.expect),
                                 note=args.note, incident=args.incident)
    saved = RG.add_case(path, case)
    _print(saved)
    print(f"added {saved['id']} to {path}: it copies the decision's logged state, review it before committing",
          file=sys.stderr)
    return 0


def cmd_check(args: argparse.Namespace) -> int:
    from .core.checklist import format_table, run

    svc = _service(args)
    _config_ok(svc, block=False)       # the findings go to stderr here; the checklist's config.valid item decides
    res = run(svc, model=args.model)
    if args.json:
        _print(res)
    else:
        print(format_table(res))
    return 0 if res["ok"] else 1


def cmd_config(args: argparse.Namespace) -> int:
    from .core import config_check as CC

    if args.config_cmd == "check":
        project, findings = CC.check_file(args.project)
        path = project.path if project is not None else args.project
        n_err, n_warn = len(CC.errors(findings)), len(CC.warnings(findings))
        if args.json:
            _print({"config": str(path) if path else None, "ok": n_err == 0, "errors": n_err, "warnings": n_warn,
                    "findings": findings})
            return 1 if n_err else 0
        for line in CC.format_findings(findings):
            print(line)
        if path is None:
            print("no rlcd.yaml here or in a parent folder (and RLCD_PROJECT is not set): the default project is "
                  "used, there is nothing to check. `rlcd init NAME` writes one.")
        else:
            print(f"{path}: {CC.summary(findings)}")
        return 1 if n_err else 0
    project = _project(args)
    view = json.loads(json.dumps(CC.effective(project), default=str))
    if args.json:
        _print(view)
    else:
        import yaml

        print(yaml.safe_dump(view, sort_keys=False, allow_unicode=True, width=110), end="")
    for line in CC.format_findings(CC.check_project(project)):
        print(f"rlcd.yaml: {line}", file=sys.stderr)
    return 0


# ----------------------------------------------------------------------------- parser

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="rlcd", description="Open RLCD: calibrated typed decisions.")
    p.add_argument("--version", action="version", version=f"open-rlcd {VERSION}")
    p.add_argument("--project", help="project directory or rlcd.yaml (default: nearest rlcd.yaml)")
    p.add_argument("--home", help="artifacts directory (default: the project's home, or RLCD_HOME)")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("init", help="create a project with rlcd.yaml")
    s.add_argument("name")
    s.add_argument("--pack", choices=sorted(TEMPLATES), default="documents")
    s.add_argument("--force", action="store_true")
    s.set_defaults(fn=cmd_init)

    s = sub.add_parser("config", help="rlcd.yaml: check it key by key, or show what it resolves to")
    gsub = s.add_subparsers(dest="config_cmd", required=True)
    g = gsub.add_parser("check", help="unknown keys, wrong types, values out of range, secrets in the file "
                                      "(exit 1 on an error)")
    g.add_argument("--json", action="store_true", help="print the findings as JSON")
    g = gsub.add_parser("show", help="the effective configuration: home, pack, heads, thresholds, rollout")
    g.add_argument("--json", action="store_true", help="print JSON instead of YAML")
    s.set_defaults(fn=cmd_config)

    s = sub.add_parser("ingest", help="documents pack: read a local folder into the corpus (uses the project's "
                                      "encoder)",
                       description="Read a folder into the documents corpus: chunks and BM25 under RLCD_HOME, vectors "
                                   "and the entity graph locally or in the stores pack_config.vector / graph name "
                                   "(index mode). Remote hosts need security.allow_remote: true or "
                                   "RLCD_ALLOW_REMOTE=1; credentials come from environment variables only.")
    s.add_argument("path", nargs="?", help="folder of documents (default: pack_config.corpus)")
    s.add_argument("--qdrant", metavar="URL", help="upsert the vectors into Qdrant at URL (index mode; needs "
                                                   "--collection unless pack_config.vector names one; API key from "
                                                   "the variable named by pack_config.vector.api_key_env)")
    s.add_argument("--collection", metavar="NAME", help="the Qdrant collection open-rlcd writes its chunks into "
                                                        "(created with the encoder's dimension if missing)")
    s.add_argument("--neo4j", metavar="URI", help="write the entity graph into Neo4j at URI, e.g. "
                                                  "bolt://localhost:7687 (password from the variable named by "
                                                  "pack_config.graph.password_env)")
    s.add_argument("--csv", help=argparse.SUPPRESS)     # not a documents-pack input: refused with a message
    s.add_argument("--max-chars", type=int, dest="max_chars", help="chunk size in characters")
    s.set_defaults(fn=cmd_ingest)

    s = sub.add_parser("label", help="labels: bootstrap with an LLM judge, audit, choose what to audit next")
    lsub = s.add_subparsers(dest="label_cmd", required=True)
    b = lsub.add_parser("bootstrap", help="weak labels from an LLM judge")
    b.add_argument("--judge", help="claude-cli[:model] | anthropic[:model] | ollama:<model> | openai:<url>|<model>")
    b.add_argument("--heads")
    b.add_argument("--limit", type=int)
    b.add_argument("--batch", type=int)
    a = lsub.add_parser("audit", help="human review in the terminal")
    a.add_argument("--heads")
    a.add_argument("--n", type=int, default=300,
                   help="questions this session, spread over the selected heads (a head needs ~150 trusted rows "
                        "for a 30-row holdout: audit a few heads at a time with --heads)")
    a.add_argument("--queue", action="store_true", help="audit the `rlcd label next` queue")
    a.add_argument("--annotator")
    nx = lsub.add_parser("next", help="queue the most informative rows for audit")
    nx.add_argument("--head")
    nx.add_argument("--n", type=int, default=50)
    lsub.add_parser("status", help="label counts per head and source")
    s.set_defaults(fn=cmd_label)

    s = sub.add_parser("train", help="train every head (or --heads) and print the calibration summary")
    s.add_argument("--heads")
    s.add_argument("--calibration", choices=["auto", "sigmoid", "isotonic", "temperature"])
    s.add_argument("--seed", type=int)
    s.add_argument("--holdout-fraction", type=float, dest="holdout_fraction")
    s.add_argument("--n-splits", type=int, dest="n_splits")
    s.add_argument("--feedback-weight", type=float, dest="feedback_weight")
    s.set_defaults(fn=cmd_train)

    s = sub.add_parser("report", help="write MODEL_CARD.md and report.html, open the page")
    s.add_argument("--model")
    s.add_argument("--out")
    s.add_argument("--no-open", action="store_true")
    s.set_defaults(fn=cmd_report)

    s = sub.add_parser("serve", help="run the HTTP API")
    s.add_argument("--host")
    s.add_argument("--port", type=int)
    s.add_argument("--log-level", default="info", dest="log_level")
    s.add_argument("--insecure-no-auth", action="store_true", dest="insecure_no_auth",
                   help="serve on a non-loopback host although RLCD_API_KEY is not set. Without this flag `rlcd "
                        "serve` refuses that (exit 2). RLCD_API_KEY protects every route except /health and /ready; "
                        "RLCD_ADMIN_KEY protects only POST /v1/train, POST /v1/automation and POST "
                        "/v1/receipts/{id}/confirm (and is accepted wherever RLCD_API_KEY is), so an admin key "
                        "without an API key leaves every other route open")
    s.set_defaults(fn=cmd_serve)

    s = sub.add_parser("bench", help="latency per stage (p50/p95/p99) on this machine")
    s.add_argument("--n", type=int, default=200)
    s.add_argument("--model")
    s.add_argument("--json", action="store_true")
    s.set_defaults(fn=cmd_bench)

    s = sub.add_parser("export", help="write a model version as an open-weight bundle directory")
    s.add_argument("dest")
    s.add_argument("--model")
    s.set_defaults(fn=cmd_export)

    s = sub.add_parser("import", help="register a bundle directory as a new version")
    s.add_argument("src")
    s.add_argument("--keep-latest", action="store_true", dest="keep_latest")
    s.set_defaults(fn=cmd_import)

    s = sub.add_parser("push", help="upload a bundle to the Hugging Face Hub (opt-in)")
    s.add_argument("--hf", required=True, help="repo id, e.g. my-org/my-rlcd")
    s.add_argument("--model")
    s.add_argument("--private", action="store_true")
    s.add_argument("--yes", action="store_true")
    s.set_defaults(fn=cmd_push)

    s = sub.add_parser("demo", help="build and train a synthetic documents project")
    s.add_argument("--dir", default="rlcd-demo")
    s.add_argument("--encoder", choices=["hashing", "onnx"], default="hashing")
    s.add_argument("--seed", type=int, default=7)
    s.add_argument("--queries", type=int, default=900)
    s.add_argument("--export", help="also export the trained bundle to this (empty or new) directory")
    s.set_defaults(fn=cmd_demo)

    s = sub.add_parser("decide", help="one decision against the local model")
    s.add_argument("--query")
    s.add_argument("--state", help="JSON state")
    s.add_argument("--stakes", choices=["low", "medium", "high"])
    s.add_argument("--heads", help="comma-separated heads to decide with (default: every head; as `heads` of "
                                   "POST /v1/decide)")
    s.add_argument("--model")
    s.set_defaults(fn=cmd_decide)

    s = sub.add_parser("calibration", help="print a head's holdout calibration summary")
    s.add_argument("head")
    s.add_argument("--model")
    s.set_defaults(fn=cmd_calibration)

    # ---- the control layer
    s = sub.add_parser("ask", help="answer typed questions on one state (a System One call, no HTTP)")
    s.add_argument("--query", help="shorthand for --state '{\"query\": ...}'")
    s.add_argument("--state", help="the state: inline JSON, or @file.json")
    s.add_argument("--heads", help="comma-separated heads (default: every head)")
    s.add_argument("--battery", help="a battery name (optionally name@version) instead of --heads")
    s.add_argument("--strict", action="store_true", help="refuse a state that lacks inputs training rows had")
    s.add_argument("--model")
    s.set_defaults(fn=cmd_ask)

    s = sub.add_parser("batteries", help="list the versioned question sets (batteries)")
    s.add_argument("--model")
    s.set_defaults(fn=cmd_batteries)

    s = sub.add_parser("receipts", help="decision receipts: verify the chain, show, check staleness, prune states")
    rsub = s.add_subparsers(dest="receipts_cmd", required=True)
    rsub.add_parser("verify", help="check every receipt's digest and the chain between them")
    r = rsub.add_parser("show", help="print one decision record")
    r.add_argument("decision_id")
    r = rsub.add_parser("check", help="does a decision still approve acting on this state now? (exit 1 if stale)")
    r.add_argument("decision_id")
    g = r.add_mutually_exclusive_group(required=True)
    g.add_argument("--state", help="the state about to be acted on: inline JSON or @file.json")
    g.add_argument("--state-digest", dest="state_digest")
    r.add_argument("--max-age", type=float, dest="max_age", help="seconds")
    r = rsub.add_parser("confirm", help="record that a person confirmed a confirm-tier decision")
    r.add_argument("decision_id")
    r.add_argument("--by", help="who confirmed (default: $USER)")
    r.add_argument("--note")
    r = rsub.add_parser("prune", help="drop logged states older than a date (digests and the chain stay)")
    r.add_argument("--before", required=True, help="ISO 8601 date, e.g. 2026-10-01")
    s.set_defaults(fn=cmd_receipts)

    s = sub.add_parser("automation", help="the kill switch: status | on | off (off caps decisions at confirm)")
    s.add_argument("state", nargs="?", choices=["status", "on", "off"])
    s.add_argument("--reason")
    s.add_argument("--by")
    s.set_defaults(fn=cmd_automation)

    s = sub.add_parser("promote", help="point rlcd-latest at a version (no retrain)")
    s.add_argument("version")
    s.add_argument("--reason")
    s.add_argument("--by", help="who did it, for the registry history (default: $USER, else \"unknown\")")
    s.set_defaults(fn=cmd_promote)

    s = sub.add_parser("rollback", help="point rlcd-latest back at the previous version (or --to VERSION)")
    s.add_argument("--to")
    s.add_argument("--reason")
    s.add_argument("--by", help="who did it, for the registry history (default: $USER, else \"unknown\")")
    s.set_defaults(fn=cmd_promote)

    s = sub.add_parser("metrics", help="operations metrics over the logged decisions, feedback and outcomes")
    s.add_argument("--since", help="ISO 8601 date or time, inclusive (e.g. 2026-11-01)")
    s.add_argument("--until", help="ISO 8601 date or time, exclusive")
    s.add_argument("--model", help="only decisions of this model version (its holdout is compared)")
    s.add_argument("--kind", choices=["decide", "evaluate", "systemone", "rank"])
    s.add_argument("--head", help="only decisions this head took part in")
    s.set_defaults(fn=cmd_metrics)

    s = sub.add_parser("review", help="the review queue: confirm and abstain decisions without feedback (read-only)")
    s.add_argument("--limit", type=int, default=50)
    s.add_argument("--head", help="only decisions this head took part in and has no feedback for")
    s.add_argument("--kind", choices=["decide", "evaluate"])
    s.add_argument("--confirm-only", action="store_true", dest="confirm_only", help="leave out abstained decisions")
    s.add_argument("--newest-first", action="store_true", dest="newest_first")
    s.set_defaults(fn=cmd_review)

    s = sub.add_parser("replay", help="re-run logged decisions under two versions: classes, tiers, bands, labels")
    s.add_argument("--base", help="the version to compare with (default: what rlcd-latest pointed at before the "
                                  "candidate, or rlcd-latest when the candidate is another version)")
    s.add_argument("--candidate", help="the version under test (default: rlcd-latest)")
    s.add_argument("--since", help="only decisions made at or after this ISO 8601 date or time")
    s.add_argument("--limit", type=int, help="only the N most recent decisions")
    s.add_argument("--heads", help="comma-separated heads (or systemone question ids); default: every head")
    s.add_argument("--examples", type=int, default=20,
                   help="decision ids listed per head where one version was right and the other wrong")
    s.set_defaults(fn=cmd_replay)

    s = sub.add_parser("regress", help="run the regression cases against a version (exit 1 if any fails)")
    s.add_argument("--model", help="the version under test (default: rlcd-latest)")
    s.add_argument("--cases", help="the cases file (default: rlcd.yaml regressions:, else ./regressions.jsonl)")
    s.add_argument("--all", action="store_true", help="print passing cases too")
    s.set_defaults(fn=cmd_regress)

    s = sub.add_parser("case", help="regression cases: add one from a logged decision")
    csub = s.add_subparsers(dest="case_cmd", required=True)
    c = csub.add_parser("add", help="turn a logged decision into a regression case")
    c.add_argument("--from-decision", dest="from_decision", required=True, help="the decision_id")
    c.add_argument("--head", help="the head, or the question id of a systemone receipt (default: the decision's "
                                  "only head)")
    c.add_argument("--expect", action="append", required=True,
                   help="key=value, repeatable: class=X, class_in=X,Y, tier=T, tier_at_most=T, tier_at_least=T, "
                        "p_at_least=P, p_at_most=P (tiers by the policy's names, or 'abstain')")
    c.add_argument("--note")
    c.add_argument("--incident", help="the incident this case comes from (e.g. INC-42)")
    c.add_argument("--cases", help="the cases file (default: rlcd.yaml regressions:, else ./regressions.jsonl)")
    s.set_defaults(fn=cmd_case)

    s = sub.add_parser("check", help="the pre-launch checklist: pass / warn / fail / info per item (exit 1 on a fail)")
    s.add_argument("--model", help="the version to check (default: rlcd-latest)")
    s.add_argument("--json", action="store_true", help="print the result as JSON instead of the table")
    s.set_defaults(fn=cmd_check)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        misplaced = _misplaced(args)
        if misplaced:
            print(f"rlcd: {misplaced}", file=sys.stderr)
            return 2
        return int(args.fn(args) or 0)
    except KeyboardInterrupt:
        return 130
    except Exception as e:  # one readable line for teammates; RLCD_DEBUG=1 for the trace
        if os.environ.get("RLCD_DEBUG") == "1":
            raise
        from .core.contract import ContractError

        msg = e.message if isinstance(e, ContractError) else f"{type(e).__name__}: {e}"
        print(f"rlcd: {msg}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
