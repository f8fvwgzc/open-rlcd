"""`rlcd` — the command teammates use.

    rlcd init my-business --pack documents     write my-business/rlcd.yaml (packs: documents, tabular)
    rlcd ingest ./data                         build the local corpus (documents pack; tabular has no ingest step)
    rlcd label bootstrap --judge claude-cli    weak labels from an LLM judge (documents: needs queries.jsonl)
    rlcd label audit --heads answerable        human review in the terminal (~150+ trusted rows per head)
    rlcd label next --head answerable          the most informative rows to audit next
    rlcd train                                 every head; prints the calibration summary
    rlcd report                                MODEL_CARD.md + report.html (+ per-head JSON)
    rlcd serve --port 8095                     the HTTP API
    rlcd bench                                 p50 / p95 / p99 per stage on this machine
    rlcd export ./bundle | rlcd import ./bundle | rlcd push --hf user/repo
    rlcd demo                                  synthetic documents project, trained, in ./rlcd-demo
"""
from __future__ import annotations

import argparse
import json
import os
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
                        "features": "auto", "categorical": "auto", "hash_categories": False},
        "policy": {"abstain": "hold"},
    },
}
# `rlcd init` offers only the packs that ship on this branch (the trading adapter lives on its own branch).

HEADER = """# Open RLCD project configuration. Docs: docs/teammate-quickstart.md
# Secrets (ANTHROPIC_API_KEY, RLCD_API_KEY, database passwords) come from the environment, never from this file.
"""


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
            "serve": {"host": "127.0.0.1", "port": 8095, "log_states": True},
            "security": {"allow_remote": False}}
    write_config(root / "rlcd.yaml", data, HEADER)
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
            "write ./queries.jsonl: real questions, one {\"query\": \"...\"} per line (bootstrap labels these)",
            "the encoder in rlcd.yaml is ONNX: it needs the documents extra (`uv sync --extra documents`) and "
            "downloads ~135 MB on the first ingest; offline, set pack_config.encoder to {kind: hashing, dim: 256}",
            "rlcd ingest ./data",
            "rlcd label bootstrap        # needs `claude` on PATH, or --judge anthropic | ollama:<model>",
            "rlcd label audit --heads answerable,grounded --n 600   # your labels calibrate the model",
            "rlcd train",
        ]
    else:
        steps = [
            f"cd {root}",
            "copy your table to ./data.csv (or point pack_config.data at it)",
            "edit rlcd.yaml: pack_config.label = the outcome column (id / group / time columns if you have them); "
            "hash_categories: true if a categorical column could hold personal data (bundles store its values)",
            "rlcd train",
        ]
    return "\n".join(f"  {i}. {st}" for i, st in enumerate(steps, 1))


def cmd_ingest(args: argparse.Namespace) -> int:
    svc = _service(args)
    opts = {"path": args.path, "csv": args.csv, "qdrant": args.qdrant, "collection": args.collection,
            "neo4j": args.neo4j, "max_chars": args.max_chars}
    _print(svc.pack.ingest(svc.project, {k: v for k, v in opts.items() if v is not None}))
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

        judge = make_judge(args.judge or project.get("labeling.judge", "claude-cli"), project)
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


def _summary_table(job: dict) -> str:
    rows = [f"{'head':<18}{'status':<10}{'holdout':>8}{'logloss':>9}{'base':>8}{'BSS':>8}{'ECE':>7}  "
            f"{'skill':<6}verdict(medium)"]
    for name, h in job["heads"].items():
        if h.get("status") != "trained":
            rows.append(f"{name:<18}{h.get('status', '?'):<10}  {h.get('error', '')}")
            continue
        m = h["metrics"]

        def f(x, nd=3):
            return "–" if x is None else f"{x:.{nd}f}"
        v = (h.get("verdict") or {}).get("medium", {})
        rows.append(f"{name:<18}{'trained':<10}{m.get('n_holdout') or 0:>8}{f(m.get('log_loss')):>9}"
                    f"{f(m.get('baseline_log_loss')):>8}{f(m.get('brier_skill_score')):>8}{f(m.get('ece')):>7}  "
                    f"{'yes' if m.get('has_skill') else 'NO':<6}{v.get('max_tier', '–')}")
    return "\n".join(rows)


def cmd_train(args: argparse.Namespace) -> int:
    from .core.contract import TrainRequest

    svc = _service(args)
    body: dict = {}
    if args.heads:
        body["heads"] = args.heads.split(",")
    for k in ("calibration", "seed", "holdout_fraction", "n_splits", "feedback_weight"):
        v = getattr(args, k, None)
        if v is not None:
            body[k] = v
    job = svc.trainer.start(TrainRequest(**body), wait=True)
    print(f"model {job.get('model') or '(none saved)'} — job {job['status']}")
    print(_summary_table(job))
    for name, h in job["heads"].items():
        if h.get("status") == "trained" and not (h.get("metrics") or {}).get("has_skill"):
            print(f"\n{name}: no skill over the base rate on the holdout — it will abstain. "
                  f"{(h.get('verdict') or {}).get('medium', {}).get('reason', '')}")
    if job.get("error") and not job.get("model"):
        print(f"\n{job['error']}", file=sys.stderr)
        return 1
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
    _maybe_demo(svc)
    host = args.host or svc.project.get("serve.host", "127.0.0.1")
    port = int(args.port or svc.project.get("serve.port", 8095))
    if host not in ("127.0.0.1", "localhost", "::1") and not os.environ.get("RLCD_API_KEY"):
        print(f"warning: serving on {host} without RLCD_API_KEY: anyone who can reach this port can query and "
              f"train. Set RLCD_API_KEY.", file=sys.stderr)
    uvicorn.run(create_app(svc), host=host, port=port, log_level=args.log_level)
    return 0


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
    state: Any = json.loads(args.state) if args.state else {"query": args.query}
    _print(svc.decide(DecideRequest(model=args.model, state=state, stakes=args.stakes)))
    return 0


def cmd_calibration(args: argparse.Namespace) -> int:
    svc = _service(args)
    rep = svc.calibration_view(args.head, args.model)
    keep = ("head", "model", "n_holdout", "log_loss", "brier", "brier_skill_score", "ece", "accuracy", "skill",
            "baseline", "judge_agreement", "conformal", "data")
    _print({k: rep.get(k) for k in keep if k in rep} | {"verdict": rep.get("policy", {}).get("verdict")})
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

    s = sub.add_parser("ingest", help="documents pack: read a local folder into the corpus (uses the project's "
                                      "encoder)")
    s.add_argument("path", nargs="?", help="folder of documents (default: pack_config.corpus)")
    # remote connectors are not implemented yet: the pack refuses them with a clear message, so they stay out of
    # the help instead of advertising a flow that cannot run
    for flag in ("--csv", "--qdrant", "--collection", "--neo4j"):
        s.add_argument(flag, help=argparse.SUPPRESS)
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
    s.add_argument("--model")
    s.set_defaults(fn=cmd_decide)

    s = sub.add_parser("calibration", help="print a head's holdout calibration summary")
    s.add_argument("head")
    s.add_argument("--model")
    s.set_defaults(fn=cmd_calibration)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
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
