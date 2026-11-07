"""Build the synthetic documents project, train all eight heads, and show one decide() and one evaluate() call.

    uv run python examples/documents/build_demo.py                       # offline: hashing encoder, ~15 s
    uv run python examples/documents/build_demo.py --encoder onnx        # multilingual-e5-small (downloads once)
    uv run python examples/documents/build_demo.py --docs ~/my-docs --query "What is our refund window?"
    uv run python examples/documents/build_demo.py --dir /tmp/rlcd-docs-demo --queries 1500 --seed 3

What this script does: writes rlcd.yaml for the documents pack into --dir (default examples/documents/var/, kept
out of git by the repository's `var/` rule), writes the synthetic corpus of the fictional Kestrovia Logistics and
its labelled items (`packs/documents/synth.py`), trains through core.service.Service, prints each head's holdout
summary, then asks the trained model for a plan (decide) and a verdict on two drafted answers (evaluate).

With --docs it also ingests your folder (txt, md, html, pdf, json, jsonl) into the same local corpus, so decide()
retrieves from your documents too. What it does not do: it does not label your documents. The heads were trained
on synthetic rows only, so their probabilities are calibrated for that synthetic world, not for your traffic;
label your own queries (`rlcd label bootstrap` / `audit`) and retrain before trusting them. Metrics on synthetic
data are a sanity check of the pipeline, not accuracy.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ENCODERS = {"hashing": {"kind": "hashing", "dim": 256},
            "onnx": {"kind": "onnx", "model_id": "intfloat/multilingual-e5-small"}}


def summary(job: dict) -> str:
    lines = [f"{'head':15s} {'status':8s} {'skill':5s} {'acc':>6s} {'base':>6s} {'ECE':>6s}  verdict (low/medium/high)"]
    for name, h in job["heads"].items():
        if h["status"] != "trained":
            lines.append(f"{name:15s} {h['status']:8s} {h.get('error', '')[:80]}")
            continue
        m = h["metrics"]
        v = "/".join(h["verdict"][s]["max_tier"] for s in ("low", "medium", "high"))
        lines.append(f"{name:15s} {'trained':8s} {'yes' if m['has_skill'] else 'NO':5s} {m['accuracy']:6.3f} "
                     f"{m['baseline_accuracy']:6.3f} {m['ece']:6.3f}  {v}")
    return "\n".join(lines)


def show_decision(d: dict) -> None:
    print(f"  action: {d['action']}   plan: {json.dumps(d['plan'], ensure_ascii=False)}")
    for name, h in d["heads"].items():
        dec = h["decision"]
        print(f"    {name:15s} -> {dec['class']!s:16s} p={dec['p']:.3f} tier={dec['tier']}")
    for w in d["warnings"][:4]:
        print(f"    warning: {w[:150]}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dir", default=str(HERE / "var"), help="project directory (default examples/documents/var)")
    ap.add_argument("--encoder", choices=sorted(ENCODERS), default="hashing")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--queries", type=int, default=900, help="synthetic queries (default 900)")
    ap.add_argument("--docs", help="a folder of your own documents to ingest for retrieval")
    ap.add_argument("--query", action="append", help="a request to decide on (repeatable; default: three examples)")
    args = ap.parse_args(argv)

    os.environ.pop("RLCD_HOME", None)       # the project's own home, not a global one
    from open_rlcd.core.config import load_project, write_config
    from open_rlcd.core.contract import DecideRequest, EvaluateRequest, TrainRequest
    from open_rlcd.core.service import Service
    from open_rlcd.packs.documents import write_synth
    from open_rlcd.packs.documents.world import build_world

    root = Path(args.dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    enc = ENCODERS[args.encoder]
    write_config(root / "rlcd.yaml", {"name": "documents-demo", "pack": "documents", "home": "./home",
                                      "pack_config": {"encoder": enc}},
                 "# synthetic documents demo (examples/documents/build_demo.py)")
    project = load_project(root)
    t0 = time.perf_counter()
    info = write_synth(project.home, enc, seed=args.seed, n_queries=args.queries)
    print(f"synthetic data: {info['corpus']['docs']} documents, items {info['items']}, languages {info['languages']} "
          f"({time.perf_counter() - t0:.1f} s)")
    svc = Service(project)
    if args.docs:
        out = svc.pack.ingest(project, {"path": args.docs})
        print(f"ingested {args.docs}: {out['docs_added']} new / {out['docs_updated']} updated documents, "
              f"{out['chunks_total']} chunks in the corpus")
    t0 = time.perf_counter()
    job = svc.trainer.start(TrainRequest(seed=args.seed), wait=True)
    print(f"\ntraining: {job['status']} ({time.perf_counter() - t0:.1f} s), model {job['model']}\n")
    print(summary(job))
    if job["status"] not in ("succeeded", "partial"):
        print(job.get("error"), file=sys.stderr)
        return 1

    for query in args.query or ["What is the warranty period of the Velobox 300?",
                                "Who leads the team that owns the Velobox 300?",
                                "What's the weather in Lisbon tomorrow?"]:
        print(f"\ndecide: {query!r}")
        show_decision(svc.decide(DecideRequest(state={"query": query})))

    corpus = svc.pack.corpus(project, svc.provider(svc.resolve(None)))
    sheet = corpus.chunk("products/velobox-300#0")
    question = "What is the price of the Velobox 300?"
    price = build_world(args.seed).docs["products/velobox-300"].meta["price"]
    for label, answer in (("faithful", f"The price of Velobox 300 is {price:,} USD."),
                          ("changed number", f"The price of Velobox 300 is {price + 170:,} USD.")):
        r = svc.evaluate(EvaluateRequest(state={"query": question, "evidence": [sheet], "answer": answer}))
        g = r["heads"]["grounded"]["answer"]["noul"]
        print(f"\nevaluate ({label}): {answer!r}\n  verdict {r['verdict']} (action {r['action']}, stakes {r['stakes']} "
              f"from {r['stakes_from']}), P(grounded) = {g:.3f}")
    print(f"\nproject in {root}. Serve it with:  cd {root} && uv run rlcd serve")
    print("Synthetic data only: these numbers are a pipeline check, not accuracy on real traffic.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
