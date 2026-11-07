"""`rlcd bench`: latency per stage on this machine, p50 / p95 / p99.

Stages (each measured on its own, single request unless noted):
  encoder.query        embed one query (cache bypassed: the real model cost)
  encoder.passages32   embed a batch of 32 passages (cache bypassed)
  features.<view>      build one feature row (encoder cache warm: what a repeated query costs)
  head.<name>          one calibrated head on one prepared feature row (the decision model itself)
  ann.search           top-10 vector search in the local corpus index (when a corpus exists)
  graph.expand         2-hop entity expansion in the local graph (when a corpus exists)
  decide.end_to_end    the pack's whole /v1/decide in process (no HTTP)

No claims beyond what is printed: numbers depend on the CPU, thread settings, corpus size and text length.
"""
from __future__ import annotations

import platform
import time
from collections.abc import Callable
from typing import Any

import numpy as np


def _measure(fn: Callable[[], Any], n: int, warmup: int = 3) -> dict:
    for _ in range(warmup):
        fn()
    ts = np.empty(n)
    for i in range(n):
        t0 = time.perf_counter_ns()
        fn()
        ts[i] = (time.perf_counter_ns() - t0) / 1e6
    return {"n": n, "p50_ms": float(np.percentile(ts, 50)), "p95_ms": float(np.percentile(ts, 95)),
            "p99_ms": float(np.percentile(ts, 99)), "mean_ms": float(ts.mean())}


def _sample_states(service: Any, view: str, k: int = 32) -> list[Any]:
    try:
        from .labeling.store import LabelStore

        items = LabelStore(service.home).items(view)
    except Exception:
        items = []
    return [it.state for it in items[:k]]


def run(service: Any, n: int = 200, model: str | None = None, progress: Callable[[str], None] | None = None) -> dict:
    say = progress or (lambda m: None)
    version = service.resolve(model)
    out: dict[str, Any] = {"model": version, "machine": {"platform": platform.platform(),
                                                         "processor": platform.processor() or platform.machine(),
                                                         "python": platform.python_version()}, "stages": {}}
    stages = out["stages"]
    bundle = service.bundle(version)
    if bundle is None:
        out["note"] = "no trained model: only untrained answers could be timed; train first"
        return out
    provider = service.provider(version)
    enc = getattr(provider, "encoder", None)
    base_enc = getattr(enc, "encoder", enc)   # bypass the embedding cache for the encoder stages
    if base_enc is not None:
        say("encoder")
        rng = np.random.default_rng(0)
        words = ["policy", "warranty", "price", "team", "office", "return", "invoice", "manager", "product", "date"]

        def text(m: int) -> str:
            return " ".join(rng.choice(words, size=m)) + f" {rng.integers(1e9)}"
        stages["encoder.query"] = _measure(lambda: base_enc.encode([text(12)], kind="query"), max(10, n // 4))
        stages["encoder.passages32"] = _measure(lambda: base_enc.encode([text(120) for _ in range(32)],
                                                                        kind="passage"), max(5, n // 20))
    for name in bundle.names():
        h = bundle.head(name)
        spec = h.spec
        states = _sample_states(service, spec.view)
        if not states:
            continue
        say(f"features {spec.view}")
        key = f"features.{spec.view}"
        if key not in stages:
            stages[key] = _measure(lambda s=states[0], v=spec.view: provider.transform(v, [s]), max(10, n // 4))
        x = provider.transform(spec.view, [states[0]])
        ens = h.ensemble
        say(f"head {name}")
        stages[f"head.{name}"] = _measure(lambda e=ens, xx=x: e.predict_proba(xx), n)
    corpus = None
    try:
        from .packs.documents.corpus import Corpus

        if enc is not None:
            corpus = Corpus.open(service.home, enc, service.project.pack_config)
    except Exception:
        corpus = None
    if corpus is not None:
        say("ann")
        qv = enc.encode(["what is the warranty of the product"], kind="query")[0]
        vs = getattr(corpus, "vectors", None)
        if vs is not None:
            stages["ann.search"] = _measure(lambda: vs.search(qv, 10), n)
            out["corpus_vectors"] = vs.count()
        gs = getattr(corpus, "graph", None)
        if gs is not None:
            stages["graph.expand"] = _measure(lambda: gs.expand(["Warranty"], hops=2), max(10, n // 4))
    pack = service.pack_for(version)
    if pack.supports("decide"):
        states = _sample_states(service, "query", 8)
        if states:
            from .core.contract import DecideRequest

            say("decide")
            req = DecideRequest(model=version, state=states[0])
            stages["decide.end_to_end"] = _measure(lambda: service.decide(req), max(10, n // 10))
    return out


def format_table(result: dict) -> str:
    rows = [f"model {result['model']} on {result['machine']['processor']} ({result['machine']['platform']})", "",
            f"{'stage':<34}{'p50 ms':>10}{'p95 ms':>10}{'p99 ms':>10}{'n':>7}"]
    for k, v in result["stages"].items():
        rows.append(f"{k:<34}{v['p50_ms']:>10.3f}{v['p95_ms']:>10.3f}{v['p99_ms']:>10.3f}{v['n']:>7}")
    if result.get("note"):
        rows += ["", result["note"]]
    rows += ["", "Heads are the decision model; the encoder is usually the dominant cost. Numbers are this machine's."]
    return "\n".join(rows)
