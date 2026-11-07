"""`rlcd bench`: latency per stage on this machine, p50 / p95 / p99.

Stages (each measured on its own, single request unless noted):
  encoder.query          embed one query (embedding cache bypassed: the real model cost)
  encoder.passages32     embed a batch of 32 passages (embedding cache bypassed)
  features.<view>        build one feature row for one repeated state (provider and encoder caches warm: what a
                         repeated request costs)
  head.<name>            one calibrated head on one prepared feature row (the decision model itself)
  ann.search             top-10 vector search in the local corpus index (when a corpus exists)
  graph.expand           2-hop entity expansion in the local graph from a real entity of the corpus (the
                         best-connected entity that is not a hub; skipped when the graph has none)
  decide.end_to_end      the pack's whole /v1/decide in process (no HTTP) on distinct sampled query states, cycled,
                         with the query's embedding-cache lookup and the provider's feature cache bypassed: what a
                         new query costs over an ingested corpus (the stage's `cache` field says what was bypassed)
  decide.repeated_query  the same /v1/decide on one query repeated: every cache warm, for comparison

No claims beyond what is printed: numbers depend on the CPU, thread settings, corpus size and text length.
"""
from __future__ import annotations

import contextlib
import itertools
import platform
import time
from collections.abc import Callable, Iterator
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
    """Up to `k` distinct states of `view` from the label store (the project's real items), in store order."""
    import json

    try:
        from .labeling.store import LabelStore

        items = LabelStore(service.home).items(view)
    except Exception:
        items = []
    out, seen = [], set()
    for it in items:
        try:
            key = json.dumps(it.state, sort_keys=True, default=str)
        except (TypeError, ValueError):
            key = repr(it.state)
        if key in seen:
            continue
        seen.add(key)
        out.append(it.state)
        if len(out) >= k:
            break
    return out


def _base_encoder(enc: Any) -> Any:
    """The model behind any caching wrappers (features.text.CachedEncoder keeps it in `.inner`)."""
    seen = 0
    while enc is not None and seen < 8:
        inner = getattr(enc, "inner", None)
        if inner is None or not hasattr(inner, "encode"):
            return enc
        enc, seen = inner, seen + 1
    return enc


@contextlib.contextmanager
def _caches_bypassed(provider: Any) -> Iterator[list[str]]:
    """Inside the block, query texts skip the encoder's embedding cache (no reads, no writes: the user's cache is
    not filled with bench queries) and the provider's featurised-state cache is off, so every call pays what a
    query never seen before pays. Passages (corpus chunks) still come from the embedding cache, as they do after
    `rlcd ingest`. Yields what was bypassed; everything is restored on exit."""
    bypassed: list[str] = []
    enc = getattr(provider, "encoder", None)
    base = _base_encoder(enc)
    patched = enc is not None and base is not enc
    if patched:
        cached = enc.encode

        def encode(texts: list[str], kind: str = "passage") -> Any:
            return base.encode(texts, kind=kind) if kind == "query" else cached(texts, kind=kind)

        enc.encode = encode               # instance attribute: shadows CachedEncoder.encode for the block
        bypassed.append("embedding cache for the query")
    size = getattr(provider, "_cache_size", None)
    if isinstance(size, int) and not isinstance(size, bool) and size > 0:
        provider._cache_size = 0          # documents provider: 0 = no featurised-state cache
        bypassed.append("feature cache")
    try:
        yield bypassed
    finally:
        if patched:
            enc.__dict__.pop("encode", None)
        if "feature cache" in bypassed:
            provider._cache_size = size


def _graph_seed(gs: Any) -> str | None:
    """A real entity of the local graph to expand from: the best-connected entity node that expansion actually
    walks (not a hub above `max_fanout`), as its surface form. None when the graph has no entity."""
    g = getattr(gs, "g", None)
    if g is None:
        return None
    fanout = int(getattr(gs, "max_fanout", 200) or 200)
    labels = getattr(gs, "labels", {}) or {}
    best: tuple[int, str] | None = None
    try:
        for node, attrs in g.nodes():
            if not (str(node).startswith("ent:") and (attrs or {}).get("kind", "ent") == "ent"):
                continue
            deg = int(g.degree(node))
            if deg > fanout:
                continue
            if best is None or deg > best[0] or (deg == best[0] and str(node) < best[1]):
                best = (deg, str(node))
    except Exception:
        return None
    if best is None:
        return None
    key = best[1][4:]
    return str(labels.get(key) or key)


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
    base_enc = _base_encoder(enc)   # bypass the embedding cache for the encoder stages
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
        qv = base_enc.encode(["what is the warranty of the product"], kind="query")[0]   # not cached either
        vs = getattr(corpus, "vectors", None)
        if vs is not None:
            stages["ann.search"] = _measure(lambda: vs.search(qv, 10), n)
            out["corpus_vectors"] = vs.count()
        gs = getattr(corpus, "graph", None)
        if gs is not None:
            seed = _graph_seed(gs)
            hits = gs.expand([seed], hops=2) if seed else []
            if hits:
                say("graph")
                stages["graph.expand"] = {**_measure(lambda: gs.expand([seed], hops=2), max(10, n // 4)),
                                          "seed": seed, "results": len(hits)}
            else:
                out.setdefault("skipped", {})["graph.expand"] = "the corpus graph has no entity to expand from"
    pack = service.pack_for(version)
    if pack.supports("decide"):
        m, warmup = max(10, n // 10), 3
        states = _sample_states(service, "query", m + warmup)
        if states:
            from .core.contract import DecideRequest

            say("decide")
            reqs = [DecideRequest(model=version, state=s) for s in states]
            following = itertools.cycle(reqs).__next__

            with _caches_bypassed(provider) as bypassed:
                stage = _measure(lambda: service.decide(following()), m, warmup)
            cache = (f"cold: {' and '.join(bypassed)} bypassed" if bypassed else
                     "no cache to bypass found: calls after the first pass over the states may hit caches")
            stages["decide.end_to_end"] = {**stage, "distinct_states": len(states),
                                           "repeats_states": len(states) < m + warmup, "cache": cache}
            again = reqs[0]
            stages["decide.repeated_query"] = {**_measure(lambda: service.decide(again), m), "distinct_states": 1,
                                               "cache": "warm: one query repeated"}
    return out


def format_table(result: dict) -> str:
    rows = [f"model {result['model']} on {result['machine']['processor']} ({result['machine']['platform']})", "",
            f"{'stage':<34}{'p50 ms':>10}{'p95 ms':>10}{'p99 ms':>10}{'n':>7}"]
    notes = []
    for k, v in result["stages"].items():
        rows.append(f"{k:<34}{v['p50_ms']:>10.3f}{v['p95_ms']:>10.3f}{v['p99_ms']:>10.3f}{v['n']:>7}")
        if v.get("cache"):
            notes.append(f"{k}: {v['cache']} ({v.get('distinct_states')} distinct state(s))")
        if v.get("seed"):
            notes.append(f"{k}: from the entity {v['seed']!r} ({v.get('results')} results)")
    for k, why in (result.get("skipped") or {}).items():
        notes.append(f"{k}: skipped, {why}")
    if notes:
        rows += ["", *notes]
    if result.get("note"):
        rows += ["", result["note"]]
    rows += ["", "Heads are the decision model; the encoder is usually the dominant cost. Numbers are this machine's."]
    return "\n".join(rows)
