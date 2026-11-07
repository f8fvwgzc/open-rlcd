"""`rlcd label next`: which items a human should label next for one head.

score = w_uncertainty * uncertainty + w_disagreement * disagreement, then a greedy k-center pass for diversity:

  * uncertainty    from the calibrated probabilities of the latest version: binary heads (1 - max p) / 0.5,
                   multiclass heads the normalised entropy; both in [0, 1];
  * disagreement   the judge's label differs from the model's argmax, scaled by the judge's confidence (0.5 when
                   the judge gave none): a confident judge contradicting the model is worth a human look;
  * diversity      items are picked one at a time; each pick adds w_diversity * (distance to the nearest
                   already-picked row / the largest such distance) on standardised feature rows, so a cluster of
                   near-duplicates does not fill the batch. With diversity on, exact feature duplicates of a picked
                   row are left out of this batch.

An untrained head (no trained version yet) falls back to low judge confidence (items without a judge label count
as fully uncertain) plus the same diversity pass.

The queue is written to <home>/labels/queue.jsonl (rows of other heads are kept) and `rlcd label audit
--strategy queue` walks it.

What this does not do: produce a representative sample. Rows picked for being hard make a trusted set that is
harder than production traffic; since calibrators are fitted and the holdout is scored on trusted rows, keep the
stratified audit (a proportional sample) as the backbone of the trusted labels and use this queue on top. The
audit records which sampling produced each human label in its `note` ("sample=queue" / "sample=stratified").
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

import numpy as np

from ..core.contract import ContractError
from ..core.head import HeadSpec
from ..core.training import now_iso
from .prompts import label_index
from .store import LabelItem, LabelStore

QUEUE_FILE = "queue.jsonl"
DEFAULT_WEIGHTS = {"uncertainty": 1.0, "disagreement": 0.5, "diversity": 0.3}
POOL_PER_PICK = 20          # the diversity pass looks at the best max(POOL_MIN, POOL_PER_PICK * n) candidates
POOL_MIN = 1000
DUPLICATE_EPS = 1e-9


# ----------------------------------------------------------------------------- the queue file

def queue_path(store: LabelStore) -> Path:
    return store.dir / QUEUE_FILE


def read_queue(store: LabelStore, head: str | None = None) -> list[dict]:
    p = queue_path(store)
    if not p.exists():
        return []
    out = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(r, dict) and (head is None or r.get("head") == head):
            out.append(r)
    return out


def write_queue(store: LabelStore, head: str, rows: list[dict]) -> Path:
    """Replace `head`'s rows in the queue file, keeping the other heads' rows (atomic write)."""
    keep = [r for r in read_queue(store) if r.get("head") != head]
    p = queue_path(store)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".jsonl.tmp")
    tmp.write_text("".join(json.dumps(r, ensure_ascii=False, default=str) + "\n" for r in [*keep, *rows]),
                   encoding="utf-8")
    tmp.replace(p)
    return p


# ----------------------------------------------------------------------------- scoring

def uncertainty(P: np.ndarray) -> np.ndarray:
    """In [0, 1]: binary (1 - max p) / 0.5, multiclass normalised entropy."""
    P = np.clip(np.asarray(P, float), 0.0, 1.0)
    k = P.shape[1]
    if k == 2:
        return np.clip((1.0 - P.max(axis=1)) / 0.5, 0.0, 1.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        h = -(np.where(P > 0, P * np.log(P), 0.0)).sum(axis=1)
    return np.clip(h / np.log(k), 0.0, 1.0)


def _standardise(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, float)
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)   # all-NaN columns ("mean of empty slice"): handled below
        mu = np.nanmean(X, axis=0)
        sd = np.nanstd(X, axis=0)
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sd = np.where(np.isfinite(sd) & (sd > 1e-12), sd, 1.0)
    Z = (X - mu) / sd
    return np.where(np.isfinite(Z), Z, 0.0)


def greedy_diverse(base: np.ndarray, X: np.ndarray, n: int, w_div: float) -> list[tuple[int, float, float]]:
    """Greedy k-center-weighted selection: (index, value at pick time, diversity term) in pick order.

    With w_div <= 0 this is a plain ranking by `base` (ties keep the input order)."""
    base = np.asarray(base, float)
    m = len(base)
    n = min(int(n), m)
    if n <= 0:
        return []
    if w_div <= 0:
        order = np.argsort(-base, kind="stable")[:n]
        return [(int(i), float(base[i]), 1.0) for i in order]
    pool = np.argsort(-base, kind="stable")[:max(POOL_MIN, POOL_PER_PICK * n)]
    Z = _standardise(X[pool])
    b = base[pool]
    d = np.full(len(pool), np.inf)
    avail = np.ones(len(pool), bool)
    picks: list[tuple[int, float, float]] = []
    for _ in range(n):
        if picks:
            finite = d[avail & np.isfinite(d)]
            scale = float(finite.max()) if finite.size and finite.max() > 0 else 1.0
            div = np.clip(d / scale, 0.0, 1.0)
        else:
            div = np.ones(len(pool))
        ok = avail & (d > DUPLICATE_EPS)
        if not ok.any():
            break
        val = np.where(ok, b + w_div * div, -np.inf)
        j = int(np.argmax(val))
        picks.append((int(pool[j]), float(val[j]), float(div[j])))
        avail[j] = False
        d = np.minimum(d, np.sqrt(((Z - Z[j]) ** 2).sum(axis=1)))
    return picks


def _features(service: Any, version: str, spec: HeadSpec, items: list[LabelItem],
              loaded: Any) -> tuple[np.ndarray, list[int], list[str]]:
    """Feature rows for the items; a state the provider refuses is dropped (one failing row never sinks the rest)."""
    states = [it.state for it in items]
    try:
        return service.features(version, spec, states, loaded), list(range(len(items))), []
    except ContractError:
        pass
    rows, keep, bad = [], [], []
    for i, s in enumerate(states):
        try:
            rows.append(service.features(version, spec, [s], loaded)[0])
            keep.append(i)
        except ContractError as e:
            bad.append(f"{items[i].id}: {e}")
    width = len(rows[0]) if rows else 0
    return (np.vstack(rows) if rows else np.zeros((0, width), np.float32)), keep, bad


# ----------------------------------------------------------------------------- next batch

def next_batch(project: Any, service: Any, head: str, n: int = 50, weights: dict | None = None,
               store: LabelStore | None = None, version: str | None = None, write: bool = True) -> list[dict]:
    """The `n` items of `head`'s view (without a human label for it) most worth a human label, best first.

    Each row: {head, item_id, rank, score, why, uncertainty, disagreement, diversity, model_version, trained, at}.
    Written to <home>/labels/queue.jsonl unless write=False. `weights` overrides DEFAULT_WEIGHTS
    (keys uncertainty, disagreement, diversity; diversity 0 = plain ranking)."""
    if store is None:
        store = LabelStore(project.home if project is not None else service.home)
    version = service.resolve(version)
    spec = service.spec(head, version)
    w = {**DEFAULT_WEIGHTS, **(weights or {})}
    human, judge = store.human_labels(head), store.judge_labels(head)
    cands = [it for it in store.items(spec.view) if (str(it.id), head) not in human]
    if not cands:
        if write:
            write_queue(store, head, [])
        return []
    loaded = service.loaded(version, head)
    X, keep, _bad = _features(service, version, spec, cands, loaded)
    cands = [cands[i] for i in keep]
    if not cands:
        if write:
            write_queue(store, head, [])
        return []

    jl = [judge.get((str(it.id), head)) for it in cands]
    j_idx = np.full(len(cands), -1)
    j_conf = np.full(len(cands), np.nan)
    for i, r in enumerate(jl):
        if r is None:
            continue
        try:
            j_idx[i] = label_index(spec, r.get("label"))
        except (ValueError, TypeError):
            continue
        c = r.get("confidence")
        if isinstance(c, (int, float)) and not isinstance(c, bool) and 0.0 <= float(c) <= 1.0:
            j_conf[i] = float(c)

    trained = loaded is not None
    labels = spec.labels
    if trained:
        P = loaded.ensemble.predict_proba(X)
        unc = uncertainty(P)
        am = P.argmax(axis=1)
        dis = np.where((j_idx >= 0) & (j_idx != am), np.where(np.isnan(j_conf), 0.5, j_conf), 0.0)
    else:
        P = None
        unc = np.where(j_idx < 0, 1.0, np.where(np.isnan(j_conf), 0.5, 1.0 - np.nan_to_num(j_conf)))
        dis = np.zeros(len(cands))
    base = float(w["uncertainty"]) * unc + float(w["disagreement"]) * dis
    picks = greedy_diverse(base, X, n, float(w["diversity"]))

    at = now_iso()
    rows = []
    for rank, (i, val, div) in enumerate(picks, start=1):
        why = []
        if trained:
            k = int(am[i])
            why.append(f"model: P({labels[k]})={P[i, k]:.2f} (uncertainty {unc[i]:.2f})")
            if dis[i] > 0:
                jc = "no confidence" if np.isnan(j_conf[i]) else f"confidence {j_conf[i]:.2f}"
                why.append(f"judge says {labels[j_idx[i]]!r} ({jc}), model says {labels[k]!r}")
        elif j_idx[i] >= 0:
            conf = "no confidence" if np.isnan(j_conf[i]) else f"confidence {j_conf[i]:.2f}"
            why.append(f"head not trained in {version}: judge says {labels[j_idx[i]]!r} with {conf}")
        else:
            why.append(f"head not trained in {version}: no judge label")
        if w["diversity"] > 0 and rank > 1 and div < 0.25:
            why.append("close to an item already in this batch")
        rows.append({"head": head, "item_id": str(cands[i].id), "rank": rank, "score": round(val, 6),
                     "why": "; ".join(why), "uncertainty": round(float(unc[i]), 6),
                     "disagreement": round(float(dis[i]), 6), "diversity": round(div, 6),
                     "model_version": version, "trained": trained, "at": at})
    if write:
        write_queue(store, head, rows)
    return rows
