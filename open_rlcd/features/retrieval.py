"""Retrieval signals: how strong, consistent, fresh and on-topic a set of retrieved candidates is.

A candidate is a plain dict; only `id` is required:

    {id, text?, score? (ANN / cosine similarity), source? ("vector" | "graph" | "lexical" | "hybrid"),
     sources? (list of the sources that found it, when "hybrid"), rank? (1 = best), doc_id?,
     updated_at? (ISO 8601 or epoch seconds / ms), authority? (0-1), lexical_score? (BM25),
     graph? {path_length?, degree?}}

`retrieval_features` summarises the whole set (for `answerable`, `route`, ...); `chunk_features` describes one
candidate relative to the others (for `chunk_relevant`). Both return the same keys in the same order every time
and use NaN for anything the candidates do not say; they never raise on missing or malformed fields. NaN is
"unknown" to the heads (imputed at training), never a fake zero.

What these features do not do: they do not read the documents again or call any store; they only summarise what
the retriever returned.
"""
from __future__ import annotations

import math
from datetime import UTC, datetime
from typing import Any

import numpy as np

from .lexical import coverage

NAN = float("nan")
SOURCES = ("vector", "graph", "lexical", "hybrid")
SOFTMAX_TAU = 0.05      # cosine scores live in [-1, 1]; a raw softmax would be flat, so sharpen it


def to_epoch(ts: Any) -> float | None:
    """Epoch seconds from epoch seconds / milliseconds or an ISO 8601 string (naive = UTC); None if unknown."""
    if ts is None or ts == "" or isinstance(ts, bool):
        return None
    if isinstance(ts, (int, float, np.integer, np.floating)):
        v = float(ts)
        if not math.isfinite(v):
            return None
        return v / 1000.0 if abs(v) > 1e11 else v
    s = str(ts).strip()
    try:
        return to_epoch(float(s))
    except ValueError:
        pass
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.timestamp()


def _num(x: Any) -> float:
    if x is None or isinstance(x, bool):
        return NAN
    try:
        v = float(x)
    except (TypeError, ValueError):
        return NAN
    return v if math.isfinite(v) else NAN


def _clean(candidates: Any) -> list[dict]:
    if not isinstance(candidates, (list, tuple)):
        return []
    return [c for c in candidates if isinstance(c, dict) and c.get("id") is not None]


def _ordered(cands: list[dict]) -> list[dict]:
    """Best first: by `rank` when every candidate has one, else by `score` (NaN last), else as given."""
    ranks = [_num(c.get("rank")) for c in cands]
    if cands and all(not math.isnan(r) for r in ranks):
        return [c for _, _, c in sorted(zip(ranks, range(len(cands)), cands, strict=True), key=lambda t: t[:2])]
    scores = [_num(c.get("score")) for c in cands]
    if any(not math.isnan(s) for s in scores):
        keyed = [(-s if not math.isnan(s) else math.inf, i) for i, s in enumerate(scores)]
        return [cands[i] for _, i in sorted(keyed)]
    return list(cands)


def _sources(c: dict) -> set[str]:
    """The sources that found a candidate; a bare "hybrid" (no `sources` list) counts as every source."""
    srcs = c.get("sources")
    if isinstance(srcs, (list, tuple, set)) and srcs:
        return {str(s) for s in srcs if str(s) in SOURCES and s != "hybrid"} or {"vector", "graph", "lexical"}
    s = c.get("source")
    if s == "hybrid":
        return {"vector", "graph", "lexical"}
    return {str(s)} if s in SOURCES else set()


def _doc(c: dict) -> str:
    return str(c.get("doc_id") if c.get("doc_id") is not None else c.get("id"))


def _graph(c: dict, key: str) -> float:
    g = c.get("graph")
    return _num(g.get(key)) if isinstance(g, dict) else NAN


def _mean(xs: list[float]) -> float:
    xs = [x for x in xs if not math.isnan(x)]
    return float(np.mean(xs)) if xs else NAN


def _min(xs: list[float]) -> float:
    xs = [x for x in xs if not math.isnan(x)]
    return float(min(xs)) if xs else NAN


def _max(xs: list[float]) -> float:
    xs = [x for x in xs if not math.isnan(x)]
    return float(max(xs)) if xs else NAN


def _ref_time(cands: list[dict], now: int | float | None) -> float | None:
    if now is not None:
        return float(now)
    ts = [t for t in (to_epoch(c.get("updated_at")) for c in cands) if t is not None]
    return max(ts) if ts else None


def _age_days(c: dict, ref: float | None) -> float:
    t = to_epoch(c.get("updated_at"))
    if t is None or ref is None:
        return NAN
    return max(0.0, (ref - t) / 86400.0)


def _text(c: dict) -> str | None:
    t = c.get("text")
    return t if isinstance(t, str) and t.strip() else None


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return NAN
    return len(a & b) / len(a | b)


# ----------------------------------------------------------------------------- set-level features

RETRIEVAL_FEATURES: list[str] = [
    "r_n_candidates", "r_n_docs", "r_top1", "r_top2", "r_gap12", "r_mean_topk", "r_std_topk", "r_entropy_topk",
    "r_share_vector", "r_share_graph", "r_share_lexical", "r_share_hybrid",
    "r_agree_vector_graph", "r_agree_vector_lexical",
    "r_age_min_days", "r_age_mean_days", "r_authority_mean", "r_authority_max",
    "r_path_len_min", "r_path_len_mean", "r_degree_mean",
    "r_coverage_union", "r_coverage_max",
]


def retrieval_features(query: str, candidates: list[dict], now: int | None = None, k: int = 10) -> dict[str, float]:
    """Summary of a candidate set in `RETRIEVAL_FEATURES` order.

    * counts: candidates, distinct documents (an empty set is 0, not unknown);
    * scores: top-1, top-2, their gap, mean / std of the top-k scores, normalised entropy of a softmax
      (temperature 0.05) over the top-k scores: low = one clear winner, 1 = all alike;
    * sources: share of the top-k found by each source; agreement = Jaccard overlap of the documents in the
      vector top-k and the graph (resp. lexical) top-k;
    * freshness: min / mean age in days of the top-k relative to `now` (default: the newest timestamp in the
      set, so only relative age is measured); authority mean / max; graph path length min / mean, degree mean;
    * coverage: share of the query's content terms found in the union of the top-k texts, and the best single
      candidate's coverage.
    """
    f = dict.fromkeys(RETRIEVAL_FEATURES, NAN)
    try:
        cands = _clean(candidates)
        k = max(1, int(k))
        f["r_n_candidates"] = float(len(cands))
        f["r_n_docs"] = float(len({_doc(c) for c in cands}))
        if not cands:
            return f
        top = _ordered(cands)[:k]
        scores = sorted((s for s in (_num(c.get("score")) for c in cands) if not math.isnan(s)), reverse=True)
        if scores:
            f["r_top1"] = scores[0]
            if len(scores) > 1:
                f["r_top2"] = scores[1]
                f["r_gap12"] = scores[0] - scores[1]
            topk = np.asarray(scores[:k], dtype=np.float64)
            f["r_mean_topk"] = float(topk.mean())
            f["r_std_topk"] = float(topk.std())
            if len(topk) > 1:
                z = (topk - topk.max()) / SOFTMAX_TAU
                p = np.exp(z) / np.exp(z).sum()
                f["r_entropy_topk"] = float(-(p * np.log(np.clip(p, 1e-300, None))).sum() / math.log(len(topk)))
            else:
                f["r_entropy_topk"] = 0.0
        with_src = [c for c in top if c.get("source") in SOURCES]
        if with_src:
            for s in SOURCES:
                f[f"r_share_{s}"] = sum(1 for c in with_src if c.get("source") == s) / len(with_src)
        ordered_all = _ordered(cands)
        by_src = {s: [_doc(c) for c in ordered_all if s in _sources(c)][:k] for s in ("vector", "graph", "lexical")}
        f["r_agree_vector_graph"] = _jaccard(set(by_src["vector"]), set(by_src["graph"]))
        f["r_agree_vector_lexical"] = _jaccard(set(by_src["vector"]), set(by_src["lexical"]))
        ref = _ref_time(cands, now)
        ages = [_age_days(c, ref) for c in top]
        f["r_age_min_days"], f["r_age_mean_days"] = _min(ages), _mean(ages)
        auth = [_num(c.get("authority")) for c in top]
        f["r_authority_mean"], f["r_authority_max"] = _mean(auth), _max(auth)
        paths = [_graph(c, "path_length") for c in top]
        f["r_path_len_min"], f["r_path_len_mean"] = _min(paths), _mean(paths)
        f["r_degree_mean"] = _mean([_graph(c, "degree") for c in top])
        texts = [t for t in (_text(c) for c in top) if t]
        if texts:
            f["r_coverage_union"] = coverage(query, "\n".join(texts))
            f["r_coverage_max"] = max(coverage(query, t) for t in texts)
    except Exception:  # features must never take a request down; whatever was computed stays, the rest is NaN
        pass
    return f


# ----------------------------------------------------------------------------- per-candidate features

CHUNK_FEATURES: list[str] = [
    "c_score", "c_rank", "c_gap_top", "c_zscore", "c_rank_share",
    "c_src_vector", "c_src_graph", "c_src_lexical", "c_src_hybrid",
    "c_age_days", "c_authority", "c_path_len", "c_degree", "c_coverage", "c_log_chars", "c_lexical_score",
]


def chunk_features(query: str, chunk: dict, candidates: list[dict] | None = None,
                   now: int | None = None) -> dict[str, float]:
    """One candidate relative to its set, in `CHUNK_FEATURES` order.

    score; rank (given, else its position in the ordered set); gap to the best score in the set; z-score of its
    score among the set; rank share = rank / set size; source one-hots (NaN when the source is unknown); age in
    days (relative to `now`, else to the newest timestamp in the set); authority; graph path length and degree;
    query coverage; log(1 + characters); BM25 score when the retriever gave one."""
    f = dict.fromkeys(CHUNK_FEATURES, NAN)
    try:
        if not isinstance(chunk, dict):
            return f
        cands = _clean(candidates)
        cid = chunk.get("id")
        if cid is not None and not any(c.get("id") == cid for c in cands):
            cands = [*cands, chunk]
        score = _num(chunk.get("score"))
        f["c_score"] = score
        rank = _num(chunk.get("rank"))
        if math.isnan(rank) and cands and cid is not None:
            for i, c in enumerate(_ordered(cands)):
                if c.get("id") == cid:
                    rank = float(i + 1)
                    break
        f["c_rank"] = rank
        if cands and not math.isnan(rank):
            f["c_rank_share"] = rank / len(cands)
        scores = np.asarray([s for s in (_num(c.get("score")) for c in cands) if not math.isnan(s)], dtype=np.float64)
        if not math.isnan(score) and scores.size:
            f["c_gap_top"] = float(scores.max() - score)
            if scores.size > 1:
                sd = float(scores.std())
                f["c_zscore"] = float((score - scores.mean()) / sd) if sd > 1e-12 else 0.0
        src = chunk.get("source")
        if src in SOURCES:
            for s in SOURCES:
                f[f"c_src_{s}"] = float(src == s)
        f["c_age_days"] = _age_days(chunk, _ref_time(cands or [chunk], now))
        f["c_authority"] = _num(chunk.get("authority"))
        f["c_path_len"] = _graph(chunk, "path_length")
        f["c_degree"] = _graph(chunk, "degree")
        text = _text(chunk)
        if text is not None:
            f["c_coverage"] = coverage(query, text)
            f["c_log_chars"] = math.log1p(len(text))
        f["c_lexical_score"] = _num(chunk.get("lexical_score"))
    except Exception:  # see retrieval_features
        pass
    return f
