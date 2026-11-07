"""Retrieval and chunk features: fixed keys, NaN for anything the candidates do not say, never an exception."""
from __future__ import annotations

import math

import pytest

from open_rlcd.features.retrieval import (
    CHUNK_FEATURES,
    RETRIEVAL_FEATURES,
    chunk_features,
    retrieval_features,
    to_epoch,
)

DAY = 86400
NOW = 1_700_000_000

FULL = [
    {"id": "a#0", "doc_id": "a", "text": "Northwind Metals CEO is Ingrid Solberg", "score": 0.82, "source": "hybrid",
     "sources": ["vector", "graph"], "rank": 1, "updated_at": NOW - 2 * DAY, "authority": 0.9,
     "graph": {"path_length": 2, "degree": 5}, "lexical_score": 4.0},
    {"id": "b#0", "doc_id": "b", "text": "Zentrix X1 parts are supplied by Northwind Metals", "score": 0.80,
     "source": "vector", "rank": 2, "updated_at": "2023-11-14T22:13:20Z", "authority": 0.5},
    {"id": "c#3", "doc_id": "c", "text": "Bluebird Bakery sells bread", "score": 0.31, "source": "lexical",
     "rank": 3, "updated_at": (NOW - 30 * DAY) * 1000},
    {"id": "b#1", "doc_id": "b", "text": "The X1 range is 12.5 km", "score": 0.55, "source": "graph", "rank": 4,
     "graph": {"path_length": 1, "degree": 3}},
]


def nan(x: float) -> bool:
    return isinstance(x, float) and math.isnan(x)


def test_to_epoch():
    assert to_epoch(NOW) == NOW and to_epoch(NOW * 1000) == NOW and to_epoch(str(NOW)) == NOW
    assert to_epoch("1970-01-02") == DAY and to_epoch("1970-01-01T00:00:10Z") == 10
    assert to_epoch("1970-01-01T09:00:00+09:00") == 0
    for bad in (None, "", "yesterday", float("nan"), True, [1]):
        assert to_epoch(bad) is None


@pytest.mark.parametrize("cands", [[], None, "garbage", [None, 3, "x", {"no_id": 1}]])
def test_retrieval_features_empty(cands):
    f = retrieval_features("anything", cands)
    assert list(f) == RETRIEVAL_FEATURES and len(set(RETRIEVAL_FEATURES)) == len(RETRIEVAL_FEATURES)
    assert f["r_n_candidates"] == 0.0 and f["r_n_docs"] == 0.0
    assert all(nan(v) for k, v in f.items() if k not in ("r_n_candidates", "r_n_docs"))


def test_retrieval_features_partial_candidates():
    f = retrieval_features("bakery bread", [{"id": "x", "text": "Bluebird Bakery sells bread"}, {"id": "y"},
                                            {"id": "z", "score": "not a number", "authority": "high",
                                             "updated_at": "someday", "graph": "nope", "source": "carrier pigeon"}])
    assert list(f) == RETRIEVAL_FEATURES
    assert f["r_n_candidates"] == 3.0 and f["r_n_docs"] == 3.0
    assert f["r_coverage_union"] == 1.0 and f["r_coverage_max"] == 1.0
    for k in ("r_top1", "r_top2", "r_gap12", "r_mean_topk", "r_entropy_topk", "r_share_vector",
              "r_agree_vector_graph", "r_age_min_days", "r_authority_mean", "r_path_len_min", "r_degree_mean"):
        assert nan(f[k]), k


def test_retrieval_features_full():
    f = retrieval_features("Who is the CEO of Northwind Metals?", FULL, now=NOW, k=10)
    assert f["r_n_candidates"] == 4 and f["r_n_docs"] == 3
    assert f["r_top1"] == pytest.approx(0.82) and f["r_top2"] == pytest.approx(0.80)
    assert f["r_gap12"] == pytest.approx(0.02)
    assert f["r_mean_topk"] == pytest.approx((0.82 + 0.80 + 0.31 + 0.55) / 4)
    assert 0.0 < f["r_entropy_topk"] < 1.0
    assert f["r_share_hybrid"] == f["r_share_vector"] == f["r_share_lexical"] == f["r_share_graph"] == 0.25
    # vector docs {a, b}; graph docs {a, b} (b#1 is graph); lexical docs {c}
    assert f["r_agree_vector_graph"] == 1.0 and f["r_agree_vector_lexical"] == 0.0
    assert f["r_age_min_days"] == pytest.approx(0.0, abs=1e-6)   # b was updated exactly at NOW
    assert f["r_age_mean_days"] == pytest.approx((2 + 0 + 30) / 3)
    assert f["r_authority_mean"] == pytest.approx(0.7) and f["r_authority_max"] == 0.9
    assert f["r_path_len_min"] == 1 and f["r_path_len_mean"] == 1.5 and f["r_degree_mean"] == 4
    assert f["r_coverage_union"] == 1.0 and 0 < f["r_coverage_max"] <= 1.0


def test_retrieval_features_topk_and_relative_freshness():
    f = retrieval_features("x", FULL, k=1)
    assert f["r_mean_topk"] == pytest.approx(0.82) and f["r_entropy_topk"] == 0.0
    assert f["r_share_hybrid"] == 1.0
    assert f["r_age_min_days"] == pytest.approx(2.0)                # newest timestamp in the set is the reference
    one_big = retrieval_features("x", [{"id": "a", "score": 0.9}, {"id": "b", "score": 0.1}, {"id": "c", "score": 0.1}])
    flat = retrieval_features("x", [{"id": "a", "score": 0.5}, {"id": "b", "score": 0.5}, {"id": "c", "score": 0.5}])
    assert one_big["r_entropy_topk"] < 0.01 and flat["r_entropy_topk"] == pytest.approx(1.0)


def test_retrieval_features_bare_hybrid_counts_for_every_source():
    f = retrieval_features("x", [{"id": "a#0", "doc_id": "a", "source": "hybrid"},
                                 {"id": "b#0", "doc_id": "b", "source": "vector"}])
    assert f["r_agree_vector_graph"] == pytest.approx(0.5) and f["r_agree_vector_lexical"] == pytest.approx(0.5)


def test_chunk_features_full():
    f = chunk_features("Who is the CEO of Northwind Metals?", FULL[3], FULL, now=NOW)
    assert list(f) == CHUNK_FEATURES and len(set(CHUNK_FEATURES)) == len(CHUNK_FEATURES)
    assert f["c_score"] == 0.55 and f["c_rank"] == 4 and f["c_rank_share"] == 1.0
    assert f["c_gap_top"] == pytest.approx(0.82 - 0.55)
    assert f["c_zscore"] < 0
    assert (f["c_src_vector"], f["c_src_graph"], f["c_src_lexical"], f["c_src_hybrid"]) == (0.0, 1.0, 0.0, 0.0)
    assert nan(f["c_age_days"]) and nan(f["c_authority"]) and nan(f["c_lexical_score"])
    assert f["c_path_len"] == 1 and f["c_degree"] == 3
    assert f["c_coverage"] == 0.0 and f["c_log_chars"] == pytest.approx(math.log1p(len(FULL[3]["text"])))
    top = chunk_features("Northwind Metals CEO", FULL[0], FULL, now=NOW)
    assert top["c_gap_top"] == 0.0 and top["c_zscore"] > 0 and top["c_src_hybrid"] == 1.0
    assert top["c_age_days"] == pytest.approx(2.0) and top["c_authority"] == 0.9 and top["c_lexical_score"] == 4.0


def test_chunk_features_rank_inferred_and_missing_data():
    cands = [{"id": "a", "score": 0.9}, {"id": "b", "score": 0.4}, {"id": "c", "score": 0.7}]
    f = chunk_features("q", {"id": "c", "score": 0.7}, cands)
    assert f["c_rank"] == 2 and f["c_rank_share"] == pytest.approx(2 / 3)
    lone = chunk_features("q", {"id": "z"})
    assert list(lone) == CHUNK_FEATURES
    assert lone["c_rank"] == 1 and all(nan(v) for k, v in lone.items() if k not in ("c_rank", "c_rank_share"))
    for bad in ({}, None, "x", {"id": "q", "score": "?", "graph": [1], "updated_at": "never"}):
        out = chunk_features("q", bad, [None, {"id": 1}])
        assert list(out) == CHUNK_FEATURES
    single = chunk_features("q", {"id": "a", "score": 0.5}, [{"id": "a", "score": 0.5}])
    assert single["c_gap_top"] == 0.0 and nan(single["c_zscore"])
