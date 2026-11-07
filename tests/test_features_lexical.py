"""Lexical signals: multilingual tokens, BM25, coverage, number normalisation / support, query and pair features."""
from __future__ import annotations

import json
import math

import numpy as np
import pytest

from open_rlcd.features.lexical import (
    BM25,
    PAIR_FEATURES,
    QUERY_FEATURES,
    bm25_scores,
    content_tokens,
    coverage,
    number_support,
    numbers,
    pair_features,
    query_features,
    tokens,
)

# ----------------------------------------------------------------------------- tokens

def test_tokens_english_casefold_and_punctuation():
    assert tokens("Acme's Q3-report: revenue, UP 12%!") == ["acme", "s", "q3", "report", "revenue", "up", "12"]
    assert tokens("... --- !!! ___") == []


def test_tokens_japanese_bigrams_and_single_chars():
    assert tokens("東京本社") == ["東京", "京本", "本社"]
    assert tokens("円") == ["円"]
    assert tokens("iPhone15は便利") == ["iphone15", "は便", "便利"]
    assert tokens("売上、利益。") == ["売上", "利益"]          # punctuation splits runs


def test_tokens_mongolian_cyrillic_words():
    assert tokens("Улаанбаатар хотод ХЭДЭН ажилтан байна вэ?") == ["улаанбаатар", "хотод", "хэдэн", "ажилтан",
                                                                 "байна", "вэ"]


def test_content_tokens_drop_stopwords_in_three_languages():
    assert content_tokens("What is the revenue of Acme?") == ["revenue", "acme"]
    assert content_tokens("Улаанбаатар хотод хэдэн ажилтан байна вэ") == ["улаанбаатар", "хотод", "ажилтан"]
    ja = content_tokens("東京の売上はいくらですか")
    assert "東京" in ja and "売上" in ja and "くら" not in ja and "です" not in ja


# ----------------------------------------------------------------------------- BM25

DOCS = ["Acme Robotics reported quarterly revenue growth",
        "Bluebird Bakery sells sourdough bread",
        "Revenue revenue revenue everywhere",
        "東京本社の従業員数は350人です",
        "Улаанбаатар дахь салбар 2019 онд нээгдсэн"]


def test_bm25_ranks_the_relevant_document_first():
    bm = BM25([tokens(d) for d in DOCS])
    s = bm.scores(tokens("Acme revenue growth"))
    assert int(np.argmax(s)) == 0 and s[1] == 0.0
    assert int(np.argmax(bm.scores(tokens("東京本社の従業員")))) == 3
    assert int(np.argmax(bm.scores(tokens("Улаанбаатар салбар")))) == 4
    assert np.all(bm.scores(tokens("nothing matches here zzz")) == 0.0)
    assert np.all(s >= 0.0)


def test_bm25_json_round_trip_and_empty():
    bm = BM25([tokens(d) for d in DOCS], k1=1.5, b=0.6)
    again = BM25.from_dict(json.loads(json.dumps(bm.to_dict())))
    q = tokens("revenue bread 東京")
    assert np.allclose(bm.scores(q), again.scores(q)) and again.k1 == 1.5 and again.b == 0.6
    assert BM25([]).scores(["x"]).shape == (0,)
    assert BM25.from_dict(BM25([]).to_dict()).scores(["x"]).shape == (0,)


def test_bm25_scores_helper():
    s = bm25_scores("sourdough bread", DOCS)
    assert s.shape == (len(DOCS),) and int(np.argmax(s)) == 1


# ----------------------------------------------------------------------------- coverage

def test_coverage():
    assert coverage("What was Acme revenue in 2024?", "In 2024 Acme revenue was 1,200") == 1.0
    assert coverage("Acme revenue bread", "Acme revenue") == pytest.approx(2 / 3)
    assert coverage("", "anything") == 0.0
    assert coverage("what is it", "it is") == pytest.approx(2 / 3)   # all stopwords: falls back to all tokens
    assert coverage("東京本社の従業員", "東京本社の従業員数は350人") == 1.0


# ----------------------------------------------------------------------------- numbers

@pytest.mark.parametrize("text, expected", [
    ("1,200", {"1200"}), ("1200", {"1200"}), ("１２００", {"1200"}), ("1 200", {"1", "200"}),
    ("12.5%", {"12.5%"}), ("12.50 %", {"12.5%"}), ("１２．５％", {"12.5%"}),
    ("120万円", {"1200000"}), ("1.2 million", {"1200000"}), ("3 сая төгрөг", {"3000000"}),
    ("12,5", {"12", "5"}), ("1,234,567.80", {"1234567.8"}), ("price 0.50", {"0.5"}),
    ("2024-03-15", {"2024", "3", "15"}), ("no digits", set()),
])
def test_numbers_normalisation(text, expected):
    assert numbers(text) == expected


def test_number_support():
    assert number_support("Revenue was 1,200 units, up 12.5%", "sales: 1200 units (+12.5%)") == (1.0, 0)
    share, missing = number_support("It costs 1,300 and weighs 12 kg", "It costs 1200 and weighs 12 kg")
    assert share == 0.5 and missing == 1
    assert number_support("No numbers here", "1 2 3") == (1.0, 0)
    assert number_support("growth of 12.5%", "growth was 12.5 percent") == (1.0, 0)   # bare value supports %
    assert number_support("従業員は３５０人", "従業員数は350人です") == (1.0, 0)


# ----------------------------------------------------------------------------- query features

def test_query_features_keys_fixed_and_defaults():
    f = query_features("")
    assert list(f) == QUERY_FEATURES and len(set(QUERY_FEATURES)) == len(QUERY_FEATURES)
    assert all(isinstance(v, float) for v in f.values())
    assert all(math.isfinite(v) for v in f.values())


@pytest.mark.parametrize("query, on", [
    ("Who manages the Oslo plant?", ["q_qw_who", "q_ends_question"]),
    ("When did the Ulaanbaatar branch open?", ["q_qw_when"]),
    ("How many employees work at HQ?", ["q_qw_how", "q_qw_how_many"]),
    ("Compare Acme and Zentrix revenue in 2024", ["q_cue_compare", "q_has_digits"]),
    ("List all suppliers of Northwind", ["q_cue_list"]),
    ("Please summarize the contract", ["q_cue_summary"]),
    ("Extract the invoice number and amount", ["q_cue_extract"]),
    ("I want a refund, this is unacceptable", ["q_cue_complaint"]),
    ("Our lawyer says we will sue for breach of contract", ["q_cue_legal"]),
    ("What is Ingrid's home address and phone number?", ["q_cue_personal_data", "q_qw_what"]),
    ('Find "supply chain risk" in the report', ["q_has_quotes"]),
    ("東京本社の従業員は何人ですか", ["q_qw_what", "q_qw_how_many", "q_question_particle"]),
    ("東京と大阪の売上を比較してください", ["q_cue_compare"]),
    ("議事録を要約して", ["q_cue_summary"]),
    ("いつ開業しましたか？", ["q_qw_when", "q_ends_question"]),
    ("Улаанбаатар салбарыг хэн удирддаг вэ?", ["q_qw_who", "q_question_particle"]),
    ("Салбарууд хэзээ нээгдсэн бэ", ["q_qw_when"]),
    ("Бүх нийлүүлэгчдийг жагсаа", ["q_cue_list"]),
    ("Хоёр гэрээг харьцуул", ["q_cue_compare"]),
    ("what about it?", ["q_has_pronoun", "q_ambiguous"]),
])
def test_query_feature_cues(query, on):
    f = query_features(query)
    for key in on:
        assert f[key] == 1.0, (query, key)


def test_query_features_scripts_and_negatives():
    f = query_features("Acme 東京 Улаанбаатар")
    assert f["q_share_latin"] > 0 and f["q_share_cjk"] > 0 and f["q_share_cyrillic"] > 0
    assert f["q_share_latin"] + f["q_share_cjk"] + f["q_share_cyrillic"] == pytest.approx(1.0)
    g = query_features("Describe the onboarding process for new warehouse staff")
    assert g["q_cue_legal"] == g["q_cue_complaint"] == g["q_cue_personal_data"] == 0.0
    assert g["q_qw_who"] == 0.0 and g["q_ambiguous"] == 0.0 and g["q_very_short"] == 0.0
    assert query_features("whole wholesale")["q_qw_who"] == 0.0          # exact words, not prefixes


# ----------------------------------------------------------------------------- pair features

def test_pair_features():
    f = pair_features('"supply chain" risk in 2024', "The supply chain risk report for 2024", bm25=3.5)
    assert list(f) == PAIR_FEATURES
    assert f["p_coverage"] == 1.0 and f["p_phrase_hit"] == 1.0 and f["p_bm25"] == 3.5
    assert f["p_numbers_overlap"] == 1.0 and f["p_query_has_numbers"] == 1.0 and f["p_same_script"] == 1.0
    g = pair_features("Acme revenue", "東京本社の従業員数は350人です")
    assert g["p_coverage"] == 0.0 and g["p_jaccard"] == 0.0 and g["p_same_script"] == 0.0 and g["p_bm25"] == 0.0
    assert g["p_numbers_overlap"] == 1.0 and g["p_query_has_numbers"] == 0.0
    h = pair_features("revenue in 2023", "revenue in 2024")
    assert h["p_numbers_overlap"] == 0.0 and h["p_phrase_hit"] == 0.0
    assert pair_features("", "")["p_coverage"] == 0.0
    assert all(math.isfinite(v) for v in pair_features("x", "y", bm25=float("nan")).values())
