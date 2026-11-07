"""DocumentsProvider: feature names / shapes, rebuild from a bundle config, validation (422), answer support.
Offline: the hashing encoder only."""
from __future__ import annotations

import numpy as np
import pytest

from open_rlcd.core.config import Project
from open_rlcd.core.contract import ContractError
from open_rlcd.features.lexical import PAIR_FEATURES, QUERY_FEATURES
from open_rlcd.features.retrieval import CHUNK_FEATURES, RETRIEVAL_FEATURES
from open_rlcd.features.text import HashingEncoder, make_encoder
from open_rlcd.packs.base import FeatureProvider
from open_rlcd.packs.documents import VIEWS, DocumentsPack, DocumentsProvider
from open_rlcd.packs.documents.provider import A_SUPPORT, RISK, dates, entity_coverage, split_sentences

DIM = 64
ENC = {"kind": "hashing", "dim": DIM}
CANDS = [
    {"id": "p1#0", "doc_id": "p1", "title": "Velobox 300 product sheet", "score": 0.71, "source": "hybrid",
     "sources": ["graph", "lexical", "vector"], "rank": 1, "lexical_score": 7.2, "updated_at": "2024-11-02",
     "authority": 0.8, "graph": {"path_length": 1, "degree": 4},
     "text": "The Velobox 300 (SKU KV-4821) is a parcel locker. It sells for 1,250 USD and comes with a 24-month "
             "warranty. The product is owned by the Parcel Systems team."},
    {"id": "p2#0", "doc_id": "p2", "title": "Arcternix Hub product sheet", "score": 0.52, "source": "vector",
     "rank": 2, "updated_at": 1717200000, "authority": 0.8,
     "text": "The Arcternix Hub (SKU KV-1180) is a vehicle tracker. It sells for 320 USD."},
    {"id": "t1#0", "doc_id": "t1", "title": "Parcel Systems team", "score": 0.33, "source": "graph", "rank": 3,
     "graph": {"path_length": 2, "degree": 3},
     "text": "The Parcel Systems team is part of the Engineering department. Its team lead is Mara Okonkwo."},
]
Q = "What is the price of the Velobox 300?"
STATES = {
    "query": [{"query": Q, "candidates": CANDS, "now": "2025-03-01T00:00:00Z"}, {"query": "Hello there"}],
    "query_chunk": [{"query": Q, "chunk": c, "candidates": CANDS} for c in CANDS] + [{"query": Q, "chunk": CANDS[0]}],
    "query_evidence": [{"query": Q, "evidence": CANDS[:1], "candidates": CANDS}, {"query": Q, "evidence": []},
                       {"query": Q, "evidence": ["Velobox 300 costs 1,250 USD."]}],
    "query_evidence_answer": [{"query": Q, "evidence": CANDS[:1], "answer": "The Velobox 300 costs 1,250 USD."},
                              {"query": Q, "evidence": CANDS[:1], "answer": "The Velobox 300 costs 1,450 USD."}],
}


@pytest.fixture
def project(tmp_path):
    return Project(root=tmp_path, data={"pack": "documents", "home": str(tmp_path / "var"),
                                        "pack_config": {"encoder": ENC}})


@pytest.fixture
def provider(tmp_path):
    return DocumentsProvider(make_encoder(ENC, tmp_path / "var"))


def test_protocol_names_and_groups(provider):
    assert isinstance(provider, FeatureProvider)
    d = DIM
    expect = {"query": d + len(QUERY_FEATURES) + len(RETRIEVAL_FEATURES),
              "query_chunk": d + 1 + len(PAIR_FEATURES) + 1 + len(CHUNK_FEATURES),
              "query_evidence": d + 9 + len(RETRIEVAL_FEATURES),
              "query_evidence_answer": d + 1 + len(A_SUPPORT) + len(RISK)}
    for view in VIEWS:
        names = provider.names(view)
        assert len(names) == expect[view] == len(set(names))
        groups = provider.groups(view)
        assert set(groups) == set(names)
        assert set(groups.values()) <= {"embedding", "lexical", "retrieval", "answer_support"}
        assert provider.describe(view)["example"].startswith("{")
    assert provider.names("query")[:2] == ["emb_0", "emb_1"]
    assert provider.groups("query_evidence_answer")["a_num_share"] == "answer_support"
    assert provider.groups("query_chunk")["c_rank"] == "retrieval"
    assert provider.feature_version.startswith("documents-v1:hashing-64-s0:")
    cfg = provider.to_config()
    assert cfg["version"] == 1 and cfg["encoder"]["kind"] == "hashing" and cfg["encoder"]["dim"] == DIM
    assert cfg["views"] == {v: len(provider.names(v)) for v in VIEWS}
    with pytest.raises(ContractError, match="unknown view"):
        provider.names("nope")


def test_transform_shapes_are_stable_and_rebuild_identically(provider, project):
    rebuilt = DocumentsPack().provider(project, provider.to_config())
    fresh = DocumentsProvider(HashingEncoder(DIM), {"cache_size": 0})       # no cache, no CachedEncoder
    assert rebuilt.feature_version == provider.feature_version == fresh.feature_version
    for view, states in STATES.items():
        X = provider.transform(view, states)
        assert X.dtype == np.float32 and X.shape == (len(states), len(provider.names(view)))
        assert not np.isinf(X).any()
        np.testing.assert_array_equal(X, provider.transform(view, states))          # cache hit
        np.testing.assert_allclose(X, rebuilt.transform(view, states), atol=1e-6, equal_nan=True)
        np.testing.assert_allclose(X, fresh.transform(view, states), atol=1e-6, equal_nan=True)
        assert provider.names(view) == rebuilt.names(view)
    assert provider.transform("query", []).shape == (0, len(provider.names("query")))


def test_features_say_what_they_should(provider):
    names = provider.names("query_evidence_answer")
    X = provider.transform("query_evidence_answer", STATES["query_evidence_answer"])
    good, bad = X[0], X[1]
    col = {n: i for i, n in enumerate(names)}
    assert good[col["a_num_share"]] == 1.0 and good[col["a_num_unsupported"]] == 0.0
    assert bad[col["a_num_unsupported"]] == 1.0 and bad[col["a_num_local_support"]] < 1.0
    assert good[col["q_risk_money"]] == 1.0 and good[col["q_risk_safety"]] == 0.0
    qc = provider.names("query_chunk")
    Xc = provider.transform("query_chunk", STATES["query_chunk"][:2])
    cc = {n: i for i, n in enumerate(qc)}
    assert Xc[0, cc["qc_entity_coverage"]] == 1.0 and Xc[1, cc["qc_entity_coverage"]] == 0.0
    assert Xc[0, cc["p_bm25"]] > Xc[1, cc["p_bm25"]] and Xc[0, cc["c_rank"]] == 1.0
    qe = provider.names("query_evidence")
    Xe = provider.transform("query_evidence", STATES["query_evidence"][:2])
    ce = {n: i for i, n in enumerate(qe)}
    assert Xe[0, ce["e_n_evidence"]] == 1 and Xe[0, ce["e_n_candidates"]] == 3
    assert Xe[1, ce["e_n_evidence"]] == 0 and np.isnan(Xe[1, ce["e_cos_max"]]) and np.isnan(Xe[1, ce["qe_0"]])
    # the /v1/rank shape: the chunk's fields next to the query instead of under "chunk"
    flat = provider.transform("query_chunk", [{"query": Q, **CANDS[0], "candidates": CANDS}])
    np.testing.assert_allclose(flat, Xc[:1], equal_nan=True)


def test_one_encoder_call_per_kind(tmp_path):
    calls = []

    class Counting(HashingEncoder):
        def encode(self, texts, kind="passage"):
            calls.append((kind, len(texts)))
            return super().encode(texts, kind)

    p = DocumentsProvider(Counting(DIM), {"cache_size": 0})
    p.transform("query_evidence_answer", STATES["query_evidence_answer"] * 3)
    assert sorted(k for k, _ in calls) == ["passage", "query"]
    assert dict(calls)["query"] == 1                       # the same query once, however many states carry it


@pytest.mark.parametrize("view,state,needle", [
    ("query", "just a string", "must be an object"),
    ("query", {"candidates": []}, "query is required"),
    ("query", {"query": "   "}, "non-empty"),
    ("query", {"query": 3}, "query must be a string"),
    ("query", {"query": "q", "candidates": "abc"}, "candidates must be an array"),
    ("query", {"query": "q", "candidates": [3]}, "candidates[0]"),
    ("query", {"query": "q", "candidates": [{"id": "a"}] * 201}, "at most 200"),
    ("query", {"query": "q", "now": "yesterday-ish"}, "now must be ISO"),
    ("query_chunk", {"query": "q"}, "chunk is required"),
    ("query_chunk", {"query": "q", "chunk": {"id": "a", "text": 5}}, "chunk[0].text must be a string"),
    ("query_evidence", {"query": "q"}, "evidence is required"),
    ("query_evidence_answer", {"query": "q", "evidence": []}, "answer is required"),
    ("query_evidence_answer", {"query": "q", "evidence": [], "answer": ""}, "non-empty"),
])
def test_bad_states_are_422_with_a_precise_message(provider, view, state, needle):
    with pytest.raises(ContractError) as e:
        provider.transform(view, [STATES[view][0], state])
    assert e.value.status == 422 and needle in e.value.message
    assert e.value.details["state_index"] == 1 or "must be an object" in e.value.message


def test_bundle_config_mismatch_is_refused(provider, project):
    cfg = provider.to_config()
    with pytest.raises(ContractError) as e:
        DocumentsPack().provider(project, {**cfg, "encoder": {**cfg["encoder"], "sha256": "0" * 64}})
    assert e.value.kind == "encoder_mismatch" and "different encoder" in e.value.message
    with pytest.raises(ContractError) as e:
        DocumentsPack().provider(project, {**cfg, "views": {**cfg["views"], "query": 3}})
    assert e.value.kind == "feature_schema_mismatch"
    with pytest.raises(ContractError, match="version"):
        DocumentsPack().provider(project, {**cfg, "version": 9})


def test_text_helpers():
    assert split_sentences("It weighs 3.2 kg. Price: 1,250 USD!\nNext line") == \
        ["It weighs 3.2 kg.", "Price: 1,250 USD!", "Next line"]
    assert split_sentences("価格は1,250 USDです。保証は24か月です。") == ["価格は1,250 USDです。", "保証は24か月です。"]
    # a Latin name glued to Japanese text is still an entity (the graph's patterns need a word boundary)
    assert entity_coverage("Robotics Labの製品は？", "担当チームはRobotics Labです。") == 1.0
    assert entity_coverage("Robotics Labの製品は？", "担当チームはCold Chainです。") == 0.0
    assert np.isnan(entity_coverage("what does it cost?", "anything"))
    assert dates("launched on 2023-04-12 and 2024年3月1日") == {(2023, 4, 12), (2024, 3, 1)}
