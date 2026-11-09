"""Regression tests for the reviewed defects of the documents pack, its features and the local connectors (F07-F12,
F27, F30-F33, F39, F46). Each test fails on the code before its fix. Trained-model checks of the same fixes live in
test_documents_pack.py (they share its trained service)."""
from __future__ import annotations

import json
import time
from types import SimpleNamespace

import numpy as np
import pytest

from open_rlcd.connectors.local_graph import LocalGraphStore, common_katakana, entities
from open_rlcd.core import policy as PL
from open_rlcd.core.config import Project
from open_rlcd.core.contract import ContractError, DecideRequest
from open_rlcd.core.service import Service
from open_rlcd.features.lexical import number_support, numbers
from open_rlcd.features.retrieval import to_epoch
from open_rlcd.features.text import make_encoder
from open_rlcd.packs.documents import DocumentsPack
from open_rlcd.packs.documents import decide as D
from open_rlcd.packs.documents.corpus import Corpus
from open_rlcd.packs.documents.provider import DocumentsProvider, _risk_cues, describe
from open_rlcd.packs.documents.search import candidates
from open_rlcd.packs.documents.synth import EMPTY_EVIDENCE_SHARE, Generator, partial_level, write_synth
from open_rlcd.packs.documents.world import build_world

ENC = {"kind": "hashing", "dim": 256}


# ----------------------------------------------------------------------------- a scripted service for decide()

def _dec(cls, p, tier, calibrated=True, max_tier="act", uncapped=None):
    return {"class": cls, "p": p, "tier": tier, "uncapped_tier": uncapped or tier, "max_tier": max_tier,
            "calibrated": calibrated, "warnings": []}


class Scripted:
    """Just enough of core.service.Service for the documents pack's decide() / evaluate(): every head answers
    what `script` says, and every state a head saw is kept in `seen`."""

    def __init__(self, script: dict, relevance: list[float] | None = None, guard: bool = False):
        self.script, self.relevance, self.guard = script, relevance, guard
        self.seen: dict[str, list] = {}
        self.project = SimpleNamespace(home=None, pack_config={}, get=lambda key, default=None: default)
        self.logs = SimpleNamespace(decision=lambda decision_id: None)
        self.logged: list[dict] = []

    def resolve(self, model):
        return "rlcd-0.1.1"

    def specs(self, version):
        return {n: SimpleNamespace(name=n) for n in (*D.DECIDE_HEADS, *D.EVALUATE_HEADS)}

    def policy(self, version):
        return PL.PolicyConfig()

    def probabilities(self, version, spec, states, warnings=None):
        p = np.asarray(self.relevance if self.relevance is not None else [0.9] * len(states))
        return np.c_[1 - p, p], True

    def head_decision(self, version, spec, state, *, stakes=None, explore=False, warnings=None, **kw):
        self.seen.setdefault(spec.name, []).append(state)
        out = dict(self.script[spec.name])
        if self.guard and spec.name == "answerable" and warnings is not None:
            warnings.append("head 'answerable': the request lacks 261 input feature(s) that almost every training "
                            "row had (embedding, retrieval; e.g. qe_0): ... Send the same fields as in training")
        out["decision"] = {**out["decision"], "stakes": stakes}
        return out

    def record(self, version, kind, heads, extra=None, **receipt):
        return "d-1"

    # the control-layer helpers decide() / evaluate() call: no filters, no rollout, a record kept in memory
    def filter_options(self, specs, request, heads=None):
        return {}, {}

    def attach_contract(self, result, head, allowed, considered):
        return None

    def head_log(self, state, r, **extra):
        d = r["decision"]
        return {"state": state, "class": d["class"], "p": d["p"], "tier": d["tier"], **extra}

    def rollout_summary(self, results, action, cfg, proposed_action=None):
        return {"mode": "full", "capped": False, "proposed_action": proposed_action or action}

    @staticmethod
    def proposed_tier(d, cfg):
        return d["tier"]

    def decision_contract(self, kind, heads, request, **params):
        return {"digest": None, "battery": None}

    def log_decision(self, version, kind, heads, extra=None, **receipt):
        row = {"decision_id": "d-1", "kind": kind, "model": version, "heads": heads, **(extra or {}),
               "receipt": {"digest": None}}
        self.logged.append({**row, **receipt})
        return row


def _answering(**over):
    s = {"response_type": {"answer": {}, "decision": _dec("direct_answer", 0.95, "act")},
         "route": {"answer": {}, "decision": _dec("vector", 0.9, "act")},
         "needs_graph": {"answer": {}, "decision": _dec("false", 0.9, "act")},
         "chunk_relevant": {"answer": {}, "decision": _dec("true", 0.9, "act")},
         "answerable": {"answer": {}, "decision": _dec("true", 0.97, "act")}}
    s.update(over)
    return s


CANDS = [{"id": "a#0", "text": "The Velobox 300 costs 3,900 USD.", "updated_at": "2025-01-01T00:00:00Z"},
         {"id": "b#0", "text": "Unrelated text.", "updated_at": "2024-06-01T00:00:00Z"}]


# ----------------------------------------------------------------------------- F07: `now` defaults to the request time

def test_f07_decide_sends_now_to_every_view_when_the_request_has_none():
    svc = Scripted(_answering())
    t0 = time.time()
    DocumentsPack().decide(svc, {"state": {"query": "What is the price of the Velobox 300?", "candidates": CANDS}})
    for head in ("response_type", "answerable"):
        now = to_epoch(svc.seen[head][0]["now"])
        assert now is not None and abs(now - t0) < 120, (head, svc.seen[head][0].get("now"))
    # the caller's own `now` wins
    svc2 = Scripted(_answering())
    DocumentsPack().decide(svc2, {"state": {"query": "q", "candidates": CANDS, "now": "2025-03-01T00:00:00Z"}})
    assert svc2.seen["route"][0]["now"] == "2025-03-01T00:00:00Z"
    assert "request" in describe("query")["state"]["now"]


def test_f07_age_features_are_absolute_with_the_default_now(tmp_path):
    svc = Scripted(_answering())
    DocumentsPack().decide(svc, {"state": {"query": "Velobox 300 price?", "candidates": CANDS}})
    prov = DocumentsProvider(make_encoder(ENC, tmp_path))
    names = prov.names("query")
    x = prov.transform("query", [svc.seen["response_type"][0]])[0]
    age = x[names.index("r_age_min_days")]
    expected = (time.time() - to_epoch(CANDS[0]["updated_at"])) / 86400
    assert abs(age - expected) < 1.0          # not 0.0 (= relative to the newest candidate, never seen in training)


# ----------------------------------------------------------------------------- F30 / F39: empty evidence

def test_f30_no_candidate_reaches_min_relevance_plan_retrieves_more_without_blaming_the_caller():
    svc = Scripted(_answering(), relevance=[0.43, 0.10], guard=True)
    d = DocumentsPack().decide(svc, {"state": {"query": "How much does standard shipping cost?", "candidates": CANDS}})
    assert d["plan"]["use_chunks"] == [] and d["heads"]["chunk_relevant"]["kept"] == 0
    assert svc.seen["answerable"][0]["evidence"] == []
    assert d["action"] == "escalate" and d["plan"]["next"] == "retrieve_more" and d["plan"]["call_llm"] is False
    assert "min_relevance" in d["plan"]["reason"] and "0.43" in d["plan"]["reason"]
    assert "answerable" in d["plan"]["chain"]
    assert not any("Send the same fields" in w for w in d["warnings"]), d["warnings"]
    assert any("empty evidence set" in w for w in d["warnings"])


def test_f30_no_candidates_at_all():
    svc = Scripted(_answering())
    d = DocumentsPack().decide(svc, {"state": {"query": "q", "candidates": []}})
    assert d["action"] == "escalate" and d["plan"]["next"] == "retrieve_more" and d["plan"]["call_llm"] is False


def test_f30_synthetic_data_trains_answerable_on_empty_evidence(tmp_path):
    info = write_synth(tmp_path / "home", ENC, seed=7, n_queries=300)
    from open_rlcd.labeling.store import LabelStore

    store = LabelStore(tmp_path / "home")
    items = store.items("query_evidence")
    empty = [i for i in items if i.state.get("evidence") == []]
    labels = store.human_labels("answerable")
    assert len(empty) / len(items) > 0.05           # above the input guard's 5 %: the state is a training state
    assert all(labels[(i.id, "answerable")]["label"] in (False, "false") for i in empty)
    assert EMPTY_EVIDENCE_SHARE > 0 and info["labels"]["answerable"]["classes"]["false"] >= len(empty)


# ----------------------------------------------------------------------------- F31: abstained response_type

@pytest.mark.parametrize("cls", ["out_of_scope", "clarify_question", "escalate_human"])
def test_f31_next_does_not_follow_an_abstained_response_type(cls):
    svc = Scripted(_answering(response_type={"answer": {}, "decision": _dec(cls, 0.39, "escalate")}))
    d = DocumentsPack().decide(svc, {"state": {"query": "標準配送の料金はいくらですか？", "candidates": CANDS}})
    assert d["action"] == "escalate" and d["plan"]["next"] == "escalate_to_human" and d["plan"]["call_llm"] is False
    assert "abstained" in d["plan"]["reason"]
    sure = Scripted(_answering(response_type={"answer": {}, "decision": _dec(cls, 0.9, "confirm")}))
    d2 = DocumentsPack().decide(sure, {"state": {"query": "q", "candidates": CANDS}})
    assert d2["plan"]["next"] == D.NEXT_FOR[cls]


# ----------------------------------------------------------------------------- F12: evaluate stakes and the risk-3 cap

def _evaluator(risk_cls, risk_p, risk_tier, max_tier="act", calibrated=True):
    return {"risk_tier": {"answer": {"score": risk_cls}, "decision": _dec(str(risk_cls), risk_p, risk_tier,
                                                                         calibrated, max_tier)},
            "grounded": {"answer": {}, "decision": _dec("true", 0.99, "act")},
            "complete": {"answer": {"score": 3}, "decision": _dec("3", 0.99, "act")}}


STATE = {"query": "give me a short overview of the Data retention policy", "evidence": ["Records: 5 years."],
         "answer": "Customer records are kept 5 years."}


def test_f12_an_abstained_risk_decision_never_lowers_the_stakes():
    svc = Scripted(_evaluator(0, 0.37, "escalate"))
    r = DocumentsPack().evaluate(svc, {"state": STATE})
    assert r["stakes"] == "high" and r["stakes_from"] == "fail_safe"
    assert svc.seen and any("abstained" in w for w in r["warnings"])
    confident = Scripted(_evaluator(0, 0.9, "act"))
    r2 = DocumentsPack().evaluate(confident, {"state": STATE})
    assert r2["stakes"] == "low" and r2["stakes_from"] == "risk_tier"
    noskill = Scripted(_evaluator(0, 0.9, "escalate", max_tier="escalate"))
    assert DocumentsPack().evaluate(noskill, {"state": STATE})["stakes_from"] == "fail_safe"


def test_f12_risk_three_caps_at_review_even_when_its_decision_abstained():
    r = DocumentsPack().evaluate(Scripted(_evaluator(3, 0.45, "escalate")), {"state": STATE})
    assert r["action"] == "confirm" and r["verdict"] == "review" and r["stakes"] == "high"
    assert any("risk_tier 3" in w for w in r["warnings"])
    noskill = DocumentsPack().evaluate(Scripted(_evaluator(3, 0.5, "escalate", max_tier="escalate")),
                                       {"state": STATE, "stakes": "low"})
    assert noskill["action"] == "confirm"          # level 3 is still the most likely level: review, never ship
    untrained =DocumentsPack().evaluate(Scripted(_evaluator(3, 0.25, "escalate", calibrated=False)),
                                         {"state": STATE, "stakes": "low"})
    assert untrained["action"] == "act"           # an untrained head's prior is not evidence of level 3


# ----------------------------------------------------------------------------- F27: a broken store is a 503

def test_f27_broken_vector_store_is_503_connector_unavailable_without_paths(tmp_path, monkeypatch):
    home = tmp_path / "home"
    monkeypatch.setenv("RLCD_HOME", str(home))
    enc = make_encoder(ENC, home)
    Corpus(home, enc).ingest_records([{"id": "a", "text": "The Zentrix Hub costs 1,250 USD."},
                                      {"id": "b", "text": "The Harbor Team owns the Zentrix Hub."}])
    svc = Service(Project(root=tmp_path, data={"pack": "documents", "pack_config": {"encoder": ENC}}), home=home)
    assert svc.decide(DecideRequest(state={"query": "Zentrix Hub price?"}))["candidates_from"] == "corpus"
    meta = home / "corpus" / "vectors" / "meta.json"
    m = json.loads(meta.read_text())
    meta.write_text(json.dumps({**m, "count": m["count"] + 1}))
    import os

    st = (home / "corpus" / "meta.json").stat()          # the pack re-opens the corpus when its meta changes
    os.utime(home / "corpus" / "meta.json", (st.st_atime, st.st_mtime + 10))
    with pytest.raises(ContractError) as e:
        svc.decide(DecideRequest(state={"query": "Zentrix Hub price?"}))
    assert e.value.status == 503 and e.value.kind == "connector_unavailable"
    body = json.dumps(e.value.body())
    assert str(tmp_path) not in body and str(tmp_path.resolve()) not in body and "/private/" not in body
    assert "ConnectorError" in body and "vectors" in body


def test_f27_messages_lose_absolute_paths():
    assert D._no_paths("x: /srv/data/home/corpus/vectors: bad", "/srv/data/home") == "x: <RLCD_HOME>/corpus/vectors: bad"
    assert D._no_paths("/etc/a/graph.json: unknown graph format", None) == "graph.json: unknown graph format"
    assert D._no_paths("ratio 1/2 stays", None) == "ratio 1/2 stays"


# ----------------------------------------------------------------------------- F11: risk cues on token boundaries

@pytest.mark.parametrize("query", ["When was my order shipped?", "The Velobox 300 stopped working after two days.",
                                   "What happened to the shipment?", "Has the order been dropped?",
                                   "Is a firewall required for remote logins?", "Where do I leave feedback?",
                                   "Can I get a coffee machine?", "What is the costume policy for the party?",
                                   "Энэ үнэн үү?"])
def test_f11_no_risk_cue_inside_ordinary_words(query):
    assert not any(_risk_cues(query).values()), (query, _risk_cues(query))


@pytest.mark.parametrize("query, key", [
    ("What PPE is required?", "q_risk_safety"), ("Is there a fire exit?", "q_risk_safety"),
    ("What is the fee?", "q_risk_money"), ("Prices in $?", "q_risk_money"), ("Refunds for the Velobox?", "q_risk_money"),
    ("How much does it cost?", "q_risk_money"), ("Velobox 300の価格は？", "q_risk_money"),
    ("Velobox 300-ийн үнэ юу вэ?", "q_risk_money"), ("フォークリフトの安全規則", "q_risk_safety"),
    ("Аюулгүй ажиллагааны журам", "q_risk_safety"), ("What first aid is needed after an injury?", "q_risk_medical"),
    ("Ослын үед юу хийх вэ?", "q_risk_medical"),
])
def test_f11_real_cues_still_fire(query, key):
    assert _risk_cues(query)[key] == 1.0


# ----------------------------------------------------------------------------- F33: number formats

@pytest.mark.parametrize("text, expected", [
    ("2万5000円", {"25000"}), ("1万5千円", {"15000"}), ("3億", {"300000000"}), ("1億2000万", {"120000000"}),
    ("1億2千万", {"120000000"}), ("5百円", {"500"}), ("120万円", {"1200000"}), ("1.2万", {"12000"}),
    ("1 250 000 төгрөг", {"1250000"}), ("1 250 000₮", {"1250000"}), ("1 200 000 kg", {"1200000"}),
    ("1 200", {"1", "200"}), ("items 1 200 300", {"1", "200", "300"}), ("2024年3月1日", {"2024", "3", "1"}),
])
def test_f33_composite_and_space_grouped_numbers(text, expected):
    assert numbers(text) == expected


def test_f33_number_support_across_conventions():
    assert number_support("宿泊費の上限は2万5000円です。", "宿泊費の上限は1泊25,000円です。") == (1.0, 0)
    assert number_support("上限は1万5千円です。", "上限は15,000円です。") == (1.0, 0)
    assert number_support("Үнэ нь 1 250 000 төгрөг.", "Үнэ нь 1,250,000 төгрөг.") == (1.0, 0)
    assert number_support("上限は2万6000円です。", "上限は25,000円です。") == (0.0, 1)


# ----------------------------------------------------------------------------- F09 / F10: entity extraction

def test_f10_latin_names_glued_to_cjk_and_mid_sentence_name_words():
    assert entities("チームはRobotics Labです") == ["Robotics Lab"]
    assert entities("担当チームはRobotics Labです。") == ["Robotics Lab"]
    assert entities("Robotics Labチームが担当しています。") == ["Robotics Lab"]
    assert entities("2022-03-29-нд худалдаанд гарсан бөгөөд Last Mile баг хариуцдаг.") == ["Last Mile"]
    assert entities("The Last Mile team owns it.") == ["Last Mile"]
    terms = build_world(7).terms
    assert "Үйл ажиллагааны хэлтэс" in terms
    assert entities("Last Mile баг нь Үйл ажиллагааны хэлтэст харьяалагддаг", terms) == \
        ["Last Mile", "Үйл ажиллагааны хэлтэс"]


def test_f09_common_katakana_and_units_are_not_entities():
    assert entities("Halvarne Cartを担当するチームのリーダーは誰ですか？") == ["Halvarne Cart"]
    assert entities("チームリーダーは田中 美咲です。メンバーは12名です。", ["田中 美咲"]) == ["田中 美咲"]
    assert common_katakana("チームリーダー") and common_katakana("チーム") and not common_katakana("ゼントリクス")
    assert entities("ゼントリクス社はノースウィンド・メタルズから部品を調達。") == ["ゼントリクス", "ノースウィンド・メタルズ"]
    assert entities("Velobox 300（SKU KV-1234）の価格は350 USDです。") == ["Velobox", "KV-1234"]
    assert entities("It costs 1,250 USD and weighs 12 KG.") == []


# ----------------------------------------------------------------------------- F08: hop-2 ranking and hubs

def _team_graph(tmp_path) -> LocalGraphStore:
    g = LocalGraphStore(tmp_path / "g.json", use_networkx=False)
    # ids chosen so that an id tie-break puts the team page last
    g.add_document("products/x", [("products/x#0", "The Zorbel X (SKU KV-1) costs 10 USD; the Harbor Team owns it.")],
                   links=["teams/harbor"])
    for i in range(3):
        g.add_document(f"products/s{i}", [(f"products/s{i}#0", f"The Quenta S{i} (SKU KV-2{i}) costs 20 USD; the "
                                                                f"Harbor Team owns it.")])
    g.add_document("news/a", [("news/a#0", "Visitors tried the Zorbel X and the Quenta S0 at Northgate.")])
    g.add_document("teams/harbor", [("teams/harbor#0", "The Harbor Team works from Northgate. Its lead is Mara Okonkwo.")])
    return g


def test_f08_the_linked_team_page_ranks_first_among_two_hop_rows(tmp_path):
    g = _team_graph(tmp_path)
    rows = g.expand(["Zorbel X"], hops=2)
    hop1 = [r["id"] for r in rows if r["path_length"] == 1]
    hop2 = [r["id"] for r in rows if r["path_length"] == 2]
    assert hop1 == ["products/x#0", "news/a#0"]                  # its own sheet before the news item (focus)
    assert hop2[0] == "teams/harbor#0", hop2                     # link + team name beat the siblings' team name
    assert not g.g.has_node("ent:usd") and not g.g.has_node("ent:sku")


def test_f08_hubs_are_neither_seeds_nor_walked_through(tmp_path):
    g = LocalGraphStore(tmp_path / "h.json", use_networkx=False, hub_min=2)
    for i in range(30):
        g.add_document(f"p{i}", [(f"p{i}#0", f"Acme Corp page {i} mentions Widget{i} Line.")])
    assert g.hub_degree() == 2                                   # max(hub_min 2, ceil(5 % of 30)) = 2
    assert g.expand(["Acme Corp"], hops=1) == []                 # a hub is not a seed
    rows = g.expand(["Widget3 Line"], hops=2)
    assert [r["id"] for r in rows] == ["p3#0"]                   # and nothing is reached through it
    small = LocalGraphStore(tmp_path / "s.json", use_networkx=False)      # default floor: 10 mentions are no hub
    for i in range(8):
        small.add_document(f"p{i}", [(f"p{i}#0", f"Acme Corp page {i} mentions Widget{i} Line.")])
    assert small.hub_degree() == 10 and len(small.expand(["Acme Corp"], hops=1)) == 8


def test_f08_graph_rebuilt_when_the_entity_extractor_changed(tmp_path):
    enc = make_encoder(ENC, tmp_path)
    c = Corpus(tmp_path, enc)
    c.ingest_records([{"id": "a", "title": "A", "text": "チームはRobotics Labです。価格は350 USDです。"}])
    data = json.loads((tmp_path / "corpus" / "graph.json").read_text())
    data["entities"] = 1                                          # pretend an older extractor wrote it
    data["nodes"] = [n for n in data["nodes"] if n["id"] != "ent:robotics lab"] + [{"id": "ent:lab", "kind": "ent"}]
    data["links"] = [x for x in data["links"] if x["target"] != "ent:robotics lab"]
    (tmp_path / "corpus" / "graph.json").write_text(json.dumps(data))
    again = Corpus(tmp_path, enc)
    assert again.graph.g.has_node("ent:robotics lab") and not again.graph.g.has_node("ent:lab")
    again.ingest_records([{"id": "a", "title": "A", "text": "チームはRobotics Labです。価格は350 USDです。"}])
    assert json.loads((tmp_path / "corpus" / "graph.json").read_text())["entities"] == 2   # written at next save


@pytest.fixture(scope="module")
def synth_corpus(tmp_path_factory):
    home = tmp_path_factory.mktemp("synthcorpus")
    w = build_world(7)
    c = Corpus(home, make_encoder(ENC, home), {"graph": {"kind": "local", "terms": w.terms}})
    c.ingest_records([d.record() for d in w.docs.values()])
    return SimpleNamespace(corpus=c, world=w)


LEAD = {"en": "Who leads the team that owns the {P}?", "ja": "{P}を担当するチームのリーダーは誰ですか？",
        "mn": "{P}-г хариуцдаг багийн ахлагч хэн бэ?"}


@pytest.mark.parametrize("lang", ["en", "ja", "mn"])
def test_f08_hop2_recall_on_the_synthetic_corpus(synth_corpus, lang):
    """The team page of 'who leads the team that owns X?': before the fixes graph.expand top-5 found it for 0/18
    (en, mn) and 3/18 (ja) products, candidates() for 16/18, 8/18 and 17/18. Now 15/18 and 18/18 in all three."""
    c, w = synth_corpus.corpus, synth_corpus.world
    team = {d.subject: d.chunk_id for d in w.teams}
    graph_hits = cand_hits = 0
    for p in w.products:
        q = LEAD[lang].format(P=p.subject)
        tid = team[p.meta["team"]]
        graph_hits += tid in [r["id"] for r in c.graph.expand(c.graph.entities(q), hops=2, limit=5)]
        cand_hits += tid in [x["id"] for x in candidates(c, q, 5)]
    assert graph_hits >= 12, graph_hits
    assert cand_hits >= 17, cand_hits


def test_f10_list_query_finds_every_product_of_the_team(synth_corpus):
    c, w = synth_corpus.corpus, synth_corpus.world
    by = {d.subject: d.chunk_id for d in w.products}
    for team, prods in w.team_products.items():
        if not prods:
            continue
        gold = {by[p] for p in prods}
        for q in (f"List all products owned by the {team} team.", f"{team}チームが担当する製品をすべて挙げてください。",
                  f"{team} багийн хариуцдаг бүх бүтээгдэхүүнийг жагсаана уу."):
            got = {x["id"] for x in candidates(c, q, 5)}
            assert gold <= got, (q, gold - got)


# ----------------------------------------------------------------------------- F32: synthetic labels

def test_f32_labels_follow_the_head_definitions():
    w = build_world(7)
    g = Generator(w, 7)
    seen = {"office": False, "missing": False, "fake_partner": False}
    for _ in range(3000):
        q = g.query()
        if q.family == "multihop" and q.facts:
            assert partial_level(q, [0], 2) == 0 and partial_level(q, [1], 2) == 3     # link alone / asked fact
        if q.family == "multihop" and q.template.endswith(":office"):
            assert q.risk == 1
            seen["office"] = True
        if q.family == "prod_extract" and q.gold and not q.answerable:
            assert len(q.gold) == 1 and q.gold[0].startswith("products/")
            seen["missing"] = True
        if q.family == "compare" and not q.answerable:
            assert len(q.gold) == 1 and q.gold[0].startswith("products/")
            seen["fake_partner"] = True
    assert all(seen.values()), seen


# ----------------------------------------------------------------------------- F46: JSON records without ids

def test_f46_json_records_without_ids_get_content_ids_and_shrinking_leaves_no_stale_copy(tmp_path):
    kb = tmp_path / "kb"
    kb.mkdir()
    (kb / "arr.json").write_text(json.dumps([{"text": "first"}, {"text": "second"}, {"text": "third"},
                                             {"id": "r1", "text": "with id"}]), encoding="utf-8")
    enc = make_encoder({"kind": "hashing", "dim": 64}, tmp_path)
    c = Corpus(tmp_path / "home", enc)
    c.ingest_folder(kb)
    before = {c.chunks[c.docs[d]["chunk_ids"][0]]["text"]: d for d in c.docs}
    assert before["with id"] == "r1" and before["second"].startswith("kb/arr.json:")
    (kb / "arr.json").write_text(json.dumps([{"text": "second"}, {"text": "third"}, {"id": "r1", "text": "with id"}]),
                                 encoding="utf-8")
    import os

    st = (kb / "arr.json").stat()
    os.utime(kb / "arr.json", (st.st_atime, st.st_mtime + 10))   # a later mtime (records without dates take it)
    c2 = Corpus(tmp_path / "home", enc)
    out = c2.ingest_folder(kb)
    assert out["docs_added"] == 0 and out["docs_removed"] == 1
    assert out["docs_updated"] + out["docs_unchanged"] == 3
    after = {c2.chunks[c2.docs[d]["chunk_ids"][0]]["text"]: d for d in c2.docs}
    assert sorted(after) == ["second", "third", "with id"]
    assert after["second"] == before["second"] and after["third"] == before["third"] and after["with id"] == "r1"
    assert set(c2.vectors.ids()) == set(c2.chunks)
    (kb / "arr.json").write_text("{broken", encoding="utf-8")           # unreadable: nothing is removed
    out = Corpus(tmp_path / "home", enc).ingest_folder(kb)
    assert out["docs_removed"] == 0 and out["docs_total"] == 3


def test_f46_identical_records_in_one_file_stay_apart(tmp_path):
    kb = tmp_path / "kb"
    kb.mkdir()
    (kb / "dup.jsonl").write_text('{"text": "same"}\n{"text": "same"}\n', encoding="utf-8")
    c = Corpus(tmp_path / "home", make_encoder({"kind": "hashing", "dim": 64}, tmp_path))
    out = c.ingest_folder(kb)
    assert out["docs_added"] == 2 and len({d for d in c.docs}) == 2
