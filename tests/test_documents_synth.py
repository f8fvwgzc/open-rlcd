"""The synthetic RAG generator: deterministic, multilingual, labels consistent with the documents. Offline."""
from __future__ import annotations

import random

from open_rlcd.features.lexical import numbers
from open_rlcd.features.text import make_encoder
from open_rlcd.labeling.store import LabelStore
from open_rlcd.packs.documents.corpus import Corpus
from open_rlcd.packs.documents.synth import Generator, complete_level, compose, corrupt_value, write_synth
from open_rlcd.packs.documents.world import PRODUCT_NAMES, build_world

ENC = {"kind": "hashing", "dim": 64}


def snapshot(home):
    store = LabelStore(home)
    items = sorted((i.id, i.view, i.group, i.t, repr(i.state)) for i in store.items())
    labels = sorted((k, repr(v["label"]), v["source"]) for k, v in store.human_labels().items())
    return items, labels


def test_generator_is_deterministic(tmp_path):
    a = write_synth(tmp_path / "a", ENC, seed=11, n_queries=60)
    b = write_synth(tmp_path / "b", ENC, seed=11, n_queries=60)
    assert {k: v for k, v in a.items() if k != "home"} == {k: v for k, v in b.items() if k != "home"}
    assert snapshot(tmp_path / "a") == snapshot(tmp_path / "b")
    c = write_synth(tmp_path / "c", ENC, seed=12, n_queries=60)
    assert c["seed"] == 12 and snapshot(tmp_path / "c") != snapshot(tmp_path / "a")
    again = write_synth(tmp_path / "a", ENC, seed=11, n_queries=60)       # re-running adds no new items
    assert again["items_added"] == 0 and snapshot(tmp_path / "a")[0] == snapshot(tmp_path / "b")[0]


def test_items_are_trusted_grouped_and_timed(tmp_path):
    info = write_synth(tmp_path / "h", ENC, seed=7, n_queries=80)
    store = LabelStore(tmp_path / "h")
    items = store.items()
    assert sum(info["items"].values()) == len(items) and set(info["items"]) == {
        "query", "query_chunk", "query_evidence", "query_evidence_answer"}
    assert all(i.group and i.t and i.display.startswith("Request:") for i in items)
    assert {v["source"] for v in store.human_labels().values()} == {"synthetic"}
    # every item of one query shares the query's template group, so paraphrases never straddle the holdout
    by_query: dict[str, set] = {}
    for i in items:
        by_query.setdefault(i.id.split(":")[0], set()).add(i.group)
    assert all(len(g) == 1 for g in by_query.values())
    assert set(info["languages"]) == {"en", "ja", "mn"}
    corpus = Corpus.open(tmp_path / "h", make_encoder(ENC, tmp_path / "h"))
    assert corpus is not None and corpus.stats()["docs"] == info["corpus"]["docs"]
    full = minimal = 0
    for it in items:                 # candidates are real retrieval hits from that corpus
        for c in (it.state.get("candidates") or [])[:3]:
            assert corpus.chunk(c["id"]) is not None and c.get("text")
            if "rank" in c:          # full retrieval metadata
                assert "score" in c
                full += 1
            else:                    # metadata dropout: what a caller with another retriever sends
                assert set(c) <= {"id", "text", "score"}
                minimal += 1
    assert full > minimal > 0


def test_world_links_and_unanswerable_entities():
    w = build_world(7)
    assert build_world(7).docs.keys() == w.docs.keys()
    langs = {d.lang for d in w.docs.values()}
    assert langs == {"en", "ja", "mn"}
    for p in w.products:             # product sheet -> owning team (named and linked); team page names no product
        team = p.meta["team"]
        assert team in p.text and p.links == [f"teams/{team.lower().replace(' ', '-')}"]
        tdoc = next(t for t in w.teams if t.subject == team)
        assert not any(n in tdoc.text for n in PRODUCT_NAMES) and w.leads[team] in tdoc.text
    for d in [*w.products, *w.fact_docs, *w.teams]:          # every stated fact is in its document's text
        for f in d.facts.values():
            if f.kind in ("number", "months"):
                assert numbers(f.value[d.lang]) <= numbers(d.text), (d.id, f.key)
            else:
                assert f.value[d.lang] in d.text, (d.id, f.key)
    g = Generator(w, 3)
    fakes = [g._prod_attr() for _ in range(300)]
    for q in fakes:
        if not q.answerable:
            assert q.gold == [] and q.facts == []


def test_answers_corruption_and_levels():
    w = build_world(7)
    rng = random.Random(0)
    p = w.products[0]
    price = p.facts["price"]
    wrong = corrupt_value(price, "en", rng, w, p.text)
    assert wrong != price.value["en"] and not numbers(wrong) <= numbers(p.text)
    lead = next(t for t in w.teams if t.subject == p.meta["team"]).facts["lead"]
    other = corrupt_value(lead, "en", rng, w, p.text)
    assert other in w.names["person"] and other != lead.value["en"]
    launch = p.facts["launch"]
    assert corrupt_value(launch, "en", rng, w, "") != launch.value["en"]
    g = Generator(w, 5)
    q = next(q for q in (g._prod_extract() for _ in range(50)) if q.answerable)
    full, part = compose(q, "en"), compose(q, "en", keep=[0])
    assert all(f.value["en"] in full for _, f in q.facts) and len(part) < len(full)
    assert [complete_level(k, 3) for k in range(4)] == [0, 1, 2, 3] and complete_level(1, 2) == 1
    ja = compose(q, "ja", style=1)
    assert "について" in ja and q.facts[0][0] in ja
