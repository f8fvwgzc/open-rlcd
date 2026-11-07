"""The documents pack's local corpus: chunking, multilingual ingest of txt / md / html / json / jsonl / pdf,
idempotent re-ingest, hybrid retrieval (vector + graph + BM25) and the two-hop entity path. Offline, hashing
encoder only."""
from __future__ import annotations

import importlib.util
import json
import os
import time
from pathlib import Path

import numpy as np
import pytest

from open_rlcd.core.contract import ContractError
from open_rlcd.features.retrieval import RETRIEVAL_FEATURES, chunk_features, retrieval_features
from open_rlcd.features.text import HashingEncoder, make_encoder
from open_rlcd.packs.documents.corpus import Corpus, chunk_text

HAVE_PDF = importlib.util.find_spec("pypdf") is not None   # the 'documents' extra; core-only CI has no pypdf

# ----------------------------------------------------------------------------- chunking


def test_chunk_text_respects_limits_and_paragraphs():
    paras = [f"Paragraph {i} talks about topic {i}. " * 6 for i in range(12)]
    chunks = chunk_text("\n\n".join(paras), max_chars=400, overlap=60)
    assert len(chunks) > 3 and all(len(c) <= 400 for c in chunks)
    assert all(c.strip() for c in chunks)
    # consecutive chunks overlap: the next one starts with a piece of the previous one's end
    for a, b in zip(chunks, chunks[1:], strict=False):
        head = b.split("\n", 1)[0]
        assert head and head in a
    short = chunk_text("One short paragraph.\n\nAnother one.", max_chars=400, overlap=60)
    assert short == ["One short paragraph.\n\nAnother one."]
    assert chunk_text("", 400, 60) == [] and chunk_text("   \n\n  ", 400, 60) == []


def test_chunk_text_long_sentences_and_cjk():
    long_en = " ".join(["word"] * 1000)
    chunks = chunk_text(long_en, max_chars=300, overlap=50)
    assert all(len(c) <= 300 for c in chunks) and len(chunks) >= 15
    ja = "東京本社の従業員数は三百五十人です。" * 80
    jc = chunk_text(ja, max_chars=200, overlap=30)
    assert all(len(c) <= 200 for c in jc) and len(jc) >= 7
    assert "".join(jc).count("東京本社") >= 80                       # nothing lost (overlap may repeat)
    no_punct = "あ" * 1000
    assert all(len(c) <= 200 for c in chunk_text(no_punct, max_chars=200, overlap=30))


# ----------------------------------------------------------------------------- fixtures

def _minimal_pdf(text: str) -> bytes:
    """A one-page PDF with one line of Helvetica text, xref offsets computed (pypdf reads it cleanly)."""
    stream = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode("latin-1")
    objs = [b"<< /Type /Catalog /Pages 2 0 R >>",
            b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R "
            b"/Resources << /Font << /F1 5 0 R >> >> >>",
            b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream",
            b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"]
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for i, body in enumerate(objs, 1):
        offsets.append(len(out))
        out += b"%d 0 obj\n" % i + body + b"\nendobj\n"
    xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objs) + 1)
    out += b"".join(b"%010d 00000 n \n" % o for o in offsets)
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (len(objs) + 1, xref)
    return bytes(out)


@pytest.fixture
def docs(tmp_path) -> Path:
    d = tmp_path / "acme-docs"
    (d / "suppliers").mkdir(parents=True)
    (d / ".hidden").mkdir()
    (d / "zentrix.txt").write_text(
        "Zentrix X1 drones are assembled in Tallinn from parts supplied by Northwind Metals.\n\n"
        "The X1 has a flight range of 12.5 km and costs 1,200 USD.", encoding="utf-8")
    (d / "suppliers" / "northwind.md").write_text(
        "# Northwind Metals\n\nNorthwind Metals' chief executive is Ingrid Solberg. The company is headquartered "
        "in Bergen and employs 480 people.\n\nSee also [the drone note](../zentrix.txt).", encoding="utf-8")
    (d / "bakery.html").write_text(
        "<html><head><title>Bluebird Bakery</title><style>.x{color:red}</style>"
        "<script>var token = 'do-not-index';</script></head><body><h1>Bluebird Bakery</h1>"
        "<p>Bluebird Bakery sells sourdough bread every morning.</p><script>alert('no')</script></body></html>",
        encoding="utf-8")
    (d / "tokyo.json").write_text(json.dumps([
        {"id": "ja-hq", "title": "東京本社", "text": "東京本社の従業員数は350人です。営業時間は9時から18時までです。",
         "updated_at": "2024-05-01T00:00:00Z", "authority": 0.8},
        {"id": "ja-osaka", "title": "大阪支社", "body": "大阪支社は2021年に開設されました。",
         "updated_at": "2024-06-01", "authority": 7},
        {"title": "no text here"},
    ], ensure_ascii=False), encoding="utf-8")
    (d / "mongolia.jsonl").write_text("\n".join([
        json.dumps({"id": "mn-ub", "title": "Улаанбаатар салбар",
                    "content": "Улаанбаатар дахь салбар 2019 онд нээгдсэн. Салбарыг Бат-Эрдэнэ удирддаг.",
                    "authority": 0.9, "links": ["ja-hq"]}, ensure_ascii=False),
        "this line is not json",
    ]), encoding="utf-8")
    (d / "report.pdf").write_bytes(_minimal_pdf("Annual safety audit of the Tallinn assembly plant passed"))
    (d / "notes.csv").write_text("a,b\n1,2\n", encoding="utf-8")
    (d / ".secret.txt").write_text("hidden file must not be ingested", encoding="utf-8")
    (d / ".hidden" / "x.txt").write_text("hidden dir must not be ingested", encoding="utf-8")
    return d


@pytest.fixture
def enc() -> HashingEncoder:
    return HashingEncoder(dim=256)


# ----------------------------------------------------------------------------- ingest

def test_ingest_folder_reads_every_format(tmp_path, docs, enc):
    corpus = Corpus(tmp_path / "home", enc)
    assert not corpus.exists() and Corpus.open(tmp_path / "home", enc) is None
    r = corpus.ingest_folder(docs)
    assert r["files"] == 6                                          # csv, hidden file and hidden dir skipped
    n_docs = 7 if HAVE_PDF else 6                                   # without pypdf the PDF is skipped with a reason
    assert r["docs_added"] == n_docs and r["docs_unchanged"] == 0 and r["docs_total"] == n_docs
    assert r["chunks_total"] == r["chunks_added"] >= n_docs
    reasons = " ".join(s["reason"] for s in r["skipped"])
    assert "records without text" in reasons
    assert corpus.exists()
    assert set(corpus.docs) == {"acme-docs/zentrix.txt", "acme-docs/suppliers/northwind.md", "acme-docs/bakery.html",
                                "ja-hq", "ja-osaka", "mn-ub"} | ({"acme-docs/report.pdf"} if HAVE_PDF else set())
    html = corpus.chunk("acme-docs/bakery.html#0")
    assert html["title"] == "Bluebird Bakery" and "sourdough" in html["text"]
    assert "do-not-index" not in html["text"] and "alert" not in html["text"] and "color" not in html["text"]
    hq = corpus.doc("ja-hq")
    assert hq["authority"] == 0.8 and hq["updated_at"] == 1714521600 and hq["title"] == "東京本社"
    assert corpus.doc("ja-osaka")["authority"] == 1.0                # clipped to [0, 1]
    assert corpus.doc("mn-ub")["links"] == ["ja-hq"]
    assert corpus.doc("acme-docs/suppliers/northwind.md")["links"] == ["acme-docs/zentrix.txt"]
    if HAVE_PDF:
        assert "safety audit" in corpus.chunk("acme-docs/report.pdf#0")["text"]
    else:
        assert "pypdf" in reasons
    assert corpus.chunk("missing#0") is None
    for name in ("meta.json", "docs.jsonl", "chunks.jsonl", "bm25.json", "graph.json"):
        assert (tmp_path / "home" / "corpus" / name).is_file()
    assert sorted(p.name for p in (tmp_path / "home" / "corpus" / "vectors").iterdir()) == \
        ["items.jsonl", "meta.json", "vectors.npy"]
    stats = corpus.stats()
    assert stats["docs"] == n_docs and stats["vectors"] == stats["chunks"] and stats["graph"]["chunk"] == stats["chunks"]


def test_reingest_is_idempotent_and_updates_changed_docs(tmp_path, docs, enc):
    corpus = Corpus(tmp_path / "home", enc)
    first = corpus.ingest_folder(docs)
    meta_before = (tmp_path / "home" / "corpus" / "chunks.jsonl").read_bytes()
    again = corpus.ingest_folder(docs)
    assert again["docs_unchanged"] == first["docs_added"] and again["docs_added"] == again["docs_updated"] == 0
    assert again["chunks_added"] == 0
    assert (tmp_path / "home" / "corpus" / "chunks.jsonl").read_bytes() == meta_before

    f = docs / "zentrix.txt"
    f.write_text("Zentrix X2 drones replaced the X1 line in 2025.", encoding="utf-8")
    os.utime(f, (time.time() + 5, time.time() + 5))
    reopened = Corpus.open(tmp_path / "home", enc)
    third = reopened.ingest_folder(docs)
    assert third["docs_updated"] == 1 and third["docs_unchanged"] == first["docs_added"] - 1
    assert "X2" in reopened.chunk("acme-docs/zentrix.txt#0")["text"]
    assert reopened.chunk("acme-docs/zentrix.txt#1") is None         # the old second chunk is gone everywhere
    assert "acme-docs/zentrix.txt#1" not in reopened.vectors
    assert reopened.vectors.count() == len(reopened.chunks)


# ----------------------------------------------------------------------------- retrieve

@pytest.fixture
def corpus(tmp_path, docs, enc) -> Corpus:
    c = Corpus(tmp_path / "home", enc)
    c.ingest_folder(docs, max_chars=300, overlap=40)
    return Corpus.open(tmp_path / "home", enc)


@pytest.mark.parametrize("query, doc", [
    ("What is the flight range of the Zentrix X1?", "acme-docs/zentrix.txt"),
    ("sourdough bread bakery", "acme-docs/bakery.html"),
    ("東京本社の従業員数は何人ですか", "ja-hq"),
    ("Улаанбаатар салбарыг хэн удирддаг вэ?", "mn-ub"),
    ("safety audit of the assembly plant", "acme-docs/report.pdf"),
])
def test_retrieve_finds_the_right_chunk(corpus, query, doc):
    if doc.endswith(".pdf") and not HAVE_PDF:
        pytest.skip("pypdf (documents extra) not installed")
    cands = corpus.retrieve(query, k=3)
    assert cands and cands[0]["doc_id"] == doc
    assert [c["rank"] for c in cands] == list(range(1, len(cands) + 1))
    top = cands[0]
    assert top["source"] in {"vector", "graph", "lexical", "hybrid"} and top["text"]
    assert -1.0 <= top["score"] <= 1.0 and np.isfinite(top["lexical_score"])
    assert len(cands) <= 9


def test_retrieve_two_hop_entity_question_uses_the_graph(corpus):
    q = "Who is the chief executive of the company that supplies Zentrix?"
    cands = corpus.retrieve(q, k=3, graph_hops=2)
    north = next(c for c in cands if "chief executive" in c["text"])
    assert "graph" in north["sources"] and north["graph"]["path_length"] == 2
    assert north["graph"]["via"] == "Northwind Metals" and north["graph"]["degree"] >= 2
    zen = next(c for c in cands if c["id"] == "acme-docs/zentrix.txt#0")
    assert "graph" in zen["sources"] and zen["graph"]["path_length"] == 1
    no_graph = corpus.retrieve(q, k=3, graph_hops=0)
    assert all("graph" not in c["sources"] for c in no_graph)
    # the candidates are exactly what the retrieval / chunk features consume
    f = retrieval_features(q, cands, k=3)
    assert list(f) == RETRIEVAL_FEATURES and f["r_n_candidates"] == len(cands) and f["r_path_len_min"] == 1
    cf = chunk_features(q, north, cands)
    assert cf["c_path_len"] == 2 and cf["c_rank"] == north["rank"]


def test_retrieve_edge_cases(tmp_path, corpus, enc):
    assert corpus.retrieve("", k=5) == []
    assert Corpus(tmp_path / "empty-home", enc).retrieve("anything") == []
    assert len(corpus.retrieve("Zentrix", k=1)) <= 3


def test_ingest_records_remove_and_numpy_backend(tmp_path, enc):
    c = Corpus(tmp_path / "home", enc, config={"vector": {"kind": "local", "backend": "numpy"},
                                               "graph": {"kind": "local", "terms": ["本社"]}})
    r = c.ingest_records([{"id": "r1", "title": "Policy", "text": "Refunds are issued within 14 days.",
                           "updated_at": 1700000000, "authority": 0.4},
                          {"id": "r2", "text": "東京本社は品川にあります。"}, {"id": "r3"}, "not a dict"])
    assert r["docs_added"] == 2 and r["docs_skipped"] == 2 and c.vectors.backend == "numpy"
    assert c.retrieve("refund within 14 days", k=1)[0]["id"] == "r1#0"
    graph_hit = [x for x in c.retrieve("本社はどこ", k=2) if "graph" in x["sources"]]
    assert graph_hit and graph_hit[0]["doc_id"] == "r2"                # kanji term from the configured list
    assert c.remove(["r1", "missing"]) == 1 and c.chunk("r1#0") is None and c.vectors.count() == 1
    assert all(x["doc_id"] != "r1" for x in c.retrieve("refund", k=5))


def test_encoder_mismatch_and_reembed(tmp_path, docs, enc):
    Corpus(tmp_path / "home", enc).ingest_folder(docs)
    other = HashingEncoder(dim=128)
    with pytest.raises(ContractError) as e:
        Corpus.open(tmp_path / "home", other)
    assert e.value.kind == "encoder_mismatch"
    c = Corpus(tmp_path / "home", other, config={"on_encoder_change": "reembed"})
    assert c.vectors.dim == 128 and c.vectors.count() == len(c.chunks)
    assert c.retrieve("sourdough bread", k=1)[0]["doc_id"] == "acme-docs/bakery.html"
    assert Corpus.open(tmp_path / "home", other) is not None          # meta now records the new encoder


def test_cached_encoder_and_unsupported_connectors(tmp_path, docs):
    enc = make_encoder({"kind": "hashing", "dim": 64}, tmp_path / "home")
    c = Corpus(tmp_path / "home", enc)
    c.ingest_folder(docs)
    assert c.retrieve("Bluebird Bakery", k=1)[0]["doc_id"] == "acme-docs/bakery.html"
    for cfg in ({"vector": {"kind": "qdrant"}}, {"graph": {"kind": "neo4j"}}):
        with pytest.raises(ContractError, match="not available yet") as e:
            Corpus(tmp_path / "home", enc, config=cfg)
        assert e.value.kind == "unsupported_connector"
    with pytest.raises(ContractError):
        c.ingest_folder(tmp_path / "does-not-exist")


def test_ingest_single_file(tmp_path, docs, enc):
    c = Corpus(tmp_path / "home", enc)
    r = c.ingest_folder(docs / "zentrix.txt")
    assert r["files"] == 1 and list(c.docs) == ["zentrix.txt"]
