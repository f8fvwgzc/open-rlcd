"""Remote connectors (M7): Qdrant, pgvector and Neo4j, the documents corpus on top of them, `rlcd ingest --qdrant /
--collection / --neo4j`, and the lazy imports that keep a core-only install working.

* Qdrant runs for real: qdrant-client's in-process mode (`url: ":memory:"`), full upsert / search / fetch / delete
  round trips, attach mode against collections built here with their own payload schema.
* pgvector and Neo4j run against fakes that record every statement with its parameters and transaction mode (no
  server is started): they prove parameterised SQL / Cypher, read-only attach mode, dimension checks, the 503
  mapping and the host checks, not that the SQL / Cypher runs on a real server. Tests that need a live server are
  marked `network` and skipped unless RLCD_TEST_NETWORK=1 (plus the server's address in the environment).

Nothing here opens a network connection, except to a loopback port that was just closed (connection refused).
"""
from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import uuid
from datetime import UTC, datetime
from decimal import Decimal
from importlib.util import find_spec
from pathlib import Path

import numpy as np
import pytest
import yaml

from open_rlcd.cli import main
from open_rlcd.connectors import base as B
from open_rlcd.connectors.factory import check_store, make_graph_store, make_vector_store
from open_rlcd.core.config import Project
from open_rlcd.core.contract import ContractError, DecideRequest
from open_rlcd.core.service import Service
from open_rlcd.features.text import HashingEncoder
from open_rlcd.packs.documents import DocumentsPack
from open_rlcd.packs.documents.corpus import Corpus

HAVE_QDRANT = find_spec("qdrant_client") is not None
HAVE_PSYCOPG = find_spec("psycopg") is not None
HAVE_NEO4J = find_spec("neo4j") is not None
needs_qdrant = pytest.mark.skipif(not HAVE_QDRANT, reason="qdrant-client (extra 'qdrant') not installed")
needs_psycopg = pytest.mark.skipif(not HAVE_PSYCOPG, reason="psycopg (extra 'pgvector') not installed")
needs_neo4j = pytest.mark.skipif(not HAVE_NEO4J, reason="neo4j driver (extra 'neo4j') not installed")

DIM = 64
ENC_CFG = {"kind": "hashing", "dim": DIM}
RECS = [
    {"id": "zentrix", "title": "Zentrix X1", "text": "Zentrix X1 drones are assembled in Tallinn from parts supplied "
     "by Northwind Metals. The X1 has a flight range of 12.5 km.", "updated_at": 1714521600, "authority": 0.8},
    {"id": "northwind", "title": "Northwind Metals", "text": "Northwind Metals' chief executive is Ingrid Solberg. "
     "The company is headquartered in Bergen.", "links": ["zentrix"]},
    {"id": "bakery", "title": "Bluebird Bakery", "text": "Bluebird Bakery sells sourdough bread every morning."},
    {"id": "tokyo", "title": "東京本社", "text": "東京本社の従業員数は350人です。営業時間は9時から18時までです。"},
]
QUERIES = ["Who is the chief executive of the company that supplies Zentrix?", "sourdough bread",
           "東京本社の従業員数は何人ですか", "flight range of the X1"]


@pytest.fixture
def enc() -> HashingEncoder:
    return HashingEncoder(dim=DIM)


def uniq(prefix: str = "c") -> str:
    return f"{prefix}_{uuid.uuid4().hex[:10]}"


def closed_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def summary(cands: list[dict]) -> list[tuple]:
    return [(c["id"], c["source"], round(c["score"], 4), (c.get("graph") or {}).get("path_length")) for c in cands]


# ============================================================================ lazy imports (core-only installs)

BLOCKER = """
import sys
class _Block:
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in {"qdrant_client", "psycopg", "psycopg_binary", "pgvector", "neo4j"}:
            raise ImportError(f"blocked {name}")
        return None
sys.meta_path.insert(0, _Block())
import os
os.environ["RLCD_PG_DSN"] = "postgresql://localhost/x"
import open_rlcd, open_rlcd.cli, open_rlcd.serve.api
import open_rlcd.connectors.qdrant, open_rlcd.connectors.pgvector, open_rlcd.connectors.neo4j
from open_rlcd.connectors.factory import make_vector_store, make_graph_store
from open_rlcd.packs.documents import corpus, decide, search
from open_rlcd.core.contract import ContractError
out = []
for build in (lambda: make_vector_store({"kind": "qdrant", "url": ":memory:", "collection": "x"}, 8).connect(),
              lambda: make_vector_store({"kind": "pgvector", "table": "t"}, 8),
              lambda: make_graph_store({"kind": "neo4j", "uri": "bolt://localhost:7687"}).connect()):
    try:
        build()
        out.append("no error")
    except ContractError as e:
        out.append(f"{e.status} {e.kind} {e.details.get('extra')} {e.message}")
print("\\n".join(out))
"""


def test_core_imports_and_names_the_missing_extra_without_connector_libraries():
    """With qdrant-client, psycopg, pgvector and neo4j unimportable, everything imports, and using a remote store
    fails with a 503 that names the extra to install."""
    r = subprocess.run([sys.executable, "-c", BLOCKER], capture_output=True, text=True, timeout=120,
                       cwd=str(Path(__file__).resolve().parent.parent))
    assert r.returncode == 0, r.stderr
    lines = r.stdout.strip().splitlines()
    assert len(lines) == 3
    for line, extra in zip(lines, ("qdrant", "pgvector", "neo4j"), strict=True):
        assert line.startswith(f"503 connector_unavailable {extra} ") and f"'{extra}' extra" in line
        assert f"uv sync --extra {extra}" in line


# ============================================================================ configuration and security

def test_credentials_never_come_from_rlcd_yaml(monkeypatch):
    cases = [({"kind": "qdrant", "url": "http://localhost:6333", "collection": "c", "api_key": "s3cret-k"},
              "vector", "api_key_env"),
             ({"kind": "pgvector", "table": "t", "dsn": "postgresql://u:s3cret-k@localhost/db"}, "vector", "dsn_env"),
             ({"kind": "neo4j", "uri": "bolt://localhost:7687", "password": "s3cret-k"}, "graph", "password_env")]
    for cfg, what, alt in cases:
        with pytest.raises(ContractError) as e:
            make_vector_store(cfg, DIM) if what == "vector" else make_graph_store(cfg)
        assert e.value.kind == "connector_config" and alt in e.value.message and "s3cret" not in e.value.message
    for cfg, what in (({"kind": "qdrant", "url": "http://admin:s3cret-k@localhost:6333", "collection": "c"}, "vector"),
                      ({"kind": "neo4j", "uri": "bolt://neo4j:s3cret-k@localhost:7687"}, "graph")):
        with pytest.raises(ContractError, match="carries credentials") as e:
            make_vector_store(cfg, DIM) if what == "vector" else make_graph_store(cfg)
        assert "s3cret" not in e.value.message
    with pytest.raises(ContractError, match="name of an environment variable"):
        make_vector_store({"kind": "qdrant", "url": "http://localhost:6333", "collection": "c",
                           "api_key_env": "sk-live-123"}, DIM)
    with pytest.raises(ContractError, match="unknown key"):           # a typo never falls back to a default
        make_vector_store({"kind": "qdrant", "url": "http://localhost:6333", "colection": "c", "collection": "d"},
                          DIM)


@pytest.mark.parametrize("cfg, what", [
    ({"kind": "qdrant", "url": "http://qdrant.example.com:6333", "collection": "c"}, "vector"),
    ({"kind": "qdrant", "url": "https://10.0.0.7:6333", "collection": "c", "mode": "attach"}, "vector"),
    ({"kind": "neo4j", "uri": "neo4j+s://graph.example.org:7687"}, "graph"),
])
def test_remote_hosts_are_refused_unless_allowed(cfg, what, monkeypatch):
    monkeypatch.delenv("RLCD_ALLOW_REMOTE", raising=False)
    build = (lambda **kw: make_vector_store(cfg, DIM, **kw)) if what == "vector" else \
        (lambda **kw: make_graph_store(cfg, **kw))
    with pytest.raises(ContractError, match="refusing non-loopback host") as e:
        build()
    assert e.value.kind == "connector_config" and "RLCD_ALLOW_REMOTE=1" in e.value.message
    assert build(allow_remote=True) is not None            # building opens no connection
    monkeypatch.setenv("RLCD_ALLOW_REMOTE", "1")
    assert build() is not None


def test_corpus_takes_allow_remote_from_the_project(tmp_path, monkeypatch):
    monkeypatch.delenv("RLCD_ALLOW_REMOTE", raising=False)
    vec = {"kind": "qdrant", "url": "http://qdrant.example.com:6333", "collection": "c", "mode": "attach"}
    project = Project(root=tmp_path, data={"pack": "documents", "pack_config": {"encoder": ENC_CFG, "vector": vec}})
    project.home_override = tmp_path / "home"
    pack = DocumentsPack()
    with pytest.raises(ContractError, match="refusing non-loopback host"):
        pack.corpus(project, pack.provider(project))
    allowed = Project(root=tmp_path, data={**project.data, "security": {"allow_remote": True}})
    allowed.home_override = tmp_path / "home"
    c = Corpus(tmp_path / "home", HashingEncoder(dim=DIM), {"vector": vec, "allow_remote": True})
    assert c.attached and c.exists() and c.vectors.target == "collection 'c'"
    from open_rlcd.packs.documents.settings import project_corpus_settings
    assert project_corpus_settings(allowed)["allow_remote"] is True
    assert project_corpus_settings(project)["allow_remote"] is False


def test_check_store_refuses_offline_what_building_the_store_refuses(monkeypatch):
    """`factory.check_store` is what `rlcd config check` runs for a remote store: the store's own checks without its
    client library, an environment variable or a connection. Qdrant and Neo4j build the store to check it (building
    opens no connection), so a refusal is word for word the one the pack gets. The one difference: any host passes,
    because whether a remote host is allowed depends on the environment the store is built in."""
    from open_rlcd.connectors import neo4j as N4
    from open_rlcd.connectors import qdrant as QD

    def offline(*a, **kw):
        raise AssertionError("checking a configuration needs no client library and no environment variable")

    for mod in (QD, N4):
        monkeypatch.setattr(mod, "require_extra", offline)
        monkeypatch.setattr(mod, "env_secret", offline)
    monkeypatch.delenv("RLCD_ALLOW_REMOTE", raising=False)
    cypher = "MATCH (e:Entity)<-[:MENTIONS]-(c:Chunk) WHERE e.key IN $keys RETURN c.id AS id, 1 AS path_length"
    cases = [({"kind": "qdrant", "url": "http://localhost:6333"}, "vector"),
             ({"kind": "qdrant", "url": "http://localhost:6333", "path": "./q", "collection": "c"}, "vector"),
             ({"kind": "qdrant", "url": "http://localhost:6333", "collection": "c", "table": "t"}, "vector"),
             ({"kind": "qdrant", "url": "http://admin:s3cret-k@localhost:6333", "collection": "c"}, "vector"),
             ({"kind": "qdrant", "url": ":memory:", "collection": "c", "mode": "attach", "text_field": None}, "vector"),
             ({"kind": "neo4j", "uri": "bolt://localhost:7687", "mode": "attach"}, "graph"),
             ({"kind": "neo4j", "uri": "bolt://localhost:7687", "expand_cypher": cypher + " DETACH DELETE c"}, "graph"),
             ({"kind": "neo4j", "uri": "bolt://localhost:7687", "degree_cypher": "RETURN 1 AS degree"}, "graph"),
             ({"kind": "neo4j", "uri": "bolt://localhost:7687", "password_env": "s3cret-k pw"}, "graph")]
    for cfg, what in cases:
        with pytest.raises(ContractError) as built:
            make_vector_store(cfg, DIM) if what == "vector" else make_graph_store(cfg)
        with pytest.raises(ContractError) as checked:
            check_store(cfg, what)
        assert checked.value.message == built.value.message and checked.value.kind == "connector_config", cfg
        assert "s3cret" not in checked.value.message
    check_store({"kind": "neo4j", "mode": "attach", "uri": "bolt://localhost:7687", "expand_cypher": cypher}, "graph")
    far = {"kind": "qdrant", "url": "http://qdrant.example.com:6333", "collection": "c"}
    with pytest.raises(ContractError, match="refusing non-loopback host"):
        make_vector_store(far, DIM)
    check_store(far, "vector")
    check_store({"kind": "local", "backend": "numpy"}, "vector")             # a local store: nothing to refuse here


# ============================================================================ Qdrant (real, in-process)

def _brute(X: np.ndarray, q: np.ndarray, k: int) -> list[int]:
    s = (X / np.linalg.norm(X, axis=1, keepdims=True)) @ (q / np.linalg.norm(q))
    return list(np.argsort(-s, kind="stable")[:k])


@needs_qdrant
def test_qdrant_index_round_trip_matches_brute_force():
    from open_rlcd.connectors.qdrant import QdrantVectorStore, point_id

    rng = np.random.default_rng(0)
    X = rng.normal(size=(150, 24)).astype(np.float32)
    ids = [f"doc{i % 9}/file.md#{i}" for i in range(150)]               # string ids: stored as uuid5 point ids
    store = QdrantVectorStore(uniq(), 24, url=":memory:")
    store.upsert(ids, X, [{"doc_id": f"doc{i % 9}", "text": f"chunk {i}", "authority": 0.5} for i in range(150)])
    assert store.count() == 150 and isinstance(store, B.VectorStore)
    for q in rng.normal(size=(8, 24)).astype(np.float32):
        hits = store.search(q, k=6)
        want = _brute(X, q, 6)
        assert [h.id for h in hits] == [ids[i] for i in want]
        cos = [float(X[i] @ q / np.linalg.norm(X[i]) / np.linalg.norm(q)) for i in want]
        assert np.allclose([h.score for h in hits], cos, atol=1e-4)
        assert all(h.payload["text"].startswith("chunk ") and h.payload["doc_id"].startswith("doc") for h in hits)
    hits = store.search(X[3], k=50, filter={"doc_id": "doc3"})
    assert hits and {h.payload["doc_id"] for h in hits} == {"doc3"} and hits[0].id == ids[3]
    got = store.fetch([ids[0], ids[7], "missing#0"])
    assert set(got) == {ids[0], ids[7]} and got[ids[7]][1]["text"] == "chunk 7"
    assert np.allclose(got[ids[0]][0] / np.linalg.norm(got[ids[0]][0]), X[0] / np.linalg.norm(X[0]), atol=1e-5)
    V = store.vectors([ids[1], "missing#0"])
    assert np.isclose(np.linalg.norm(V[0]), 1.0) and np.isnan(V[1]).all()
    assert store.delete([ids[0], ids[1]]) == 2 and store.count() == 148 and store.search(X[0], k=1)[0].id != ids[0]
    assert point_id(ids[0]) == point_id(ids[0]) != point_id(ids[1])
    with pytest.raises(ValueError):
        store.upsert(["a", "a"], np.ones((2, 24)), [{}, {}])
    with pytest.raises(ValueError):
        store.search(np.ones(5))
    assert store.search(X[0], k=0) == []


@needs_qdrant
def test_corpus_in_qdrant_index_mode_matches_the_local_corpus(tmp_path, enc):
    name = uniq()
    cfg = {"vector": {"kind": "qdrant", "url": ":memory:", "collection": name}}
    remote = Corpus(tmp_path / "remote", enc, cfg)
    local = Corpus(tmp_path / "local", enc)
    r = remote.ingest_records(RECS)
    local.ingest_records(RECS)
    assert r["docs_added"] == 4 and remote.vectors.count() == len(remote.chunks)
    assert not (tmp_path / "remote" / "corpus" / "vectors").exists()   # the vectors live in Qdrant only
    meta = json.loads((tmp_path / "remote" / "corpus" / "meta.json").read_text())
    assert meta["stores"]["vector"] == {"kind": "qdrant", "collection": name} and meta["vector_backend"] == "qdrant"
    for q in QUERIES:
        a, b = remote.retrieve(q, k=3), local.retrieve(q, k=3)
        assert [c["id"] for c in a] == [c["id"] for c in b]
        assert np.allclose([c["score"] for c in a], [c["score"] for c in b], atol=1e-4)
        assert [c["sources"] for c in a] == [c["sources"] for c in b]
    two_hop = next(c for c in remote.retrieve(QUERIES[0], k=3) if c["id"] == "northwind#0")
    assert two_hop["graph"]["path_length"] == 2 and "graph" in two_hop["sources"]
    # the collection describes itself: the payload carries the candidate fields
    hit = remote.vectors.search(enc.encode(["sourdough bread"], kind="query")[0], k=1)[0]
    assert hit.id == "bakery#0" and hit.payload["text"].startswith("Bluebird") and hit.payload["doc_id"] == "bakery"
    # re-opened by another Corpus (the in-process store is shared), a changed document replaces its points
    again = Corpus.open(tmp_path / "remote", enc, cfg)
    assert [c["id"] for c in again.retrieve("sourdough bread", k=2)][0] == "bakery#0"
    again.ingest_records([{**RECS[2], "text": "Bluebird Bakery now sells rye bread.\n\n" + "More text. " * 200}])
    assert again.vectors.count() == len(again.chunks) > len(local.chunks)
    assert again.remove(["bakery"]) == 1 and again.vectors.count() == len(again.chunks) == len(local.chunks) - 1
    assert again.chunk("bakery#0") is None and again.vectors.fetch(["bakery#0"]) == {}


@needs_qdrant
def test_switching_an_ingested_corpus_to_qdrant_needs_one_ingest(tmp_path, enc):
    Corpus(tmp_path / "h", enc).ingest_records(RECS)
    cfg = {"vector": {"kind": "qdrant", "url": ":memory:", "collection": uniq()}}
    with pytest.raises(ContractError, match="indexed into other stores") as e:
        Corpus(tmp_path / "h", enc, cfg)              # serving would search an empty collection: refused
    assert e.value.kind == "connector_config" and "rlcd ingest" in e.value.message
    moved = Corpus(tmp_path / "h", enc, cfg, reindex=True)
    assert moved.reindexed == ["vector"] and moved.vectors.count() == len(moved.chunks)
    opened = Corpus.open(tmp_path / "h", enc, cfg)
    assert opened.reindexed == [] and opened.retrieve("sourdough bread", k=1)[0]["id"] == "bakery#0"
    with pytest.raises(ContractError, match="the local vector index"):     # and back: the local index may be stale
        Corpus.open(tmp_path / "h", enc)
    assert Corpus(tmp_path / "h", enc, reindex=True).reindexed == ["vector"]


def _make_attached(enc, *, size: int = DIM, distance: str = "Cosine", named: str | None = None, scale: bool = False,
                   id_key: str | None = None) -> tuple[str, list[dict]]:
    """A collection in the shared in-process Qdrant with somebody else's payload schema and integer point ids."""
    from qdrant_client import models

    from open_rlcd.connectors.qdrant import QdrantVectorStore

    client = QdrantVectorStore(uniq(), DIM, url=":memory:")._make_client()
    name = uniq("kb")
    vp = models.VectorParams(size=size, distance=models.Distance(distance))
    client.create_collection(name, vectors_config={named: vp} if named else vp)
    docs = [{"body": r["text"], "source": r["id"], "heading": r["title"], "modified": "2024-05-01T00:00:00Z",
             "trust": 7 if r["id"] == "bakery" else 0.6} for r in RECS]
    if size == DIM:
        vecs = enc.encode([f"{d['heading']}\n{d['body']}" for d in docs], kind="passage")
        if scale:
            vecs = vecs * np.arange(1, len(docs) + 1, dtype=np.float32)[:, None] * 3.0
    else:
        vecs = np.random.default_rng(1).normal(size=(len(docs), size)).astype(np.float32)
    pts = []
    for i, (d, v) in enumerate(zip(docs, vecs, strict=True), start=1):
        payload = {**d, **({id_key: f"kb-{i}"} if id_key else {})}
        pts.append(models.PointStruct(id=i, vector={named: v.tolist()} if named else v.tolist(), payload=payload))
    client.upsert(name, points=pts, wait=True)
    return name, docs


ATTACH_FIELDS = {"text_field": "body", "doc_id_field": "source", "title_field": "heading",
                 "updated_at_field": "modified", "authority_field": "trust"}
WRITES = ("upsert", "delete", "create_collection", "recreate_collection", "delete_collection", "set_payload",
          "overwrite_payload", "delete_payload", "clear_payload", "update_vectors", "delete_vectors",
          "batch_update_points", "upload_points", "upload_collection", "update_collection", "create_payload_index")


@pytest.fixture
def no_qdrant_writes(monkeypatch):
    """Any write to Qdrant fails the test (installed after the test's own collections are built)."""
    from qdrant_client import QdrantClient

    def install():
        for m in WRITES:
            monkeypatch.setattr(QdrantClient, m, lambda *a, _m=m, **k: pytest.fail(f"attach mode called {_m}"))
    return install


@needs_qdrant
def test_qdrant_attach_mode_reads_an_existing_collection_and_never_writes(tmp_path, enc, no_qdrant_writes):
    name, docs = _make_attached(enc)
    no_qdrant_writes()
    cfg = {"vector": {"kind": "qdrant", "mode": "attach", "url": ":memory:", "collection": name, **ATTACH_FIELDS},
           "graph": {"kind": "local"}}
    c = Corpus(tmp_path / "home", enc, cfg)
    assert c.attached and c.exists() and c.graph is None and Corpus.open(tmp_path / "home", enc, cfg) is not None
    c.connect()
    cands = c.retrieve("sourdough bread", k=3)
    top = cands[0]
    assert top["id"] == "3" and top["doc_id"] == "bakery" and top["title"] == "Bluebird Bakery"
    assert top["text"].startswith("Bluebird Bakery sells") and top["updated_at"] == "2024-05-01T00:00:00Z"
    assert top["authority"] == 1.0                                # 7 clipped to [0, 1], like local documents
    assert all(c_["sources"] == ["vector"] and np.isnan(c_["lexical_score"]) for c_ in cands)   # no BM25 here
    want = enc.encode(["sourdough bread"], kind="query")[0]
    v = enc.encode([f"{docs[2]['heading']}\n{docs[2]['body']}"], kind="passage")[0]
    assert top["score"] == pytest.approx(float(v @ want / np.linalg.norm(v) / np.linalg.norm(want)), abs=1e-4)
    assert c.chunk("3")["text"] == docs[2]["body"] and c.chunk("99") is None
    assert c.stats()["vectors"] == 4 and c.stats()["vector_mode"] == "attach" and c.stats()["docs"] is None
    for write in (lambda: c.ingest_records(RECS), lambda: c.remove(["bakery"]), lambda: c.reembed(),
                  lambda: c.vectors.upsert(["x"], np.ones((1, DIM)), [{}]), lambda: c.vectors.delete(["3"])):
        with pytest.raises(ContractError) as e:
            write()
        assert e.value.kind == "read_only_connector"
    assert not (tmp_path / "home" / "corpus").exists()          # nothing stored locally either


@needs_qdrant
def test_qdrant_attach_refuses_a_collection_of_another_dimension(tmp_path, enc):
    name, _ = _make_attached(enc, size=32)
    for mode in ("attach", "index"):
        c = Corpus(tmp_path / mode, enc, {"vector": {"kind": "qdrant", "mode": mode, "url": ":memory:",
                                                     "collection": name, **(ATTACH_FIELDS if mode == "attach" else {})}})
        with pytest.raises(ContractError) as e:
            c.connect() if mode == "attach" else c.ingest_records(RECS)
        assert e.value.kind == "encoder_mismatch" and "32-dimensional" in e.value.message
        assert "64-dimensional" in e.value.message and "Query vectors come from open-rlcd's encoder" in e.value.message
        assert "pass candidates in the request" in e.value.message
    with pytest.raises(ContractError, match="encoder_mismatch|32-dimensional"):
        Corpus(tmp_path / "x", enc, {"vector": {"kind": "qdrant", "mode": "attach", "url": ":memory:",
                                                "collection": name}}).retrieve("anything")
    missing = Corpus(tmp_path / "m", enc, {"vector": {"kind": "qdrant", "mode": "attach", "url": ":memory:",
                                                      "collection": uniq("absent")}})
    with pytest.raises(ContractError, match="does not exist") as e:
        missing.connect()
    assert e.value.kind == "connector_config"


@needs_qdrant
def test_qdrant_attach_dot_distance_named_vectors_and_payload_ids(tmp_path, enc):
    name, docs = _make_attached(enc, distance="Dot", named="dense", scale=True, id_key="chunk_key")
    c = Corpus(tmp_path / "h", enc, {"vector": {"kind": "qdrant", "mode": "attach", "url": ":memory:",
                                                "collection": name, "id_field": "chunk_key", **ATTACH_FIELDS}})
    cands = c.retrieve("sourdough bread", k=4)
    q = enc.encode(["sourdough bread"], kind="query")[0]
    for cand in cands:
        i = int(cand["id"].split("-")[1]) - 1                 # ids come from the payload field, not the point id
        v = enc.encode([f"{docs[i]['heading']}\n{docs[i]['body']}"], kind="passage")[0]
        # a dot-product collection with unnormalised vectors: the score is still the cosine, computed here
        assert cand["score"] == pytest.approx(float(v @ q / np.linalg.norm(v) / np.linalg.norm(q)), abs=1e-4)
    assert c.vectors.vector_name == "dense"
    got = c.vectors.fetch(["kb-2", "kb-9", "2"])
    assert set(got) == {"kb-2"} and got["kb-2"][1]["doc_id"] == "northwind"


@needs_qdrant
@pytest.mark.filterwarnings("ignore:Api key is used with an insecure connection")
def test_qdrant_unreachable_is_a_503_without_url_or_key(tmp_path, enc, monkeypatch, caplog):
    monkeypatch.setenv("RLCD_TEST_QDRANT_KEY", "qk-very-secret")
    port = closed_port()
    c = Corpus(tmp_path / "h", enc, {"vector": {"kind": "qdrant", "mode": "attach", "collection": "kb",
                                                "url": f"http://127.0.0.1:{port}", "timeout": 2,
                                                "api_key_env": "RLCD_TEST_QDRANT_KEY"}})
    with caplog.at_level("WARNING", logger="open_rlcd.connectors"), pytest.raises(ContractError) as e:
        c.connect()
    assert e.value.status == 503 and e.value.kind == "connector_unavailable"
    msg = json.dumps(e.value.body())
    assert str(port) not in msg and "127.0.0.1" not in msg and "qk-very-secret" not in msg and "collection 'kb'" in msg
    assert "qk-very-secret" not in caplog.text


@needs_qdrant
def test_qdrant_api_key_comes_from_the_named_environment_variable(monkeypatch):
    import qdrant_client

    from open_rlcd.connectors.qdrant import QdrantVectorStore

    seen = []
    monkeypatch.setattr(qdrant_client, "QdrantClient", lambda **kw: seen.append(kw) or object())
    store = QdrantVectorStore("c", DIM, url="http://localhost:6333", api_key_env="RLCD_TEST_QDRANT_KEY")
    monkeypatch.setenv("RLCD_TEST_QDRANT_KEY", "qk-1")
    store._make_client()
    monkeypatch.delenv("RLCD_TEST_QDRANT_KEY")
    store._make_client()
    assert [s["api_key"] for s in seen] == ["qk-1", None] and seen[0]["url"] == "http://localhost:6333"
    assert seen[0]["check_compatibility"] is False


# ---------------------------------------------------------------------------- decide() and the pack on Qdrant

def _project(tmp_path: Path, **pack_config) -> Project:
    p = Project(root=tmp_path, data={"pack": "documents", "pack_config": {"encoder": ENC_CFG, **pack_config}})
    p.home_override = tmp_path / "home"
    return p


@needs_qdrant
def test_decide_retrieves_from_an_attached_collection(tmp_path, enc, monkeypatch):
    name, _ = _make_attached(enc)
    vec = {"kind": "qdrant", "mode": "attach", "url": ":memory:", "collection": name, **ATTACH_FIELDS}
    svc = Service(_project(tmp_path, vector=vec), home=tmp_path / "home")
    out = svc.decide(DecideRequest(state={"query": "Who sells sourdough bread?"}))
    assert out["candidates_from"] == "corpus" and out["ranked"][0]["id"] in {"1", "2", "3", "4"}
    assert "3" in [r["id"] for r in out["ranked"]]
    # an attached collection of another dimension: retrieval refused, the plan says so (no evidence, a warning)
    other, _ = _make_attached(enc, size=32)
    svc2 = Service(_project(tmp_path / "b", vector={**vec, "collection": other}), home=tmp_path / "b" / "home")
    out2 = svc2.decide(DecideRequest(state={"query": "Who sells sourdough bread?"}))
    assert out2["candidates_from"] == "none" and out2["plan"]["use_chunks"] == []
    assert any("32-dimensional" in w for w in out2["warnings"])
    # a store that cannot be reached: a 503, never a plan without evidence
    down = {**vec, "url": f"http://127.0.0.1:{closed_port()}", "timeout": 2}
    svc3 = Service(_project(tmp_path / "c", vector=down), home=tmp_path / "c" / "home")
    with pytest.raises(ContractError) as e:
        svc3.decide(DecideRequest(state={"query": "Who sells sourdough bread?"}))
    assert e.value.status == 503 and e.value.kind == "connector_unavailable" and "127.0.0.1" not in e.value.message


def _cli_project(tmp_path: Path, **pack_config) -> tuple[Path, Path]:
    root = tmp_path / "proj"
    (root / "data").mkdir(parents=True)
    for r in RECS:
        (root / "data" / f"{r['id']}.md").write_text(f"# {r['title']}\n\n{r['text']}", encoding="utf-8")
    (root / "rlcd.yaml").write_text(yaml.safe_dump({"name": "p", "pack": "documents", "home": "./var",
                                                    "pack_config": {"encoder": ENC_CFG, **pack_config}}),
                                    encoding="utf-8")
    return root, root / "var"


@needs_qdrant
def test_cli_ingest_into_qdrant(tmp_path, capsys):
    root, home = _cli_project(tmp_path)
    name = uniq()
    argv = ["--project", str(root), "--home", str(home), "ingest", str(root / "data")]
    assert main([*argv, "--qdrant", ":memory:", "--collection", name]) == 0
    io = capsys.readouterr()
    res = json.loads(io.out)
    assert res["docs_added"] == 4 and res["stores"]["vector"] == {"kind": "qdrant", "collection": name}
    assert res["corpus"]["vectors"] == res["chunks_total"] and res["corpus"]["vector_backend"] == "qdrant"
    assert "note:" in io.err and "pack_config" in io.err and name in io.err
    assert main([*argv, "--collection", "only"]) == 1
    assert "--collection needs --qdrant" in capsys.readouterr().err
    # rlcd.yaml still says local: serving refuses to search an index it was not told about ...
    project = Project(root=root, data=yaml.safe_load((root / "rlcd.yaml").read_text()))
    project.home_override = home
    pack = DocumentsPack()
    with pytest.raises(ContractError, match="indexed into other stores"):
        pack.corpus(project, pack.provider(project))
    # ... until rlcd.yaml names the same collection
    project.data["pack_config"]["vector"] = {"kind": "qdrant", "url": ":memory:", "collection": name}
    corpus = pack.corpus(project, pack.provider(project))
    assert corpus.retrieve("sourdough bread", k=1)[0]["id"].startswith("data/bakery.md")
    # attach mode is never turned into a writable store from the command line
    project.data["pack_config"]["vector"]["mode"] = "attach"
    with pytest.raises(ContractError) as e:
        pack.ingest(project, {"path": str(root / "data"), "qdrant": ":memory:"})
    assert e.value.kind == "read_only_connector"
    with pytest.raises(ContractError) as e:
        pack.ingest(project, {"path": str(root / "data")})
    assert e.value.kind == "read_only_connector" and "nothing to ingest" in e.value.message


# ============================================================================ pgvector (fake psycopg connection)

class FakeCursor:
    def __init__(self, conn: FakePg):
        self.conn, self._rows, self.rowcount = conn, [], 0

    def __enter__(self) -> FakeCursor:
        return self

    def __exit__(self, *a) -> None:
        return None

    def _record(self, query, params) -> str:
        sql = query.as_string(None) if hasattr(query, "as_string") else str(query)
        self.conn.statements.append({"sql": sql, "params": params, "read_only": self.conn.read_only})
        verb = sql.split()[0].upper()
        if self.conn.read_only and verb in {"INSERT", "UPDATE", "DELETE", "CREATE", "DROP", "ALTER", "TRUNCATE"}:
            raise RuntimeError("cannot execute in a read-only transaction")      # what Postgres would say
        if self.conn.fail_on and self.conn.fail_on in sql:
            raise self.conn.fail_exc
        return sql

    def execute(self, query, params=None) -> None:
        sql = self._record(query, params)
        c = self.conn
        if "JOIN pg_type" in sql:
            self._rows = [("vector", c.typmod)] if c.exists and c.vector_col in c.columns else []
        elif sql.startswith("SELECT a.attname"):
            self._rows = [(x,) for x in sorted(c.columns)] if c.exists else []
        elif sql.startswith("CREATE TABLE"):
            c.exists = True
        elif "vector_dims(" in sql:
            self._rows = [(c.stored_dims,)] if c.stored_dims else []
        elif "<=>" in sql:
            q = np.asarray(json.loads(params[0]), dtype=np.float32)
            k = params[-1]
            scored = []
            for cid, r in c.rows.items():
                v = r["emb"]
                s = float(v @ q / (np.linalg.norm(v) * np.linalg.norm(q)))
                scored.append((-s, cid, r, s))
            scored.sort(key=lambda t: (t[0], t[1]))
            self._rows = [(cid, r.get("doc_id"), r.get("title"), r.get("text"), r.get("updated_at"),
                           r.get("authority"), s) for _, cid, r, s in scored[:k]]
        elif "= ANY(%s::text[])" in sql and sql.startswith("SELECT"):
            want = params[0]
            self._rows = [(cid, r.get("doc_id"), r.get("title"), r.get("text"), r.get("updated_at"),
                           r.get("authority"), "[" + ",".join(str(float(x)) for x in r["emb"]) + "]")
                          for cid, r in c.rows.items() if cid in want]
        elif sql.startswith("DELETE FROM"):
            gone = [i for i in params[0] if i in c.rows]
            for i in gone:
                del c.rows[i]
            self.rowcount = len(gone)
        elif "count(*)" in sql:
            self._rows = [(len(c.rows),)]
        else:
            self._rows = []

    def executemany(self, query, rows) -> None:
        sql = self._record(query, rows)
        assert sql.startswith("INSERT INTO")
        for i, doc_id, title, text, upd, auth, emb in rows:
            self.conn.rows[i] = {"doc_id": doc_id, "title": title, "text": text, "updated_at": upd, "authority": auth,
                                 "emb": np.asarray(json.loads(emb), dtype=np.float32)}

    def fetchall(self) -> list:
        return list(self._rows)

    def fetchone(self):
        return self._rows[0] if self._rows else None


class FakePg:
    """One psycopg connection to a database holding one pgvector table: records every statement (SQL rendered
    from psycopg.sql, parameters, read-only state) and answers the statements PgVectorStore sends."""

    def __init__(self, *, exists=False, typmod=16, stored_dims=None, columns=None, vector_col="embedding",
                 fail_on=None, fail_exc=None):
        self.exists, self.typmod, self.stored_dims, self.vector_col = exists, typmod, stored_dims, vector_col
        self.columns = set(columns or {"id", "doc_id", "title", "text", "updated_at", "authority", "embedding"})
        self.rows: dict[str, dict] = {}
        self.statements: list[dict] = []
        self.read_only = None
        self.commits = self.rollbacks = 0
        self.closed = False
        self.fail_on, self.fail_exc = fail_on, fail_exc

    def cursor(self) -> FakeCursor:
        return FakeCursor(self)

    def commit(self) -> None:
        self.commits += 1

    def rollback(self) -> None:
        self.rollbacks += 1

    def close(self) -> None:
        self.closed = True

    def sql(self) -> str:
        return "\n".join(s["sql"] for s in self.statements)


DSN = "postgresql://rlcd:hunter2-pw@localhost:5432/docs"


@pytest.fixture
def pg(monkeypatch):
    """psycopg.connect returns the FakePg given to `use()`; every DSN it was called with is recorded."""
    import psycopg

    state = {"conn": None, "dsns": [], "kwargs": []}

    def connect(dsn, **kw):
        state["dsns"].append(dsn)
        state["kwargs"].append(kw)
        if isinstance(state["conn"], Exception):
            raise state["conn"]
        return state["conn"]

    monkeypatch.setattr(psycopg, "connect", connect)
    monkeypatch.setenv("RLCD_PG_DSN", DSN)

    def use(conn):
        state["conn"] = conn
        return conn
    state["use"] = use
    return state


@needs_psycopg
def test_pgvector_index_mode_sql_is_parameterised(pg):
    from open_rlcd.connectors.pgvector import PgVectorStore

    db = pg["use"](FakePg(exists=False, typmod=16))
    store = PgVectorStore("public.rlcd_chunks", 16)
    rng = np.random.default_rng(2)
    X = rng.normal(size=(30, 16)).astype(np.float32)
    nasty = "x'); DROP TABLE rlcd_chunks; --"
    ids = [nasty] + [f"d{i % 4}#{i}" for i in range(1, 30)]
    store.upsert(ids, X, [{"doc_id": f"d{i % 4}", "title": "T", "text": f"text {i}", "updated_at": 1700000000 + i,
                           "authority": 0.5} for i in range(30)])
    sql = db.sql()
    assert pg["dsns"] == [DSN] and db.read_only is None and db.commits >= 2       # index mode: read-write
    assert "CREATE EXTENSION IF NOT EXISTS vector" in sql
    assert 'CREATE TABLE IF NOT EXISTS "public"."rlcd_chunks" ("id" text PRIMARY KEY' in sql and "vector(16)" in sql
    assert "USING hnsw" in sql and "vector_cosine_ops" in sql
    ins = next(s for s in db.statements if s["sql"].startswith("INSERT INTO"))
    assert '("id") DO UPDATE SET "doc_id" = EXCLUDED."doc_id"' in ins["sql"] and "%s::vector" in ins["sql"]
    assert ins["params"][0][0] == nasty and ins["params"][0][6].startswith("[")       # values travel as parameters
    for q in rng.normal(size=(5, 16)).astype(np.float32):
        hits = store.search(q, k=4)
        assert [h.id for h in hits] == [ids[i] for i in _brute(X, q, 4)]
        s = db.statements[-1]
        assert "<=> %s::vector" in s["sql"] and s["sql"].endswith("LIMIT %s") and s["params"][-1] == 4
        assert json.loads(s["params"][0]) == pytest.approx(q.tolist(), abs=1e-6)
    got = store.fetch([nasty, "d1#1", "nope"])
    assert set(got) == {nasty, "d1#1"} and got["d1#1"][1]["text"] == "text 1"
    assert db.statements[-1]["params"] == [[nasty, "d1#1", "nope"]] and "::text = ANY(%s::text[])" in db.sql()
    store.search(X[1], k=3, filter={"doc_id": "d1", "title": ["T", "U"]})
    s = db.statements[-1]
    assert 'WHERE "doc_id" = %s AND "title"::text = ANY(%s::text[])' in s["sql"] and s["params"][1:3] == ["d1", ["T", "U"]]
    assert store.delete([nasty, "d1#1"]) == 2 and store.count() == 28
    assert all(nasty not in st["sql"] for st in db.statements)          # never formatted into SQL text
    assert all("hunter2" not in st["sql"] for st in db.statements)


@needs_psycopg
def test_pgvector_attach_mode_is_read_only(pg):

    cols = {"chunk_id", "body", "document_id", "heading", "modified_at", "trust", "emb"}
    db = pg["use"](FakePg(exists=True, typmod=8, columns=cols, vector_col="emb"))
    db.rows = {"k1": {"doc_id": "D1", "title": "Policy", "text": "Refunds within 14 days.",
                      "updated_at": datetime(2024, 5, 1, tzinfo=UTC), "authority": Decimal("0.75"),
                      "emb": np.arange(1, 9, dtype=np.float32)},
               "k2": {"doc_id": "D2", "title": None, "text": "Shipping takes 3 days.", "updated_at": None,
                      "authority": None, "emb": -np.arange(1, 9, dtype=np.float32)}}
    store = make_vector_store({"kind": "pgvector", "mode": "attach", "table": "kb_chunks", "id_field": "chunk_id",
                               "vector_field": "emb", "text_field": "body", "doc_id_field": "document_id",
                               "title_field": "heading", "updated_at_field": "modified_at", "authority_field": "trust"},
                              8)
    hits = store.search(np.arange(1, 9, dtype=np.float32), k=2)
    assert db.read_only is True and db.commits == 0 and db.rollbacks >= 2     # BEGIN READ ONLY, rolled back
    assert [h.id for h in hits] == ["k1", "k2"] and hits[0].score == pytest.approx(1.0)
    assert hits[0].payload == {"doc_id": "D1", "title": "Policy", "text": "Refunds within 14 days.",
                               "updated_at": "2024-05-01T00:00:00+00:00", "authority": 0.75}
    sel = db.statements[-1]["sql"]
    assert '"chunk_id"::text AS "id"' in sel and '"body" AS "text"' in sel and '"emb" <=> %s::vector' in sel
    n = len(db.statements)
    for write in (lambda: store.upsert(["x"], np.ones((1, 8)), [{}]), lambda: store.delete(["k1"])):
        with pytest.raises(ContractError) as e:
            write()
        assert e.value.kind == "read_only_connector"
    assert len(db.statements) == n                                    # refused before any statement was sent
    verbs = {s["sql"].split()[0].upper() for s in db.statements}
    assert verbs == {"SELECT"} and all(s["read_only"] for s in db.statements)


@needs_psycopg
def test_pgvector_dimension_and_mapping_checks(pg):
    from open_rlcd.connectors.pgvector import PgVectorStore

    pg["use"](FakePg(exists=True, typmod=32))
    with pytest.raises(ContractError) as e:
        PgVectorStore("chunks", 16, mode="attach").connect()
    assert e.value.kind == "encoder_mismatch" and "32-dimensional" in e.value.message
    assert "Query vectors come from open-rlcd's encoder" in e.value.message
    pg["use"](FakePg(exists=True, typmod=-1, stored_dims=24))         # untyped vector column: a stored row says 24
    with pytest.raises(ContractError, match="24-dimensional"):
        PgVectorStore("chunks", 16, mode="attach").connect()
    pg["use"](FakePg(exists=True, typmod=16, columns={"id", "text", "embedding"}))
    with pytest.raises(ContractError, match="no column") as e:          # doc_id / title / ... mapped by default
        PgVectorStore("chunks", 16, mode="attach").connect()
    assert e.value.kind == "connector_config" and "doc_id" in e.value.message
    ok = PgVectorStore("chunks", 16, mode="attach", fields={"doc_id_field": None, "title_field": None,
                                                            "updated_at_field": None, "authority_field": None})
    ok.connect()
    pg["use"](FakePg(exists=False))
    with pytest.raises(ContractError, match="never creates") as e:
        PgVectorStore("chunks", 16, mode="attach").connect()
    assert e.value.kind == "connector_config"


@needs_psycopg
def test_pgvector_failures_are_503_without_dsn_or_password(pg, caplog):
    import psycopg

    from open_rlcd.connectors.pgvector import PgVectorStore

    pg["use"](psycopg.OperationalError(f'connection to server at "localhost" failed: password authentication failed '
                                       f'for user "rlcd" ({DSN})'))
    with caplog.at_level("WARNING", logger="open_rlcd.connectors"), pytest.raises(ContractError) as e:
        PgVectorStore("chunks", 16).search(np.ones(16))
    body = json.dumps(e.value.body())
    assert e.value.status == 503 and e.value.kind == "connector_unavailable" and "OperationalError" in body
    for secret in ("hunter2", "postgresql://", "localhost", "5432"):
        assert secret not in body
    assert "hunter2" not in caplog.text and "OperationalError" in caplog.text
    # a failure in the middle of a query: 503, the connection is dropped and reopened on the next call
    db = pg["use"](FakePg(exists=True, typmod=16, fail_on="<=>", fail_exc=psycopg.OperationalError("server closed")))
    store = PgVectorStore("chunks", 16)
    with pytest.raises(ContractError) as e:
        store.search(np.ones(16))
    assert e.value.status == 503 and db.closed
    db2 = pg["use"](FakePg(exists=True, typmod=16))
    assert store.search(np.ones(16)) == [] and len(pg["dsns"]) == 3 and not db2.closed


@needs_psycopg
def test_pgvector_dsn_comes_from_the_environment_and_its_hosts_are_checked(pg, monkeypatch):
    from open_rlcd.connectors.pgvector import PgVectorStore

    pg["use"](FakePg(exists=True, typmod=16))
    monkeypatch.delenv("RLCD_ALLOW_REMOTE", raising=False)
    monkeypatch.delenv("RLCD_PG_DSN")
    with pytest.raises(ContractError, match="RLCD_PG_DSN") as e:
        PgVectorStore("chunks", 16)
    assert e.value.kind == "connector_config"
    monkeypatch.setenv("KB_DSN", "postgresql://u:pw-x@db.internal:5432/kb")
    with pytest.raises(ContractError, match="refusing non-loopback host 'db.internal'") as e:
        make_vector_store({"kind": "pgvector", "table": "t", "dsn_env": "KB_DSN"}, 16)
    assert "pw-x" not in e.value.message
    assert make_vector_store({"kind": "pgvector", "table": "t", "dsn_env": "KB_DSN"}, 16, allow_remote=True)
    for dsn in ("host=/var/run/postgresql dbname=kb", "postgresql:///kb?host=/tmp", "host=localhost,127.0.0.1 dbname=x",
                "dbname=kb"):
        monkeypatch.setenv("KB_DSN", dsn)
        assert PgVectorStore("t", 16, dsn_env="KB_DSN")
    for dsn in ("host=localhost,db.internal dbname=x", "hostaddr=10.1.2.3 dbname=x"):
        monkeypatch.setenv("KB_DSN", dsn)
        with pytest.raises(ContractError, match="refusing non-loopback host"):
            PgVectorStore("t", 16, dsn_env="KB_DSN")
    monkeypatch.setenv("KB_DSN", "dbname=kb")
    monkeypatch.setenv("PGHOST", "db.internal")            # libpq would use PGHOST: so is the check
    with pytest.raises(ContractError, match="db.internal"):
        PgVectorStore("t", 16, dsn_env="KB_DSN")
    with pytest.raises(ContractError, match="schema.table"):
        PgVectorStore("a.b.c", 16)


@needs_psycopg
def test_pgvector_check_config_refuses_what_the_store_refuses_without_psycopg_or_the_dsn(pg, monkeypatch):
    """`rlcd config check` runs PgVectorStore.check_config (factory.check_store): it shares the constructor's own
    checks, so it refuses a configuration with the message building the store gives, and it needs neither psycopg
    nor the DSN in the environment to do so."""
    from open_rlcd.connectors import pgvector as PG

    cases = [{"kind": "pgvector"}, {"kind": "pgvector", "table": "a.b.c"},
             {"kind": "pgvector", "table": "t", "mode": "attach", "id_field": None},
             {"kind": "pgvector", "table": "t", "mode": "attach", "text_field": ""},
             {"kind": "pgvector", "table": "t", "dsn_env": "postgresql://localhost/db"},
             {"kind": "pgvector", "table": "t", "dsn": DSN}, {"kind": "pgvector", "table": "t", "collection": "c"}]
    for cfg in cases:
        with pytest.raises(ContractError) as built:
            make_vector_store(cfg, DIM)
        with pytest.raises(ContractError) as checked:
            check_store(cfg, "vector")
        assert checked.value.message == built.value.message and checked.value.kind == "connector_config", cfg
        assert "hunter2" not in checked.value.message

    def offline(*a, **kw):
        raise AssertionError("checking a configuration needs no client library and no environment variable")

    monkeypatch.setattr(PG, "require_extra", offline)
    monkeypatch.setattr(PG, "env_secret", offline)
    monkeypatch.delenv("RLCD_PG_DSN")
    check_store({"kind": "pgvector", "mode": "attach", "table": "public.kb", "id_field": "chunk_id",
                 "authority_field": None}, "vector")
    with pytest.raises(AssertionError):                      # building the store does need them
        make_vector_store({"kind": "pgvector", "table": "t"}, DIM)


@needs_psycopg
def test_corpus_on_pgvector_index_mode_matches_the_local_corpus(tmp_path, enc, pg):
    db = pg["use"](FakePg(exists=False, typmod=DIM))
    remote = Corpus(tmp_path / "r", enc, {"vector": {"kind": "pgvector", "table": "rlcd_chunks"}})
    local = Corpus(tmp_path / "l", enc)
    remote.ingest_records(RECS)
    local.ingest_records(RECS)
    assert set(db.rows) == set(remote.chunks) and db.rows["bakery#0"]["text"].startswith("Bluebird")
    for q in QUERIES:
        a, b = remote.retrieve(q, k=3), local.retrieve(q, k=3)
        assert [c["id"] for c in a] == [c["id"] for c in b]
        assert np.allclose([c["score"] for c in a], [c["score"] for c in b], atol=1e-4)
    meta = json.loads((tmp_path / "r" / "corpus" / "meta.json").read_text())
    assert meta["stores"]["vector"] == {"kind": "pgvector", "table": "rlcd_chunks"}
    assert "hunter2" not in json.dumps(meta)


# ============================================================================ Neo4j (fake driver)

class FakeResult:
    def __init__(self, rows: list[dict]):
        self.rows = rows

    def __iter__(self):
        return iter([FakeRecord(r) for r in self.rows])

    def consume(self) -> None:
        return None


class FakeRecord(dict):
    def data(self) -> dict:
        return dict(self)


class FakeNeo4j:
    """A neo4j.Driver stand-in: records every query with its parameters, the session's access mode and whether it
    ran in execute_read or execute_write; `answer(query, params)` gives the rows."""

    def __init__(self, answer=None, fail: Exception | None = None):
        self.answer = answer or (lambda q, p: [])
        self.fail = fail
        self.calls: list[dict] = []
        self.closed = False

    def verify_connectivity(self) -> None:
        if self.fail is not None:
            raise self.fail

    def session(self, **kw):
        driver = self

        class Tx:
            def __init__(self, how: str):
                self.how = how

            def run(self, query, parameters=None, **kwp):
                params = dict(parameters or {}, **kwp)
                driver.calls.append({"query": query, "params": params, "how": self.how,
                                     "access": kw.get("default_access_mode"), "database": kw.get("database")})
                return FakeResult(driver.answer(query, params))

        class Session:
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return None

            def execute_read(self, fn, *a, **k):
                return fn(Tx("read"), *a, **k)

            def execute_write(self, fn, *a, **k):
                return fn(Tx("write"), *a, **k)

        return Session()

    def close(self) -> None:
        self.closed = True


@pytest.fixture
def neo(monkeypatch):
    """neo4j.GraphDatabase.driver returns the FakeNeo4j given to `use()`; (uri, auth, config) are recorded."""
    import neo4j

    state = {"driver": FakeNeo4j(), "opened": []}

    def driver(uri, auth=None, **config):
        state["opened"].append({"uri": uri, "auth": auth, **config})
        return state["driver"]

    monkeypatch.setattr(neo4j.GraphDatabase, "driver", driver)

    def use(d):
        state["driver"] = d
        return d
    state["use"] = use
    return state


TEMPLATE = ("MATCH (e:Entity)<-[:MENTIONS]-(c:Chunk) WHERE e.key IN $keys AND c.status <> 'SET by import' "
            "RETURN c.chunk_id AS id, 1 AS path_length, COUNT { (c)--() } AS degree, e.name AS via, c.body AS text, "
            "c.doc AS doc_id LIMIT $limit")


@needs_neo4j
def test_neo4j_attach_template_runs_verbatim_with_parameters_in_read_transactions(neo):
    import neo4j

    drv = neo["use"](FakeNeo4j(answer=lambda q, p: [{"id": "c1", "path_length": 1, "degree": 4, "via": "Zentrix",
                                                      "text": "Zentrix X1 ...", "doc_id": "d1"},
                                                     {"id": "c1", "path_length": 2, "degree": 1},
                                                     {"id": 7, "path_length": "2", "degree": None},
                                                     {"path_length": 1}]))
    store = make_graph_store({"kind": "neo4j", "mode": "attach", "uri": "bolt://localhost:7687", "database": "kb",
                              "expand_cypher": TEMPLATE, "max_fanout": 50})
    nasty = "Zentrix'}) DETACH DELETE n //"
    rows = store.expand(["Zentrix", nasty], hops=2, limit=5)
    assert rows == [{"id": "c1", "path_length": 1, "degree": 4, "via": "Zentrix", "text": "Zentrix X1 ...",
                     "doc_id": "d1"}, {"id": "7", "path_length": 2, "degree": None, "via": None}]
    (call,) = drv.calls
    assert call["query"] == TEMPLATE                               # verbatim: nothing formatted into it
    assert call["how"] == "read" and call["access"] == neo4j.READ_ACCESS and call["database"] == "kb"
    assert call["params"]["terms"] == ["Zentrix", nasty] and "zentrix" in call["params"]["keys"]
    assert {k: call["params"][k] for k in ("hops", "max_edges", "limit", "max_fanout")} == \
        {"hops": 2, "max_edges": 4, "limit": 5, "max_fanout": 50}
    assert store.expand([], hops=2) == [] and len(drv.calls) == 1     # no entity: no query at all
    assert store.neighbors(["c1"]) == [] and store.chunk_degree("c1") is None   # no templates for them configured
    for write in (lambda: store.add_document("d", [("d#0", "Zentrix")]), lambda: store.remove_document("d")):
        with pytest.raises(ContractError) as e:
            write()
        assert e.value.kind == "read_only_connector"
    assert all(c["how"] == "read" for c in drv.calls) and len(drv.calls) == 1


@needs_neo4j
@pytest.mark.parametrize("template, msg", [
    ("MATCH (e) WHERE e.key IN $keys MERGE (x:Seen) RETURN e.id AS id, 1 AS path_length", "MERGE"),
    ("MATCH (c:Chunk) WHERE c.key IN $keys DETACH DELETE c RETURN 1 AS id", "DETACH"),
    ("MATCH (c:Chunk) WHERE c.key IN $keys SET c.seen = true RETURN c.id AS id", "SET"),
    ("LOAD CSV FROM 'file:///x' AS row RETURN row[0] AS id, $keys AS k", "LOAD"),
    ("MATCH (c:Chunk) RETURN c.id AS id, 1 AS path_length", "must use the parameter"),
    ("", "non-empty"),
])
def test_neo4j_templates_with_writes_or_without_parameters_are_refused(template, msg):
    with pytest.raises(ContractError, match=msg) as e:
        make_graph_store({"kind": "neo4j", "mode": "attach", "uri": "bolt://localhost:7687",
                          "expand_cypher": template})
    assert e.value.kind == "connector_config"
    with pytest.raises(ContractError, match="needs expand_cypher"):
        make_graph_store({"kind": "neo4j", "mode": "attach", "uri": "bolt://localhost:7687"})


@needs_neo4j
def test_neo4j_index_mode_writes_open_rlcds_graph_with_parameters(neo):
    import neo4j

    from open_rlcd.connectors import neo4j as N

    drv = neo["use"](FakeNeo4j(answer=lambda q, p: [{"degree": 3}] if q == N.DEGREE_CYPHER else []))
    store = make_graph_store({"kind": "neo4j", "uri": "bolt://localhost:7687"}, terms=["東京本社"])
    store.add_document("acme/a.md", [("acme/a.md#0", "Zentrix X1 drones use parts supplied by Northwind Metals."),
                                     ("acme/a.md#1", "東京本社が管理。")], links=["acme/b.md", "acme/a.md"])
    writes = [c for c in drv.calls if c["how"] == "write"]
    assert [c["query"] for c in writes[:3]] == list(N.SCHEMA)
    assert [c["query"] for c in writes[3:]] == [N.DROP_CHUNKS, N.DROP_LINKS, N.UPSERT_DOC, N.UPSERT_LINKS]
    assert all(c["access"] == neo4j.WRITE_ACCESS for c in writes)
    up = writes[5]["params"]
    assert up["doc_id"] == "acme/a.md" and [ch["id"] for ch in up["chunks"]] == ["acme/a.md#0", "acme/a.md#1"]
    keys = {e["key"] for ch in up["chunks"] for e in ch["entities"]}
    assert {"zentrix x1", "northwind metals", "東京本社"} <= keys
    assert writes[6]["params"]["links"] == ["acme/b.md"]             # no self link
    assert all("acme" not in c["query"] and "Zentrix" not in c["query"] for c in drv.calls)
    store.add_document("acme/b.md", [("acme/b.md#0", "Northwind Metals is run by Ingrid Solberg.")])
    assert sum(1 for c in drv.calls if c["query"] in N.SCHEMA) == 3   # constraints are created once
    n = len(drv.calls)
    store.expand(["Zentrix X1"], hops=2, limit=7)
    store.expand(["Zentrix X1"], hops=9, limit=7)
    two, deep = drv.calls[n:]
    assert two["query"] == N.EXPAND_CYPHER and two["how"] == "read" and two["access"] == neo4j.READ_ACCESS
    assert two["params"] == {"terms": ["Zentrix X1"], "keys": ["zentrix x1"], "hops": 2, "max_edges": 4, "limit": 7,
                             "max_fanout": 200}
    assert deep["query"] == N.EXPAND_CYPHER_DEEP and deep["params"]["hops"] == 3 and deep["params"]["max_edges"] == 6
    store.neighbors(["acme/b.md#0"], hops=1)
    assert drv.calls[-1]["query"] == N.NEIGHBORS_CYPHER and drv.calls[-1]["params"]["ids"] == ["acme/b.md#0"]
    assert store.chunk_degree("acme/a.md#0") == 3 and drv.calls[-1]["params"] == {"id": "acme/a.md#0"}
    store.remove_document("acme/a.md")
    assert [c["query"] for c in drv.calls[-3:]] == [N.DROP_CHUNKS, N.DROP_LINKS, N.DROP_DOC]
    # literal variable-length bounds (Cypher takes no parameter there), the requested depth as a parameter filter
    assert "*1..4]" in N.EXPAND_CYPHER and "*1..6]" in N.EXPAND_CYPHER_DEEP and "*1..4]" in N.NEIGHBORS_CYPHER
    assert all("length(p) <= $max_edges" in q for q in (N.EXPAND_CYPHER, N.EXPAND_CYPHER_DEEP, N.NEIGHBORS_CYPHER))


@needs_neo4j
def test_neo4j_credentials_and_failures(neo, monkeypatch, caplog):
    import neo4j

    monkeypatch.setenv("RLCD_TEST_NEO4J_PW", "neo-pw-123")
    monkeypatch.setenv("RLCD_TEST_NEO4J_USER", "reader")
    store = make_graph_store({"kind": "neo4j", "mode": "attach", "uri": "bolt://localhost:7687",
                              "user_env": "RLCD_TEST_NEO4J_USER", "password_env": "RLCD_TEST_NEO4J_PW",
                              "expand_cypher": TEMPLATE})
    store.connect()
    assert neo["opened"][-1]["auth"] == ("reader", "neo-pw-123") and neo["opened"][-1]["uri"] == "bolt://localhost:7687"
    make_graph_store({"kind": "neo4j", "uri": "bolt://localhost:7687"}).connect()
    assert neo["opened"][-1]["auth"] is None                          # no password_env: no authentication
    neo["use"](FakeNeo4j(fail=neo4j.exceptions.ServiceUnavailable(
        "Couldn't connect to localhost:7687 (resolved to ('127.0.0.1:7687',)) with neo-pw-123")))
    with caplog.at_level("WARNING", logger="open_rlcd.connectors"), pytest.raises(ContractError) as e:
        make_graph_store({"kind": "neo4j", "uri": "bolt://localhost:7687",
                          "password_env": "RLCD_TEST_NEO4J_PW"}).expand(["Zentrix"])
    body = json.dumps(e.value.body())
    assert e.value.status == 503 and e.value.kind == "connector_unavailable" and "ServiceUnavailable" in body
    assert "localhost" not in body and "7687" not in body and "neo-pw-123" not in body
    assert "neo-pw-123" not in caplog.text
    neo["use"](FakeNeo4j(fail=neo4j.exceptions.AuthError("The client is unauthorized due to authentication failure.")))
    with pytest.raises(ContractError) as e:
        make_graph_store({"kind": "neo4j", "uri": "bolt://localhost:7687",
                          "password_env": "RLCD_TEST_NEO4J_PW"}).connect()
    assert e.value.status == 503 and "RLCD_TEST_NEO4J_PW" in e.value.message and "neo-pw-123" not in e.value.message


# ---------------------------------------------------------------------------- the corpus on Neo4j (+ Qdrant)

@needs_neo4j
def test_corpus_with_local_vectors_and_a_neo4j_index_graph(tmp_path, enc, neo):
    from open_rlcd.connectors import neo4j as N

    def answer(q, p):
        if q == N.EXPAND_CYPHER and "zentrix x1" in p["keys"]:
            return [{"id": "zentrix#0", "path_length": 1, "degree": 4, "via": "Zentrix X1"},
                    {"id": "northwind#0", "path_length": 2, "degree": 3, "via": "Northwind Metals"},
                    {"id": "ghost#0", "path_length": 2, "degree": 1}]           # not in the corpus: dropped
        return []

    drv = neo["use"](FakeNeo4j(answer=answer))
    c = Corpus(tmp_path / "h", enc, {"graph": {"kind": "neo4j", "uri": "bolt://localhost:7687"}})
    c.ingest_records(RECS)
    upserts = [x for x in drv.calls if x["query"] == N.UPSERT_DOC]
    assert sorted(x["params"]["doc_id"] for x in upserts) == sorted(r["id"] for r in RECS)
    assert not (tmp_path / "h" / "corpus" / "graph.json").exists()
    meta = json.loads((tmp_path / "h" / "corpus" / "meta.json").read_text())
    assert meta["stores"]["graph"] == {"kind": "neo4j", "database": None, "mode": "index"}
    cands = c.retrieve("Who supplies parts for the Zentrix X1?", k=3)
    north = next(x for x in cands if x["id"] == "northwind#0")
    assert "graph" in north["sources"] and north["graph"]["path_length"] == 2
    assert all(x["id"] != "ghost#0" for x in cands)
    c.remove(["bakery"])
    assert drv.calls[-1]["query"] == N.DROP_DOC and drv.calls[-1]["params"] == {"doc_id": "bakery"}
    # back to the local graph: refused until an ingest rebuilds graph.json from the chunks
    with pytest.raises(ContractError, match="indexed into other stores"):
        Corpus(tmp_path / "h", enc)
    back = Corpus(tmp_path / "h", enc, reindex=True)
    assert back.reindexed == ["graph"] and (tmp_path / "h" / "corpus" / "graph.json").is_file()
    assert any(x["graph"]["path_length"] == 2 for x in back.retrieve("Who supplies parts for the Zentrix X1?", k=3)
               if "graph" in x)


@needs_neo4j
@needs_qdrant
def test_corpus_with_qdrant_and_neo4j_both_attached(tmp_path, enc, neo, no_qdrant_writes):
    name, docs = _make_attached(enc, id_key="chunk_key")
    no_qdrant_writes()

    cfg = {"vector": {"kind": "qdrant", "mode": "attach", "url": ":memory:", "collection": name,
                      "id_field": "chunk_key", **ATTACH_FIELDS},
           "graph": {"kind": "neo4j", "mode": "attach", "uri": "bolt://localhost:7687", "expand_cypher": TEMPLATE}}
    c = Corpus(tmp_path / "h", enc, cfg)
    q = "Who supplies Zentrix?"
    qv = enc.encode([q], kind="query")[0]
    top3 = [h.id for h in c.vectors.search(qv, k=3)]
    outside = next(f"kb-{i}" for i in range(1, 5) if f"kb-{i}" not in top3)   # the one the vector search misses

    def answer(q, p):
        return [{"id": outside, "path_length": 1, "degree": 2},                                # in Qdrant
                {"id": "ext-9", "path_length": 2, "degree": 1, "text": "Northwind also supplies Orion Labs.",
                 "doc_id": "ext"},                                                             # text from the graph
                {"id": "gone-1", "path_length": 1, "degree": 1}]                               # unknown anywhere

    drv = neo["use"](FakeNeo4j(answer=answer))
    cands = {x["id"]: x for x in c.retrieve(q, k=3)}
    assert "gone-1" not in cands and set(cands) == {*top3, outside, "ext-9"}
    hit = cands[outside]
    assert hit["sources"] == ["graph"] and hit["graph"]["path_length"] == 1
    d = docs[int(outside.split("-")[1]) - 1]
    v = enc.encode([f"{d['heading']}\n{d['body']}"], kind="passage")[0]
    # a graph hit the vector search did not return: fetched by id and scored by the same cosine
    assert hit["score"] == pytest.approx(float(v @ qv / np.linalg.norm(v) / np.linalg.norm(qv)), abs=1e-4)
    assert hit["text"] == d["body"] and hit["doc_id"] == d["source"]
    ext = cands["ext-9"]
    assert ext["text"].startswith("Northwind also") and ext["sources"] == ["graph"] and np.isnan(ext["score"])
    assert all(x["how"] == "read" for x in drv.calls)
    with pytest.raises(ContractError) as e:
        c.ingest_records(RECS)
    assert e.value.kind == "read_only_connector"


@needs_neo4j
def test_cli_ingest_into_neo4j(tmp_path, capsys, neo, monkeypatch):
    from open_rlcd.connectors import neo4j as N

    monkeypatch.delenv("RLCD_ALLOW_REMOTE", raising=False)
    drv = neo["use"](FakeNeo4j())
    root, home = _cli_project(tmp_path, graph={"kind": "local", "terms": ["東京本社"], "hub_min": 5})
    argv = ["--project", str(root), "--home", str(home), "ingest", str(root / "data")]
    assert main([*argv, "--neo4j", "bolt://localhost:7687"]) == 0
    io = capsys.readouterr()
    res = json.loads(io.out)
    assert res["stores"]["graph"]["kind"] == "neo4j" and res["corpus"]["graph"]["backend"] == "neo4j"
    assert len([c for c in drv.calls if c["query"] == N.UPSERT_DOC]) == 4 and "note:" in io.err
    tokyo = next(c for c in drv.calls if c["query"] == N.UPSERT_DOC and c["params"]["doc_id"].endswith("tokyo.md"))
    assert "東京本社" in {e["key"] for ch in tokyo["params"]["chunks"] for e in ch["entities"]}   # terms kept
    assert main([*argv, "--neo4j", "bolt://graph.example.org:7687"]) == 1
    assert "refusing non-loopback host" in capsys.readouterr().err


# ============================================================================ live servers (skipped by default)

@pytest.mark.network
@needs_qdrant
def test_live_qdrant_index_round_trip(tmp_path, enc):
    url = os.environ.get("RLCD_TEST_QDRANT_URL")
    if not url:
        pytest.skip("set RLCD_TEST_QDRANT_URL (e.g. http://localhost:6333)")
    c = Corpus(tmp_path / "h", enc, {"vector": {"kind": "qdrant", "url": url, "collection": uniq("rlcd_test")},
                                     "allow_remote": True})
    c.ingest_records(RECS)
    assert c.retrieve("sourdough bread", k=1)[0]["id"] == "bakery#0"


@pytest.mark.network
@needs_psycopg
def test_live_pgvector_index_and_attach_round_trip(tmp_path, enc):
    """Against a real Postgres with the vector extension (RLCD_TEST_PG_DSN): index mode creates its table and
    matches the local corpus; attach mode maps someone else's columns, runs in READ ONLY transactions (the server
    refuses a write) and refuses a table of another dimension."""
    import psycopg

    from open_rlcd.connectors.pgvector import PgVectorStore, vector_literal

    dsn = os.environ.get("RLCD_TEST_PG_DSN")
    if not dsn:
        pytest.skip("set RLCD_TEST_PG_DSN to a Postgres with the vector extension")
    table, kb, small, untyped = uniq("rlcd_test"), uniq("kb_test"), uniq("kb_small"), uniq("kb_untyped")
    vec = {"kind": "pgvector", "dsn_env": "RLCD_TEST_PG_DSN", "table": table}
    try:
        c = Corpus(tmp_path / "h", enc, {"vector": vec, "allow_remote": True})
        local = Corpus(tmp_path / "l", enc)
        c.ingest_records(RECS)
        local.ingest_records(RECS)
        for q in QUERIES:
            a, b = c.retrieve(q, k=3), local.retrieve(q, k=3)
            assert [x["id"] for x in a] == [x["id"] for x in b]
            assert np.allclose([x["score"] for x in a], [x["score"] for x in b], atol=1e-4)
        c.ingest_records([{**RECS[2], "text": "Bluebird Bakery now sells rye bread."}])
        assert c.vectors.count() == len(c.chunks) and c.remove(["bakery"]) == 1
        assert c.vectors.count() == len(c.chunks) == len(local.chunks) - 1
        # somebody else's table: integer ids, other column names, timestamptz, numeric
        with psycopg.connect(dsn) as conn:
            conn.execute(f"CREATE TABLE {kb} (pk bigint PRIMARY KEY, body text, src text, ts timestamptz, "
                         f"trust numeric, emb vector({DIM}))")
            conn.execute(f"CREATE TABLE {small} (id text PRIMARY KEY, text text, emb vector(32))")
            conn.execute(f"CREATE TABLE {untyped} (id text PRIMARY KEY, text text, emb vector)")
            for i, r in enumerate(RECS, 1):
                v = enc.encode([r["text"]], kind="passage")[0]
                conn.execute(f"INSERT INTO {kb} VALUES (%s, %s, %s, now(), 0.5, %s::vector)",
                             (i, r["text"], r["id"], vector_literal(v)))
                conn.execute(f"INSERT INTO {untyped} VALUES (%s, %s, %s::vector)",
                             (r["id"], r["text"], vector_literal(np.ones(24))))
        att_cfg = {"kind": "pgvector", "mode": "attach", "dsn_env": "RLCD_TEST_PG_DSN", "table": kb, "id_field": "pk",
                   "vector_field": "emb", "text_field": "body", "doc_id_field": "src", "title_field": None,
                   "updated_at_field": "ts", "authority_field": "trust"}
        att = Corpus(tmp_path / "a", enc, {"vector": att_cfg, "allow_remote": True})
        top = att.retrieve("sourdough bread", k=2)[0]
        assert top["id"] == "3" and top["doc_id"] == "bakery" and top["authority"] == 0.5
        assert isinstance(top["updated_at"], str) and np.isnan(top["lexical_score"])
        assert att.chunk("3")["text"].startswith("Bluebird")
        with pytest.raises(ContractError) as e:             # the server itself refuses writes in attach mode
            att.vectors._run(lambda cur: cur.execute(f"DELETE FROM {kb}"))
        assert e.value.status == 503 and "ReadOnlySqlTransaction" in e.value.message
        assert att.vectors.count() == 4
        for t, dims in ((small, 32), (untyped, 24)):
            with pytest.raises(ContractError, match=f"{dims}-dimensional") as e:
                PgVectorStore(t, DIM, dsn_env="RLCD_TEST_PG_DSN", mode="attach",
                              fields={"vector_field": "emb", "doc_id_field": None, "title_field": None,
                                      "updated_at_field": None, "authority_field": None}).connect()
            assert e.value.kind == "encoder_mismatch"
    finally:
        with psycopg.connect(dsn, autocommit=True) as conn:
            for t in (table, kb, small, untyped):
                conn.execute(f"DROP TABLE IF EXISTS {t}")


@pytest.mark.network
@needs_neo4j
def test_live_neo4j_index_round_trip(tmp_path, enc):
    if not os.environ.get("RLCD_TEST_NEO4J_URI"):
        pytest.skip("set RLCD_TEST_NEO4J_URI (and RLCD_TEST_NEO4J_PASSWORD) to a disposable Neo4j 5 database")
    g = {"kind": "neo4j", "uri": os.environ["RLCD_TEST_NEO4J_URI"], "password_env": "RLCD_TEST_NEO4J_PASSWORD"}
    c = Corpus(tmp_path / "h", enc, {"graph": g, "allow_remote": True})
    c.ingest_records(RECS)
    hits = [x for x in c.retrieve(QUERIES[0], k=3) if "graph" in x["sources"]]
    assert any(x["id"] == "northwind#0" and x["graph"]["path_length"] == 2 for x in hits)
    c.remove([r["id"] for r in RECS])
