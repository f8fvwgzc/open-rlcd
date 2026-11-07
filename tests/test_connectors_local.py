"""Local connectors: host allow-listing, the local vector store (exact vs brute force, persistence, filters) and
the local entity graph (multi-hop expansion with and without networkx, persistence)."""
from __future__ import annotations

import json
import sys

import numpy as np
import pytest

from open_rlcd.connectors import local_vector
from open_rlcd.connectors.base import ConnectorError, GraphStore, Hit, VectorStore, unavailable
from open_rlcd.connectors.local_graph import LocalGraphStore, entities, normalize_entity
from open_rlcd.connectors.local_vector import LocalVectorStore
from open_rlcd.connectors.security import check_host, remote_allowed
from open_rlcd.core.config import Project
from open_rlcd.core.contract import ContractError

# ----------------------------------------------------------------------------- security

LOCAL = ["localhost", "LOCALHOST", "127.0.0.1", "127.5.6.7:8080", "http://localhost:6333", "https://127.0.0.1/x",
         "[::1]:6333", "::1", "http://[::1]:8000/v1", "::ffff:127.0.0.1", "bolt://localhost:7687",
         "postgresql://rlcd@localhost:5432/docs", "unix:///var/run/qdrant.sock", "/var/run/postgresql",
         "postgresql:///docs?host=/var/run/postgresql", "postgresql:///docs", "http+unix://%2Ftmp%2Fs.sock/"]
REMOTE = ["example.com", "http://10.0.0.5:6333", "https://api.anthropic.com/v1", "0.0.0.0", "http://0.0.0.0:80",
          "http://localhost.evil.com", "192.168.1.2:5432", "postgresql://u:pw@db.internal:5432/x", "2001:db8::1",
          "bolt://neo4j.example.org:7687", "http://127.0.0.1.nip.io"]


@pytest.mark.parametrize("target", LOCAL)
def test_check_host_allows_loopback_and_unix_sockets(target):
    assert check_host(target) in {"localhost", "127.0.0.1", "127.5.6.7", "::1", "::ffff:127.0.0.1", "unix"}


@pytest.mark.parametrize("target", REMOTE)
def test_check_host_refuses_remote_hosts_by_default(target, monkeypatch):
    monkeypatch.delenv("RLCD_ALLOW_REMOTE", raising=False)
    with pytest.raises(ValueError, match="RLCD_ALLOW_REMOTE=1") as e:
        check_host(target)
    assert "security.allow_remote" in str(e.value)
    assert check_host(target, allow_remote=True)
    monkeypatch.setenv("RLCD_ALLOW_REMOTE", "1")
    assert check_host(target)
    monkeypatch.setenv("RLCD_ALLOW_REMOTE", "true")       # only the exact value "1" opts in
    with pytest.raises(ValueError):
        check_host(target)


@pytest.mark.parametrize("bad", ["", "   ", None, "http://"])
def test_check_host_rejects_empty(bad):
    with pytest.raises(ValueError):
        check_host(bad)


def test_remote_allowed_sources(monkeypatch, tmp_path):
    monkeypatch.delenv("RLCD_ALLOW_REMOTE", raising=False)
    assert remote_allowed(None) is False and remote_allowed({}) is False
    assert remote_allowed({"allow_remote": True}) and remote_allowed({"security": {"allow_remote": True}})
    assert remote_allowed({"pack_config": {"allow_remote": True}})
    assert remote_allowed(Project(root=tmp_path, data={"security": {"allow_remote": True}}))
    assert remote_allowed(Project(root=tmp_path, data={"pack_config": {"allow_remote": True}}))
    assert not remote_allowed(Project(root=tmp_path, data={}))
    monkeypatch.setenv("RLCD_ALLOW_REMOTE", "1")
    assert remote_allowed(None) and remote_allowed(Project(root=tmp_path, data={}))


def test_unavailable_is_a_503_contract_error():
    e = unavailable("qdrant at localhost:6333 is down", url="http://localhost:6333")
    assert isinstance(e, ContractError) and e.status == 503 and e.kind == "connector_unavailable"
    assert e.body() == {"error": "qdrant at localhost:6333 is down", "kind": "connector_unavailable",
                        "url": "http://localhost:6333"}
    assert issubclass(ConnectorError, Exception)


# ----------------------------------------------------------------------------- vector store

def _data(n=400, dim=24, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, dim)).astype(np.float32)
    ids = [f"c{i}" for i in range(n)]
    payloads = [{"doc_id": f"d{i % 7}", "n": i} for i in range(n)]
    return X, ids, payloads


def _brute(X, q, k, rows=None):
    Xn = X / np.linalg.norm(X, axis=1, keepdims=True)
    qn = q / np.linalg.norm(q)
    s = Xn @ qn
    idx = np.arange(len(X)) if rows is None else np.asarray(rows)
    order = idx[np.argsort(-s[idx], kind="stable")][:k]
    return [f"c{i}" for i in order], s[order]


BACKENDS = ["numpy", pytest.param("faiss", marks=pytest.mark.skipif(local_vector._importable("faiss") is None,
                                                                     reason="faiss not installed"))]


@pytest.mark.parametrize("backend", BACKENDS)
def test_vector_store_exact_matches_brute_force(tmp_path, backend):
    X, ids, payloads = _data()
    store = LocalVectorStore(tmp_path / "v", dim=24, backend=backend)
    assert store.backend == backend and isinstance(store, VectorStore)
    store.upsert(ids[:250], X[:250], payloads[:250])
    store.upsert(ids[250:], X[250:], payloads[250:])     # appends after the index was built incrementally
    store.search(X[0], k=3)
    assert store.count() == 400
    rng = np.random.default_rng(1)
    for q in rng.normal(size=(10, 24)).astype(np.float32):
        hits = store.search(q, k=7)
        want_ids, want_scores = _brute(X, q, 7)
        assert [h.id for h in hits] == want_ids
        assert np.allclose([h.score for h in hits], want_scores, atol=1e-5)
        assert all(isinstance(h, Hit) and h.payload["doc_id"].startswith("d") for h in hits)
    assert len(store.search(X[0], k=1000)) == 400
    assert store.search(X[0], k=0) == []


@pytest.mark.parametrize("backend", BACKENDS)
def test_vector_store_filter_upsert_delete_and_persistence(tmp_path, backend):
    X, ids, payloads = _data(120, 16)
    store = LocalVectorStore(tmp_path / "v", dim=16, backend=backend)
    store.upsert(ids, X, payloads)
    q = X[5]
    hits = store.search(q, k=5, filter={"doc_id": "d3"})
    rows = [i for i in range(120) if i % 7 == 3]
    assert [h.id for h in hits] == _brute(X, q, 5, rows)[0]
    assert {h.payload["doc_id"] for h in store.search(q, k=50, filter={"doc_id": ["d1", "d2"]})} == {"d1", "d2"}
    assert store.search(q, k=5, filter={"doc_id": "nope"}) == []

    store.upsert(["c7"], -X[5], [{"doc_id": "moved"}])   # overwrite: c7 now points away from q
    assert store.count() == 120 and store.payload("c7") == {"doc_id": "moved"}
    assert store.search(q, k=120)[-1].id == "c7"
    assert store.delete(["c5", "c9", "missing"]) == 2 and "c5" not in store and store.count() == 118
    assert store.search(q, k=1)[0].id != "c5"

    store.save()
    files = sorted(p.name for p in (tmp_path / "v").iterdir())
    assert files == ["items.jsonl", "meta.json", "vectors.npy"]
    np.load(tmp_path / "v" / "vectors.npy", allow_pickle=False)      # loads without pickle
    again = LocalVectorStore.load(tmp_path / "v", backend=backend)
    assert again.count() == 118 and again.ids() == store.ids() and again.dim == 16
    for qq in X[:5]:
        a, b = store.search(qq, k=6), again.search(qq, k=6)
        assert [h.id for h in a] == [h.id for h in b] and [h.payload for h in a] == [h.payload for h in b]
    assert np.isnan(again.vectors(["c5"])).all() and np.allclose(np.linalg.norm(again.vectors(["c1"])), 1.0)


def test_vector_store_validation(tmp_path):
    store = LocalVectorStore(tmp_path / "v", dim=4, backend="numpy")
    assert store.search(np.ones(4), k=3) == []
    with pytest.raises(ValueError):
        store.upsert(["a"], np.ones((1, 5)), [{}])
    with pytest.raises(ValueError):
        store.upsert(["a", "a"], np.ones((2, 4)), [{}, {}])
    store.upsert(["a", "z"], np.array([[1, 0, 0, 0], [0, 0, 0, 0]], np.float32))
    assert store.search(np.array([2.0, 0, 0, 0]), k=1)[0].score == pytest.approx(1.0)
    assert store.search(np.ones(4), k=2)[1].score == 0.0               # an all-zero vector scores 0
    with pytest.raises(ValueError):
        store.search(np.ones(3))
    with pytest.raises(ValueError):
        LocalVectorStore(tmp_path / "w", dim=4, backend="annoy")
    with pytest.raises(ConnectorError):
        LocalVectorStore.load(tmp_path / "nothing-here")


def test_vector_store_missing_backend_and_auto(tmp_path, monkeypatch):
    if local_vector._importable("hnswlib") is None:
        with pytest.raises(ConnectorError, match="documents"):
            LocalVectorStore(tmp_path, dim=4, backend="hnswlib")
    monkeypatch.setattr(local_vector, "_importable", lambda name: None)
    assert LocalVectorStore(tmp_path, dim=4).backend == "numpy"


@pytest.mark.skipif(local_vector._importable("faiss") is None, reason="faiss not installed")
def test_vector_store_switches_to_hnsw_above_threshold(tmp_path, monkeypatch):
    monkeypatch.setattr(local_vector, "HNSW_THRESHOLD", 100)
    X, ids, payloads = _data(600, 16, seed=3)
    store = LocalVectorStore(tmp_path / "v", dim=16, backend="faiss")
    store.upsert(ids, X, payloads)
    rng = np.random.default_rng(5)
    recall = []
    for q in rng.normal(size=(20, 16)).astype(np.float32):
        got = {h.id for h in store.search(q, k=10)}
        recall.append(len(got & set(_brute(X, q, 10)[0])) / 10)
    assert store._index_kind == "faiss-hnsw" and np.mean(recall) >= 0.9


# ----------------------------------------------------------------------------- entities

def test_entities_three_languages():
    assert entities("Zentrix X1 drones use parts supplied by Northwind Metals.") == \
        ["Zentrix X1", "X1", "Northwind Metals"]
    assert entities("Northwind Metals' CEO is Ingrid Solberg. The plant is in Oslo.") == \
        ["Northwind Metals", "CEO", "Ingrid Solberg", "Oslo"]
    assert entities("Who supplies Zentrix?") == ["Zentrix"]
    assert entities("Revenue grew; revenue is up.") == []          # a capitalised ordinary word is not a name
    assert entities('Please ship SKU-4471 with "Q3 batch" labels.') == ["SKU-4471", "Q3 batch", "Q3"]
    assert entities("ゼントリクス社はノースウィンド・メタルズから部品を調達。東京本社が管理。", extra=["東京本社"]) == \
        ["ゼントリクス", "ノースウィンド・メタルズ", "東京本社"]
    assert entities("Улаанбаатар хотод Номин Трейд компанийн салбар бий. Салбарыг Бат-Эрдэнэ удирддаг.") == \
        ["Улаанбаатар", "Номин Трейд", "Бат-Эрдэнэ"]
    assert normalize_entity("Улаанбаатарын") == normalize_entity("Улаанбаатар") == "улаанбаатар"
    assert normalize_entity("  ACME's ") == "acme"


# ----------------------------------------------------------------------------- graph store

CHAIN = {
    "a": [("a#0", "Zentrix X1 drones use parts supplied by Northwind Metals."), ("a#1", "The X1 flies for an hour.")],
    "b": [("b#0", "Northwind Metals is run by Ingrid Solberg.")],
    "c": [("c#0", "Ingrid Solberg previously led Oslo Harbour Logistics.")],
    "d": [("d#0", "Bluebird Bakery sells sourdough bread.")],
}


def _graph(path, **kw) -> LocalGraphStore:
    g = LocalGraphStore(path, **kw)
    for doc, chunks in CHAIN.items():
        g.add_document(doc, chunks)
    return g


def _summary(rows):
    return [(r["id"], r["path_length"]) for r in rows]


@pytest.mark.parametrize("mode", ["networkx", "fallback", "missing"])
def test_graph_multi_hop_expand(tmp_path, monkeypatch, mode):
    if mode == "missing":
        monkeypatch.setitem(sys.modules, "networkx", None)    # import networkx -> ImportError
        g = _graph(tmp_path / "g.json")
        assert g.backend == "dict"
    elif mode == "fallback":
        g = _graph(tmp_path / "g.json", use_networkx=False)
        assert g.backend == "dict"
    else:
        pytest.importorskip("networkx")
        g = _graph(tmp_path / "g.json")
        assert g.backend == "networkx"
    assert isinstance(g, GraphStore)
    assert _summary(g.expand(["Zentrix"], hops=1)) == [("a#0", 1)]
    two = g.expand(["Zentrix"], hops=2)
    assert _summary(two) == [("a#0", 1), ("a#1", 2), ("b#0", 2)]   # a#1 through doc a, b#0 through Northwind
    assert two[2]["via"] == "Northwind Metals" and two[2]["degree"] == 3
    assert _summary(g.expand(["Zentrix"], hops=3))[-1] == ("c#0", 3)
    assert g.expand(["Bluebird Bakery"], hops=3) == [{"id": "d#0", "path_length": 1, "degree": 2,
                                                      "via": "Bluebird Bakery", "seed": "Bluebird Bakery"}]
    assert g.expand(["Unknown Corp"], hops=2) == [] and g.expand([], hops=2) == []
    assert len(g.expand(["Zentrix"], hops=3, limit=2)) == 2
    assert _summary(g.neighbors(["b#0"], hops=1)) == [("a#0", 1), ("c#0", 1)]
    assert g.neighbors(["nope"]) == []


def test_graph_backends_agree_and_json_round_trip(tmp_path):
    pytest.importorskip("networkx")
    nx_g = _graph(tmp_path / "nx.json")
    dict_g = _graph(tmp_path / "dict.json", use_networkx=False)
    for terms in (["Zentrix"], ["Ingrid Solberg"], ["Oslo Harbour Logistics", "Bluebird Bakery"]):
        assert nx_g.expand(terms, hops=3) == dict_g.expand(terms, hops=3)
    nx_g.save()
    data = json.loads((tmp_path / "nx.json").read_text(encoding="utf-8"))
    assert data["format"] == "open-rlcd-graph/1" and data["nodes"] and data["links"]
    for use_nx in (None, False):
        loaded = LocalGraphStore.load(tmp_path / "nx.json", use_networkx=use_nx)
        assert loaded.expand(["Zentrix"], hops=3) == nx_g.expand(["Zentrix"], hops=3)
        assert loaded.stats()["nodes"] == nx_g.stats()["nodes"] and loaded.stats()["edges"] == nx_g.stats()["edges"]


def test_graph_readd_remove_links_and_hubs(tmp_path):
    g = _graph(tmp_path / "g.json", use_networkx=False)
    g.add_document("b", [("b#0", "Northwind Metals is now run by Maria Lind.")])   # re-ingest replaces chunks
    assert g.expand(["Ingrid Solberg"], hops=1) == [{"id": "c#0", "path_length": 1, "degree": 3,
                                                     "via": "Ingrid Solberg", "seed": "Ingrid Solberg"}]
    assert _summary(g.expand(["Maria Lind"], hops=1)) == [("b#0", 1)]
    g.add_document("d", CHAIN["d"], links=["c"])                                 # explicit doc -> doc link
    assert ("c#0", 2) in _summary(g.expand(["Bluebird Bakery"], hops=2))
    g.add_document("d", CHAIN["d"])                                              # link dropped on re-ingest
    assert _summary(g.expand(["Bluebird Bakery"], hops=2)) == [("d#0", 1)]
    g.add_document("c", CHAIN["c"], links=["d"])
    g.add_document("d", CHAIN["d"])                                              # c's own link survives d's re-ingest
    assert ("c#0", 2) in _summary(g.expand(["Bluebird Bakery"], hops=2))
    g.remove_document("a")
    assert g.expand(["Zentrix"], hops=2) == [] and "ent:zentrix x1" not in dict(g.g.nodes())

    hub = LocalGraphStore(tmp_path / "hub.json", use_networkx=False, max_fanout=3)
    for i in range(5):
        hub.add_document(f"p{i}", [(f"p{i}#0", f"Acme Corp page {i} mentions Widget{i} Line.")])
    hub.add_document("x", [("x#0", "Widget0 Line parts come from Acme Corp and Delta Forge.")])
    rows = _summary(hub.expand(["Delta Forge"], hops=2))
    assert rows[0] == ("x#0", 1) and ("p0#0", 2) in rows        # via Widget0 Line (degree 2)
    assert ("p3#0", 2) not in rows                               # not through the Acme Corp hub (degree 6 > 3)
