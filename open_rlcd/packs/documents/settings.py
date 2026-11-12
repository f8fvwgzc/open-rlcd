"""pack_config of the documents pack, read in one place.

    pack_config:
      encoder: {kind: onnx, model_id: intfloat/multilingual-e5-small}   # or {kind: hashing, dim: 256} offline
      corpus: ./data                     # folder `rlcd ingest` reads when no path is given
      queries: ./queries.jsonl           # query log for `rlcd label bootstrap` (CSV with a `query` column, or JSONL)
      chunking: {max_chars: 1200, overlap: 150}
      vector: {kind: local}              # or {kind: qdrant | pgvector, mode: index | attach, ...}
      graph: {kind: local, terms: [...]} # known entity names (needed for kanji names); or {kind: neo4j, ...}
      retrieve_k: 5                      # Corpus.retrieve(k): each retriever's top-k, fused (up to 3k candidates)
      label_top_k: 5                     # label tasks: chunks per query for query_chunk / query_evidence items

`config_keys()` lists every key the pack reads (these and the less common ones) for `rlcd config check`, and
`config_findings()` checks `vector` / `graph` against the store their `kind` names. The remote connector keys
(url / collection / dsn_env / table / uri / password_env / field mappings / Cypher templates) are documented in
open_rlcd/connectors/{qdrant,pgvector,neo4j}.py and docs/configuration.md.
"""
from __future__ import annotations

import re
from typing import Any

from ...connectors.security import remote_allowed

DEFAULT_ENCODER = {"kind": "onnx", "model_id": "intfloat/multilingual-e5-small"}
DEFAULT_RETRIEVE_K = 5


def encoder_settings(pack_config: dict) -> dict:
    enc = pack_config.get("encoder")
    return dict(enc) if isinstance(enc, dict) and enc else dict(DEFAULT_ENCODER)


def corpus_settings(pack_config: dict) -> dict:
    """The `Corpus(config=...)` dict built from pack_config."""
    chunking = pack_config.get("chunking") if isinstance(pack_config.get("chunking"), dict) else {}
    out: dict = {}
    for key in ("vector", "graph"):
        if isinstance(pack_config.get(key), dict):
            out[key] = dict(pack_config[key])
    for key in ("max_chars", "overlap"):
        if chunking.get(key) is not None:
            out[key] = int(chunking[key])
    for key in ("rrf_k", "batch_size", "on_encoder_change"):
        if pack_config.get(key) is not None:
            out[key] = pack_config[key]
    return out


def project_corpus_settings(project: Any) -> dict:
    """`corpus_settings` of a project plus `allow_remote` from its security settings (`security.allow_remote`,
    `pack_config.allow_remote` or RLCD_ALLOW_REMOTE=1): remote connector hosts are refused without it."""
    out = corpus_settings(project.pack_config)
    out["allow_remote"] = remote_allowed(project)
    return out


def retrieve_k(pack_config: dict) -> int:
    try:
        return max(1, int(pack_config.get("retrieve_k") or DEFAULT_RETRIEVE_K))
    except (TypeError, ValueError):
        return DEFAULT_RETRIEVE_K


def config_keys() -> dict:
    """Every `pack_config` key the documents pack reads, with its type and default, for `rlcd config check`
    (core/config_check.py). Where each is read: encoder -> features/text.py `make_encoder`; corpus, chunking ->
    `DocumentsPack.ingest`; queries, label_top_k -> tasks.py; vector, graph, rrf_k, batch_size, on_encoder_change
    -> corpus.py `Corpus`; retrieve_k -> `retrieve_k` above; feature_cache -> `DocumentsPack.provider`;
    max_evidence_age_days, require_evidence_dates -> decide.py; allow_remote -> connectors/security.py.

    `vector` and `graph` hold the keys of every store kind (connectors/factory.py `store_keys`: the local stores'
    and the `KEYS` of connectors/{qdrant,pgvector,neo4j}.py); which of them a kind reads is `config_findings`'
    business. The defaults of `batch_size` and `id_field` depend on the kind, so they are left to the docs."""
    from ...connectors.factory import GRAPH_KINDS, MODES, VECTOR_KINDS
    from ...connectors.local_vector import BACKENDS
    from ...connectors.pgvector import DEFAULT_DSN_ENV
    from ...core.config_check import Key
    from ...features.text import DEFAULT_ONNX_FILE

    def column(default: str | None) -> Key:
        # attach mode: the payload key (qdrant) / column (pgvector) holding a candidate field; null = none
        return Key("str", default=default)

    vector = {
        "kind": Key("str", default="local", choices=VECTOR_KINDS),
        "backend": Key("str", default="auto", choices=("auto", *BACKENDS)),                # local
        "mode": Key("str", default="index", choices=MODES),                                # qdrant, pgvector
        "timeout": Key("int", default=10, min=1),
        "batch_size": Key("int", min=1),
        "id_field": column(None), "text_field": column("text"), "doc_id_field": column("doc_id"),
        "title_field": column("title"), "updated_at_field": column("updated_at"),
        "authority_field": column("authority"),
        "url": Key("str"), "path": Key("str"), "collection": Key("str"),                   # qdrant
        "api_key_env": Key("str"), "vector_name": Key("str"),
        "table": Key("str"), "dsn_env": Key("str", default=DEFAULT_DSN_ENV),              # pgvector
        "vector_field": Key("str", default="embedding"), "create_index": Key("bool", default=True),
    }
    graph = {
        "kind": Key("str", default="local", choices=GRAPH_KINDS),
        "terms": Key("list"),                                                              # local, neo4j
        "max_fanout": Key("int", default=200, min=1),
        "hub_share": Key("number", default=0.05, min=0, max=1),                            # local
        "hub_min": Key("int", default=10, min=0),
        "mode": Key("str", default="index", choices=MODES),                                # neo4j
        "uri": Key("str"), "user": Key("str", default="neo4j"), "user_env": Key("str"),
        "password_env": Key("str"), "database": Key("str"), "timeout": Key("int", default=10, min=1),
        "expand_cypher": Key("str"), "neighbors_cypher": Key("str"), "degree_cypher": Key("str"),
    }
    return {
        "encoder": Key("map", keys={
            "kind": Key("str", default="onnx", choices=("onnx", "hashing")),
            "model_id": Key("str", default=DEFAULT_ENCODER["model_id"]),
            "file": Key("str", default=DEFAULT_ONNX_FILE),
            "dim": Key("int", default=384, min=1),
            "seed": Key("int", default=0),
            "max_length": Key("int", default=512, min=1),
            "batch_size": Key("int", default=32, min=1),
            "quantize_locally": Key("bool", default=False),
            "sha256": Key("str"),
            "cache": Key("bool", default=True),
        }),
        "corpus": Key("str"),
        "queries": Key("str"),
        "chunking": Key("map", keys={"max_chars": Key("int", default=1200, min=1),
                                     "overlap": Key("int", default=150, min=0)}),
        "vector": Key("map", keys=vector),
        "graph": Key("map", keys=graph),
        "retrieve_k": Key("int", default=DEFAULT_RETRIEVE_K, min=1),
        "label_top_k": Key("int", default=DEFAULT_RETRIEVE_K, min=1),
        "rrf_k": Key("number", default=60, min=0, open_min=True),
        "batch_size": Key("int", default=64, min=1),
        "on_encoder_change": Key("str", default="error", choices=("error", "reembed")),
        "feature_cache": Key("int", default=1024, min=0),
        "max_evidence_age_days": Key("number", min=0),
        "require_evidence_dates": Key("bool", default=False),
        "allow_remote": Key("bool", default=False),
    }


def config_findings(pack_config: Any) -> list[dict]:
    """What `rlcd config check` adds about `vector` and `graph` once every key has the right type
    (DocumentsPack.check_config): each is checked against the store its `kind` names, as that store reads it.

    * A key of another kind is refused by a remote store when it is built (an error here) and ignored by a local
      one (a warning: `collection:` without `kind: qdrant` builds a local index instead).
    * A remote store's own checks run offline (connectors/factory.py `check_store`): the collection / table / uri
      it needs, a `*_env` that is not a variable name, a URL with credentials, a Cypher template with a write
      clause, ... No client library, environment variable or connection is used, and any host passes: whether a
      remote host is allowed depends on the environment the store runs in (RLCD_ALLOW_REMOTE)."""
    from ...connectors.factory import KINDS, check_store, store_keys
    from ...core.contract import ContractError

    out: list[dict] = []
    for what in ("vector", "graph"):
        cfg = pack_config.get(what) if isinstance(pack_config, dict) else None
        if not isinstance(cfg, dict) or str(cfg.get("kind") or "local") not in KINDS[what]:
            continue                # not a mapping, or a kind the key check refuses
        kind = str(cfg.get("kind") or "local")
        own = store_keys(what, kind)
        # a remote store refuses a key it does not read even when it is empty (connectors/base.py `check_keys`)
        foreign = sorted(str(k) for k, v in cfg.items() if k not in own and (kind != "local" or v is not None))
        for key in foreign:
            owners = " and ".join(k for k in KINDS[what] if key in store_keys(what, k))
            if kind == "local":
                out.append({"level": "warn", "path": f"pack_config.{what}.{key}",
                            "message": f"is ignored: `kind` is local{'' if cfg.get('kind') else ' (not set)'}, and "
                                       f"the local {what} store does not read it"
                                       + (f". It is a key of {owners}" if owners else "")})
            else:
                out.append({"level": "error", "path": f"pack_config.{what}.{key}",
                            "message": (f"is a key of {owners}, not of {kind}" if owners else f"is not a key of "
                                        f"{kind}") + f": the {kind} store refuses it. Its keys are: "
                                                     f"{', '.join(sorted(own))}"})
        if kind == "local" or foreign:
            continue
        try:
            check_store(cfg, what)
        except ContractError as e:
            # "pack_config.graph.expand_cypher contains ..." is reported at that key, without repeating it
            m = re.match(rf"(pack_config\.{what}\.[a-z_]+)\b\s*", e.message)
            out.append({"level": "error", "path": m.group(1) if m else f"pack_config.{what}",
                        "message": e.message[m.end():] if m else e.message})
    return out
