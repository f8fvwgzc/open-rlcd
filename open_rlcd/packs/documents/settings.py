"""pack_config of the documents pack, read in one place.

    pack_config:
      encoder: {kind: onnx, model_id: intfloat/multilingual-e5-small}   # or {kind: hashing, dim: 256} offline
      corpus: ./data                     # folder `rlcd ingest` reads when no path is given
      queries: ./queries.jsonl           # query log for `rlcd label bootstrap` (CSV with a `query` column, or JSONL)
      chunking: {max_chars: 1200, overlap: 150}
      vector: {kind: local}              # only local connectors exist so far
      graph: {kind: local, terms: [...]} # known entity names (needed for kanji names)
      retrieve_k: 5                      # Corpus.retrieve(k): each retriever's top-k, fused (up to 3k candidates)
      label_top_k: 5                     # label tasks: chunks per query for query_chunk / query_evidence items

`config_keys()` lists every key the pack reads (these and the less common ones) for `rlcd config check`.
"""
from __future__ import annotations

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
    max_evidence_age_days, require_evidence_dates -> decide.py; allow_remote -> connectors/security.py."""
    from ...connectors.local_vector import BACKENDS
    from ...core.config_check import Key
    from ...features.text import DEFAULT_ONNX_FILE

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
        "vector": Key("map", keys={"kind": Key("str", default="local", choices=("local",)),
                                   "backend": Key("str", default="auto", choices=("auto", *BACKENDS))}),
        "graph": Key("map", keys={"kind": Key("str", default="local", choices=("local",)),
                                  "terms": Key("list"),
                                  "max_fanout": Key("int", default=200, min=1),
                                  "hub_share": Key("number", default=0.05, min=0, max=1),
                                  "hub_min": Key("int", default=10, min=0)}),
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
