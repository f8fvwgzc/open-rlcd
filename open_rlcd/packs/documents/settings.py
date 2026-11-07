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
