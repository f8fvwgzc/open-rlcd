"""Feature layer: text encoders (`text`), lexical signals (`lexical`) and retrieval signals (`retrieval`).

Importing this package is cheap: it needs numpy only. The ONNX encoder imports onnxruntime / tokenizers /
huggingface_hub lazily, when one is constructed.
"""
from .lexical import (
    BM25,
    PAIR_FEATURES,
    QUERY_FEATURES,
    bm25_scores,
    coverage,
    number_support,
    numbers,
    pair_features,
    query_features,
    tokens,
)
from .retrieval import CHUNK_FEATURES, RETRIEVAL_FEATURES, chunk_features, retrieval_features
from .text import (
    CachedEncoder,
    Encoder,
    EncoderError,
    EncoderMismatch,
    EncoderUnavailable,
    HashingEncoder,
    OnnxEncoder,
    encoder_config,
    l2_normalize,
    make_encoder,
    mean_pool,
)

__all__ = [
    "BM25", "CHUNK_FEATURES", "PAIR_FEATURES", "QUERY_FEATURES", "RETRIEVAL_FEATURES", "CachedEncoder", "Encoder",
    "EncoderError", "EncoderMismatch", "EncoderUnavailable", "HashingEncoder", "OnnxEncoder", "bm25_scores",
    "chunk_features", "coverage", "encoder_config", "l2_normalize", "make_encoder", "mean_pool", "number_support",
    "numbers", "pair_features", "query_features", "retrieval_features", "tokens",
]
