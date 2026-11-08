"""A tiny ONNX text encoder for offline tests, written with a few lines of protobuf (no `onnx` package needed).

The graph is one Gather over a token-embedding table: input_ids [batch, seq] -> last_hidden_state [batch, seq, D].
`OnnxEncoder` mean-pools and L2-normalises it, so it is a (weak) bag-of-subwords sentence encoder that ONNX
Runtime runs like any exported model. `tiny_tokenizer` trains a small byte-level BPE tokenizer on given texts.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


def _varint(n: int) -> bytes:
    out = bytearray()
    while True:
        b, n = n & 0x7F, n >> 7
        out.append(b | 0x80 if n else b)
        if not n:
            return bytes(out)


def _ld(num: int, payload: bytes) -> bytes:          # length-delimited field
    return _varint(num << 3 | 2) + _varint(len(payload)) + payload


def _vi(num: int, v: int) -> bytes:                  # varint field
    return _varint(num << 3) + _varint(v)


def _s(num: int, text: str) -> bytes:
    return _ld(num, text.encode())


def tiny_onnx(emb: np.ndarray) -> bytes:
    """ModelProto bytes: ir_version 8, opset 13, Gather(emb, input_ids)."""
    V, D = emb.shape
    tensor = _vi(1, V) + _vi(1, D) + _vi(2, 1) + _s(8, "emb") + _ld(9, np.ascontiguousarray(emb, "<f4").tobytes())
    node = _s(1, "emb") + _s(1, "input_ids") + _s(2, "last_hidden_state") + _s(3, "gather") + _s(4, "Gather")

    def dim_param(name: str) -> bytes:
        return _ld(1, _s(2, name))

    in_type = _ld(1, _vi(1, 7) + _ld(2, dim_param("batch") + dim_param("seq")))                  # int64
    out_type = _ld(1, _vi(1, 1) + _ld(2, dim_param("batch") + dim_param("seq") + _ld(1, _vi(1, D))))   # float
    graph = (_ld(1, node) + _s(2, "tiny") + _ld(5, tensor) + _ld(11, _s(1, "input_ids") + _ld(2, in_type))
             + _ld(12, _s(1, "last_hidden_state") + _ld(2, out_type)))
    return _vi(1, 8) + _s(2, "open-rlcd-test") + _ld(7, graph) + _ld(8, _s(1, "") + _vi(2, 13))


def tiny_tokenizer(texts: list[str], path: Path, vocab_size: int = 500) -> int:
    """Train a byte-level BPE tokenizer on `texts`, save it as tokenizer.json at `path`; returns the vocab size."""
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers

    tok = Tokenizer(models.BPE(unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.train_from_iterator(texts, trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=["[PAD]", "[UNK]"],
                                                       initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
                                                       show_progress=False))
    path.parent.mkdir(parents=True, exist_ok=True)
    tok.save(str(path))
    return tok.get_vocab_size()
