"""Encoders: the offline hashing encoder, the sqlite cache, config round trips, and the ONNX encoder's offline
behaviour (no download attempted) and inference plumbing (with a fake session and a locally built tokenizer)."""
from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from open_rlcd.features import text as text_mod
from open_rlcd.features.text import (
    CachedEncoder,
    Encoder,
    EncoderMismatch,
    EncoderUnavailable,
    HashingEncoder,
    OnnxEncoder,
    encoder_config,
    l2_normalize,
    make_encoder,
    mean_pool,
)


def cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))


# ----------------------------------------------------------------------------- hashing encoder

def test_hashing_shape_dtype_and_unit_norm():
    enc = HashingEncoder(dim=128, seed=3)
    X = enc.encode(["Acme quarterly revenue", "東京本社", "Улаанбаатар салбар"])
    assert X.shape == (3, 128) and X.dtype == np.float32
    assert np.allclose(np.linalg.norm(X, axis=1), 1.0, atol=1e-5)
    assert enc.id == "hashing-128-s3" and len(enc.sha256) == 64
    assert isinstance(enc, Encoder)


def test_hashing_is_deterministic_across_instances_and_processes():
    texts = ["Zentrix X1 drones", "ゼントリクス社の部品", "Бат-Эрдэнэ удирддаг"]
    a = HashingEncoder(64).encode(texts)
    b = HashingEncoder(64).encode(texts)
    assert np.array_equal(a, b)
    code = ("import json,sys; from open_rlcd.features.text import HashingEncoder; "
            "print(json.dumps(HashingEncoder(64).encode(json.loads(sys.argv[1])).round(6).tolist()))")
    env = {**os.environ, "PYTHONHASHSEED": "12345"}
    out = subprocess.run([sys.executable, "-c", code, json.dumps(texts)], capture_output=True, text=True, env=env,
                         check=True, timeout=60)
    assert np.allclose(np.asarray(json.loads(out.stdout)), a.round(6), atol=1e-6)


def test_hashing_seed_kind_and_empty_text():
    t = ["the same text"]
    assert not np.allclose(HashingEncoder(64, seed=0).encode(t), HashingEncoder(64, seed=1).encode(t))
    enc = HashingEncoder(64)
    assert np.array_equal(enc.encode(t, kind="query"), enc.encode(t, kind="passage"))
    empty = enc.encode(["", "   ", "!!!"])
    assert np.all(empty == 0.0)
    assert enc.encode([]).shape == (0, 64)


def test_hashing_unicode_normalisation():
    enc = HashingEncoder(128)
    full, half = enc.encode(["ＡＣＭＥ　Ｒｏｂｏｔｉｃｓ １２００", "acme robotics 1200"])
    assert np.allclose(full, half)


@pytest.mark.parametrize("a, b, c", [
    ("The quarterly revenue report for Acme Robotics", "Acme Robotics quarterly revenue report",
     "Bluebird Bakery sells sourdough bread every morning"),
    ("東京本社の従業員数は350人です", "東京本社の従業員は何人ですか", "大阪は今日も雨が降っている"),
    ("Улаанбаатар дахь салбар 2019 онд нээгдсэн", "Улаанбаатар салбар хэзээ нээгдсэн бэ",
     "Дархан хотын цаг агаар өнөөдөр сайхан"),
])
def test_hashing_similar_texts_are_closer_than_unrelated(a, b, c):
    va, vb, vc = HashingEncoder(384).encode([a, b, c])
    assert cos(va, vb) > cos(va, vc) + 0.1


def test_hashing_rejects_tiny_dim():
    with pytest.raises(ValueError):
        HashingEncoder(dim=4)


# ----------------------------------------------------------------------------- maths helpers

def test_l2_normalize_rows_and_zero_rows():
    x = np.array([[3.0, 4.0], [0.0, 0.0]])
    y = l2_normalize(x)
    assert y.dtype == np.float32
    assert np.allclose(y[0], [0.6, 0.8]) and np.all(y[1] == 0.0)
    assert np.allclose(l2_normalize(np.array([0.0, 2.0])), [0.0, 1.0])


def test_mean_pool_ignores_padding():
    h = np.array([[[1.0, 2.0], [3.0, 4.0], [100.0, 100.0]],
                  [[5.0, 5.0], [9.0, 9.0], [9.0, 9.0]]])
    m = np.array([[1, 1, 0], [1, 0, 0]])
    assert np.allclose(mean_pool(h, m), [[2.0, 3.0], [5.0, 5.0]])
    assert np.all(mean_pool(h, np.zeros((2, 3))) == 0.0)


# ----------------------------------------------------------------------------- cache

class Counting:
    """An encoder that counts how many texts it was asked to encode."""

    def __init__(self) -> None:
        self.inner = HashingEncoder(32)
        self.id, self.dim, self.sha256 = self.inner.id, self.inner.dim, self.inner.sha256
        self.calls: list[list[str]] = []

    def encode(self, texts, kind="passage"):
        self.calls.append(list(texts))
        return self.inner.encode(texts, kind)


def test_cached_encoder_hits_cache_and_survives_reopen(tmp_path):
    base = Counting()
    cached = CachedEncoder(base, tmp_path)
    first = cached.encode(["alpha", "beta", "alpha"])
    assert base.calls == [["alpha", "beta"]]          # misses only, deduplicated
    assert np.allclose(first, base.inner.encode(["alpha", "beta", "alpha"]))
    again = cached.encode(["beta", "alpha"])
    assert len(base.calls) == 1 and np.allclose(again, first[[1, 0]])
    assert cached.hits >= 2 and cached.misses == 2
    cached.encode(["alpha"], kind="query")            # a different kind is a different key
    assert base.calls[-1] == ["alpha"]
    cached.close()
    assert (tmp_path / f"{base.sha256[:16]}.sqlite").is_file()

    base2 = Counting()
    reopened = CachedEncoder(base2, tmp_path)
    assert np.allclose(reopened.encode(["alpha", "beta"]), first[:2])
    assert base2.calls == [] and len(reopened) == 3
    assert reopened.id == base2.id and reopened.dim == 32 and isinstance(reopened, Encoder)


# ----------------------------------------------------------------------------- config round trip

def test_make_encoder_hashing_and_config_round_trip(tmp_path):
    enc = make_encoder({"kind": "hashing", "dim": 96, "seed": 2}, tmp_path)
    assert isinstance(enc, CachedEncoder) and enc.dim == 96
    assert enc.path.parent == tmp_path / "cache" / "embeddings"
    cfg = encoder_config(enc)
    assert cfg == {"kind": "hashing", "id": "hashing-96-s2", "dim": 96, "sha256": enc.sha256, "seed": 2}
    again = make_encoder(json.loads(json.dumps(cfg)), tmp_path)
    assert again.sha256 == enc.sha256
    assert np.allclose(again.encode(["x y z"]), enc.encode(["x y z"]))
    with pytest.raises(EncoderMismatch):
        make_encoder({**cfg, "sha256": "0" * 64}, tmp_path)
    assert isinstance(make_encoder({"kind": "hashing", "cache": False}, tmp_path), HashingEncoder)
    with pytest.raises(ValueError, match="unknown encoder kind"):
        make_encoder({"kind": "word2vec"}, tmp_path)


# ----------------------------------------------------------------------------- ONNX: offline

def _no_network(monkeypatch):
    def refuse(*a, **k):
        raise AssertionError("network access attempted")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)


def test_onnx_offline_and_not_cached_raises_clear_error_without_downloading(tmp_path, monkeypatch):
    hub = pytest.importorskip("huggingface_hub")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    _no_network(monkeypatch)
    seen = []
    real = hub.hf_hub_download

    def spy(*a, **k):
        seen.append(k.get("local_files_only"))
        return real(*a, **k)

    monkeypatch.setattr(hub, "hf_hub_download", spy)
    with pytest.raises(EncoderUnavailable) as e:
        OnnxEncoder(home=tmp_path)
    msg = str(e.value)
    assert "HF_HUB_OFFLINE" in msg and "run once online" in msg.lower() and "hashing" in msg
    assert seen and all(x is True for x in seen)
    with pytest.raises(EncoderUnavailable):
        make_encoder(None, tmp_path)                  # the default is the ONNX encoder


# ----------------------------------------------------------------------------- ONNX: plumbing with fakes

def _fake_model_dir(tmp_path: Path) -> Path:
    tokenizers = pytest.importorskip("tokenizers")
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace

    words = ["<pad>", "[UNK]", "query", "passage", ":", "alpha", "beta", "gamma", "delta"]
    tok = tokenizers.Tokenizer(WordLevel({w: i for i, w in enumerate(words)}, unk_token="[UNK]"))
    tok.pre_tokenizer = Whitespace()
    d = tmp_path / "fake-model"
    (d / "onnx").mkdir(parents=True)
    tok.save(str(d / "tokenizer.json"))
    (d / "onnx" / "model_qint8_avx512_vnni.onnx").write_bytes(b"not really an onnx model")
    return d


class _FakeSession:
    """Embeds token id i as a one-hot-ish vector, like a model whose hidden states depend only on the token."""

    DIM = 8

    def __init__(self, path, sess_options=None, providers=None):
        self.providers = providers

    def get_inputs(self):
        return [type("I", (), {"name": n})() for n in ("input_ids", "attention_mask", "token_type_ids")]

    def get_outputs(self):
        return [type("O", (), {"shape": ["batch", "seq", self.DIM]})()]

    def run(self, _names, feed):
        ids = feed["input_ids"]
        assert feed["attention_mask"].shape == ids.shape and not feed["token_type_ids"].any()
        h = np.zeros((*ids.shape, self.DIM), np.float32)
        for (b, s), i in np.ndenumerate(ids):
            h[b, s, i % self.DIM] = 1.0
            h[b, s, (i * 3 + 1) % self.DIM] += 0.5
        return [h]


@pytest.fixture
def fake_onnx(tmp_path, monkeypatch):
    ort = pytest.importorskip("onnxruntime")
    hub = pytest.importorskip("huggingface_hub")
    d = _fake_model_dir(tmp_path)
    calls = []

    def fake_download(repo_id, filename, cache_dir=None, local_files_only=False):
        calls.append((repo_id, filename, cache_dir, local_files_only))
        return str(d / filename)

    monkeypatch.setattr(hub, "hf_hub_download", fake_download)
    monkeypatch.setattr(ort, "InferenceSession", _FakeSession)
    return d, calls


def test_onnx_encoder_plumbing_with_fake_session(tmp_path, fake_onnx):
    d, calls = fake_onnx
    enc = OnnxEncoder(model_id="intfloat/multilingual-e5-small", home=tmp_path / "home", batch_size=2)
    assert enc.dim == _FakeSession.DIM and enc.prefixes == ("query: ", "passage: ")
    assert enc.sha256 == text_mod.file_sha256(d / "onnx" / "model_qint8_avx512_vnni.onnx")
    assert {c[1] for c in calls} == {"onnx/model_qint8_avx512_vnni.onnx", "tokenizer.json"}
    assert all(c[2] == str(tmp_path / "home" / "encoders") for c in calls)
    texts = ["alpha beta gamma delta alpha", "beta", "gamma delta", "alpha"]
    X = enc.encode(texts)
    assert X.shape == (4, _FakeSession.DIM) and np.allclose(np.linalg.norm(X, axis=1), 1.0, atol=1e-5)
    # batching sorts by length; the output order must still match the input order
    one_by_one = np.vstack([enc.encode([t]) for t in texts])
    assert np.allclose(X, one_by_one, atol=1e-6)
    assert not np.allclose(enc.encode(["beta"], kind="query"), enc.encode(["beta"], kind="passage"))
    cfg = encoder_config(enc)
    assert cfg["kind"] == "onnx" and cfg["sha256"] == enc.sha256 and cfg["model_id"] == enc.model_id
    rebuilt = make_encoder(cfg, tmp_path / "home")
    assert rebuilt.sha256 == enc.sha256 and np.allclose(rebuilt.encode(texts), X, atol=1e-6)


def test_onnx_encoder_refuses_a_different_model_file(tmp_path, fake_onnx):
    with pytest.raises(EncoderMismatch, match="must not be served"):
        OnnxEncoder(home=tmp_path, expected_sha256="f" * 64)


# ----------------------------------------------------------------------------- ONNX: the real model

@pytest.mark.network
def test_real_e5_small_download_and_japanese_paraphrase(tmp_path, monkeypatch):
    hub = pytest.importorskip("huggingface_hub")
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.setattr(hub.constants, "HF_HUB_OFFLINE", False, raising=False)
    enc = OnnxEncoder(home=tmp_path)
    assert enc.dim == 384
    q = enc.encode(["東京本社の従業員は何人ですか"], kind="query")[0]
    p = enc.encode(["東京本社の従業員数は350人です。", "Бороо орж байна, шүхрээ аваарай."], kind="passage")
    assert cos(q, p[0]) > cos(q, p[1])


def test_onnx_quantize_locally_without_onnx_package_explains(tmp_path, fake_onnx, monkeypatch):
    monkeypatch.setitem(sys.modules, "onnx", None)          # import onnx -> ImportError
    with pytest.raises(text_mod.EncoderError, match="quantize_locally needs the 'onnx' package"):
        OnnxEncoder(home=tmp_path, quantize_locally=True)
    _d, calls = fake_onnx
    assert calls[-1][1] == "onnx/model.onnx"                # the fp32 file is what gets quantized
