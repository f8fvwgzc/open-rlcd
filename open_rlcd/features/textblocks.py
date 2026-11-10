"""Text feature blocks for a pack of your own: a few columns of a feature row, made from one text field.

Two blocks, one interface (`names()`, `transform(texts)`, `groups()`, `to_config()`, `signature`):

* `HashedNgrams` - word n-grams (character bigrams inside Japanese / CJK runs, see `lexical.tokens`) hashed into
  a fixed number of columns. Needs numpy only, downloads nothing, works offline. `fit(texts)` learns the
  vocabulary and with it what to call each column, so a decision's `reasons` read `text:refund` instead of
  `h402`. It sees surface forms only ("car" and "automobile" are unrelated to it).
* `TextEmbeddings` - dense vectors from a frozen encoder (`features/text.py`), through the on-disk cache under
  `<home>/cache/embeddings`. `TextEmbeddings.for_project(project)` uses the encoder rlcd.yaml configures in
  `pack_config.encoder` (as the documents pack does) and falls back to the offline hashing encoder when there
  is none. Its columns have no readable names (`emb_017`); give them a group so `reason_groups` reads well.

Both serialise to plain JSON: put `block.to_config()` inside your provider's `to_config()` and rebuild with
`text_block(config, home=project.home)` in `pack.provider(project, config)`. No pickle: a bundle stores the
n-gram vocabulary, or the encoder's identity and the hash of its model file, and the rebuilt block gives the
same numbers (an encoder whose file differs is refused, `EncoderMismatch`).

`signature` is a short hash of everything that decides a block's columns. Put it in your provider's
`feature_version` (`f"tickets-v1:{block.signature}"`): the service then refuses precomputed features made with
another vocabulary or encoder instead of scoring them.

What this does not do: no TF-IDF weighting, no stemming, no fine-tuning. Two n-grams that share a column are one
feature (the name shows the most frequent one and how many others share it): raise `dim` when many do. The
vocabulary is stored in the bundle as text, so it holds words of your training texts: mind that when the texts
are confidential and the bundle is shared.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from .lexical import tokens
from .text import Encoder, encoder_config, l2_normalize, make_encoder


def _ngrams(text: str, lo: int, hi: int) -> Counter:
    toks = tokens(text)
    out: Counter = Counter()
    for n in range(lo, hi + 1):
        for i in range(len(toks) - n + 1):
            out[" ".join(toks[i:i + n])] += 1
    return out


def _signature(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()[:12]


class HashedNgrams:
    """Token n-grams hashed into `dim` columns (unsigned, log1p of the counts, each row L2-normalised).

        block = HashedNgrams(dim=2048, prefix="text").fit(training_texts)
        block.names()            # ["text:refund", "text:invoice (+1)", "text:#2", ...]
        block.transform(texts)   # (len(texts), 2048) float32

    `ngrams=(1, 2)` means unigrams and bigrams of `lexical.tokens` (casefolded words; character bigrams for
    Japanese / CJK). The hash is blake2b with `seed`, the same in every process.

    `fit(texts, min_df=2, max_vocabulary=20000)` learns the vocabulary: the n-grams found in at least `min_df`
    training texts (the most frequent `max_vocabulary` of them). After it, only those n-grams count, and each
    column is named after the one found in most training texts among those that hash to it, with " (+k)" when k
    others share the column; a column with none is `<prefix>:#<index>` and stays zero. So a name in a decision's
    `reasons` is an n-gram the text really has, or one of the k that share its column - never an unseen one that
    happened to hash there. Without `fit`, every n-gram counts and the columns are `<prefix>:#<index>`.
    An empty text is an all-zero row."""

    VERSION = 1
    kind = "ngrams"

    def __init__(self, dim: int = 2048, ngrams: tuple[int, int] = (1, 2), prefix: str = "text", seed: int = 0):
        lo, hi = (int(ngrams[0]), int(ngrams[1])) if not isinstance(ngrams, int) else (1, int(ngrams))
        if int(dim) < 8 or not 1 <= lo <= hi <= 5:
            raise ValueError("HashedNgrams needs dim >= 8 and ngrams (lo, hi) with 1 <= lo <= hi <= 5")
        self.dim, self.ngrams, self.prefix, self.seed = int(dim), (lo, hi), str(prefix), int(seed)
        self.vocabulary: list[str] | None = None      # most frequent first; None until fit
        self._known: frozenset[str] | None = None
        self._labels: dict[int, str] = {}
        self._key = self.seed.to_bytes(8, "little", signed=True)
        self._slots: dict[str, int] = {}

    def _slot(self, gram: str) -> int:
        hit = self._slots.get(gram)
        if hit is None:
            d = hashlib.blake2b(gram.encode("utf-8"), digest_size=8, key=self._key).digest()
            hit = int.from_bytes(d, "little") % self.dim
            if len(self._slots) > 1_000_000:
                self._slots.clear()
            self._slots[gram] = hit
        return hit

    def fit(self, texts: list[str], min_df: int = 2, max_vocabulary: int = 20000) -> HashedNgrams:
        """Learn the vocabulary and the column names from training texts. Returns self."""
        df: Counter = Counter()
        for t in texts:
            df.update(_ngrams(t, *self.ngrams).keys())
        ranked = sorted((g for g, n in df.items() if n >= int(min_df)), key=lambda g: (-df[g], g))
        return self._set_vocabulary(ranked[:int(max_vocabulary)])

    def _set_vocabulary(self, vocabulary: list[str]) -> HashedNgrams:
        self.vocabulary = [str(g) for g in vocabulary]
        self._known = frozenset(self.vocabulary)
        first: dict[int, str] = {}
        shared: Counter = Counter()
        for gram in self.vocabulary:                  # most frequent first: the first one names the column
            slot = self._slot(gram)
            first.setdefault(slot, gram)
            shared[slot] += 1
        self._labels = {s: g + (f" (+{shared[s] - 1})" if shared[s] > 1 else "") for s, g in first.items()}
        return self

    def names(self) -> list[str]:
        return [f"{self.prefix}:{self._labels.get(i, f'#{i}')}" for i in range(self.dim)]

    def groups(self, group: str | None = None) -> dict[str, str]:
        """{column name: group} for the provider's `groups()` (default group: the prefix)."""
        return dict.fromkeys(self.names(), group or self.prefix)

    def transform(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        known = self._known
        for r, t in enumerate(texts):
            for gram, c in _ngrams(t, *self.ngrams).items():
                if known is None or gram in known:
                    out[r, self._slot(gram)] += np.log1p(c)
        return l2_normalize(out)

    @property
    def signature(self) -> str:
        return _signature(self.to_config())

    def to_config(self) -> dict:
        return {"kind": self.kind, "version": self.VERSION, "dim": self.dim, "ngrams": list(self.ngrams),
                "prefix": self.prefix, "seed": self.seed, "vocabulary": self.vocabulary}

    @classmethod
    def from_config(cls, config: dict) -> HashedNgrams:
        if int(config.get("version") or cls.VERSION) != cls.VERSION:
            raise ValueError(f"HashedNgrams config version {config.get('version')!r} is not {cls.VERSION}: it was "
                             f"written by another open-rlcd version; retrain")
        block = cls(dim=int(config.get("dim") or 2048), ngrams=tuple(config.get("ngrams") or (1, 2)),
                    prefix=str(config.get("prefix") or "text"), seed=int(config.get("seed") or 0))
        vocabulary = config.get("vocabulary")
        return block if vocabulary is None else block._set_vocabulary(list(vocabulary))


class TextEmbeddings:
    """Dense vectors of a frozen text encoder as feature columns `<prefix>_000 ...`.

        block = TextEmbeddings.for_project(project)      # pack_config.encoder, cached under <home>/cache
        block.transform(texts)                           # (len(texts), encoder.dim) float32, L2-normalised rows

    Pass any `Encoder` (`features.make_encoder(config, home)` gives a cached one) to use another."""

    kind = "embedding"

    def __init__(self, encoder: Encoder, prefix: str = "emb", text_kind: str = "passage"):
        self.encoder, self.prefix, self.text_kind = encoder, str(prefix), str(text_kind)

    @classmethod
    def for_project(cls, project: Any, prefix: str = "emb", default: dict | None = None) -> TextEmbeddings:
        """The encoder rlcd.yaml names in `pack_config.encoder` ({kind: hashing, dim: 384} offline, or
        {kind: onnx, model_id: ...} with the `documents` extra), cached under the project home. When the key is
        absent: `default`, else the offline hashing encoder (nothing is downloaded unless rlcd.yaml asks)."""
        cfg = (getattr(project, "pack_config", None) or {}).get("encoder") or default or {"kind": "hashing"}
        return cls(make_encoder(cfg, Path(project.home)), prefix=prefix)

    def names(self) -> list[str]:
        return [f"{self.prefix}_{i:03d}" for i in range(int(self.encoder.dim))]

    def groups(self, group: str | None = None) -> dict[str, str]:
        return dict.fromkeys(self.names(), group or self.prefix)

    def transform(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, int(self.encoder.dim)), dtype=np.float32)
        return np.asarray(self.encoder.encode([str(t) for t in texts], kind=self.text_kind), dtype=np.float32)

    @property
    def signature(self) -> str:
        return _signature([self.kind, self.prefix, self.text_kind, self.encoder.sha256])

    def to_config(self) -> dict:
        return {"kind": self.kind, "prefix": self.prefix, "text_kind": self.text_kind,
                "encoder": encoder_config(self.encoder)}

    @classmethod
    def from_config(cls, config: dict, home: str | Path) -> TextEmbeddings:
        """Rebuild from `to_config()`. The encoder's stored sha256 is checked: another model file is refused."""
        return cls(make_encoder(config.get("encoder") or {"kind": "hashing"}, Path(home)),
                   prefix=str(config.get("prefix") or "emb"), text_kind=str(config.get("text_kind") or "passage"))


def text_block(config: dict, home: str | Path | None = None) -> HashedNgrams | TextEmbeddings:
    """The block a `to_config()` describes: `{"kind": "ngrams", ...}` or `{"kind": "embedding", ...}` (which needs
    `home`, the project home, for the encoder files and the cache)."""
    kind = str((config or {}).get("kind") or "")
    if kind == HashedNgrams.kind:
        return HashedNgrams.from_config(config)
    if kind == TextEmbeddings.kind:
        if home is None:
            raise ValueError("an embedding block needs home= (the project home: encoder files and the cache)")
        return TextEmbeddings.from_config(config, home)
    raise ValueError(f"unknown text block kind {kind!r}; use 'ngrams' or 'embedding'")
