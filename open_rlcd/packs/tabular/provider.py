"""TabularProvider: a row `{column: value}` -> one float32 feature row, identically at train and serve time.

    numeric column      -> one feature `<col>`; NaN when missing, non-finite or unparseable (LightGBM routes
                           NaN natively; logistic regression fills it with the training mean)
    categorical column  -> one-hot over a vocabulary fixed when the provider is learned: the `max_categories`
                           most frequent values (stored sorted by value, so new rows that only shift
                           frequencies keep the same schema), plus `<col>=__other__` (a value outside the
                           vocabulary, including one never seen in training) and `<col>=__missing__` (null /
                           NA). A value that differs from a vocabulary entry only in letter case maps to it
                           when that is unambiguous.
    text column         -> `text_buckets` features `text:<col>#<i>` (pack_config.text): the signed feature hashing of
                           `open_rlcd.features.text.HashingEncoder` (word unigrams and character 2-4 grams within
                           words after NFKC + casefold, log1p-dampened counts, L2-normalised; blake2b, so the same
                           text gives the same row in every process and on every machine). Stateless: no
                           vocabulary is learned or stored, so nothing of the training text travels in a bundle,
                           and because the signal is in character n-grams it does not depend on spaces between
                           words (English, Japanese, Mongolian). A missing text is the zero vector. What it is not:
                           it matches shared words and character runs, not meaning, and 256 buckets collide, so
                           a single bucket in `reasons` stands for several n-grams.

The schema (column order, kinds, vocabularies, NA tokens) is the whole state of the provider. It lives in
`to_config()`, which the bundle stores, so a served model rebuilds exactly the transform it was trained with;
`feature_version` is "tabular-v1:" + the first 12 hex characters of the schema's SHA-256, so a request with
precomputed features of another schema is refused with a 422. A text column is part of the schema (its bucket
count and the encoder's own config), so a model trained without it has another feature_version than one trained
with it; schemas without text columns hash exactly as they did before text columns existed.

Hashed vocabularies (`pack_config.hash_categories: true`): each vocabulary entry is stored as
"h:" + the first 16 hex characters of SHA-256(salt, 0x1f, value), plus the same hash of its lower-cased form for
the case-insensitive match, so neither the bundle (model.json, the heads' feature names) nor /v1/heads shows a raw
value; the transform hashes each incoming value the same way and behaves exactly as with raw values. The salt is
random (16 bytes), kept in the bundle next to the hashes, and the tabular pack keeps one per project home
(<home>/tabular/category_salt), so retraining the same table keeps the same schema and untouched heads are carried
over. What hashing does not do: anyone holding the bundle can still test a guessed value (hash it with the stored
salt), so small value sets (country, plan) stay recoverable by trying them all; it stops a reader from simply
reading names, emails or phone numbers out of a shared bundle. Raw mode (the default) is unchanged and keeps the
feature_version of bundles trained before the option existed.

What it does not do: no scaling (the base models handle that), no interaction or date features, no target
encoding (it never looks at labels, so the vocabulary learned from every row of the file does not leak
outcomes), and no missing-value indicator for numeric columns. A request must carry every feature column as a
key; `null` is an explicit "missing" and is accepted. Raw vocabularies are values of the training data and
travel with every bundle (`rlcd export` / `push` warn about them): for a categorical column that could hold
personal data, set `hash_categories: true`, or better, leave the column out of the features.
"""
from __future__ import annotations

import hashlib
import json
import re
import secrets
from collections import Counter
from dataclasses import dataclass
from typing import Any

import numpy as np

from ...core.contract import ContractError
from ...features.text import HashingEncoder
from .data import DEFAULT_TEXT_BUCKETS, Table, TabularConfig, TabularConfigError, canon, to_float

VIEW = "row"
KIND = "tabular"
SCHEMA_VERSION = 1
OTHER, MISSING = "__other__", "__missing__"
ID_LIKE_MIN_ROWS = 100
LONG_STRING_CHARS = 12        # median length from which mostly-unique categorical values read as free text / ids
RARE_OTHER_SHARE = 0.05       # an unseen category is worth a warning when __other__ was this rare in training
TEXT_PREFIX = "text:"
_SALT = re.compile(r"[0-9a-f]{32}")


def new_salt() -> str:
    """A fresh random salt for hashed vocabularies (16 bytes, hex)."""
    return secrets.token_hex(16)


def category_hash(salt: str, value: str) -> str:
    """How a categorical value is stored when vocabularies are hashed: "h:" + 16 hex characters (64 bits)."""
    return "h:" + hashlib.sha256(f"{salt}\x1f{value}".encode()).hexdigest()[:16]


@dataclass(frozen=True)
class Column:
    name: str
    kind: str                             # numeric | categorical | text
    vocabulary: tuple[str, ...] = ()      # categorical: the most frequent values sorted by value (raw), or their
    #                                       salted hashes sorted by hash (hashed)
    hashed: bool = False                  # vocabulary holds category_hash(salt, value), not values
    folded: tuple[str, ...] = ()          # hashed: category_hash(salt, value.lower()) per vocabulary entry
    buckets: int = 0                      # text: number of hashed features

    def features(self) -> list[str]:
        if self.kind == "numeric":
            return [self.name]
        if self.kind == "text":
            return [f"{TEXT_PREFIX}{self.name}#{i}" for i in range(self.buckets)]
        return [f"{self.name}={v}" for v in (*self.vocabulary, OTHER, MISSING)]

    def to_dict(self) -> dict:
        d: dict[str, Any] = {"name": self.name, "kind": self.kind}
        if self.kind == "categorical":
            d["vocabulary"] = list(self.vocabulary)
            if self.hashed:          # raw columns keep their old dict, so old bundles keep their feature_version
                d["hashed"] = True
                d["vocabulary_folded"] = list(self.folded)
        if self.kind == "text":      # the encoder's config is in the schema: another hashing is another version
            d["buckets"] = self.buckets
            d["encoder"] = text_encoder(self.buckets).config()
        return d


def text_encoder(buckets: int) -> HashingEncoder:
    """The stateless encoder of a text column: HashingEncoder with `buckets` dimensions and the fixed seed 0."""
    return HashingEncoder(dim=int(buckets), seed=0)


def _text(v: Any, na: frozenset[str]) -> str:
    """A text cell as the string that is hashed; "" (the zero vector) when missing."""
    if isinstance(v, (list, tuple)):
        v = " ".join(str(x) for x in v if x is not None)
    return canon(v, na) or ""


def _long_strings(present: list[str], max_categories: int) -> tuple[int, int] | None:
    """(distinct, median length) when a column's non-missing values are mostly unique long strings: at least
    ID_LIKE_MIN_ROWS values, more distinct ones than the vocabulary holds, at least half of them distinct, and
    a median length of LONG_STRING_CHARS or more. None otherwise (short codes and real categories)."""
    if len(present) < ID_LIKE_MIN_ROWS:
        return None
    distinct = len(set(present))
    if distinct <= max_categories or distinct < 0.5 * len(present):
        return None
    med = int(np.median([len(v) for v in present]))
    return (distinct, med) if med >= LONG_STRING_CHARS else None


def _numeric(values: list[Any], na: frozenset[str]) -> bool:
    """Every non-missing value parses as a finite number (and there is at least one)."""
    seen = False
    for v in values:
        if canon(v, na) is None:
            continue
        if isinstance(v, bool) or np.isnan(to_float(v, na)):
            return False
        seen = True
    return seen


class TabularProvider:
    """FeatureProvider for the tabular pack. Build it with `learn` (from the training table) or `from_config`
    (from a bundle); the constructor takes an already-fixed schema."""

    def __init__(self, columns: list[Column], na_values: frozenset[str], source: dict | None = None,
                 notes: list[str] | None = None, salt: str | None = None):
        if not columns:
            raise TabularConfigError("the tabular provider has no feature columns: list some under "
                                     "pack_config.features")
        names = [c.name for c in columns]
        if len(set(names)) != len(names):
            raise TabularConfigError(f"duplicate feature columns: {names}")
        hashed = [c for c in columns if c.kind == "categorical" and c.hashed]
        if hashed and not (isinstance(salt, str) and _SALT.fullmatch(salt)):
            raise TabularConfigError("hashed categorical vocabularies need their salt (32 hex characters)")
        for c in hashed:
            if len(c.folded) != len(c.vocabulary) or len(set(c.vocabulary)) != len(c.vocabulary):
                raise TabularConfigError(f"column {c.name!r}: hashed vocabulary is malformed")
        self.columns = list(columns)
        self.na_values = frozenset(na_values)
        self.source = dict(source or {})
        self.notes = list(notes or [])
        self.salt = salt if hashed else None
        self._names = [f for c in self.columns for f in c.features()]
        self._groups = {f: c.name for c in self.columns for f in c.features()}
        # per column: (offset of its first feature, kind, value -> position within its block)
        self._layout: list[tuple[Column, int, dict[str, int], dict[str, int]]] = []
        self._encoders = {c.name: text_encoder(c.buckets) for c in self.columns if c.kind == "text"}
        off = 0
        for c in self.columns:
            pos = {v: i for i, v in enumerate(c.vocabulary)}
            lower: dict[str, int] = {}
            # case-insensitive keys: the lower-cased value (raw) or the hash of it (hashed)
            folds = c.folded if c.hashed else tuple(v.lower() for v in c.vocabulary)
            for i, f in enumerate(folds):
                lower[f] = -1 if f in lower else i   # -1: ambiguous ignoring case
            self._layout.append((c, off, pos, lower))
            off += len(c.features())
        self.feature_version = f"tabular-v1:{hashlib.sha256(self._schema_json().encode()).hexdigest()[:12]}"

    # ------------------------------------------------------------------ construction
    @classmethod
    def learn(cls, table: Table, cfg: TabularConfig, exclude: set[str], salt: str | None = None) -> TabularProvider:
        """Fix the schema from the training table: which columns are features, which are categorical, and each
        categorical column's vocabulary. `exclude`: label columns of every target (never features). With
        `cfg.hash_categories`, vocabularies are stored hashed with `salt` (a fresh random one when None)."""
        na = cfg.na_values
        if cfg.hash_categories and salt is None:
            salt = new_salt()
        reserved = set(exclude) | set(cfg.meta_columns)
        notes: list[str] = []
        table.require(cfg.meta_columns, "pack_config id / group / time / segment")
        # a misspelt column under `exclude` would silently stay a feature: it must exist
        table.require(list(cfg.exclude), "pack_config.exclude")
        table.require(list(cfg.text), "pack_config.text")
        clash = sorted(set(cfg.text) & reserved)
        if clash:
            raise TabularConfigError(f"pack_config.text lists {clash}, which are label / id / group / time / segment "
                                     f"columns")
        if cfg.features is not None:
            table.require(list(cfg.features), "pack_config.features")
            clash = sorted(set(cfg.features) & reserved)
            if clash:
                raise TabularConfigError(f"pack_config.features lists {clash}, which are label / id / group / time / "
                                         f"segment columns: a label used as a feature leaks the answer")
            clash = sorted(set(cfg.features) & set(cfg.exclude))
            if clash:
                raise TabularConfigError(f"pack_config.features and pack_config.exclude both list {clash}")
            feats = [c for c in cfg.features if c not in cfg.text]
        else:
            feats = []
            for c in table.columns:
                if c in reserved or c in cfg.exclude or c in cfg.text:
                    continue
                vals = [canon(v, na) for v in table.data[c]]
                present = [v for v in vals if v is not None]
                if not present:
                    notes.append(f"column {c!r} is empty: left out of the auto features")
                    continue
                distinct = len(set(present))
                numeric = c not in (cfg.categorical or ()) and _numeric(table.data[c], na)
                # (a column the config lists under `categorical` was asked for: it stays, with the note below)
                if (not numeric and c not in (cfg.categorical or ()) and len(present) >= ID_LIKE_MIN_ROWS
                        and distinct > cfg.max_categories and distinct >= 0.9 * len(present)):
                    notes.append(f"column {c!r} looks like an identifier or free text ({distinct} distinct values in "
                                 f"{len(present)} rows): left out of the auto features. Free text belongs under "
                                 f"pack_config.text (hashed n-gram features), not under pack_config.features "
                                 f"(that one-hot encodes whole strings)")
                    continue
                feats.append(c)
        if cfg.categorical is not None:
            table.require(list(cfg.categorical), "pack_config.categorical")
            stray = sorted(set(cfg.categorical) - set(feats) - reserved)
            if stray:
                raise TabularConfigError(f"pack_config.categorical lists {stray}, which are not feature columns")
        columns = []
        other_rows: dict[str, int] = {}     # per categorical column: training rows encoded as __other__
        for c in feats:
            # a listed column is categorical even when its values are numbers (zip codes, plan ids); a column
            # whose values are not all numbers is categorical either way (as a number it would be all NaN)
            cat = c in (cfg.categorical or ()) or not _numeric(table.data[c], na)
            if not cat:
                columns.append(Column(c, "numeric"))
                bad = sum(1 for v in table.data[c] if canon(v, na) is not None and np.isnan(to_float(v, na)))
                if bad:
                    notes.append(f"numeric column {c!r}: {bad} values are not numbers and are read as missing")
                continue
            present = [v for v in (canon(x, na) for x in table.data[c]) if v is not None]
            counts = Counter(present)
            ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
            vocab = tuple(sorted(v for v, _ in ranked[:cfg.max_categories]))
            rare = sum(n for _, n in ranked[cfg.max_categories:])
            other_rows[c] = rare
            if len(ranked) > cfg.max_categories:
                notes.append(f"categorical column {c!r}: {len(ranked)} distinct values; the {cfg.max_categories} most "
                             f"frequent are one-hot encoded, {rare} rows of rarer values go to {OTHER}")
            long = _long_strings(present, cfg.max_categories)
            if long is not None:
                stored = ("Its values are stored only as salted hashes (hash_categories: true)" if cfg.hash_categories
                          else f"With hash_categories: false its {len(vocab)} most frequent values are stored word "
                               f"for word in every bundle, readable by anyone who gets the bundle")
                notes.append(f"categorical column {c!r} holds mostly unique long strings ({long[0]} distinct values "
                             f"in {len(present)} rows, typical length {long[1]}): use pack_config.text (hashed "
                             f"n-gram features, nothing of the text is stored) if it is free text, or "
                             f"pack_config.exclude; one-hot encoding whole strings carries almost no signal. "
                             f"{stored}")
            if cfg.hash_categories:
                pairs = sorted((category_hash(salt, v), category_hash(salt, v.lower())) for v in vocab)
                columns.append(Column(c, "categorical", tuple(h for h, _ in pairs), hashed=True,
                                      folded=tuple(f for _, f in pairs)))
            else:
                columns.append(Column(c, "categorical", vocab))
        cats = [col.name for col in columns if col.kind == "categorical"]
        if cats and cfg.hash_categories:
            notes.append(f"categorical columns {cats}: values are stored as salted SHA-256 hashes "
                         f"(pack_config.hash_categories); the salt is in the bundle, so a guessed value can still be "
                         f"checked")
        elif cats:
            notes.append(f"categorical columns {cats}: their most frequent raw values are stored in every bundle; set "
                         f"pack_config.hash_categories: true to store salted hashes instead")
        for c in cfg.text:
            columns.append(Column(c, "text", buckets=cfg.text_buckets))
            empty = sum(1 for v in table.data[c] if not _text(v, na))
            if empty == table.n:
                notes.append(f"text column {c!r} is empty in every row: its features are all zero")
        if cfg.text:
            notes.append(f"text columns {list(cfg.text)}: {cfg.text_buckets} hashed word / character n-gram features "
                         f"each; no vocabulary is stored, so no text from the file is in the bundle")
        source = {"file": table.path.name, "rows": table.n, "learned_from": "every row of the file (no labels)",
                  "other_rows": other_rows}
        return cls(columns, na, source, notes, salt=salt if cfg.hash_categories else None)

    @classmethod
    def from_config(cls, config: dict) -> TabularProvider:
        """Rebuild the provider a bundle was trained with. The stored feature_version must match the rebuilt
        schema's hash: a hand-edited config cannot pass for the trained one."""
        if not isinstance(config, dict) or config.get("kind") != KIND:
            raise TabularConfigError(f"not a tabular provider config (kind {config.get('kind')!r})"
                                     if isinstance(config, dict) else "tabular provider config must be a mapping")
        if int(config.get("schema_version", 0)) != SCHEMA_VERSION:
            raise TabularConfigError(f"tabular provider schema_version {config.get('schema_version')!r} is not "
                                     f"supported (this build reads {SCHEMA_VERSION})")
        cols = []
        for d in config.get("columns") or []:
            if d.get("kind") == "text":
                buckets = d.get("buckets", DEFAULT_TEXT_BUCKETS)
                if not isinstance(buckets, int) or isinstance(buckets, bool) or buckets < 8:
                    raise TabularConfigError(f"text column {d.get('name')!r}: bad bucket count {buckets!r}")
                have = text_encoder(buckets).config()
                if d.get("encoder") != have:
                    raise TabularConfigError(
                        f"text column {d.get('name')!r} was hashed with encoder {d.get('encoder')!r}; this build "
                        f"hashes with {have!r}, which gives other features: retrain, or use the open-rlcd version "
                        f"the bundle was trained with")
                cols.append(Column(str(d["name"]), "text", buckets=buckets))
                continue
            if d.get("kind") not in ("numeric", "categorical"):
                raise TabularConfigError(f"column {d.get('name')!r}: unknown kind {d.get('kind')!r}")
            hashed = d.get("kind") == "categorical" and d.get("hashed") is True
            cols.append(Column(str(d["name"]), d["kind"], tuple(str(v) for v in d.get("vocabulary") or ()),
                               hashed=hashed,
                               folded=tuple(str(v) for v in d.get("vocabulary_folded") or ()) if hashed else ()))
        p = cls(cols, frozenset(config.get("na_values") or ()), config.get("source"), config.get("notes"),
                salt=config.get("category_salt"))
        want = config.get("feature_version")
        if want is not None and want != p.feature_version:
            raise TabularConfigError(f"tabular provider config says feature_version {want!r} but its schema hashes "
                                     f"to {p.feature_version!r}: the config was modified")
        return p

    def _schema(self) -> dict:
        out: dict[str, Any] = {"columns": [c.to_dict() for c in self.columns], "na_values": sorted(self.na_values)}
        if self.salt is not None:    # only with hashed vocabularies: raw schemas hash exactly as before
            out["category_salt"] = self.salt
        return out

    @property
    def hashed(self) -> bool:
        """True when categorical vocabularies are stored as salted hashes."""
        return self.salt is not None

    def _schema_json(self) -> str:
        return json.dumps(self._schema(), sort_keys=True, separators=(",", ":"), ensure_ascii=False)

    def to_config(self) -> dict:
        return {"kind": KIND, "schema_version": SCHEMA_VERSION, **self._schema(),
                "feature_version": self.feature_version, "source": self.source, "notes": self.notes}

    def data_card(self) -> dict:
        """Extra data-card fields (picked up by the trainer)."""
        return {"tabular": {"source": self.source, "numeric_columns": [c.name for c in self.columns
                                                                       if c.kind == "numeric"],
                            "categorical_columns": {c.name: len(c.vocabulary) for c in self.columns
                                                    if c.kind == "categorical"},
                            "categorical_values": "salted sha256 hashes" if self.hashed else "raw values",
                            **({"text_columns": {c.name: c.buckets for c in self.columns if c.kind == "text"}}
                               if self._encoders else {}),
                            "features": len(self._names), "notes": self.notes}}

    # ------------------------------------------------------------------ FeatureProvider protocol
    def _view(self, view: str) -> None:
        if view != VIEW:
            raise ContractError(f"the tabular pack has one feature view, {VIEW!r}; got {view!r}")

    def names(self, view: str) -> list[str]:
        self._view(view)
        return list(self._names)

    def groups(self, view: str) -> dict[str, str]:
        self._view(view)
        return dict(self._groups)

    def required(self) -> list[str]:
        return [c.name for c in self.columns]

    def describe(self, view: str) -> dict:
        self._view(view)
        cols = {}
        for c in self.columns:
            if c.kind == "numeric":
                cols[c.name] = {"type": "number", "missing": "null"}
            elif c.kind == "text":
                cols[c.name] = {"type": "text", "features": c.buckets, "missing": "null (all features zero)",
                                "encoding": "hashed word and character n-grams; no vocabulary is stored"}
            elif c.hashed:   # the values are not in the bundle: nothing to list
                cols[c.name] = {"type": "category", "categories": [], "more_categories": len(c.vocabulary),
                                "hashed": True, "unseen": f"encoded as {OTHER}",
                                "missing": f"null (encoded as {MISSING})"}
            else:
                shown = list(c.vocabulary[:25])
                cols[c.name] = {"type": "category", "categories": shown,
                                "more_categories": max(0, len(c.vocabulary) - len(shown)),
                                "unseen": f"encoded as {OTHER}", "missing": f"null (encoded as {MISSING})"}
        return {"state": "an object {column: value}: one row of the training table's feature columns",
                "required": self.required(), "columns": cols,
                "notes": ["every listed column must be present (use null for a missing value); other keys are "
                          "ignored", "numbers may be sent as JSON numbers or numeric strings"]}

    def transform(self, view: str, states: list[Any]) -> np.ndarray:
        """Feature rows for `states` (each a dict {column: value}); ContractError (422) on a malformed state."""
        self._view(view)
        na = self.na_values
        X = np.zeros((len(states), len(self._names)), np.float32)
        memo: dict[tuple[str, str], int] = {}     # hashed columns: (column, value) -> position, per call
        texts: dict[str, list[str]] = {name: [] for name in self._encoders}
        for r, s in enumerate(states):
            if not isinstance(s, dict):
                raise ContractError("tabular heads need state = an object {column: value} (one table row)",
                                    expected_columns=self.required())
            missing = [c.name for c in self.columns if c.name not in s]
            if missing:
                raise ContractError(f"state is missing column(s) {missing}; send every feature column (null for a "
                                    f"missing value)", missing_columns=missing, expected_columns=self.required())
            for col, off, pos, lower in self._layout:
                v = s[col.name]
                if col.kind == "numeric":
                    X[r, off] = to_float(v, na)
                    continue
                if col.kind == "text":
                    texts[col.name].append(_text(v, na))
                    continue
                key = canon(v, na)
                if key is None:
                    j = len(col.vocabulary) + 1
                elif col.hashed:
                    j = memo.get((col.name, key), -2)
                    if j == -2:
                        j = pos.get(category_hash(self.salt, key), -1)
                        if j < 0:
                            j = lower.get(category_hash(self.salt, key.lower()), -1)
                        if j < 0:
                            j = len(col.vocabulary)
                        memo[(col.name, key)] = j
                else:
                    j = pos.get(key, -1)
                    if j < 0:
                        j = lower.get(key.lower(), -1)
                    if j < 0:
                        j = len(col.vocabulary)
                X[r, off + j] = 1.0
        for col, off, _pos, _lower in self._layout:
            if col.kind == "text" and states:
                X[:, off:off + col.buckets] = self._encoders[col.name].encode(texts[col.name])
        return X

    def input_warnings(self, view: str, states: list[Any]) -> list[str]:
        """What a caller should know about how `states` were read, beyond what the feature rows show: a value in
        a numeric column that is not a number (it is read as missing, like null), and a categorical value the
        training vocabulary does not hold (it is encoded as __other__) when __other__ was rare or absent in
        training, so the model has little or no evidence about it. One message per column; never an error, and
        states `transform` would refuse are skipped. Values of hashed columns are not repeated in the message."""
        self._view(view)
        na = self.na_values
        rows = (self.source or {}).get("rows")
        seen_other = (self.source or {}).get("other_rows")
        out: list[str] = []
        for col, _off, pos, lower in self._layout:
            if col.kind == "text":
                continue
            bad: list[str] = []
            for s in states:
                if not isinstance(s, dict) or col.name not in s:
                    continue
                v = s[col.name]
                key = canon(v, na)
                if key is None:
                    continue
                if col.kind == "numeric":
                    if isinstance(v, bool) or not np.isnan(to_float(v, na)):
                        continue
                    try:
                        nonfinite = not np.isfinite(float(str(v).strip()))   # inf / 1e999: a number, but unusable
                    except ValueError:
                        nonfinite = False
                    bad.append("a non-finite number" if nonfinite else repr(key[:40]))
                elif col.hashed:
                    if (category_hash(self.salt, key) not in pos
                            and lower.get(category_hash(self.salt, key.lower()), -1) < 0):
                        bad.append("(not shown: hashed column)")
                elif key not in pos and lower.get(key.lower(), -1) < 0:
                    bad.append(repr(key[:40]))
            if not bad:
                continue
            shown = ", ".join(dict.fromkeys(bad[:3]))
            if col.kind == "numeric":
                out.append(f"column {col.name!r}: {shown} is not a number, so the value was read as missing (the "
                           f"same as null). Send a JSON number or a numeric string")
                continue
            n_other = seen_other.get(col.name) if isinstance(seen_other, dict) else None
            if n_other is None:
                how = f"it is encoded as {OTHER}"
            elif not n_other:
                how = (f"it is encoded as {OTHER}, which no training row had: the model has no evidence about it "
                       f"and answers as if the column said nothing")
            elif rows and n_other / rows < RARE_OTHER_SHARE:
                how = (f"it is encoded as {OTHER}, like only {n_other} of the {rows} training rows: the model has "
                       f"little evidence about it")
            else:
                continue      # __other__ is an ordinary, well-represented value of this column
            out.append(f"column {col.name!r}: value {shown} was not among the {len(col.vocabulary)} values seen in "
                       f"training; {how}. Check the spelling, or retrain once rows with it have outcomes")
        return out
