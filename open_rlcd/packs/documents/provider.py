"""The documents pack's feature provider: request states -> fixed, named feature rows for four views.

    query                  {query, candidates?, now?}
    query_chunk            {query, chunk, candidates?, now?}          (or the chunk's own fields at top level)
    query_evidence         {query, evidence, candidates?, now?}
    query_evidence_answer  {query, evidence, answer}

A candidate / chunk / evidence item is a retrieval hit as `features.retrieval` describes it: `{id, text?, title?,
score?, source?, sources?, rank?, doc_id?, updated_at?, authority?, lexical_score?, graph?}`; a plain string is
accepted as `{id: "e<i>", text: <string>}`. `now` (ISO 8601 or epoch seconds) is the reference time for document
age; without it age is measured relative to the newest candidate.

Feature blocks (names are fixed for a given encoder dimension `d`; groups in brackets):

  query                  emb_0..emb_{d-1} query embedding [embedding]; QUERY_FEATURES q_* [lexical];
                         RETRIEVAL_FEATURES r_* over the candidates [retrieval]
  query_chunk            qc_0..qc_{d-1} = query * chunk embedding, qc_cosine [embedding]; PAIR_FEATURES p_* (BM25
                         within the candidate set when one is given), qc_entity_coverage [lexical];
                         CHUNK_FEATURES c_* [retrieval]
  query_evidence         qe_0..qe_{d-1} = query * mean(evidence embeddings), e_cos_max / mean / min [embedding];
                         e_coverage, e_entity_coverage [lexical]; RETRIEVAL_FEATURES r_* over the evidence and
                         counts e_n_evidence, e_n_candidates, e_share_kept, e_log_chars [retrieval]
  query_evidence_answer  ae_0..ae_{d-1} = answer * mean(evidence embeddings), a_cos_query [embedding];
                         the answer-support block a_* [answer_support]; risk cues of the query q_cue_* / q_risk_*
                         [lexical]

Why these: the heads must generalise to queries and documents they never saw, so the features describe the
*relation* between query, evidence and answer (products of embeddings, coverage, support) rather than the
documents themselves. NaN means "unknown" (no text, no candidates), never a fake zero; the heads impute it.

Every transform encodes its texts in at most two encoder calls (one for queries, kind "query"; one for passages,
kind "passage", which covers chunks, answers and sentences), each text once. A small LRU cache keyed by the state
makes the repeated transforms of one decision (probabilities, explanations) free.

What this does not do: it does not retrieve anything (candidates come with the state), it does not check that
the evidence really belongs to the corpus, and its answer support is lexical / embedding similarity, not
entailment: a paraphrase that changes meaning without changing words or numbers can still look supported.
"""
from __future__ import annotations

import json
import math
import re
import threading
from collections import OrderedDict
from typing import Any

import numpy as np

from ...connectors.local_graph import entities, normalize_entity
from ...core.contract import MAX_CANDIDATES, ContractError
from ...features.lexical import (
    BM25,
    PAIR_FEATURES,
    QUERY_FEATURES,
    content_tokens,
    coverage,
    number_support,
    numbers,
    pair_features,
    query_features,
    tokens,
)
from ...features.retrieval import (
    CHUNK_FEATURES,
    RETRIEVAL_FEATURES,
    chunk_features,
    retrieval_features,
    to_epoch,
)
from ...features.text import Encoder, encoder_config

NAN = float("nan")
VIEWS = ("query", "query_chunk", "query_evidence", "query_evidence_answer")
FEATURE_SCHEMA = "documents-v1"
SUPPORT_COSINE = 0.8            # an answer sentence counts as "supported" by embedding above this cosine
RETRIEVAL_WINDOW = 10           # top-k window of the retrieval summary features
MAX_TEXT_CHARS = 100_000        # per text field: longer is refused (a request, not a corpus)
DEFAULT_CACHE = 1024

E_EXTRA = ["e_cos_max", "e_cos_mean", "e_cos_min", "e_coverage", "e_entity_coverage", "e_n_evidence",
           "e_n_candidates", "e_share_kept", "e_log_chars"]
A_SUPPORT = ["a_sent_cos_min", "a_sent_cos_mean", "a_sent_share_08", "a_lex_support_min", "a_lex_support_mean",
             "a_num_share", "a_num_unsupported", "a_n_numbers", "a_num_local_support", "a_date_support",
             "a_token_support", "a_entity_support", "a_query_coverage", "a_log_chars", "a_n_sentences",
             "a_n_evidence"]
RISK_FROM_LEXICAL = ["q_cue_complaint", "q_cue_legal", "q_cue_personal_data"]
RISK_CUES: dict[str, tuple[str, ...]] = {
    # substrings of the casefolded query; deliberately short lists of unambiguous words in en / ja / mn
    "q_risk_money": ("price", "cost", "refund", "fee", "discount", "invoice", "payment", "warranty", "usd", "$",
                     "価格", "料金", "返金", "値段", "円", "保証", "үнэ", "төлбөр", "буцаан", "төгрөг", "баталгаа"),
    "q_risk_safety": ("safety", "hazard", "forklift", "protective", "ppe", "fire", "evacuat", "chemical",
                      "安全", "危険", "フォークリフト", "保護具", "避難", "аюулгүй", "аюултай", "хамгаалалтын",
                      "түймэр"),
    "q_risk_medical": ("injur", "first aid", "medical", "hospital", "accident", "illness", "けが", "怪我", "救急",
                       "医療", "事故", "гэмтэл", "анхны тусламж", "эмнэлэг", "осол"),
}
RISK = [*RISK_FROM_LEXICAL, *RISK_CUES]
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+|(?<=[。！？])|\n+")
_CJK_CHARS = "\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\uac00-\ud7af"
_SCRIPT_GLUE = re.compile(rf"(?<=[{_CJK_CHARS}])(?=[A-Za-z0-9])|(?<=[A-Za-z0-9])(?=[{_CJK_CHARS}])")
_DATE = re.compile(r"(?<!\d)(\d{4})\s*[-/.年]\s*(\d{1,2})\s*[-/.月]\s*(\d{1,2})日?(?!\d)")


# ----------------------------------------------------------------------------- small helpers

def split_sentences(text: str) -> list[str]:
    """Sentences of `text`: split after . ! ? followed by whitespace (so "3.2 kg" stays whole), after 。！？, and
    at newlines. Empty pieces are dropped."""
    return [s.strip() for s in _SENT_SPLIT.split(str(text or "")) if s and s.strip()]


def passage_text(c: dict) -> str:
    """How a chunk is embedded: "title\\ntext" unless the text already opens with its title (the corpus does the
    same, so the embedding cache is shared)."""
    title, text = str(c.get("title") or ""), str(c.get("text") or "")
    if not text.strip():
        return ""
    return f"{title}\n{text}" if title and not text.lstrip().startswith(title) else text


def _ents(text: str) -> set[str]:
    # a space where Latin meets CJK: "チームはRobotics Labです" has no word boundary before "Robotics" otherwise, and
    # the graph's surface patterns would miss the name
    spaced = _SCRIPT_GLUE.sub(" ", str(text or ""))
    return {k for k in (normalize_entity(e) for e in entities(spaced)) if len(k) >= 2}


def dates(text: str) -> set[tuple[int, int, int]]:
    """Calendar dates written as 2024-03-01, 2024/3/1 or 2024年3月1日 (so a changed day is caught as a whole
    date, not hidden by the same number elsewhere)."""
    out = set()
    for m in _DATE.finditer(str(text or "")):
        y, mo, d = (int(x) for x in m.groups())
        if 1 <= mo <= 12 and 1 <= d <= 31:
            out.add((y, mo, d))
    return out


def entity_coverage(query: str, text: str) -> float:
    """Share of the entities named in `query` (names, codes, quoted terms) that `text` mentions; NaN when the query
    names none. Matching is on normalised surface forms, so it is cheap and language-agnostic, not NER."""
    q = _ents(query)
    if not q:
        return NAN
    low = normalize_entity(text)
    found = _ents(text)
    return sum(1 for e in q if e in found or e in low) / len(q)


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na < 1e-12 or nb < 1e-12:
        return NAN
    return float(a @ b / (na * nb))


def _risk_cues(query: str) -> dict[str, float]:
    q = str(query).casefold()
    return {k: float(any(c in q for c in cues)) for k, cues in RISK_CUES.items()}


class _Texts:
    """Collects the distinct texts of one transform call and embeds them in one call per kind."""

    def __init__(self) -> None:
        self.queries: dict[str, int] = {}
        self.passages: dict[str, int] = {}
        self.Q: np.ndarray | None = None
        self.P: np.ndarray | None = None

    def q(self, text: str) -> None:
        if text:
            self.queries.setdefault(text, len(self.queries))

    def p(self, text: str) -> None:
        if text:
            self.passages.setdefault(text, len(self.passages))

    def encode(self, enc: Encoder) -> None:
        if self.queries:
            self.Q = np.asarray(enc.encode(list(self.queries), kind="query"), np.float32)
        if self.passages:
            self.P = np.asarray(enc.encode(list(self.passages), kind="passage"), np.float32)

    def qv(self, text: str, dim: int) -> np.ndarray | None:
        return None if not text or self.Q is None else self.Q[self.queries[text]]

    def pv(self, text: str, dim: int) -> np.ndarray | None:
        return None if not text or self.P is None else self.P[self.passages[text]]


# ----------------------------------------------------------------------------- validation

def _bad(i: int, view: str, msg: str, **details: Any) -> ContractError:
    return ContractError(f"{view} state #{i}: {msg}", view=view, state_index=i, **details)


def _text_field(i: int, view: str, where: str, v: Any, required: bool) -> str:
    if v is None:
        if required:
            raise _bad(i, view, f"{where} is required (a non-empty string)", field=where)
        return ""
    if not isinstance(v, str):
        raise _bad(i, view, f"{where} must be a string, got {type(v).__name__}", field=where)
    if required and not v.strip():
        raise _bad(i, view, f"{where} must be a non-empty string", field=where)
    if len(v) > MAX_TEXT_CHARS:
        raise _bad(i, view, f"{where} is longer than {MAX_TEXT_CHARS} characters", field=where)
    return v


def _item(i: int, view: str, where: str, j: int, c: Any, prefix: str) -> dict:
    if isinstance(c, str):
        return {"id": f"{prefix}{j}", "text": _text_field(i, view, f"{where}[{j}]", c, False)}
    if not isinstance(c, dict):
        raise _bad(i, view, f"{where}[{j}] must be an object {{id, text, ...}} or a string, got {type(c).__name__}",
                   field=where)
    _text_field(i, view, f"{where}[{j}].text", c.get("text"), False)
    _text_field(i, view, f"{where}[{j}].title", c.get("title"), False)
    if c.get("id") is None:
        c = {**c, "id": f"{prefix}{j}"}
    return c


def _items(i: int, view: str, where: str, v: Any, required: bool, prefix: str) -> list[dict] | None:
    if v is None:
        if required:
            raise _bad(i, view, f"{where} is required: an array of retrieved chunks {{id, text, ...}}", field=where)
        return None
    if not isinstance(v, (list, tuple)):
        raise _bad(i, view, f"{where} must be an array of chunks, got {type(v).__name__}", field=where)
    if len(v) > MAX_CANDIDATES:
        raise _bad(i, view, f"{where} has {len(v)} items; at most {MAX_CANDIDATES}", field=where,
                   limit=MAX_CANDIDATES)
    return [_item(i, view, where, j, c, prefix) for j, c in enumerate(v)]


def _now(i: int, view: str, v: Any) -> float | None:
    if v is None or v == "":
        return None
    t = to_epoch(v)
    if t is None:
        raise _bad(i, view, f"now must be ISO 8601 or epoch seconds, got {v!r}", field="now")
    return t


def normalise_state(view: str, i: int, s: Any) -> dict:
    """A validated copy of one state with list items as dicts. Raises ContractError (422) naming the field."""
    if view not in VIEWS:
        raise ContractError(f"unknown view {view!r} for the documents provider; views: {list(VIEWS)}")
    if not isinstance(s, dict):
        raise _bad(i, view, f"must be an object like {describe(view)['example']}, got {type(s).__name__}")
    out: dict[str, Any] = {"query": _text_field(i, view, "query", s.get("query"), True)}
    out["now"] = _now(i, view, s.get("now"))
    out["candidates"] = _items(i, view, "candidates", s.get("candidates"), False, "c")
    if view == "query_chunk":
        chunk = s.get("chunk")
        if chunk is None:
            # the /v1/rank shape: shared {query} + each candidate's own fields merged into one state
            rest = {k: v for k, v in s.items() if k not in ("query", "candidates", "now", "chunk")}
            if "text" not in rest and "id" not in rest:
                raise _bad(i, view, "chunk is required: the retrieved chunk {id, text, ...} to judge (or give its "
                                    "fields, at least text, next to query)", field="chunk")
            chunk = rest
        out["chunk"] = _item(i, view, "chunk", 0, chunk, "chunk")
    elif view in ("query_evidence", "query_evidence_answer"):
        out["evidence"] = _items(i, view, "evidence", s.get("evidence"), True, "e")
        if view == "query_evidence_answer":
            out["answer"] = _text_field(i, view, "answer", s.get("answer"), True)
    return out


# ----------------------------------------------------------------------------- the provider

def describe(view: str) -> dict:
    """Input documentation of one view (served in /v1/heads)."""
    cand = "{id, text, title?, score?, source?, sources?, rank?, doc_id?, updated_at?, authority?, lexical_score?, " \
           "graph?: {path_length?, degree?}}"
    common = {"query": "string, the user's request (required)",
              "candidates": f"optional array of retrieved chunks {cand}; plain strings are accepted as {{text}}",
              "now": "optional reference time (ISO 8601 or epoch seconds) for document age"}
    docs = {
        "query": ({**common}, {"query": "What is the warranty of the Velobox 300?", "candidates": []}),
        "query_chunk": ({**common, "chunk": f"the retrieved chunk to judge {cand} (required; or its fields next "
                                            f"to query, as /v1/rank merges them)"},
                        {"query": "...", "chunk": {"id": "doc#0", "text": "..."}}),
        "query_evidence": ({**common, "evidence": "array of the chunks kept as evidence (required, may be empty)"},
                           {"query": "...", "evidence": [{"id": "doc#0", "text": "..."}]}),
        "query_evidence_answer": ({"query": common["query"],
                                   "evidence": "array of the chunks the answer was written from (required)",
                                   "answer": "string, the drafted answer to judge (required)"},
                                  {"query": "...", "evidence": [{"id": "doc#0", "text": "..."}], "answer": "..."}),
    }
    if view not in docs:
        raise ContractError(f"unknown view {view!r} for the documents provider; views: {list(VIEWS)}")
    fields, example = docs[view]
    return {"view": view, "state": fields, "example": json.dumps(example, ensure_ascii=False)}


class DocumentsProvider:
    """FeatureProvider of the documents pack. `encoder` is any `features.text.Encoder` (normally a CachedEncoder
    from `make_encoder`); `config` may set `cache_size` (LRU of featurised states, default 1024, 0 = off)."""

    def __init__(self, encoder: Encoder, config: dict | None = None):
        self.encoder = encoder
        self.config = dict(config or {})
        self.dim = int(encoder.dim)
        self.feature_version = f"{FEATURE_SCHEMA}:{encoder.id}:{encoder.sha256[:12]}"
        d = self.dim
        self._names = {
            "query": [*(f"emb_{i}" for i in range(d)), *QUERY_FEATURES, *RETRIEVAL_FEATURES],
            "query_chunk": [*(f"qc_{i}" for i in range(d)), "qc_cosine", *PAIR_FEATURES, "qc_entity_coverage",
                            *CHUNK_FEATURES],
            "query_evidence": [*(f"qe_{i}" for i in range(d)), *E_EXTRA, *RETRIEVAL_FEATURES],
            "query_evidence_answer": [*(f"ae_{i}" for i in range(d)), "a_cos_query", *A_SUPPORT, *RISK],
        }
        self._cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._cache_size = int(self.config.get("cache_size", DEFAULT_CACHE) or 0)
        self._lock = threading.Lock()

    def reserve_cache(self, n: int) -> None:
        """Grow the state cache to hold `n` rows (training: heads sharing a view are featurised once)."""
        with self._lock:
            self._cache_size = max(self._cache_size, int(n))

    # ------------------------------------------------------------------ protocol
    def names(self, view: str) -> list[str]:
        if view not in self._names:
            raise ContractError(f"unknown view {view!r} for the documents provider; views: {list(VIEWS)}")
        return list(self._names[view])

    def groups(self, view: str) -> dict[str, str]:
        out = {}
        for n in self.names(view):
            if re.fullmatch(r"(emb|qc|qe|ae)_\d+", n) or n in ("qc_cosine", "a_cos_query") or n.startswith("e_cos"):
                g = "embedding"
            elif n.startswith(("r_", "c_")) or n in ("e_n_evidence", "e_n_candidates", "e_share_kept", "e_log_chars"):
                g = "retrieval"
            elif n.startswith("a_"):
                g = "answer_support"
            else:
                g = "lexical"
            out[n] = g
        return out

    def describe(self, view: str) -> dict:
        return {**describe(view), "features": len(self.names(view)), "feature_version": self.feature_version}

    def to_config(self) -> dict:
        return {"encoder": encoder_config(self.encoder), "views": {v: len(n) for v, n in self._names.items()},
                "version": 1}

    def data_card(self) -> dict:
        return {"encoder": encoder_config(self.encoder),
                "features": {v: len(n) for v, n in self._names.items()}, "feature_version": self.feature_version}

    def transform(self, view: str, states: list[Any]) -> np.ndarray:
        names = self.names(view)
        out = np.full((len(states), len(names)), np.nan, np.float32)
        keys: list[str | None] = [None] * len(states)
        todo: list[int] = []
        for i, s in enumerate(states):
            if self._cache_size:
                try:
                    keys[i] = view + "\x00" + json.dumps(s, sort_keys=True, ensure_ascii=False, default=str)
                except (TypeError, ValueError):
                    keys[i] = None
            hit = None
            if keys[i] is not None:
                with self._lock:
                    hit = self._cache.get(keys[i])
                    if hit is not None:
                        self._cache.move_to_end(keys[i])
            if hit is not None:
                out[i] = hit
            else:
                todo.append(i)
        if todo:
            norm = [normalise_state(view, i, states[i]) for i in todo]
            rows = getattr(self, f"_{view}")(norm)
            for j, i in enumerate(todo):
                out[i] = rows[j]
                if keys[i] is not None:
                    with self._lock:
                        self._cache[keys[i]] = rows[j].copy()
                        while len(self._cache) > self._cache_size:
                            self._cache.popitem(last=False)
        return out

    # ------------------------------------------------------------------ views
    def _row(self, view: str, emb: np.ndarray | None, extra: dict[str, float]) -> np.ndarray:
        names = self._names[view]
        row = np.full(len(names), np.nan, np.float32)
        if emb is not None:
            row[: self.dim] = emb
        for j in range(self.dim, len(names)):
            v = extra.get(names[j], NAN)
            row[j] = NAN if v is None else v
        return row

    def _query(self, states: list[dict]) -> np.ndarray:
        tx = _Texts()
        for s in states:
            tx.q(s["query"])
        tx.encode(self.encoder)
        rows = []
        for s in states:
            f = {**query_features(s["query"]),
                 **retrieval_features(s["query"], s["candidates"] or [], now=s["now"], k=RETRIEVAL_WINDOW)}
            rows.append(self._row("query", tx.qv(s["query"], self.dim), f))
        return np.vstack(rows) if rows else np.zeros((0, len(self._names["query"])), np.float32)

    def _query_chunk(self, states: list[dict]) -> np.ndarray:
        tx = _Texts()
        for s in states:
            tx.q(s["query"])
            tx.p(passage_text(s["chunk"]))
        tx.encode(self.encoder)
        rows = []
        for s in states:
            q, c = s["query"], s["chunk"]
            qv, cv = tx.qv(q, self.dim), tx.pv(passage_text(c), self.dim)
            emb = qv * cv if qv is not None and cv is not None else None
            text = str(c.get("text") or "")
            f: dict[str, float] = {"qc_cosine": _cos(qv, cv) if emb is not None else NAN}
            if text.strip():
                bm25 = None
                cands = s["candidates"]
                if cands:
                    texts = [str(x.get("text") or "") for x in cands]
                    ids = [x.get("id") for x in cands]
                    if c.get("id") not in ids:
                        texts.append(text)
                        ids.append(c.get("id"))
                    scores = BM25([tokens(t) for t in texts]).scores(tokens(q))
                    bm25 = float(scores[ids.index(c.get("id"))])
                f.update(pair_features(q, text, bm25))
                f["qc_entity_coverage"] = entity_coverage(q, f"{c.get('title') or ''}\n{text}")
            f.update(chunk_features(q, c, s["candidates"], now=s["now"]))
            rows.append(self._row("query_chunk", emb, f))
        return np.vstack(rows) if rows else np.zeros((0, len(self._names["query_chunk"])), np.float32)

    def _evidence_matrix(self, tx: _Texts, evidence: list[dict]) -> np.ndarray | None:
        vecs = [tx.pv(passage_text(e), self.dim) for e in evidence]
        vecs = [v for v in vecs if v is not None]
        return np.vstack(vecs) if vecs else None

    def _query_evidence(self, states: list[dict]) -> np.ndarray:
        tx = _Texts()
        for s in states:
            tx.q(s["query"])
            for e in s["evidence"]:
                tx.p(passage_text(e))
        tx.encode(self.encoder)
        rows = []
        for s in states:
            q, ev = s["query"], s["evidence"]
            qv = tx.qv(q, self.dim)
            E = self._evidence_matrix(tx, ev)
            emb = qv * E.mean(axis=0) if E is not None and qv is not None else None
            f: dict[str, float] = {}
            if E is not None and qv is not None:
                cos = [_cos(qv, e) for e in E]
                cos = [x for x in cos if not math.isnan(x)]
                if cos:
                    f.update(e_cos_max=max(cos), e_cos_mean=float(np.mean(cos)), e_cos_min=min(cos))
            joined = "\n".join(f"{e.get('title') or ''}\n{e.get('text') or ''}" for e in ev)
            f["e_coverage"] = coverage(q, joined) if joined.strip() else 0.0
            f["e_entity_coverage"] = entity_coverage(q, joined)     # 0 for no evidence, NaN when q names nothing
            f["e_n_evidence"] = float(len(ev))
            cands = s["candidates"]
            if cands is not None:
                f["e_n_candidates"] = float(len(cands))
                f["e_share_kept"] = len(ev) / len(cands) if cands else NAN
            f["e_log_chars"] = math.log1p(sum(len(str(e.get("text") or "")) for e in ev))
            f.update(retrieval_features(q, ev, now=s["now"], k=RETRIEVAL_WINDOW))
            rows.append(self._row("query_evidence", emb, f))
        return np.vstack(rows) if rows else np.zeros((0, len(self._names["query_evidence"])), np.float32)

    def _query_evidence_answer(self, states: list[dict]) -> np.ndarray:
        tx = _Texts()
        split: list[tuple[list[str], list[str]]] = []
        for s in states:
            tx.q(s["query"])
            tx.p(s["answer"])
            for e in s["evidence"]:
                tx.p(passage_text(e))
            a_sents = split_sentences(s["answer"])
            e_sents = [x for e in s["evidence"] for x in split_sentences(str(e.get("text") or ""))]
            for x in (*a_sents, *e_sents):
                tx.p(x)
            split.append((a_sents, e_sents))
        tx.encode(self.encoder)
        rows = []
        for s, (a_sents, e_sents) in zip(states, split, strict=True):
            q, ans, ev = s["query"], s["answer"], s["evidence"]
            qv, av = tx.qv(q, self.dim), tx.pv(ans, self.dim)
            E = self._evidence_matrix(tx, ev)
            emb = av * E.mean(axis=0) if E is not None and av is not None else None
            f: dict[str, float] = {"a_cos_query": _cos(qv, av) if qv is not None and av is not None else NAN}
            f.update(self._support(tx, ans, ev, a_sents, e_sents))
            f["a_query_coverage"] = coverage(q, ans)
            f["a_log_chars"] = math.log1p(len(ans))
            f["a_n_sentences"] = float(len(a_sents))
            f["a_n_evidence"] = float(len(ev))
            qf = query_features(q)
            f.update({k: qf[k] for k in RISK_FROM_LEXICAL})
            f.update(_risk_cues(q))
            rows.append(self._row("query_evidence_answer", emb, f))
        return np.vstack(rows) if rows else np.zeros((0, len(self._names["query_evidence_answer"])), np.float32)

    def _support(self, tx: _Texts, ans: str, ev: list[dict], a_sents: list[str], e_sents: list[str]) -> dict:
        """The answer-support block: how much of the answer the evidence backs, sentence by sentence."""
        f: dict[str, float] = {}
        joined = "\n".join(f"{e.get('title') or ''}\n{e.get('text') or ''}" for e in ev)
        if a_sents and e_sents:
            A = np.vstack([tx.pv(x, self.dim) for x in a_sents])
            S = np.vstack([tx.pv(x, self.dim) for x in e_sents])
            an = np.linalg.norm(A, axis=1, keepdims=True)
            sn = np.linalg.norm(S, axis=1, keepdims=True)
            C = (A / np.where(an > 1e-12, an, 1.0)) @ (S / np.where(sn > 1e-12, sn, 1.0)).T
            best = C.max(axis=1)
            f.update(a_sent_cos_min=float(best.min()), a_sent_cos_mean=float(best.mean()),
                     a_sent_share_08=float((best > SUPPORT_COSINE).mean()))
            e_tok = [set(tokens(x)) for x in e_sents]
            lex = []
            for x in a_sents:
                ct = set(content_tokens(x)) or set(tokens(x))
                if ct:
                    lex.append(max(len(ct & et) / len(ct) for et in e_tok))
            if lex:
                f.update(a_lex_support_min=float(min(lex)), a_lex_support_mean=float(np.mean(lex)))
        share, unsupported = number_support(ans, joined)
        f.update(a_num_share=share, a_num_unsupported=float(unsupported), a_n_numbers=float(len(numbers(ans))))
        # numbers must be backed by the chunk the sentence draws on, not by a coincidence elsewhere in the evidence
        texts = [f"{e.get('title') or ''}\n{e.get('text') or ''}" for e in ev]
        if texts:
            toks = [set(tokens(t)) for t in texts]
            local = []
            for x in a_sents:
                if not numbers(x):
                    continue
                st = set(tokens(x))
                best = max(range(len(texts)), key=lambda j: (len(st & toks[j]), -j))
                local.append(number_support(x, texts[best])[0])
            if local:
                f["a_num_local_support"] = float(min(local))
        ad = dates(ans)
        if ad:
            f["a_date_support"] = len(ad & dates(joined)) / len(ad)
        at = set(content_tokens(ans))
        if at:
            jt = set(tokens(joined))
            f["a_token_support"] = len(at & jt) / len(at)
        ae = _ents(ans)
        if ae:
            low = normalize_entity(joined)
            je = _ents(joined)
            f["a_entity_support"] = sum(1 for e in ae if e in je or e in low) / len(ae)
        return f
