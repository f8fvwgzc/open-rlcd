"""Label tasks for a real deployment: turn a query log into items a human or an LLM judge can label.

    pack_config.queries: ./queries.jsonl      (or .csv with a `query` column, or a .json array)

A JSONL row: `{"query": "...", "time": ISO | epoch, "group": "...", "answer": "...", "evidence": [...]}`; only
`query` is required. `time` is also each state's `now`, the reference time of the document-age features; log it:
/v1/decide measures age from the time of the request, and a row without `time` measures it relative to the newest
candidate (another distribution) or not at all. `evidence` (what the production system actually showed the LLM) is a list of chunk objects
`{id, text, title?, ...}`, chunk ids of the local corpus, or plain strings.

For every query, retrieval runs against the local corpus (top `label_top_k`, default 5) and yields:
  * one `query` item (response_type, route, needs_graph), with the retrieved candidates;
  * one `query_chunk` item per top-k candidate (chunk_relevant);
  * one `query_evidence` item with the top-k as evidence, plus one with the logged evidence when there is some
    (answerable);
  * one `query_evidence_answer` item when the log has an answer (grounded, complete, risk_tier).

Each item carries a `display` text (the request, numbered chunks with document titles, the answer) so the judge
and `rlcd label audit` read the same thing a person would. Item ids hash the state, so re-running over the same
log adds nothing twice.

What this does not do: it does not label anything (that is `rlcd label bootstrap` / `audit`), it does not
deduplicate near-identical queries, and without an ingested corpus it can only produce `query` items (plus
evidence / answer items for rows that carry their own evidence).
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ...core.config import Project
from ...core.contract import ContractError
from ...labeling.store import LabelItem
from ..base import epoch
from .settings import DEFAULT_RETRIEVE_K, project_corpus_settings

DISPLAY_CHUNK_CHARS = 700
DISPLAY_LIST = 8


# ----------------------------------------------------------------------------- display

def clean_candidate(c: dict) -> dict:
    """A retrieved candidate as stored in a state or a log: NaN -> None (strict JSON); the graph's `via` label is
    dropped (features use path_length and degree only)."""
    out = {}
    for k, v in c.items():
        if isinstance(v, float) and not math.isfinite(v):
            v = None
        if k == "graph" and isinstance(v, dict):
            v = {g: x for g, x in v.items() if g in ("path_length", "degree")}
        out[k] = v
    return out


def _cut(text: Any, n: int) -> str:
    s = " ".join(str(text or "").split())
    return s if len(s) <= n else s[: n - 1] + "…"


def _chunk_lines(chunks: list[Any], n_chars: int = DISPLAY_CHUNK_CHARS) -> list[str]:
    lines = []
    for i, c in enumerate(chunks[:DISPLAY_LIST], 1):
        if isinstance(c, str):
            lines.append(f"[{i}] {_cut(c, n_chars)}")
            continue
        c = c if isinstance(c, dict) else {}
        head = " - ".join(x for x in (str(c.get("title") or "").strip(), f"({c.get('id')})") if x)
        lines.append(f"[{i}] {head}\n    {_cut(c.get('text'), n_chars)}")
    if len(chunks) > DISPLAY_LIST:
        lines.append(f"... and {len(chunks) - DISPLAY_LIST} more")
    return lines


def display_for(view: str, state: dict) -> str:
    """The text a labeller reads for one item of `view`."""
    q = f"Request: {state.get('query')}"
    if view == "query":
        cands = state.get("candidates") or []
        top = _chunk_lines(cands, 160) if cands else ["(none)"]
        return "\n".join([q, "", "Top retrieved chunks (context only):", *top])
    if view == "query_chunk":
        c = state.get("chunk") or {}
        return "\n".join([q, "", "Chunk to judge:", *_chunk_lines([c], 1500)])
    ev = state.get("evidence") or []
    parts = [q, "", "Evidence:", *(_chunk_lines(ev) if ev else ["(no evidence)"])]
    if view == "query_evidence_answer":
        parts += ["", "Drafted answer:", str(state.get("answer") or "")]
    return "\n".join(parts)


# ----------------------------------------------------------------------------- reading the log

def read_queries(path: Path) -> tuple[list[dict], list[str]]:
    """Rows of a query log (CSV / TSV with a `query` column, JSONL, or a JSON array) and notes about skipped rows."""
    path = Path(path)
    suffix = path.suffix.lower()
    notes: list[str] = []
    raw: list[Any] = []
    try:
        if suffix in (".csv", ".tsv"):
            with open(path, encoding="utf-8-sig", newline="") as f:
                reader = csv.DictReader(f, delimiter="\t" if suffix == ".tsv" else ",")
                if "query" not in (reader.fieldnames or []):
                    raise ContractError(f"{path.name}: the query log needs a 'query' column; columns are "
                                        f"{reader.fieldnames}", kind="config")
                raw = list(reader)
        elif suffix in (".jsonl", ".ndjson"):
            with open(path, encoding="utf-8") as f:
                for n, line in enumerate(f, 1):
                    if not line.strip():
                        continue
                    try:
                        raw.append(json.loads(line))
                    except json.JSONDecodeError:
                        notes.append(f"line {n}: not valid JSON, skipped")
        elif suffix == ".json":
            obj = json.loads(path.read_text(encoding="utf-8"))
            raw = obj if isinstance(obj, list) else obj.get("queries", []) if isinstance(obj, dict) else []
        else:
            raise ContractError(f"{path.name}: the query log must be .csv, .tsv, .jsonl or .json", kind="config")
    except OSError as e:
        raise ContractError(f"cannot read the query log {path}: {e.strerror}", kind="not_found") from None
    rows = []
    for i, r in enumerate(raw):
        if not isinstance(r, dict) or not isinstance(r.get("query"), str) or not r["query"].strip():
            notes.append(f"row {i}: no non-empty 'query', skipped")
            continue
        rows.append(r)
    return rows, notes


def _evidence(raw: Any, corpus: Any) -> list[dict] | None:
    if raw is None:
        return None
    if not isinstance(raw, list):
        raw = [raw]
    out = []
    for j, e in enumerate(raw):
        if isinstance(e, dict):
            out.append(clean_candidate({**e, "id": e.get("id") if e.get("id") is not None else f"evidence:{j}"}))
        elif isinstance(e, str):
            ch = corpus.chunk(e) if corpus is not None else None
            out.append(clean_candidate(ch) if ch is not None else {"id": f"evidence:{j}", "text": e})
    return out


def _item_id(view: str, state: dict) -> str:
    h = hashlib.sha256(json.dumps(state, sort_keys=True, ensure_ascii=False, default=str).encode()).hexdigest()
    return f"log:{view}:{h[:20]}"


# ----------------------------------------------------------------------------- tasks

def label_tasks(project: Project, provider: Any, corpus: Any = None) -> list[LabelItem]:
    """Label items for every query of `pack_config.queries`. `corpus` defaults to the project's local corpus
    (opened with the provider's encoder); without one, retrieval is skipped."""
    cfg = project.pack_config
    if not cfg.get("queries"):
        raise ContractError("pack_config.queries is not set: point it at a query log (CSV with a 'query' column, or "
                            "JSONL rows {query, time?, group?, answer?, evidence?})", kind="config")
    path = project.resolve_path(cfg["queries"])
    if not path.is_file():
        raise ContractError(f"query log {path} not found (pack_config.queries)", kind="not_found")
    rows, _ = read_queries(path)
    k = max(1, int(cfg.get("label_top_k") or DEFAULT_RETRIEVE_K))
    from .search import candidates

    if corpus is None:
        from .corpus import Corpus

        corpus = Corpus.open(project.home, provider.encoder, project_corpus_settings(project))
    items: list[LabelItem] = []
    seen: set[str] = set()

    def add(view: str, state: dict, group: str, t: int | None, meta: dict) -> None:
        iid = _item_id(view, state)
        if iid in seen:
            return
        seen.add(iid)
        items.append(LabelItem(id=iid, view=view, state=state, display=display_for(view, state), group=group, t=t,
                               meta=meta))

    for i, r in enumerate(rows):
        query = r["query"].strip()
        t = epoch(r.get("time"))
        group = str(r.get("group") or f"query:{hashlib.sha256(query.encode()).hexdigest()[:12]}")
        meta = {"source": path.name, "row": i}
        base: dict[str, Any] = {"query": query}
        if t is not None:
            base["now"] = datetime.fromtimestamp(t, UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
        cands = candidates(corpus, query, k) if corpus is not None else []
        add("query", {**base, "candidates": cands}, group, t, meta)
        top = cands[:k]
        for c in top:
            add("query_chunk", {**base, "chunk": c, "candidates": cands}, group, t, meta)
        if top:
            add("query_evidence", {**base, "evidence": top, "candidates": cands}, group, t, meta)
        logged = _evidence(r.get("evidence"), corpus)
        if logged:
            add("query_evidence", {**base, "evidence": logged, "candidates": cands or None}, group, t, meta)
        answer = r.get("answer")
        if isinstance(answer, str) and answer.strip():
            ev = logged if logged is not None else top
            add("query_evidence_answer", {"query": query, "evidence": ev, "answer": answer.strip()}, group, t, meta)
    return items
