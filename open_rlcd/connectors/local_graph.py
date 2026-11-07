"""A local entity graph over documents and chunks, for multi-hop retrieval without a graph database.

Nodes: `doc:<id>`, `chunk:<id>`, `ent:<normalised entity>`. Edges: doc - chunk (the chunk belongs to the doc),
chunk - entity (the chunk mentions it), optional doc - doc (explicit links between documents). The graph is
undirected. It uses networkx when installed and an equivalent dict-of-dicts adjacency otherwise; both give the
same results (neighbours are visited in sorted order).

Distances are counted in **hops between chunks**: a chunk that mentions a query entity is 1 hop away; a chunk
reached through one more entity it shares with a 1-hop chunk (or through the same / a linked document) is 2
hops away, and so on. `expand(["Zentrix"], hops=2)` therefore answers "chunks about Zentrix, and chunks about
the things those chunks mention".

Entities come from cheap, language-aware surface patterns (`entities`): capitalised multi-word phrases and
capitalised words mid-sentence (Latin and Mongolian Cyrillic), ALL-CAPS codes and product ids, quoted terms,
katakana runs, and terms from a configured list (needed for kanji names, which have no capitalisation). It is
not named-entity recognition: it misses lower-case names and picks up some capitalised non-entities; the
features built on it (path length, degree) are weighed by calibrated heads, not trusted blindly.

Persistence: one JSON file (node-link style), never pickle.
"""
from __future__ import annotations

import importlib
import json
import os
import re
import unicodedata
from collections import deque
from pathlib import Path
from typing import Any

FORMAT = "open-rlcd-graph/1"

# ----------------------------------------------------------------------------- entity extraction

def _wordset(text: str) -> frozenset[str]:
    return frozenset(text.split())


_LAT_WORD = r"[A-Z][A-Za-z0-9&'\-]*"
_CYR_UP, _CYR_LO = "А-ЯЁӨҮ", "а-яёөүА-ЯЁӨҮA-Za-z0-9\\-"
_CAP_RUN = re.compile(rf"\b{_LAT_WORD}(?:[ \t]+{_LAT_WORD})*")
_CYR_RUN = re.compile(rf"(?<![\w])[{_CYR_UP}][{_CYR_LO}]+(?:[ \t]+[{_CYR_UP}][{_CYR_LO}]+)*")
_CODE = re.compile(r"(?<![\w-])(?=[A-Z0-9_-]*[A-Z])(?=[A-Z0-9_-]*(?:\d|[A-Z]{2}))[A-Z0-9][A-Z0-9_-]*[A-Z0-9](?![\w-])")
_QUOTED = re.compile(r"[\"“„«「『]([^\"”»」』\n]{2,80})[\"”»」』]")
_KATAKANA = re.compile(r"[\u30a1-\u30fa\u30fc\u30fb]{2,}")
_CJK = re.compile(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\uac00-\ud7af]")
_SENT_START = re.compile(r"([.!?。！？:;\n]\s*|[\"“„«(\[]\s*)$")
_LEADING = _wordset("""a an the this that these those in on at for of and or but if when where who what which why how
does do did is are was were will would can could should shall may might must please our your their its it he she
they we i my me us them his her there here then also after before during since while with from by to as all each
every some any no not new see yes ok okay hi hello dear thanks thank note notes total both many most more other
such first second third finally overall according based following last next until unlike like however therefore
thus today yesterday tomorrow now currently recently meanwhile moreover furthermore although though because so
summary introduction conclusion background overview table figure section chapter page part step""")
_CYR_LEADING = _wordset("энэ тэр манай танай бүх ямар хэн юу хаана хэзээ яагаад яаж аль")
# multi-letter case endings only: single letters (dative -д / -т) end too many stems to strip safely
_MN_SUFFIXES = ("уудын", "үүдийн", "ийнхээ", "ынхаа", "аас", "ээс", "оос", "өөс", "тай", "тэй", "той", "ийн", "ын",
                "руу", "рүү", "ийг", "ыг")


def normalize_entity(text: str) -> str:
    """The graph key of an entity: NFKC, casefold, collapsed spaces, possessive 's dropped, and one common
    Mongolian case suffix stripped from Cyrillic words (so "Улаанбаатарт" and "Улаанбаатар" meet)."""
    s = unicodedata.normalize("NFKC", str(text)).casefold().strip(" \t\n\"'“”「」『』・ー-")
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"['’]s\b", "", s)
    words = []
    for w in s.split(" "):
        if re.fullmatch(r"[а-яёөү]+", w):
            for suf in _MN_SUFFIXES:
                if w.endswith(suf) and len(w) - len(suf) >= 4:
                    w = w[: -len(suf)]
                    break
        words.append(w)
    return " ".join(w for w in words if w)


def _at_sentence_start(text: str, pos: int) -> bool:
    return pos == 0 or bool(_SENT_START.search(text[max(0, pos - 8):pos]))


def _split_possessive(words: list[str]) -> list[list[str]]:
    """"Northwind Metals' CEO" -> [["Northwind", "Metals"], ["CEO"]]: a possessive ends a name."""
    parts: list[list[str]] = [[]]
    for w in words:
        end = w.endswith(("'", "’", "'s", "’s"))
        w = re.sub(r"(['’]s|['’])$", "", w).strip("-")
        if w:
            parts[-1].append(w)
        if end:
            parts.append([])
    return [p for p in parts if p]


def _lower_elsewhere(word: str, text: str) -> bool:
    """True when `word` also occurs in lower case in `text`: then it is an ordinary word, not a name."""
    return re.search(rf"(?<!\w){re.escape(word.lower())}(?!\w)", text) is not None


def _cap_phrases(text: str, run: re.Pattern, leading: frozenset[str]) -> list[str]:
    """Capitalised phrases. A single capitalised word counts mid-sentence; at a sentence start (where any word
    is capitalised) only if it is not a common sentence opener and never appears in lower case in the text."""
    out = []
    lowered = None
    for m in run.finditer(text):
        words = m.group(0).split()
        sentence_start = _at_sentence_start(text, m.start())
        while words and words[0].casefold().strip("'’") in leading:   # "The Acme Robotics" -> "Acme Robotics"
            words = words[1:]
            sentence_start = False
        if sentence_start and len(words) >= 2:
            # an ordinary word capitalised only because it opens the sentence ("Салбарыг Бат-Эрдэнэ ...")
            if lowered is None:
                lowered = set(re.findall(r"\w+", text))      # case kept: only words written in lower case count
            if words[0].lower() in lowered or normalize_entity(words[0]) in lowered:
                words = words[1:]
                sentence_start = False
        for i, part in enumerate(_split_possessive(words)):
            first = sentence_start and i == 0
            if len(part) >= 2:
                out.append(" ".join(part))
                continue
            w = part[0]
            if len(w) < 3 or w.casefold() in leading:
                continue
            if first and _lower_elsewhere(w, text):
                continue
            out.append(w)
    return out


def entities(text: str, extra: list[str] | None = None) -> list[str]:
    """Entity mentions in `text`, deduplicated by `normalize_entity`, in order of first appearance.

    `extra` is a list of known terms (any script; needed for kanji / hanzi names) matched as case-insensitive
    substrings."""
    s = unicodedata.normalize("NFKC", str(text or ""))
    found: list[tuple[int, str]] = []
    for m in _QUOTED.finditer(s):
        found.append((m.start(1), m.group(1).strip()))
    for m in _CODE.finditer(s):
        found.append((m.start(), m.group(0)))
    for phrase in _cap_phrases(s, _CAP_RUN, _LEADING):
        found.append((s.find(phrase), phrase))
    for phrase in _cap_phrases(s, _CYR_RUN, _CYR_LEADING):
        found.append((s.find(phrase), phrase))
    for m in _KATAKANA.finditer(s):
        k = m.group(0).strip("・")
        if len(k.replace("ー", "")) >= 2:
            found.append((m.start(), k))
    if extra:
        low = s.casefold()
        for term in extra:
            t = unicodedata.normalize("NFKC", str(term)).strip()
            if len(t) >= 2 and (i := low.find(t.casefold())) >= 0:
                found.append((i, t))
    out, seen = [], set()
    for _, e in sorted(found, key=lambda t: t[0]):
        key = normalize_entity(e)
        if len(key) >= 2 and key not in seen:
            seen.add(key)
            out.append(e)
    return out


# ----------------------------------------------------------------------------- adjacency backends

class _DictGraph:
    """The networkx-free fallback: an undirected graph as {node: {neighbour: edge attrs}}."""

    def __init__(self) -> None:
        self.adj: dict[str, dict[str, dict]] = {}
        self.attrs: dict[str, dict] = {}

    def add_node(self, n: str, **attrs: Any) -> None:
        self.adj.setdefault(n, {})
        self.attrs.setdefault(n, {}).update(attrs)

    def add_edge(self, a: str, b: str, **attrs: Any) -> None:
        self.add_node(a)
        self.add_node(b)
        self.adj[a][b] = dict(attrs)
        self.adj[b][a] = self.adj[a][b]

    def remove_node(self, n: str) -> None:
        for m in self.adj.pop(n, {}):
            self.adj[m].pop(n, None)
        self.attrs.pop(n, None)

    def has_node(self, n: str) -> bool:
        return n in self.adj

    def edge(self, a: str, b: str) -> dict | None:
        return self.adj.get(a, {}).get(b)

    def neighbors(self, n: str) -> list[str]:
        return list(self.adj.get(n, ()))

    def degree(self, n: str) -> int:
        return len(self.adj.get(n, ()))

    def nodes(self) -> list[tuple[str, dict]]:
        return [(n, self.attrs.get(n, {})) for n in self.adj]

    def edges(self) -> list[tuple[str, str, dict]]:
        out, seen = [], set()
        for a, nb in self.adj.items():
            for b, d in nb.items():
                if (b, a) not in seen:
                    seen.add((a, b))
                    out.append((a, b, d))
        return out

    def number_of_nodes(self) -> int:
        return len(self.adj)

    def number_of_edges(self) -> int:
        return sum(len(v) for v in self.adj.values()) // 2


class _NxGraph:
    """Thin wrapper giving networkx.Graph the same small interface as `_DictGraph`."""

    def __init__(self, nx: Any) -> None:
        self.g = nx.Graph()

    def add_node(self, n: str, **attrs: Any) -> None:
        self.g.add_node(n, **attrs)

    def add_edge(self, a: str, b: str, **attrs: Any) -> None:
        self.g.add_edge(a, b, **attrs)

    def remove_node(self, n: str) -> None:
        if self.g.has_node(n):
            self.g.remove_node(n)

    def has_node(self, n: str) -> bool:
        return self.g.has_node(n)

    def edge(self, a: str, b: str) -> dict | None:
        return dict(self.g.edges[a, b]) if self.g.has_edge(a, b) else None

    def neighbors(self, n: str) -> list[str]:
        return list(self.g.neighbors(n)) if self.g.has_node(n) else []

    def degree(self, n: str) -> int:
        return int(self.g.degree(n)) if self.g.has_node(n) else 0

    def nodes(self) -> list[tuple[str, dict]]:
        return list(self.g.nodes(data=True))

    def edges(self) -> list[tuple[str, str, dict]]:
        return list(self.g.edges(data=True))

    def number_of_nodes(self) -> int:
        return self.g.number_of_nodes()

    def number_of_edges(self) -> int:
        return self.g.number_of_edges()


_SEP = "\n"   # separates the ids of the documents that declared a doc-doc link


def _declarers(attrs: dict | None) -> set[str]:
    return {x for x in str((attrs or {}).get("by") or "").split(_SEP) if x}


def _networkx() -> Any | None:
    try:
        return importlib.import_module("networkx")
    except ImportError:
        return None


# ----------------------------------------------------------------------------- the store

class LocalGraphStore:
    """Entity graph of docs and chunks, persisted as JSON at `path` (see the module docstring).

    `terms`: known entity names to match as substrings (kanji names, product names without capitals).
    `use_networkx`: None = networkx when importable, False = the dict fallback, True = require networkx.
    `max_fanout`: an entity mentioned by more chunks than this is a hub (a company name on every page); it is
    still a seed, but expansion does not walk through it, so one hub cannot pull in the whole corpus."""

    def __init__(self, path: Path, terms: list[str] | None = None, use_networkx: bool | None = None,
                 max_fanout: int = 200):
        self.path = Path(path)
        self.terms = [str(t) for t in (terms or [])]
        self.max_fanout = int(max_fanout)
        nx = _networkx() if use_networkx is not False else None
        if use_networkx is True and nx is None:
            from .base import ConnectorError

            raise ConnectorError("networkx is not installed; install the 'documents' extra or use the fallback")
        self.g: _NxGraph | _DictGraph = _NxGraph(nx) if nx is not None else _DictGraph()
        self.labels: dict[str, str] = {}          # ent key -> first surface form seen

    @property
    def backend(self) -> str:
        return "networkx" if isinstance(self.g, _NxGraph) else "dict"

    def entities(self, text: str, extra: list[str] | None = None) -> list[str]:
        return entities(text, [*self.terms, *(extra or [])])

    def stats(self) -> dict:
        kinds: dict[str, int] = {}
        for n, _ in self.g.nodes():
            k = n.split(":", 1)[0]
            kinds[k] = kinds.get(k, 0) + 1
        return {"nodes": self.g.number_of_nodes(), "edges": self.g.number_of_edges(), **kinds,
                "backend": self.backend}

    # ------------------------------------------------------------------ writes

    def add_document(self, doc_id: str, chunks: list[tuple[str, str]], links: list[str] | None = None) -> None:
        """(Re)place a document: its chunks (id, text), the entities they mention, and links to other docs."""
        doc = f"doc:{doc_id}"
        self.remove_document(doc_id, keep_node=True)
        self.g.add_node(doc, kind="doc")
        for cid, text in chunks:
            c = f"chunk:{cid}"
            self.g.add_node(c, kind="chunk", doc=str(doc_id))
            self.g.add_edge(doc, c, kind="has_chunk")
            for e in self.entities(text):
                key = normalize_entity(e)
                if not key:
                    continue
                self.labels.setdefault(key, e)
                n = f"ent:{key}"
                self.g.add_node(n, kind="ent", label=self.labels[key])
                self.g.add_edge(c, n, kind="mentions")
        for other in links or []:
            o = f"doc:{other}"
            if o != doc:
                if not self.g.has_node(o):
                    self.g.add_node(o, kind="doc", placeholder=True)
                by = _declarers(self.g.edge(doc, o)) | {str(doc_id)}
                self.g.add_edge(doc, o, kind="links", by=_SEP.join(sorted(by)))

    def remove_document(self, doc_id: str, keep_node: bool = False) -> None:
        doc = f"doc:{doc_id}"
        if not self.g.has_node(doc):
            return
        orphan_check = set()
        for n in self.g.neighbors(doc):
            if n.startswith("chunk:"):
                orphan_check.update(m for m in self.g.neighbors(n) if m.startswith("ent:"))
                self.g.remove_node(n)
        kept = {}                 # links other documents declared survive a re-ingest or removal of this one
        for n in self.g.neighbors(doc):
            if n.startswith("doc:"):
                by = _declarers(self.g.edge(doc, n)) - {str(doc_id)}
                if by:
                    kept[n] = by
        self.g.remove_node(doc)
        if keep_node or kept:
            self.g.add_node(doc, kind="doc", **({} if keep_node else {"placeholder": True}))
            for n, by in kept.items():
                self.g.add_edge(doc, n, kind="links", by=_SEP.join(sorted(by)))
        for e in orphan_check:
            if self.g.degree(e) == 0:
                self.g.remove_node(e)
                self.labels.pop(e[4:], None)

    # ------------------------------------------------------------------ reads

    def _seeds(self, terms: list[str]) -> list[str]:
        """Entity nodes named by `terms`: exact key match, else entities whose key contains the term as whole
        words (Latin / Cyrillic) or as a substring (CJK), or that the term contains."""
        seeds: list[str] = []
        ent_keys = None
        for t in terms:
            key = normalize_entity(t)
            if len(key) < 2:
                continue
            node = f"ent:{key}"
            if self.g.has_node(node):
                seeds.append(node)
                continue
            if ent_keys is None:
                ent_keys = [n[4:] for n, _ in self.g.nodes() if n.startswith("ent:")]
            padded = f" {key} "          # keys are single-spaced: whole-word containment is a padded substring
            cjk = bool(_CJK.search(key))
            for k in ent_keys:
                if padded in f" {k} " or (len(k) >= 3 and f" {k} " in padded) \
                        or (cjk and (key in k or (len(k) >= 2 and k in key))):
                    seeds.append(f"ent:{k}")
        return list(dict.fromkeys(seeds))

    def _walk(self, starts: list[str], max_edges: int) -> dict[str, tuple[int, str, str]]:
        """Multi-source BFS: node -> (edge distance, predecessor, originating start). Hubs are not walked
        through (except as starts)."""
        best: dict[str, tuple[int, str, str]] = {s: (0, "", s) for s in starts}
        queue = deque(starts)
        while queue:
            n = queue.popleft()
            d, _, origin = best[n]
            if d >= max_edges:
                continue
            if d > 0 and n.startswith("ent:") and self.g.degree(n) > self.max_fanout:
                continue
            for m in sorted(self.g.neighbors(n)):
                if m not in best:
                    best[m] = (d + 1, n, origin)
                    queue.append(m)
        return best

    def _chunk_rows(self, best: dict[str, tuple[int, str, str]], hop_of, exclude: set[str], limit: int,
                    seeds: set[str] | None = None) -> list[dict]:
        rows = []
        for n, (d, prev, origin) in best.items():
            if not n.startswith("chunk:") or n in exclude:
                continue
            hops = hop_of(d)
            seed_hits = sum(1 for m in self.g.neighbors(n) if m in seeds) if seeds else 0
            rows.append({"id": n[6:], "path_length": hops, "degree": self.g.degree(n),
                         "via": self._label(prev), "seed": self._label(origin), "_hits": seed_hits})
        rows.sort(key=lambda r: (r["path_length"], -r["_hits"], r["id"]))
        for r in rows:
            r.pop("_hits")
        return rows[: max(0, int(limit))]

    def _label(self, node: str) -> str:
        if node.startswith("ent:"):
            return self.labels.get(node[4:], node[4:])
        return node

    def expand(self, terms: list[str], hops: int = 2, limit: int = 50) -> list[dict]:
        """Chunks within `hops` chunk-hops of the entities named by `terms`, nearest first (then by how many
        of those entities they mention): [{id, path_length, degree, via, seed}]."""
        hops = max(1, int(hops))
        seeds = self._seeds([str(t) for t in terms or []])
        if not seeds:
            return []
        best = self._walk(seeds, max_edges=2 * hops)
        return self._chunk_rows(best, lambda d: (d + 1) // 2, set(), limit, set(seeds))

    def neighbors(self, ids: list[str], hops: int = 1, limit: int = 50) -> list[dict]:
        """Chunks within `hops` chunk-hops of the given chunks (through a shared entity or document)."""
        hops = max(1, int(hops))
        starts = [f"chunk:{i}" for i in ids if self.g.has_node(f"chunk:{i}")]
        if not starts:
            return []
        best = self._walk(starts, max_edges=2 * hops)
        return self._chunk_rows(best, lambda d: (d + 1) // 2, set(starts), limit)

    def chunk_degree(self, chunk_id: str) -> int:
        return self.g.degree(f"chunk:{chunk_id}")

    # ------------------------------------------------------------------ persistence

    def to_dict(self) -> dict:
        return {"format": FORMAT, "directed": False, "terms": self.terms,
                "nodes": [{"id": n, **{k: v for k, v in a.items() if isinstance(v, (str, int, float, bool))}}
                          for n, a in self.g.nodes()],
                "links": [{"source": a, "target": b, **{k: v for k, v in d.items() if isinstance(v, str)}}
                          for a, b, d in self.g.edges()]}

    def save(self) -> Path:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(self.path.name + ".tmp")
        tmp.write_text(json.dumps(self.to_dict(), ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, self.path)
        return self.path

    @classmethod
    def load(cls, path: Path, terms: list[str] | None = None, use_networkx: bool | None = None,
             max_fanout: int = 200) -> LocalGraphStore:
        path = Path(path)
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("format") != FORMAT:
            raise ValueError(f"{path}: unknown graph format {data.get('format')!r}")
        store = cls(path, terms=terms if terms is not None else data.get("terms"), use_networkx=use_networkx,
                    max_fanout=max_fanout)
        for node in data.get("nodes") or []:
            attrs = {k: v for k, v in node.items() if k != "id"}
            store.g.add_node(str(node["id"]), **attrs)
            if str(node["id"]).startswith("ent:") and attrs.get("label"):
                store.labels[str(node["id"])[4:]] = str(attrs["label"])
        for link in data.get("links") or []:
            attrs = {k: v for k, v in link.items() if k not in ("source", "target")}
            store.g.add_edge(str(link["source"]), str(link["target"]), **attrs)
        return store
