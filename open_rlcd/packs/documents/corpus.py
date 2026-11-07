"""The local document corpus of the documents pack: ingest a folder, chunk it, embed it, index it, retrieve.

    <home>/corpus/
      meta.json        format, encoder config (id, dim, sha256), chunking, counts, updated_at
      docs.jsonl       one line per document: id, title, source, sha256, updated_at, authority, links, chunk ids
      chunks.jsonl     one line per chunk: id ("<doc id>#<n>"), doc_id, ord, title, text, updated_at, authority
      bm25.json        BM25 inverted index over the chunks (plain JSON)
      vectors/         LocalVectorStore (vectors.npy + items.jsonl + meta.json)
      graph.json       LocalGraphStore (doc / chunk / entity graph)

`retrieve(query)` runs three retrievers and fuses them with reciprocal-rank fusion (RRF, k = 60): vector top-k
(cosine on the encoder's query embedding), graph expansion from the entities named in the query (multi-hop,
`graph_hops`), and BM25 top-k. Every candidate gets the exact cosine similarity of its stored vector (so graph
and lexical hits have a comparable `score` too), its BM25 score, and graph distance when the graph found it.

Ingestion is idempotent: a document whose content hash (text + title + metadata + chunking settings) did not
change is skipped; a changed one replaces its old chunks everywhere. Ingestion is additive: documents that
disappeared from the folder stay until `remove()` is called.

What this is not: a database. One writer at a time, everything is loaded into memory, BM25 is rebuilt on every
ingest. It is meant for a laptop-sized corpus (up to some hundred thousand chunks); remote connectors (Qdrant,
pgvector, Neo4j) are the path beyond that and are not implemented here yet.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

import numpy as np

from ...connectors.local_graph import LocalGraphStore
from ...connectors.local_vector import LocalVectorStore
from ...core.contract import ContractError
from ...features.lexical import BM25, tokens
from ...features.retrieval import to_epoch
from ...features.text import Encoder, encoder_config

FORMAT = "open-rlcd-corpus/1"
TEXT_SUFFIXES = {".txt", ".md", ".markdown"}
HTML_SUFFIXES = {".html", ".htm"}
SUPPORTED = TEXT_SUFFIXES | HTML_SUFFIXES | {".pdf", ".json", ".jsonl"}
MAX_FILE_BYTES = 50 << 20
MAX_REPORTED_SKIPS = 50


# ----------------------------------------------------------------------------- chunking

_SENTENCE = re.compile(r"[^.!?。！？]*[.!?。！？]+[\"'”」』)\]]*\s*|[^.!?。！？]+$")


def _clean_paragraph(p: str) -> str:
    lines = [re.sub(r"[ \t ]+", " ", ln).strip() for ln in p.split("\n")]
    return "\n".join(ln for ln in lines if ln)


def _hard_split(s: str, limit: int) -> list[str]:
    out = []
    while len(s) > limit:
        cut = s.rfind(" ", int(limit * 0.8), limit)
        cut = cut if cut > 0 else limit
        out.append(s[:cut].strip())
        s = s[cut:].strip()
    if s:
        out.append(s)
    return out


def _split_long(p: str, limit: int) -> list[str]:
    """A paragraph longer than `limit`: pack whole sentences, cut a too-long sentence at a space (or hard)."""
    pieces, buf = [], ""
    for m in _SENTENCE.finditer(p):
        s = m.group(0)
        if not s.strip():
            continue
        if len(s.strip()) > limit:
            if buf.strip():
                pieces.append(buf.strip())
            buf = ""
            pieces.extend(_hard_split(s.strip(), limit))
        elif len(buf) + len(s) <= limit:
            buf += s
        else:
            pieces.append(buf.strip())
            buf = s
    if buf.strip():
        pieces.append(buf.strip())
    return [x for x in pieces if x]


def _tail(s: str, n: int) -> str:
    """The last ~n characters of a chunk, starting at a sentence or word boundary when there is one nearby."""
    if n <= 0:
        return ""
    if len(s) <= n:
        return s
    t = s[-n:]
    m = re.search(r"[.!?。！？]\s*|\s", t[: n // 2])
    return (t[m.end():] if m else t).strip()


def chunk_text(text: str, max_chars: int = 1200, overlap: int = 150) -> list[str]:
    """Paragraph-aware chunks of at most `max_chars` characters.

    Paragraphs (blank-line separated) are packed together while they fit; a paragraph that does not fit alone is
    split at sentence ends (`.!?` and `。！？`), and a sentence that still does not fit is cut at a space (or
    hard, for scripts without spaces). Consecutive chunks share about `overlap` characters (the tail of the
    previous chunk, starting at a word / sentence boundary) whenever that fits within `max_chars`."""
    max_chars = max(50, int(max_chars))
    overlap = max(0, min(int(overlap), max_chars // 3))
    unit_limit = max_chars - overlap - 1 if overlap else max_chars
    text = str(text or "").replace("\r\n", "\n").replace("\r", "\n")
    units: list[str] = []
    for p in re.split(r"\n\s*\n", text):
        p = _clean_paragraph(p)
        if not p:
            continue
        units.extend([p] if len(p) <= max_chars else _split_long(p, unit_limit))
    chunks: list[str] = []
    cur = ""
    for u in units:
        if not cur:
            cur = u
        elif len(cur) + 2 + len(u) <= max_chars:
            cur = f"{cur}\n\n{u}"
        else:
            chunks.append(cur)
            tail = _tail(cur, overlap)
            cur = f"{tail}\n{u}" if tail and len(tail) + 1 + len(u) <= max_chars else u
    if cur:
        chunks.append(cur)
    return chunks


# ----------------------------------------------------------------------------- readers

class _HTMLText(HTMLParser):
    """Visible text of an HTML page: script / style / template content dropped, block tags become paragraph
    breaks, <title> kept apart, hrefs collected (for links between local documents)."""

    SKIP = {"script", "style", "noscript", "template", "svg", "head"}
    BLOCK = {"p", "div", "br", "li", "ul", "ol", "tr", "table", "section", "article", "header", "footer", "h1",
             "h2", "h3", "h4", "h5", "h6", "blockquote", "pre", "hr", "dd", "dt", "main", "aside", "nav", "form"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.title_parts: list[str] = []
        self.hrefs: list[str] = []
        self._skip = 0
        self._in_title = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "title":
            self._in_title = True
        elif tag in self.SKIP:
            self._skip += 1
        if tag in self.BLOCK:
            self.parts.append("\n\n")
        if tag == "a":
            href = dict(attrs).get("href")
            if href:
                self.hrefs.append(href)

    def handle_endtag(self, tag: str) -> None:
        if tag == "title":
            self._in_title = False
        elif tag in self.SKIP and self._skip:
            self._skip -= 1
        if tag in self.BLOCK:
            self.parts.append("\n\n")

    def handle_data(self, data: str) -> None:
        if self._in_title:
            self.title_parts.append(data)
        elif not self._skip:
            self.parts.append(data)

    def text(self) -> str:
        return re.sub(r"\n\s*\n+", "\n\n", "".join(self.parts)).strip()

    def title(self) -> str:
        return " ".join("".join(self.title_parts).split())


_MD_TITLE = re.compile(r"^\s{0,3}#{1,6}\s+(.+?)\s*#*\s*$", re.MULTILINE)
_MD_LINK = re.compile(r"\[[^\]]*\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")


def _local_links(hrefs: list[str], file: Path, root: Path, root_name: str) -> list[str]:
    """Doc ids of the local documents that `hrefs` point at (same ingest root only)."""
    out = []
    for h in hrefs:
        u = urlparse(h)
        if u.scheme or u.netloc or not u.path:
            continue
        target = (file.parent / unquote(u.path)).resolve()
        try:
            rel = target.relative_to(root)
        except ValueError:
            continue
        if target.suffix.lower() in SUPPORTED:
            out.append(f"{root_name}/{rel.as_posix()}")
    return list(dict.fromkeys(out))


def _first(rec: dict, keys: tuple[str, ...]) -> Any:
    for k in keys:
        v = rec.get(k)
        if v is not None and v != "":
            return v
    return None


def _authority(v: Any) -> float | None:
    try:
        a = float(v)
    except (TypeError, ValueError):
        return None
    return min(1.0, max(0.0, a)) if math.isfinite(a) else None


def _links(v: Any) -> list[str]:
    if isinstance(v, str):
        return [v]
    if isinstance(v, (list, tuple)):
        return [str(x) for x in v if isinstance(x, (str, int))]
    return []


def record_to_doc(rec: dict, default_id: str, source: str = "") -> dict | None:
    """A JSON record -> a document dict, or None when it has no text. Text from `text` / `content` / `body`."""
    text = _first(rec, ("text", "content", "body"))
    if isinstance(text, list):
        text = "\n\n".join(str(t) for t in text if t)
    if not isinstance(text, str) or not text.strip():
        return None
    rid = _first(rec, ("id", "doc_id", "_id"))
    title = _first(rec, ("title", "name", "subject"))
    ts = to_epoch(_first(rec, ("updated_at", "updated", "modified", "date", "timestamp")))
    return {"id": str(rid) if rid is not None else default_id, "title": str(title) if title else "",
            "text": text, "updated_at": int(ts) if ts is not None else None,
            "authority": _authority(rec.get("authority")), "links": _links(rec.get("links")), "source": source}


def _json_records(raw: str, suffix: str) -> tuple[list[dict], int]:
    bad = 0
    if suffix == ".jsonl":
        recs = []
        for line in raw.splitlines():
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                bad += 1
                continue
            if isinstance(obj, dict):
                recs.append(obj)
            else:
                bad += 1
        return recs, bad
    obj = json.loads(raw)
    if isinstance(obj, dict):
        for key in ("documents", "records", "items", "data"):
            if isinstance(obj.get(key), list):
                obj = obj[key]
                break
        else:
            return [obj], 0
    if isinstance(obj, list):
        recs = [o for o in obj if isinstance(o, dict)]
        return recs, len(obj) - len(recs)
    return [], 1


def read_file(path: Path, root: Path, root_name: str) -> tuple[list[dict], list[dict]]:
    """Documents in one file, and skip notes ({path, reason}). Raises nothing for bad content."""
    rel = path.relative_to(root).as_posix() if path != root else path.name
    doc_id = f"{root_name}/{rel}" if path != root else path.name
    suffix = path.suffix.lower()
    mtime = int(path.stat().st_mtime)
    base = {"id": doc_id, "updated_at": mtime, "authority": None, "links": [], "source": str(path)}
    try:
        if path.stat().st_size > MAX_FILE_BYTES:
            return [], [{"path": rel, "reason": f"larger than {MAX_FILE_BYTES >> 20} MB"}]
        if suffix in TEXT_SUFFIXES:
            text = path.read_text(encoding="utf-8", errors="replace").lstrip("﻿")
            m = _MD_TITLE.search(text) if suffix != ".txt" else None
            links = _local_links(_MD_LINK.findall(text), path, root, root_name) if suffix != ".txt" else []
            title = m.group(1).strip() if m else ""
            return [{**base, "title": title or path.stem, "title_is_filename": not title, "text": text,
                     "links": links}], []
        if suffix in HTML_SUFFIXES:
            parser = _HTMLText()
            parser.feed(path.read_text(encoding="utf-8", errors="replace"))
            parser.close()
            return [{**base, "title": parser.title() or path.stem, "title_is_filename": not parser.title(),
                     "text": parser.text(),
                     "links": _local_links(parser.hrefs, path, root, root_name)}], []
        if suffix == ".pdf":
            try:
                from pypdf import PdfReader
            except ImportError:
                return [], [{"path": rel, "reason": "PDF support needs pypdf: install the 'documents' extra"}]
            reader = PdfReader(str(path))
            if reader.is_encrypted and not reader.decrypt(""):
                return [], [{"path": rel, "reason": "encrypted PDF"}]
            text = "\n\n".join((p.extract_text() or "") for p in reader.pages)
            meta_title = getattr(reader.metadata, "title", None) if reader.metadata else None
            return [{**base, "title": str(meta_title or path.stem), "title_is_filename": not meta_title,
                     "text": text}], []
        if suffix in (".json", ".jsonl"):
            recs, bad = _json_records(path.read_text(encoding="utf-8", errors="replace"), suffix)
            docs, notes = [], []
            for i, r in enumerate(recs):
                d = record_to_doc(r, default_id=f"{doc_id}:{i}", source=str(path))
                if d is None:
                    bad += 1
                    continue
                if d["updated_at"] is None:
                    d["updated_at"] = mtime
                docs.append(d)
            if bad:
                notes.append({"path": rel, "reason": f"{bad} records without text or not objects"})
            return docs, notes
    except Exception as e:  # one unreadable file never stops an ingest
        return [], [{"path": rel, "reason": f"{type(e).__name__}: {e}"[:300]}]
    return [], [{"path": rel, "reason": "unsupported file type"}]


def _walk(root: Path) -> list[Path]:
    """Supported files under `root`, sorted; hidden files / directories and symlinks are skipped (a symlink
    could point outside the folder the user chose to ingest)."""
    if root.is_file():
        return [root] if root.suffix.lower() in SUPPORTED else []
    out = []
    for d, dirs, files in os.walk(root, followlinks=False):
        dirs[:] = sorted(x for x in dirs if not x.startswith(".") and not os.path.islink(os.path.join(d, x)))
        for f in sorted(files):
            p = Path(d) / f
            if f.startswith(".") or p.is_symlink() or p.suffix.lower() not in SUPPORTED:
                continue
            out.append(p)
    return out


# ----------------------------------------------------------------------------- the corpus

def _opens_with_title(rows: list[dict], title: str) -> bool:
    """True when the first chunk already opens with the title (a Markdown heading)."""
    return bool(rows) and rows[0]["text"].lstrip("# ").startswith(title)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def _write_json(path: Path, obj: Any) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def _read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


class Corpus:
    """A local, multilingual document corpus: chunks + vectors + entity graph + BM25 under `<home>/corpus`.

    config keys (all optional): `vector: {kind: local, backend: auto|faiss|hnswlib|numpy}`, `graph: {kind:
    local, terms: [...known entity names...], max_fanout: 200}`, `max_chars`, `overlap`, `rrf_k` (60),
    `batch_size` (64), `on_encoder_change: error | reembed`."""

    def __init__(self, home: Path, encoder: Encoder, config: dict | None = None):
        self.home = Path(home)
        self.dir = self.home / "corpus"
        self.encoder = encoder
        self.config = dict(config or {})
        vcfg = dict(self.config.get("vector") or {"kind": "local"})
        gcfg = dict(self.config.get("graph") or {"kind": "local"})
        for name, c in (("vector", vcfg), ("graph", gcfg)):
            kind = str(c.get("kind") or "local").lower()
            if kind != "local":
                raise ContractError(f"{name} connector kind {kind!r} is not available yet; the documents pack "
                                    f"supports {name}: {{kind: local}} for now", kind="unsupported_connector")
        self.max_chars = int(self.config.get("max_chars") or 1200)
        self.overlap = int(self.config.get("overlap") if self.config.get("overlap") is not None else 150)
        self.rrf_k = float(self.config.get("rrf_k") or 60)
        self.batch_size = max(1, int(self.config.get("batch_size") or 64))
        self._backend = str(vcfg.get("backend") or "auto")
        self._terms = [str(t) for t in (gcfg.get("terms") or [])]
        self._fanout = int(gcfg.get("max_fanout") or 200)
        self.docs: dict[str, dict] = {}
        self.chunks: dict[str, dict] = {}
        self.bm25: BM25 | None = None
        self._bm25_ids: list[str] = []
        self._bm25_pos: dict[str, int] = {}
        self.vectors = LocalVectorStore(self.dir / "vectors", encoder.dim, backend=self._backend)
        self.graph = LocalGraphStore(self.dir / "graph.json", terms=self._terms, max_fanout=self._fanout)
        if self.exists():
            self._load()

    # ------------------------------------------------------------------ state

    def exists(self) -> bool:
        return (self.dir / "meta.json").is_file() and (self.dir / "chunks.jsonl").is_file()

    @classmethod
    def open(cls, home: Path, encoder: Encoder, config: dict | None = None) -> Corpus | None:
        """The corpus under `home`, or None when nothing was ingested there yet."""
        c = cls(home, encoder, config)
        return c if c.exists() else None

    def _load(self) -> None:
        meta = json.loads((self.dir / "meta.json").read_text(encoding="utf-8"))
        if meta.get("format") != FORMAT:
            raise ContractError(f"{self.dir}: unknown corpus format {meta.get('format')!r}", kind="corpus_format")
        self.docs = {d["id"]: d for d in _read_jsonl(self.dir / "docs.jsonl")}
        self.chunks = {c["id"]: c for c in _read_jsonl(self.dir / "chunks.jsonl")}
        if (self.dir / "vectors" / "meta.json").is_file():
            self.vectors = LocalVectorStore.load(self.dir / "vectors", backend=self._backend)
        if (self.dir / "graph.json").is_file():
            # configured terms win; without any, the terms the graph was built with are kept
            self.graph = LocalGraphStore.load(self.dir / "graph.json", terms=self._terms or None,
                                              max_fanout=self._fanout)
        bm = self.dir / "bm25.json"
        if bm.is_file():
            d = json.loads(bm.read_text(encoding="utf-8"))
            self.bm25 = BM25.from_dict(d)
            self._bm25_ids = [str(i) for i in d.get("ids") or []]
            self._bm25_pos = {cid: i for i, cid in enumerate(self._bm25_ids)}
        else:
            self._rebuild_bm25()
        enc = meta.get("encoder") or {}
        if enc.get("sha256") != self.encoder.sha256 or int(enc.get("dim") or 0) != int(self.encoder.dim):
            if str(self.config.get("on_encoder_change") or "error") == "reembed":
                self.reembed()
            else:
                raise ContractError(
                    f"the corpus in {self.dir} was embedded with encoder {enc.get('id')!r} "
                    f"(sha256 {str(enc.get('sha256'))[:16]}...), not {self.encoder.id!r} "
                    f"(sha256 {self.encoder.sha256[:16]}...): its vectors would be meaningless. Re-embed it "
                    f"(config on_encoder_change: reembed) or use the original encoder.", kind="encoder_mismatch")

    def stats(self) -> dict:
        return {"docs": len(self.docs), "chunks": len(self.chunks), "vectors": self.vectors.count(),
                "vector_backend": self.vectors.backend, "graph": self.graph.stats(), "encoder": self.encoder.id}

    def chunk(self, chunk_id: str) -> dict | None:
        c = self.chunks.get(str(chunk_id))
        return dict(c) if c is not None else None

    def doc(self, doc_id: str) -> dict | None:
        d = self.docs.get(str(doc_id))
        return dict(d) if d is not None else None

    # ------------------------------------------------------------------ ingest

    def ingest_folder(self, path: Path, max_chars: int = 1200, overlap: int = 150) -> dict:
        """Ingest every supported file under `path` (.txt .md .markdown .html .htm .pdf .json .jsonl).

        Document ids are `<folder name>/<relative path>` (`:<n>` appended per record of a JSON file without
        ids), so re-ingesting the same folder updates in place. Returns counts and up to 50 skip notes."""
        root = Path(path).expanduser().resolve()
        if not root.exists():
            raise ContractError(f"nothing to ingest: {root} does not exist", kind="not_found")
        files = _walk(root)
        root_name = root.name if root.is_dir() else ""
        docs, notes = [], []
        for f in files:
            d, n = read_file(f, root, root_name)
            docs.extend(d)
            notes.extend(n)
        out = self._ingest(docs, max_chars, overlap, notes)
        out["files"] = len(files)
        return out

    def ingest_records(self, records: list[dict], max_chars: int | None = None, overlap: int | None = None) -> dict:
        """Ingest documents given as dicts `{id, title?, text, updated_at?, authority?, links?}`."""
        docs, notes = [], []
        for i, r in enumerate(records or []):
            d = record_to_doc(r, default_id=f"record:{i}", source="records") if isinstance(r, dict) else None
            if d is None:
                notes.append({"path": f"record {i}", "reason": "no text"})
                continue
            docs.append(d)
        return self._ingest(docs, self.max_chars if max_chars is None else max_chars,
                            self.overlap if overlap is None else overlap, notes)

    @staticmethod
    def _hash(doc: dict, max_chars: int, overlap: int) -> str:
        key = {k: doc.get(k) for k in ("title", "text", "updated_at", "authority", "links")}
        key["chunking"] = [int(max_chars), int(overlap)]
        return hashlib.sha256(json.dumps(key, sort_keys=True, ensure_ascii=False).encode()).hexdigest()

    def _passage(self, title: str, text: str) -> str:
        return f"{title}\n{text}" if title and not text.lstrip().startswith(title) else text

    def _drop_doc(self, doc_id: str) -> None:
        old = self.docs.pop(doc_id, None)
        ids = [cid for cid in (old or {}).get("chunk_ids", [])]
        for cid in ids:
            self.chunks.pop(cid, None)
        self.vectors.delete(ids)
        self.graph.remove_document(doc_id)

    def _ingest(self, docs: list[dict], max_chars: int, overlap: int, notes: list[dict] | None = None) -> dict:
        notes = list(notes or [])
        self.max_chars, self.overlap = int(max_chars), int(overlap)
        added = updated = unchanged = 0
        new_chunks: list[dict] = []
        for d in docs:
            if not str(d.get("text") or "").strip():
                notes.append({"path": d.get("source") or d["id"], "reason": "no text"})
                continue
            h = self._hash(d, max_chars, overlap)
            old = self.docs.get(d["id"])
            if old is not None and old.get("sha256") == h:
                unchanged += 1
                continue
            if old is not None:
                self._drop_doc(d["id"])
                updated += 1
            else:
                added += 1
            pieces = chunk_text(d["text"], max_chars, overlap)
            title = str(d.get("title") or "")
            rows = [{"id": f"{d['id']}#{i}", "doc_id": d["id"], "ord": i, "title": title, "text": t,
                     "updated_at": d.get("updated_at"), "authority": d.get("authority")} for i, t in enumerate(pieces)]
            for r in rows:
                self.chunks[r["id"]] = r
            self.docs[d["id"]] = {"id": d["id"], "title": title, "source": d.get("source") or "", "sha256": h,
                                  "updated_at": d.get("updated_at"), "authority": d.get("authority"),
                                  "links": list(d.get("links") or []), "chunk_ids": [r["id"] for r in rows]}
            # entities: the chunk text; the first chunk also gets a real title (a file name is not a name)
            graph_title = "" if d.get("title_is_filename") or not title or _opens_with_title(rows, title) else f"{title}.\n"
            self.graph.add_document(d["id"], [(r["id"], (graph_title if r["ord"] == 0 else "") + r["text"])
                                              for r in rows], links=d.get("links") or [])
            new_chunks.extend(rows)
        for s in range(0, len(new_chunks), self.batch_size):
            batch = new_chunks[s:s + self.batch_size]
            vecs = self.encoder.encode([self._passage(r["title"], r["text"]) for r in batch], kind="passage")
            self.vectors.upsert([r["id"] for r in batch], vecs, [{"doc_id": r["doc_id"]} for r in batch])
        if added or updated or not self.exists():
            self._rebuild_bm25()
            self.save()
        return {"docs_added": added, "docs_updated": updated, "docs_unchanged": unchanged,
                "docs_skipped": len(notes), "chunks_added": len(new_chunks), "docs_total": len(self.docs),
                "chunks_total": len(self.chunks), "skipped": notes[:MAX_REPORTED_SKIPS], "encoder": self.encoder.id}

    def remove(self, doc_ids: list[str]) -> int:
        """Remove documents (and their chunks, vectors and graph nodes); returns how many existed."""
        n = 0
        for d in doc_ids:
            if str(d) in self.docs:
                self._drop_doc(str(d))
                n += 1
        if n:
            self._rebuild_bm25()
            self.save()
        return n

    def reembed(self) -> int:
        """Re-encode every chunk with the current encoder (after an encoder change); returns the chunk count."""
        self.vectors = LocalVectorStore(self.dir / "vectors", self.encoder.dim, backend=self._backend)
        rows = list(self.chunks.values())
        for s in range(0, len(rows), self.batch_size):
            batch = rows[s:s + self.batch_size]
            vecs = self.encoder.encode([self._passage(r["title"], r["text"]) for r in batch], kind="passage")
            self.vectors.upsert([r["id"] for r in batch], vecs, [{"doc_id": r["doc_id"]} for r in batch])
        self.save()
        return len(rows)

    def _rebuild_bm25(self) -> None:
        self._bm25_ids = list(self.chunks)
        self.bm25 = BM25([tokens(f"{c.get('title') or ''} {c['text']}") for c in self.chunks.values()])
        self._bm25_pos = {cid: i for i, cid in enumerate(self._bm25_ids)}

    def save(self) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        _write_jsonl(self.dir / "docs.jsonl", list(self.docs.values()))
        _write_jsonl(self.dir / "chunks.jsonl", list(self.chunks.values()))
        if self.bm25 is not None:
            _write_json(self.dir / "bm25.json", {"ids": self._bm25_ids, **self.bm25.to_dict()})
        self.vectors.save()
        self.graph.save()
        _write_json(self.dir / "meta.json", {
            "format": FORMAT, "encoder": encoder_config(self.encoder), "docs": len(self.docs),
            "chunks": len(self.chunks), "chunking": {"max_chars": self.max_chars, "overlap": self.overlap},
            "vector_backend": self.vectors.backend, "graph_backend": self.graph.backend})

    # ------------------------------------------------------------------ retrieve

    def retrieve(self, query: str, k: int = 10, graph_hops: int = 2) -> list[dict]:
        """Candidates for `query` from vector, graph and lexical retrieval, fused by reciprocal rank.

        Each retriever contributes up to `k` chunks, so the merged list holds up to 3k candidates, best first:
        `{id, doc_id, title, text, score (cosine), source (vector | graph | lexical | hybrid), sources, rank,
        rrf, lexical_score (BM25), updated_at, authority, graph? {path_length, degree, via}}`."""
        if not self.chunks or not str(query or "").strip():
            return []
        k = max(1, int(k))
        qv = np.asarray(self.encoder.encode([str(query)], kind="query"), dtype=np.float32)[0]
        found: dict[str, dict] = {}

        def add(cid: str, source: str, rank: int) -> dict:
            info = found.setdefault(cid, {"sources": [], "rrf": 0.0})
            if source not in info["sources"]:
                info["sources"].append(source)
                info["rrf"] += 1.0 / (self.rrf_k + rank)
            return info

        for r, h in enumerate(self.vectors.search(qv, k=k)):
            if h.id in self.chunks:
                add(h.id, "vector", r + 1)
        if graph_hops and int(graph_hops) > 0:
            terms = self.graph.entities(str(query))
            graph_hits = [g for g in self.graph.expand(terms, hops=int(graph_hops), limit=k) if g["id"] in self.chunks]
            for r, g in enumerate(graph_hits):
                add(g["id"], "graph", r + 1)["graph"] = {"path_length": g["path_length"], "degree": g["degree"],
                                                         "via": g.get("via")}
        lex = self.bm25.scores(tokens(str(query))) if self.bm25 is not None else np.zeros(0)
        if lex.size:
            order = np.lexsort((np.arange(lex.size), -lex))[:k]
            for r, i in enumerate(order):
                if lex[i] > 0 and self._bm25_ids[i] in self.chunks:
                    add(self._bm25_ids[i], "lexical", r + 1)
        if not found:
            return []
        ids = list(found)
        qn = qv / (np.linalg.norm(qv) or 1.0)
        cos = self.vectors.vectors(ids) @ qn
        out = []
        for cid, s in zip(ids, cos, strict=True):
            c, info = self.chunks[cid], found[cid]
            li = self._bm25_pos.get(cid)
            cand = {"id": cid, "doc_id": c["doc_id"], "title": c.get("title") or "", "text": c["text"],
                    "score": float(s) if np.isfinite(s) else float("nan"),
                    "source": info["sources"][0] if len(info["sources"]) == 1 else "hybrid",
                    "sources": sorted(info["sources"]), "rrf": info["rrf"],
                    "lexical_score": float(lex[li]) if li is not None and li < lex.size else float("nan"),
                    "updated_at": c.get("updated_at"), "authority": c.get("authority")}
            if "graph" in info:
                cand["graph"] = info["graph"]
            out.append(cand)
        out.sort(key=lambda c: (-c["rrf"], -(c["score"] if np.isfinite(c["score"]) else -np.inf), c["id"]))
        for i, c in enumerate(out):
            c["rank"] = i + 1
        return out
