"""Candidate retrieval for the documents pack: `Corpus.retrieve` plus one step along explicit document links.

`candidates(corpus, query, k)` is the single retrieval call used by `decide()`, the label tasks and the synthetic
generator, so the candidates the heads learn from look exactly like the ones they judge at serve time.

Why the link step: a multi-hop question ("who leads the team that owns X?") needs the document X's page links
to. The entity graph ranks such pages first among its 2-hop rows, but all 1-hop rows (every chunk that names X,
news items included) come before them and only `k` graph hits are kept. Explicit links (Markdown / HTML
hyperlinks, `links` of JSON records) are a far more specific signal, so the documents linked from the top
`LINK_FROM` candidates and from every 1-hop graph hit (a chunk that names a query entity, wherever fusion ranked
it) are added (at most `MAX_LINKED` chunks, in candidate order), scored by the same cosine as every candidate and
marked as graph hits at path length 2. It adds candidates, never removes or reorders the retriever's.

What this does not do: it does not follow links transitively, and it does not read links out of chunk text (only
what the ingest recorded per document).
"""
from __future__ import annotations

from typing import Any

import numpy as np

from .tasks import clean_candidate

LINK_FROM = 3
MAX_LINKED = 4


def candidates(corpus: Any, query: str, k: int) -> list[dict]:
    """Fused candidates for `query` (best first, `rank` 1..n), plus chunks of documents the top ones link to."""
    cands = [clean_candidate(c) for c in corpus.retrieve(query, k=k)]
    if not cands:
        return cands
    have = {c["id"] for c in cands}
    linked: list[str] = []
    sources = [c for i, c in enumerate(cands) if i < LINK_FROM or (c.get("graph") or {}).get("path_length") == 1]
    for c in sources:
        doc = corpus.doc(c.get("doc_id")) if c.get("doc_id") is not None else None
        for target in (doc or {}).get("links") or []:
            for cid in (corpus.doc(target) or {}).get("chunk_ids") or []:
                if cid not in have and cid not in linked:
                    linked.append(cid)
    linked = linked[:MAX_LINKED]
    if not linked:
        return cands
    qv = np.asarray(corpus.encoder.encode([str(query)], kind="query"), np.float32)[0]
    qn = qv / (np.linalg.norm(qv) or 1.0)
    vecs = corpus.vectors.vectors(linked)
    for cid, v in zip(linked, vecs, strict=True):
        ch = corpus.chunk(cid)
        if ch is None:
            continue
        s = float(v @ qn)
        cands.append({"id": cid, "doc_id": ch["doc_id"], "title": ch.get("title") or "", "text": ch["text"],
                      "score": s if np.isfinite(s) else None, "source": "graph", "sources": ["graph"], "rrf": 0.0,
                      "lexical_score": None, "updated_at": ch.get("updated_at"), "authority": ch.get("authority"),
                      "graph": {"path_length": 2,
                                "degree": corpus.graph.chunk_degree(cid) if corpus.graph is not None else None},
                      "rank": len(cands) + 1})
    return cands
