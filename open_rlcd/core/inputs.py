"""Question inputs: which part of one shared state a head reads.

A System One request carries one state and many questions, and each head expects its own view (the documents
`answerable` head reads `{query, evidence}`, `grounded` reads `{query, evidence, answer}`). A question can say
where its head's inputs live in a larger, named state, so one evidence packet serves every question:

    "path":   "rag"                                     the head reads state["rag"] as its whole state
    "inputs": {"query": "request.text",                 the head's state is built field by field from paths
               "evidence": "retrieval.passages",
               "answer": "llm_proposal.text"}
    both:     `inputs` paths are read below `path`

A path is a dotted list of keys; a segment that is a non-negative integer indexes an array (`items.0.text`).
Keys that themselves contain a dot cannot be addressed (rename them in the state builder).

A path that does not resolve fails closed: a 422 `missing_evidence` naming the question and the path. The head is
never asked to judge a state with a guessed or empty field in place of the one the question named; whether a
present-but-empty field is acceptable is the head's own validation (the documents provider requires a non-empty
query, for example). Values are passed through unchanged, and the paths a question read are part of its contract
(they are in the question digest of a receipt).
"""
from __future__ import annotations

import re
from typing import Any

from .contract import ContractError

MAX_PATH_DEPTH = 32
MAX_PATH_CHARS = 512
MAX_INPUTS = 64
_TARGET = re.compile(r"^[A-Za-z_][A-Za-z0-9_\-]{0,63}$")


def parse_path(path: Any, *, question: str | None = None, where: str = "path") -> tuple[str | int, ...]:
    """`"a.b.0"` -> `("a", "b", 0)`; `""` or `"."` -> `()` (the whole state). 422 on a malformed path."""
    if not isinstance(path, str):
        raise ContractError(f"question '{question}': {where} must be a string like 'request.text'",
                            question=question, field=where)
    p = path.strip()
    if p in ("", "."):
        return ()
    if len(p) > MAX_PATH_CHARS:
        raise ContractError(f"question '{question}': {where} is longer than {MAX_PATH_CHARS} characters",
                            question=question, field=where)
    parts = p.split(".")
    if len(parts) > MAX_PATH_DEPTH:
        raise ContractError(f"question '{question}': {where} {path!r} is deeper than {MAX_PATH_DEPTH} levels",
                            question=question, field=where)
    if any(not s for s in parts):
        raise ContractError(f"question '{question}': {where} {path!r} has an empty segment", question=question,
                            field=where)
    # a canonical non-negative integer ("0", "7", not "007") indexes an array; any other segment is an object key
    return tuple(int(s) if s.isascii() and s.isdigit() and str(int(s)) == s else s for s in parts)


def resolve(state: Any, parts: tuple[str | int, ...], *, question: str | None = None, path: str = "") -> Any:
    """The value at `parts` in `state`; 422 `missing_evidence` when a segment does not resolve."""
    cur = state
    for i, seg in enumerate(parts):
        here = ".".join(str(s) for s in parts[: i + 1])
        if isinstance(seg, int) and isinstance(cur, list):
            if seg >= len(cur):
                raise _missing(question, path, f"{here} is out of range (the array has {len(cur)} items)")
            cur = cur[seg]
        elif isinstance(cur, dict):
            key = str(seg)
            if key not in cur:
                raise _missing(question, path, f"{here} is not in the state")
            cur = cur[key]
        else:
            raise _missing(question, path, f"{here} cannot be read: the value above it is a "
                                           f"{type(cur).__name__}, not an object or array")
    return cur


def _missing(question: str | None, path: str, why: str) -> ContractError:
    return ContractError(f"question '{question}': {why}; the question names it as evidence, so it is not answered "
                         f"from a guess (send the field, or fix the question's path)", kind="missing_evidence",
                         question=question, path=path)


class InputSpec:
    """A question's validated `path` / `inputs` (both optional). `project(state)` builds the head's state."""

    def __init__(self, path: Any = None, inputs: Any = None, *, question: str | None = None):
        self.question = question
        self.path: str | None = None
        self._root: tuple[str | int, ...] = ()
        if path is not None:
            self._root = parse_path(path, question=question, where="path")
            self.path = ".".join(str(s) for s in self._root)
        self.inputs: dict[str, str] | None = None
        self._fields: dict[str, tuple[str | int, ...]] = {}
        if inputs is not None:
            if not isinstance(inputs, dict) or not inputs:
                raise ContractError(f"question '{question}': inputs must be a non-empty object mapping each field "
                                    f"the head reads to a path in the state", question=question, field="inputs")
            if len(inputs) > MAX_INPUTS:
                raise ContractError(f"question '{question}': at most {MAX_INPUTS} inputs", question=question,
                                    field="inputs", limit=MAX_INPUTS)
            norm: dict[str, str] = {}
            for target, src in inputs.items():
                t = str(target)
                if not _TARGET.fullmatch(t):
                    raise ContractError(f"question '{question}': input name {t!r} must be an identifier "
                                        f"(letters, digits, _ and -)", question=question, field="inputs")
                parts = parse_path(src, question=question, where=f"inputs.{t}")
                self._fields[t] = parts
                norm[t] = ".".join(str(s) for s in parts)
            self.inputs = norm

    @property
    def active(self) -> bool:
        return self.path is not None or self.inputs is not None

    def to_dict(self) -> dict:
        """The part of the question contract this spec adds (for digests and receipts)."""
        out: dict = {}
        if self.path is not None:
            out["path"] = self.path
        if self.inputs is not None:
            out["inputs"] = dict(sorted(self.inputs.items()))
        return out

    def project(self, state: Any) -> Any:
        """The state the head reads. Without a spec, the request state itself."""
        if not self.active:
            return state
        root = resolve(state, self._root, question=self.question, path=self.path or "") if self._root else state
        if self.inputs is None:
            return root
        return {t: resolve(root, parts, question=self.question, path=self.inputs[t])
                for t, parts in self._fields.items()}
