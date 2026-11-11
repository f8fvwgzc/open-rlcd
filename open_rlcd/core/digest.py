"""Canonical JSON and content digests: the identity of a state, a question set, a policy, a receipt.

Two JSON values that mean the same thing get the same digest: object keys are sorted, separators are minimal,
text is UTF-8 (not escaped), and values JSON cannot carry are normalised first (numpy scalars and arrays become
numbers and lists, tuples and sets become lists, non-string keys become strings, anything else its `str`). NaN and
infinities print as `NaN` / `Infinity` / `-Infinity`, and a lone surrogate (which a JSON parser accepts as `\\ud800`)
is encoded as such (`surrogatepass`), so a digest never fails on a state the service accepted — however deeply it
is nested: the normalisation walks the value without recursion.

    digest({"b": 1, "a": [1, 2]}) == digest({"a": [1, 2], "b": 1})   # "sha256:…"

A digest is an identity, not a secret: it says whether two things are the same thing, and nothing about who made
them. Receipts chain their digests (core/receipts.py) so that an edited or deleted line is detectable.
"""
from __future__ import annotations

import hashlib
import json
import sys
from typing import Any

PREFIX = "sha256:"
_SCALARS = (str, bool, int, float)


def plain(obj: Any) -> Any:
    """`obj` as plain JSON-able Python: dicts with string keys, lists, str, int, float, bool, None. Writing
    `plain(x)` as JSON and reading it back gives a value with the same digest as `x`. Iterative: a deeply nested
    value does not hit the recursion limit."""
    out: list = [None]
    stack: list[tuple[Any, Any, Any]] = [(obj, out, 0)]
    while stack:
        src, parent, key = stack.pop()
        while True:
            if src is None or isinstance(src, _SCALARS):
                parent[key] = src
                break
            if isinstance(src, dict):
                d: dict = {}
                parent[key] = d
                for k in src:                      # keys in order: on a collision ("1" and 1) the last one wins
                    d[str(k)] = None
                for k, v in reversed(list(src.items())):
                    stack.append((v, d, str(k)))
                break
            if isinstance(src, (list, tuple)):
                lst: list = [None] * len(src)
                parent[key] = lst
                for i in range(len(src) - 1, -1, -1):
                    stack.append((src[i], lst, i))
                break
            if isinstance(src, (set, frozenset)):
                items = [plain(v) for v in src]
                parent[key] = sorted(items, key=lambda v: json.dumps(v, sort_keys=True, default=str))
                break
            tolist = getattr(src, "tolist", None)       # numpy scalars and arrays
            if callable(tolist):
                src = tolist()
                continue
            parent[key] = str(src)
            break
    return out[0]


def dumps(obj: Any, *, sort_keys: bool = False) -> str:
    """JSON text of a plain value; deep nesting is handled by raising the recursion limit for the encoder only."""
    try:
        return json.dumps(obj, sort_keys=sort_keys, separators=(",", ":"), ensure_ascii=False, allow_nan=True)
    except RecursionError:
        limit = sys.getrecursionlimit()
        sys.setrecursionlimit(max(limit, 100_000))
        try:
            return json.dumps(obj, sort_keys=sort_keys, separators=(",", ":"), ensure_ascii=False, allow_nan=True)
        finally:
            sys.setrecursionlimit(limit)


def encode(text: str) -> bytes:
    """UTF-8 bytes of JSON text; a lone surrogate passes through instead of failing the write."""
    return text.encode("utf-8", "surrogatepass")


def decode(data: bytes) -> str:
    return data.decode("utf-8", "surrogatepass")


def canonical(obj: Any) -> bytes:
    """The canonical UTF-8 JSON bytes of `obj` (sorted keys, no whitespace)."""
    return encode(dumps(plain(obj), sort_keys=True))


def digest_bytes(data: bytes) -> str:
    return PREFIX + hashlib.sha256(data).hexdigest()


def digest(obj: Any) -> str:
    """`sha256:<hex>` of the canonical JSON of `obj`."""
    return digest_bytes(canonical(obj))


def short(d: str | None, n: int = 12) -> str | None:
    """A digest shortened for messages: `sha256:0123456789ab`."""
    if not d:
        return d
    return d[: len(PREFIX) + n] if d.startswith(PREFIX) else d[:n]
