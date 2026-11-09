"""Batteries: named, versioned question sets — a product interface, not a prompt.

A battery is the set of typed questions one decision point asks about one state (an intake battery, a preflight
battery before the LLM, a postflight battery on its output, a tool-call battery). Like any interface it has an
owner, a version, a changelog and a list of consumers (which route or automation reads each answer), so a
seemingly harmless edit cannot silently change an unrelated automation path, and two question definitions are
never mixed under one name in a dashboard.

Declared by the pack (pack.yaml `batteries:`) and by the project (rlcd.yaml `batteries:`; a project entry
replaces the pack's battery of the same name):

    batteries:
      preflight:
        version: "1"                     # change it whenever the meaning changes
        owner: search-platform
        description: before any LLM call, on the request
        changelog: ["1: response_type, route, needs_graph"]
        consumers: {kind: prompt template, route: retriever, multi_hop: graph expansion}
        questions:
          kind:      {type: choice, head: response_type}
          route:     {type: choice, head: route}
          multi_hop: {type: noul, head: needs_graph}

A System One request names a battery instead of listing questions: `{"battery": "preflight", "state": ...}`, or
pinned to a version, `{"battery": "preflight@1"}`. A pinned version that is not the current one is a 422
`battery_version_mismatch` (the caller's thresholds were chosen for the old meaning). The response and the receipt
carry `{name, version, digest}`; the digest covers the question contract (types, heads, criteria, options, paths)
and nothing else, so it changes exactly when a question's meaning can change. `rlcd check` fails when a battery's
digest differs from the one receipts recorded under the same version: a change shipped without a version bump.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from .contract import ContractError, Question, parse_questions, questions_digest

_NAME = re.compile(r"^[A-Za-z0-9_\-]{1,64}$")
_VERSION = re.compile(r"^[A-Za-z0-9_.\-]{1,32}$")


@dataclass(frozen=True)
class Battery:
    name: str
    version: str
    questions_raw: dict
    owner: str | None = None
    description: str = ""
    changelog: tuple[str, ...] = ()
    consumers: dict = field(default_factory=dict)
    source: str = "pack"                      # pack | project

    def questions(self) -> list[Question]:
        return parse_questions(self.questions_raw)

    @property
    def digest(self) -> str:
        return questions_digest(self.questions())

    def ref(self) -> dict:
        return {"name": self.name, "version": self.version, "digest": self.digest}

    def to_dict(self) -> dict:
        return {"name": self.name, "version": self.version, "owner": self.owner, "description": self.description,
                "changelog": list(self.changelog), "consumers": dict(self.consumers), "source": self.source,
                "digest": self.digest,
                "questions": {q.id: q.contract() for q in self.questions()}}


def _battery(name: str, d: Any, source: str) -> Battery:
    if not _NAME.fullmatch(str(name)):
        raise ValueError(f"battery name {name!r}: letters, digits, _ and -, at most 64")
    if not isinstance(d, dict):
        raise ValueError(f"battery {name!r} must be a mapping with version and questions")
    known = {"version", "owner", "description", "changelog", "consumers", "questions"}
    extra = sorted(set(d) - known)
    if extra:
        raise ValueError(f"battery {name!r}: unknown fields {extra}; known: {sorted(known)}")
    version = d.get("version")
    if isinstance(version, (float, bool)):
        # YAML reads an unquoted 1.10 as the float 1.1: another, possibly older, version string
        raise ValueError(f"battery {name!r}: quote the version in YAML (version: \"{version}\" is read as a number)")
    if version is None or not _VERSION.fullmatch(str(version)):
        raise ValueError(f"battery {name!r} needs a version (letters, digits, '.', '_', '-'; e.g. \"1\" or \"2.1\")")
    q = d.get("questions")
    if not isinstance(q, dict) or not q:
        raise ValueError(f"battery {name!r} needs questions: a mapping of question id -> question")
    try:
        parse_questions(q)        # structural validation now; head binding happens per model version
    except ContractError as e:
        raise ValueError(f"battery {name!r}: {e.message}") from None
    changelog = d.get("changelog") or []
    if isinstance(changelog, str):
        changelog = [changelog]
    if not isinstance(changelog, list):
        raise ValueError(f"battery {name!r}: changelog is a list of entries")
    consumers = d.get("consumers") or {}
    if not isinstance(consumers, dict):
        raise ValueError(f"battery {name!r}: consumers maps a question id to who reads its answer")
    unknown = sorted(set(map(str, consumers)) - set(map(str, q)))
    if unknown:
        raise ValueError(f"battery {name!r}: consumers name questions it does not ask: {unknown}")
    return Battery(name=str(name), version=str(version), questions_raw=dict(q), owner=d.get("owner"),
                   description=str(d.get("description") or ""), changelog=tuple(str(c) for c in changelog),
                   consumers={str(k): v for k, v in consumers.items()}, source=source)


def load(pack_section: dict | None, project_section: dict | None) -> dict[str, Battery]:
    """The batteries of a pack overlaid by the project's (same name: the project's replaces the pack's)."""
    out: dict[str, Battery] = {}
    for source, section in (("pack", pack_section), ("project", project_section)):
        if not section:
            continue
        if not isinstance(section, dict):
            raise ValueError("batteries must be a mapping of name -> battery")
        for name, d in section.items():
            out[str(name)] = _battery(str(name), d, source)
    return out


def parse_ref(ref: Any) -> tuple[str, str | None]:
    """`"preflight"` -> ("preflight", None); `"preflight@2"` -> ("preflight", "2")."""
    if not isinstance(ref, str) or not ref.strip():
        raise ContractError("battery must be a string: a battery name, optionally pinned as name@version",
                            field="battery")
    name, _, version = ref.strip().partition("@")
    if not _NAME.fullmatch(name) or (version and not _VERSION.fullmatch(version)):
        raise ContractError(f"battery {ref!r} is not a battery name (optionally name@version)", field="battery")
    return name, version or None


def resolve(ref: Any, batteries: dict[str, Battery]) -> Battery:
    """The battery a request names. Unknown -> 422 listing the batteries; a pinned version that is not the
    current one -> 422 battery_version_mismatch."""
    name, pinned = parse_ref(ref)
    if name not in batteries:
        raise ContractError(f"no battery named {name!r}", kind="unknown_battery",
                            available=sorted(batteries))
    b = batteries[name]
    if pinned is not None and pinned != b.version:
        raise ContractError(f"battery {name} is version {b.version}, the request pinned {pinned}: the questions may "
                            f"mean something else now. Re-check your thresholds against the new version (its "
                            f"changelog is at GET /v1/batteries), then pin {name}@{b.version}",
                            kind="battery_version_mismatch", battery=name, expected=pinned, current=b.version,
                            changelog=list(b.changelog))
    return b
