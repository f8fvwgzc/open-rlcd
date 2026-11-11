"""Regression cases: incidents turned into checks that every later model version must pass.

The guide (X.H, VIII.H, X.C, XI.C): every incident should leave at least one durable artifact, and the first is a
regression case: a state, the question it was asked, and what the answer must be. `rlcd regress` runs the cases
against a version before it is promoted; `rlcd replay` (core/replay.py) compares two versions on logged traffic.

Cases are JSON lines in the project, `<project root>/regressions.jsonl` (rlcd.yaml `regressions: path/to/file`
overrides it), so they are reviewed and versioned with the project rather than kept with the artifacts in
RLCD_HOME. Two kinds:

    head-level     {"kind": "head", "head": "route", "state": {...}, "stakes"?: "high", "strict"?: false,
                    "expect": {"class"?, "class_in"?, "tier"?, "tier_at_most"?, "tier_at_least"?,
                               "p_at_least"?, "p_at_most"?}}
                   one head on one state through `service.head_decision`. `class` is the head's most probable class
                   (for a gate head too, whose decision always commits to `true`: there `class` says which side of
                   0.5 the head is on, and the gate is the tier). `p` is the probability of the expected class, or,
                   without `class`, of the class the decision commits to (P(true) for a gate head). Tiers are the
                   version's policy names; "abstain" names the abstain action whatever the policy calls it.
    request-level  {"kind": "decide" | "evaluate" | "systemone", "request": {...},
                    "expect": {"action"?, "action_at_most"?, "action_at_least"?, "next"?, "verdict"?,
                               "answers"?: {question id: {"choice"? | "score"? | "noul_at_least"? | "noul_at_most"?}}}}
                   the whole request through `service.decide` / `evaluate` / `systemone` (pack logic included), with
                   `model` set to the version under test. `answers` is for systemone cases, the others for decide and
                   evaluate (`next` is the documents plan's next step, `verdict` evaluate's ship / review / reject).

Either kind may expect a refusal instead: {"expect": {"error": "missing_evidence"}} passes when the call is refused
with that error kind (an incomplete or malformed state must stay refused). Optional fields: `note`, `incident`, and
`source` (the decision or feedback row the case was copied from). `add_case` assigns `id` and `added_at`.

Cases judge the model and its policy as trained, not the deployment: they run with rollout mode `full`, the kill
switch on, and the Bernoulli gate on each version's own holdout table instead of the live ledger (`as_trained`). So
switching automation off during an incident cannot make a case pass, the outcomes posted since cannot make it flip,
and a case gives the same result wherever the version is installed. Every case runs inside `service.dry_run()`:
nothing is logged.

What it does not do: it does not write cases by itself (an incident review does: `rlcd case add --from-decision`,
or feedback tagged "regression"), it does not decide what to do about a failing case, and an untrained head can
neither pass nor fail one (that is an error, not a pass). A case copies a state into the project directory: logged
states were redacted when they were logged (serve.redact), but they can still hold customer data — review the file
before committing it.
"""
from __future__ import annotations

import copy
import json
import os
import re
import threading
import uuid
from collections import Counter
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import ValidationError

from .contract import STAKES_NAME, ContractError, DecideRequest, EvaluateRequest, SystemOneRequest
from .replay import entry_head, holdout_table
from .rollout import Rollout, SwitchState
from .training import now_iso

try:  # POSIX: several processes appending to one cases file still write whole lines
    import fcntl
except ImportError:  # pragma: no cover
    fcntl = None  # type: ignore[assignment]

DEFAULT_FILE = "regressions.jsonl"
ABSTAIN = "abstain"            # names tier 0 in a case, whatever the policy calls its abstain action
HEAD_EXPECT = ("class", "class_in", "tier", "tier_at_most", "tier_at_least", "p_at_least", "p_at_most", "error")
REQUESTS: dict[str, Any] = {"decide": DecideRequest, "evaluate": EvaluateRequest, "systemone": SystemOneRequest}
_DECISION_EXPECT = ("action", "action_at_most", "action_at_least", "next", "verdict", "error")
REQUEST_EXPECT = {"decide": _DECISION_EXPECT, "evaluate": _DECISION_EXPECT, "systemone": ("answers", "error")}
ANSWER_EXPECT = ("choice", "score", "noul_at_least", "noul_at_most")
_TEXT = ("class", "tier", "tier_at_most", "tier_at_least", "action", "action_at_most", "action_at_least", "next",
         "verdict", "error")
_LOCK = threading.Lock()


class _Unjudgeable(Exception):
    """The case cannot be judged against this version (an untrained head, a tier or class it does not have)."""


def _bad(message: str, **details: Any) -> ContractError:
    return ContractError(message, kind="bad_case", **details)


# ----------------------------------------------------------------------------- the cases file

def cases_path(project: Any) -> Path:
    """Where the project's regression cases live: rlcd.yaml `regressions:` (relative to the project root), else
    `<project root>/regressions.jsonl`."""
    raw = project.get("regressions")
    if raw is None or raw == "":
        raw = DEFAULT_FILE
    if not isinstance(raw, (str, os.PathLike)):
        raise ContractError("rlcd.yaml `regressions:` is the path of the regression cases file (JSON lines)",
                            kind="bad_config", status=500)
    return project.resolve_path(raw)


def load_cases(path: str | os.PathLike) -> list[dict]:
    """The cases of a file, in order ([] when it does not exist). A line that is not a valid case is kept as
    {id, line, invalid: reason}, so `run_cases` reports it as an error instead of silently dropping a check."""
    p = Path(path)
    if not p.exists():
        return []
    out: list[dict] = []
    for n, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
        except ValueError as e:
            out.append({"id": f"line-{n}", "line": n, "invalid": f"{p.name} line {n} is not JSON ({e})"})
            continue
        try:
            case = validate_case(raw)
        except Exception as e:
            why = e.message if isinstance(e, ContractError) else f"{type(e).__name__}: {e}"
            rid = raw.get("id") if isinstance(raw, dict) and isinstance(raw.get("id"), str) and raw["id"] else None
            out.append({"id": rid or f"line-{n}", "line": n, "invalid": f"{p.name} line {n}: {why}"})
            continue
        if not case.get("id"):
            case["id"] = f"line-{n}"
        out.append(case)
    return out


def add_case(path: str | os.PathLike, case: dict) -> dict:
    """Validate `case`, give it a fresh `id` ("case-<hex>") and `added_at`, append it to the file (created with its
    directory when missing) and return it as stored. Any `id` / `added_at` the case carried is replaced."""
    case = validate_case(case)
    rest = {k: v for k, v in case.items() if k not in ("id", "added_at", "kind", "line", "invalid")}
    rec = {"id": f"case-{uuid.uuid4().hex[:12]}", "added_at": now_iso(), "kind": case["kind"], **rest}
    data = (json.dumps(rec, ensure_ascii=False, separators=(",", ":"), default=str) + "\n").encode("utf-8")
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with _LOCK, open(p, "a+b") as f:
        if fcntl is not None:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            f.seek(0, os.SEEK_END)
            if f.tell() > 0:       # a hand-edited file without a final newline must not swallow the new case
                f.seek(-1, os.SEEK_END)
                if f.read(1) != b"\n":
                    data = b"\n" + data
            f.write(data)
            f.flush()
        finally:
            if fcntl is not None:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
    return rec


# ----------------------------------------------------------------------------- validation

def validate_case(case: Any) -> dict:
    """Check a case's shape (not whether any version passes it). Returns a copy with `kind` set ("head" when
    absent); raises ContractError kind `bad_case` naming the problem."""
    if not isinstance(case, dict):
        raise _bad("a regression case is a JSON object")
    out = dict(case)
    kind = out.get("kind") or "head"
    if not isinstance(kind, str):
        raise _bad(f"case kind must be 'head' or one of {sorted(REQUESTS)}, got {kind!r}")
    if kind == "head":
        _check_head_case(out)
    elif kind in REQUESTS:
        _check_request_case(out, kind)
    else:
        raise _bad(f"case kind must be 'head' or one of {sorted(REQUESTS)}, got {kind!r}")
    out["kind"] = kind
    for k in ("id", "note", "incident", "added_at"):
        if out.get(k) is not None and not isinstance(out[k], str):
            raise _bad(f"{k} must be a string")
    if out.get("source") is not None and not isinstance(out["source"], dict):
        raise _bad("source must be an object (where the case came from)")
    return out


def _check_head_case(c: dict) -> None:
    head = c.get("head")
    if not isinstance(head, str) or not head.strip():
        raise _bad("a head-level case names its head (`head`)")
    if "request" in c:
        raise _bad("a head-level case carries a `state`; a whole request needs kind decide, evaluate or systemone")
    if c.get("state") is None:
        raise _bad(f"case for head {head!r}: `state` is required (the state the head judges)")
    stakes = c.get("stakes")
    if stakes is not None and (not isinstance(stakes, str) or not re.fullmatch(STAKES_NAME, stakes)):
        raise _bad(f"stakes must be a stakes level name, got {stakes!r}")
    if c.get("strict") is not None and not isinstance(c["strict"], bool):
        raise _bad("strict must be true or false")
    _check_expect(c.get("expect"), HEAD_EXPECT)


def _check_request_case(c: dict, kind: str) -> None:
    req = c.get("request")
    if not isinstance(req, dict):
        raise _bad(f"a {kind} case carries the JSON body of the call in `request`")
    if "state" in c or "head" in c:
        raise _bad(f"a {kind} case keeps its state inside `request`; a top-level `head` and `state` make a "
                   f"head-level case")
    if req.get("explore"):
        raise _bad("a regression case must be deterministic: explore (Thompson sampling) is not allowed")
    try:
        REQUESTS[kind].model_validate(req)
    except ValidationError as e:
        raise _bad(f"request is not a valid {kind} request: {_pydantic(e)}") from None
    _check_expect(c.get("expect"), REQUEST_EXPECT[kind])


def _check_expect(expect: Any, allowed: Sequence[str]) -> None:
    if not isinstance(expect, dict) or not expect:
        raise _bad("expect must be a non-empty object: what the answer must be")
    unknown = sorted(str(k) for k in expect if k not in allowed)
    if unknown:
        raise _bad(f"expect has unknown keys {unknown}; this kind of case takes {list(allowed)}")
    if "error" in expect and len(expect) > 1:
        raise _bad("expect.error stands alone: a refused call has no answer to check")
    for k in _TEXT:
        if k in expect and (not isinstance(expect[k], str) or not expect[k].strip()):
            raise _bad(f"expect.{k} must be a non-empty string")
    if "class_in" in expect:
        v = expect["class_in"]
        if not isinstance(v, list) or not v or not all(isinstance(x, str) and x for x in v):
            raise _bad("expect.class_in must be a non-empty array of class names")
    _check_bounds(expect, "p_at_least", "p_at_most", "expect")
    if "answers" in expect:
        answers = expect["answers"]
        if not isinstance(answers, dict) or not answers:
            raise _bad("expect.answers maps a question id to what its answer must be")
        for qid, a in answers.items():
            where = f"expect.answers.{qid}"
            if not isinstance(a, dict) or not a:
                raise _bad(f"{where} must be a non-empty object")
            bad = sorted(str(k) for k in a if k not in ANSWER_EXPECT)
            if bad:
                raise _bad(f"{where} has unknown keys {bad}; it takes {list(ANSWER_EXPECT)}")
            if "choice" in a and (not isinstance(a["choice"], str) or not a["choice"]):
                raise _bad(f"{where}.choice must be an option name")
            if "score" in a and (isinstance(a["score"], bool) or not isinstance(a["score"], int) or a["score"] < 0):
                raise _bad(f"{where}.score must be a level index (0, 1, ...)")
            _check_bounds(a, "noul_at_least", "noul_at_most", where)


def _check_bounds(d: dict, lo: str, hi: str, where: str) -> None:
    for k in (lo, hi):
        if k in d:
            v = d[k]
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not 0 <= float(v) <= 1:
                raise _bad(f"{where}.{k} must be a probability in [0, 1]")
    if lo in d and hi in d and float(d[lo]) > float(d[hi]):
        raise _bad(f"{where}: {lo} {d[lo]} is above {hi} {d[hi]}: no answer could pass")


def _pydantic(e: ValidationError) -> str:
    parts = []
    for err in e.errors()[:3]:
        loc = ".".join(str(x) for x in err.get("loc", ()))
        parts.append(f"{loc}: {err.get('msg')}" if loc else str(err.get("msg")))
    return "; ".join(parts)


def parse_expect(items: Iterable[str]) -> dict:
    """Head-level expectations as the CLI takes them (`--expect`, repeatable): class=red, class_in=red,green,
    tier=confirm, tier_at_most=confirm, tier_at_least=confirm, p_at_least=0.8, p_at_most=0.2."""
    out: dict[str, Any] = {}
    for item in items:
        k, sep, v = str(item).partition("=")
        k, v = k.strip(), v.strip()
        if not sep or not k or not v:
            raise _bad(f"--expect takes key=value (e.g. class=red), got {item!r}")
        if k not in HEAD_EXPECT or k == "error":
            raise _bad(f"--expect: unknown key {k!r}; keys: {[x for x in HEAD_EXPECT if x != 'error']}")
        if k == "class_in":
            out[k] = [x.strip() for x in v.split(",") if x.strip()]
        elif k in ("p_at_least", "p_at_most"):
            try:
                out[k] = float(v)
            except ValueError:
                raise _bad(f"--expect {k} takes a probability, got {v!r}") from None
        else:
            out[k] = v
    return out


# ----------------------------------------------------------------------------- cases from the logs

def _logged_entry(row: dict, decision_id: str, head: str | None) -> tuple[str, Any]:
    """(key, entry) of `head` in a decision record: a head name, or a systemone receipt's question id."""
    heads = row.get("heads") if isinstance(row.get("heads"), dict) else {}
    if head is None:
        if len(heads) != 1:
            raise ContractError(f"decision {decision_id} involved heads {list(heads)}: name the one the case is for",
                                heads=list(heads))
        key = next(iter(heads))
        return str(key), heads[key]
    if head in heads:
        return head, heads[head]
    keys = [str(k) for k, h in heads.items() if isinstance(h, dict) and h.get("head") == head]
    if len(keys) > 1:
        raise ContractError(f"decision {decision_id} asked head {head!r} in several questions {keys}: name the "
                            f"question id", heads=list(heads))
    if not keys:
        raise ContractError(f"decision {decision_id} has no head {head!r}; its heads: {list(heads)}",
                            kind="unknown_head", heads=list(heads))
    return keys[0], heads[keys[0]]


def _canonical(spec: Any, label: Any) -> str:
    """A label as the head names it (yes -> true, a score level's description -> its name)."""
    try:
        return spec.labels[spec.label_index(label)]
    except (ValueError, IndexError, TypeError):
        raise _bad(f"{label!r} is not a class of head {spec.name!r}: {list(spec.labels)}") from None


def _canonical_expect(service: Any, head: str, expect: Any) -> dict:
    """Class names checked against the head as the latest version defines it (a typo is refused when the case is
    made, not discovered when it fails)."""
    if not isinstance(expect, dict) or not expect:
        raise _bad("expect must be a non-empty object: what the answer must be")
    out = dict(expect)
    try:
        spec = service.specs(service.resolve(None)).get(head)
    except Exception:   # the pack cannot load here: keep the names as given, run_cases checks them per version
        spec = None
    if spec is None:
        return out
    if isinstance(out.get("class"), str):
        out["class"] = _canonical(spec, out["class"])
    if isinstance(out.get("class_in"), list):
        out["class_in"] = [_canonical(spec, c) for c in out["class_in"]]
    return out


def _no_state(where: str, entry: Any) -> ContractError:
    pruned = isinstance(entry, dict) and entry.get("state_digest") and "state" not in entry
    why = ("its state was pruned (`rlcd receipts prune`)" if pruned else
           "it was logged without a state (serve.log_states: false)")
    return ContractError(f"{where}: {why}; a regression case needs the state the head judged", kind="no_state")


def case_from_decision(service: Any, decision_id: str, *, head: str | None, expect: dict, note: str | None = None,
                       incident: str | None = None) -> dict:
    """A head-level case built from a logged decision: the state the head judged (as logged, so redacted as the log
    is), the stakes it was decided at, and `expect`. `head` may be a head name or the question id of a systemone
    receipt (None: the decision's only head). Not stored: pass it to `add_case`. ContractError when the decision,
    the head or its logged state is missing."""
    row = service.logs.decision(decision_id)
    if row is None:
        raise ContractError(f"unknown decision_id {decision_id!r}", kind="unknown_decision")
    key, entry = _logged_entry(row, decision_id, head)
    name = entry_head(key, entry)
    if not isinstance(entry, dict) or entry.get("state") is None:
        raise _no_state(f"decision {decision_id}, head {name!r}", entry)
    case: dict[str, Any] = {"kind": "head", "head": name, "state": copy.deepcopy(entry["state"])}
    if isinstance(entry.get("stakes"), str) and entry["stakes"]:
        case["stakes"] = entry["stakes"]
    case["expect"] = _canonical_expect(service, name, expect)
    if note:
        case["note"] = str(note)
    if incident:
        case["incident"] = str(incident)
    src: dict[str, Any] = {"decision_id": str(decision_id), "kind": row.get("kind"), "model": row.get("model"),
                           "created_at": row.get("created_at"),
                           "logged": {k: entry.get(k) for k in ("class", "p", "tier")}}
    if key != name:
        src["question"] = key
    case["source"] = src
    return validate_case(case)


def case_from_feedback(service: Any, feedback_row: dict) -> dict:
    """A head-level case from a feedback row (a reviewer's correction): its state (else the logged state of its
    decision), the stakes the decision was made at, and the corrected label as the expected class. Not stored:
    pass it to `add_case`. ContractError when no state can be found."""
    if not isinstance(feedback_row, dict):
        raise _bad("a feedback row is an object")
    head, label = feedback_row.get("head"), feedback_row.get("label")
    if not isinstance(head, str) or not head or label is None:
        raise _bad("a feedback row names its head and its label")
    did = feedback_row.get("decision_id")
    row = service.logs.decision(did) if did else None
    key, entry = None, None
    if row is not None:
        heads = row.get("heads") if isinstance(row.get("heads"), dict) else {}
        keys = [str(k) for k, h in heads.items() if isinstance(h, dict) and entry_head(str(k), h) == head]
        if len(keys) == 1:
            key, entry = keys[0], heads[keys[0]]
    state = feedback_row.get("state")
    if state is None and entry is not None:
        state = entry.get("state")
    if state is None:
        raise _no_state(f"feedback {feedback_row.get('id')!r} for head {head!r}", entry)
    case: dict[str, Any] = {"kind": "head", "head": head, "state": copy.deepcopy(state)}
    if entry is not None and isinstance(entry.get("stakes"), str) and entry["stakes"]:
        case["stakes"] = entry["stakes"]
    case["expect"] = _canonical_expect(service, head, {"class": str(label)})
    if isinstance(feedback_row.get("note"), str) and feedback_row["note"]:
        case["note"] = feedback_row["note"]
    src: dict[str, Any] = {"feedback_id": feedback_row.get("id"), "decision_id": did,
                           "source": feedback_row.get("source"), "received_at": feedback_row.get("received_at")}
    for k in ("tags", "missing_evidence"):
        if feedback_row.get(k):
            src[k] = feedback_row[k]
    if row is not None:
        src["model"] = row.get("model")
    if entry is not None:
        src["logged"] = {k: entry.get(k) for k in ("class", "p", "tier")}
        if key != head:
            src["question"] = key
    case["source"] = src
    return validate_case(case)


# ----------------------------------------------------------------------------- running cases

class _SwitchOn:
    """The kill switch as a regression case sees it: on (a case judges the model, not today's switch)."""

    def state(self) -> SwitchState:
        return SwitchState()

    def cap(self) -> tuple[int, str | None]:
        return 2, None

    def history(self, limit: int = 50) -> list[dict]:
        return []


def _full_rollout() -> Rollout:
    return Rollout()


def _own_holdout(version: str, head_name: str, head: Any) -> dict | None:
    return holdout_table(head)


def as_trained(service: Any) -> Any:
    """A view of `service` that decides as each version was trained: rollout mode `full`, the kill switch on, and
    the Bernoulli gate on the version's own holdout table instead of the live ledger. A shallow copy: it shares the
    registry, logs, caches and the dry-run flag with `service`; only those three lookups differ."""
    view = copy.copy(service)
    view.rollout = _full_rollout
    view.switch = _SwitchOn()
    view.gate_table = _own_holdout
    return view


def run_cases(service: Any, cases: Iterable[Any], *, model: str | None = None) -> dict:
    """Run every case against `model` (default: rlcd-latest), as trained (see `as_trained`), inside a dry run.

    Returns {model, total, passed, failed, errors, results: [{id, ok, status, got, expect, why, note?,
    incident?}]}. `status` is passed / failed / error: an error is a case that could not be judged (invalid, a
    head the version lacks or never trained, a refusal the case did not expect, a tier name the policy does not
    have) and counts against the version like a failure."""
    version = service.resolve(model)
    view = as_trained(service)
    results: list[dict] = []
    with service.dry_run(), view.dry_run():
        for i, case in enumerate(cases):
            results.append(_run(view, version, case, i))
    n = Counter(r["status"] for r in results)
    return {"model": version, "total": len(results), "passed": n["passed"], "failed": n["failed"],
            "errors": n["error"], "results": results}


def _outcome(ok: bool, got: Any, expect: Any, why: str | None) -> dict:
    return {"ok": ok, "status": "passed" if ok else "failed", "got": got, "expect": expect, "why": why}


def _error(expect: Any, why: str, got: Any = None) -> dict:
    return {"ok": False, "status": "error", "got": got, "expect": expect, "why": why}


def _run(view: Any, version: str, case: Any, i: int) -> dict:
    res: dict[str, Any] = {"id": f"#{i + 1}"}
    if isinstance(case, dict):
        if isinstance(case.get("id"), str) and case["id"]:
            res["id"] = case["id"]
        for k in ("note", "incident"):
            if isinstance(case.get(k), str) and case[k]:
                res[k] = case[k]
        if case.get("invalid"):
            return {**res, **_error(case.get("expect"), str(case["invalid"]))}
    try:
        checked = validate_case(case)
    except Exception as e:   # one malformed case never stops the suite
        why = e.message if isinstance(e, ContractError) else f"{type(e).__name__}: {e}"
        return {**res, **_error(case.get("expect") if isinstance(case, dict) else None, why)}
    case = checked
    expect = case["expect"]
    try:
        if case["kind"] == "head":
            got, problems = _head_case(view, version, case)
        else:
            got, problems = _request_case(view, version, case)
    except _Unjudgeable as e:
        return {**res, **_error(expect, str(e))}
    except ContractError as e:
        got = {"error": e.kind, "message": e.message}
        want = expect.get("error")
        if want is None:
            return {**res, **_error(expect, f"the call was refused ({e.kind}): {e.message}", got)}
        ok = e.kind == want
        return {**res, **_outcome(ok, got, expect, None if ok else f"refused with {e.kind} ({e.message}); "
                                                                   f"expected {want}")}
    except Exception as e:   # one broken case never stops the suite
        return {**res, **_error(expect, f"{type(e).__name__}: {e}")}
    if "error" in expect:
        problems = [f"expected a refusal ({expect['error']}), but the call was answered"]
    return {**res, **_outcome(not problems, got, expect, "; ".join(problems) or None)}


def _tier_names(view: Any, version: str) -> tuple[str, ...]:
    cfg = view.policy(version)
    return tuple(cfg.tier_name(i) for i in range(3))


def _tier_rank(name: str, names: Sequence[str]) -> int:
    if name == ABSTAIN:
        return 0
    if name in names:
        return list(names).index(name)
    raise _Unjudgeable(f"{name!r} is not a tier of this version's policy: {list(names)} (or {ABSTAIN!r})")


def _tier_problems(what: str, got: str, expect: dict, names: Sequence[str]) -> list[str]:
    """`what` = tier (head-level) or action (request-level): equal / at most / at least, in the policy's order."""
    keys = {what: "", f"{what}_at_most": "at most ", f"{what}_at_least": "at least "}
    wanted = {k: _tier_rank(expect[k], names) for k in keys if k in expect}
    if not wanted:
        return []
    if got not in names:
        return [f"{what} {got!r} is not a tier of this version's policy {list(names)}"]
    g = list(names).index(got)
    out = []
    for k, w in wanted.items():
        bad = g != w if k == what else g > w if k.endswith("_at_most") else g < w
        if bad:
            out.append(f"{what} {got}, expected {keys[k]}{expect[k]}")
    return out


def _class(spec: Any, version: str, label: Any) -> str:
    try:
        return spec.labels[spec.label_index(label)]
    except (ValueError, IndexError, TypeError):
        raise _Unjudgeable(f"{label!r} is not a class of head {spec.name!r} in {version}: "
                           f"{list(spec.labels)}") from None


def _head_case(view: Any, version: str, case: dict) -> tuple[dict, list[str]]:
    spec = view.spec(case["head"], version)
    state = case["state"]
    warnings: list[str] = []
    missing: list[str] = []
    P, calibrated = view.probabilities(version, spec, [state], warnings, missing_out=missing)
    if not calibrated:
        raise _Unjudgeable(f"head {spec.name!r} is not trained in {version}: its answer is the uninformed prior, "
                           f"which can neither pass nor fail a case")
    probs = np.asarray(P[0], float)
    r = view.head_decision(version, spec, state, stakes=case.get("stakes"), warnings=warnings, probs=probs,
                           calibrated=True, explain=False, missing_inputs=missing, strict=bool(case.get("strict")))
    d = r["decision"]
    expect = case["expect"]
    labels = list(spec.labels)
    top = labels[int(np.argmax(probs))]
    want = _class(spec, version, expect["class"]) if "class" in expect else None
    p = float(probs[labels.index(want)]) if want is not None else float(d["p"])
    got: dict[str, Any] = {"class": top, "p": round(p, 6), "tier": d["tier"], "stakes": d["stakes"],
                           "probabilities": {lab: round(float(x), 6) for lab, x in zip(labels, probs, strict=True)}}
    if missing:
        got["missing_inputs"] = missing[:10]
    problems: list[str] = []
    if want is not None and top != want:
        problems.append(f"class {top} (p {float(probs[labels.index(top)]):.3f}), expected {want} (p {p:.3f})")
    if "class_in" in expect:
        allowed = [_class(spec, version, c) for c in expect["class_in"]]
        if top not in allowed:
            problems.append(f"class {top}, expected one of {allowed}")
    problems += _tier_problems("tier", d["tier"], expect, _tier_names(view, version))
    if "p_at_least" in expect and p < float(expect["p_at_least"]):
        problems.append(f"p {p:.3f} is below {expect['p_at_least']}")
    if "p_at_most" in expect and p > float(expect["p_at_most"]):
        problems.append(f"p {p:.3f} is above {expect['p_at_most']}")
    return got, problems


def _request_case(view: Any, version: str, case: dict) -> tuple[dict, list[str]]:
    kind = case["kind"]
    try:
        req = REQUESTS[kind].model_validate({**case["request"], "model": version})
    except ValidationError as e:
        raise ContractError(f"request is not a valid {kind} request: {_pydantic(e)}") from None
    call = {"decide": view.decide, "evaluate": view.evaluate, "systemone": view.systemone}[kind]
    out = call(req)
    expect = case["expect"]
    got: dict[str, Any] = {k: out[k] for k in ("action", "verdict") if out.get(k) is not None}
    nxt = out.get("next")
    if nxt is None and isinstance(out.get("plan"), dict):
        nxt = out["plan"].get("next")
    if nxt is not None:
        got["next"] = nxt
    problems: list[str] = []
    if any(k in expect for k in ("action", "action_at_most", "action_at_least")):
        if "action" not in got:
            problems.append("the response has no action")
        else:
            problems += _tier_problems("action", got["action"], expect, _tier_names(view, version))
    for k in ("next", "verdict"):
        if k in expect and got.get(k) != expect[k]:
            problems.append(f"{k} {got.get(k)!r}, expected {expect[k]!r}")
    if "answers" in expect:
        answers = out.get("answers") or {}
        got["answers"] = {}
        for qid, want in expect["answers"].items():
            a = answers.get(qid)
            if not isinstance(a, dict):
                problems.append(f"no answer for question {qid!r}")
                continue
            if a.get("calibrated") is False:
                raise _Unjudgeable(f"question {qid!r}: head {a.get('head')!r} is not trained in {version}: its "
                                   f"answer is the uninformed prior, which can neither pass nor fail a case")
            got["answers"][qid] = {k: a[k] for k in ("type", "choice", "score", "noul") if a.get(k) is not None}
            problems += _answer_problems(qid, a, want)
    return got, problems


def _answer_problems(qid: str, a: dict, want: dict) -> list[str]:
    out = []
    if "choice" in want and a.get("choice") != want["choice"]:
        out.append(f"{qid}: choice {a.get('choice')!r}, expected {want['choice']!r}")
    if "score" in want and a.get("score") != want["score"]:
        out.append(f"{qid}: score {a.get('score')!r}, expected {want['score']!r}")
    noul = a.get("noul")
    if "noul_at_least" in want and (noul is None or float(noul) < float(want["noul_at_least"])):
        out.append(f"{qid}: noul {noul!r} is below {want['noul_at_least']}")
    if "noul_at_most" in want and (noul is None or float(noul) > float(want["noul_at_most"])):
        out.append(f"{qid}: noul {noul!r} is above {want['noul_at_most']}")
    return out
