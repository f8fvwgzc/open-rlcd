"""Receipts: every decision as a tamper-evident, replayable record (the guide's "execution ledger").

A receipt binds what was decided to everything the decision depended on:

    state_digest        the request state as the caller sent it (the state itself is logged only with
                        serve.log_states, secrets redacted)
    questions           the question contract: a battery {name, version, digest}, or the digest of the ad-hoc
                        questions / the heads and parameters a decide / evaluate call used
    model, pack, feature_version, policy_digest, stakes, rollout (mode, automation, whether it capped)
    answers             per head: class, p, tier, the full distribution, uncertainty sources
    action / plan / verdict, latency, warnings
    audit               what the caller passed to be kept with the decision: the baseline route it would have taken
                        without open-rlcd (counterfactual), the LLM artifact's provenance, deterministic check
                        results, the revision it belongs to

`contract_digest` = digest(model, pack, feature version, question contract, policy): with `state_digest` it identifies
what was asked about which state. It is not a cache key for a decide / evaluate result, whose tier also depends on
the rollout, the kill switch, the live Bernoulli table, `explore`, the clock and the corpus. Receipts are lines of decisions.jsonl (the log /v1/feedback refers to by
decision_id; older lines without a receipt stay readable). Each receipt carries the digest of the receipt before
it (`prev`) and its own digest, so an edited, inserted or deleted line breaks the chain (`rlcd receipts verify`).
Truncating the newest lines cannot be detected from the file alone: anchor the head digest elsewhere
(`rlcd receipts verify` prints it). Appends hold an exclusive lock on the file (POSIX), so several processes
writing one home still produce one chain; a torn last line (a crash mid-write) is terminated before the next append
and is reported, never chained.

Confirmations (confirmations.jsonl: a person confirmed a confirm decision) form a second chain of the same kind:
each line carries its own `receipt` part with `prev` and `digest`. A confirmation counts for a decision only when
it names that decision and the state the decision judged and matches its own digest (`confirmation_problem`): a
line without a digest, or one edited after it was written, confirms nothing, and `rlcd receipts verify` reports
both chains. The digest is not a signature: anyone who can write RLCD_HOME can append a confirmation that counts
(and verifies, with the right `prev`). RLCD_ADMIN_KEY on the confirm route, file permissions on the home and an
external anchor of both heads are what limit that.

Staleness (`check`): a receipt approves what it saw, not what exists now. Before a side effect, the caller sends
the state it is about to act on; the receipt still applies only if that state has the same digest, the model is
the same (when the decision used the `rlcd-latest` alias: it still points at the same version), the policy and
battery version are unchanged, automation was not switched off or capped since (for an `act` decision), and the
receipt is younger than `max_age_s`. Two kinds of decision never apply on their own: one that abstained (escalate
/ hold / a rejected answer) approves no action at all, and a `confirm` decision applies only once a person has
confirmed it (the service adds that check).
"""
from __future__ import annotations

import json
import os
import threading
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .digest import decode, digest, dumps, encode, plain

try:  # POSIX only; elsewhere the in-process lock still serialises one server's writes
    import fcntl
except ImportError:  # pragma: no cover
    fcntl = None  # type: ignore[assignment]

FORMAT = "open-rlcd-receipt/1"
CONFIRMATION_FORMAT = "open-rlcd-confirmation/1"
MAX_WARNINGS = 20
MAX_AUDIT_BYTES = 64 << 10
MAX_LOGGED_STATE_BYTES = 2 << 20        # logged head states per record (the rest keep only their digest)
_BLOCK = 1 << 16


# ----------------------------------------------------------------------------- building

def contract_digest(*, model: str, pack: dict | None, feature_version: str | None, questions: str | None,
                    policy: str | None) -> str:
    return digest({"model": model, "pack": pack, "feature_version": feature_version, "questions": questions,
                   "policy": policy})


def bounded_audit(audit: dict | None) -> dict | None:
    """The caller-supplied audit fields, refused when they would bloat the log (422 at the edge is better than a
    silently cut receipt)."""
    if not audit:
        return None
    out = {k: v for k, v in audit.items() if v is not None}
    if not out:
        return None
    size = len(json.dumps(plain(out), ensure_ascii=False, default=str).encode())
    if size > MAX_AUDIT_BYTES:
        from .contract import ContractError

        raise ContractError(f"audit fields ({', '.join(sorted(out))}) are {size} bytes as JSON; at most "
                            f"{MAX_AUDIT_BYTES} are kept per receipt", field="audit", limit=MAX_AUDIT_BYTES)
    return out


def make(*, requested_model: str | None, model: str, pack: dict | None, feature_version: str | None,
         state: Any, questions: dict, policy_digest: str | None, rollout: dict | None, latency_ms: float | None,
         warnings: list[str] | None, audit: dict | None = None) -> dict:
    """The `receipt` part of a decision record (prev / digest are set when it is appended)."""
    qd = questions.get("digest")
    out = {"format": FORMAT, "requested_model": requested_model, "pack": pack, "feature_version": feature_version,
           "state_digest": digest(state), "questions": questions, "policy_digest": policy_digest,
           "rollout": rollout, "latency_ms": latency_ms,
           "warnings": list(warnings or [])[:MAX_WARNINGS],
           "contract_digest": contract_digest(model=model, pack=pack, feature_version=feature_version, questions=qd,
                                              policy=policy_digest)}
    a = bounded_audit(audit)
    if a:
        out["audit"] = a
    return out


def summary(row: dict) -> dict:
    """The compact receipt returned with a response."""
    r = row.get("receipt") or {}
    return {"id": row.get("decision_id"), "digest": r.get("digest"), "prev": r.get("prev"),
            "state_digest": r.get("state_digest"), "contract_digest": r.get("contract_digest"),
            "questions_digest": (r.get("questions") or {}).get("digest")}


def row_digest(row: dict) -> str:
    """The digest of a decision record: everything but receipt.digest itself, with each logged head state replaced
    by its digest. The chain binds what each head judged without depending on the logged copy, so states can be
    pruned for retention (`prune_states`) while every receipt still verifies. A head that still carries its state
    is digested from that state, never from a `state_digest` stored next to it: an edited state cannot hide behind
    the original's digest."""
    r = dict(row.get("receipt") or {}) if isinstance(row.get("receipt"), dict) else row.get("receipt")
    if isinstance(r, dict):
        r.pop("digest", None)
    heads = row.get("heads")
    if isinstance(heads, dict):
        out = {}
        for name, h in heads.items():
            if isinstance(h, dict) and "state" in h:
                h = {k: v for k, v in h.items() if k != "state"}
                h["state_digest"] = digest(heads[name]["state"])
            out[name] = h
        heads = out
    body = {**row, "receipt": r}
    if "heads" in row:
        body["heads"] = heads
    return digest(body)


def prune_states(path: Path, before: str) -> dict:
    """Drop the logged head states of receipts created before `before` (ISO 8601), keeping each state's digest so
    the chain still verifies. Feedback can no longer train from a pruned decision (it has no state).

    The file is rewritten in place under the append lock (same inode, so a writer waiting for the lock appends to
    the rewritten file, never to a replaced one); a copy of the original is kept as decisions.jsonl.bak until the
    rewrite succeeds. Run it while the server is idle: a long log takes a while to rewrite."""
    path = Path(path)
    if not path.exists():
        return {"pruned": 0, "rows": 0, "before": before}
    cutoff = _parse_time(before)
    if cutoff is None:
        raise ValueError(f"before must be an ISO 8601 date or time, got {before!r}")
    pruned = rows = 0
    backup = path.with_suffix(".jsonl.bak")
    with open(path, "r+b") as f:
        if fcntl is not None:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            original = f.read()
            backup.write_bytes(original)
            out = []
            for line in original.split(b"\n"):
                if not line.strip():
                    continue
                rows += 1
                try:
                    row = json.loads(decode(line))
                except ValueError:
                    out.append(line)
                    continue
                changed = False
                created = _parse_time(row.get("created_at")) if isinstance(row, dict) else None
                heads = row.get("heads") if isinstance(row, dict) else None
                if created is not None and created < cutoff and isinstance(heads, dict):
                    for h in heads.values():
                        if isinstance(h, dict) and "state" in h:
                            h["state_digest"] = digest(h["state"])   # from the state itself, never a stored one
                            del h["state"]
                            changed = True
                if changed:
                    pruned += 1
                    line = encode(dumps(row))
                out.append(line)
            data = b"".join(x + b"\n" for x in out)
            f.seek(0)
            f.write(data)
            f.truncate()
            f.flush()
            os.fsync(f.fileno())
            backup.unlink(missing_ok=True)
        finally:
            if fcntl is not None:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
    return {"pruned": pruned, "rows": rows, "before": before}


def _parse_time(v: Any) -> datetime | None:
    if not v:
        return None
    try:
        t = datetime.fromisoformat(str(v).replace("Z", "+00:00"))
    except ValueError:
        return None
    return t if t.tzinfo is not None else t.replace(tzinfo=UTC)


# ----------------------------------------------------------------------------- the chained log

def _reverse_lines(f) -> Iterator[bytes]:
    """The lines of a binary file, last first (a final fragment without a newline included)."""
    f.seek(0, os.SEEK_END)
    pos = f.tell()
    head = b""
    while pos > 0:
        step = min(_BLOCK, pos)
        pos -= step
        f.seek(pos)
        parts = (f.read(step) + head).split(b"\n")
        head = parts[0]
        for line in reversed(parts[1:]):
            if line.strip():
                yield line
    if head.strip():
        yield head


def _last_digest(f) -> str | None:
    for line in _reverse_lines(f):
        try:
            row = json.loads(decode(line))
        except ValueError:
            continue    # a torn line is never part of the chain
        # a line whose `receipt` is not an object (written by hand, or by another tool) is skipped, as verify_chain
        # skips it: one bad line must not stop every later append
        r = row.get("receipt") if isinstance(row, dict) else None
        d = r.get("digest") if isinstance(r, dict) else None
        if d:
            return d
    return None


class ChainedLog:
    """decisions.jsonl with a hash chain over its receipt lines."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self._lock = threading.RLock()
        # (file size, modification time, last digest) after our own last append: any other write changes the time
        self._tip: tuple[int, int, str | None] | None = None

    def append(self, row: dict) -> dict:
        """Chain and append one record (which must have a `receipt` part). Returns the record as written."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock, open(self.path, "a+b") as f:
            if fcntl is not None:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                f.seek(0, os.SEEK_END)
                size = f.tell()
                mtime = os.fstat(f.fileno()).st_mtime_ns
                if self._tip is not None and self._tip[:2] == (size, mtime):
                    prev = self._tip[2]
                else:            # another writer (or a restart) since our last append: read the tip from the file
                    prev = _last_digest(f) if size else None
                    f.seek(0, os.SEEK_END)
                if size:
                    f.seek(size - 1)
                    if f.read(1) != b"\n":
                        f.seek(0, os.SEEK_END)
                        f.write(b"\n")       # terminate a torn last line: it must not swallow this record
                rec = plain(row)
                rec["receipt"] = {**(rec.get("receipt") or {}), "prev": prev}
                rec["receipt"].pop("digest", None)
                rec["receipt"]["digest"] = row_digest(rec)
                f.seek(0, os.SEEK_END)
                f.write(encode(dumps(rec) + "\n"))
                f.flush()
                self._tip = (f.tell(), os.fstat(f.fileno()).st_mtime_ns, rec["receipt"]["digest"])
                return rec
            finally:
                if fcntl is not None:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def verify_chain(path: Path) -> dict:
    """Walk a chained log (decisions.jsonl, confirmations.jsonl) and check every line's receipt: its own digest, and
    that it names the receipt before it."""
    path = Path(path)
    out: dict[str, Any] = {"path": path.name, "lines": 0, "receipts": 0, "legacy": 0, "unreadable": [], "ok": True,
                           "breaks": [], "head": None}
    if not path.exists():
        return out
    prev: str | None = None
    started = False
    with open(path, "rb") as f:
        for n, line in enumerate(f, 1):
            if not line.strip():
                continue
            out["lines"] += 1
            try:
                row = json.loads(decode(line))
            except ValueError:
                out["unreadable"].append(n)
                continue
            r = row.get("receipt") if isinstance(row, dict) else None
            if not isinstance(r, dict) or not r.get("digest"):
                if started:
                    # every record written since the chain started has a receipt: a line without one after it was
                    # inserted (or written by an open-rlcd that predates receipts, into the same home)
                    out["breaks"].append({"line": n, "decision_id": row.get("decision_id")
                                          if isinstance(row, dict) else None,
                                          "why": "a record without a receipt after the chain started (inserted, or "
                                                 "written by an older open-rlcd)"})
                else:
                    out["legacy"] += 1
                continue
            started = True
            out["receipts"] += 1
            did = row.get("decision_id")
            try:
                same = row_digest(row) == r["digest"]
            except Exception:  # a malformed record (heads not an object, ...) is a break, never a crash
                same = False
            if not same:
                out["breaks"].append({"line": n, "decision_id": did, "why": "the record does not match its digest "
                                                                            "(edited after it was written)"})
            if r.get("prev") != prev:
                out["breaks"].append({"line": n, "decision_id": did,
                                      "why": "it does not name the receipt before it (a receipt was inserted, "
                                             "deleted or reordered here)"})
            prev = r["digest"]
    out["head"] = prev
    out["ok"] = not out["breaks"]
    return out


# ----------------------------------------------------------------------------- staleness

def _age_s(created_at: str | None, now: datetime) -> float | None:
    t = _parse_time(created_at)
    return None if t is None else (now - t).total_seconds()


def _was_act(row: dict) -> bool:
    """Did this decision authorise an automatic action? A decide / evaluate record says so in its overall `action`
    (a head that acted inside a plan that escalates authorises nothing); a record without one (/v1/rank, older
    lines) counts any head at `act`."""
    if "action" in row:
        return row.get("action") == "act"
    return any((h or {}).get("tier") == "act" for h in (row.get("heads") or {}).values())


def needs_confirmation(row: dict) -> bool:
    """A decision at `confirm` (its overall action, or for a record without one any head at confirm) authorises an
    action only after a person confirmed it."""
    if "action" in row:
        return row.get("action") == "confirm"
    return any((h or {}).get("tier") == "confirm" for h in (row.get("heads") or {}).values())


def confirmation_problem(conf: dict, row: dict) -> str | None:
    """Why a confirmations.jsonl line does not confirm decision `row`, or None when it does: it must name the
    decision and the state the decision judged, and match its own digest, so a line without one or edited after it
    was written confirms nothing. The digest is unkeyed: it catches edits, not a forger who can write RLCD_HOME. The
    line's place in the chain is `verify_chain`'s to judge (walking the chain on every check would add nothing a
    forger could not satisfy by naming the current head)."""
    if not isinstance(conf, dict) or conf.get("decision_id") != row.get("decision_id"):
        return "it names another decision"
    if conf.get("state_digest") != (row.get("receipt") or {}).get("state_digest"):
        return "it names another state than the one the decision judged"
    c = conf.get("receipt")
    if not isinstance(c, dict) or not c.get("digest"):
        return "it carries no digest (written by hand, or before confirmations were chained: confirm it again)"
    try:
        same = row_digest(conf) == c["digest"]
    except Exception:  # a malformed line confirms nothing, it never breaks the check
        same = False
    return None if same else "it does not match its digest (edited after it was written)"


def check(row: dict, *, state_digest: str, latest: str, known_models: list[str], policy_digest: str | None,
          automation_cap: int, battery_version: str | None = None, max_age_s: float | None = None,
          now: datetime | None = None, aliases: tuple = (None, "", "rlcd-latest")) -> dict:
    """Does a receipt still approve acting on the state the caller is about to act on?

    `latest`: the version rlcd-latest points at now; `known_models`: the versions the registry serves now;
    `policy_digest`: the policy of the receipt's model version as it is now; `automation_cap`: the strongest tier
    the rollout and kill switch allow now (2 = act); `battery_version`: the current version of the receipt's
    battery (None when it had none, or the battery no longer exists)."""
    now = now or datetime.now(UTC)
    r = row.get("receipt") or {}
    reasons: list[str] = []
    checks: dict[str, Any] = {}
    same_state = state_digest == r.get("state_digest")
    checks["state"] = "same" if same_state else "changed"
    if not same_state:
        reasons.append("the state differs from the one this decision judged (a changed file, policy text, tool "
                       "argument or answer needs a new decision)")
    model = row.get("model")
    requested = r.get("requested_model")
    if requested in aliases:
        checks["model"] = "same" if latest == model else "moved"
        if latest != model:
            reasons.append(f"rlcd-latest now points at {latest}, the decision was made by {model}")
    else:
        checks["model"] = "pinned" if model in known_models else "gone"
        if model not in known_models:
            reasons.append(f"model {model} is no longer served")
    if r.get("policy_digest") is not None:
        same_policy = policy_digest == r.get("policy_digest")
        checks["policy"] = "same" if same_policy else "changed"
        if not same_policy:
            reasons.append("the policy of this model version changed since the decision")
    battery = (r.get("questions") or {}).get("battery")
    if battery:
        checks["battery"] = "same" if battery_version == battery.get("version") else "changed"
        if battery_version != battery.get("version"):
            now_v = "removed" if battery_version is None else f"now version {battery_version}"
            reasons.append(f"battery {battery.get('name')} was version {battery.get('version')}, {now_v}")
    if "action" in row and row.get("action") not in ("act", "confirm"):
        # an abstained decision (escalate / hold / a rejected answer) approves no action, whatever else holds
        checks["approval"] = "none"
        reasons.append(f"the decision approves no action (its action is {row.get('action')!r}): a person decides")
    if _was_act(row):
        checks["automation"] = "allowed" if automation_cap >= 2 else "capped"
        if automation_cap < 2:
            reasons.append("automation is switched off or capped by the rollout now: an act decision made before "
                           "no longer authorises an automatic action")
    age = _age_s(row.get("created_at"), now)
    checks["age_s"] = None if age is None else round(age, 3)
    if max_age_s is not None and (age is None or age > float(max_age_s)):
        reasons.append(f"the decision is older than max_age_s={max_age_s:g}")
    return {"decision_id": row.get("decision_id"), "kind": row.get("kind"), "model": model,
            "action": row.get("action"), "current": not reasons, "stale_reasons": reasons, "checks": checks}
