"""Reviewing an LLM's output: deterministic checks first, then a finite revision loop.

The evaluator turns a drafted answer into ship / review / reject through a decision table with explicit
precedence (the guide, "convert review into action"):

    1. malformed state                                -> 422 (nothing is judged)
    2. a deterministic check the caller ran failed    -> reject; the heads are not asked (`decided_by: checks`)
    3. the artifact is partial or missing             -> reject; a half answer never ships (`decided_by: artifact`)
       cited chunks outside the evidence              -> reject (`decided_by: citations`)
       an answer that leaks a secret                  -> reject (`decided_by: output_scan`)
    4. a revision past its budget, or the same
       property failing twice, or an unchanged state  -> reject; escalate or narrow the task (`decided_by: revision`)
    5. the heads (grounded, complete, risk_tier ...)  -> ship only when every gate is favourable (`decided_by: heads`)

A stopped revision loop outranks rules 2-3: its `decided_by` is `revision`, the earlier rule's reason is kept in
`warnings`.

A favourable review is evidence, not authority: `act` / ship means the answer is eligible for the caller's next
policy gate, never that a consequential action was authorised.

Artifact (`artifact` in the request; also kept in the receipt as provenance):
    {status: complete | partial | none, finish_reason?, model?, provider?, request_digest?, latency_ms?, usage?}
A provider that streamed half an answer before disconnecting must say `partial`, not look like a complete answer.

Checks (`checks`): the caller's own deterministic validation, run before open-rlcd is asked: parsers, schemas,
tests, allowlists. `{name: true | false | {passed: bool, detail?: str}}`. Any false rejects without a semantic
review: a favourable semantic answer must never override a failed exact check.

Revisions (`revision: {of: <decision_id of the previous evaluation>}`): when a review fails, the response carries
`revise` — which properties failed, which evidence the answer must stay within — for the generator's next attempt.
Each revision names the evaluation it revises; open-rlcd counts the attempts along that chain and stops the loop:
an attempt past `evaluate.max_revisions` (default 2), a property that fails in two consecutive attempts, or a
revision whose state is identical to the one it revises (asking again over the same evidence is sampling until a
convenient answer arrives, not recovery) is rejected with the reason, so the caller escalates or narrows the task.
"""
from __future__ import annotations

from typing import Any

from .contract import ContractError

ARTIFACT_STATUS = ("complete", "partial", "none")
DEFAULT_MAX_REVISIONS = 2
MAX_CHECKS = 64


def parse_artifact(raw: Any) -> dict | None:
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ContractError("artifact must be an object {status, finish_reason?, model?, provider?, ...}",
                            field="artifact")
    status = raw.get("status", "complete")
    if status not in ARTIFACT_STATUS:
        raise ContractError(f"artifact.status must be one of {list(ARTIFACT_STATUS)}, got {status!r}",
                            field="artifact.status")
    for k in ("finish_reason", "model", "provider", "request_digest"):
        if raw.get(k) is not None and not isinstance(raw[k], str):
            raise ContractError(f"artifact.{k} must be a string", field=f"artifact.{k}")
    if raw.get("latency_ms") is not None and not isinstance(raw["latency_ms"], (int, float)):
        raise ContractError("artifact.latency_ms must be a number", field="artifact.latency_ms")
    return {**raw, "status": status}


def parse_checks(raw: Any) -> dict[str, dict] | None:
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ContractError("checks must be an object {check name: true | false | {passed, detail?}}",
                            field="checks")
    if len(raw) > MAX_CHECKS:
        raise ContractError(f"at most {MAX_CHECKS} checks", field="checks", limit=MAX_CHECKS)
    out: dict[str, dict] = {}
    for name, v in raw.items():
        if isinstance(v, bool):
            out[str(name)] = {"passed": v}
        elif isinstance(v, dict) and isinstance(v.get("passed"), bool):
            detail = v.get("detail")
            if detail is not None and not isinstance(detail, str):
                raise ContractError(f"checks.{name}.detail must be a string", field=f"checks.{name}")
            out[str(name)] = {"passed": v["passed"], **({"detail": detail} if detail else {})}
        else:
            raise ContractError(f"checks.{name} must be true, false or {{passed: bool, detail?: string}}",
                                field=f"checks.{name}")
    return out


def precheck(artifact: dict | None, checks: dict[str, dict] | None) -> dict | None:
    """The first deterministic rule that rejects the proposal, by precedence; None when every one passes."""
    failed = [n for n, c in (checks or {}).items() if not c["passed"]]
    if failed:
        details = "; ".join(f"{n}: {checks[n]['detail']}" if checks[n].get("detail") else n for n in failed)
        return {"rule": "checks", "failed": failed,
                "reason": f"deterministic checks failed ({details}): no semantic review can override them"}
    if artifact is not None and artifact["status"] != "complete":
        fr = artifact.get("finish_reason")
        what = "no answer was produced" if artifact["status"] == "none" else "the answer is partial"
        return {"rule": "artifact", "status": artifact["status"],
                "reason": f"{what}{f' (finish_reason {fr})' if fr else ''}: a partial or missing artifact never ships; "
                          f"retry the generation or fall back"}
    return None


def output_scan(answer: str, project: Any) -> dict | None:
    """Output screening (the guide VI.B): an answer that leaks a secret (and, with `pii`, an e-mail address or a
    card number) never ships, whatever the heads say. rlcd.yaml `evaluate.output_scan`: [secrets] by default,
    [secrets, pii], or false."""
    from .redact import kinds_from_config, redact_text

    try:
        kinds = kinds_from_config(project.get("evaluate.output_scan", True))
    except ValueError as e:
        raise ContractError(f"rlcd.yaml evaluate.output_scan: {e}", kind="bad_config", status=500) from None
    if not kinds:
        return None
    from .redact import _SECRETS, SHAPES

    # shapes only: the `assignment` heuristic would reject documentation ("Password: minimum twelve characters")
    _, found = redact_text(str(answer or ""), kinds, skip={k for k, _ in _SECRETS if k not in SHAPES})
    if not found:
        return None
    return {"rule": "output_scan", "found": found,
            "reason": f"the answer contains {', '.join(found)}: it must not ship (the generator leaked what the "
                      f"output scan looks for)"}


def parse_revision(raw: Any) -> str | None:
    """The decision_id a revision revises."""
    if raw is None:
        return None
    of = raw.get("of") if isinstance(raw, dict) else None
    if not isinstance(of, str) or not of:
        raise ContractError("revision must be {of: <decision_id of the evaluation this answer revises>}",
                            field="revision")
    return of


def revision_context(previous: dict | None, of: str, state_digest: str) -> dict:
    """What the loop knows from the evaluation being revised: attempt number, the properties that failed there,
    whether the state is unchanged."""
    if previous is None:
        raise ContractError(f"revision.of: unknown decision_id {of!r}", kind="unknown_decision", field="revision")
    if previous.get("kind") != "evaluate":
        raise ContractError(f"revision.of: decision {of} is a {previous.get('kind')}, not an evaluation",
                            field="revision")
    r = previous.get("receipt") or {}
    prev_rev = ((r.get("audit") or {}).get("revision") or {})
    attempt = int(prev_rev.get("attempt") or 1) + 1
    failed = list(previous.get("failed") or [])
    return {"of": of, "attempt": attempt, "previously_failed": failed,
            "unchanged": state_digest == r.get("state_digest")}


def revision_verdict(ctx: dict | None, failed_now: list[str], max_revisions: int) -> str | None:
    """The reason the revision loop stops here (the answer is rejected), or None."""
    if ctx is None:
        return None
    n = ctx["attempt"]
    if ctx["unchanged"]:
        return (f"attempt {n} (revision {n - 1}) sent exactly the state of evaluation {ctx['of']}: asking again over "
                f"the same evidence is not a recovery; change the answer or the evidence, or escalate")
    if n - 1 > max_revisions:
        return (f"revision budget exhausted: this is attempt {n}, after {n - 1} revisions (evaluate.max_revisions = "
                f"{max_revisions}); escalate or narrow the task")
    twice = sorted(set(failed_now) & set(ctx["previously_failed"]))
    if twice:
        return (f"{', '.join(twice)} failed in two consecutive attempts: the request may be impossible, the context "
                f"insufficient, or the verification contract wrong; escalate or narrow the task")
    return None


def max_revisions(project: Any) -> int:
    v = project.get("evaluate.max_revisions", DEFAULT_MAX_REVISIONS)
    try:
        n = int(v)
    except (TypeError, ValueError):
        raise ContractError("rlcd.yaml evaluate.max_revisions must be an integer", kind="bad_config",
                            status=500) from None
    if n < 0:
        raise ContractError("rlcd.yaml evaluate.max_revisions must be >= 0", kind="bad_config", status=500)
    return n
