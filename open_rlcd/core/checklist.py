"""The pre-launch checklist (`rlcd check`): the field guide's launch list, checked mechanically.

The field guide this control layer follows (docs/control-layer.md) closes with a pre-launch checklist: required
fields validated; no-match options present; deterministic rules run before AI; secrets minimised; provider routes
approved; failure behaviour defined; thresholds evaluated on held-out cases; receipts replayable; rollback tested;
human review staffed for the volume the policy will create; and, from its production sections, named owners. Most of
these are facts open-rlcd can read about one project — the trained heads and their holdout verdicts, the head specs,
rlcd.yaml, the server's environment, the receipt chain, the registry — so `run` reads them and returns one line per
item, each with a status:

    pass   checked, and it holds
    warn   checked, and it deserves a look before launch (a warning never makes the run fail)
    fail   checked, and it does not hold: `ok` is false and `rlcd check` exits 1
    info   a setting worth knowing, or something open-rlcd cannot see (your callers, your validators)

Check ids (stable, for scripts and CI):

    config.valid           fail when rlcd.yaml has an error (`rlcd config check`), warn on its warnings
    heads.trained          fail when no head is trained, warn when some are not
    skill.<head>           per trained head: warn without skill over the base rate; its verdict per stakes level
    thresholds.holdout     fail when a head has no holdout verdict for a stakes level the version serves, or was
                           verified under other act thresholds than the ones served; warn below min_verify_rows
    thresholds.config      warn when rlcd.yaml `policy` differs from the policy the version serves (a version keeps
                           the thresholds it was verified with until a retrain verifies new ones)
    options.no_match       warn when a choice head has no escape option (none, other, unknown, out_of_scope, ...)
    score.anchors          warn when score levels lack observable descriptions (only low / medium / high, ...)
    heads.described        warn when a head has no description, or a choice head no option descriptions
    noul.positive          warn when a noul head does not say what yes means (`positive`)
    options.overlap        warn when two options (or levels) of one head have the same description
    inputs.required        info: how a missing field fails closed
    deterministic.first    info: your deterministic checks run before any head (open-rlcd cannot see them)
    routes.contracts       fail when a route contract names a head or an option the model does not have
    security.api_key       fail when rlcd.yaml serve.host is not loopback and RLCD_API_KEY is not set
    security.allow_remote  warn when connectors and judges may reach non-loopback hosts
    security.admin_key     warn when RLCD_ADMIN_KEY is unset and the log holds confirm decisions, however old (a
                           decision caller can confirm its own); info: whether training, the switch and
                           confirmations need it
    secrets.redact         warn when serve.redact is off
    secrets.log_states     info: whether request states are logged
    receipts.chain         fail when the hash chain of decisions.jsonl or of confirmations.jsonl does not verify
    receipts.mode          info: serve.receipts
    model.bundle           fail when the checked version's bundle does not validate
    rollback.ready         warn without an earlier version to roll back to, fail when that version does not load
    rollout.mode           info: the rollout mode
    automation.switch      warn when automation is switched off (or the switch cannot be written from here); info
                           says whether the switch was ever rehearsed
    batteries.bound        fail when a battery question does not bind to a head; warn when its head is untrained
    batteries.versions     fail when receipts recorded a battery version under another digest than it has now; warn
                           when batteries exist but no receipt records one (serve.receipts: decisions skips them)
    regressions.cases      fail when a regression case fails (needs open_rlcd.core.regress; info without it)
    review.capacity        warn when confirm decisions per day exceed serve.review_capacity_per_day (or it is unset)
    owners                 warn unless rlcd.yaml `owners` names pack, heads, policy, review and incidents
    failure.behaviour      info: a caller treats an error as "review unavailable", never as approval

Each check is isolated: one that raises becomes a `fail` naming the exception type, and the others still run.

What this does not do: it changes nothing (regression cases run in a dry run: nothing is logged, no switch is
flipped, no ledger moves); it reads the project, the environment and RLCD_HOME and calls no LLM — only the regression
cases go through the heads, exactly as `rlcd regress` runs them (so through the pack's feature provider and, for a
request case, the stores the pack is configured with); and it cannot see your application, so the items that live
there are reminders (info), never passes. A run without a fail is not a launch approval: warnings are judgment calls
for the owners, and the checks only see what open-rlcd sees.
"""
from __future__ import annotations

import functools
import importlib
import math
import os
import re
import textwrap
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from ..connectors.security import is_loopback
from . import bundle as BD
from . import policy as PL
from .contract import ContractError
from .digest import digest, short

STATUSES = ("pass", "warn", "fail", "info")
# options that let a closed choice say "none of these" (guide III.G, IV.C); compared lower-case, "-" and " " as "_"
NO_MATCH = frozenset({"none", "other", "unknown", "out_of_scope", "no_match", "needs_review"})
NO_MATCH_PREFIXES = ("escalate", "clarify", "none_", "other_")
# words that rank without describing (guide IV.H): a level made only of these has no observable anchor
VAGUE_WORDS = frozenset({
    "a", "an", "the", "and", "or", "of", "is", "it", "very", "quite", "rather", "somewhat", "slightly", "extremely",
    "not", "no", "yes", "low", "lower", "lowest", "medium", "mid", "middle", "moderate", "high", "higher", "highest",
    "bad", "good", "ok", "okay", "fine", "poor", "fair", "great", "excellent", "average", "acceptable", "none",
    "some", "most", "all", "little", "much", "minimal", "partial", "partially", "full", "fully", "neutral", "positive",
    "negative", "weak", "strong", "minor", "major", "best", "worst", "better", "worse", "level", "score", "tier",
    "grade", "rating"})
# rlcd.yaml `owners` (guide X.J: state builder, question contract, thresholds, provider registry, review queue,
# incident response)
OWNER_ROLES = {
    "pack": "the state builder: pack, features, data",
    "heads": "the question contract: heads and batteries",
    "policy": "thresholds, rollout and route contracts",
    "review": "the review queue",
    "incidents": "incident response and the kill switch",
}
REVIEW_WINDOW_DAYS = 7
REVIEW_ACTIONS = ("confirm",)          # the actions a person reviews before anything happens (guide VII.B, X.I)
REGRESSIONS_FILE = "regressions.jsonl"
REGRESS_MODULE = "open_rlcd.core.regress"
CONTRACT_DOC = "docs/contract.md"
_DAY_S = 86400.0
_INDEX = re.compile(r"^\s*\d+\s*[:.)\-–—]\s*")     # a leading "0:", "1.", "2)" or "3 -" before a level description
_WORD = re.compile(r"[^\W_]+")
_LABEL = {"pass": "PASS", "warn": "WARN", "fail": "FAIL", "info": "info"}


@dataclass(frozen=True)
class Check:
    id: str
    status: str
    title: str
    detail: str

    def to_dict(self) -> dict:
        return {"id": self.id, "status": self.status, "title": self.title, "detail": self.detail}


Outcome = tuple[str, str] | list[Check]          # what a check function returns: (status, detail) or several checks
_CHECKS: list[tuple[str, str, Callable[[_Ctx], Outcome]]] = []


def _check(cid: str, title: str) -> Callable:
    """Register a check: `fn(ctx)` returns (status, detail), or a list of Checks (one per head)."""
    def deco(fn: Callable[[_Ctx], Outcome]) -> Callable[[_Ctx], Outcome]:
        _CHECKS.append((cid, title, fn))
        return fn
    return deco


# ----------------------------------------------------------------------------- running

def run(service: Any, *, model: str | None = None, now: datetime | None = None, error_text: bool = True) -> dict:
    """Run every check against `model` (default: rlcd-latest) of `service`'s project.

    Returns {"ok", "model", "checks": [{id, status, title, detail}], "counts": {pass, warn, fail, info}}; `ok` is
    false only when some check fails. An unknown `model` is a ContractError (422 unknown_model), as at every endpoint.
    `now` fixes the clock of the review-load window (tests); with `error_text=False` a check that raised reports only
    the exception type, not its message (for an HTTP route: exception text can carry paths or a DSN)."""
    version = service.resolve(model)
    ctx = _Ctx(service, version, now or datetime.now(UTC), error_text)
    checks: list[Check] = []
    for cid, title, fn in _CHECKS:
        checks.extend(_safe(cid, title, functools.partial(fn, ctx), error_text))
    checks = [Check(c.id, c.status, c.title, _scrub(c.detail, service)) for c in checks]
    counts = {s: sum(1 for c in checks if c.status == s) for s in STATUSES}
    return {"ok": counts["fail"] == 0, "model": version, "checks": [c.to_dict() for c in checks], "counts": counts}


def format_table(result: dict, width: int = 100) -> str:
    """`run`'s result as the table `rlcd check` prints: status, id and title per check with its detail wrapped
    under it, then one line that says whether anything fails."""
    checks = result.get("checks") or []
    idw = max([len(c["id"]) for c in checks] + [8])
    lines = [f"rlcd check: model {result.get('model')}", ""]
    for c in checks:
        lines.append(f"{_LABEL.get(c['status'], c['status']):<4}  {c['id']:<{idw}}  {c['title']}")
        lines += textwrap.wrap(str(c["detail"]), width=width, initial_indent=" " * 6, subsequent_indent=" " * 6,
                               break_long_words=False, break_on_hyphens=False)
    counts = result.get("counts") or {}
    summary = ", ".join(f"{counts.get(s, 0)} {s}" for s in STATUSES)
    failing = [c["id"] for c in checks if c["status"] == "fail"]
    lines.append("")
    if failing:
        lines.append(f"NOT READY ({summary}): fix {', '.join(failing)} before launch")
    elif counts.get("warn"):
        lines.append(f"no check fails ({summary}); the warnings are for the owners to judge before launch")
    else:
        lines.append(f"no check fails ({summary})")
    return "\n".join(lines)


class _Ctx:
    """What several checks read, computed once per run. A failure is cached and raised again to every check that
    needs the value, so each of them reports it (the others are unaffected)."""

    def __init__(self, service: Any, version: str, now: datetime, error_text: bool):
        self.service, self.version, self.now, self.error_text = service, version, now, error_text
        self._memo: dict[str, tuple[bool, Any]] = {}

    def _get(self, key: str, fn: Callable[[], Any]) -> Any:
        if key not in self._memo:
            try:
                self._memo[key] = (True, fn())
            except Exception as e:
                self._memo[key] = (False, e)
        ok, value = self._memo[key]
        if not ok:
            raise value
        return value

    @property
    def project(self) -> Any:
        return self.service.project

    @property
    def specs(self) -> dict:
        return self._get("specs", lambda: dict(self.service.specs(self.version)))

    @property
    def off(self) -> set:
        """The heads the project switches off (Service.switched_off): not served even when this version has them."""
        return self._get("off", lambda: set(self.service.switched_off(self.version)))

    @property
    def policy(self) -> PL.PolicyConfig:
        return self._get("policy", lambda: self.service.policy(self.version))

    @property
    def trained(self) -> dict:
        """{head: LoadedHead} for the heads this version trained (or carried over)."""
        return self._get("trained", self._trained)

    def _trained(self) -> dict:
        out = {}
        for name in self.specs:
            h = self.service.loaded(self.version, name)
            if h is not None:
                out[name] = h
        return out

    @property
    def batteries(self) -> dict:
        return self._get("batteries", lambda: self.service.batteries(self.version))

    @property
    def scan(self) -> dict:
        return self._get("scan", lambda: _scan(self.service, self.now))


def _safe(cid: str, title: str, fn: Callable[[], Outcome], error_text: bool) -> list[Check]:
    try:
        out = fn()
    except Exception as e:   # a check never raises: a crash is a failed check, and the other checks still run
        return [Check(cid, "fail", title, _raised(e, error_text))]
    if isinstance(out, tuple):
        status, detail = out
        if status not in STATUSES:
            return [Check(cid, "fail", title, f"the check returned an unknown status {status!r}")]
        return [Check(cid, status, title, str(detail))]
    return list(out)


def _raised(e: BaseException, error_text: bool) -> str:
    if isinstance(e, ContractError):   # written for callers (422 bodies carry it): always shown
        return f"the check raised {type(e).__name__} ({e.kind}): {e.message}"
    msg = str(e).strip()
    if error_text and msg:
        return f"the check raised {type(e).__name__}: {msg[:500]}"
    return f"the check raised {type(e).__name__}"


def _scrub(text: str, service: Any) -> str:
    """Absolute paths of the home and the project root become RLCD_HOME / <project> in every detail."""
    for path, name in ((getattr(service, "home", None), "RLCD_HOME"),
                       (getattr(getattr(service, "project", None), "root", None), "<project>")):
        p = str(path) if path is not None else ""
        if len(p) > 1:
            text = text.replace(p, name)
    return text


def _names(names: list[str] | tuple[str, ...]) -> str:
    return ", ".join(str(n) for n in names)


def _holdout_rows(summary: dict) -> int | None:
    for v in ((summary.get("data") or {}).get("holdout_rows"), (summary.get("metrics") or {}).get("n_holdout")):
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            return int(v)
    return None


def _time(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        t = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    return t if t.tzinfo is not None else t.replace(tzinfo=UTC)


# ----------------------------------------------------------------------------- configuration

@_check("config.valid", "rlcd.yaml is valid")
def _config_valid(ctx: _Ctx) -> Outcome:
    """The findings of `rlcd config check` (core/config_check.py), against the pack this service runs."""
    from .config_check import check_project, errors, warnings

    project = ctx.project
    if getattr(project, "path", None) is None and not project.data:
        return "info", "no rlcd.yaml: the default project is used (pack and home from the environment or defaults)"
    findings = check_project(project, ctx.service.pack)
    bad, odd = errors(findings), warnings(findings)

    def listed(items: list[dict]) -> str:
        # without error_text only the keys are named: a message can quote a value or a path of the server
        shown = [f"{f['path']}: {f['message']}" if ctx.error_text else str(f["path"]) for f in items[:5]]
        more = f"; and {len(items) - 5} more" if len(items) > 5 else ""
        return "; ".join(shown) + more

    if bad:
        return "fail", (f"{len(bad)} error{'' if len(bad) == 1 else 's'} in rlcd.yaml: {listed(bad)}. `rlcd config "
                        f"check` lists every finding; `rlcd train` and `rlcd serve` refuse to start until it is clean")
    if odd:
        return "warn", f"{len(odd)} warning{'' if len(odd) == 1 else 's'} in rlcd.yaml: {listed(odd)}"
    return "pass", "every key of rlcd.yaml is known and every value is of its type and in its range"


# ----------------------------------------------------------------------------- heads and thresholds

@_check("heads.trained", "Heads are trained")
def _heads_trained(ctx: _Ctx) -> Outcome:
    specs, trained = ctx.specs, ctx.trained
    if not specs:
        return "fail", f"the pack defines no heads for {ctx.version}: there is nothing to answer with"
    untrained = [n for n in specs if n not in trained]
    if not trained:
        return "fail", (f"no head is trained in {ctx.version}: every answer is the uninformed prior and every "
                        f"decision abstains. Train with `rlcd train` (or POST /v1/train)")
    if untrained:
        return "warn", (f"{len(trained)} of {len(specs)} heads are trained in {ctx.version}; not trained: "
                        f"{_names(untrained)} (they answer with the uninformed prior and every decision of theirs "
                        f"abstains)")
    return "pass", f"all {len(specs)} heads are trained in {ctx.version}"


@_check("skill", "Each trained head shows skill on its holdout")
def _skill(ctx: _Ctx) -> Outcome:
    out: list[Check] = []
    for name, h in ctx.trained.items():
        out += _safe(f"skill.{name}", f"Head {name} shows skill on its holdout",
                     functools.partial(_head_skill, ctx, name, h), ctx.error_text)
    return out


def _head_skill(ctx: _Ctx, name: str, h: Any) -> Outcome:
    summary = h.summary                 # head.json, checked against its sha256
    m = summary.get("metrics") or {}
    n = _holdout_rows(summary)
    rows = f"{n} trusted holdout rows" if n is not None else "its trusted holdout rows"
    verdict = _verdict_text(h.verdict, list(ctx.policy.stakes))
    if not m.get("has_skill"):
        return "warn", (f"no skill over the base rate on {rows}: every decision of this head abstains until a "
                        f"retrain shows skill; {verdict}")
    bss = m.get("brier_skill_score")
    bss_text = f" (Brier skill score {bss:.3f})" if isinstance(bss, (int, float)) else ""
    try:
        gate = gate_lines(ctx.service.gate_outlook(ctx.version, name))
    except Exception:       # a reading aid: the verdict above is the check
        gate = []
    gate_text = (". Live decisions also pass the Bernoulli gate, per (class, probability bin): " + "; ".join(gate)
                 if gate else "")
    return "pass", f"skill over the base rate on {rows}{bss_text}; {verdict}{gate_text}"


def gate_lines(outlook: dict, shown: int = 4) -> list[str]:
    """`Service.gate_outlook` as one line per stakes level whose holdout verdict is act: how many of the cells
    that can act already pass the Bernoulli gate on holdout evidence, and how many correct outcomes (POST
    /v1/feedback with the decision_id) each of the others needs before its decisions move from confirm to act."""
    lines: list[str] = []
    for level, o in outlook.items():
        cells = o.get("cells") or []
        waiting = [c for c in cells if not c["passes"]]
        empty = o.get("from_empty")
        fresh = (f"a cell with no evidence yet needs {empty} correct outcomes in a row" if empty is not None
                 else "a cell with no evidence yet cannot pass")
        if not waiting:
            lines.append(f"{level}: all {len(cells)} cells that can act (p >= {o['act']:g}) pass on holdout evidence")
            continue

        def one(c: dict) -> str:
            need = (f"needs {c['needed']} correct" if c["needed"] is not None
                    else "cannot pass on correct outcomes alone")
            return f"{c['class']} at p {c['lo']:g}-{c['hi']:g} {need} ({c['wins']} of {c['wins'] + c['losses']} right)"

        more = f", and {len(waiting) - shown} more" if len(waiting) > shown else ""
        lines.append(f"{level}: {len(cells) - len(waiting)} of {len(cells)} cells that can act (p >= {o['act']:g}) "
                     f"pass on holdout evidence; the others start at confirm until correct outcomes arrive via "
                     f"POST /v1/feedback: {', '.join(one(c) for c in waiting[:shown])}{more} ({fresh}; the gate "
                     f"wants P(accuracy > {o['min_accuracy']:g}) >= {o['threshold']:g})")
    return lines


def _verdict_text(verdict: dict, served: list[str]) -> str:
    """'verdict: low confirm, medium act, high act (low: <why it is capped>)'."""
    levels = [*served, *[lv for lv in verdict if lv not in served]]
    tiers: list[str] = []
    capped: dict[str, list[str]] = {}
    for lv in levels:
        v = verdict.get(lv)
        if not isinstance(v, dict):
            tiers.append(f"{lv} (no verdict)")
            continue
        tiers.append(f"{lv} {v.get('max_tier')}")
        tier = v.get("tier")
        if (not isinstance(tier, int) or tier < 2) and v.get("reason"):
            capped.setdefault(str(v["reason"]), []).append(lv)
    text = "verdict: " + ", ".join(tiers)
    if capped:
        text += " (" + "; ".join(f"{'/'.join(lvs)}: {why}" for why, lvs in capped.items()) + ")"
    return text


@_check("thresholds.holdout", "Thresholds were verified on held-out rows")
def _thresholds(ctx: _Ctx) -> Outcome:
    trained = ctx.trained
    if not trained:
        return "info", "no head is trained: no threshold has been evaluated on held-out rows yet"
    cfg = ctx.policy
    k = cfg.min_verify_rows
    fails: list[str] = []
    warns: list[str] = []
    sizes: dict[str, int] = {}
    for name, h in trained.items():
        try:
            verdict = h.verdict
            missing = [lv for lv in cfg.stakes if not isinstance(verdict.get(lv), dict) or "tier" not in verdict[lv]]
            if missing:
                fails.append(f"{name} has no holdout verdict at {_names(missing)} stakes, so its decisions there are "
                             f"not capped by holdout evidence")
            n = _holdout_rows(h.summary)
            if n is None:
                warns.append(f"{name}: its bundle does not record how many holdout rows it was scored on")
            else:
                sizes[name] = n
                if n < k:
                    warns.append(f"{name} was scored on {n} trusted holdout rows, fewer than policy.min_verify_rows "
                                 f"{k}: acting is not verified, so its decisions stay at confirm or below")
            raw = ((h.report or {}).get("policy") or {}).get("config")
            if not isinstance(raw, dict):
                warns.append(f"{name}: its holdout report does not say which thresholds it was verified with")
            else:
                drift = _act_drift(cfg, h.spec, PL.PolicyConfig.from_dict(raw))
                if drift:
                    fails.append(f"{name} was verified under other thresholds than {ctx.version} serves ({drift}): "
                                 f"retrain it so that what is served is what was verified")
        except Exception as e:
            fails.append(f"{name}: {_raised(e, ctx.error_text)}")
    if fails:
        return "fail", "; ".join(fails + warns)
    if warns:
        return "warn", "; ".join(warns)
    fewest = min(sizes.items(), key=lambda kv: kv[1])
    return "pass", (f"each of the {len(trained)} trained heads has a holdout verdict for every stakes level "
                    f"({_names(list(cfg.stakes))}), computed under the act thresholds {ctx.version} serves, on at "
                    f"least {k} trusted holdout rows (fewest: {fewest[0]}, {fewest[1]})")


def _act_drift(served: PL.PolicyConfig, spec: Any, verified: PL.PolicyConfig) -> str | None:
    """How the act thresholds `served` gives one head differ from those its holdout verdict was computed with (the
    verdict depends on act, min_verify_rows and miscalibration_p, not on confirm or min_accuracy); None if alike."""
    labels = (spec.labels[1],) if spec.decision == "gate" else tuple(spec.labels)   # a gate commits to `true`
    diffs: list[str] = []
    for lv in served.stakes:
        if lv not in verified.stakes:
            continue                      # reported as a stakes level without a verdict
        pairs = {c: (verified.level_for(spec.name, c, lv).act, served.level_for(spec.name, c, lv).act)
                 for c in labels}
        changed = {c: p for c, p in pairs.items() if p[0] != p[1]}
        if not changed:
            continue
        if len(changed) == len(pairs) and len(set(changed.values())) == 1:
            a, b = next(iter(changed.values()))
            diffs.append(f"act at {lv} stakes: verified {a:g}, served {b:g}")
        else:
            diffs.append(f"act at {lv} stakes for " + ", ".join(f"{c} verified {a:g}, served {b:g}"
                                                                 for c, (a, b) in changed.items()))
    for key in ("min_verify_rows", "miscalibration_p"):
        a, b = getattr(verified, key), getattr(served, key)
        if a != b:
            diffs.append(f"{key}: verified {a:g}, served {b:g}")
    return "; ".join(diffs) or None


@_check("thresholds.config", "rlcd.yaml thresholds are the ones served")
def _policy_config(ctx: _Ctx) -> Outcome:
    b = ctx.service.bundle(ctx.version)
    if b is None or not b.meta.get("policy"):
        return "info", f"{ctx.version} records no policy of its own: rlcd.yaml `policy` is served as written"
    drift = policy_drift(ctx.service, ctx.version)
    if drift is None:
        return "pass", f"rlcd.yaml `policy` is the policy {ctx.version} was trained and verified with"
    return "warn", (f"rlcd.yaml differs from what {ctx.version} serves (served -> rlcd.yaml: {drift}): a version "
                    f"keeps the thresholds it was verified with, so the new ones apply only after `rlcd train` "
                    f"verifies them on a holdout")


def policy_drift(service: Any, version: str) -> str | None:
    """How rlcd.yaml differs from what `version` serves, as 'key served -> configured; ...': its `policy`, and the
    `stakes` and `conformal` of each trained head (a version serves the policy stored in its bundle and its heads
    as they were trained). None when they agree, or when the version stores no policy (rlcd.yaml is then served as
    written). `rlcd check` (thresholds.config), `rlcd decide`, `rlcd ask` and `rlcd serve` all report this one
    text."""
    b = service.bundle(version)
    if b is None or not b.meta.get("policy"):
        return None
    served, configured = service.policy(version).to_dict(), service.policy_config().to_dict()
    parts = [_diff(served, configured)] if digest(served) != digest(configured) else []
    try:
        wanted = service.pack_for(version).heads(service.project)
    except Exception:       # the pack cannot build its heads from this file: `config.valid` reports that
        wanted = {}
    for name in b.names():
        spec = wanted.get(name)
        if spec is None:
            continue
        trained = b.head(name).spec
        for field in ("stakes", "conformal"):
            was, now = getattr(trained, field), getattr(spec, field)
            if now != was:
                parts.append(f"heads.{name}.{field} {was!r} -> {now!r}")
    return "; ".join(parts) or None


def _flat(d: Any, prefix: str = "") -> dict[str, Any]:
    if not isinstance(d, dict) or not d:
        return {prefix.rstrip("."): d} if prefix else {}
    out: dict[str, Any] = {}
    for k, v in d.items():
        out.update(_flat(v, f"{prefix}{k}."))
    return out


def _diff(a: dict, b: dict, limit: int = 6) -> str:
    fa, fb = _flat(a), _flat(b)
    keys = [k for k in sorted(set(fa) | set(fb)) if fa.get(k) != fb.get(k)]
    shown = [f"{k} {fa.get(k)!r} -> {fb.get(k)!r}" for k in keys[:limit]]
    if len(keys) > limit:
        shown.append(f"{len(keys) - limit} more")
    return "; ".join(shown)


# ----------------------------------------------------------------------------- the question contract

def _is_escape(option: str) -> bool:
    o = re.sub(r"[\s\-]+", "_", str(option).strip().casefold())
    return o in NO_MATCH or o.startswith(NO_MATCH_PREFIXES)


def _description_key(text: Any) -> str:
    """A level / option description as compared for overlap: index prefix dropped, case and spacing folded."""
    t = " ".join(str(text).split()).casefold()
    return _INDEX.sub("", t).strip(" .;:")


def _vague(text: Any, cls: str) -> bool:
    """Does a score level description lack an observable anchor (empty, just its name, only ranking words)?"""
    t = _description_key(text)
    words = _WORD.findall(t)
    if not words or t == str(cls).strip().casefold():
        return True
    return all(w in VAGUE_WORDS or w.isdigit() for w in words)


@_check("options.no_match", "Choice heads have a no-match option")
def _no_match(ctx: _Ctx) -> Outcome:
    choice = {n: s for n, s in ctx.specs.items() if s.type == "choice"}
    if not choice:
        return "info", "no choice heads"
    escapes = {n: [c for c in s.classes if _is_escape(c)] for n, s in choice.items()}
    missing = [n for n, e in escapes.items() if not e]
    if missing:
        return "warn", (f"no escape option in {_names(missing)} (none, other, unknown, out_of_scope, no_match, "
                        f"needs_review, or an option named escalate* / clarify*): a closed choice must pick one of its "
                        f"options even when none fits, and the winner can look confident although the state never "
                        f"supported it. Add one and label the cases that fit no option")
    return "pass", "every choice head can say that no option fits: " + "; ".join(
        f"{n} ({_names(e)})" for n, e in escapes.items())


@_check("score.anchors", "Score levels have observable anchors")
def _anchors(ctx: _Ctx) -> Outcome:
    score = {n: s for n, s in ctx.specs.items() if s.type == "score"}
    if not score:
        return "info", "no score heads"
    problems = []
    for n, s in score.items():
        if not s.levels:
            problems.append(f"{n} has no level descriptions, only the level names {list(s.classes)}")
            continue
        vague = [lv for lv, cls in zip(s.levels, s.classes, strict=True) if _vague(lv, cls)]
        if vague:
            problems.append(f"{n}: {_names([repr(v) for v in vague])}")
    if problems:
        return "warn", ("levels without an observable anchor (what a reviewer can check in the state, not only low / "
                        "medium / high or good / bad): " + "; ".join(problems))
    return "pass", f"every level of {_names(list(score))} describes what it looks like"


@_check("heads.described", "Heads and options are described")
def _described(ctx: _Ctx) -> Outcome:
    specs = ctx.specs
    if not specs:
        return "info", "no heads"
    no_desc = [n for n, s in specs.items() if not str(s.description or "").strip()]
    no_opts = [n for n, s in specs.items() if s.type == "choice" and (not s.levels or all(
        _description_key(d) == _description_key(str(c).replace("_", " ")) or _description_key(d) == _description_key(c)
        for c, d in zip(s.classes, s.levels, strict=True)))]     # none, or each one only repeats its option's name
    if not no_desc and not no_opts:
        return "pass", f"all {len(specs)} heads have a description, and every choice head describes its options"
    parts = []
    if no_desc:
        parts.append(f"no description: {_names(no_desc)}")
    if no_opts:
        parts.append(f"choice heads without option descriptions (`levels`): {_names(no_opts)}")
    return "warn", ("; ".join(parts) + ". The descriptions are the question contract that labellers and LLM judges "
                    "read: without them two people can label the same case differently")


@_check("noul.positive", "Noul heads say what yes means")
def _positive(ctx: _Ctx) -> Outcome:
    nouls = {n: s for n, s in ctx.specs.items() if s.type == "noul"}
    if not nouls:
        return "info", "no noul heads"
    missing = [n for n, s in nouls.items() if not str(s.positive or "").strip()]
    if missing:
        return "warn", (f"{_names(missing)}: no `positive` saying what yes means, so the yes condition can change "
                        f"from one reviewer to another")
    return "pass", f"every noul head ({_names(list(nouls))}) says what yes means"


@_check("options.overlap", "Options of one head are distinct")
def _overlap(ctx: _Ctx) -> Outcome:
    described = {n: s for n, s in ctx.specs.items() if s.levels}
    if not described:
        return "info", "no head describes its options or levels, so there is nothing to compare"
    problems = []
    for n, s in described.items():
        seen: dict[str, list[str]] = {}
        for cls, d in zip(s.classes, s.levels, strict=True):
            seen.setdefault(_description_key(d), []).append(cls)
        same = [v for v in seen.values() if len(v) > 1]
        if same:
            problems.append(f"{n}: " + "; ".join(" / ".join(v) for v in same))
    if problems:
        return "warn", ("options with the same description (overlapping options split one behaviour between two "
                        "labels): " + "; ".join(problems))
    return "pass", f"no two options or levels share a description ({_names(list(described))})"


@_check("inputs.required", "Missing inputs fail closed")
def _inputs(ctx: _Ctx) -> Outcome:
    text = ("a question whose `path` / `inputs` do not resolve in the state is a 422 missing_evidence, never a guess. "
            "A state that lacks features almost every training row had is capped at confirm on /v1/decide and "
            "/v1/evaluate and answered with a warning on /v1/systemone; `strict: true` refuses it (422) on all "
            "three. That guard sees only features that come out missing: an optional field left out of the state can "
            "read as an ordinary value instead and change the answer without a warning (on /v1/systemone and "
            "/v1/rank the documents pack reads absent candidates as zero candidates; /v1/decide retrieves them from "
            "the corpus, or warns when none is ingested). Your state builder still validates required fields and "
            "fails closed when a record cannot be loaded")
    try:
        questions = [q for b in ctx.batteries.values() for q in b.questions()]
    except ContractError:
        questions = []       # batteries.bound reports a broken battery configuration
    if questions:
        named = sum(1 for q in questions if q.inputs.to_dict())
        text += (f". {named} of {len(questions)} battery question(s) name their inputs (`path` / `inputs`); the "
                 f"others read the request state as their head's own view")
    return "info", text


@_check("deterministic.first", "Deterministic rules run before any head")
def _deterministic(ctx: _Ctx) -> Outcome:
    return "info", ("open-rlcd cannot see your validators: run schema, identity, capability and allowlist checks "
                    "before calling it; remove the options code rules out (`allowed`, route contracts) before the "
                    "choice; send your output checks as `checks` to /v1/evaluate, where a failed check rejects without "
                    "asking a head")


@_check("routes.contracts", "Route contracts match the heads")
def _routes(ctx: _Ctx) -> Outcome:
    book = ctx.service.routes()
    if not book.routes:
        return "info", ("no route contracts (rlcd.yaml `routes:`): when a choice option selects a provider, model or "
                        "tool, declare what it may receive (allowed_data), needs (requires) and falls back to, so "
                        "deterministic filters run before the choice")
    specs = ctx.specs
    # a head the project switches off is not served (Service.specs leaves it out), so its contracts are kept and
    # never consulted, as config check says too (a pack that serves such a head anyway has it in specs: checked)
    off = {n for n in ctx.off if n not in specs}
    served = [h for h in book.routes if h not in off]
    missing, refused = [], []
    for head in served:
        spec = specs.get(head)
        if spec is None:
            missing.append(f"routes.{head}: {ctx.version} has no head {head!r}")
            continue
        try:
            book.check_head(head, spec)
        except ContractError as e:
            refused.append(e.message)
    if missing or refused:
        parts = []
        if refused:
            parts.append("; ".join(refused) + " (a /v1/decide that applies route contracts to the head answers "
                                              "500 bad_config)")
        if missing:
            parts.append("; ".join(missing) + " (no decision consults these contracts: train the head or remove "
                                              "them)")
        return "fail", "; ".join(parts)
    any_data = [f"{h}.{o}" for h in served for o, c in book.routes[h].items()
                if "allowed_data" not in c and c.get("enabled") is not False]
    uncovered = [f"{h}.{o}" for h in served for o in specs[h].classes if o not in book.routes[h]]
    text = f"contracts for {_names(served)}" if served else "no contracts for a served head"
    if any_data:
        text += f"; accept every data class (no allowed_data): {_names(any_data[:10])}"
    if uncovered:
        text += f"; options without a contract (never filtered): {_names(uncovered[:10])}"
    kept = [h for h in book.routes if h in off]
    if kept:
        text += f"; kept but not consulted while the head is switched off in rlcd.yaml: {_names(kept)}"
    return "pass", text


# ----------------------------------------------------------------------------- security and secrets

@_check("security.api_key", "The API is not open off loopback")
def _api_key(ctx: _Ctx) -> Outcome:
    host = str(ctx.project.get("serve.host", "127.0.0.1") or "127.0.0.1")
    key = bool(os.environ.get("RLCD_API_KEY"))
    if is_loopback(host):
        tail = "RLCD_API_KEY is set" if key else "set RLCD_API_KEY before serving on any other interface"
        return "pass", f"serve.host {host} is loopback (`rlcd serve --host` overrides it); {tail}"
    if not key:
        return "fail", (f"serve.host is {host}, not loopback, and RLCD_API_KEY is not set: anyone who can reach the "
                        f"port can query, read receipts, retrain and switch automation back on. Set RLCD_API_KEY "
                        f"(and RLCD_ADMIN_KEY for training and the kill switch)")
    return "pass", f"serve.host {host} is not loopback and RLCD_API_KEY is set"


@_check("security.allow_remote", "Connectors and judges stay on loopback")
def _allow_remote(ctx: _Ctx) -> Outcome:
    sources = []
    if os.environ.get("RLCD_ALLOW_REMOTE") == "1":
        sources.append("RLCD_ALLOW_REMOTE=1")
    if ctx.project.section("security").get("allow_remote"):
        sources.append("security.allow_remote")
    if ctx.project.pack_config.get("allow_remote"):
        sources.append("pack_config.allow_remote")
    if sources:
        return "warn", (f"remote hosts are allowed ({_names(sources)}): connectors and LLM judges may send documents, "
                        f"queries and labels to non-loopback hosts, and the claude-cli / anthropic judges to Anthropic. Make sure every such host is an approved route for "
                        f"the data it receives")
    return "pass", ("connectors and LLM judges only talk to loopback hosts and unix sockets (the claude-cli and "
                    "anthropic judges are refused)")


@_check("security.admin_key", "The operator's controls need their own key")
def _admin_key(ctx: _Ctx) -> Outcome:
    if os.environ.get("RLCD_ADMIN_KEY"):
        return "info", ("RLCD_ADMIN_KEY is set: POST /v1/train, POST /v1/automation and POST "
                        "/v1/receipts/{id}/confirm need it, so a decision caller can neither retrain, switch "
                        "automation back on nor confirm its own confirm decisions")
    text = ("RLCD_ADMIN_KEY is not set: whoever holds RLCD_API_KEY (or anyone, without one) can retrain "
            "(POST /v1/train), switch automation back on (POST /v1/automation) and confirm confirm decisions (POST "
            "/v1/receipts/{id}/confirm), so a person's confirmation is not enforced")
    confirms = ctx.scan["confirms"]
    if confirms:
        return "warn", (f"{text}: the log holds {confirms} confirm decision(s) that the caller who asked for them "
                        f"could confirm itself")
    return "info", text


@_check("secrets.redact", "Secrets are redacted from logs")
def _redact(ctx: _Ctx) -> Outcome:
    kinds = ctx.service.redact_kinds()
    if not kinds:
        return "warn", ("serve.redact is off: a credential inside a state, a note or a reason is written verbatim to "
                        "decisions.jsonl, feedback.jsonl and the other append-only logs, which are never rotated")
    text = (f"serve.redact: {_names(kinds)}, applied to what the logs keep: states, audit fields and warnings "
            f"(decisions.jsonl), feedback, outcomes, confirmations (who, note), the kill switch and the registry "
            f"history (who, reason). Pattern-based: a bare high-entropy string is kept, so the state builder must "
            f"still keep secrets out of states")
    if "pii" not in kinds:
        text += ("; e-mail addresses and card numbers are kept, reviewers' addresses included (serve.redact: "
                 "[secrets, pii] masks them)")
    return "pass", text


@_check("secrets.log_states", "What the decision log keeps")
def _log_states(ctx: _Ctx) -> Outcome:
    if ctx.project.get("serve.log_states", True) is not False:
        return "info", ("serve.log_states is on: request states are kept (redacted) in decisions.jsonl so feedback can "
                        "train from a decision_id and decisions can be replayed; `rlcd receipts prune --before DATE` "
                        "drops old states and keeps their digests")
    return "info", ("serve.log_states is off: decisions.jsonl keeps state digests only, so feedback must send the "
                    "state itself and logged decisions cannot be replayed")


# ----------------------------------------------------------------------------- receipts, bundles, rollback

@_check("receipts.chain", "The receipt chain verifies")
def _receipts(ctx: _Ctx) -> Outcome:
    res = ctx.service.verify_receipts()
    conf = res.get("confirmations") or {}
    breaks = [("decisions.jsonl", b) for b in res.get("breaks") or []]
    breaks += [("confirmations.jsonl", b) for b in conf.get("breaks") or []]
    legacy = f"; {res['legacy']} older lines carry no receipt" if res.get("legacy") else ""
    if breaks:
        where, first = breaks[0]
        files = list(dict.fromkeys(w for w, _ in breaks))
        return "fail", (f"{len(breaks)} break(s) in {' and '.join(files)}, first at line {first.get('line')}"
                        f"{f' of {where}' if len(files) > 1 else ''} (decision {first.get('decision_id')}): "
                        f"{first.get('why')}. `rlcd receipts verify` lists them all")
    unreadable = [f"decisions.jsonl line {n}" for n in res.get("unreadable") or []]
    unreadable += [f"confirmations.jsonl line {n}" for n in conf.get("unreadable") or []]
    torn = (f"; {len(unreadable)} unreadable line(s): {_names(unreadable[:5])} (a write torn by a crash, or an edit)"
            if unreadable else "")
    if not res.get("receipts"):
        if conf.get("receipts"):
            n = conf["receipts"]
            return "warn", (f"{n} confirmation{'' if n == 1 else 's'} chained (head {conf.get('head')}), but "
                            f"decisions.jsonl holds no receipt for the decisions they confirm (archived, truncated "
                            f"or replaced?){torn}")
        text = "no receipts yet" + (f" ({res['lines']} lines without one)" if res.get("lines") else
                                    ": no decision has been logged")
        return ("warn" if unreadable else "info"), text + torn
    # the full digests: an anchor is only as strong as what is written down
    text = f"{res['receipts']} receipts chained, head {res.get('head')}"
    if conf.get("receipts"):
        n = conf["receipts"]
        text += f"; {n} confirmation{'' if n == 1 else 's'} chained, head {conf.get('head')}"
    text += (f": anchor {'both head digests' if conf.get('receipts') else 'the head digest'} outside RLCD_HOME to "
             f"detect truncation{legacy}")
    return ("warn", text + torn) if unreadable else ("pass", text)


@_check("receipts.mode", "Which calls leave a receipt")
def _receipts_mode(ctx: _Ctx) -> Outcome:
    if ctx.service.receipts_mode() == "all":
        return "info", "serve.receipts: all: every /v1/decide, /v1/evaluate, /v1/systemone and /v1/rank call is recorded"
    return "info", ("serve.receipts: decisions: /v1/decide and /v1/evaluate are recorded; /v1/systemone and /v1/rank "
                    "only when the request asks for a receipt")


@_check("model.bundle", "The checked version's bundle validates")
def _model_bundle(ctx: _Ctx) -> Outcome:
    b = ctx.service.bundle(ctx.version)
    if b is None:
        return "info", f"{ctx.version} is the untrained service: there is no bundle to validate"
    try:
        meta = BD.validate(b.path)
    except BD.BundleError as e:
        return "fail", f"the bundle of {ctx.version} does not validate: {e}"
    return "pass", (f"{ctx.version}: every file of its {len(meta.get('heads') or {})} heads matches its sha256 in "
                    f"model.json (model.json itself, policy included, is not hash-covered)")


@_check("rollback.ready", "There is a version to roll back to")
def _rollback(ctx: _Ctx) -> Outcome:
    registry = ctx.service.registry
    reg = registry.registry()
    if not reg.get("versions"):
        return "warn", "no trained version yet: there is nothing to roll back to"
    latest = registry.latest_id()
    # only an operator's move rehearses the rollback: every train also appends a history entry (kind "train")
    moves = sum(1 for h in reg.get("history") or [] if isinstance(h, dict) and h.get("kind") in ("promote", "rollback"))
    drill = (f"rlcd-latest has been moved {moves} time(s) by promote / rollback" if moves else
             "rlcd-latest has never been moved by promote / rollback: rehearse `rlcd rollback` and `rlcd promote` "
             "before launch")
    prev = registry.previous_latest()
    if prev is None:
        return "warn", (f"rlcd-latest is {latest} and there is no earlier version: `rlcd rollback` has no target "
                        f"until another version is trained or imported; {drill}")
    try:
        b = registry.bundle(prev)
        BD.validate(b.path)
        ctx.service.pack_for(prev)
        for name in b.names():
            h = b.head(name)
            _ = h.summary              # hash-checked head.json: the verdict the head enforces
            h.holdout_ledger()         # the table its Bernoulli gate restarts from
    except (BD.BundleError, ContractError, OSError, ValueError, KeyError) as e:
        why = e.message if isinstance(e, ContractError) else f"{type(e).__name__}: {e}"
        return "fail", f"`rlcd rollback` would point rlcd-latest at {prev}, which does not load ({why})"
    return "pass", (f"`rlcd rollback` would point rlcd-latest ({latest}) at {prev}: its bundle validates and the "
                    f"verdicts and holdout tables of its {len(b.names())} heads load; {drill}")


# ----------------------------------------------------------------------------- rollout and kill switch

@_check("rollout.mode", "Rollout mode")
def _rollout(ctx: _Ctx) -> Outcome:
    ro = ctx.service.rollout()
    text = {
        "full": "full: no rollout cap, the policy alone decides what acts",
        "shadow": "shadow: decisions are made, logged and returned, nothing is automated (capped at confirm)",
        "assist": "assist: every decision is a recommendation a person confirms (capped at confirm)",
        "limited": (f"limited: act only at stakes {_names(ro.limited_stakes) or 'none'}"
                    + (f" for heads {_names(ro.limited_heads)}" if ro.limited_heads else "")
                    + ", confirm everywhere else"),
    }[ro.mode]
    if ro.head_modes:
        text += "; per head: " + ", ".join(f"{h} {m}" for h, m in ro.head_modes)
    try:
        # a head the project switches off is not served at all, so a mode set for it holds nothing back
        latest = ctx.service.resolve(None)
        known = set(ctx.service.specs(latest)) | ctx.service.switched_off(latest)
    except Exception:
        known = set()
    named = {h for h, _ in ro.head_modes} | set(ro.limited_heads or ())
    unknown = sorted(named - known) if known else []
    if unknown:
        # a misspelled head stays at the global mode: the operator believes it is held back while it acts
        return "fail", (f"rollout names heads the model does not have: {unknown} (heads: {sorted(known)}); "
                        f"{text}")
    return "info", text


@_check("automation.switch", "The kill switch")
def _switch(ctx: _Ctx) -> Outcome:
    st = ctx.service.switch.state()
    if not st.enabled:
        who = "".join(f" {k} {v}" for k, v in (("by", st.by), ("at", st.at)) if v)
        how = ("it is forced off by RLCD_AUTOMATION=off in this environment" if st.forced_by_env
               else "`rlcd automation on` switches it back on")
        return "warn", (f"automation is switched off ({st.reason or 'no reason given'}{',' if who else ''}{who}): every "
                        f"decision is capped at confirm; {how}")
    if not os.access(ctx.service.home, os.W_OK):
        return "warn", ("automation is on, but RLCD_HOME is not writable from here: the kill switch cannot be flipped "
                        "from this machine")
    text = ("automation is on; `rlcd automation off --reason ...` (or POST /v1/automation) caps every decision at "
            "confirm at once, without a model update or a restart")
    try:
        offs = sum(1 for h in ctx.service.switch.history(10_000) if isinstance(h, dict) and h.get("enabled") is False)
    except (OSError, ValueError):
        offs = 0
    return "info", text + (f"; it has been switched off {offs} time(s) before" if offs else
                           "; it has never been switched off: rehearse `rlcd automation off` and `on` before launch")


# ----------------------------------------------------------------------------- batteries

@_check("batteries.bound", "Battery questions bind to trained heads")
def _batteries_bound(ctx: _Ctx) -> Outcome:
    bats = ctx.batteries
    if not bats:
        return "info", "no batteries (pack.yaml / rlcd.yaml `batteries:`): System One callers list their questions"
    specs, trained = ctx.specs, ctx.trained
    broken: list[str] = []
    warns: list[str] = []
    untrained: list[str] = []
    for b in bats.values():
        ref = f"{b.name}@{b.version}"
        if not b.owner:
            warns.append(f"{ref} has no owner")
        for q in b.questions():
            spec = specs.get(q.head)
            if spec is None:
                broken.append(f"{ref} question {q.id}: {ctx.version} has no head {q.head!r}")
                continue
            try:
                notes = q.bind(spec)
            except ContractError as e:
                broken.append(f"{ref}: {e.message}")
                continue
            warns += [f"{ref}: {n}" for n in notes]
            if q.head not in trained:
                untrained.append(f"{ref} {q.id} ({q.head})")
    if untrained:
        warns.append(f"questions whose head is not trained (it answers with the uninformed prior): "
                     f"{_names(untrained)}")
    refs = _names([f"{b.name}@{b.version}" for b in bats.values()])
    if broken:
        return "fail", "; ".join(broken + warns) + " (every call naming the battery is refused)"
    if warns:
        return "warn", f"{refs}: " + "; ".join(warns)
    return "pass", f"{refs}: every question binds to a trained head of {ctx.version}"


@_check("batteries.versions", "Battery changes come with a version bump")
def _battery_versions(ctx: _Ctx) -> Outcome:
    recorded = ctx.scan["batteries"]
    try:
        bats = ctx.batteries
    except ContractError:
        return "info", "the battery configuration does not load (batteries.bound says why): nothing to compare"
    try:
        every_call = ctx.service.receipts_mode() == "all"
    except ContractError:
        every_call = False      # receipts.mode reports the bad setting
    partial = ("" if every_call else
               "with serve.receipts: decisions, /v1/systemone leaves a receipt (and so a battery version and digest "
               "to compare) only when the request sends \"receipt\": true; set serve.receipts: all to compare every "
               "battery call")
    if not recorded:
        if not bats:
            return "info", "no battery is defined, and no receipt names one"
        if every_call:
            return "info", "no receipt names a battery yet"
        return "warn", (f"no receipt names a battery, so a battery changed without a version bump goes unnoticed: "
                        f"{partial}")
    changed: list[str] = []
    older: list[str] = []
    same = 0
    for (name, ver), digests in sorted(recorded.items()):
        b = bats.get(name)
        if b is None or ver != b.version:
            older.append(f"{name}@{ver}" + (" (no longer defined)" if b is None else ""))
            continue
        current = b.digest
        for d, (n, first) in digests.items():
            if d == current:
                same += n
            else:
                changed.append(f"{name}@{ver}: {n} receipt(s) recorded digest {short(d)} (first: decision {first}), "
                               f"it is {short(current)} now")
    if changed:
        return "fail", ("changed without a version bump: " + "; ".join(changed) + ". Bump the version (and the "
                        "changelog) whenever a question's meaning changes, so a caller pinned to the old version gets "
                        "a 422 instead of new semantics under the old name")
    text = f"{same} receipt(s) recorded the current digest of their battery version"
    if older:
        text += f"; receipts of earlier versions: {_names(older)}"
    if partial:
        text += f" (only calls that asked for a receipt are compared: {partial})"
    return "pass", text


# ----------------------------------------------------------------------------- regression cases

def _regress() -> Any | None:
    """core/regress.py, or None when it is not installed (or lacks the functions this check calls)."""
    try:
        mod = importlib.import_module(REGRESS_MODULE)
    except ModuleNotFoundError as e:
        if e.name == REGRESS_MODULE:
            return None
        raise                         # the module is there but needs something that is not: a broken install
    if not all(callable(getattr(mod, f, None)) for f in ("cases_path", "load_cases", "run_cases")):
        return None
    return mod


def _cases_file(project: Any) -> Path | None:
    """The cases file by the rules of core/regress.py `cases_path`, for when that module is missing: rlcd.yaml
    `regressions:` (relative to the project root), else <project root>/regressions.jsonl. None: not a path."""
    raw = project.get("regressions")
    if raw is None or raw == "":
        raw = REGRESSIONS_FILE
    return project.resolve_path(raw) if isinstance(raw, (str, os.PathLike)) else None


def _shown(path: Path, project: Any) -> str:
    try:
        return str(Path(path).resolve().relative_to(Path(project.root).resolve()))
    except ValueError:
        return Path(path).name


@_check("regressions.cases", "Regression cases pass")
def _regressions(ctx: _Ctx) -> Outcome:
    project = ctx.project
    configured = project.get("regressions")
    mod = _regress()
    path = mod.cases_path(project) if mod is not None else _cases_file(project)
    if path is None:
        return "warn", "rlcd.yaml `regressions:` must be the path of the regression cases file (JSON lines)"
    if not Path(path).is_file():
        if configured not in (None, ""):
            return "warn", f"rlcd.yaml `regressions:` names {configured}, which is not a file"
        return "info", (f"no regression cases ({REGRESSIONS_FILE} in the project, or rlcd.yaml `regressions:`): every "
                        f"incident should leave one, so each new version is checked against what went wrong before")
    where = _shown(path, project)
    if mod is None:
        n = sum(1 for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip())
        return "info", (f"{n} regression case(s) in {where}; the regression runner ({REGRESS_MODULE}) is not "
                        f"installed here, so they were not run")
    cases = mod.load_cases(path)          # a line that is not a valid case comes back as an error to report
    if not cases:
        return "info", f"{where} holds no regression case yet"
    with ctx.service.dry_run():           # run_cases runs in a dry run itself: nothing is logged either way
        res = mod.run_cases(ctx.service, cases, model=ctx.version)
    results = res.get("results") if isinstance(res, dict) else None
    if not isinstance(results, list):
        return "warn", f"the regression runner returned a result this check cannot read for the cases of {where}"
    bad = [r for r in results if not (isinstance(r, dict) and r.get("ok") is True)]
    if bad:
        errors = sum(1 for r in bad if isinstance(r, dict) and r.get("status") == "error")
        shown = "; ".join(f"{r.get('id')}: {r.get('why')}" if isinstance(r, dict) else str(r) for r in bad[:3])
        more = f"; and {len(bad) - 3} more" if len(bad) > 3 else ""
        judged = f" ({errors} of them could not be judged, which counts as failing)" if errors else ""
        return "fail", f"{len(bad)} of {len(results)} case(s) in {where} fail on {ctx.version}{judged}: {shown}{more}"
    return "pass", (f"all {len(results)} case(s) in {where} pass on {ctx.version} (judged as trained: rollout full, "
                    f"kill switch on, each version's own holdout gate)")


# ----------------------------------------------------------------------------- the decision log: review load

def _scan(service: Any, now: datetime) -> dict:
    """One pass over decisions.jsonl: the battery refs receipts recorded, the confirm decisions of the whole log, and
    per UTC day the confirm and abstain decisions of the last REVIEW_WINDOW_DAYS days up to `now` (counted as
    core/ops.py counts its review load)."""
    since = now - timedelta(days=REVIEW_WINDOW_DAYS)
    per_day: dict[str, list[int]] = {}           # UTC date -> [confirm, abstain]
    abstained_next: dict[str, int] = {}          # where the abstentions went: the decision's `next`
    capped = window = confirms = 0
    first: datetime | None = None
    bats: dict[tuple[str, str], dict[str, list]] = {}
    for row in service.logs.decisions():
        if not isinstance(row, dict):
            continue
        r = row.get("receipt") if isinstance(row.get("receipt"), dict) else {}
        q = r.get("questions") if isinstance(r.get("questions"), dict) else {}
        bat = q.get("battery")
        if isinstance(bat, dict) and bat.get("name") and bat.get("digest"):
            slot = bats.setdefault((str(bat["name"]), str(bat.get("version"))), {}).setdefault(
                str(bat["digest"]), [0, row.get("decision_id")])
            slot[0] += 1
        if row.get("action") in REVIEW_ACTIONS:
            confirms += 1                         # however old: any of them can still be confirmed
        t = _time(row.get("created_at"))
        if t is None or t < since or t > now:
            continue
        window += 1
        first = t if first is None or t < first else first
        action = row.get("action")
        day = t.astimezone(UTC).date().isoformat()
        if action in REVIEW_ACTIONS:
            per_day.setdefault(day, [0, 0])[0] += 1
            ro = r.get("rollout") if isinstance(r.get("rollout"), dict) else {}
            # capped by the rollout or the switch *and* the policy alone would have acted (a head capped inside a
            # plan that confirms anyway is not a review the cap created)
            if ro.get("capped") and ro.get("proposed_action", "act") == "act":
                capped += 1
        elif isinstance(action, str) and action != "act":     # the abstain action, whatever the policy calls it
            per_day.setdefault(day, [0, 0])[1] += 1
            plan = row.get("plan") if isinstance(row.get("plan"), dict) else {}
            nxt = row.get("next") or plan.get("next")           # evaluate records `next`, decide its plan's
            key = nxt if isinstance(nxt, str) else ""
            abstained_next[key] = abstained_next.get(key, 0) + 1
    days = 0.0
    if first is not None:
        days = max(1.0, min(float(REVIEW_WINDOW_DAYS), (now - first).total_seconds() / _DAY_S))
    return {"batteries": bats, "window": window, "per_day": per_day, "capped": capped, "days": days,
            "abstained_next": abstained_next, "confirms": confirms}


# where an abstained decision went next (its `next`): only some of these reach a person
_NEXT_WHERE = {
    "escalate_to_human": "escalated to a person",
    "human_review": "sent to a person",
    "revise": "sent back to the generator",
    "retry_generation": "sent back to the generator",
    "retrieve_more": "sent back for more evidence",
    "ask_clarifying_question": "asked back for clarification",
    "refuse_out_of_scope": "refused as out of scope",
}


def _abstained_where(counts: dict[str, int]) -> str:
    """'2 escalated to a person, 1 sent back to the generator': abstentions by where they went next."""
    where: dict[str, int] = {}
    for nxt, n in counts.items():
        phrase = _NEXT_WHERE.get(nxt) or (f"next {nxt}" if nxt else "with no next step recorded")
        where[phrase] = where.get(phrase, 0) + n
    return ", ".join(f"{n} {p}" for p, n in sorted(where.items(), key=lambda kv: (-kv[1], kv[0])))


@_check("review.capacity", "Human review is staffed for the volume")
def _review(ctx: _Ctx) -> Outcome:
    raw = ctx.project.get("serve.review_capacity_per_day")
    cap: float | None = None
    if raw is not None:
        if isinstance(raw, bool) or not isinstance(raw, (int, float)) or not math.isfinite(raw) or raw <= 0:
            return "fail", f"serve.review_capacity_per_day must be a positive number of reviews a day, got {raw!r}"
        cap = float(raw)
    s = ctx.scan
    if not s["window"]:
        tail = (f" (capacity {cap:g} a day)" if cap is not None else
                "; declare serve.review_capacity_per_day once traffic flows")
        return "info", f"no decisions in the last {REVIEW_WINDOW_DAYS} days to measure the review load against{tail}"
    per_day, days = s["per_day"], s["days"]
    total = sum(v[0] for v in per_day.values())
    abstained = sum(v[1] for v in per_day.values())
    peak_day, peak = max(((d, v[0]) for d, v in per_day.items()), key=lambda kv: (kv[1], kv[0]), default=("", 0))
    mean = total / days
    load = (f"{total} {'/'.join(REVIEW_ACTIONS)} decision(s) over the last {round(days, 2):g} day(s): {mean:.1f} a day "
            f"on average, {peak} on the busiest UTC day" + (f" ({peak_day})" if peak else ""))
    notes = ""
    if abstained:
        notes += f"; {abstained} more abstained ({_abstained_where(s.get('abstained_next') or {})}; not counted as reviews)"
    if s["capped"]:
        notes += (f"; {s['capped']} of the reviews were capped by the rollout or the kill switch (the policy alone "
                  f"would have acted)")
    if cap is None:
        if not total:
            return "info", (f"{load}{notes}; declare serve.review_capacity_per_day before the policy starts sending "
                            f"decisions to review")
        return "warn", (f"{load}{notes}; serve.review_capacity_per_day is not declared, so nothing says the review "
                        f"queue is staffed for this volume")
    over = sorted(d for d, v in per_day.items() if v[0] > cap)
    if over or mean > cap:
        avg = "; so is the average" if mean > cap else ""
        return "warn", (f"{load}; {len(over)} day(s) above the review capacity of {cap:g} a day ({_names(over)}){avg}"
                        f"{notes}. Staff the queue or narrow the traffic this decision path handles; lowering "
                        f"thresholds to shrink the queue automates unverified decisions")
    return "pass", f"{load}: within the review capacity of {cap:g} a day{notes}"


# ----------------------------------------------------------------------------- ownership and failure behaviour

def _named(v: Any) -> bool:
    if isinstance(v, str):
        return bool(v.strip())
    if isinstance(v, (list, tuple)):
        return bool(v) and all(isinstance(x, str) and x.strip() for x in v)
    return False


@_check("owners", "Named owners")
def _owners(ctx: _Ctx) -> Outcome:
    owners = ctx.project.data.get("owners")
    roles = "; ".join(f"{r} ({what})" for r, what in OWNER_ROLES.items())
    if owners is None:
        return "warn", (f"rlcd.yaml has no `owners:`: name who owns {roles}. An integration without owners decays as "
                        f"policies, traffic and vendors change")
    if not isinstance(owners, dict):
        return "warn", f"rlcd.yaml `owners` must map each role to a person or team: {roles}"
    missing = [r for r in OWNER_ROLES if not _named(owners.get(r))]
    if missing:
        return "warn", "no owner named for " + "; ".join(f"{r} ({OWNER_ROLES[r]})" for r in missing)
    return "pass", "; ".join(f"{r}: {_names(owners[r]) if isinstance(owners[r], list) else owners[r]}"
                             for r in OWNER_ROLES)


@_check("failure.behaviour", "Callers fail closed")
def _failure(ctx: _Ctx) -> Outcome:
    return "info", (f"a caller must treat an error, a timeout or an unreachable service as 'review unavailable', never "
                    f"as approval: fall back to proposal-only or a person ({CONTRACT_DOC}, failure behaviour for "
                    f"callers; the Python client's fail_closed does this). open-rlcd cannot check your callers")
