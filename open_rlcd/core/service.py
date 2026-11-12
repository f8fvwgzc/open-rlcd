"""The service: everything the HTTP endpoints (and the CLI) need, without HTTP.

Answers questions bound to heads (System One contract), ranks candidates, makes decisions through the policy,
records decisions as receipts, and takes feedback (immediate Bernoulli update + a training row for the next train).

The control layer around the heads (docs/control-layer.md): versioned question batteries, question inputs read from
one shared state (fail closed on a missing field), options removed by deterministic code before the semantic
choice, uncertainty sources on every decision, the operator's rollout mode and kill switch (they only ever take
automation away), and a hash-chained receipt for every decision that a caller can check for staleness before it
acts.
"""
from __future__ import annotations

import contextlib
import threading
import time
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from .. import LATEST, MODEL_PREFIX, VERSION
from . import battery as BT
from . import ops as OPS
from . import policy as PL
from . import receipts as R
from . import uncertainty as U
from .bernoulli import Ledger
from .bundle import Bundle, LoadedHead
from .config import Project, load_project
from .contract import (
    MAX_CANDIDATES,
    ContractError,
    DecideRequest,
    EvaluateRequest,
    FeedbackRequest,
    OutcomeRequest,
    Question,
    RankRequest,
    SystemOneRequest,
    build_answer,
    parse_questions,
    positive_probability,
    questions_digest,
    restrict,
)
from .digest import digest
from .feedback import FeedbackLog
from .head import HeadSpec
from .redact import kinds_from_config, redact
from .registry import NotServable, Registry, UnknownModel
from .rollout import AutomationSwitch, Rollout, effective_cap
from .routes import Constraints, RouteBook
from .training import now_iso

RECEIPT_MODES = ("decisions", "all")
_MISSING = object()


def _merge(a: dict, b: dict) -> dict:
    out = dict(a)
    for k, v in (b or {}).items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def is_precomputed(state: Any) -> bool:
    """A state is a precomputed feature vector only when it has both `x` and `feature_version` (the rule the packs
    use at training time): a raw state may well have a feature column called `x`."""
    return isinstance(state, dict) and "x" in state and "feature_version" in state


def table_is_live(table: dict | None, version: str | None) -> bool:
    """Does a head's Bernoulli table hold the evidence of `version`'s head? A table belongs to the version that
    trained the head and to every later version that carried that head over unchanged (`valid_for`)."""
    if not table or not version:
        return False
    return table.get("model_version") == version or version in (table.get("valid_for") or [])


def _committed(spec: HeadSpec, a: dict) -> tuple[str, float]:
    """(class, probability) an answer commits to by the policy's rule (`policy.chosen`: a gate head commits to
    `true`, every other head to its most probable allowed class), so an outcome posted for a /v1/systemone receipt
    scores the Bernoulli ledger exactly as one posted for a /v1/decide decision."""
    if a["type"] == "choice":   # the answer's own choice (among the allowed options), never a rounded re-argmax
        return a["choice"], round(float(a["probabilities"].get(a["choice"], 0.0)), 6)
    if a["type"] == "noul":
        p = float(a["noul"])
        vec = [1.0 - p, p]
    else:
        vec = [float(a["probabilities"][str(i)]) for i in range(len(spec.labels))]
    c, p = PL.chosen(spec, np.asarray(vec, float))
    return spec.labels[int(c[0])], round(float(p[0]), 6)


def restrict_questions(questions: list[Question], allowed: dict) -> list[Question]:
    """Apply a request's `allowed: {question id: [options]}` (deterministic filters of this request) to its choice
    questions, intersected with a question's own `allowed` (a battery question keeps its fixed contract; the request
    can only narrow it)."""
    by_id = {q.id: q for q in questions}
    unknown = [k for k in allowed if k not in by_id]
    if unknown:
        raise ContractError(f"allowed names questions this request does not ask: {unknown}", field="allowed",
                            questions=list(by_id))
    for qid, opts in allowed.items():
        q = by_id[qid]
        if q.type != "choice":
            raise ContractError(f"allowed.{qid}: a {q.type} question has no options to remove", field="allowed",
                                question=qid)
        if not isinstance(opts, list) or not opts or not all(isinstance(o, str) for o in opts):
            raise ContractError(f"allowed.{qid} must be a non-empty array of option names", field="allowed",
                                question=qid)
        keep = [o for o in dict.fromkeys(opts) if q.allowed is None or o in q.allowed]
        if not keep:
            raise ContractError(f"allowed.{qid}: none of {opts} is among the question's own allowed options "
                                f"{list(q.allowed or ())}", field="allowed", question=qid)
        q.allowed = tuple(keep)
    return questions


def json_safe(obj: Any) -> Any:
    """A copy with NaN / Infinity as null: a logged state may hold them (JSON parsers accept them), a JSON response
    must not."""
    import math

    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    return obj


AUDIT_FIELDS = ("baseline", "artifact", "checks", "revision", "tags")


def request_audit(extra: dict | None) -> dict | None:
    """What a decide / evaluate caller asks to keep with the decision in its receipt: `baseline` (the route or
    action its existing path would take without open-rlcd: the counterfactual), `artifact` (provenance of the LLM
    output: serving model, provider, finish reason, request digest), `checks` (its deterministic validation
    results), `revision` (the decision this one revises) and `tags` (free-form labels for metrics)."""
    out = {k: (extra or {}).get(k) for k in AUDIT_FIELDS if (extra or {}).get(k) is not None}
    return out or None


def request_allowed(raw: Any, names: Sequence[str]) -> dict[str, list[str]]:
    """A decide request's `allowed`: {choice head: [options deterministic code kept]}."""
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ContractError("allowed must be an object mapping a choice head to the options it may choose",
                            field="allowed")
    unknown = [h for h in raw if h not in names]
    if unknown:
        raise ContractError(f"allowed names heads that take no part in this decision: {unknown}", field="allowed",
                            heads=list(names))
    return {str(k): v for k, v in raw.items()}


class Service:
    def __init__(self, project: Project | None = None, home: Any = None, pack: Any = None):
        self.project = project or load_project()
        if home is not None:   # the registry, the logs and the pack's label store / corpus share one home
            self.project.home_override = Path(home).expanduser().resolve()
        self.registry = Registry(self.project.home)
        self.logs = FeedbackLog(self.registry.home)
        self._packs: dict[str, Any] = {}
        if pack is not None:
            self._packs[pack.name] = pack
            self._default_pack = pack.name
        else:
            self._default_pack = self.project.pack
        self._providers: dict[str, Any] = {}
        self.rng = np.random.default_rng()
        self.switch = AutomationSwitch(self.registry.home)
        self._local = threading.local()      # dry runs (replay, regression cases) record nothing
        self._verified_cache: dict[tuple[str, str], PL.PolicyConfig | None] = {}
        from .jobs import Trainer

        self.trainer = Trainer(self)

    # ------------------------------------------------------------------ lookups
    @property
    def ledger(self) -> Ledger:
        return self.registry.ledger

    @property
    def home(self):
        return self.registry.home

    def load_pack(self, name: str):
        """A pack by the name rlcd.yaml gives it (a built-in name or a `module:Class` path). The instance is cached
        under that key and under its own `.name`, the name bundles record, so a custom pack is found again when a
        trained version is served."""
        if name not in self._packs:
            from ..packs.base import load_pack

            pack = load_pack(name, root=self.project.root)
            self._packs[name] = pack
            self._packs.setdefault(str(getattr(pack, "name", name)), pack)
        return self._packs[name]

    @property
    def pack(self):
        """The project's pack (what `rlcd train` trains)."""
        return self.load_pack(self._default_pack)

    def resolve(self, model: str | None, serve: bool = True) -> str:
        """The versioned id of `model`. With `serve` (every path that answers a request), a weight-edit candidate
        whose holdout gate failed is refused: it is kept for inspection (`serve=False`: reports, exports, the
        calibration view), never used to answer."""
        try:
            version = self.registry.resolve(model)
        except UnknownModel:
            raise ContractError(f"unknown model {model!r}", kind="unknown_model",
                                available=self.registry.available()) from None
        if serve and not self.registry.servable(version):
            raise self._never_serves(version)
        return version

    def _never_serves(self, version: str, reason: str | None = None) -> ContractError:
        """The 422 for a weight-edit candidate whose holdout gate failed, asked to answer or to become latest."""
        reason = reason or ((self.registry.entry(version) or {}).get("edit") or {}).get("reason")
        return ContractError(f"model {version} is a weight-edit candidate that did not pass its holdout gate "
                             f"({reason}); it is kept for inspection and never serves", kind="unknown_model",
                             available=[v for v in self.registry.available()
                                        if v == LATEST or self.registry.servable(v)])

    def bundle(self, version: str) -> Bundle | None:
        return self.registry.bundle(version)

    def pack_for(self, version: str):
        """The pack a version was trained with. A bundle's `pack.name` is data, never an import path: it may only
        select this project's own configured pack (matched by the pack's `.name`) or a built-in pack; anything else
        is refused (a bundle shared by someone else must not make this process import and run code)."""
        from ..packs.base import BUILTIN

        b = self.bundle(version)
        name = (b.meta.get("pack") or {}).get("name") if b is not None else None
        if not name:
            return self.pack
        if not isinstance(name, str):
            raise ContractError(f"model {version}: the bundle's pack name is not a string", kind="bad_bundle",
                                version=version)
        try:
            own = self.pack
        except Exception:
            if name in BUILTIN:     # the project's pack cannot load here, but the bundle names a built-in one
                return self.load_pack(name)
            raise
        if name == getattr(own, "name", None):
            return own
        if name in BUILTIN:
            return self.load_pack(name)
        raise ContractError(f"model {version} was trained with pack {name!r}, which is neither this project's pack "
                            f"({getattr(own, 'name', self._default_pack)!r}) nor a built-in pack {sorted(BUILTIN)}. A "
                            f"bundle never names code to import: set `pack:` in rlcd.yaml to the pack it was trained "
                            f"with", kind="bad_bundle", status=422, version=version)

    def specs(self, version: str) -> dict[str, HeadSpec]:
        """The pack's current heads, overlaid by the heads this version was trained with. A head the project switches
        off (`switched_off`: `heads.<name>.enabled: false`; in the tabular pack also a target switched off in
        `pack_config` or no longer listed there) is not served even when the version has it trained: switching a
        question off is not a retrain (the trained head stays in the bundle, so switching it on again needs none)."""
        out = dict(self.pack_for(version).heads(self.project))
        b = self.bundle(version)
        if b is not None:
            off = self.switched_off(version)
            for name in b.names():
                if name not in off:
                    out[name] = b.head(name).spec
        return out

    def switched_off(self, version: str) -> set[str]:
        """The heads the project switches off (`packs.base.switched_off`, asked of the pack `version` was trained
        with, given the heads it trained): not served even when `version` has them. `rlcd check` reads the same."""
        from ..packs.base import switched_off

        b = self.bundle(version)
        return switched_off(self.pack_for(version), self.project, b.names() if b is not None else ())

    def untrained_hint(self, version: str) -> str:
        """What to add to an error when `version` is the untrained service ('' for a trained version)."""
        if self.bundle(version) is not None:
            return ""
        return (f" No model is trained yet ({version} is the untrained service): run `rlcd train` in the project "
                f"(or POST /v1/train), then ask again; `rlcd check` says what is missing.")

    def spec(self, head: str, version: str, question: str | None = None) -> HeadSpec:
        specs = self.specs(version)
        if head not in specs:
            raise ContractError(
                f"no head named {head!r}. RLCD only answers questions bound to a trained head: set the "
                f"question's `head` field (or use a head name as the question id)."
                + (self.untrained_hint(version) if not specs else ""), kind="unknown_head",
                question=question, available_heads=list(specs))
        return specs[head]

    def loaded(self, version: str, head: str) -> LoadedHead | None:
        b = self.bundle(version)
        return None if b is None else b.head(head)

    def has_skill(self, version: str, head: str) -> bool:
        """Did the head's trusted holdout show reliable skill over the base rate? False when untrained."""
        h = self.loaded(version, head)
        return bool(h is not None and (h.summary.get("metrics") or {}).get("has_skill"))

    def provider(self, version: str):
        """The feature provider a version was trained with (rebuilt from the bundle's pack config)."""
        if version not in self._providers:
            pack, b = self.pack_for(version), self.bundle(version)
            cfg = ((b.meta.get("pack") or {}).get("config") if b is not None else None)
            self._providers[version] = pack.provider(self.project, cfg)
        return self._providers[version]

    def policy(self, version: str) -> PL.PolicyConfig:
        """The policy `version` serves: the one stored in its bundle (the thresholds it was verified with), else
        rlcd.yaml's. A stored policy that cannot be served (a bundle trained before `rlcd config check` existed, with
        `abstain: act`, or a damaged model.json) fails closed: the version refuses to decide (422 `bad_bundle`)
        instead of handing a caller an abstention named `act`."""
        b = self.bundle(version)
        if b is not None and b.meta.get("policy"):
            try:
                return PL.PolicyConfig.from_dict(b.meta["policy"])
            except (ValueError, TypeError) as e:
                raise ContractError(
                    f"model {version} cannot decide: the policy stored in its bundle is not usable ({e}). It was "
                    f"trained before rlcd.yaml was checked: fix `policy` in rlcd.yaml (`rlcd config check`) and "
                    f"retrain with `rlcd train`, or point rlcd-latest at a sound version (`rlcd rollback`)",
                    kind="bad_bundle", status=422, version=version) from None
        return self.policy_config()

    def policy_config(self) -> PL.PolicyConfig:
        try:
            return PL.PolicyConfig.from_dict(_merge(self.pack.policy_defaults(), self.project.section("policy")))
        except (ValueError, TypeError) as e:
            raise ContractError(f"rlcd.yaml policy: {e}", kind="bad_config", status=500) from None

    def policy_digest(self, version: str) -> str:
        return digest(self.policy(version).to_dict())

    # ------------------------------------------------------------------ control-layer configuration
    def rollout(self) -> Rollout:
        try:
            return Rollout.from_config(self.project.data.get("rollout"))
        except (ValueError, TypeError) as e:
            raise ContractError(f"rlcd.yaml rollout: {e}", kind="bad_config", status=500) from None

    def batteries(self, version: str) -> dict[str, BT.Battery]:
        """The batteries of the pack `version` was trained with, overlaid by rlcd.yaml `batteries:`."""
        pack = self.pack_for(version)
        defaults = pack.defaults() if callable(getattr(pack, "defaults", None)) else {}
        try:
            return BT.load((defaults or {}).get("batteries"), self.project.data.get("batteries"))
        except (ValueError, TypeError) as e:
            raise ContractError(f"batteries: {e}", kind="bad_config", status=500) from None

    def routes(self) -> RouteBook:
        try:
            return RouteBook.from_config(self.project.data.get("routes"))
        except (ValueError, TypeError) as e:
            raise ContractError(f"rlcd.yaml {e}", kind="bad_config", status=500) from None

    def filter_options(self, specs: dict[str, HeadSpec], request: dict | None,
                       heads: Sequence[str] | None = None) -> tuple[dict[str, tuple[str, ...]], dict[str, list]]:
        """Deterministic filters before the semantic choices: the request's `allowed` options intersected with the
        route contracts that survive its `capabilities` / `sensitivity` / `latency_budget_ms`.
        Returns ({head: options kept}, {head: [{option, kept, reason}]}); a head absent from the first map is
        unrestricted. An empty tuple means every option was removed."""
        request = request or {}
        names = list(heads) if heads is not None else list(specs)
        names = [n for n in names if n in specs]
        raw = request_allowed(request.get("allowed"), names)
        book = self.routes()
        cons = Constraints.from_request(request)
        allowed: dict[str, tuple[str, ...]] = {}
        considered: dict[str, list] = {}
        for name in names:
            spec = specs[name]
            keep: list[str] | None = None
            if name in raw:
                keep = list(self.check_allowed(spec, raw[name]))
            if book.has(name):
                book.check_head(name, spec)
                kept, cons_list = book.filter(name, spec.classes, cons)
                if keep is not None:   # options the request removed itself are reported as such
                    cons_list = [c if c["option"] in keep or not c["kept"]
                                 else {**c, "kept": False, "reason": "not in the request's allowed options"}
                                 for c in cons_list]
                    kept = [o for o in kept if o in keep]
                considered[name] = cons_list
                keep = kept
            if keep is not None:
                allowed[name] = tuple(keep)
        return allowed, considered

    def no_option_decision(self, version: str, spec: HeadSpec, state: Any, *, stakes: str | None,
                           warnings: list[str], considered: list | None = None) -> dict:
        """A choice head whose every option deterministic filters removed: it commits to nothing and abstains, so
        the caller escalates. The head's distribution is still reported (what it would have preferred)."""
        cfg = self.policy(version)
        stakes_name = stakes or spec.stakes
        level = cfg.level(stakes_name)
        P, calibrated = self.probabilities(version, spec, [state], warnings)
        ans = build_answer(None, spec, P[0], bool(calibrated), self.has_skill(version, spec.name), allowed=None)
        abstain = cfg.tier_name(0)
        why = "; ".join(f"{c['option']}: {c['reason']}" for c in (considered or []) if not c["kept"])
        msg = (f"head '{spec.name}': deterministic filters removed every option ({why or 'allowed is empty'}): "
               f"nothing to choose from, the decision abstains")
        if msg not in warnings:
            warnings.append(msg)
        d = {"head": spec.name, "decision": spec.decision, "stakes": stakes_name, "class": None, "commit": None,
             "p": 0.0, "tier": abstain, "uncapped_tier": abstain, "max_tier": abstain,
             "thresholds": {**asdict(level), "conformal": None}, "bernoulli": None, "gate": None,
             "calibrated": bool(calibrated), "warnings": [msg],
             "rollout": {"mode": self.rollout().mode_for(spec.name), "automation": self.switch.state().enabled,
                         "cap": cfg.tier_name(effective_cap(self.rollout(), self.switch, stakes=stakes_name,
                                                            head=spec.name)[0]), "capped": False}}
        return {"answer": {**ans, "allowed": [], "excluded_mass": 1.0}, "decision": d,
                "uncertainty": {"sources": ["options_removed"], "margin": None, "entropy": None, "top2": []}}

    def decision_contract(self, kind: str, heads: Sequence[str], request: dict | None, **params: Any) -> dict:
        """The question contract of a decide / evaluate call for its receipt: which heads, the route contracts they
        consult, and every request parameter other than the state that can change the outcome."""
        request = request or {}
        book = self.routes()
        keys = ("allowed", "capabilities", "sensitivity", "latency_budget_ms", "strict")
        body = {"kind": kind, "heads": list(heads), "routes": {h: book.routes[h] for h in heads if book.has(h)},
                "params": {**{k: request.get(k) for k in keys if request.get(k) is not None}, **params}}
        return {"digest": digest(body), "battery": None}

    def attach_contract(self, result: dict, head: str, allowed: tuple[str, ...] | None,
                        considered: list | None) -> None:
        """Put the chosen option's route contract (and its usable fallback) on a head result."""
        if considered is None:
            return
        book = self.routes()
        result["considered"] = considered
        result["contract"] = book.resolve(head, result["decision"]["class"], list(allowed or ()))

    def receipts_mode(self) -> str:
        mode = str(self.project.get("serve.receipts", "decisions") or "decisions")
        if mode not in RECEIPT_MODES:
            raise ContractError(f"rlcd.yaml serve.receipts must be one of {list(RECEIPT_MODES)}, got {mode!r}",
                                kind="bad_config", status=500)
        return mode

    def redact_kinds(self) -> tuple[str, ...]:
        try:
            return kinds_from_config(self.project.get("serve.redact", True))
        except (ValueError, TypeError) as e:
            raise ContractError(f"rlcd.yaml {e}", kind="bad_config", status=500) from None

    @contextlib.contextmanager
    def dry_run(self):
        """Within this block decisions are made but not logged (replay, regression cases, previews)."""
        prev = getattr(self._local, "dry", False)
        self._local.dry = True
        try:
            yield
        finally:
            self._local.dry = prev

    @property
    def is_dry(self) -> bool:
        return bool(getattr(self._local, "dry", False))

    # ------------------------------------------------------------------ features and probabilities
    def features(self, version: str, spec: HeadSpec, states: list[Any], head: LoadedHead | None) -> np.ndarray:
        """Feature rows for `states`; a precomputed {x, feature_names, feature_version} is checked against the
        version's feature schema, anything else goes through the pack's provider."""
        names = head.ensemble.feature_names if head is not None else None
        fv = head.ensemble.feature_version if head is not None else None
        rows: list[np.ndarray | None] = [None] * len(states)
        todo: list[int] = []
        for i, s in enumerate(states):
            if is_precomputed(s):
                rows[i] = self._check_vector(spec, s, names, fv)
            else:
                todo.append(i)
        if todo:
            provider = self.provider(version)
            if fv is not None and provider.feature_version != fv:
                raise ContractError(f"head '{spec.name}' ({version}) was trained on feature version '{fv}' but the "
                                    f"configured provider builds '{provider.feature_version}'",
                                    kind="feature_version_mismatch", expected=fv, got=provider.feature_version)
            try:
                X = provider.transform(spec.view, [states[i] for i in todo])
            except ContractError as e:
                # a state with a vector `x` but no feature_version was most likely meant as precomputed features
                if any(isinstance(states[i], dict) and isinstance(states[i].get("x"), list) for i in todo):
                    raise ContractError(f"state.feature_version is required with state.x (a precomputed feature "
                                        f"vector); read as a raw state instead: {e.message}", head=spec.name) from None
                raise
            for j, i in enumerate(todo):
                rows[i] = X[j]
        return np.vstack([np.asarray(r, np.float32).reshape(1, -1) for r in rows])

    @staticmethod
    def _check_vector(spec: HeadSpec, s: dict, names: list[str] | None, fv: str | None) -> np.ndarray:
        if "feature_version" not in s:
            raise ContractError("state.feature_version is required with state.x", head=spec.name)
        if fv is not None and s["feature_version"] != fv:
            raise ContractError(
                f"feature_version mismatch: head '{spec.name}' was trained on '{fv}', got {s['feature_version']!r}. "
                f"Retrain, or send features of the trained version.", kind="feature_version_mismatch",
                expected=fv, got=s["feature_version"])
        if names is not None and s.get("feature_names") is not None and list(s["feature_names"]) != list(names):
            raise ContractError(f"feature_names do not match the {len(names)} names head '{spec.name}' was trained "
                                f"on (same feature_version, different columns)", kind="feature_names_mismatch")
        x = s["x"]
        if not isinstance(x, (list, tuple)) or (names is not None and len(x) != len(names)):
            n = len(names) if names is not None else "the trained number of"
            raise ContractError(f"x must be an array of {n} numbers (null = missing) in feature_names order",
                                kind="bad_feature_vector")
        try:
            return np.array([np.nan if v is None else float(v) for v in x], np.float32)
        except (TypeError, ValueError):
            raise ContractError("x must contain only numbers or null", kind="bad_feature_vector") from None

    def probabilities(self, version: str, spec: HeadSpec, states: list[Any],
                      warnings: list[str] | None = None,
                      missing_out: list[str] | None = None) -> tuple[np.ndarray, bool]:
        """(P (n, K) in spec.labels order, calibrated). An untrained head answers with the uninformed prior.
        With `warnings`, the input guard adds a warning when the request lacks features (missing or non-finite) that
        almost every training row had; `missing_out`, when given, receives those feature names (the policy caps
        such a decision at confirm)."""
        head = self.loaded(version, spec.name)
        k = len(spec.labels)
        if head is None:
            return np.full((len(states), k), 1.0 / k), False
        X = self.features(version, spec, states, head)
        if warnings is not None or missing_out is not None:
            missing = head.ensemble.missing_inputs(X)
            if missing_out is not None:
                missing_out.extend(m for m in missing if m not in missing_out)
            if missing and warnings is not None:
                groups = self.provider(version).groups(spec.view) if missing else {}
                shown = sorted({groups.get(f, f) for f in missing})
                msg = (f"head '{spec.name}': the request lacks {len(missing)} input feature(s) that almost every "
                       f"training row had ({', '.join(shown[:6])}; e.g. {', '.join(missing[:4])}): the probabilities "
                       f"were calibrated on inputs that had them and may not hold. Send the same fields as in "
                       f"training (GET /v1/heads describes them)")
                if msg not in warnings:
                    warnings.append(msg)
        if warnings is not None:
            # a provider may say more about raw inputs than "missing" (a value that is not a number, a category
            # no training row had): optional hook `input_warnings(view, states) -> list[str]`
            hook = getattr(self.provider(version), "input_warnings", None)
            raw = [s for s in states if not is_precomputed(s)]
            for w in (hook(spec.view, raw) if callable(hook) and raw else []):
                msg = f"head '{spec.name}': {w}"
                if msg not in warnings:
                    warnings.append(msg)
        return head.ensemble.predict_proba(X), True

    def _warn_head(self, version: str, spec: HeadSpec, calibrated: bool, warnings: list[str],
                   stakes: str | None = None) -> None:
        if not calibrated:
            msg = (f"head '{spec.name}' is not trained in {version}: the answer is the uninformed prior "
                   f"(calibrated=false). Train it with `rlcd train` or POST /v1/train.")
        else:
            verdict = self.loaded(version, spec.name).verdict or {}
            v = verdict.get(stakes or spec.stakes)
            if v is None and verdict:   # a stakes level the verdict does not know: warn as for the weakest level
                v = min(verdict.values(), key=lambda x: x.get("tier", 2))
            v = v or {}
            if v.get("tier", 2) > 0:
                return
            msg = f"head '{spec.name}': {v.get('reason')}"
        if msg not in warnings:
            warnings.append(msg)

    def missing_evidence(self, version: str, spec: HeadSpec, missing: Sequence[str],
                         question: str | None = None) -> ContractError:
        """The 422 of a strict request whose state lacks inputs almost every training row had."""
        try:
            groups = self.provider(version).groups(spec.view)
        except Exception:
            groups = {}
        shown = sorted({groups.get(f, f) for f in missing})
        return ContractError(f"head '{spec.name}': the state lacks {len(missing)} input(s) that almost every "
                             f"training row had ({', '.join(shown[:6])}); a strict request is not answered from "
                             f"incomplete evidence. Send the same fields as in training (GET /v1/heads describes "
                             f"them)", kind="missing_evidence", head=spec.name, question=question,
                             missing=list(missing)[:50], missing_groups=shown)

    @staticmethod
    def _removed_warning(name: str, answer: dict, warnings: list[str]) -> None:
        m = answer.get("excluded_mass")
        if m is not None and m >= U.LARGE_REMOVED_MASS:
            w = (f"{name}: the head put {m:.2f} of its probability on options the request removed (allowed): the "
                 f"right answer is likely not among the allowed options")
            if w not in warnings:
                warnings.append(w)

    # ------------------------------------------------------------------ the evaluation endpoint
    def answer(self, version: str, q: Question, state: Any, warnings: list[str], *, strict: bool = False) -> dict:
        spec = self.spec(q.head, version, q.id)
        for note in q.bind(spec):
            if note not in warnings:
                warnings.append(note)
        s = q.inputs.project(state)
        missing: list[str] = []
        P, calibrated = self.probabilities(version, spec, [s], warnings, missing_out=missing)
        if strict and missing and calibrated:
            raise self.missing_evidence(version, spec, missing, q.id)
        self._warn_head(version, spec, calibrated, warnings)
        ans = build_answer(q, spec, P[0], calibrated, self.has_skill(version, spec.name))
        self._removed_warning(f"question '{q.id}'", ans, warnings)
        return ans

    def systemone(self, req: SystemOneRequest) -> dict:
        t0 = time.perf_counter()
        version = self.resolve(req.model)
        bat = BT.resolve(req.battery, self.batteries(version)) if req.battery is not None else None
        questions = bat.questions() if bat is not None else parse_questions(req.questions)
        if req.allowed:
            questions = restrict_questions(questions, req.allowed)
        warnings: list[str] = []
        # speculative fan-out: one state, every question answered independently by its own head
        answers = {q.id: self.answer(version, q, req.state, warnings, strict=req.strict) for q in questions}
        latency = round((time.perf_counter() - t0) * 1000, 2)
        out: dict[str, Any] = {"model": version, "answers": answers,
                               "usage": {"input_tokens": 0, "output_tokens": 0, "questions": len(questions),
                                         "latency_ms": latency}}
        if warnings:
            out["warnings"] = warnings
        if bat is not None:
            out["battery"] = bat.ref()
        if req.receipt or self.receipts_mode() == "all":
            heads = {}
            for q in questions:
                a = answers[q.id]
                cls, p = _committed(self.spec(q.head, version, q.id), a)
                heads[q.id] = {"head": q.head, "class": cls, "p": p, "tier": None,
                               "probabilities": a.get("probabilities"), "state": q.inputs.project(req.state)}
            row = self.log_decision(
                version, "systemone", heads, None, request_state=req.state, requested_model=req.model,
                questions={"digest": questions_digest(questions), "battery": bat.ref() if bat else None},
                latency_ms=latency, warnings=warnings)
            out["receipt"] = R.summary(row)
        return out

    def rank(self, req: RankRequest) -> dict:
        t0 = time.perf_counter()
        version = self.resolve(req.model)
        if not req.candidates:
            raise ContractError("candidates must be a non-empty array of {id, state}")
        if len(req.candidates) > MAX_CANDIDATES:
            raise ContractError(f"at most {MAX_CANDIDATES} candidates per call", limit=MAX_CANDIDATES)
        raw_q = req.question
        if raw_q is None:
            head = getattr(self.pack_for(version), "rank_head", None)
            if not head:
                raise ContractError("question is required: the noul (or choice + option) head to rank by")
            raw_q = {"type": self.spec(head, version).type, "head": head}
        q = Question("question", raw_q)
        spec = self.spec(q.head, version, q.id)
        q.bind(spec)
        option = req.option or q.option
        states = []
        for c in req.candidates:
            state = c.state
            if isinstance(req.state, dict) and isinstance(c.state, dict):
                state = {**req.state, **c.state}   # shared state, overridden per candidate
            elif c.state is None:
                state = req.state
            states.append(q.inputs.project(state))
        warnings: list[str] = []
        P, calibrated = self.probabilities(version, spec, states, warnings)   # one batch: the encoder runs once
        self._warn_head(version, spec, calibrated, warnings)
        ranked = []
        skill = self.has_skill(version, spec.name)
        for c, p in zip(req.candidates, P, strict=True):
            ans = build_answer(q, spec, p, calibrated, skill)
            ranked.append({"id": c.id, "probability": positive_probability(ans, option), "answer": ans})
        # stable: ties keep the request order; a candidate the head does not apply to goes last
        ranked.sort(key=lambda r: -1.0 if r["probability"] is None else r["probability"], reverse=True)
        for i, r in enumerate(ranked):
            r["rank"] = i + 1
        latency = round((time.perf_counter() - t0) * 1000, 2)
        out = {"model": version, "head": q.head, "option": option if q.type == "choice" else None,
               "candidates": ranked,
               "usage": {"input_tokens": 0, "output_tokens": 0, "questions": len(ranked),
                         "latency_ms": latency}}
        if warnings:
            out["warnings"] = warnings
        if req.receipt or self.receipts_mode() == "all":
            row = self.log_decision(
                version, "rank", {}, {"head": q.head, "option": out["option"],
                                      "ranking": [{"id": r["id"], "probability": r["probability"]} for r in ranked]},
                request_state={"state": req.state, "candidates": [c.model_dump() for c in req.candidates]},
                requested_model=req.model, questions={"digest": questions_digest([q]), "battery": None},
                latency_ms=latency, warnings=warnings)
            out["receipt"] = R.summary(row)
        return out

    # ------------------------------------------------------------------ decisions
    def gate_table(self, version: str, head_name: str, head: LoadedHead | None) -> dict | None:
        """The Bernoulli evidence a decision of `version` is gated on: the live table when it belongs to that
        version (holdout + outcomes scored since), else the version's own holdout table — a client that pins an
        older version keeps the evidence that version was verified with instead of losing its act tier."""
        if head is None:
            return None
        table = self.ledger.table(head_name)
        if table_is_live(table, version):
            return table
        archived = self.ledger.archived(head_name, version)   # the outcomes it collected while it was live
        if archived is not None:
            return archived
        try:
            return head.holdout_ledger()
        except Exception:   # an old bundle without its table: no evidence, the gate stays closed
            return None

    def check_allowed(self, spec: HeadSpec, allowed: Any) -> tuple[str, ...]:
        """Validate the options a decide request kept for a choice head (422 naming the problem)."""
        if spec.type != "choice":
            raise ContractError(f"allowed: head '{spec.name}' is a {spec.type} head; only a choice head has options "
                                f"to remove", field="allowed", head=spec.name)
        if not isinstance(allowed, (list, tuple)) or not allowed or not all(isinstance(o, str) for o in allowed):
            raise ContractError(f"allowed.{spec.name} must be a non-empty array of option names", field="allowed",
                                head=spec.name)
        unknown = [o for o in allowed if o not in spec.classes]
        if unknown:
            raise ContractError(f"allowed.{spec.name} names options {unknown} that head '{spec.name}' does not have; "
                                f"its options are {list(spec.classes)}", field="allowed", head=spec.name,
                                classes=list(spec.classes))
        return tuple(dict.fromkeys(allowed))

    def head_decision(self, version: str, spec: HeadSpec, state: Any, *, stakes: str | None = None,
                      explore: bool = False, warnings: list[str] | None = None, probs: np.ndarray | None = None,
                      calibrated: bool | None = None, explain: bool = True,
                      missing_inputs: list[str] | None = None, allowed: Sequence[str] | None = None,
                      strict: bool = False) -> dict:
        """Answer one head on one state and run it through the policy. Returns {answer, decision, reasons,
        uncertainty}.

        A request that lacks inputs almost every training row had (the input guard) is capped at confirm: its
        probabilities were calibrated on inputs that had them. The answer still says calibrated=true (the head is
        trained; the wire contract keeps that meaning) and the warning says why; with `strict` it is refused
        instead (422 missing_evidence). When `probs` are passed in, pass `missing_inputs` too (from
        `probabilities(..., missing_out=...)`), or the guard cannot cap.

        `allowed` (choice heads): the options deterministic code kept. The decision commits to the most probable
        allowed option with its calibrated, unrenormalised probability, so thresholds keep meaning error rates.
        Then the operator's rollout mode and kill switch cap the tier (never below confirm; `decision.rollout`)."""
        if probs is None:
            missing: list[str] = []
            P, calibrated = self.probabilities(version, spec, [state], warnings, missing_out=missing)
            probs = P[0]
            if missing_inputs is None:
                missing_inputs = missing
        if strict and missing_inputs and calibrated:
            raise self.missing_evidence(version, spec, missing_inputs)
        if allowed is not None:
            allowed = self.check_allowed(spec, allowed)
        pol, _removed, removed_mass = restrict(spec.labels, probs, allowed)
        head = self.loaded(version, spec.name)
        cfg = self.policy(version)
        live = self.gate_table(version, spec.name, head)
        d = PL.decide(spec, np.asarray(pol, float), cfg, stakes=stakes, verdict=head.verdict if head else None,
                      conformal_threshold=head.conformal_threshold if head else None, ledger=live,
                      explore=explore, rng=self.rng, calibrated=bool(calibrated), missing_inputs=missing_inputs)
        self._apply_verified(d, version, spec, head)
        self._apply_rollout(d, spec, cfg)
        ans = build_answer(None, spec, probs, bool(calibrated), self.has_skill(version, spec.name), allowed=allowed)
        vt = None
        if head is not None and calibrated:
            vt = ((head.verdict or {}).get(d["stakes"]) or {}).get("tier")
        out = {"answer": ans, "decision": d,
               "uncertainty": U.classify(probs, spec.labels, d, missing_inputs=missing_inputs,
                                         removed_mass=removed_mass, verdict_tier=vt)}
        if warnings is not None and allowed is not None:
            self._removed_warning(f"head '{spec.name}'", ans, warnings)
        if head is not None and head.set_quantile is not None and spec.type != "noul":
            from .calibration import prediction_set

            ks = prediction_set(probs, head.set_quantile) or []
            out["prediction_set"] = [spec.labels[k] for k in ks]
        if explain and head is not None and isinstance(state, dict) and not is_precomputed(state):
            try:
                x = self.features(version, spec, [state], head)
                ci = spec.labels.index(d["class"])
                reasons, groups = head.ensemble.explain(x, ci, groups=self.provider(version).groups(spec.view))
                out["reasons"], out["reason_groups"] = reasons, groups
            except ContractError:
                raise
            except Exception as e:  # explanations are a reading aid: never fail a decision over them
                out["reasons"] = [{"text": f"attributions unavailable: {type(e).__name__}"}]
        if warnings is not None:
            self._warn_head(version, spec, bool(calibrated), warnings, stakes)
            for w in d["warnings"]:
                if w not in warnings:
                    warnings.append(w)
        return out

    def verified_policy(self, version: str, head: str, loaded: LoadedHead | None) -> PL.PolicyConfig | None:
        """The policy a head's holdout verdict was computed under (its report.json), cached per version and head;
        None for an untrained head or a report without one."""
        key = (version, head)
        cache = self._verified_cache
        if key not in cache:
            cfg = None
            if loaded is not None:
                try:
                    raw = (loaded.report.get("policy") or {}).get("config")
                    cfg = PL.PolicyConfig.from_dict(raw) if isinstance(raw, dict) else None
                except Exception:   # an old or damaged report: nothing to compare with, the verdict cap still holds
                    cfg = None
            cache[key] = cfg
        return cache[key]

    def _apply_verified(self, d: dict, version: str, spec: HeadSpec, loaded: LoadedHead | None) -> None:
        """A head is served under the act threshold its holdout verified, or a stricter one: when the version serves
        a looser act threshold than the head was verified with (a head carried over into a version whose policy
        changed), acting is not verified and the decision is capped at confirm."""
        if d.get("class") is None or not d.get("calibrated") or d["tier"] != "act":
            return
        vcfg = self.verified_policy(version, spec.name, loaded)
        if vcfg is None:
            return
        try:
            vl = vcfg.level_for(spec.name, d["class"], d["stakes"])
        except ContractError:      # the level is unknown to the verified policy: policy.decide already capped it
            return
        served = float((d.get("thresholds") or {}).get("act", 1.0))
        if served + 1e-12 < vl.act:
            why = (f"head '{spec.name}' was verified with act >= {vl.act:g} at {d['stakes']} stakes, this version serves "
                   f"act >= {served:g}: acting below the verified threshold is not verified, so act becomes confirm")
            d.setdefault("caps", []).append({"from": "act", "to": "confirm", "reason": why})
            d["tier"] = "confirm"
            if why not in d["warnings"]:
                d["warnings"].append(why)

    def _apply_rollout(self, d: dict, spec: HeadSpec, cfg: PL.PolicyConfig) -> None:
        """Cap one policy decision by the rollout mode and the kill switch (in place). The cap is never below
        confirm: abstaining stays the model's own statement."""
        ro = self.rollout()
        cap, why = effective_cap(ro, self.switch, stakes=d["stakes"], head=spec.name)
        names = [cfg.tier_name(i) for i in range(3)]
        t = names.index(d["tier"])
        info: dict[str, Any] = {"mode": ro.mode_for(spec.name), "automation": self.switch.state().enabled,
                                "cap": names[cap],
                                "capped": t > cap}
        if t > cap:
            info["proposed_tier"] = d["tier"]
            info["reason"] = "; ".join(why)
            d["tier"] = names[cap]
            msg = f"{'; '.join(why)} (act became {names[cap]})"
            if msg not in d["warnings"]:
                d["warnings"].append(msg)
        d["rollout"] = info

    def head_log(self, state: Any, r: dict, **extra: Any) -> dict:
        """What a decision record keeps of one head decision (head_decision's output) on `state`."""
        d = r["decision"]
        rec = {"state": state, "class": d["class"], "p": d["p"], "tier": d["tier"], "stakes": d.get("stakes"),
               "probabilities": (r.get("answer") or {}).get("probabilities"),
               "excluded_mass": (r.get("answer") or {}).get("excluded_mass"),
               "uncertainty": (r.get("uncertainty") or {}).get("sources")}
        if (d.get("rollout") or {}).get("capped"):
            rec["proposed_tier"] = d["rollout"].get("proposed_tier")
        rec.update(extra)
        return rec

    @staticmethod
    def proposed_tier(d: dict, cfg: PL.PolicyConfig) -> str:
        """The tier a head decision would have without the rollout cap: its tier before the cap, still lowered by
        every deterministic cap a pack applied after it (`caps`)."""
        names = [cfg.tier_name(i) for i in range(3)]
        t = (d.get("rollout") or {}).get("proposed_tier") or d["tier"]
        for c in d.get("caps") or []:
            if c.get("to") in names and names.index(c["to"]) < names.index(t):
                t = c["to"]
        return t

    def rollout_summary(self, results: list[dict], action: str | None, cfg: PL.PolicyConfig,
                        proposed_action: str | None = None) -> dict:
        """The response's `rollout`: mode, kill switch, and the action that would have been taken without the
        rollout cap. A pack whose action is not simply the weakest head (a plan chain, rules) passes its own
        `proposed_action`, computed by the same rule from the heads' proposed tiers."""
        decisions = [r.get("decision", r) for r in results]
        capped = any((d.get("rollout") or {}).get("capped") for d in decisions)
        out = {**self.rollout().to_dict(), "automation": self.switch.state().to_dict(), "capped": capped}
        if action is not None:
            names = [cfg.tier_name(i) for i in range(3)]
            if proposed_action is None:
                proposed_action = (PL.weakest([{"tier": self.proposed_tier(d, cfg)} for d in decisions], cfg)
                                   if capped and decisions else action)
            # the rollout cap only lowers act to confirm: an abstained action was not its doing, and what would have
            # happened without it is never weaker than what happened
            if action == names[0] or (proposed_action in names and action in names
                                      and names.index(proposed_action) < names.index(action)):
                proposed_action = action
            out["proposed_action"] = proposed_action
        return out

    def log_decision(self, version: str, kind: str, heads: dict[str, dict], extra: dict | None = None, *,
                     request_state: Any = _MISSING, questions: dict | None = None, requested_model: str | None = None,
                     latency_ms: float | None = None, warnings: list[str] | None = None, audit: dict | None = None,
                     rollout: dict | None = None) -> dict:
        """Log a decision as a receipt so feedback can refer to it and a caller can check it before acting.
        `heads`: name -> head_log(...) (or at least {state, class, p, tier}). Returns the record as written (in a
        dry run: as it would have been, with decision_id "dry-run"). States are logged with serve.log_states
        (default on), secrets redacted (serve.redact); the receipt keeps the digest of the state as sent."""
        log_states = self.project.get("serve.log_states", True) is not False
        kinds = self.redact_kinds() if log_states else ()
        rec_heads = {}
        for name, h in heads.items():
            r = {k: h.get(k) for k in ("class", "p", "tier")}
            for k in ("head", "stakes", "probabilities", "excluded_mass", "proposed_tier", "uncertainty"):
                if h.get(k) is not None:
                    r[k] = h[k]
            if log_states:
                r["state"] = redact(h.get("state"), kinds)[0] if kinds else h.get("state")
            rec_heads[name] = r
        if log_states:   # a record never grows past MAX_LOGGED_STATE_BYTES of states (many questions, big states)
            budget = R.MAX_LOGGED_STATE_BYTES
            for r in rec_heads.values():
                if "state" not in r:
                    continue
                size = len(R.encode(R.dumps(R.plain(r["state"]))))
                if size > budget:
                    r["state_digest"] = digest(r.pop("state"))
                    r["state_omitted"] = f"{size} bytes: over the per-record log budget; send the state with feedback"
                else:
                    budget -= size
        b = self.bundle(version)
        meta = b.meta if b is not None else {}
        pk = meta.get("pack") or {}
        pack = {"name": pk.get("name") or getattr(self.pack_for(version), "name", None),
                "version": pk.get("version") or getattr(self.pack_for(version), "version", None)}
        if request_state is _MISSING:   # older callers: the state of the first head stands for the request
            request_state = next(iter(heads.values()), {}).get("state") if heads else None
        q = questions or {"digest": digest({"kind": kind, "heads": sorted(heads)}), "battery": None}
        rk = self.redact_kinds()
        if rk:
            audit = redact(audit, rk)[0] if audit else audit
            warnings = [redact(w, rk)[0] for w in (warnings or [])]
        receipt = R.make(requested_model=requested_model, model=version, pack=pack,
                         feature_version=meta.get("feature_version"), state=request_state, questions=q,
                         policy_digest=self.policy_digest(version), rollout=rollout, latency_ms=latency_ms,
                         warnings=warnings, audit=audit)
        row = {"kind": kind, "model": version, "heads": rec_heads, **(extra or {}), "receipt": receipt}
        if self.is_dry:
            return {"decision_id": "dry-run", "created_at": now_iso(), **row}
        return self.logs.add_receipt(row)

    def record(self, version: str, kind: str, heads: dict[str, dict], extra: dict | None = None,
               **receipt: Any) -> str:
        """Log a decision (see log_decision) and return its decision_id."""
        return self.log_decision(version, kind, heads, extra, **receipt)["decision_id"]

    def decide(self, req: DecideRequest) -> dict:
        version = self.resolve(req.model)
        if req.stakes is not None:   # an unknown level is a 422 listing the configured ones, before anything runs
            self.policy(version).level(req.stakes)
        pack = self.pack_for(version)
        if pack.supports("decide"):
            return pack.decide(self, req.model_dump())
        return self.generic_decide(req)

    def generic_decide(self, req: DecideRequest) -> dict:
        t0 = time.perf_counter()
        version = self.resolve(req.model)
        specs = self.specs(version)
        names = req.heads or list(specs)
        if not names:
            why = self.untrained_hint(version) or (f" {version} serves no head: every head is switched off in rlcd.yaml, "
                                                   f"or the pack defines none.")
            raise ContractError("no heads to decide with." + why)
        extra = req.model_extra or {}
        strict = bool(extra.get("strict"))
        hspecs = {name: self.spec(name, version) for name in names}
        allowed, considered = self.filter_options(hspecs, extra, names)
        warnings: list[str] = []
        results, logged = {}, {}
        for name in names:
            spec = hspecs[name]
            if name in allowed and not allowed[name]:
                r = self.no_option_decision(version, spec, req.state, stakes=req.stakes, warnings=warnings,
                                            considered=considered.get(name))
            else:
                r = self.head_decision(version, spec, req.state, stakes=req.stakes, explore=req.explore,
                                       warnings=warnings, allowed=allowed.get(name), strict=strict)
            self.attach_contract(r, name, allowed.get(name), considered.get(name))
            results[name] = r
            logged[name] = self.head_log(req.state, r)
        cfg = self.policy(version)
        tier = PL.weakest([r["decision"] for r in results.values()], cfg)
        latency = round((time.perf_counter() - t0) * 1000, 2)
        rollout = self.rollout_summary(list(results.values()), tier, cfg)
        row = self.log_decision(version, "decide", logged, {"action": tier, "stakes": req.stakes},
                                request_state=req.state, requested_model=req.model, latency_ms=latency,
                                warnings=warnings, audit=request_audit(extra), rollout=rollout,
                                questions=self.decision_contract("decide", names, extra, stakes=req.stakes))
        return {"decision_id": row["decision_id"], "model": version, "action": tier, "tier": tier,
                "explore": bool(req.explore), "heads": results, "policy": cfg.to_dict(), "warnings": warnings,
                "rollout": rollout, "receipt": R.summary(row),
                "usage": {"input_tokens": 0, "output_tokens": 0, "questions": len(results), "latency_ms": latency}}

    def evaluate(self, req: EvaluateRequest) -> dict:
        version = self.resolve(req.model)
        if req.stakes is not None:
            self.policy(version).level(req.stakes)
        pack = self.pack_for(version)
        if not pack.supports("evaluate"):
            raise ContractError(f"pack '{pack.name}' has no evaluator mode", kind="unsupported")
        return pack.evaluate(self, req.model_dump())

    # ------------------------------------------------------------------ the reinforcement loop
    def feedback(self, req: FeedbackRequest) -> dict:
        version = self.registry.latest_id()
        head, decision, dh, question = req.head, None, None, None
        if req.decision_id:
            decision = self.logs.decision(req.decision_id)
            if decision is None:
                raise ContractError(f"unknown decision_id {req.decision_id!r}", kind="unknown_decision")
            # a decision record keys its heads by head name (decide / evaluate) or by question id (a /v1/systemone
            # receipt, whose entries name their head): `head` may be either
            dheads = decision.get("heads") or {}
            if head is None:
                if len(dheads) != 1:
                    raise ContractError(f"decision {req.decision_id} involved heads {list(dheads)}: name the one "
                                        f"this outcome is for in `head`", heads=list(dheads))
                key = next(iter(dheads))
            elif head in dheads:
                key = head
            else:
                keys = [k for k, h in dheads.items() if (h or {}).get("head") == head]
                key = keys[0] if len(keys) == 1 else None
                if len(keys) > 1:
                    raise ContractError(f"decision {req.decision_id} asked head '{head}' in several questions "
                                        f"{keys}: name the question id in `head`", heads=list(dheads))
            if key is not None:
                dh = dheads[key]
                head = (dh or {}).get("head") or key
                if key != head:
                    question = key   # a /v1/systemone receipt: one head may be asked in several questions
        if not head:
            raise ContractError("head is required (or a decision_id of a single-head decision, which implies it)")
        spec = self.spec(head, version)
        try:
            yi = spec.label_index(req.label)
        except ValueError as e:
            raise ContractError(str(e), head=head, classes=list(spec.labels)) from None
        label = spec.labels[yi]
        row: dict = {"decision_id": req.decision_id, "head": head, "label": label, "weight": req.weight,
                     "source": req.source}
        if question is not None:
            row["question"] = question
        if req.x is not None:
            # a vector is only usable at the next train if it has the current model's feature schema: check it now
            # instead of promising a training row that every pack would skip
            if not req.feature_version:
                raise ContractError("feature_version is required with x (the feature version the vector was built "
                                    "with; GET /v1/heads lists it)", head=head)
            loaded = self.loaded(version, head)
            if loaded is not None:
                names, fv = loaded.ensemble.feature_names, loaded.ensemble.feature_version
            else:
                prov = self.provider(version)
                names, fv = prov.names(spec.view), prov.feature_version
            state = {"x": req.x, "feature_names": req.feature_names, "feature_version": req.feature_version}
            self._check_vector(spec, state, names, fv)
            row.update(state=state)
        elif req.state is not None:
            kinds = self.redact_kinds()
            row["state"] = redact(req.state, kinds)[0] if kinds else req.state
        elif dh is not None and dh.get("state") is not None:
            row["state"] = dh["state"]
        else:
            raise ContractError("feedback needs the state the label belongs to: `state`, `x` + `feature_version`, "
                                "or the decision_id of a logged decision (with serve.log_states on)")
        if decision is not None:
            row["time"] = decision.get("created_at")
        rk = self.redact_kinds()
        for k in ("source", "tags", "note", "missing_evidence"):
            v = getattr(req, k, None)
            if v:
                row[k] = redact(v, rk)[0] if rk else v
        update = None
        if dh is not None:
            table = self.ledger.table(head)
            if table is None:
                update = {"updated": False, "reason": f"head '{head}' is not trained: it has no posterior table yet"}
            elif dh.get("class") is None:
                update = {"updated": False, "reason": "the decision committed to no class (every option was removed)"}
            elif table_is_live(table, decision.get("model")):
                # each prediction counts once: a receipt that asked one head in two questions holds two predictions,
                # scored as two trials (decision_id#question); a head asked once keeps the plain decision_id
                asked = [k for k, h in (decision.get("heads") or {}).items() if ((h or {}).get("head") or k) == head]
                trial = req.decision_id if question is None or len(asked) < 2 else f"{req.decision_id}#{question}"
                update = self.ledger.update(head, trial, dh["class"], float(dh["p"]), label)
            else:
                update = {"updated": False, "reason": "the decision was made by a model version other than the one "
                                                      "the posterior table belongs to"}
        saved = self.logs.add_feedback(row)
        out = {"id": saved["id"], "stored": True, "head": head, "label": label,
               "feedback_rows": self.logs.feedback_counts().get(head, 0), "bernoulli": update,
               "note": "used as an up-weighted training row at the next train"}
        if "regression" in (req.tags or []):
            from . import regress as RG  # a reviewer's correction becomes a case every later version must pass

            try:
                case = RG.add_case(RG.cases_path(self.project), RG.case_from_feedback(self, saved))
                out["regression_case"] = {"id": case["id"], "stored": True}
            except ContractError as e:          # the feedback is stored either way; say why there is no case
                out["regression_case"] = {"stored": False, "reason": e.message}
            except OSError as e:                # the project directory is not writable from this process
                out["regression_case"] = {"stored": False, "reason": f"the regression cases file could not be "
                                                                      f"written ({type(e).__name__})"}
        return out

    def outcome(self, req: OutcomeRequest) -> dict:
        """A decision-level outcome (POST /v1/outcome), joined with its decision by the metrics."""
        decision = self.logs.decision(req.decision_id)
        if decision is None:
            raise ContractError(f"unknown decision_id {req.decision_id!r}", kind="unknown_decision")
        row = {k: v for k, v in req.model_dump().items() if v is not None}
        rk = self.redact_kinds()
        if rk:
            row = {k: (redact(v, rk)[0] if k in ("note", "provider", "route_taken", "tags", "verified_by") else v)
                   for k, v in row.items()}
        row["kind"] = decision.get("kind")
        row["model"] = decision.get("model")
        saved = self.logs.add_outcome(row)
        return {"id": saved["id"], "stored": True, "decision_id": req.decision_id,
                "note": "joined with the decision by GET /v1/metrics; outcomes never train a head"}

    # ------------------------------------------------------------------ views
    def head_view(self, version: str, spec: HeadSpec, pending: dict) -> dict:
        head = self.loaded(version, spec.name)
        out = {"name": spec.name, "type": spec.type, "classes": list(spec.classes), "view": spec.view,
               "decision": spec.decision, "stakes": spec.stakes, "conformal": spec.conformal,
               "description": spec.description, "input": spec.input or self._describe(version, spec.view),
               "model": spec.model, "trained": head is not None, "trained_at": None, "model_version": None,
               "metrics": None, "verdict": None, "n_train": None, "positive": spec.positive,
               "feedback_rows": pending.get(spec.name, 0)}
        if spec.type == "score":
            out["legend"] = {str(i): d for i, d in enumerate(spec.legend)}
        if head is not None:
            e = head.entry
            out.update(trained_at=e.get("trained_at"), model_version=version, metrics=e.get("metrics"),
                       verdict=head.verdict, n_train=(e.get("data") or {}).get("development_rows"),
                       model=e.get("model"), conformal_threshold=head.conformal_threshold)
        return out

    def _describe(self, version: str, view: str) -> dict:
        try:
            return self.provider(version).describe(view)
        except Exception:
            return {}

    def heads_view(self, model: str | None) -> dict:
        version = self.resolve(model, serve=False)
        pending = self.logs.feedback_counts()
        return {"model": version, "heads": [self.head_view(version, s, pending) for s in self.specs(version).values()]}

    def models_view(self) -> dict:
        reg = self.registry.registry()
        latest = self.registry.latest_id()
        data: list[dict] = [{"id": LATEST, "alias_of": latest}]
        if not reg["versions"]:
            data.append({"id": latest, "trained_at": None, "heads": {},
                         "note": "untrained: every answer is the uninformed prior until `rlcd train`"})
        for vid in sorted(reg["versions"], key=lambda v: int(v.rsplit(".", 1)[1]), reverse=True):
            v = reg["versions"][vid]
            row = {"id": vid, "trained_at": v.get("created_at"), "pack": v.get("pack"),
                   "retrained": v.get("retrained", []), "imported_from": v.get("imported_from"),
                   "heads": self._version_heads(vid, v.get("heads", {}))}
            if v.get("edit"):   # a weight-edit candidate (additive field): whether it serves, and why
                row["edit"] = v["edit"]
            data.append(row)
        return {"data": data}

    def _version_heads(self, vid: str, trained: dict) -> dict:
        """Every head of a version, as the reference lists them: `trained`, `n_train`, `trained_at`,
        `model_version` and `metrics` (plus `verdict` and `model`); heads the pack defines but this version never
        trained are listed with trained=false."""
        out: dict[str, dict] = {}
        b = None
        try:
            b = self.bundle(vid)
        except Exception:   # a broken or missing bundle must not take the listing down
            b = None
        for name, h in trained.items():
            entry = ((b.meta.get("heads") or {}).get(name) or {}) if b is not None else {}
            row = {**h, "trained": True, "model_version": vid,
                   "n_train": (entry.get("data") or {}).get("development_rows")}
            loaded = b.head(name) if b is not None else None
            if loaded is not None:   # the verdict the head enforces (hash-checked head.json), not registry.json's
                with contextlib.suppress(Exception):
                    row["verdict"] = loaded.verdict
            out[name] = row
        try:
            names = list(self.specs(vid))
        except Exception:
            names = []
        for name in names:
            out.setdefault(name, {"trained": False, "n_train": None, "trained_at": None, "model_version": None,
                                  "metrics": None, "verdict": None, "model": None})
        return out

    def trained(self, head: str, model: str | None) -> tuple[str, HeadSpec, LoadedHead]:
        version = self.resolve(model, serve=False)
        spec = self.spec(head, version)
        h = self.loaded(version, head)
        if h is None:
            raise ContractError(f"head '{head}' is not trained in {version}: there is no holdout report yet",
                                kind="untrained", head=head)
        return version, spec, h

    def calibration_view(self, head: str, model: str | None, segment: str | None = None) -> dict:
        version, spec, h = self.trained(head, model)
        rep = dict(h.report)
        meta = {"head": head, "type": spec.type, "model": version, "trained_at": h.entry.get("trained_at"),
                "base_model": h.entry.get("model"), "calibration": h.entry.get("calibration"),
                "data": h.entry.get("data"), "notes": h.entry.get("notes", []), "scope": "trusted holdout rows",
                "primary_metrics": ["log_loss", "brier"]}
        if segment is not None:
            segs = rep.get("by_segment") or {}
            if segment not in segs:
                raise ContractError(f"no holdout slice for segment {segment!r}", segments=sorted(segs))
            return {**meta, "scope": f"trusted holdout rows of segment {segment}", **segs[segment]}
        return {**meta, **rep}

    def bernoulli_view(self, head: str, model: str | None, stakes: str | None = None) -> dict:
        from . import bernoulli as B

        version, spec, h = self.trained(head, model)
        cfg = self.policy(version)
        level = cfg.level(stakes or spec.stakes)
        table = self.ledger.table(head)
        live = table_is_live(table, version)
        view = B.table_view(table if live else h.holdout_ledger(), level.min_accuracy, cfg.bernoulli())
        return {"head": head, "model": version, "stakes": stakes or spec.stakes,
                "required_accuracy": level.min_accuracy, "config": cfg.bernoulli().to_dict(),
                "table_is_live": live, **view,
                "policy_holdout": (h.report.get("policy") or {}).get("holdout"),
                "notes": ["bins: every decision is a Bernoulli trial of its committed class and probability bin; "
                          "counts start from the trusted holdout and grow with each outcome posted to "
                          "/v1/feedback with a decision_id",
                          "the table belongs to one model version and restarts from the new holdout at a train"]}

    # ------------------------------------------------------------------ receipts
    def receipt_view(self, decision_id: str) -> dict:
        row = self.logs.decision(decision_id)
        if row is None:
            raise ContractError(f"unknown decision_id {decision_id!r}", kind="unknown_decision", status=404)
        return json_safe(row)

    def check_receipt(self, decision_id: str, *, state: Any = _MISSING, state_digest: str | None = None,
                      max_age_s: float | None = None) -> dict:
        """Does decision `decision_id` still approve acting on `state` now? (core/receipts.py `check`)"""
        row = self.receipt_view(decision_id)
        r = row.get("receipt") if isinstance(row.get("receipt"), dict) else {}     # a hand-edited line: no receipt
        if not r.get("state_digest"):
            raise ContractError(f"decision {decision_id} was logged before receipts existed: it cannot be checked",
                                kind="no_receipt")
        if state is _MISSING and state_digest is None:
            raise ContractError("send the state you are about to act on (`state`) or its digest (`state_digest`): a "
                                "receipt approves the state it judged, nothing else", field="state")
        sd = digest(state) if state is not _MISSING else str(state_digest)
        reg = self.registry.registry()
        latest = self.registry.latest_id()
        known = list(reg["versions"]) or [latest]
        model = row.get("model")
        try:
            pd = self.policy_digest(model) if model in known else None
        except Exception:   # a bundle that no longer loads: the policy cannot be compared, the model check says why
            pd = None
        bat = (r.get("questions") or {}).get("battery")
        bv = None
        if bat:
            try:
                b = self.batteries(model if model in known else latest).get(bat.get("name"))
                bv = b.version if b is not None else None
            except ContractError:
                bv = None
        cap = 2
        for name, h in (row.get("heads") or {}).items():
            if (h or {}).get("tier") == "act":
                c, _ = effective_cap(self.rollout(), self.switch, stakes=h.get("stakes") or row.get("stakes"),
                                     head=h.get("head") or name)
                cap = min(cap, c)
        if row.get("action") == "act" and self.switch.cap()[0] < cap:
            cap = self.switch.cap()[0]
        out = R.check(row, state_digest=sd, latest=latest, known_models=known, policy_digest=pd,
                      automation_cap=cap, battery_version=bv, max_age_s=max_age_s)
        # a confirm decision approves an action only once a person confirmed it (the guide's layered gate); a line in
        # confirmations.jsonl counts only when it names this decision's state and matches its own digest
        if R.needs_confirmation(row):
            valid, ignored = [], []
            for c in self.logs.confirmations(decision_id):
                why = R.confirmation_problem(c, row)
                if why is None:
                    valid.append(c)
                else:
                    ignored.append(why)
            if valid:
                out["checks"]["confirmation"] = "confirmed"
                out["confirmed"] = valid[-1]
            else:
                out["checks"]["confirmation"] = "missing"
                reason = (f"a confirm decision: a person must confirm it before the action "
                          f"(POST /v1/receipts/{decision_id}/confirm)")
                if ignored:
                    reason += (f"; {len(ignored)} confirmation line(s) ignored: "
                               f"{'; '.join(dict.fromkeys(ignored))}")
                out["stale_reasons"].append(reason)
                out["current"] = False
        return out

    def confirm_receipt(self, decision_id: str, *, by: str, note: str | None = None,
                        state_digest: str | None = None) -> dict:
        """Record that a person confirmed a decision (POST /v1/receipts/{id}/confirm). The confirmation belongs to
        the receipt and so to its state: a changed state still makes the receipt stale."""
        row = self.receipt_view(decision_id)
        r = row.get("receipt") if isinstance(row.get("receipt"), dict) else {}     # a hand-edited line: no receipt
        if state_digest is not None and state_digest != r.get("state_digest"):
            raise ContractError("state_digest is not the digest of the state this decision judged: confirm the "
                                "decision you reviewed", kind="state_mismatch", field="state_digest")
        rk = self.redact_kinds()
        note = redact(note, rk)[0] if (note and rk) else note
        by = redact(by, rk)[0] if (by and rk) else by
        rec = self.logs.add_confirmation({"decision_id": decision_id, "by": by, "note": note,
                                          "state_digest": r.get("state_digest"), "action": row.get("action")})
        return {"confirmed": True, "decision_id": decision_id, "id": rec["id"], "at": rec["received_at"], "by": by,
                "digest": rec["receipt"]["digest"]}

    def verify_receipts(self) -> dict:
        """The chain of decisions.jsonl (top level, as before) and of confirmations.jsonl (`confirmations`); `ok`
        only when both verify. Anchor both heads outside RLCD_HOME to detect truncation."""
        out = R.verify_chain(self.logs.decision_log.path)
        conf = R.verify_chain(self.logs.confirmation_log.path)
        out["confirmations"] = conf
        out["ok"] = bool(out["ok"] and conf["ok"])
        return out

    # ------------------------------------------------------------------ operations metrics and the review queue
    def review_capacity(self) -> float | None:
        """rlcd.yaml serve.review_capacity_per_day: confirm decisions the reviewers can take per day (None: unset)."""
        v = self.project.get("serve.review_capacity_per_day")
        if v is None:
            return None
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not v > 0:
            raise ContractError(f"rlcd.yaml serve.review_capacity_per_day must be a positive number, got {v!r}",
                                kind="bad_config", status=500)
        return float(v)

    def abstain_names(self) -> tuple[str, ...]:
        """The names the abstain action goes by in the logs: the built-in ones and this project's policy.abstain."""
        names = list(OPS.ABSTAIN_NAMES)
        with contextlib.suppress(Exception):   # a pack that cannot load must not take the metrics down
            names.append(self.policy_config().abstain)
        return tuple(dict.fromkeys(names))

    def holdouts(self, version: str, head: str | None = None) -> dict[str, dict]:
        """{head: ops.holdout_summary(report.json)} for the trained heads of `version` ({} when it has none)."""
        try:
            b = self.bundle(version)
        except Exception:      # a version no longer registered, or a bundle that does not load
            return {}
        out: dict[str, dict] = {}
        for name in (b.names() if b is not None else []):
            if head is not None and name != head:
                continue
            h = b.head(name)
            try:
                out[name] = OPS.holdout_summary(h.report, decision=h.spec.decision, model=version)
            except Exception:  # an old or damaged bundle without report.json: nothing to compare with
                continue
        return out

    def metrics_view(self, *, since: str | None = None, until: str | None = None, model: str | None = None,
                     kind: str | None = None, head: str | None = None) -> dict:
        """GET /v1/metrics (core/ops.py) over decisions.jsonl, feedback.jsonl and outcomes.jsonl. The holdout of
        `model` (default: the version rlcd-latest points at) is put next to that version's online numbers."""
        version = None
        if model:
            try:
                version = self.resolve(model)
            except ContractError:   # a version no longer served still has decisions in the log
                version = model
        target = version or self.registry.latest_id()
        holdout = self.holdouts(target, head)
        out = OPS.metrics(self.logs.decisions(), self.logs.feedback(), self.logs.outcomes(), since=since,
                          until=until, model=version, kind=kind, head=head, holdout=holdout,
                          review_capacity_per_day=self.review_capacity(), abstain_names=self.abstain_names())
        out["holdout_model"] = target if holdout else None
        return out

    def review_view(self, *, limit: int = 50, head: str | None = None, kind: str | None = None,
                    include_abstain: bool = True, oldest_first: bool = True) -> dict:
        """GET /v1/review: confirm and abstain decisions without feedback, oldest first (core/ops.py)."""
        limit = max(0, int(limit))
        items = OPS.review_queue(self.logs.decisions(), self.logs.feedback(), limit=None, head=head, kind=kind,
                                 include_abstain=include_abstain, oldest_first=oldest_first,
                                 abstain_names=self.abstain_names())
        return {"pending": len(items), "shown": min(len(items), limit), "items": json_safe(items[:limit]),
                "capacity_per_day": self.review_capacity(),
                "resolve": "POST /v1/feedback {decision_id, head, label, source, tags?: [\"regression\"], note?, "
                           "missing_evidence?}: correct the route, name missing evidence, or keep the case as a "
                           "regression fixture; POST /v1/receipts/{id}/confirm when a person confirms the action"}

    # ------------------------------------------------------------------ automation and rollout
    def automation_view(self) -> dict:
        return {"rollout": self.rollout().to_dict(), "automation": self.switch.state().to_dict(),
                "history": self.switch.history(20)}

    def _redacted(self, *values: str | None) -> tuple[str | None, ...]:
        """Free-text audit fields (who, why) as serve.redact keeps them: a key pasted into a reason is masked."""
        rk = self.redact_kinds()
        return tuple(redact(v, rk)[0] if (v and rk) else v for v in values)

    def set_automation(self, enabled: bool, *, reason: str | None = None, by: str | None = None) -> dict:
        reason, by = self._redacted(reason, by)
        self.switch.set(bool(enabled), reason=reason, by=by, at=now_iso())
        return self.automation_view()

    def promote(self, version: str, *, reason: str | None = None, by: str | None = None,
                kind: str = "promote") -> dict:
        """Point rlcd-latest at `version` (kind "rollback" or "promote"): no retrain, no code change. A weight-edit
        candidate whose holdout gate failed is refused: it never serves."""
        reason, by = self._redacted(reason, by)
        try:
            return self.registry.set_latest(version, reason=reason, by=by, kind=kind)
        except UnknownModel:
            raise ContractError(f"unknown model {version!r}", kind="unknown_model",
                                available=self.registry.available()) from None
        except NotServable as e:
            raise self._never_serves(e.version, e.reason) from None

    # ------------------------------------------------------------------ batteries
    def batteries_view(self, model: str | None = None) -> dict:
        version = self.resolve(model)
        specs = self.specs(version)
        out = []
        for b in self.batteries(version).values():
            d = b.to_dict()
            problems = []
            for q in b.questions():
                spec = specs.get(q.head)
                if spec is None:
                    problems.append(f"question '{q.id}': no head named {q.head!r} in {version}")
                    continue
                try:
                    q.bind(spec)
                except ContractError as e:
                    problems.append(e.message)
                d["questions"][q.id]["trained"] = self.loaded(version, q.head) is not None
                if not d["questions"][q.id]["trained"]:
                    problems.append(f"question '{q.id}': head {q.head!r} is not trained in {version}")
            d["problems"] = problems
            out.append(d)
        return {"model": version, "batteries": out}

    def gate_outlook(self, version: str, head: str) -> dict:
        """Where a trained head stands with the Bernoulli gate, per stakes level whose holdout verdict is `act`.

        The gate keeps counts per (committed class, probability bin), seeded from the holdout. A cell the holdout
        filled well passes at once; a cell it never (or rarely) reached starts at confirm, and needs correct
        outcomes posted to /v1/feedback before it acts. Computed with the gate itself (`bernoulli.posterior` against
        the level's `min_accuracy` and `policy.gate_min_p_above`), on the evidence a decision would be gated on now:

            {level: {"act": 0.85, "min_accuracy": 0.75, "threshold": 0.9, "from_empty": 8,
                     "cells": [{"class", "bin", "lo", "hi", "wins", "losses", "passes", "needed"}]}}

        `needed`: correct outcomes in a row until the cell passes (0 when it passes; None when 10,000 would not
        do); `from_empty`: the same number for a cell without any evidence. {} for an untrained head, or one whose
        verdict allows `act` at no level."""
        from . import bernoulli as B

        loaded = self.loaded(version, head)
        if loaded is None:
            return {}
        spec, cfg = loaded.spec, self.policy(version)
        bcfg = cfg.bernoulli()
        table = self.gate_table(version, head, loaded) or {}
        bins = table.get("bins") or {}
        labels = (spec.labels[1],) if spec.decision == "gate" else tuple(spec.labels)
        conformal = loaded.conformal_threshold if spec.decision == "gate" else None

        def needed(wins: float, losses: float, required: float) -> int | None:
            def ok(n: int) -> bool:
                return (B.posterior(wins + n, losses, required, bcfg)["p_above_required"] or 0.0) >= bcfg.min_p_above
            if ok(0):
                return 0
            hi = 1
            while not ok(hi):
                hi *= 2
                if hi > 10_000:
                    return None
            lo = hi // 2          # not ok(lo), ok(hi): more wins only ever raise the posterior mass above `required`
            while hi - lo > 1:
                mid = (lo + hi) // 2
                lo, hi = (lo, mid) if ok(mid) else (mid, hi)
            return hi

        out: dict[str, dict] = {}
        for level in cfg.stakes:
            v = (loaded.verdict or {}).get(level) or {}
            if v.get("tier") != 2:
                continue
            cells = []
            for c in labels:
                lv = cfg.level_for(spec.name, c, level)
                act = max(lv.act, conformal) if conformal is not None else lv.act
                for k in range(B.bin_of(act), B.BINS):
                    w, l_ = (bins.get(c) or [[0, 0]] * B.BINS)[k]
                    n = needed(w, l_, lv.min_accuracy)
                    cells.append({"class": c, "bin": k, "lo": round(k / B.BINS, 2), "hi": round((k + 1) / B.BINS, 2),
                                  "wins": int(w), "losses": int(l_), "passes": n == 0, "needed": n})
            base = cfg.level(level)
            out[level] = {"act": base.act, "min_accuracy": base.min_accuracy, "threshold": bcfg.min_p_above,
                          "from_empty": needed(0, 0, base.min_accuracy), "cells": cells}
        return out

    def readiness(self) -> dict:
        """Can this process decide? {"ready", "model", "reasons"}: ready when the pack loads, the served version has
        at least one trained head and its policy can be served. `/health` says the process is up; this says a
        decision would come from a trained model (GET /ready answers 503 until then)."""
        version = self.registry.latest_id()
        reasons: list[str] = []
        try:
            specs = self.specs(version)
        except Exception as e:      # the text can carry paths or a DSN: it goes to the log, as in health()
            import logging

            logging.getLogger("open_rlcd").error("readiness: the pack cannot load: %s", e)
            specs = None
            reasons.append(f"the pack cannot load ({type(e).__name__}; details are in the server log)")
        if specs is not None:
            if self.bundle(version) is None:
                reasons.append(f"no model is trained ({version} is the untrained service): run `rlcd train` or "
                               f"import a bundle")
            elif not any(self.loaded(version, n) is not None for n in specs):
                reasons.append(f"{version} serves no trained head (every trained head is switched off in rlcd.yaml)")
            else:
                try:
                    self.policy(version)
                except ContractError as e:
                    reasons.append(e.message if e.kind == "bad_bundle" else
                                   "the policy cannot be served (details: GET /v1/check)")
        return {"ready": not reasons, "model": version, "reasons": reasons}

    def health(self) -> dict:
        version = self.registry.latest_id()
        running = self.trainer.running()
        heads = {}
        try:
            for n, s in self.specs(version).items():
                heads[n] = {"trained": self.loaded(version, s.name) is not None}
        except Exception as e:  # health must answer even when the pack cannot load; the text (paths, DSNs) is logged
            import logging

            logging.getLogger("open_rlcd").error("health: the pack cannot load: %s", e)
            heads = {"error": f"{type(e).__name__}: the pack cannot load (details are in the server log)"}
        b = self.bundle(version)
        try:
            rollout = self.rollout().mode
        except ContractError:
            rollout = "invalid"
        return {"status": "ok", "service": "open-rlcd", "version": VERSION, "model": version, "alias": LATEST,
                "pack": ((b.meta.get("pack") or {}).get("name") if b else self._default_pack),
                "heads": heads, "training": running["job_id"] if running else None, "home": str(self.home),
                "untrained_id": f"{MODEL_PREFIX}0", "rollout": rollout,
                "automation": self.switch.state().enabled, "ready": self.readiness()["ready"]}
