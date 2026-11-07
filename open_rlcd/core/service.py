"""The service: everything the HTTP endpoints (and the CLI) need, without HTTP.

Answers questions bound to heads (System One contract), ranks candidates, makes decisions through the policy,
records decisions, and takes feedback (immediate Bernoulli update + a training row for the next train).
"""
from __future__ import annotations

import contextlib
import time
from pathlib import Path
from typing import Any

import numpy as np

from .. import LATEST, MODEL_PREFIX, VERSION
from . import policy as PL
from .bernoulli import Ledger
from .bundle import Bundle, LoadedHead
from .config import Project, load_project
from .contract import (
    MAX_CANDIDATES,
    ContractError,
    DecideRequest,
    EvaluateRequest,
    FeedbackRequest,
    Question,
    RankRequest,
    SystemOneRequest,
    build_answer,
    parse_questions,
    positive_probability,
)
from .feedback import FeedbackLog
from .head import HeadSpec
from .registry import Registry, UnknownModel


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

            pack = load_pack(name)
            self._packs[name] = pack
            self._packs.setdefault(str(getattr(pack, "name", name)), pack)
        return self._packs[name]

    @property
    def pack(self):
        """The project's pack (what `rlcd train` trains)."""
        return self.load_pack(self._default_pack)

    def resolve(self, model: str | None) -> str:
        try:
            return self.registry.resolve(model)
        except UnknownModel:
            raise ContractError(f"unknown model {model!r}", kind="unknown_model",
                                available=self.registry.available()) from None

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
        """The pack's current heads, overlaid by the heads this version was trained with."""
        out = dict(self.pack_for(version).heads(self.project))
        b = self.bundle(version)
        if b is not None:
            for name in b.names():
                out[name] = b.head(name).spec
        return out

    def spec(self, head: str, version: str, question: str | None = None) -> HeadSpec:
        specs = self.specs(version)
        if head not in specs:
            raise ContractError(
                f"no head named {head!r}. RLCD only answers questions bound to a trained head: set the "
                f"question's `head` field (or use a head name as the question id).", kind="unknown_head",
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
        b = self.bundle(version)
        if b is not None and b.meta.get("policy"):
            return PL.PolicyConfig.from_dict(b.meta["policy"])
        return self.policy_config()

    def policy_config(self) -> PL.PolicyConfig:
        return PL.PolicyConfig.from_dict(_merge(self.pack.policy_defaults(), self.project.section("policy")))

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

    # ------------------------------------------------------------------ the evaluation endpoint
    def answer(self, version: str, q: Question, state: Any, warnings: list[str]) -> dict:
        spec = self.spec(q.head, version, q.id)
        q.bind(spec)
        P, calibrated = self.probabilities(version, spec, [state], warnings)
        self._warn_head(version, spec, calibrated, warnings)
        return build_answer(q, spec, P[0], calibrated, self.has_skill(version, spec.name))

    def systemone(self, req: SystemOneRequest) -> dict:
        t0 = time.perf_counter()
        version = self.resolve(req.model)
        questions = parse_questions(req.questions)
        warnings: list[str] = []
        # speculative fan-out: one state, every question answered independently by its own head
        answers = {q.id: self.answer(version, q, req.state, warnings) for q in questions}
        out = {"model": version, "answers": answers,
               "usage": {"input_tokens": 0, "output_tokens": 0, "questions": len(questions),
                         "latency_ms": round((time.perf_counter() - t0) * 1000, 2)}}
        if warnings:
            out["warnings"] = warnings
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
            states.append(state)
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
        out = {"model": version, "head": q.head, "option": option if q.type == "choice" else None,
               "candidates": ranked,
               "usage": {"input_tokens": 0, "output_tokens": 0, "questions": len(ranked),
                         "latency_ms": round((time.perf_counter() - t0) * 1000, 2)}}
        if warnings:
            out["warnings"] = warnings
        return out

    # ------------------------------------------------------------------ decisions
    def head_decision(self, version: str, spec: HeadSpec, state: Any, *, stakes: str | None = None,
                      explore: bool = False, warnings: list[str] | None = None, probs: np.ndarray | None = None,
                      calibrated: bool | None = None, explain: bool = True,
                      missing_inputs: list[str] | None = None) -> dict:
        """Answer one head on one state and run it through the policy. Returns {answer, decision, reasons}.

        A request that lacks inputs almost every training row had (the input guard) is capped at confirm: its
        probabilities were calibrated on inputs that had them. The answer still says calibrated=true (the head is
        trained; the wire contract keeps that meaning) and the warning says why. When `probs` are passed in, pass
        `missing_inputs` too (from `probabilities(..., missing_out=...)`), or the guard cannot cap."""
        if probs is None:
            missing: list[str] = []
            P, calibrated = self.probabilities(version, spec, [state], warnings, missing_out=missing)
            probs = P[0]
            if missing_inputs is None:
                missing_inputs = missing
        head = self.loaded(version, spec.name)
        cfg = self.policy(version)
        table = self.ledger.table(spec.name) if head is not None else None
        live = table if table_is_live(table, version) else None
        d = PL.decide(spec, probs, cfg, stakes=stakes, verdict=head.verdict if head else None,
                      conformal_threshold=head.conformal_threshold if head else None, ledger=live,
                      explore=explore, rng=self.rng, calibrated=bool(calibrated), missing_inputs=missing_inputs)
        ans = build_answer(None, spec, probs, bool(calibrated), self.has_skill(version, spec.name))
        out = {"answer": ans, "decision": d}
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

    def record(self, version: str, kind: str, heads: dict[str, dict], extra: dict | None = None) -> str:
        """Log a decision so feedback can refer to it. `heads`: name -> {state, class, p}."""
        log_states = self.project.get("serve.log_states", True) is not False
        rec_heads = {}
        for name, h in heads.items():
            r = {"class": h.get("class"), "p": h.get("p"), "tier": h.get("tier")}
            if log_states:
                r["state"] = h.get("state")
            rec_heads[name] = r
        return self.logs.add_decision({"kind": kind, "model": version, "heads": rec_heads, **(extra or {})})

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
            raise ContractError("no heads to decide with")
        warnings: list[str] = []
        results, logged = {}, {}
        for name in names:
            spec = self.spec(name, version)
            r = self.head_decision(version, spec, req.state, stakes=req.stakes, explore=req.explore,
                                   warnings=warnings)
            results[name] = r
            logged[name] = {"state": req.state, "class": r["decision"]["class"], "p": r["decision"]["p"],
                            "tier": r["decision"]["tier"]}
        cfg = self.policy(version)
        tier = PL.weakest([r["decision"] for r in results.values()], cfg)
        did = self.record(version, "decide", logged)
        return {"decision_id": did, "model": version, "action": tier, "tier": tier,
                "explore": bool(req.explore), "heads": results, "policy": cfg.to_dict(), "warnings": warnings,
                "usage": {"input_tokens": 0, "output_tokens": 0, "questions": len(results),
                          "latency_ms": round((time.perf_counter() - t0) * 1000, 2)}}

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
        head, decision = req.head, None
        if req.decision_id:
            decision = self.logs.decision(req.decision_id)
            if decision is None:
                raise ContractError(f"unknown decision_id {req.decision_id!r}", kind="unknown_decision")
            if head is None:
                dheads = list(decision.get("heads") or {})
                if len(dheads) != 1:
                    raise ContractError(f"decision {req.decision_id} involved heads {dheads}: name the one this "
                                        f"outcome is for in `head`", heads=dheads)
                head = dheads[0]
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
        dh = ((decision or {}).get("heads") or {}).get(head) if decision else None
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
            row["state"] = req.state
        elif dh is not None and dh.get("state") is not None:
            row["state"] = dh["state"]
        else:
            raise ContractError("feedback needs the state the label belongs to: `state`, `x` + `feature_version`, "
                                "or the decision_id of a logged decision (with serve.log_states on)")
        if decision is not None:
            row["time"] = decision.get("created_at")
        update = None
        if dh is not None:
            table = self.ledger.table(head)
            if table is None:
                update = {"updated": False, "reason": f"head '{head}' is not trained: it has no posterior table yet"}
            elif table_is_live(table, decision.get("model")):
                update = self.ledger.update(head, req.decision_id, dh["class"], float(dh["p"]), label)
            else:
                update = {"updated": False, "reason": "the decision was made by a model version other than the one "
                                                      "the posterior table belongs to"}
        saved = self.logs.add_feedback(row)
        return {"id": saved["id"], "stored": True, "head": head, "label": label,
                "feedback_rows": self.logs.feedback_counts().get(head, 0), "bernoulli": update,
                "note": "used as an up-weighted training row at the next train"}

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
        version = self.resolve(model)
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
            data.append({"id": vid, "trained_at": v.get("created_at"), "pack": v.get("pack"),
                         "retrained": v.get("retrained", []), "imported_from": v.get("imported_from"),
                         "heads": self._version_heads(vid, v.get("heads", {}))})
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
        version = self.resolve(model)
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

    def health(self) -> dict:
        version = self.registry.latest_id()
        running = self.trainer.running()
        heads = {}
        try:
            for n, s in self.specs(version).items():
                heads[n] = {"trained": self.loaded(version, s.name) is not None}
        except Exception as e:  # health must answer even when the pack cannot load
            heads = {"error": f"{type(e).__name__}: {e}"}
        b = self.bundle(version)
        return {"status": "ok", "service": "open-rlcd", "version": VERSION, "model": version, "alias": LATEST,
                "pack": ((b.meta.get("pack") or {}).get("name") if b else self._default_pack),
                "heads": heads, "training": running["job_id"] if running else None, "home": str(self.home),
                "untrained_id": f"{MODEL_PREFIX}0"}
