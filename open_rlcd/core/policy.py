"""The decision policy: turn calibrated probabilities into act / confirm / escalate.

For one head and one row, `p` is the calibrated probability of the class the decision commits to:
  * argmax mode (choice, score, symmetric noul)  the most likely class and its probability;
  * gate mode (a noul whose `true` means "proceed": answerable, grounded, approve, ...)  P(true).

With the thresholds of the requested stakes level:

    act       p >= act        (and p >= the conformal threshold, for gate heads that promise an error rate)
    confirm   p >= confirm    (a human, or a cheaper check, confirms before acting)
    escalate  otherwise       (the abstain action; packs may call it "hold")

If the probabilities are calibrated, acting at p >= 0.85 is wrong about 15% of the time or less: the thresholds
are a statement about acceptable error rates, which is why they are configuration and are returned with every
decision. Two caps then apply, both from evidence rather than belief:

  * the verdict of the head's holdout (`verify`): no skill over the base rate -> always escalate; too few
    trusted holdout rows, act-tier rows that came true less often than predicted, or a conformal promise that
    cannot be made -> never more than confirm;
  * the Bernoulli gate: `act` needs the accuracy posterior of predictions like this one (same head, same class,
    same probability bin; holdout + scored feedback) to put >= `gate_min_p_above` of its mass above the stakes
    level's `min_accuracy`. Otherwise act becomes confirm.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from . import bernoulli as B
from .contract import ContractError
from .metrics import _f, wilson

TIERS = ("escalate", "confirm", "act")  # index = strength; TIERS[0] is renamed to the configured abstain action


class UnknownStakes(ContractError, ValueError):
    """A stakes level the policy does not define: a 422 `validation` error listing the configured levels (still a
    ValueError for library callers)."""


@dataclass(frozen=True)
class StakesLevel:
    act: float
    confirm: float
    min_accuracy: float

    def __post_init__(self) -> None:
        if not 0 <= self.confirm <= self.act <= 1:
            raise ValueError("stakes thresholds need 0 <= confirm <= act <= 1")
        if not 0 <= self.min_accuracy <= 1:
            raise ValueError("min_accuracy must be in [0, 1]")


def default_stakes() -> dict[str, StakesLevel]:
    return {"low": StakesLevel(0.70, 0.50, 0.60), "medium": StakesLevel(0.85, 0.65, 0.75),
            "high": StakesLevel(0.95, 0.80, 0.90)}


@dataclass
class PolicyConfig:
    stakes: dict[str, StakesLevel] = field(default_factory=default_stakes)
    abstain: str = "escalate"           # the name of the abstaining action ("hold" in the tabular pack)
    min_verify_rows: int = 30           # holdout rows (and act-tier rows) needed before act is verified
    gate_min_p_above: float = 0.9       # Bernoulli gate on act
    miscalibration_p: float = 0.05      # act-tier accuracy below its mean prediction at this level -> confirm

    def level(self, stakes: str | None, default: str = "medium") -> StakesLevel:
        """The thresholds of a stakes level. An unknown level (a request's `stakes`, or a head's `stakes` override
        in rlcd.yaml) is a 422 validation error that lists the configured levels, never a 500."""
        name = stakes or default
        if name not in self.stakes:
            raise UnknownStakes(f"unknown stakes level {name!r}; configured: {list(self.stakes)}",
                                stakes=list(self.stakes))
        return self.stakes[name]

    def bernoulli(self) -> B.BernoulliConfig:
        return B.BernoulliConfig(min_p_above=self.gate_min_p_above)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["stakes"] = {k: asdict(v) for k, v in self.stakes.items()}
        return d

    @classmethod
    def from_dict(cls, d: dict | None) -> PolicyConfig:
        d = dict(d or {})
        stakes = default_stakes()
        for name, lv in (d.pop("stakes", None) or {}).items():
            base = asdict(stakes[name]) if name in stakes else {}
            stakes[name] = StakesLevel(**{**base, **lv})
        known = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(stakes=stakes, **known)

    def tier_name(self, tier: int) -> str:
        return self.abstain if tier == 0 else TIERS[tier]


def chosen(spec: Any, P: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(class index each row commits to, its probability). Gate heads always commit to `true`."""
    P = np.atleast_2d(np.asarray(P, float))
    if spec.decision == "gate":
        c = np.ones(len(P), int)
    else:
        c = P.argmax(axis=1)
    return c, P[np.arange(len(P)), c]


def tiers(p: np.ndarray, level: StakesLevel, conformal_threshold: float | None = None) -> np.ndarray:
    p = np.asarray(p, float)
    act = p >= level.act
    if conformal_threshold is not None:
        act &= p >= conformal_threshold
    return np.where(act, 2, np.where(p >= level.confirm, 1, 0))


def _stats(mask: np.ndarray, correct: np.ndarray, p: np.ndarray) -> dict:
    n = int(mask.sum())
    if n == 0:
        return {"n": 0, "correct": 0, "accuracy": None, "wilson_95": [None, None], "mean_predicted": None,
                "p_value_below_predicted": None}
    k = int(correct[mask].sum())
    lo, hi = wilson(k, n)
    mp = float(p[mask].mean())
    return {"n": n, "correct": k, "accuracy": _f(k / n), "wilson_95": [_f(lo), _f(hi)], "mean_predicted": _f(mp),
            "p_value_below_predicted": _f(B.binomial_test_below(k, n, mp))}


def evaluate(spec: Any, P: np.ndarray, y: np.ndarray, cfg: PolicyConfig,
             conformal_threshold: float | None = None) -> dict:
    """What the policy would have done on these (holdout) rows, for every stakes level."""
    c, p = chosen(spec, P)
    y = np.asarray(y, int)
    correct = c == y
    out = {}
    for name, level in cfg.stakes.items():
        t = tiers(p, level, conformal_threshold if spec.decision == "gate" else None)
        out[name] = {"thresholds": asdict(level),
                     "act": _stats(t == 2, correct, p), "act_or_confirm": _stats(t >= 1, correct, p),
                     "tiers": {cfg.tier_name(i): int((t == i).sum()) for i in range(3)}}
    out["rows"] = int(len(y))
    out["note"] = ("accuracy = the committed class came true (gate heads commit to `true`); "
                   "p_value_below_predicted: exact one-sided binomial test of the accuracy against the mean "
                   "predicted probability (small = the probabilities overstated)")
    return out


def verify(has_skill: bool, n_holdout: int, holdout_eval: dict, cfg: PolicyConfig, conformal: dict | None = None,
           spec: Any = None) -> dict:
    """The strongest tier the holdout evidence allows, per stakes level: {stakes: {max_tier, reason}}."""
    out = {}
    k = cfg.min_verify_rows
    for name in cfg.stakes:
        ev = holdout_eval.get(name, {})
        act = ev.get("act", {"n": 0})
        # p-values are rounded to 6 decimals: a tiny one becomes 0.0, which must count as significant (a falsy
        # `or 1.0` default would wave the most overconfident heads through to `act`)
        pv = act.get("p_value_below_predicted")
        if not has_skill:
            v = (0, "the holdout shows no skill over the base rate: every decision abstains until a retrain shows "
                    "skill")
        elif n_holdout < k:
            v = (1, f"only {n_holdout} trusted holdout rows (need {k}): skill is suggested, not verified; "
                    f"decisions are capped at confirm")
        elif act["n"] < k:
            v = (1, f"only {act['n']} act-tier rows at {name} stakes on the holdout (need {k}), so acting is not "
                    f"verified; decisions are capped at confirm")
        elif pv is not None and pv < cfg.miscalibration_p:
            v = (1, f"act-tier rows at {name} stakes came true {act['accuracy']} of the time against a mean "
                    f"prediction of {act['mean_predicted']} (one-sided p = {act['p_value_below_predicted']}): the "
                    f"probabilities overstate; decisions are capped at confirm")
        else:
            v = (2, f"holdout: {act['n']} act-tier rows at {name} stakes, accuracy {act['accuracy']} (95% Wilson "
                    f"{act['wilson_95'][0]}-{act['wilson_95'][1]}) against a mean prediction of "
                    f"{act['mean_predicted']}")
        if v[0] == 2 and conformal is not None and spec is not None and spec.decision == "gate":
            sel = conformal.get("selective") or {}
            chk = conformal.get("holdout") or {}
            if sel.get("threshold") is None:
                v = (1, f"conformal: {sel.get('reason', 'no threshold')}; decisions are capped at confirm")
            elif chk.get("consistent") is False:
                v = (1, f"conformal: the holdout error rate above the threshold ({chk.get('error_rate')}) exceeds "
                        f"the target {sel.get('target_error')}; decisions are capped at confirm")
        out[name] = {"max_tier": cfg.tier_name(v[0]), "tier": v[0], "reason": v[1]}
    return out


def decide(spec: Any, probs: np.ndarray, cfg: PolicyConfig, *, stakes: str | None = None,
           verdict: dict | None = None, conformal_threshold: float | None = None, ledger: dict | None = None,
           explore: bool = False, rng: np.random.Generator | None = None, calibrated: bool = True,
           missing_inputs: list[str] | None = None) -> dict:
    """One decision for one head on one row. `probs` in spec.labels order.

    `missing_inputs`: features almost every training row had that this request lacks (the service's input guard).
    The probabilities were calibrated on inputs that had them, so such a decision is capped at confirm."""
    stakes_name = stakes or spec.stakes
    level = cfg.level(stakes_name)
    P = np.asarray(probs, float).reshape(1, -1)
    c, p = chosen(spec, P)
    ci, pv = int(c[0]), float(p[0])
    thr = conformal_threshold if spec.decision == "gate" else None
    raw = int(tiers(np.array([pv]), level, thr)[0])
    warnings: list[str] = []
    cap = 0 if not calibrated else 2
    cap_reason = "the head is not trained: the answer is the uninformed prior" if not calibrated else None
    if calibrated and verdict is not None:
        v = verdict.get(stakes_name) or {}
        if "tier" in v and v["tier"] < cap:
            cap, cap_reason = int(v["tier"]), v["reason"]
    if calibrated and missing_inputs and cap > 1:
        shown = ", ".join(list(missing_inputs)[:4])
        cap, cap_reason = 1, (f"the request lacks {len(missing_inputs)} input feature(s) that almost every training "
                              f"row had ({shown}): the probabilities were calibrated on inputs that had them, so the "
                              f"decision is capped at confirm")
    tier = min(raw, cap)
    if cap_reason and raw > cap:
        warnings.append(f"head '{spec.name}': {cap_reason}")
    # Bernoulli gate on act
    label = spec.labels[ci]
    bcfg = cfg.bernoulli()
    wins = losses = 0
    if ledger is not None and label in ledger.get("bins", {}):
        wins, losses = ledger["bins"][label][B.bin_of(pv)]
    post = B.posterior(wins, losses, level.min_accuracy, bcfg)
    gate = {"mode": "posterior", "threshold": cfg.gate_min_p_above, "required_accuracy": level.min_accuracy,
            "passed": (post["p_above_required"] or 0.0) >= cfg.gate_min_p_above}
    if explore:
        draw = B.thompson(wins, losses, rng or np.random.default_rng(), bcfg)
        gate = {**gate, "mode": "thompson_sampling", "exploration": True, "sampled_accuracy": round(draw, 6),
                "passed": draw > level.min_accuracy}
        warnings.append(f"head '{spec.name}': exploration mode: the Bernoulli gate was decided by a draw from the "
                        f"accuracy posterior (Thompson sampling)")
    if tier == 2 and not gate["passed"]:
        tier = 1
        k = B.bin_of(pv)
        warnings.append(f"head '{spec.name}': Bernoulli gate: predictions like this one (P({label}) in "
                        f"{k / 10:.1f}-{(k + 1) / 10:.1f}) were right {wins} of {wins + losses} times; "
                        f"P(accuracy > {level.min_accuracy:g}) = {post['p_above_required']} is below "
                        f"{cfg.gate_min_p_above:g}, so act becomes confirm")
    if spec.decision == "gate":
        commit = label if tier > 0 else None
    else:
        commit = label
    return {"head": spec.name, "decision": spec.decision, "stakes": stakes_name, "class": label,
            "commit": commit, "p": round(pv, 6), "tier": cfg.tier_name(tier), "uncapped_tier": cfg.tier_name(raw),
            "max_tier": cfg.tier_name(cap), "thresholds": {**asdict(level), "conformal": thr},
            "bernoulli": {**post, "wins": wins, "losses": losses, "bin": B.bin_of(pv),
                          "decisions_needed": B.decisions_needed(pv, level.min_accuracy, bcfg)},
            "gate": gate, "calibrated": calibrated, "warnings": warnings}


def weakest(decisions: list[dict], cfg: PolicyConfig) -> str:
    """A chain of decisions is only as strong as its weakest link."""
    names = [cfg.tier_name(i) for i in range(3)]
    if not decisions:
        return cfg.tier_name(0)
    return names[min(names.index(d["tier"]) for d in decisions)]
