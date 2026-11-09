"""Why a decision is uncertain: name the source, so the fix can target it.

Low confidence can mean very different things, and each has a different remedy (the guide, "distinguish
uncertainty sources"): the evidence is missing (send the field, load the record), two options are genuinely
plausible (add evidence that separates them, or review the option boundaries), the distribution is spread over
everything (the request may be outside what the head was trained on), the head has no verified skill (label
more), or the track record of predictions like this one is too short (wait for outcomes). `classify` reads one
decision and says which apply; receipts keep the sources and the metrics endpoint counts them per head, so a team
can see which intervention would resolve most cases.

Sources (a decision can have several; none when it acts on a clear, verified answer):

    missing_evidence           the input guard found features almost every training row had missing
    options_removed            deterministic code removed options that held probability mass (`allowed`)
    ambiguous                  the top two options are within AMBIGUOUS_MARGIN (a noul within it of 0.5)
    diffuse                    normalised entropy above DIFFUSE_ENTROPY: no option stands out
    below_threshold            the probability of the chosen class is below the act threshold
    no_skill                   the head's holdout showed no skill over the base rate
    unverified                 the holdout could not verify acting (too few rows, overstated probabilities,
                               conformal promise not made)
    insufficient_track_record  the Bernoulli gate failed: too few right predictions like this one so far
    untrained                  the head is not trained (uninformed prior)
"""
from __future__ import annotations

import math
from collections.abc import Sequence

AMBIGUOUS_MARGIN = 0.10
DIFFUSE_ENTROPY = 0.85
LARGE_REMOVED_MASS = 0.5


def entropy(probabilities: Sequence[float]) -> float:
    """Shannon entropy normalised to [0, 1] by log(n) (0 = certain, 1 = uniform)."""
    ps = [float(p) for p in probabilities if float(p) > 0]
    n = len(probabilities)
    if n < 2:
        return 0.0
    h = -sum(p * math.log(p) for p in ps)
    return max(0.0, min(1.0, h / math.log(n)))


def margin(probabilities: Sequence[float]) -> float:
    """Gap between the two largest probabilities (1.0 for a single option)."""
    ps = sorted((float(p) for p in probabilities), reverse=True)
    return 1.0 if len(ps) < 2 else ps[0] - ps[1]


def classify(probabilities: Sequence[float], labels: Sequence[str], decision: dict, *,
             missing_inputs: Sequence[str] | None = None, removed_mass: float = 0.0,
             verdict_tier: int | None = None) -> dict:
    """{sources, margin, entropy, top2} for one head decision (`decision`: policy.decide's output;
    `verdict_tier`: the head's holdout verdict at the decision's stakes, 0 abstain / 1 confirm / 2 act — without
    it the verdict is read from the decision's max_tier, which the input guard can also lower)."""
    ps = [float(p) for p in probabilities]
    order = sorted(range(len(ps)), key=lambda i: (-ps[i], i))
    sources: list[str] = []
    calibrated = bool(decision.get("calibrated", True))
    if not calibrated:
        sources.append("untrained")
    if missing_inputs:
        sources.append("missing_evidence")
    if removed_mass > 0 and removed_mass >= LARGE_REMOVED_MASS:
        sources.append("options_removed")
    m = margin(ps)
    if len(ps) == 2:
        # a noul: ambiguous when P(true) is close to 0.5 (the two "options" are true and false)
        if abs(ps[1] - 0.5) < AMBIGUOUS_MARGIN / 2:
            sources.append("ambiguous")
    elif m < AMBIGUOUS_MARGIN:
        sources.append("ambiguous")
    h = entropy(ps)
    if len(ps) > 2 and h > DIFFUSE_ENTROPY:
        sources.append("diffuse")
    thr = (decision.get("thresholds") or {}).get("act")
    p = decision.get("p")
    if thr is not None and p is not None and float(p) < float(thr):
        sources.append("below_threshold")
    max_tier = decision.get("max_tier")
    if calibrated:
        vt = verdict_tier
        if vt is None:
            vt = 2 if max_tier == "act" else 1 if max_tier == "confirm" else 0
            if vt == 1 and missing_inputs:
                vt = None      # that cap may be the input guard's, not the holdout verdict's
        if vt == 0:
            sources.append("no_skill")
        elif vt == 1:
            sources.append("unverified")
    gate = decision.get("gate") or {}
    if decision.get("uncapped_tier") == "act" and max_tier == "act" and gate.get("passed") is False:
        sources.append("insufficient_track_record")
    top2 = [labels[i] for i in order[:2]] if len(labels) >= 2 else list(labels)
    return {"sources": sources, "margin": round(m, 6), "entropy": round(h, 6), "top2": top2}
