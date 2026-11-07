"""Synthetic churn-like data whose labels come from KNOWN models, so the true probabilities are known.

Each customer (`customer_id`) contributes 1-8 consecutive monthly rows (`month`, ISO date). Features:
tenure_months, monthly_charges (3 % missing), support_calls, plan, contract, region (2 % missing; no effect on
any label), channel (no effect; used as the segment), paperless. Labels, drawn independently per row from
functions of that row's features only (no hidden customer effect, so P(label | features) is exactly the
generating model and a well-specified model is calibrated by construction):

    churned          noul    "yes" / "no"                    logistic in the features
    plan_tier_next   choice  downgrade / same / upgrade      multinomial logit
    satisfaction     score   0-3                             adjacent-category logit (ordered)

`signal` scales every effect (0 = labels independent of the features); `noise=True` draws every label from its
base rate alone, independent of the features: a model trained on it must show no skill and abstain.
Synthetic metrics are a sanity check of the machinery, not accuracy on anyone's real customers.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

PLANS = ("basic", "standard", "premium")
CONTRACTS = ("month-to-month", "one-year", "two-year")
REGIONS = ("north", "south", "east", "west")
CHANNELS = ("web", "store", "partner")
NEXT_PLAN = ("downgrade", "same", "upgrade")
SATISFACTION_LEVELS = ("very unhappy", "unhappy", "satisfied", "very satisfied")
COLUMNS = ("row_id", "customer_id", "month", "tenure_months", "monthly_charges", "support_calls", "plan",
           "contract", "region", "channel", "paperless", "churned", "plan_tier_next", "satisfaction")
FEATURE_COLUMNS = ("tenure_months", "monthly_charges", "support_calls", "plan", "contract", "region", "channel",
                   "paperless")
_SAT_OFFSETS = np.array([0.0, 1.0, 1.2, 0.2])


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def _softmax(L: np.ndarray) -> np.ndarray:
    E = np.exp(L - L.max(axis=1, keepdims=True))
    return E / E.sum(axis=1, keepdims=True)


def true_probabilities(rows: list[dict], signal: float = 1.0, noise: bool = False) -> dict[str, np.ndarray]:
    """The generating probabilities for rows given as dicts of the feature columns (CSV strings or numbers).
    Usable on any row, e.g. a hand-written serve-time state, to compare the served probability with the truth."""
    def num(r: dict, c: str) -> float:
        v = r.get(c)
        return float("nan") if v in (None, "") else float(v)

    tenure = np.array([num(r, "tenure_months") for r in rows])
    calls = np.array([num(r, "support_calls") for r in rows])
    charges = np.array([num(r, "monthly_charges") for r in rows])
    z_ten = (tenure - 24.0) / 14.0
    z_calls = (calls - 1.5) / 1.3
    z_chg = np.where(np.isnan(charges), 0.0, (charges - 55.0) / 28.0)
    plan = np.array([str(r.get("plan")) for r in rows])
    contract = np.array([str(r.get("contract")) for r in rows])
    basic, premium = (plan == "basic").astype(float), (plan == "premium").astype(float)
    mtm, two = (contract == "month-to-month").astype(float), (contract == "two-year").astype(float)
    paperless = np.array([str(r.get("paperless")) == "yes" for r in rows], float)
    s = 0.0 if noise else float(signal)
    churn = _sigmoid(-1.3 + s * (0.9 * z_calls - 0.7 * z_ten + 0.6 * mtm - 0.7 * two + 0.35 * z_chg
                                 + 0.3 * paperless - 0.2 * premium))
    n = len(rows)
    L = np.column_stack([
        -1.2 + s * (0.8 * z_calls + 0.7 * premium - 0.5 * z_ten - 1.5 * basic),
        np.full(n, 0.8),
        -1.0 + s * (0.6 * z_ten + 0.8 * basic - 1.5 * premium - 0.5 * z_calls),
    ])
    latent = s * (-0.9 * z_calls + 0.4 * z_ten + 0.3 * premium - 0.3 * mtm)
    S = latent[:, None] * np.arange(4)[None, :] + _SAT_OFFSETS[None, :]
    return {"churned": churn, "plan_tier_next": _softmax(L), "satisfaction": _softmax(S)}


def generate(n: int = 4000, seed: int = 7, signal: float = 1.0,
             noise: bool = False) -> tuple[list[dict], dict[str, np.ndarray]]:
    """(rows as dicts of CSV strings in file order, true probabilities aligned with the rows)."""
    if n < 1:
        raise ValueError("n must be positive")
    rng = np.random.default_rng(seed)
    raw: list[tuple[int, int, dict]] = []
    cust = 0
    while len(raw) < n:
        cust += 1
        months = int(rng.integers(1, 9))
        start = int(rng.integers(0, 36))
        tenure0 = int(rng.integers(0, 48))
        plan = int(rng.choice(3, p=[0.45, 0.35, 0.20]))
        contract = int(rng.choice(3, p=[0.5, 0.3, 0.2]))
        region = "" if rng.random() < 0.02 else REGIONS[int(rng.integers(0, 4))]
        channel = CHANNELS[int(rng.integers(0, 3))]
        paperless = "yes" if rng.random() < 0.6 else "no"
        base_charge = (25.0, 55.0, 95.0)[plan] + rng.normal(0, 8)
        for m in range(months):
            if len(raw) >= n:
                break
            charges = base_charge + rng.normal(0, 3)
            calls = int(rng.poisson(1.2 + 0.8 * (plan == 0)))
            raw.append((start + m, cust, {
                "customer_id": f"C{cust:05d}",
                "month": f"{2022 + (start + m) // 12:04d}-{(start + m) % 12 + 1:02d}-01",
                "tenure_months": str(tenure0 + m),
                "monthly_charges": "" if rng.random() < 0.03 else f"{charges:.2f}",
                "support_calls": str(calls), "plan": PLANS[plan], "contract": CONTRACTS[contract],
                "region": region, "channel": channel, "paperless": paperless}))
    raw.sort(key=lambda r: (r[0], r[1]))
    rows = [r[2] for r in raw]
    truth = true_probabilities(rows, signal, noise)
    churned = rng.random(n) < truth["churned"]
    nxt = (rng.random(n)[:, None] > np.cumsum(truth["plan_tier_next"], axis=1)).sum(axis=1)
    sat = (rng.random(n)[:, None] > np.cumsum(truth["satisfaction"], axis=1)).sum(axis=1)
    for i, r in enumerate(rows):
        r["row_id"] = f"R{i + 1:06d}"
        r["churned"] = "yes" if churned[i] else "no"
        r["plan_tier_next"] = NEXT_PLAN[min(int(nxt[i]), 2)]
        r["satisfaction"] = str(min(int(sat[i]), 3))
    return [{c: r[c] for c in COLUMNS} for r in rows], truth


def write_synthetic_csv(path: str | Path, n: int = 4000, seed: int = 7, signal: float = 1.0,
                        noise: bool = False) -> dict:
    """Write the synthetic table as CSV (creating parent directories); returns a short description of it."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows, truth = generate(n, seed, signal, noise)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(COLUMNS))
        w.writeheader()
        w.writerows(rows)
    return {"path": str(path), "rows": len(rows), "customers": len({r["customer_id"] for r in rows}),
            "columns": list(COLUMNS), "signal": 0.0 if noise else signal, "noise": noise,
            "true_rates": {"churned": round(float(truth["churned"].mean()), 4),
                           "plan_tier_next": dict(zip(NEXT_PLAN, np.round(truth["plan_tier_next"].mean(0), 4)
                                                      .tolist(), strict=True)),
                           "satisfaction": np.round(truth["satisfaction"].mean(0), 4).tolist()}}


def synthetic_pack_config(path: str | Path) -> dict:
    """pack_config for the synthetic table: three heads (noul, choice, score), customer groups, monthly time."""
    return {"data": str(path), "id": "row_id", "group": "customer_id", "time": "month", "segment": "channel",
            "targets": {
                "churned": {"label": "churned", "description": "will this customer churn next month?"},
                "plan_tier_next": {"label": "plan_tier_next", "classes": list(NEXT_PLAN),
                                   "description": "the customer's plan change next month"},
                "satisfaction": {"label": "satisfaction", "type": "score", "levels": list(SATISFACTION_LEVELS),
                                 "description": "satisfaction at the next survey"}}}
