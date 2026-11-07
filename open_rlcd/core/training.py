"""Training one head: split, fit the fold ensemble, calibrate on trusted rows, report on the untouched holdout.

    |------------------------- development -------------------------|------ holdout ------|
    |  fold 1: fit | cal |                                           |  trusted rows only, |
    |  fold 2: fit ......... | cal |                                 |  scored once by the |
    |  ...                                                           |  served ensemble    |

  * base models fit on every development row on the training side of a fold: trusted, judge and feedback rows
    (feedback up-weighted). No class weights, no resampling.
  * each calibrator is fitted on the trusted rows of its fold's calibration block only;
  * the served model is the ensemble of calibrated members: exactly the model the holdout scored;
  * the holdout is never used for fitting, calibrating, choosing thresholds or the conformal cut.

The report carries: holdout calibration metrics against the base-rate forecaster, the block-bootstrap skill
test, per-stakes policy results, the verdict (strongest tier the evidence allows), the conformal threshold and
its holdout check, the judge's agreement with humans, per-segment slices and a data card.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import numpy as np

from . import bernoulli as B
from . import calibration as C
from . import metrics as M
from . import policy as P
from . import splits as S
from .dataset import Dataset
from .head import CalibratedEnsemble, HeadSpec, make_base, resolve_model

MIN_SLICE_HOLDOUT = 50


class TrainError(Exception):
    pass


def now_iso() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass
class TrainConfig:
    holdout_fraction: float = 0.2
    n_splits: int = 5
    ordering: str = "auto"              # auto | time | group | row
    gap: int = 0                        # time ordering: extra timestamps left out before each calibration block
    calibration: str = "auto"           # auto | sigmoid | isotonic | temperature
    isotonic_min_per_class: int = C.ISOTONIC_MIN_PER_CLASS
    conformal_delta: float = 0.1
    n_estimators: int | None = None     # LightGBM override
    min_trusted: int = 30               # fewest trusted rows a head needs at all
    seed: int = 7
    policy: P.PolicyConfig = field(default_factory=P.PolicyConfig)


@dataclass
class TrainedHead:
    spec: HeadSpec
    ensemble: CalibratedEnsemble
    report: dict
    metrics: dict
    verdict: dict
    conformal: dict | None
    ledger: dict
    calibration: dict
    data: dict
    notes: list[str]
    trained_at: str
    model: str

    def summary(self) -> dict:
        return {"head": self.spec.name, "type": self.spec.type, "model": self.model, "trained_at": self.trained_at,
                "metrics": self.metrics, "verdict": self.verdict, "calibration": self.calibration,
                "conformal": _conformal_summary(self.conformal), "data": self.data, "notes": self.notes}


def _conformal_summary(c: dict | None) -> dict | None:
    if not c:
        return None
    sel = c.get("selective") or {}
    return {"target_error": sel.get("target_error"), "threshold": sel.get("threshold"),
            "coverage_dev": sel.get("coverage"), "holdout": c.get("holdout"),
            "set_quantile": (c.get("sets") or {}).get("q"), "reason": sel.get("reason")}


def _prior(y: np.ndarray, k: int) -> np.ndarray:
    """Base-rate forecast with add-half smoothing (a class never seen in training is not given probability 0)."""
    counts = np.bincount(np.asarray(y, int), minlength=k).astype(float)
    return (counts + 0.5) / (counts.sum() + 0.5 * k)


def _day(ts: int | None) -> str | None:
    if ts is None or ts < 0:
        return None
    return datetime.fromtimestamp(int(ts), UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def train_head(spec: HeadSpec, ds: Dataset, cfg: TrainConfig | None = None) -> TrainedHead:
    cfg = cfg or TrainConfig()
    labels = spec.labels
    k = len(labels)
    notes: list[str] = list(ds.notes)
    if len(ds) == 0:
        raise TrainError(f"{spec.name}: no labelled rows")
    if ds.y.min() < 0 or ds.y.max() >= k:
        raise TrainError(f"{spec.name}: labels must be class indices 0..{k - 1}")
    n_trusted = int(ds.trusted.sum())
    if n_trusted < cfg.min_trusted:
        raise TrainError(f"{spec.name}: {n_trusted} trusted (human-verified or ground-truth) rows; at least "
                         f"{cfg.min_trusted} are needed to calibrate and to measure anything. Audit more rows "
                         f"(`rlcd label audit`).")
    absent = [labels[c] for c in range(k) if not (ds.y == c).any()]
    if absent:
        raise TrainError(f"{spec.name}: no labelled rows for {absent}; every class of the head needs examples to be "
                         f"learned and calibrated. Label some, or remove the class from the head.")
    present = np.unique(ds.y[ds.trusted])
    if len(present) < 2:
        raise TrainError(f"{spec.name}: all {n_trusted} trusted rows carry the label {labels[int(present[0])]!r}; "
                         f"calibrating and measuring a head needs trusted examples of at least two classes. Audit "
                         f"rows of the other classes (`rlcd label audit`).")
    model_kind, note = resolve_model(spec.model)
    if note:
        notes.append(note)
    params = dict(spec.params)
    if model_kind == "lightgbm" and cfg.n_estimators and "n_estimators" not in params:
        params["n_estimators"] = int(cfg.n_estimators)

    # ------------------------------------------------------------------ holdout
    try:
        split = S.holdout_split(ds, cfg.holdout_fraction, cfg.ordering, cfg.seed)
    except S.NotEnoughData as e:
        raise TrainError(f"{spec.name}: {e}") from e
    notes += split.notes
    mode = split.mode
    h_idx = S.scoring_order(ds, np.flatnonzero(split.holdout), mode, cfg.seed)
    n_hold = len(h_idx)
    if n_hold < 10:
        raise TrainError(f"{spec.name}: only {n_hold} trusted holdout rows after the split; need at least 10 "
                         f"(add trusted labels or raise holdout_fraction)")

    # ------------------------------------------------------------------ calibration folds
    try:
        folds, fnotes = S.calibration_folds(ds, split.dev, cfg.n_splits, mode, cfg.gap, cfg.seed, n_classes=k)
    except S.NotEnoughData as e:
        raise TrainError(f"{spec.name}: {e}") from e
    notes += fnotes
    method, why = C.choose_method(cfg.calibration, [ds.y[ca] for _, ca in folds], k, cfg.isotonic_min_per_class)

    members, fold_info = [], []
    oof_idx, oof_P = [], []
    for fi, (tr, ca) in enumerate(folds):
        base = make_base(model_kind, k, params, cfg.seed + fi)
        base.fit(ds.X[tr], ds.y[tr], ds.weight[tr])
        cal = C.Calibrator(method, k).fit(base.raw(ds.X[ca]), ds.y[ca])
        members.append((base, cal))
        oof_idx.append(ca)
        oof_P.append(cal.predict(base.raw(ds.X[ca])))
        fold_info.append({"n_train": int(len(tr)), "n_train_trusted": int(ds.trusted[tr].sum()),
                          "n_calibration": int(len(ca))})
    ensemble = CalibratedEnsemble(spec, members, ds.feature_names, ds.feature_version)
    oof_i = np.concatenate(oof_idx)
    oof = np.vstack(oof_P)

    # ------------------------------------------------------------------ holdout report
    Xh, yh = ds.X[h_idx], ds.y[h_idx]
    Ph = ensemble.predict_proba(Xh)
    dev_trusted = split.dev & ds.trusted
    pooled = _prior(ds.y[dev_trusted], k)
    prior = np.tile(pooled, (n_hold, 1))
    baseline_note = "predicts the class frequencies of the trusted development rows (add-half smoothing)"
    seg_train: dict[str, int] = {}
    if ds.segment is not None:
        for s in np.unique(ds.segment[h_idx]):
            m = dev_trusted & (ds.segment == s)
            seg_train[str(s)] = int(m.sum())
            if m.sum() >= 50:
                prior[ds.segment[h_idx] == s] = _prior(ds.y[m], k)
        baseline_note = ("predicts, for each row, the class frequencies of its own segment's trusted development "
                         "rows (pooled when the segment has fewer than 50); add-half smoothing")
    blocks = S.skill_blocks(ds, h_idx, mode)
    report = M.calibration_report(Ph, yh, labels, prior, baseline_note, blocks, cfg.seed)

    # ------------------------------------------------------------------ conformal
    conformal = None
    if spec.conformal is not None:
        target = float(spec.conformal)
        y_oof = ds.y[oof_i]
        if spec.decision == "gate":
            sel = C.selective_threshold(oof[:, 1], y_oof == 1, target, cfg.conformal_delta)
            chk = C.selective_check(Ph[:, 1], yh == 1, sel.get("threshold"), target)
            conformal = {"selective": sel, "holdout": chk}
        else:
            q = C.set_quantile(oof[np.arange(len(y_oof)), y_oof], target)
            cover = None
            if q["q"] is not None:
                sets = [C.prediction_set(Ph[i], q["q"]) for i in range(n_hold)]
                cover = {"coverage": M._f(np.mean([int(yh[i]) in s for i, s in enumerate(sets)])),
                         "mean_set_size": M._f(np.mean([len(s) for s in sets])), "target_coverage": 1 - target}
            conformal = {"sets": q, "holdout": cover}
        report["conformal"] = conformal
    threshold = (conformal or {}).get("selective", {}).get("threshold") if conformal else None

    # ------------------------------------------------------------------ policy, verdict, ledger
    pol = P.evaluate(spec, Ph, yh, cfg.policy, threshold)
    verdict = P.verify(report["skill"]["has_skill"], n_hold, pol, cfg.policy, conformal, spec)
    report["policy"] = {"config": cfg.policy.to_dict(), "holdout": pol, "verdict": verdict}
    c_h, _ = P.chosen(spec, Ph)
    ledger = B.holdout_table(Ph, yh, labels, c_h)

    # ------------------------------------------------------------------ judge agreement, slices
    if ds.judge_y is not None:
        both = ds.trusted & (ds.judge_y >= 0)
        report["judge_agreement"] = M.agreement(ds.judge_y[both], ds.y[both], labels)
    if ds.segment is not None:
        by_seg = {}
        for s in np.unique(ds.segment[h_idx]):
            m = ds.segment[h_idx] == s
            if m.sum() >= MIN_SLICE_HOLDOUT:
                rep = M.calibration_report(Ph[m], yh[m], labels, prior[m], baseline_note,
                                           blocks if isinstance(blocks, int) else blocks[m], cfg.seed)
                rep.pop("definitions")
                rep["n_train_trusted"] = seg_train.get(str(s), 0)
                by_seg[str(s)] = rep
        report["by_segment"] = by_seg

    # ------------------------------------------------------------------ data card
    data = ds.counts()
    data.update({"development_rows": int(split.dev.sum()), "development_trusted": int(dev_trusted.sum()),
                 "holdout_rows": n_hold, "purged_rows": split.purged, "ordering": mode, "split": split.cut,
                 "time_from": _day(data.get("time_from")), "time_to": _day(data.get("time_to"))})
    if mode == "time":
        data["holdout_from"] = _day(split.cut.get("time"))
    report["split"] = {"ordering": mode, "holdout_fraction": cfg.holdout_fraction, **split.cut,
                       "holdout_rows": n_hold, "development_rows": int(split.dev.sum()), "purged_rows": split.purged,
                       "folds": fold_info}
    calib = {"method": method, "why": why, "folds": fold_info, "ensemble_members": len(members),
             "fitted_on": "trusted rows of each fold's calibration block"}
    summary = M.summary(report)
    summary.update({"calibration_method": method, "verdict": {s: v["max_tier"] for s, v in verdict.items()}})
    if conformal and spec.decision == "gate":
        summary["conformal_threshold"] = threshold
    return TrainedHead(spec=spec, ensemble=ensemble, report=report, metrics=summary, verdict=verdict,
                       conformal=conformal, ledger=ledger, calibration=calib, data=data, notes=notes,
                       trained_at=now_iso(), model=model_kind)


def describe_failure(e: Exception) -> dict:
    return {"status": "failed", "error": str(e) if isinstance(e, TrainError) else f"{type(e).__name__}: {e}"}


def holdout_predictions(trained: TrainedHead, ds: Dataset, cfg: TrainConfig) -> dict[str, Any]:
    """Re-derive the holdout rows and predictions (for reports and tests)."""
    split = S.holdout_split(ds, cfg.holdout_fraction, cfg.ordering, cfg.seed)
    idx = S.scoring_order(ds, np.flatnonzero(split.holdout), split.mode, cfg.seed)
    return {"idx": idx, "P": trained.ensemble.predict_proba(ds.X[idx]), "y": ds.y[idx]}
