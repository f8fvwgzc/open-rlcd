"""Heads: a calibrated answerer for one fixed question.

A head = a HeadSpec (what it answers, from a pack) + a CalibratedEnsemble (how): one base model per calibration
fold, each followed by a calibrator fitted on that fold's trusted calibration block (the `ensemble=True` scheme
of scikit-learn's CalibratedClassifierCV); the served probability is the geometric (log-linear) pool of the
calibrated members, which, unlike their plain average, does not come out underconfident when members disagree.

Base models:
  * logreg    standardised features -> LogisticRegression. Raw scores are logits; explanations are exact
              (coef * standardised value). Saved as safetensors.
  * lightgbm  small, regularised gradient-boosted trees. Raw scores via raw_score=True; explanations are
              TreeSHAP contributions (pred_contrib). Saved as LightGBM's native text model.
              Imported lazily: on macOS LightGBM needs the OpenMP runtime (`brew install libomp`).

No class weights and no resampling anywhere: probabilities must stay frequencies.
Saving writes only JSON, safetensors and LightGBM text; loading never executes code.
"""
from __future__ import annotations

import json
import sys
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .calibration import Calibrator
from .contract import MAX_CHOICE_OPTIONS, MAX_SCORE_LEVELS, MIN_SCORE_LEVELS, QUESTION_TYPES

MODELS = ("logreg", "lightgbm", "auto")
DECISIONS = ("argmax", "gate")
NOUL_LABELS = ("false", "true")
TRUE_WORDS = {"true", "yes", "1", "y", "t", "1.0"}
FALSE_WORDS = {"false", "no", "0", "n", "f", "0.0"}


class LightGBMUnavailable(RuntimeError):
    pass


def lightgbm_module():
    try:
        import lightgbm  # noqa: PLC0415
    except (ImportError, OSError) as e:  # OSError: libomp missing on macOS
        raise LightGBMUnavailable(
            f"LightGBM cannot be imported ({type(e).__name__}: {e}). On macOS install the OpenMP runtime with "
            f"`brew install libomp`; or set `model: logreg` for this head.") from e
    return lightgbm


def lightgbm_available() -> bool:
    try:
        lightgbm_module()
        return True
    except LightGBMUnavailable:
        return False


def openmp_conflict() -> bool:
    """True when LightGBM must run single-threaded to stay alive.

    On macOS the faiss-cpu wheel ships its own OpenMP runtime next to the one LightGBM links (Homebrew libomp).
    Once both are loaded, a multithreaded parallel region in either one crashes or deadlocks the process
    (measured: a segfault in LightGBM's Dataset construction after a FAISS search; a hang the other way round).
    Single-threaded LightGBM is safe, so it runs that way while faiss is loaded: slower training, identical
    model format. This guards LightGBM only; FAISS code must limit its own threads (faiss.omp_set_num_threads)."""
    return sys.platform == "darwin" and "faiss" in sys.modules


def _thread_params() -> dict:
    return {"num_threads": 1} if openmp_conflict() else {}


@dataclass(frozen=True)
class HeadSpec:
    name: str
    type: str                                # noul | choice | score
    classes: tuple[str, ...] = ()            # choice options / score level names; () for a noul
    view: str = "default"                    # the feature view the pack's provider builds
    model: str = "logreg"                    # logreg | lightgbm | auto
    decision: str = "argmax"                 # argmax | gate (noul: true = proceed)
    stakes: str = "medium"                   # low | medium | high
    conformal: float | None = None           # target error rate (gate heads) / miscoverage (others)
    description: str = ""
    input: dict = field(default_factory=dict)
    params: dict = field(default_factory=dict)
    levels: tuple[str, ...] = ()             # score: level descriptions for `legend` (default: classes)
    positive: str | None = None              # documentation: what `true` means for a noul

    def __post_init__(self) -> None:
        if self.type not in QUESTION_TYPES:
            raise ValueError(f"head {self.name!r}: type must be one of {QUESTION_TYPES}")
        if self.type == "noul" and self.classes:
            raise ValueError(f"head {self.name!r}: a noul head has no classes")
        if self.type == "choice" and not 2 <= len(self.classes) <= MAX_CHOICE_OPTIONS:
            raise ValueError(f"head {self.name!r}: a choice head needs 2 to {MAX_CHOICE_OPTIONS} classes")
        if self.type == "score" and not MIN_SCORE_LEVELS <= len(self.classes) <= MAX_SCORE_LEVELS:
            raise ValueError(f"head {self.name!r}: a score head needs {MIN_SCORE_LEVELS} to {MAX_SCORE_LEVELS} "
                             f"levels")
        if len(set(self.classes)) != len(self.classes):
            raise ValueError(f"head {self.name!r}: duplicate classes")
        if self.model not in MODELS:
            raise ValueError(f"head {self.name!r}: model must be one of {MODELS}")
        if self.decision not in DECISIONS:
            raise ValueError(f"head {self.name!r}: decision must be one of {DECISIONS}")
        if self.decision == "gate" and self.type != "noul":
            raise ValueError(f"head {self.name!r}: decision 'gate' is for noul heads")
        if self.conformal is not None and not 0 < float(self.conformal) < 0.5:
            raise ValueError(f"head {self.name!r}: conformal target must be in (0, 0.5)")
        if self.levels and len(self.levels) != len(self.classes):
            raise ValueError(f"head {self.name!r}: {len(self.levels)} level descriptions for "
                             f"{len(self.classes)} levels")

    @property
    def labels(self) -> tuple[str, ...]:
        """Internal class order of the model (noul: false, true)."""
        return NOUL_LABELS if self.type == "noul" else tuple(self.classes)

    @property
    def legend(self) -> tuple[str, ...]:
        return tuple(self.levels) if self.levels else tuple(self.classes)

    def label_index(self, label: Any) -> int:
        """Class index of a label as people write it.

        noul: true/false, yes/no, 1/0, or the head's own `positive` value (e.g. a tabular column's "churn") for
        true. choice: a class name. score: a level name first, then a level description from `levels`, then a
        0-based index (so on levels named "1".."5" the label "4" is the level named "4", not index 4)."""
        if self.type == "noul":
            s = str(label).strip().lower()
            if isinstance(label, bool):
                return int(label)
            if self.positive is not None and s == str(self.positive).strip().lower():
                return 1
            if s in TRUE_WORDS:
                return 1
            if s in FALSE_WORDS:
                return 0
            raise ValueError(f"head {self.name!r}: noul label must be true/false (or yes/no, 1/0), got {label!r}")
        s = str(label)
        if s in self.classes:
            return self.classes.index(s)
        if self.type == "score" and self.levels and s in self.levels:
            return self.levels.index(s)
        if self.type == "score":
            try:
                i = int(float(s))
            except ValueError:
                i = -1
            if 0 <= i < len(self.classes):
                return i
        raise ValueError(f"head {self.name!r}: label must be one of {list(self.classes)}, got {label!r}")

    def to_dict(self) -> dict:
        d = asdict(self)
        d["classes"], d["levels"] = list(self.classes), list(self.levels)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> HeadSpec:
        d = dict(d)
        d["classes"] = tuple(d.get("classes") or ())
        d["levels"] = tuple(d.get("levels") or ())
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in d.items() if k in known})


# ----------------------------------------------------------------------------- base models

LOGREG_DEFAULTS = {"C": 1.0, "max_iter": 2000}
LIGHTGBM_DEFAULTS = {"n_estimators": 300, "learning_rate": 0.03, "num_leaves": 15, "max_depth": 5,
                     "subsample": 0.8, "subsample_freq": 1, "colsample_bytree": 0.7, "reg_lambda": 5.0}


class LogReg:
    kind = "logreg"

    def __init__(self, n_classes: int, params: dict | None = None, seed: int = 7):
        self.n_classes, self.params, self.seed = n_classes, {**LOGREG_DEFAULTS, **(params or {})}, seed
        self.mean = self.scale = self.coef = self.intercept = None
        self.fill = None

    def _std(self, X: np.ndarray) -> np.ndarray:
        # NaN = missing; +-inf (e.g. a log of zero in a feature provider) is treated as missing too, since a
        # linear model cannot use it and scikit-learn refuses it
        X = np.asarray(X, np.float64)
        X = np.where(np.isfinite(X), X, self.fill)
        return (X - self.mean) / self.scale

    def fit(self, X: np.ndarray, y: np.ndarray, w: np.ndarray | None = None) -> LogReg:
        from sklearn.linear_model import LogisticRegression

        X = np.asarray(X, np.float64)
        X = np.where(np.isfinite(X), X, np.nan)
        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)   # a feature never observed: its fill becomes 0
            fill = np.nanmean(X, axis=0)
        self.fill = np.where(np.isnan(fill), 0.0, fill)
        Xf = np.where(np.isnan(X), self.fill, X)
        self.mean = Xf.mean(axis=0)
        sd = Xf.std(axis=0)
        self.scale = np.where(sd > 1e-12, sd, 1.0)
        clf = LogisticRegression(C=float(self.params["C"]), max_iter=int(self.params["max_iter"]),
                                 random_state=self.seed)
        clf.fit(self._std(X), y, sample_weight=w)
        present = list(clf.classes_)
        coef = np.zeros((self.n_classes if self.n_classes > 2 else 1, X.shape[1]))
        inter = np.zeros(coef.shape[0])
        if self.n_classes == 2:
            coef[0], inter[0] = clf.coef_[0], clf.intercept_[0]
        elif len(present) == 2:  # sklearn fits a binary model when only two of the classes are present
            coef[present[1]], inter[present[1]] = clf.coef_[0] / 2, clf.intercept_[0] / 2
            coef[present[0]], inter[present[0]] = -clf.coef_[0] / 2, -clf.intercept_[0] / 2
            for c in set(range(self.n_classes)) - set(present):
                inter[c] = -30.0
        else:
            for j, c in enumerate(present):
                coef[c], inter[c] = clf.coef_[j], clf.intercept_[j]
            # classes absent from training get a very low logit
            for c in set(range(self.n_classes)) - set(present):
                inter[c] = -30.0
        self.coef, self.intercept = coef, inter
        return self

    def raw(self, X: np.ndarray) -> np.ndarray:
        Z = self._std(X) @ self.coef.T + self.intercept
        return Z[:, 0] if self.n_classes == 2 else Z

    def contributions(self, x: np.ndarray, class_index: int) -> np.ndarray:
        z = self._std(np.atleast_2d(x))[0]
        if self.n_classes == 2:
            return z * self.coef[0] * (1.0 if class_index == 1 else -1.0)
        return z * self.coef[class_index]

    def arrays(self) -> dict[str, np.ndarray]:
        return {"coef": self.coef, "intercept": self.intercept, "mean": self.mean, "scale": self.scale,
                "fill": self.fill}

    @classmethod
    def from_arrays(cls, n_classes: int, arrays: dict[str, np.ndarray], params: dict | None = None) -> LogReg:
        m = cls(n_classes, params)
        m.coef, m.intercept = np.asarray(arrays["coef"], float), np.asarray(arrays["intercept"], float)
        m.mean, m.scale, m.fill = (np.asarray(arrays[k], float) for k in ("mean", "scale", "fill"))
        return m


class LightGBM:
    kind = "lightgbm"

    def __init__(self, n_classes: int, params: dict | None = None, seed: int = 7):
        self.n_classes, self.params, self.seed = n_classes, {**LIGHTGBM_DEFAULTS, **(params or {})}, seed
        self.booster = None
        self.n_features = None

    def fit(self, X: np.ndarray, y: np.ndarray, w: np.ndarray | None = None) -> LightGBM:
        lgb = lightgbm_module()
        n = len(y)
        p = dict(self.params)
        p.setdefault("min_child_samples", int(max(10, min(300, n // 150))))
        p.update(_thread_params())   # not saved: thread count never changes the model format
        objective = "binary" if self.n_classes == 2 else "multiclass"
        extra = {} if self.n_classes == 2 else {"num_class": self.n_classes}
        clf = lgb.LGBMClassifier(objective=objective, random_state=self.seed, deterministic=True,
                                 force_row_wise=True, verbose=-1, **extra, **p)
        if self.n_classes > 2 and len(np.unique(y)) < self.n_classes:
            # LightGBM infers classes from y; pad with zero-weight rows so every class index exists
            missing = sorted(set(range(self.n_classes)) - set(np.unique(y).tolist()))
            X = np.vstack([X, np.repeat(X[:1], len(missing), axis=0)])
            y = np.concatenate([y, missing])
            w = np.concatenate([np.ones(n) if w is None else w, np.full(len(missing), 1e-9)])
        clf.fit(np.asarray(X, np.float32), y, sample_weight=w)
        self.booster = clf.booster_
        self.n_features = int(X.shape[1])
        return self

    def raw(self, X: np.ndarray) -> np.ndarray:
        R = np.asarray(self.booster.predict(np.asarray(X, np.float32), raw_score=True, **_thread_params()), float)
        return R.reshape(-1) if self.n_classes == 2 else R.reshape(len(X), self.n_classes)

    def contributions(self, x: np.ndarray, class_index: int) -> np.ndarray | None:
        f = self.n_features
        c = np.asarray(self.booster.predict(np.atleast_2d(np.asarray(x, np.float32)), pred_contrib=True,
                                            **_thread_params()), float)
        if self.n_classes == 2 and c.size == f + 1:
            return c.reshape(f + 1)[:f] * (1.0 if class_index == 1 else -1.0)
        if c.size == self.n_classes * (f + 1):
            return c.reshape(self.n_classes, f + 1)[class_index, :f]
        return None

    def text(self) -> str:
        return self.booster.model_to_string()

    @classmethod
    def from_text(cls, n_classes: int, text: str, params: dict | None = None) -> LightGBM:
        lgb = lightgbm_module()
        m = cls(n_classes, params)
        m.booster = lgb.Booster(model_str=text)
        m.n_features = int(m.booster.num_feature())
        return m


def make_base(kind: str, n_classes: int, params: dict | None, seed: int):
    if kind == "lightgbm":
        return LightGBM(n_classes, params, seed)
    if kind == "logreg":
        return LogReg(n_classes, params, seed)
    raise ValueError(f"unknown base model {kind!r}")


def resolve_model(kind: str) -> tuple[str, str | None]:
    """(model actually used, note). `auto` = LightGBM when it can be imported, else logistic regression."""
    if kind == "auto":
        if lightgbm_available():
            return "lightgbm", None
        return "logreg", "model 'auto': LightGBM is not importable here (macOS: `brew install libomp`), so " \
                         "logistic regression was used"
    if kind == "lightgbm":
        lightgbm_module()  # raises a clear error
    return kind, None


# ----------------------------------------------------------------------------- the calibrated ensemble

class CalibratedEnsemble:
    def __init__(self, spec: HeadSpec, members: list[tuple[Any, Calibrator]], feature_names: list[str],
                 feature_version: str, nan_rate: list[float] | None = None):
        self.spec, self.members = spec, members
        self.feature_names, self.feature_version = list(feature_names), feature_version
        self.n_classes = len(spec.labels)
        # share of training rows in which each feature was missing (the input guard compares requests to it)
        self.nan_rate = None if nan_rate is None else [float(v) for v in nan_rate]

    def missing_inputs(self, X: np.ndarray, threshold: float = 0.05) -> list[str]:
        """Features that were present in at least 95% of training rows but are missing (NaN) in any row of X:
        the probabilities were calibrated on inputs that had them, so they may not hold for this request."""
        if self.nan_rate is None:
            return []
        X = np.atleast_2d(np.asarray(X, np.float32))
        rate = np.asarray(self.nan_rate)
        miss = np.isnan(X).any(axis=0) & (rate < threshold)
        return [self.feature_names[i] for i in np.flatnonzero(miss)]

    @property
    def kind(self) -> str:
        return self.members[0][0].kind

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Log-linear (geometric) pool of the calibrated members, renormalised.

        The plain average of calibrated forecasts is underconfident whenever the members disagree (a linear pool
        of calibrated forecasts is underdispersed; Ranjan & Gneiting, 2010). Measured on known generators (8
        seeds x 20,000 fresh rows): top-label ECE 0.074 -> 0.054 (8 classes, noisy features), 0.032 -> 0.021
        (binary), with equal or better log loss. With one member both pools are the same."""
        X = np.atleast_2d(np.asarray(X, np.float32))
        L = np.zeros((len(X), self.n_classes))
        for base, cal in self.members:
            L += np.log(np.clip(cal.predict(base.raw(X)), 1e-12, 1.0))
        L /= len(self.members)
        P = np.exp(L - L.max(axis=1, keepdims=True))
        return P / P.sum(axis=1, keepdims=True)

    def member_proba(self, X: np.ndarray, k: int) -> np.ndarray:
        base, cal = self.members[k]
        return cal.predict(base.raw(np.atleast_2d(np.asarray(X, np.float32))))

    def contributions(self, x: np.ndarray, class_index: int) -> np.ndarray | None:
        """Per-feature contribution to the raw score of one class, averaged over the members (log-odds of the
        uncalibrated base models: a ranking of what moved this prediction, not a probability)."""
        total, used = np.zeros(len(self.feature_names)), 0
        for base, _ in self.members:
            try:
                c = base.contributions(x, class_index)
            except Exception:
                c = None
            if c is not None:
                total += c
                used += 1
        return total / used if used else None

    def explain(self, x: np.ndarray, class_index: int, top: int = 6,
                groups: dict | None = None) -> tuple[list[dict], dict]:
        """(top feature attributions, share of absolute attribution per feature group)."""
        x = np.atleast_2d(np.asarray(x, np.float32))
        c = self.contributions(x, class_index)
        if c is None:
            return [], {}
        order = np.argsort(-np.abs(c))[:top]
        reasons = [{"feature": self.feature_names[i],
                    "value": None if np.isnan(x[0, i]) else round(float(x[0, i]), 4),
                    "contribution": round(float(c[i]), 4),
                    "effect": f"{'raises' if c[i] > 0 else 'lowers'} P({self.spec.labels[class_index]})"}
                   for i in order if abs(c[i]) > 1e-9]
        tot = float(np.abs(c).sum())
        shares: dict[str, float] = {}
        if tot > 0:
            for name, v in zip(self.feature_names, c, strict=True):
                g = (groups or {}).get(name) or name.split(":", 1)[0].split(".", 1)[0]
                shares[g] = shares.get(g, 0.0) + abs(float(v)) / tot
        return reasons, {g: round(v, 4) for g, v in sorted(shares.items(), key=lambda kv: -kv[1])}

    # ------------------------------------------------------------------ save / load (no pickle)
    def save(self, directory: Path) -> list[str]:
        """Write the members and calibrators into `directory`; returns the file names written."""
        from safetensors.numpy import save_file

        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        files, members = [], []
        for k, (base, cal) in enumerate(self.members):
            if base.kind == "logreg":
                name = f"member_{k}.safetensors"
                save_file({a: np.ascontiguousarray(v, dtype=np.float64) for a, v in base.arrays().items()},
                          str(directory / name))
            else:
                name = f"member_{k}.txt"
                (directory / name).write_text(base.text(), encoding="utf-8")
            files.append(name)
            members.append({"file": name, "kind": base.kind, "params": base.params,
                            "calibrator": cal.to_dict()})
        meta = {"spec": self.spec.to_dict(), "n_classes": self.n_classes, "feature_names": self.feature_names,
                "feature_version": self.feature_version, "members": members, "nan_rate": self.nan_rate}
        (directory / "ensemble.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
        return [*files, "ensemble.json"]

    @classmethod
    def load(cls, directory: Path) -> CalibratedEnsemble:
        from safetensors.numpy import load_file

        directory = Path(directory)
        meta = json.loads((directory / "ensemble.json").read_text(encoding="utf-8"))
        spec = HeadSpec.from_dict(meta["spec"])
        n = int(meta["n_classes"])
        members = []
        for m in meta["members"]:
            name = str(m["file"])
            if "/" in name or "\\" in name or name.startswith("."):
                raise ValueError(f"bad member file name {name!r}")
            path = directory / name
            if m["kind"] == "logreg":
                if not name.endswith(".safetensors"):
                    raise ValueError(f"logreg member must be .safetensors, got {name!r}")
                base = LogReg.from_arrays(n, load_file(str(path)), m.get("params"))
            elif m["kind"] == "lightgbm":
                if not name.endswith(".txt"):
                    raise ValueError(f"lightgbm member must be .txt, got {name!r}")
                base = LightGBM.from_text(n, path.read_text(encoding="utf-8"), m.get("params"))
            else:
                raise ValueError(f"unknown member kind {m['kind']!r}")
            members.append((base, Calibrator.from_dict(m["calibrator"])))
        return cls(spec, members, meta["feature_names"], meta["feature_version"], meta.get("nan_rate"))
