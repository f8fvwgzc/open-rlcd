"""Guarded weight edits (EXPERIMENTAL): an edit is a claim, and the holdout decides.

A team may change a trained model's weights directly: by hand, with a script, or by asking an LLM to do
"weight-space programming". open-rlcd never adopts such an edit on faith. The edit goes through the same gate as
training, and it serves only if it re-earns its verdict there.

  head edit     a safetensors file of ADDITIVE deltas to one head's logistic-regression arrays, named
                `member_<k>.<coef|intercept|mean|scale|fill>` (the arrays of `member_<k>.safetensors`). The base
                models stay frozen with the edit applied; every member's calibrator is refitted on its own fold's
                trusted calibration block (the block it was calibrated on in training); the edited ensemble is
                scored once on the base's untouched holdout by `training.finish_head`, i.e. with the very report,
                skill test, conformal check, policy verdict and Bernoulli table a trained head gets.
  encoder edit  a replacement ONNX text encoder (documents pack). A new sha256 is a new feature version, so no
                trained head is valid on it: every head of the base version is retrained (`Trainer.train_heads`)
                on the new features, with the base's head specs and split settings, and scored on the same
                holdout rows. Old heads are never reused on the new encoder.

Promotion. The candidate becomes rlcd-latest only if, for every head it changes,
  * a paired block-bootstrap test on the same holdout rows shows its Brier AND its log-loss gain over the base
    version reliably positive (one-sided lower bound > 0 at level alpha = 0.05 / attempt, see below), and
  * no verdict is weaker than the base's (per stakes level),
and the base is still rlcd-latest when the candidate is saved. Otherwise the candidate is kept as a non-latest
version with the reason, and one whose gate failed is never served (`Registry.servable`, `Service.resolve`).

The holdout judges; it never chooses. An edit chosen or tuned while looking at holdout rows turns the gate into a
fit, so:
  * an edit must say what it was chosen or tuned on (`tuned_on`: none | development | holdout) and may list the
    row ids it looked at (`tuned_on_ids`). `holdout`, or any listed id that is a holdout row of the base, is
    refused before anything is scored;
  * every scoring spends some of the holdout's credibility: the k-th candidate scored on the same holdout (same
    head, same row ids) is tested at alpha = 0.05 / k, and the attempt number is published with the result. Many
    attempts on one holdout call for fresh trusted labels (a new holdout), not a looser bound.

Before an edit is scored, the base's holdout is re-derived from today's rows with the base's split settings. It
must be the one the base was scored on (same row ids, the base's published Brier / log loss reproduce, and for a
head edit the same calibration blocks); otherwise the labelled rows changed since the base was trained and the
edit is refused: retrain instead, or edit the retrained version.

What an edit cannot do (docs/weight-edits.md): it can remove an obstacle (a spurious feature, a bad bias) but it
cannot add an ability the features do not carry; errors spread over many weights are poor targets; LightGBM
members are not editable here; an int8 ONNX encoder should be edited in fp32 and re-quantized.
"""
from __future__ import annotations

import json
import mmap
import os
import re
import shutil
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from . import calibration as C
from . import metrics as M
from . import splits as S
from . import training as T
from .bundle import LoadedHead
from .bundle import sha256 as file_sha256
from .dataset import Dataset
from .head import LogReg
from .policy import TIERS, PolicyConfig

ALPHA = 0.05                                   # one-sided level of the paired test for a holdout's first edit
TUNED_ON = ("none", "development", "holdout")
EDITABLE = ("coef", "intercept", "mean", "scale", "fill")
DELTA_KEY = re.compile(r"^member_(\d+)\.([a-z]+)$")
MAX_DELTA_BYTES = 256 << 20                    # 256 MiB: head arrays are small
MAX_ONNX_BYTES = 4 << 30                       # 4 GiB
REPRODUCE_TOL = 1e-5                           # published metrics are rounded to 6 decimals


class EditRefused(Exception):
    """The edit cannot be judged fairly, so it is refused before (or instead of) being scored."""


# ----------------------------------------------------------------------------- provenance

@dataclass(frozen=True)
class Provenance:
    author: str                                # "manual", "script:<name>", "LLM: <model>", ...
    tuned_on: str                              # none | development | holdout
    tuned_on_ids: tuple[str, ...] = ()
    description: str = ""

    def to_dict(self) -> dict:
        ids = {"n": len(self.tuned_on_ids), "sha256": T.ids_sha256(self.tuned_on_ids)} if self.tuned_on_ids else None
        return {"author": self.author, "tuned_on": self.tuned_on, "tuned_on_ids": ids,
                "description": self.description}


def _id_list(v: Any, where: str) -> list[str]:
    if v in (None, "", []):
        return []
    if isinstance(v, str):
        try:
            v = json.loads(v)
        except ValueError:
            raise EditRefused(f"{where}: tuned_on_ids must be a JSON list of row ids") from None
    if not isinstance(v, (list, tuple)):
        raise EditRefused(f"{where}: tuned_on_ids must be a list of row ids")
    return [str(i) for i in v]


def provenance(meta: dict | None = None, *, author: str | None = None, tuned_on: str | None = None,
               tuned_on_ids: list[str] | None = None, description: str | None = None) -> Provenance:
    """Merge what the edit file says about itself (safetensors metadata) with what the caller states. Both are
    claims; contradicting claims are refused rather than resolved."""
    meta = dict(meta or {})

    def pick(name: str, given: Any) -> Any:
        stated = meta.get(name)
        if given not in (None, "") and stated not in (None, "") and str(given) != str(stated):
            raise EditRefused(f"the edit file says {name} = {stated!r} but the command says {given!r}")
        return given if given not in (None, "") else stated

    who = str(pick("author", author) or "").strip()
    if not who:
        raise EditRefused("say who or what produced the edit (author: 'manual', 'script:<name>', 'LLM: <model>'): an "
                          "edit without an author cannot be reviewed")
    on = str(pick("tuned_on", tuned_on) or "").strip().lower()
    if on not in TUNED_ON:
        raise EditRefused(f"say what the edit was chosen or tuned on: tuned_on must be one of {list(TUNED_ON)} "
                          f"(got {on or 'nothing'!r})")
    ids = list(dict.fromkeys([*_id_list(meta.get("tuned_on_ids"), "edit file"),
                              *_id_list(tuned_on_ids, "command")]))
    if on == "none" and ids:
        raise EditRefused(f"tuned_on = 'none' but the edit lists {len(ids)} row ids it was tuned on")
    desc = str(description if description not in (None, "") else meta.get("description") or "")
    return Provenance(author=who, tuned_on=on, tuned_on_ids=tuple(ids), description=desc)


def refuse_holdout_tuning(prov: Provenance, holdout_ids: set[str], where: str) -> None:
    """The holdout decides; it must never have chosen. Refuse an edit that says it was tuned on holdout rows."""
    if prov.tuned_on == "holdout":
        raise EditRefused(f"the edit says it was tuned on the holdout: the holdout of {where} can only judge edits "
                          f"chosen without it. Tune on development rows (or none) and resubmit")
    hit = sorted(holdout_ids.intersection(prov.tuned_on_ids))
    if hit:
        raise EditRefused(f"{len(hit)} of the {len(prov.tuned_on_ids)} rows the edit was tuned on are holdout rows of "
                          f"{where} (e.g. {hit[:3]}): an edit chosen on the holdout cannot be judged by it")


# ----------------------------------------------------------------------------- head deltas

def read_delta(path: Path) -> tuple[dict[str, np.ndarray], dict[str, str], str]:
    """(arrays, metadata, sha256) of a safetensors delta file. safetensors is plain tensors + a JSON header:
    reading it never executes code."""
    from safetensors import safe_open

    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise EditRefused(f"{path} is not a regular file")
    if path.suffix != ".safetensors":
        raise EditRefused(f"a head edit is a .safetensors file of deltas, got {path.name!r}")
    if path.stat().st_size > MAX_DELTA_BYTES:
        raise EditRefused(f"{path.name} is larger than {MAX_DELTA_BYTES >> 20} MiB: not a head delta")
    try:
        with safe_open(str(path), framework="np") as f:
            meta = dict(f.metadata() or {})
            arrays = {k: np.asarray(f.get_tensor(k)) for k in f.keys()}  # noqa: SIM118 (safe_open has no __iter__)
    except Exception as e:  # the parser raises several types on a malformed file
        raise EditRefused(f"{path.name} is not a readable safetensors file ({type(e).__name__})") from None
    if not arrays:
        raise EditRefused(f"{path.name} holds no arrays")
    return arrays, meta, file_sha256(path)


def apply_delta(ensemble: Any, delta: dict[str, np.ndarray]) -> tuple[list[LogReg], int, int, dict[str, int]]:
    """Edited copies of an ensemble's base models (their calibrators are refitted afterwards). Returns (bases,
    parameters touched, parameters in all members, touched per delta array). The base ensemble is not modified."""
    kinds = {base.kind for base, _ in ensemble.members}
    if kinds != {"logreg"}:
        raise EditRefused(f"head edits apply to logistic-regression members (safetensors arrays); this head has "
                          f"{sorted(kinds)} members. LightGBM trees are not editable here: retrain instead")
    bases = [LogReg.from_arrays(ensemble.n_classes, {a: np.array(v, dtype=float, copy=True)
                                                     for a, v in base.arrays().items()}, base.params)
             for base, _ in ensemble.members]
    total = sum(int(np.size(getattr(b, a))) for b in bases for a in EDITABLE)
    touched, where = 0, {}
    for key in sorted(delta):
        m = DELTA_KEY.match(key)
        if not m or m.group(2) not in EDITABLE:
            raise EditRefused(f"unknown array {key!r} in the edit: name each delta member_<k>.<{'|'.join(EDITABLE)}>")
        k, arr = int(m.group(1)), m.group(2)
        if k >= len(bases):
            raise EditRefused(f"{key}: the head has {len(bases)} members (member_0 .. member_{len(bases) - 1})")
        cur = getattr(bases[k], arr)
        d = np.asarray(delta[key])
        if not np.issubdtype(d.dtype, np.floating) and not np.issubdtype(d.dtype, np.integer):
            raise EditRefused(f"{key}: deltas must be numbers, got dtype {d.dtype}")
        d = d.astype(float)
        if d.shape != np.shape(cur):
            raise EditRefused(f"{key}: delta shape {d.shape} does not match the array's {np.shape(cur)}")
        if not np.isfinite(d).all():
            raise EditRefused(f"{key}: the delta holds NaN or infinite values")
        new = cur + d
        if arr == "scale" and (new <= 1e-12).any():
            raise EditRefused(f"{key}: a feature scale must stay positive")
        setattr(bases[k], arr, new)
        where[key] = int(np.count_nonzero(d))
        touched += where[key]
    if touched == 0:
        raise EditRefused("the edit changes no parameter (every delta is zero)")
    return bases, touched, total, where


# ----------------------------------------------------------------------------- the base's holdout, re-derived

@dataclass
class Baseline:
    """A base head's holdout, re-derived from today's rows and checked against what the base published."""
    plan: T.HeadPlan
    ids: np.ndarray            # holdout row ids, scoring order
    y: np.ndarray
    P: np.ndarray              # the base head's forecasts on those rows
    blocks: np.ndarray | int   # block keys of the paired test (same as the base's skill test)
    holdout_sha256: str


def split_config(head: LoadedHead, cfg: T.TrainConfig, policy: PolicyConfig) -> T.TrainConfig:
    """The settings the base head was split, calibrated and judged with (its report records them; the project's
    current settings fill what an older report lacks), under the base version's policy."""
    sp = head.report.get("split") or {}
    method = (head.summary.get("calibration") or {}).get("method") or cfg.calibration
    return replace(cfg, holdout_fraction=float(sp.get("holdout_fraction", cfg.holdout_fraction)),
                   ordering=str(sp.get("ordering") or cfg.ordering), seed=int(sp.get("seed", cfg.seed)),
                   n_splits=int(sp.get("n_splits", cfg.n_splits)), gap=int(sp.get("gap", cfg.gap)),
                   calibration=str(method), policy=policy)


def reproduce(head: LoadedHead, ds: Dataset, cfg: T.TrainConfig) -> Baseline:
    """Re-derive the base head's untouched holdout and check it is the one the base was scored on."""
    name = head.name
    if ds.feature_version != head.ensemble.feature_version:
        raise EditRefused(f"{name}: today's rows have feature version {ds.feature_version!r}, the base head was "
                          f"trained on {head.ensemble.feature_version!r}")
    try:
        T.check_rows(head.spec, ds, cfg)
        plan = T.plan_head(head.spec, ds, cfg)
    except T.TrainError as e:
        raise EditRefused(f"cannot re-derive the base's holdout: {e}") from None
    ids = np.asarray([str(i) for i in ds.ids[plan.h_idx]], dtype=object)
    if len(set(ids.tolist())) != len(ids):
        raise EditRefused(f"{name}: holdout row ids are not unique, so the base and the edit cannot be paired row "
                          f"by row")
    sha = plan.holdout_ids_sha256(ds)
    rep = head.report
    changed = (f"the labelled rows changed since {name} was trained, so its holdout cannot be re-derived: retrain "
               f"(`rlcd train`) and edit the retrained version")
    recorded = (rep.get("split") or {}).get("holdout_ids_sha256")
    if recorded and recorded != sha:
        raise EditRefused(f"{name}: the holdout re-derived from today's rows is not the one the base was scored on; "
                          f"{changed}")
    y = ds.y[plan.h_idx]
    P = head.ensemble.predict_proba(ds.X[plan.h_idx])
    Y = np.eye(P.shape[1])[y]
    got = {"n_holdout": len(ids), "brier": float(M.brier_rows(P, Y).mean()),
           "log_loss": float(M.log_loss_rows(P, Y).mean())}
    if rep.get("n_holdout") is not None and int(rep["n_holdout"]) != got["n_holdout"]:
        raise EditRefused(f"{name}: {got['n_holdout']} holdout rows today, the base was scored on "
                          f"{rep['n_holdout']}; {changed}")
    for key in ("brier", "log_loss"):
        if rep.get(key) is not None and abs(float(rep[key]) - got[key]) > REPRODUCE_TOL:
            raise EditRefused(f"{name}: the base's published holdout {key} ({rep[key]}) does not come out again on "
                              f"today's rows ({got[key]:.6f}); {changed}")
    return Baseline(plan=plan, ids=ids, y=y, P=P, blocks=S.skill_blocks(ds, plan.h_idx, plan.mode),
                    holdout_sha256=sha)


# ----------------------------------------------------------------------------- the gate

def _tier(v: dict | None, abstain: str) -> int:
    v = v or {}
    if isinstance(v.get("tier"), int):
        return int(v["tier"])
    name = v.get("max_tier")
    return 0 if name in (None, abstain) else (TIERS.index(name) if name in TIERS else 0)


def judge(base_P: np.ndarray, cand_P: np.ndarray, y: np.ndarray, blocks: np.ndarray | int, base_verdict: dict,
          cand_verdict: dict, policy: PolicyConfig, attempt: int = 1, seed: int = 7) -> dict:
    """The promotion gate for one head: paired holdout gain reliably positive AND no verdict weaker."""
    attempt = max(1, int(attempt))
    alpha = ALPHA / attempt
    test = M.paired_gain_test(base_P, cand_P, y, blocks, alpha=alpha, seed=seed)
    weaker = [f"{s}: {bv.get('max_tier')} -> {(cand_verdict.get(s) or {}).get('max_tier')}"
              for s, bv in (base_verdict or {}).items()
              if _tier(cand_verdict.get(s), policy.abstain) < _tier(bv, policy.abstain)]
    reasons = []
    if not test["passed"]:
        b, ll = test.get("brier") or {}, test.get("log_loss") or {}
        reasons.append(f"the paired holdout gain over the base is not reliably positive (Brier {b.get('mean_gain')}, "
                       f"lower bound {b.get('lower')}; log loss {ll.get('mean_gain')}, lower bound {ll.get('lower')}; "
                       f"one-sided {1 - alpha:.4g}" + (f", attempt {attempt} on this holdout" if attempt > 1 else "")
                       + ")")
    if weaker:
        reasons.append("verdicts weaker than the base's (" + "; ".join(weaker) + ")")
    return {"paired_test": test, "attempt": attempt, "alpha": M._f(alpha),
            "verdict_base": {s: v.get("max_tier") for s, v in (base_verdict or {}).items()},
            "verdict_candidate": {s: v.get("max_tier") for s, v in (cand_verdict or {}).items()},
            "weaker_verdicts": weaker, "passed": not reasons, "reasons": reasons}


@dataclass
class HeadEdit:
    trained: T.TrainedHead
    judgement: dict
    holdout_sha256: str
    parameters_touched: int
    parameters_total: int


def patch_head(head: LoadedHead, ds: Dataset, delta: dict[str, np.ndarray], prov: Provenance, cfg: T.TrainConfig,
               *, edit_sha256: str, base_version: str,
               attempts: Callable[[str, str], int] | None = None) -> HeadEdit:
    """Apply a delta to one head, recalibrate every member on its trusted calibration block and score the edited
    head once on the base's untouched holdout. `cfg` is the base's `split_config`; `attempts(head, holdout_sha)`
    counts earlier candidates judged on that holdout (the registry's). Nothing is saved here."""
    spec = head.spec
    base = reproduce(head, ds, cfg)
    refuse_holdout_tuning(prov, set(base.ids.tolist()), f"{base_version}:{spec.name}")
    plan = base.plan
    ens = head.ensemble
    if len(plan.folds) != len(ens.members):
        raise EditRefused(f"{spec.name}: {len(plan.folds)} calibration folds today, the base has {len(ens.members)} "
                          f"members: the rows changed since it was trained; retrain instead")
    rec = (head.report.get("split") or {}).get("calibration_ids_sha256")
    if rec and rec != plan.calibration_ids_sha256(ds):
        raise EditRefused(f"{spec.name}: the calibration blocks re-derived today are not the ones the base members "
                          f"were calibrated on: the rows changed since it was trained; retrain instead")
    bases, touched, total, _ = apply_delta(ens, delta)
    k = len(spec.labels)
    members, oof_P = [], []
    for (_, ca), b in zip(plan.folds, bases, strict=True):
        raw = b.raw(ds.X[ca])
        members.append((b, C.Calibrator(plan.method, k).fit(raw, ds.y[ca])))   # trusted calibration block only
        oof_P.append(T.cross_fit_scores(plan.method, k, raw, ds.y[ca]))
    notes = [*ds.notes, *plan.notes,
             f"weight edit {edit_sha256[:12]} of {base_version} by {prov.author}: {touched} of {total} parameters "
             f"touched; every member recalibrated on its trusted calibration block, scored once on the untouched "
             f"holdout"]
    th = T.finish_head(spec, ds, cfg, plan, members, oof_P, ens.kind, notes)
    attempt = (attempts(spec.name, base.holdout_sha256) if attempts else 0) + 1
    j = judge(base.P, th.holdout_P, base.y, base.blocks, head.verdict, th.verdict, cfg.policy, attempt, cfg.seed)
    th.edit = {"experimental": True, "kind": "head", "base_version": base_version, "sha256": edit_sha256,
               "provenance": prov.to_dict(), "parameters_touched": touched, "parameters_total": total,
               "holdout_ids_sha256": base.holdout_sha256, **j}
    return HeadEdit(th, j, base.holdout_sha256, touched, total)


# ----------------------------------------------------------------------------- ONNX initializers (no onnx package)

_ONNX_ITEMSIZE = {1: 4, 2: 1, 3: 1, 4: 2, 5: 2, 6: 4, 7: 8, 9: 1, 10: 2, 11: 8, 12: 4, 13: 8, 16: 2}


def _varint(buf: memoryview, i: int) -> tuple[int, int]:
    out = shift = 0
    while True:
        if i >= len(buf):
            raise ValueError("truncated varint")
        b = buf[i]
        i += 1
        out |= (b & 0x7F) << shift
        if b < 0x80:
            return out, i
        shift += 7
        if shift > 63:
            raise ValueError("varint too long")


def _fields(buf: memoryview):
    """(field number, wire type, value) of one protobuf message: an int for varints, a memoryview otherwise."""
    i = 0
    while i < len(buf):
        key, i = _varint(buf, i)
        num, wt = key >> 3, key & 7
        if wt == 0:
            v, i = _varint(buf, i)
        elif wt == 1:
            v, i = buf[i:i + 8], i + 8
        elif wt == 2:
            n, i = _varint(buf, i)
            v, i = buf[i:i + n], i + n
        elif wt == 5:
            v, i = buf[i:i + 4], i + 4
        else:
            raise ValueError(f"unsupported wire type {wt}")
        if i > len(buf):
            raise ValueError("truncated field")
        yield num, wt, v


def _tensor(buf: memoryview) -> tuple[str, tuple[int, ...], np.ndarray | None]:
    """(name, dims, values) of an ONNX TensorProto: values as one row of bytes per element (None: external data)."""
    name, dims, dtype, raw, typed, external = "", [], 0, None, [], False
    for num, wt, v in _fields(buf):
        if num == 1:
            if wt == 0:
                dims.append(int(v))
            else:
                j = 0
                while j < len(v):
                    d, j = _varint(v, j)
                    dims.append(d)
        elif num == 2:
            dtype = int(v)
        elif num == 8:
            name = bytes(v).decode("utf-8", "replace")
        elif num == 9:
            raw = bytes(v)
        elif num in (4, 10) and wt == 2:          # packed float / double
            typed.append(np.frombuffer(bytes(v), "<f4" if num == 4 else "<f8").view(np.uint8))
        elif num in (5, 7, 11):                   # int32 / int64 / uint64 (varints)
            vals = []
            if wt == 0:
                vals.append(int(v))
            else:
                j = 0
                while j < len(v):
                    d, j = _varint(v, j)
                    vals.append(d)
            typed.append(np.asarray(vals, dtype=np.uint64).view(np.uint8))
        elif num == 14 and int(v) == 1:
            external = True
    if external:
        return name, tuple(dims), None
    data = np.frombuffer(raw, np.uint8) if raw is not None else (np.concatenate(typed) if typed else
                                                                 np.zeros(0, np.uint8))
    size = int(np.prod(dims)) if dims else 1
    if size == 0:
        return name, tuple(dims), np.zeros((0, 1), np.uint8)
    width = len(data) // size if len(data) % size == 0 else _ONNX_ITEMSIZE.get(dtype, 1)
    if width == 0 or len(data) < size * width:
        raise ValueError(f"initializer {name!r}: {len(data)} bytes for {size} elements")
    return name, tuple(dims), data[: size * width].reshape(size, width)


def onnx_initializers(path: Path) -> dict[str, tuple[tuple[int, ...], np.ndarray | None]]:
    """The weights (graph initializers) of an ONNX file, read with a minimal protobuf reader (no code runs and no
    `onnx` package is needed): name -> (dims, one row of bytes per element, or None for external data)."""
    with open(path, "rb") as f:
        # mapped, not read: an encoder can be hundreds of MB. Values are copied out, and the mapping is released
        # when the last slice of it goes (closing it while the parser's slices exist would raise)
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    out = {}
    for num, wt, v in _fields(memoryview(mm)):
        if num == 7 and wt == 2:                  # ModelProto.graph
            for gnum, gwt, gv in _fields(v):
                if gnum == 5 and gwt == 2:        # GraphProto.initializer
                    name, dims, data = _tensor(gv)
                    out[name] = (dims, None if data is None else data.copy())
    return out


def count_touched(base: Path, edited: Path) -> tuple[int | None, int | None, str | None]:
    """(parameters touched, parameters in the edited model, note): elements of the graph initializers that differ
    between two ONNX files of the same architecture. None when it cannot be counted (external data, a parse
    error), with the reason."""
    try:
        a, b = onnx_initializers(base), onnx_initializers(edited)
    except (OSError, ValueError) as e:
        return None, None, f"could not read the ONNX initializers to count the parameters touched ({e})"
    if not b:
        return None, None, "the edited ONNX file has no graph initializers to count"
    touched = total = 0
    for name, (dims, data) in b.items():
        if data is None or (name in a and a[name][1] is None):
            return None, None, "the encoder keeps its weights in external data files: parameters touched not counted"
        total += len(data)
        old = a.get(name)
        if old is None or old[0] != dims or old[1].shape != data.shape:
            touched += len(data)
        else:
            touched += int((old[1] != data).any(axis=1).sum())
    return touched, total, None


# ----------------------------------------------------------------------------- orchestration (service level)

def _base(svc: Any, base: str | None) -> tuple[str, Any]:
    from .registry import UnknownModel

    try:
        version = svc.registry.resolve(base)
    except UnknownModel:
        raise EditRefused(f"unknown base version {base!r}; available: {svc.registry.available()}") from None
    if not svc.registry.servable(version):
        raise EditRefused(f"{version} is an edit candidate that failed its gate: edit a version that serves")
    b = svc.bundle(version)
    if b is None:
        raise EditRefused("no trained model to edit yet: run `rlcd train` first")
    return version, b


def _settings(svc: Any) -> tuple[T.TrainConfig, float]:
    from .contract import TrainRequest

    req = TrainRequest()
    return svc.trainer.config(req), svc.trainer.feedback_weight(req)


def _data_card(b: Any, trained: dict[str, T.TrainedHead], edit: dict, provider: Any = None) -> dict:
    card = dict(b.meta.get("data_card") or {})
    extra = getattr(provider, "data_card", None)
    if callable(extra):   # an encoder edit: the new encoder and feature version
        card.update(extra())
    card["heads"] = {**(card.get("heads") or {}), **{n: th.data for n, th in trained.items()}}
    card["weight_edit"] = {"kind": edit["kind"], "base_version": edit["base_version"], "sha256": edit["sha256"]}
    return card


def edit_head(svc: Any, *, head: str, delta_path: Path, base: str | None = None, author: str | None = None,
              tuned_on: str | None = None, tuned_on_ids: list[str] | None = None,
              description: str | None = None) -> dict:
    """`rlcd patch --delta`: judge a head edit against its base version and save it as a candidate version."""
    version, b = _base(svc, base)
    lh = b.head(head)
    if lh is None:
        raise EditRefused(f"{version} has no head {head!r}; heads: {b.names()}")
    delta, meta, sha = read_delta(Path(delta_path))
    if meta.get("head") not in (None, "", head):
        raise EditRefused(f"the edit file is for head {meta['head']!r}, not {head!r}")
    prov = provenance(meta, author=author, tuned_on=tuned_on, tuned_on_ids=tuned_on_ids, description=description)
    tcfg, fw = _settings(svc)
    cfg = split_config(lh, tcfg, svc.policy(version))
    pack, provider = svc.pack_for(version), svc.provider(version)
    ds = pack.datasets(svc.project, provider, [head], svc.logs.feedback(), fw).get(head)
    if ds is None or len(ds) == 0:
        raise EditRefused(f"{head}: no labelled rows today, so the edit cannot be recalibrated or scored")
    he = patch_head(lh, ds, delta, prov, cfg, edit_sha256=sha, base_version=version,
                    attempts=svc.registry.edit_attempts)
    j = he.judgement
    edit = {"experimental": True, "kind": "head", "base_version": version, "sha256": sha,
            "file": Path(delta_path).name, "provenance": prov.to_dict(), "author": prov.author,
            "parameters_touched": he.parameters_touched, "parameters_total": he.parameters_total,
            "holdouts": {head: he.holdout_sha256}, "heads": {head: j}, "passed": j["passed"],
            "reasons": [f"{head}: {r}" for r in j["reasons"]]}
    res = svc.registry.commit_edit(version, {head: he.trained}, edit=edit, data_card=_data_card(b, {head: he.trained},
                                                                                                  edit),
                                   notes=[f"weight edit of {head} ({sha[:12]}) on {version}"])
    svc._providers.pop(res["version"], None)
    return res


def edit_encoder(svc: Any, *, onnx_path: Path, base: str | None = None, author: str | None = None,
                 tuned_on: str | None = None, tuned_on_ids: list[str] | None = None,
                 description: str | None = None) -> dict:
    """`rlcd patch --encoder`: a replacement ONNX encoder is a new feature version, so every head of the base is
    retrained on it (never reused) and each is judged against its base head on the same holdout rows."""
    version, b = _base(svc, base)
    pconf = (b.meta.get("pack") or {}).get("config")
    enc = pconf.get("encoder") if isinstance(pconf, dict) else None
    if not isinstance(enc, dict) or str(enc.get("kind")) != "onnx":
        kind = enc.get("kind") if isinstance(enc, dict) else None
        raise EditRefused(f"an encoder edit replaces an ONNX text encoder; {version} has "
                          f"{'a ' + repr(kind) + ' encoder' if kind else 'no encoder'}. Head edits (--delta) work "
                          f"on any logistic-regression head")
    pack = svc.pack_for(version)
    if getattr(pack, "name", None) != getattr(svc.pack, "name", None):
        raise EditRefused(f"{version} was trained with pack {getattr(pack, 'name', '?')!r}; an encoder edit retrains "
                          f"with this project's pack {getattr(svc.pack, 'name', '?')!r}")
    src = Path(onnx_path)
    if src.is_symlink() or not src.is_file() or src.suffix.lower() != ".onnx":
        raise EditRefused(f"{src} is not a regular .onnx file")
    if src.stat().st_size > MAX_ONNX_BYTES:
        raise EditRefused(f"{src.name} is larger than {MAX_ONNX_BYTES >> 30} GiB")
    sha = file_sha256(src)
    if sha == enc.get("sha256"):
        raise EditRefused("the replacement encoder is byte-identical to the base's: nothing to judge")
    prov = provenance({}, author=author, tuned_on=tuned_on, tuned_on_ids=tuned_on_ids, description=description)

    # judge first what can be judged without training: the base's holdouts must re-derive, and no tuned-on row
    # may be a holdout row
    tcfg, fw = _settings(svc)
    policy = svc.policy(version)
    names = b.names()
    base_provider = svc.provider(version)
    feedback = svc.logs.feedback()
    base_ds = pack.datasets(svc.project, base_provider, names, feedback, fw)
    cfgs = {n: split_config(b.head(n), tcfg, policy) for n in names}
    baselines = {}
    for n in names:
        if base_ds.get(n) is None:
            raise EditRefused(f"{n}: no labelled rows today, so it cannot be retrained or judged")
        baselines[n] = reproduce(b.head(n), base_ds[n], cfgs[n])
        refuse_holdout_tuning(prov, set(baselines[n].ids.tolist()), f"{version}:{n}")

    # register the edited file inside the home (bundles hold no ONNX; the bundle's encoder config points here)
    rel = f"encoders/edited/{sha}.onnx"
    dest = svc.home / rel
    if not dest.is_file() or file_sha256(dest) != sha:
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(".tmp")
        shutil.copyfile(src, tmp, follow_symlinks=False)
        if file_sha256(tmp) != sha:
            tmp.unlink(missing_ok=True)
            raise EditRefused(f"{src.name} changed while it was being copied")
        os.replace(tmp, dest)
    new_cfg = {k: v for k, v in pconf.items() if k != "views"}
    new_cfg["encoder"] = {**enc, "path": rel, "sha256": sha}
    try:
        provider = pack.provider(svc.project, new_cfg)
    except Exception as e:
        raise EditRefused(f"the replacement encoder cannot be loaded: {getattr(e, 'message', None) or e}") from None
    if provider.feature_version == b.meta.get("feature_version"):
        raise EditRefused("the replacement encoder gives the same feature version as the base's")

    trained, status, cand_ds = svc.trainer.train_heads(provider, names, tcfg, fw,
                                                       specs={n: b.head(n).spec for n in names}, configs=cfgs)
    heads, holdouts, reasons = {}, {}, []
    for n in names:
        base = baselines[n]
        holdouts[n] = base.holdout_sha256
        th = trained.get(n)
        if th is None:
            heads[n] = {"passed": False, "reasons": [f"retraining failed: {status[n].get('error')}"]}
        else:
            cids = [str(i) for i in cand_ds[n].ids[th.holdout_idx]]
            pos = {i: r for r, i in enumerate(cids)}
            if len(pos) != len(cids) or set(pos) != set(base.ids.tolist()):
                heads[n] = {"passed": False, "reasons": ["the retrained head was scored on other holdout rows than "
                                                         "the base: the two cannot be paired"]}
            else:
                order = np.asarray([pos[i] for i in base.ids.tolist()])
                if not np.array_equal(cand_ds[n].y[th.holdout_idx][order], base.y):
                    heads[n] = {"passed": False, "reasons": ["holdout labels differ between the base rows and today's"]}
                else:
                    attempt = svc.registry.edit_attempts(n, base.holdout_sha256) + 1
                    heads[n] = judge(base.P, th.holdout_P[order], base.y, base.blocks, b.head(n).verdict, th.verdict,
                                     policy, attempt, cfgs[n].seed)
            th.edit = {"experimental": True, "kind": "encoder", "base_version": version, "sha256": sha,
                       "provenance": prov.to_dict(), "holdout_ids_sha256": base.holdout_sha256,
                       "note": "retrained from scratch on the edited encoder's features (a new feature version: "
                               "no head trained on the old encoder is valid on it)", **heads[n]}
        reasons += [f"{n}: {r}" for r in heads[n]["reasons"]]
    if not trained:
        raise EditRefused("no head could be retrained on the edited encoder: " + "; ".join(reasons))

    base_path = getattr(_inner(base_provider.encoder), "model_path", None)
    touched, total, why = count_touched(Path(base_path), dest) if base_path else (None, None, "base encoder file "
                                                                                    "not found")
    notes = ["every head was retrained on the edited encoder (new feature version); none was reused"]
    if why:
        notes.append(why)
    if "int8" in str(enc.get("path") or enc.get("file") or "") or (enc.get("quantize_locally") and not enc.get("path")):
        notes.append("the base encoder is int8-quantized: edit the fp32 weights and re-quantize, rather than editing "
                     "int8 values in place (docs/weight-edits.md)")
    notes.append("the local corpus (if any) was embedded with the base encoder: re-ingest, or set "
                 "pack_config.on_encoder_change: reembed, before /v1/decide retrieves with this version")
    edit = {"experimental": True, "kind": "encoder", "base_version": version, "sha256": sha, "file": src.name,
            "encoder_path": rel, "base_encoder_sha256": enc.get("sha256"), "provenance": prov.to_dict(),
            "author": prov.author, "parameters_touched": touched, "parameters_total": total,
            "feature_version": provider.feature_version, "holdouts": holdouts, "heads": heads,
            "retrained": sorted(trained), "passed": not reasons and len(trained) == len(names),
            "reasons": reasons, "notes": notes}
    pinfo = pack.info(provider)
    pc = pinfo.get("config") if isinstance(pinfo, dict) else None
    res = svc.registry.commit_edit(version, trained, edit=edit, carry=False, pack=pinfo,
                                   feature_version=provider.feature_version,
                                   encoder=pc.get("encoder") if isinstance(pc, dict) else None,
                                   data_card=_data_card(b, trained, edit, provider),
                                   notes=[f"encoder edit {sha[:12]} on {version}", *notes])
    svc._providers.pop(res["version"], None)
    return res


def _inner(enc: Any) -> Any:
    while hasattr(enc, "inner"):
        enc = enc.inner
    return enc
