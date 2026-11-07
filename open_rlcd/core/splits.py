"""How rows are kept apart: the untouched holdout and the calibration folds.

Three orderings:

  * time   rows carry timestamps. The holdout is the latest `fraction` of the trusted rows (one cut-off instant
           for everyone); development rows whose label window (`t_end`) reaches the cut are purged. Calibration
           folds are expanding windows: fit on the past, calibrate on the block that follows, a training row
           kept only if its label window closed before the block starts (plus an optional `gap`).
  * group  rows carry a group key (document, customer, ticket). Groups never straddle train / calibration /
           holdout, so near-duplicates cannot leak. Groups are ordered by their first timestamp when rows have
           one (the holdout is then the latest groups), otherwise by a seeded hash (a reproducible shuffle).
           Calibration folds are group k-fold over the development groups.
  * row    neither: every row is its own group (hashed order). Only correct when rows really are independent;
           the report says so.

Rows that are not trusted (judge labels, feedback) are never scored in the holdout and never fitted by a
calibrator: they only ever sit on the training side of a fold. Rows without a timestamp (t < 0) in time mode
are training-side only, and so are feedback rows that fall in the holdout window: they train every fold (they are
later than every calibration block, a deliberate exception to "train on the past") and the holdout rows whose
label window overlaps theirs are removed. A calibration block holding a single class is dropped: a calibrator
fitted on it would only learn "the other class never happens".
"""
from __future__ import annotations

import zlib
from dataclasses import dataclass, field

import numpy as np

from .dataset import Dataset

MODES = ("auto", "time", "group", "row")


class NotEnoughData(Exception):
    pass


@dataclass
class Split:
    mode: str
    dev: np.ndarray        # bool: rows usable for development (training and calibration)
    holdout: np.ndarray    # bool: trusted rows scored once, never fitted on
    purged: int
    cut: dict
    notes: list[str] = field(default_factory=list)


def resolve_mode(ds: Dataset, mode: str = "auto") -> str:
    if mode not in MODES:
        raise ValueError(f"ordering must be one of {MODES}, got {mode!r}")
    if mode == "auto":
        if ds.group is not None:
            return "group"
        return "time" if ds.t is not None else "row"
    if mode == "time" and ds.t is None:
        raise ValueError("ordering 'time' needs timestamps (t)")
    if mode == "group" and ds.group is None:
        raise ValueError("ordering 'group' needs a group key")
    return mode


def _hash(seed: int, key: str) -> int:
    return zlib.crc32(f"{seed}|{key}".encode())


def group_keys(ds: Dataset, mode: str) -> np.ndarray:
    """One key per row: the group for group mode, the row id for row mode."""
    if mode == "group":
        return ds.group
    return np.asarray([f"row:{i}" for i in ds.ids], dtype=object)


def group_order(keys: np.ndarray, t: np.ndarray | None, seed: int) -> dict[str, tuple]:
    """Rank of each group: by first timestamp when available (hash breaks ties), else by seeded hash."""
    first: dict[str, int] = {}
    if t is not None:
        for g, ti in zip(keys.tolist(), t.tolist(), strict=True):
            if ti >= 0 and (g not in first or ti < first[g]):
                first[g] = ti
    return {g: (first.get(g, -1) if t is not None else 0, _hash(seed, g)) for g in set(keys.tolist())}


def holdout_split(ds: Dataset, fraction: float = 0.2, mode: str = "auto", seed: int = 7) -> Split:
    mode = resolve_mode(ds, mode)
    # rows a holdout may score: trusted, and sampled representatively (rows an active-learning queue picked for
    # being hard are trusted for training and calibration, but would make the holdout unrepresentative)
    trusted = ds.trusted & ds.holdout_ok
    biased = ds.trusted & ~ds.holdout_ok
    notes: list[str] = []
    if biased.any():
        notes.append(f"{int(biased.sum())} trusted rows were chosen by active learning: they train and calibrate but "
                     f"are never scored in the holdout (they are not a representative sample)")
    if mode == "time":
        timed = ds.t >= 0
        tt = np.sort(ds.t[trusted & timed])
        if len(tt) < 2:
            raise NotEnoughData(f"{len(tt)} trusted rows with timestamps: cannot place a time holdout")
        cut = int(tt[int(np.floor((1 - fraction) * (len(tt) - 1)))])
        if cut <= int(tt[0]):
            cut = int(tt[0]) + 1
        later = timed & (ds.t >= cut)
        dev = (~timed) | ((ds.t < cut) & (ds.t_end <= cut))
        holdout = later & trusted
        # feedback inside the holdout window trains the model, so the holdout rows its label window overlaps
        # are removed: the holdout stays untouched by training
        fb_late = later & (ds.source == "feedback")
        if fb_late.any():
            dev |= fb_late
            for i in np.flatnonzero(fb_late):
                overlap = holdout & (ds.t <= ds.t_end[i]) & (ds.t_end >= ds.t[i])
                if ds.group is not None:
                    overlap &= ds.group == ds.group[i]
                holdout &= ~overlap
            notes.append(f"{int(fb_late.sum())} feedback rows fall in the holdout window: they train the model "
                         f"and the holdout rows they overlap were removed")
        purged = int(((~dev) & (~later)).sum())
        dropped = int((later & ~trusted & ~fb_late).sum())
        if dropped:
            notes.append(f"{dropped} untrusted rows in the holdout window were dropped (not scored, not trained; "
                         f"actively sampled human rows are counted here too)")
        return Split(mode, dev, holdout, purged, {"time": cut}, notes)

    keys = group_keys(ds, mode)
    rank = group_order(keys, ds.t if mode == "group" else None, seed)
    ordered = sorted(rank, key=lambda g: rank[g])
    # walk from the latest group back until the holdout holds `fraction` of the trusted rows
    trusted_per: dict[str, int] = {}
    dev_only: set[str] = set(keys[biased].tolist())   # groups holding actively sampled rows stay in development
    for g, tr in zip(keys.tolist(), trusted.tolist(), strict=True):
        trusted_per[g] = trusted_per.get(g, 0) + int(tr)
    total = sum(trusted_per.values())
    if total < 2:
        raise NotEnoughData(f"{total} trusted rows: cannot place a holdout")
    want, got, hold_groups = fraction * total, 0, set()
    # Without a time order, a group with no trusted row has nothing to score and no "future" to leak, so it stays
    # in development rather than being swallowed (and dropped) by the walk. With a time order it is dropped:
    # training on it would mean training on the holdout period. Either way the holdout's trusted rows are the same.
    time_ordered = mode == "group" and ds.t is not None
    for g in reversed(ordered):
        if got >= want:
            break
        if g in dev_only or (not trusted_per[g] and not time_ordered):
            continue
        hold_groups.add(g)
        got += trusted_per[g]
    in_hold = np.isin(keys, list(hold_groups))
    fb = ds.source == "feedback"
    fb_hold = in_hold & fb
    if fb_hold.any() and mode == "group":
        # a production outcome about a holdout group: keep it for training, take that group out of the holdout
        touched = set(keys[fb_hold].tolist())
        in_hold &= ~np.isin(keys, list(touched))
        notes.append(f"{len(touched)} holdout groups have feedback rows: they were moved to development")
    holdout = in_hold & trusted
    dev = ~in_hold
    dropped = int((in_hold & ~trusted).sum())
    if dropped:
        notes.append(f"{dropped} untrusted rows of holdout groups were dropped (not scored, not trained)")
    if mode == "row":
        notes.append("rows have neither a time nor a group key: they are treated as independent. If rows share a "
                     "document, customer or ticket, give them a group key so near-duplicates cannot leak")
    return Split(mode, dev, holdout, 0, {"holdout_groups": len(hold_groups),
                                         "groups": len(ordered)}, notes)


def calibration_folds(ds: Dataset, dev: np.ndarray, n_splits: int = 5, mode: str = "auto", gap: int = 0,
                      seed: int = 7, min_train: int = 20, min_cal: int = 5,
                      n_classes: int | None = None) -> tuple[list[tuple[np.ndarray, np.ndarray]], list[str]]:
    """Folds over the development rows: (train row indices, calibration row indices). Calibration rows are
    trusted only; training rows are every development row on the training side (trusted, judge, feedback).
    A fold is kept only if its training side has every class and enough rows and its calibration block has
    at least `min_cal` trusted rows."""
    mode = resolve_mode(ds, mode)
    k = n_classes or int(ds.y.max()) + 1
    dev_idx = np.flatnonzero(dev)
    trusted = ds.trusted
    folds: list[tuple[np.ndarray, np.ndarray]] = []
    notes: list[str] = []
    if mode == "time":
        timed = ds.t >= 0
        tdev = ds.t[dev & timed & trusted]
        uniq = np.unique(tdev)
        n_eff = int(min(n_splits, max(1, len(uniq) // max(min_cal, 1) - 1)))
        if n_eff < 2 or len(uniq) <= n_eff + 1 + gap:
            raise NotEnoughData(f"only {len(uniq)} distinct trusted timestamps in development: not enough for "
                                f"time-ordered calibration folds")
        from sklearn.model_selection import TimeSeriesSplit

        # Training side of every fold: development rows without a timestamp, and feedback rows that the holdout
        # split moved in from the holdout window. Those are later than every calibration block, so an expanding
        # window would never use them, although the split note promises that they train the model.
        late_feedback = timed & (ds.source == "feedback") & (ds.t > uniq[-1])
        always = dev & (~timed | late_feedback)
        for tr_u, te_u in TimeSeriesSplit(n_splits=n_eff, gap=gap).split(uniq):
            cal_start, cal_end = uniq[te_u[0]], uniq[te_u[-1]]
            # `gap` distinct trusted timestamps are left out before the block: training rows end before the
            # first one left out (with gap = 0 that is the block start). The label window must also have closed.
            train_cut = uniq[tr_u[-1] + 1]
            train = np.flatnonzero((dev & timed & ~always & (ds.t < train_cut) & (ds.t_end <= cal_start)) | always)
            cal = np.flatnonzero(dev & timed & trusted & (ds.t >= cal_start) & (ds.t <= cal_end))
            folds.append((train, cal))
    else:
        keys = group_keys(ds, mode)
        dev_keys = keys[dev_idx]
        rank = group_order(dev_keys, ds.t[dev_idx] if (mode == "group" and ds.t is not None) else None, seed)
        ordered = sorted(rank, key=lambda g: rank[g])
        if len(ordered) < 2:
            raise NotEnoughData(f"{len(ordered)} development groups: need at least 2 for calibration folds")
        n_eff = min(n_splits, len(ordered))
        chunks = np.array_split(np.arange(len(ordered)), n_eff)
        pos = {g: i for i, g in enumerate(ordered)}
        fold_of = np.array([np.searchsorted([c[-1] for c in chunks], pos[g]) for g in dev_keys.tolist()])
        for f in range(n_eff):
            train = dev_idx[fold_of != f]
            cal = dev_idx[(fold_of == f) & trusted[dev_idx]]
            folds.append((train, cal))
    usable = []
    single = 0
    for tr, ca in folds:
        if len(tr) < min_train or len(ca) < min_cal or len(np.unique(ds.y[tr])) < k:
            continue
        if len(np.unique(ds.y[ca])) < 2:
            # a calibrator fitted on one class learns "this never happens" and drags the ensemble average with it
            single += 1
            continue
        usable.append((tr, ca))
    dropped = len(folds) - len(usable)
    if dropped:
        notes.append(f"{dropped} of {len(folds)} calibration folds dropped (training side missing a class, fewer "
                     f"than {min_cal} trusted calibration rows, or a calibration block holding a single class"
                     f"{f': {single}' if single else ''})")
    if not usable:
        raise NotEnoughData("no calibration fold has every class on its training side and enough trusted rows "
                            "(of at least two classes) in its calibration block")
    return usable, notes


def skill_blocks(ds: Dataset, idx: np.ndarray, mode: str) -> np.ndarray | int:
    """Block keys for the skill test over the holdout rows `idx` (already in scoring order)."""
    if mode == "time":
        return 20
    return group_keys(ds, mode)[idx]


def scoring_order(ds: Dataset, idx: np.ndarray, mode: str, seed: int = 7) -> np.ndarray:
    """Holdout rows in time order (time mode) or grouped (group / row mode), for contiguous blocks."""
    idx = np.asarray(idx)
    if mode == "time":
        return idx[np.argsort(ds.t[idx], kind="stable")]
    keys = group_keys(ds, mode)[idx].tolist()
    rank = group_order(np.asarray(keys, dtype=object), None, seed)
    order = sorted(range(len(idx)), key=lambda i: (rank[keys[i]], i))
    return idx[np.asarray(order, dtype=int)]
