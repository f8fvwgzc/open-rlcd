"""How rows are kept apart: the holdout is never trained on, calibrators only see trusted rows of their own block,
and no label window or group straddles a boundary."""
import numpy as np
import pytest

from open_rlcd.core import splits as S
from open_rlcd.core.dataset import Dataset


def timed(n=1500, seed=0, horizon=40, sources=("ground_truth", "human", "judge"), p=(0.5, 0.2, 0.3), y=None):
    """Rows with timestamps (several rows may share one) and label windows of up to `horizon` seconds."""
    rng = np.random.default_rng(seed)
    t = np.sort(rng.integers(0, 6000, n))
    return Dataset(X=rng.normal(size=(n, 3)), y=rng.integers(0, 2, n) if y is None else y,
                   feature_names=["a", "b", "c"], feature_version="v", t=t, t_end=t + rng.integers(0, horizon, n),
                   source=rng.choice(sources, size=n, p=p))


def grouped(n_groups=150, seed=0, with_time=True, feedback_groups=()):
    rng = np.random.default_rng(seed)
    sizes = rng.integers(1, 12, n_groups)
    group = np.repeat([f"doc{g}" for g in range(n_groups)], sizes)
    n = len(group)
    first = rng.permutation(n_groups) * 100
    t = (np.repeat(first, sizes) + rng.integers(0, 50, n)) if with_time else None
    source = rng.choice(["human", "ground_truth", "judge"], size=n, p=[0.4, 0.3, 0.3])
    for g in feedback_groups:                         # production outcomes about an existing document
        source[np.flatnonzero(group == g)[:1]] = "feedback"
    return Dataset(X=rng.normal(size=(n, 3)), y=rng.integers(0, 2, n), feature_names=["a", "b", "c"],
                   feature_version="v", t=t, group=group, source=source)


# ----------------------------------------------------------------------------- modes

def test_resolve_mode():
    assert S.resolve_mode(grouped()) == "group"
    assert S.resolve_mode(timed()) == "time"
    plain = Dataset(X=np.zeros((4, 1)), y=[0, 1, 0, 1], feature_names=["a"], feature_version="v")
    assert S.resolve_mode(plain) == "row"
    assert S.resolve_mode(grouped(), "time") == "time"
    with pytest.raises(ValueError, match="needs timestamps"):
        S.resolve_mode(plain, "time")
    with pytest.raises(ValueError, match="needs a group key"):
        S.resolve_mode(plain, "group")
    with pytest.raises(ValueError, match="ordering must be one of"):
        S.resolve_mode(plain, "random")


# ----------------------------------------------------------------------------- time ordering

def test_time_holdout_is_the_latest_trusted_rows_with_a_purge():
    ds = timed()
    sp = S.holdout_split(ds, 0.2, "time")
    cut = sp.cut["time"]
    assert sp.mode == "time"
    assert (ds.t[sp.dev] < cut).all() and (ds.t_end[sp.dev] <= cut).all()      # no label window reaches past
    assert (ds.t[sp.holdout] >= cut).all() and ds.trusted[sp.holdout].all()
    assert not (sp.dev & sp.holdout).any()
    assert not (sp.holdout & (ds.source == "judge")).any()
    purged = ~sp.dev & (ds.t < cut)
    assert purged.sum() == sp.purged > 0 and (ds.t_end[purged] > cut).all()
    share = sp.holdout.sum() / ds.trusted.sum()
    assert 0.17 < share < 0.23
    assert any("untrusted rows in the holdout window were dropped" in n for n in sp.notes)


def test_time_folds_train_on_the_past_with_closed_label_windows():
    ds = timed()
    sp = S.holdout_split(ds, 0.2, "time")
    folds, notes = S.calibration_folds(ds, sp.dev, 5, "time")
    assert len(folds) == 5 and notes == []
    seen_cal: set[int] = set()
    sizes = []
    for tr, ca in folds:
        cal_start = ds.t[ca].min()
        assert sp.dev[tr].all() and sp.dev[ca].all() and not set(tr) & set(ca)
        assert ds.trusted[ca].all()                               # calibrators see trusted rows only
        assert ds.t[tr].max() < cal_start                         # fit on the past
        assert ds.t_end[tr].max() <= cal_start                    # whose labels were known by then
        assert (ds.source[tr] == "judge").any()                   # weak labels do train
        assert not seen_cal & set(ca)                             # calibration blocks do not overlap
        seen_cal |= set(ca.tolist())
        sizes.append(len(tr))
    assert sizes == sorted(sizes)                                 # an expanding window
    assert not seen_cal & set(np.flatnonzero(sp.holdout))


@pytest.mark.parametrize("gap", [0, 3, 8])
def test_time_folds_leave_a_gap_before_each_calibration_block(gap):
    ds = timed(horizon=1)
    sp = S.holdout_split(ds, 0.2, "time")
    folds, _ = S.calibration_folds(ds, sp.dev, 4, "time", gap=gap)
    uniq = np.unique(ds.t[sp.dev & ds.trusted])
    for tr, ca in folds:
        cal_start = ds.t[ca].min()
        between = uniq[(uniq > ds.t[tr].max()) & (uniq < cal_start)]
        assert len(between) >= gap                                # `gap` trusted timestamps are left out
        if gap == 0:
            assert len(between) == 0


def test_time_folds_with_an_expanding_window_train_on_more_with_a_smaller_gap():
    ds = timed(horizon=1)
    sp = S.holdout_split(ds, 0.2, "time")
    no_gap, _ = S.calibration_folds(ds, sp.dev, 4, "time", gap=0)
    gap, _ = S.calibration_folds(ds, sp.dev, 4, "time", gap=10)
    assert all(len(a[0]) > len(b[0]) for a, b in zip(no_gap, gap, strict=True))
    assert all(np.array_equal(a[1], b[1]) for a, b in zip(no_gap, gap, strict=True))   # same blocks


def test_feedback_in_the_holdout_window_trains_but_is_never_scored():
    base = timed(n=1200, seed=1)
    cut = S.holdout_split(base, 0.2, "time").cut["time"]
    rng = np.random.default_rng(5)
    late_t = np.array([cut + 50, cut + 400, cut + 900])
    fb = Dataset(X=rng.normal(size=(3, 3)), y=[1, 0, 1], feature_names=["a", "b", "c"], feature_version="v",
                 t=late_t, t_end=late_t + 5, source=["feedback"] * 3, weight=[5.0] * 3, ids=["fb0", "fb1", "fb2"])
    from open_rlcd.core.dataset import concat

    ds = concat([base, fb])
    sp = S.holdout_split(ds, 0.2, "time")
    is_fb = ds.source == "feedback"
    assert sp.cut["time"] == cut
    assert not sp.holdout[is_fb].any() and sp.dev[is_fb].all()
    assert any("feedback rows fall in the holdout window" in n for n in sp.notes)
    # holdout rows whose label window overlaps a feedback row were removed from the holdout
    for i in np.flatnonzero(is_fb):
        overlap = (ds.t <= ds.t_end[i]) & (ds.t_end >= ds.t[i]) & ~is_fb
        assert not (sp.holdout & overlap).any()
    folds, _ = S.calibration_folds(ds, sp.dev, 5, "time")
    for tr, ca in folds:
        assert set(np.flatnonzero(is_fb)) <= set(tr.tolist())    # they train every fold ...
        assert not is_fb[ca].any()                                # ... and never calibrate
        rest = tr[~is_fb[tr]]
        assert ds.t[rest].max() < ds.t[ca].min()                  # everything else is still the past


def test_untimed_rows_are_training_side_only():
    ds = timed(n=800, seed=2)
    t = ds.t.copy()
    t[:40] = -1
    ds = Dataset(X=ds.X, y=ds.y, feature_names=ds.feature_names, feature_version="v", t=t, source=ds.source)
    sp = S.holdout_split(ds, 0.2, "time")
    assert sp.dev[:40].all() and not sp.holdout[:40].any()
    folds, _ = S.calibration_folds(ds, sp.dev, 4, "time")
    for tr, ca in folds:
        assert set(range(40)) <= set(tr.tolist()) and not (ca < 40).any()


def test_a_single_class_calibration_block_is_dropped():
    t = np.arange(300)
    y = np.tile([0, 1], 150)
    y[40:90] = 0                                    # the first calibration block (t 44..82) holds one class
    ds = Dataset(X=np.random.default_rng(0).normal(size=(300, 2)), y=y, feature_names=["a", "b"],
                 feature_version="v", t=t)
    sp = S.holdout_split(ds, 0.2, "time")
    folds, notes = S.calibration_folds(ds, sp.dev, 5, "time", min_train=10)
    assert len(folds) == 4 and all(len(np.unique(ds.y[ca])) == 2 for _, ca in folds)
    assert "single class: 1" in notes[0]


def test_too_few_trusted_rows_cannot_be_split():
    ds = timed(n=200, sources=("judge", "human"), p=(0.995, 0.005), seed=3)
    with pytest.raises(S.NotEnoughData):
        sp = S.holdout_split(ds, 0.2, "time")
        S.calibration_folds(ds, sp.dev, 5, "time")


# ----------------------------------------------------------------------------- group ordering

def test_group_holdout_never_splits_a_group():
    ds = grouped()
    sp = S.holdout_split(ds, 0.2, "group")
    dev_groups, hold_groups = set(ds.group[sp.dev]), set(ds.group[sp.holdout])
    assert sp.mode == "group" and not dev_groups & hold_groups
    assert ds.trusted[sp.holdout].all() and not (sp.dev & sp.holdout).any()
    # untrusted rows of holdout groups are neither scored nor trained on
    in_hold = np.isin(ds.group, list(hold_groups))
    assert not (sp.dev & in_hold).any() and not (in_hold & ~ds.trusted & sp.holdout).any()
    assert 0.2 <= sp.holdout.sum() / ds.trusted.sum() < 0.3
    # with timestamps the holdout is the latest groups
    first = {g: ds.t[ds.group == g].min() for g in set(ds.group)}
    assert min(first[g] for g in hold_groups) > max(first[g] for g in dev_groups)


def test_group_holdout_without_time_is_a_seeded_shuffle():
    ds = grouped(with_time=False)
    a, b = S.holdout_split(ds, 0.2, "group", seed=1), S.holdout_split(ds, 0.2, "group", seed=1)
    c = S.holdout_split(ds, 0.2, "group", seed=2)
    assert np.array_equal(a.holdout, b.holdout) and not np.array_equal(a.holdout, c.holdout)
    assert not set(ds.group[a.dev]) & set(ds.group[a.holdout])


def _judge_only_groups(with_time):
    """Half the documents were labelled by the judge only (the usual shape of label-store data)."""
    ds = grouped(n_groups=200, with_time=with_time, seed=4)
    judge_doc = np.array([int(g[3:]) % 2 == 0 for g in ds.group])
    source = np.where(judge_doc, "judge", "human")
    return Dataset(X=ds.X, y=ds.y, feature_names=ds.feature_names, feature_version="v", t=ds.t, group=ds.group,
                   source=source), judge_doc


def test_without_time_judge_only_groups_stay_in_development():
    ds, judge_doc = _judge_only_groups(with_time=False)
    sp = S.holdout_split(ds, 0.2, "group")
    assert sp.dev[judge_doc].all()                                # nothing to score, nothing to leak: kept
    assert (sp.dev | sp.holdout).all() and not any("dropped" in n for n in sp.notes)
    assert not set(ds.group[sp.dev]) & set(ds.group[sp.holdout])
    rng = np.random.default_rng(0)
    rows = Dataset(X=rng.normal(size=(600, 2)), y=rng.integers(0, 2, 600), feature_names=["a", "b"],
                   feature_version="v", source=rng.choice(["human", "judge"], 600))
    rsp = S.holdout_split(rows, 0.2, "row")
    assert rsp.dev[rows.source == "judge"].all()


def test_with_time_judge_only_groups_of_the_holdout_period_are_dropped():
    ds, judge_doc = _judge_only_groups(with_time=True)
    sp = S.holdout_split(ds, 0.2, "group")
    first = {g: ds.t[ds.group == g].min() for g in set(ds.group)}
    start = min(first[g] for g in set(ds.group[sp.holdout]))
    late_judge = judge_doc & np.array([first[g] >= start for g in ds.group])
    assert late_judge.any() and not sp.dev[late_judge].any()      # training on them = training on the holdout period
    assert any("untrusted rows of holdout groups were dropped" in n for n in sp.notes)


def test_group_folds_never_split_a_group_and_calibrate_on_trusted_rows():
    ds = grouped()
    sp = S.holdout_split(ds, 0.2, "group")
    folds, _ = S.calibration_folds(ds, sp.dev, 5, "group")
    assert len(folds) == 5
    for tr, ca in folds:
        assert not set(ds.group[tr]) & set(ds.group[ca])
        assert ds.trusted[ca].all() and sp.dev[tr].all() and sp.dev[ca].all()
        assert (ds.source[tr] == "judge").any()
    cal = np.concatenate([ca for _, ca in folds])
    assert len(cal) == len(set(cal.tolist()))                     # each trusted row calibrates once at most


def test_feedback_rows_never_enter_the_holdout():
    ds = grouped()
    latest = S.holdout_split(ds, 0.2, "group")
    target = sorted(set(ds.group[latest.holdout]))[:2]            # two documents that would be held out
    fb = grouped(feedback_groups=target)
    sp = S.holdout_split(fb, 0.2, "group")
    is_fb = fb.source == "feedback"
    assert is_fb.sum() == 2 and not sp.holdout[is_fb].any() and sp.dev[is_fb].all()
    assert not set(target) & set(fb.group[sp.holdout])            # the whole group moved to development
    assert any("holdout groups have feedback rows" in n for n in sp.notes)
    folds, _ = S.calibration_folds(fb, sp.dev, 5, "group")
    assert all(not is_fb[ca].any() for _, ca in folds)
    assert any(is_fb[tr].any() for tr, _ in folds)


def test_row_mode_says_rows_are_treated_as_independent():
    rng = np.random.default_rng(0)
    ds = Dataset(X=rng.normal(size=(500, 2)), y=rng.integers(0, 2, 500), feature_names=["a", "b"],
                 feature_version="v")
    sp = S.holdout_split(ds, 0.2)
    assert sp.mode == "row" and any("independent" in n for n in sp.notes)
    assert 95 <= sp.holdout.sum() <= 105 and not (sp.dev & sp.holdout).any()
    folds, _ = S.calibration_folds(ds, sp.dev, 5, "row")
    assert len(folds) == 5 and all(not set(tr) & set(ca) for tr, ca in folds)


def test_skill_blocks_and_scoring_order():
    ds = grouped()
    sp = S.holdout_split(ds, 0.2, "group")
    idx = S.scoring_order(ds, np.flatnonzero(sp.holdout), "group")
    blocks = S.skill_blocks(ds, idx, "group")
    assert sorted(idx.tolist()) == sorted(np.flatnonzero(sp.holdout).tolist())
    assert np.array_equal(blocks, ds.group[idx])
    runs = [g for i, g in enumerate(blocks) if i == 0 or blocks[i - 1] != g]
    assert len(runs) == len(set(blocks))                          # each group is one contiguous run
    t = timed()
    tsp = S.holdout_split(t, 0.2, "time")
    tidx = S.scoring_order(t, np.flatnonzero(tsp.holdout)[::-1], "time")
    assert np.all(np.diff(t.t[tidx]) >= 0) and S.skill_blocks(t, tidx, "time") == 20
