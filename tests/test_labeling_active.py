"""Active learning on a tiny trained toy service: uncertainty first, human-labelled items excluded, diversity
skips exact duplicates, judge/model disagreement counts, untrained heads fall back to judge confidence."""
from __future__ import annotations

import numpy as np
import pytest
from toypack import NAMES, ToyProvider, generate

from open_rlcd.core.config import Project
from open_rlcd.core.dataset import Dataset
from open_rlcd.core.head import HeadSpec
from open_rlcd.core.service import Service
from open_rlcd.core.training import TrainConfig, train_head
from open_rlcd.labeling.active import greedy_diverse, next_batch, read_queue, uncertainty, write_queue
from open_rlcd.labeling.store import LabelItem, LabelStore
from open_rlcd.packs.base import BasePack

GOOD = HeadSpec("good", "noul", view="v")
COLOR = HeadSpec("color", "choice", ("red", "green", "blue"), view="v")


class ActPack(BasePack):
    name = "acttoy"

    def heads(self, project):
        return {"good": GOOD, "color": COLOR}

    def provider(self, project, config=None):
        return ToyProvider()

    def datasets(self, project, provider, heads, feedback=None, feedback_weight=5.0):
        return {}


@pytest.fixture(scope="module")
def trained_good():
    X, y, _ = generate(600, 2, 2.0, seed=0)
    ds = Dataset(X=X, y=y, feature_names=NAMES, feature_version="toy-v1", source=["synthetic"] * len(y),
                 group=[f"g{j // 3}" for j in range(len(y))])
    cfg = TrainConfig(n_splits=3)
    return train_head(GOOD, ds, cfg), cfg


@pytest.fixture
def setup(home, tmp_path, trained_good):
    th, cfg = trained_good
    project = Project(root=tmp_path)
    pack = ActPack()
    svc = Service(project, home=home, pack=pack)
    svc.registry.commit({"good": th}, pack=pack.info(ToyProvider()), feature_version="toy-v1", encoder=None,
                        policy=cfg.policy.to_dict())
    rng = np.random.default_rng(5)
    store = LabelStore(home)
    store.add_items([LabelItem(f"c{i}", "v", {"f": [float(v) for v in rng.normal(size=6)]}) for i in range(40)])
    return project, svc, store, th


def probs(th, store, ids):
    items = {it.id: it for it in store.items()}
    X = np.asarray([items[i].state["f"] for i in ids], np.float32)
    return th.ensemble.predict_proba(X)


def test_most_uncertain_first_and_human_labelled_excluded(setup):
    project, svc, store, th = setup
    rows = next_batch(project, svc, "good", n=10, weights={"disagreement": 0, "diversity": 0}, store=store)
    ids = [r["item_id"] for r in rows]
    all_ids = [it.id for it in store.items()]
    P = probs(th, store, all_ids)
    expected = [all_ids[i] for i in np.argsort(np.abs(P[:, 1] - 0.5), kind="stable")[:10]]
    assert ids == expected
    assert [r["rank"] for r in rows] == list(range(1, 11))
    assert all(r["trained"] and r["model_version"] == "rlcd-0.1.1" for r in rows)
    assert rows[0]["uncertainty"] >= rows[-1]["uncertainty"] and "model: P(" in rows[0]["why"]

    store.add_human(ids[0], "good", "true")
    again = next_batch(project, svc, "good", n=10, weights={"disagreement": 0, "diversity": 0}, store=store)
    assert ids[0] not in [r["item_id"] for r in again]
    assert [r["item_id"] for r in again][:9] == ids[1:]


def test_diversity_avoids_exact_duplicates(setup):
    project, svc, store, th = setup
    all_ids = [it.id for it in store.items()]
    P = probs(th, store, all_ids)
    most = store.items()[int(np.argmin(np.abs(P[:, 1] - 0.5)))]
    store.add_items([LabelItem(f"dup{k}", "v", dict(most.state)) for k in range(4)])
    rows = next_batch(project, svc, "good", n=5, store=store)
    picked = [r["item_id"] for r in rows]
    twins = {most.id, *(f"dup{k}" for k in range(4))}
    assert len(twins & set(picked)) == 1 and len(picked) == 5
    # without diversity the duplicates crowd the top of the list
    plain = next_batch(project, svc, "good", n=5, weights={"diversity": 0, "disagreement": 0}, store=store)
    assert len(twins & {r["item_id"] for r in plain}) == 5


def test_confident_judge_disagreement_ranks_first(setup):
    project, svc, store, th = setup
    all_ids = [it.id for it in store.items()]
    P = probs(th, store, all_ids)
    sure = int(np.argmax(np.abs(P[:, 1] - 0.5)))
    model_says = "true" if P[sure, 1] > 0.5 else "false"
    store.add_judge(all_ids[sure], "good", "false" if model_says == "true" else "true", "j", confidence=1.0)
    rows = next_batch(project, svc, "good", n=3, weights={"uncertainty": 0.0, "disagreement": 1.0, "diversity": 0},
                      store=store)
    assert rows[0]["item_id"] == all_ids[sure] and rows[0]["disagreement"] == 1.0
    assert "judge says" in rows[0]["why"]


def test_queue_file_keeps_other_heads(setup):
    project, svc, store, _ = setup
    write_queue(store, "color", [{"head": "color", "item_id": "c1", "score": 0.5, "why": "x"}])
    rows = next_batch(project, svc, "good", n=4, store=store)
    assert [r["item_id"] for r in read_queue(store, "good")] == [r["item_id"] for r in rows]
    assert read_queue(store, "color")[0]["item_id"] == "c1"
    next_batch(project, svc, "good", n=2, store=store)
    assert len(read_queue(store, "good")) == 2 and len(read_queue(store)) == 3


def test_untrained_head_falls_back_to_judge_confidence(setup):
    project, svc, store, _ = setup
    ids = [it.id for it in store.items()]
    for k, i in enumerate(ids[:10]):
        store.add_judge(i, "color", "red", "j", confidence=0.5 + k * 0.05)
    rows = next_batch(project, svc, "color", n=35, weights={"diversity": 0}, store=store)
    got = [r["item_id"] for r in rows]
    assert set(got[:30]) == set(ids[10:])                    # no judge label: fully uncertain
    assert got[30:] == ids[:5]                               # then the least confident judge labels
    assert not rows[0]["trained"] and "not trained" in rows[0]["why"]


def test_nothing_to_rank(setup):
    project, svc, store, _ = setup
    for it in store.items():
        store.add_human(it.id, "good", "true")
    assert next_batch(project, svc, "good", n=5, store=store) == []
    assert read_queue(store, "good") == []


def test_uncertainty_and_greedy_helpers():
    u = uncertainty(np.array([[0.5, 0.5], [0.9, 0.1], [1.0, 0.0]]))
    assert u.tolist() == pytest.approx([1.0, 0.2, 0.0])
    u3 = uncertainty(np.array([[1 / 3, 1 / 3, 1 / 3], [1.0, 0.0, 0.0]]))
    assert u3.tolist() == pytest.approx([1.0, 0.0])
    X = np.array([[0.0, 0.0], [0.0, 0.0], [5.0, 5.0], [0.1, 0.0]])
    picks = greedy_diverse(np.array([1.0, 1.0, 0.5, 0.9]), X, 3, w_div=1.0)
    assert [p[0] for p in picks][:2] == [0, 2] and 1 not in [p[0] for p in picks]
