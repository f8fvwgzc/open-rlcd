"""A train that leaves heads of rlcd-latest out of its new version says which.

A train carries a head it does not retrain over from rlcd-latest only when rlcd-latest's model.json records the same
pack, feature version and pack settings (`carry_key`; core/registry.py); a version saved by an open-rlcd release that
did not record `carry_key` has none. Otherwise the new version lacks the head: its model.json lists it under
`not_carried_over` (sorted), the train job (GET /v1/train/{job_id}) copies the list before it names the version, and
`rlcd train` prints one line on stderr, after its table. A head the project still serves answers with the uninformed
prior until a train retrains it, and the line gives that train. Before, neither the job nor `rlcd train` named such a
head; only the version's notes did.

The tabular pack on small tables, trained in well under a second: moving a column to `pack_config.exclude` changes
the feature version, and so does removing a target under `features: auto` (its label column becomes a feature).
"""
from __future__ import annotations

import io
import json
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from fastapi.testclient import TestClient

from open_rlcd.cli import _not_carried_over_line, main
from open_rlcd.core.config import load_project
from open_rlcd.core.contract import ContractError, SystemOneRequest, TrainRequest
from open_rlcd.core.jobs import Trainer
from open_rlcd.core.service import Service
from open_rlcd.packs.tabular import TabularPack
from open_rlcd.serve.api import create_app
from tests.test_core_service import train
from tests.test_tabular import small, small_table

TARGETS = {"churned": {"label": "churned", "model": "logreg"}, "next_plan": {"label": "next_plan", "model": "logreg"}}
# the reason the note in model.json and the line give: also true for a version that recorded no carry_key
WHY = ("(trained with another feature version, pack or pack settings, or by an open-rlcd release that did not record "
       "them)")
NOT_SERVED = "not served now (switched off, or no longer one of this project's heads)"


def write_project(root: Path, targets: dict | None = None, **pack_config) -> Path:
    """root/rlcd.yaml for the tabular pack on root/small.csv (features x, y, plan; label columns churned, next_plan),
    or on the table `data` names."""
    if "data" not in pack_config and not (root / "small.csv").exists():
        root.mkdir(parents=True, exist_ok=True)
        small_table(root / "small.csv")
    cfg = {"pack": "tabular", "pack_config": {"data": "small.csv", "targets": targets or TARGETS, **pack_config}}
    (root / "rlcd.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return root


def rlcd_train(root: Path, home: Path, capsys, *argv: str) -> tuple[dict, str, str]:
    """`rlcd train ...`: the model.json of the version it saved, and what it printed on stdout and on stderr."""
    capsys.readouterr()
    assert main(["--project", str(root), "--home", str(home), "train", *argv]) == 0
    printed = capsys.readouterr()
    latest = json.loads((home / "registry.json").read_text(encoding="utf-8"))["latest"]
    meta = json.loads((home / "models" / latest / "model.json").read_text(encoding="utf-8"))
    return meta, printed.out, printed.err


def said(err: str) -> list[str]:
    return [line for line in err.splitlines() if line.startswith("not carried over")]


def test_rlcd_train_names_a_head_it_leaves_out_and_the_train_that_brings_it_back(tmp_path, capsys):
    root, home = write_project(tmp_path / "p"), tmp_path / "home"
    meta, out, err = rlcd_train(root, home, capsys)
    assert set(meta["heads"]) == {"churned", "next_plan"} and "not_carried_over" not in meta
    # the same settings: a partial train carries next_plan over, adds no key and prints nothing about it
    meta, out, err = rlcd_train(root, home, capsys, "--heads", "churned")
    assert meta["heads"]["next_plan"]["trained_in"] == "rlcd-0.1.1" and "not_carried_over" not in meta
    assert "not carried over" not in out + err
    # `plan` moved to exclude: another feature version, so the partial train leaves next_plan out
    write_project(root, exclude=["plan"])
    meta, out, err = rlcd_train(root, home, capsys, "--heads", "churned")
    assert meta["version"] == "rlcd-0.1.3" and list(meta["heads"]) == ["churned"]
    assert meta["not_carried_over"] == ["next_plan"]
    assert f"not carried over {WHY}: ['next_plan']" in meta["notes"]
    assert said(err) == [f"not carried over from rlcd-0.1.2: next_plan {WHY}; until a train retrains it, it answers "
                         f"with the uninformed prior: rlcd train --heads next_plan"]
    assert "not carried over" not in out
    # what the line says: the project still serves next_plan, and rlcd-0.1.3 answers it with the uninformed prior
    svc = Service(load_project(root), home=home)
    res = svc.systemone(SystemOneRequest(state={"x": 0.9, "y": 0.5}, questions={"next_plan": {"type": "choice"}}))
    answer = res["answers"]["next_plan"]
    assert res["model"] == "rlcd-0.1.3" and answer["calibrated"] is False
    assert len(answer["probabilities"]) == 3 and len(set(answer["probabilities"].values())) == 1
    # the train it gives brings next_plan back and carries churned over: nothing left out, nothing printed about it
    meta, out, err = rlcd_train(root, home, capsys, "--heads", "next_plan")
    assert set(meta["heads"]) == {"churned", "next_plan"} and meta["heads"]["churned"]["trained_in"] == "rlcd-0.1.3"
    assert "not_carried_over" not in meta and "not carried over" not in out + err


def test_a_head_the_project_no_longer_serves_is_named_without_a_train_to_bring_it_back(tmp_path, capsys):
    """Under `features: auto`, removing the target next_plan makes its label column a feature: the next train
    leaves the head out. The project no longer serves it, so the line gives no `rlcd train --heads` for it (one that
    names it is refused: it is not a head of this project)."""
    root, home = write_project(tmp_path / "p"), tmp_path / "home"
    rlcd_train(root, home, capsys)
    write_project(root, {"churned": TARGETS["churned"]})
    meta, out, err = rlcd_train(root, home, capsys)
    assert list(meta["heads"]) == ["churned"] and meta["not_carried_over"] == ["next_plan"]
    assert said(err) == [f"not carried over from rlcd-0.1.1: next_plan {WHY}; it is {NOT_SERVED}"]
    # what the line says: no question is answered with next_plan, and a train cannot name it
    with pytest.raises(ContractError) as e:
        Service(load_project(root), home=home).systemone(
            SystemOneRequest(state={"x": 0.9, "y": 0.5}, questions={"next_plan": {"type": "choice"}}))
    assert e.value.kind == "unknown_head"
    assert main(["--project", str(root), "--home", str(home), "train", "--heads", "next_plan"]) == 1
    assert "unknown heads ['next_plan']" in capsys.readouterr().err


def four_targets(path: Path, n: int = 600) -> Path:
    """Features x, y, z and plan; four yes / no label columns, a from x, b from y, c from z and d from x and z."""
    rnd = random.Random(0)
    lines = ["x,y,z,plan,a,b,c,d"]
    for i in range(n):
        x, y, z = rnd.random(), rnd.random(), rnd.random()
        yes = [v + rnd.random() * 0.3 > 0.65 for v in (x, y, z, (x + z) / 2)]
        lines.append(f"{x:.3f},{y:.3f},{z:.3f},{'pq'[i % 2]}," + ",".join("yes" if v else "no" for v in yes))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def jobs_of(monkeypatch) -> list[dict]:
    """The record of each train job `rlcd train` runs from now on (what `Trainer.start` returns to it)."""
    jobs: list[dict] = []
    start = Trainer.start

    def recorded(self, *args, **kwargs):
        jobs.append(start(self, *args, **kwargs))
        return jobs[-1]

    monkeypatch.setattr(Trainer, "start", recorded)
    return jobs


def test_a_train_that_leaves_several_heads_out_lists_them_sorted_and_says_what_each_does_now(tmp_path, capsys,
                                                                                            monkeypatch):
    """Four targets, declared out of alphabetical order, which model.json keeps. With `plan` moved to `exclude`
    (another feature version) and d switched off, `rlcd train --heads c` leaves b, d and a out: model.json and the
    train job list them sorted, and the line says that a and b answer with the uninformed prior until a train
    retrains them, and that d is not served."""
    root, home = tmp_path / "p", tmp_path / "home"
    four_targets(root / "t.csv")
    targets = {n: {"label": n, "model": "logreg"} for n in "bdac"}
    write_project(root, targets, data="t.csv")
    meta, _, _ = rlcd_train(root, home, capsys)
    assert list(meta["heads"]) == ["b", "d", "a", "c"] and "not_carried_over" not in meta
    write_project(root, {**targets, "d": {**targets["d"], "enabled": False}}, data="t.csv", exclude=["plan"])
    jobs = jobs_of(monkeypatch)
    meta, out, err = rlcd_train(root, home, capsys, "--heads", "c")
    assert list(meta["heads"]) == ["c"] and meta["not_carried_over"] == ["a", "b", "d"]
    assert [job["not_carried_over"] for job in jobs] == [["a", "b", "d"]]
    assert f"not carried over {WHY}: ['a', 'b', 'd']" in meta["notes"]
    assert said(err) == [f"not carried over from rlcd-0.1.1: a, b, d {WHY}; until a train retrains them, a, b answer "
                         f"with the uninformed prior: rlcd train --heads a,b; d is {NOT_SERVED}"]


def test_the_reason_holds_for_a_version_saved_before_carry_key_was_recorded(tmp_path, capsys, monkeypatch):
    """An open-rlcd release that did not record `carry_key` saved its versions without one. A pack that gives one
    (the trading pack's sl_pips, tp_pips, tp2_pips and horizon) then carries none of their heads over at the first
    train after the upgrade, even with the very same settings: the note and the line say that this is a reason too.
    Here the tabular pack, which gives none, saves the first version, and is then given a carry_key."""
    root, home = write_project(tmp_path / "p"), tmp_path / "home"
    meta, _, _ = rlcd_train(root, home, capsys)
    assert "carry_key" not in meta["pack"]
    monkeypatch.setattr(TabularPack, "carry_key", lambda self, provider=None: {"window": 30})      # the upgrade
    meta, out, err = rlcd_train(root, home, capsys, "--heads", "churned")
    assert meta["pack"]["carry_key"] == {"window": 30} and list(meta["heads"]) == ["churned"]
    assert meta["not_carried_over"] == ["next_plan"] and f"not carried over {WHY}: ['next_plan']" in meta["notes"]
    assert said(err) == [f"not carried over from rlcd-0.1.1: next_plan {WHY}; until a train retrains it, it answers "
                         f"with the uninformed prior: rlcd train --heads next_plan"]
    # the train the line gives brings next_plan back; from then on the recorded carry_key carries churned over
    meta, out, err = rlcd_train(root, home, capsys, "--heads", "next_plan")
    assert meta["heads"]["churned"]["trained_in"] == "rlcd-0.1.2" and "not_carried_over" not in meta
    assert not said(err)


class Piped(io.StringIO):
    """One stream of `rlcd train 2>&1 | tee train.log`: what is written reaches the shared `pipe` when the stream is
    flushed (stdout, which Python buffers in blocks when it is a pipe) or at once (stderr)."""

    def __init__(self, pipe: list[str], buffered: bool):
        super().__init__()
        self.pipe, self.buffered = pipe, buffered

    def write(self, s: str) -> int:
        n = super().write(s)
        if not self.buffered:
            self.flush()
        return n

    def flush(self) -> None:
        self.pipe.append(self.getvalue())
        self.seek(0)
        self.truncate()


def test_with_both_streams_in_one_pipe_the_line_comes_after_the_table(tmp_path, capsys, monkeypatch):
    """`rlcd train` flushes its table before it writes the line on stderr, so `rlcd train 2>&1 | tee train.log`
    shows the line after the table (and 'Next:' last)."""
    root, home = write_project(tmp_path / "p"), tmp_path / "home"
    rlcd_train(root, home, capsys)
    write_project(root, exclude=["plan"])
    pipe: list[str] = []
    stdout = Piped(pipe, buffered=True)
    monkeypatch.setattr(sys, "stdout", stdout)
    monkeypatch.setattr(sys, "stderr", Piped(pipe, buffered=False))
    assert main(["--project", str(root), "--home", str(home), "train", "--heads", "churned"]) == 0
    stdout.flush()      # at exit
    text = "".join(pipe)
    assert text.index("model rlcd-0.1.2") < text.index("not carried over from rlcd-0.1.1") < text.index("Next:")


class Gone(io.StringIO):
    """stdout piped into a reader that has already exited (`rlcd train | true`): what is written stays in the buffer,
    and a flush fails."""

    def flush(self) -> None:
        raise BrokenPipeError(32, "Broken pipe")


def test_a_stdout_whose_reader_has_gone_still_lets_the_line_out(tmp_path, capsys, monkeypatch):
    """`rlcd train` lets the flush of its table fail: it does not stop there with `rlcd: BrokenPipeError: ...` (main
    returning 1), and the line still reaches stderr."""
    root, home = write_project(tmp_path / "p"), tmp_path / "home"
    rlcd_train(root, home, capsys)
    write_project(root, exclude=["plan"])
    monkeypatch.setattr(sys, "stdout", Gone())
    assert main(["--project", str(root), "--home", str(home), "train", "--heads", "churned"]) == 0
    assert said(capsys.readouterr().err) == [f"not carried over from rlcd-0.1.1: next_plan {WHY}; until a train "
                                             f"retrains it, it answers with the uninformed prior: rlcd train --heads "
                                             f"next_plan"]


def test_the_train_job_lists_the_heads_its_version_lacks(tmp_path, monkeypatch):
    monkeypatch.delenv("RLCD_ADMIN_KEY", raising=False)
    small_table(tmp_path / "small.csv")
    api = TestClient(create_app(small(tmp_path, {})))
    first, again = train(api), train(api, {"heads": ["churned"]})              # again: next_plan is carried over
    assert first["status"] == again["status"] == "succeeded"
    assert "not_carried_over" not in first and "not_carried_over" not in again
    svc = small(tmp_path, {}, {"exclude": ["plan"]})                           # another feature version
    job = train(TestClient(create_app(svc)), {"heads": ["churned"]})
    assert job["status"] == "succeeded" and job["model"] == "rlcd-0.1.3" and job["not_carried_over"] == ["next_plan"]
    assert svc.bundle("rlcd-0.1.3").meta["not_carried_over"] == ["next_plan"]


class Watched(dict):
    """A job record that keeps what `Trainer.get` (GET /v1/train/{job_id}) returns after each of its keys is set."""

    def __init__(self, job: dict):
        super().__init__(job)
        self.reads: list[dict] = []

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        self.reads.append(dict(self))


def test_a_job_record_that_names_the_new_version_lists_the_heads_it_lacks(tmp_path, monkeypatch):
    """A client may poll GET /v1/train/{job_id} until `model` appears. The job sets `not_carried_over` before `model`,
    so no read of the record names the new version without the heads that version lacks."""
    small_table(tmp_path / "small.csv")
    small(tmp_path, {}).trainer.start(TrainRequest(), wait=True)
    trainer = small(tmp_path, {}, {"exclude": ["plan"]}).trainer               # another feature version
    run, reads = Trainer._run, []

    def watched(self, job, req, names):          # the training thread, on a record that keeps each read of it
        record = Watched(job)
        run(self, record, req, names)
        reads.extend(record.reads)

    monkeypatch.setattr(Trainer, "_run", watched)
    trainer.start(TrainRequest(heads=["churned"]), wait=True)
    named = [r for r in reads if r["model"]]
    assert named and all(r.get("not_carried_over") == ["next_plan"] for r in named)
    assert named[-1]["model"] == "rlcd-0.1.2" and named[-1]["status"] == "succeeded"


def test_a_job_record_being_read_never_gains_a_key_meanwhile():
    """GET /v1/train/{job_id} serialises what `Trainer.get` returns while the training thread may still add
    `not_carried_over` to the record: it gets a copy, which that thread never changes."""
    trainer = Trainer(None)
    trainer.jobs["j"] = {"job_id": "j", "status": "running", "model": None}
    reading = iter(trainer.get("j").items())
    next(reading)
    trainer.jobs["j"].update(model="rlcd-0.1.3", not_carried_over=["next_plan"])    # the training thread
    assert [k for k, _ in reading] == ["status", "model"]
    assert trainer.get("j")["not_carried_over"] == ["next_plan"] and trainer.get("missing") is None


@pytest.mark.parametrize("defined, then", [
    ({"a", "b", "c"}, "until a train retrains them, they answer with the uninformed prior: rlcd train --heads a,b"),
    ({"a", "c"}, f"until a train retrains it, a answers with the uninformed prior: rlcd train --heads a; b is "
                 f"{NOT_SERVED}"),
    ({"c"}, f"they are {NOT_SERVED}"),
], ids=["both served", "one served", "neither served"])
def test_the_line_says_what_each_head_left_out_does_now(defined, then):
    """Whether the project serves a head left out (`pack.heads`) decides what the line says about it."""
    svc = SimpleNamespace(project=None, pack=SimpleNamespace(heads=lambda project: dict.fromkeys(defined)),
                          registry=SimpleNamespace(registry=lambda: {"history": [
                              {"from": "rlcd-0.1.1", "to": "rlcd-0.1.2", "kind": "train"}]}))
    job = {"model": "rlcd-0.1.2", "not_carried_over": ["a", "b"]}
    assert _not_carried_over_line(svc, job) == f"not carried over from rlcd-0.1.1: a, b {WHY}; {then}"
    assert _not_carried_over_line(svc, {"model": "rlcd-0.1.2"}) == ""
