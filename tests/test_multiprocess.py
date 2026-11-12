"""Several processes writing one RLCD_HOME at once (a server plus `rlcd train` / `import` / `promote`).

Each store's read-modify-write must be one critical section across processes: no lost ledger updates, no torn or
swallowed log lines, no version id handed out twice. The workers are real processes (spawn), started together.
"""
import json
import multiprocessing as mp
import os
import threading
import time

import pytest

from open_rlcd.core import locks
from open_rlcd.core.bernoulli import Ledger
from open_rlcd.core.feedback import JsonlLog
from open_rlcd.core.registry import Registry

PROCS = 4
CLASSES = ["a", "b"]


# ----------------------------------------------------------------------------- workers (top level: spawn pickles them)

def _wait(start_at: float) -> None:
    while time.time() < start_at:
        time.sleep(0.001)


def _ledger_worker(args) -> list[str]:
    path, worker, n, start_at = args
    ledger = Ledger(path)
    _wait(start_at)
    errors = []
    for i in range(n):
        try:
            r = ledger.update("h", f"w{worker}-{i}", "a", 0.95, "a" if i % 2 else "b")
            if not (r and r.get("updated")):
                errors.append(f"not updated: {r}")
        except Exception as e:  # the old code could fail here: two processes shared one temp file name
            errors.append(f"{type(e).__name__}: {e}")
    return errors


def _reset_worker(args) -> list[str]:
    path, worker, n, start_at = args
    ledger = Ledger(path)
    _wait(start_at)
    errors = []
    for i in range(n):
        try:
            ledger.reset(f"h{worker}", {**Ledger.empty(CLASSES, f"v{i}"), "valid_for": [f"v{i}"]})
        except Exception as e:
            errors.append(f"{type(e).__name__}: {e}")
    return errors


def _log_worker(args) -> int:
    path, worker, n, pad, start_at = args
    log = JsonlLog(path)
    _wait(start_at)
    for i in range(n):
        log.append({"w": worker, "i": i, "pad": "x" * pad})
    return n


def _commit_worker(args) -> list[str]:
    home, n, start_at = args
    reg = Registry(home)
    _wait(start_at)
    out = []
    for _ in range(n):
        try:
            out.append(reg.commit({}, pack={"name": "p"}, feature_version="f1", encoder=None, policy={}))
        except Exception as e:  # the old code: two processes pick the same id, the second finds the directory taken
            out.append(f"error: {type(e).__name__}: {e}")
    return out


def _edit_worker(args) -> list[str]:
    """`rlcd patch` saving edits that passed their gate, each judged against the version that was latest a moment
    before it is saved (another process may have moved it since)."""
    home, n, start_at = args
    reg = Registry(home)
    _wait(start_at)
    out = []
    for _ in range(n):
        try:
            base = reg.latest_id()
            res = reg.commit_edit(base, {}, edit={"kind": "head", "passed": True, "reasons": []})
            out.append(json.dumps({"version": res["version"], "base": base, "promoted": res["promoted"]}))
        except Exception as e:
            out.append(f"error: {type(e).__name__}: {e}")
    return out


def _train_or_edit_worker(args) -> list[str]:
    kind, *rest = args
    return (_edit_worker if kind == "edit" else _commit_worker)(tuple(rest))


def _promote_worker(args) -> int:
    home, versions, start_at = args
    reg = Registry(home)
    _wait(start_at)
    changed = 0
    for v in versions:
        changed += bool(reg.set_latest(v, reason="test", by=str(os.getpid()))["changed"])
    return changed


def _hold_lock(args) -> float:
    path, start_at, hold_s = args
    _wait(start_at)
    with locks.FileLock(path):
        t = time.time()
        time.sleep(hold_s)
    return t


def _warm(_=None) -> int:
    return os.getpid()


@pytest.fixture(scope="module")
def pool():
    """Spawned workers, shared by the module: importing the package once per worker is the slow part."""
    with mp.get_context("spawn").Pool(PROCS) as p:
        p.map(_warm, range(PROCS * 2))
        yield p


def _soon() -> float:
    return time.time() + 0.1


# ----------------------------------------------------------------------------- the lock itself

def test_file_lock_excludes_other_processes(tmp_path, pool):
    target = tmp_path / "data.json"
    hold = 0.3
    start = _soon()
    res = pool.map_async(_hold_lock, [(target, start, hold)])
    _wait(start + 0.1)                       # the other process holds the lock now
    t0 = time.time()
    with locks.FileLock(target):
        waited = time.time() - t0
    res.get(timeout=30)
    assert waited >= hold - 0.15
    assert (tmp_path / "data.json.lock").exists() and not target.exists()   # a sidecar: the data file is untouched


def test_file_lock_is_reentrant_and_excludes_threads(tmp_path):
    lock = locks.FileLock(tmp_path / "x.json")
    inside, overlaps = [0], [0]

    def work():
        for _ in range(200):
            with lock, lock:             # re-entrant in one thread
                inside[0] += 1
                overlaps[0] += inside[0] != 1
                inside[0] -= 1

    ts = [threading.Thread(target=work) for _ in range(4)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert overlaps[0] == 0
    with lock:                               # released properly: it can be taken again, also by a second object
        pass
    with locks.FileLock(tmp_path / "x.json"):
        pass


def test_missing_fcntl_is_a_clear_error(tmp_path, monkeypatch):
    monkeypatch.setattr(locks, "fcntl", None)
    with pytest.raises(locks.LockUnavailable, match="POSIX"), locks.FileLock(tmp_path / "x.json"):
        pass


# ----------------------------------------------------------------------------- the ledger

def test_ledger_updates_from_several_processes_are_all_counted(tmp_path, pool):
    path = tmp_path / "bernoulli.json"
    Ledger(path).reset("h", {**Ledger.empty(CLASSES, "v1"), "valid_for": ["v1"]})
    n = 150
    start = _soon()
    errors = pool.map(_ledger_worker, [(path, w, n, start) for w in range(PROCS)])
    assert [e for es in errors for e in es] == []
    t = Ledger(path).table("h")
    assert t["online_trials"] == PROCS * n
    assert len(t["applied"]) == len(set(t["applied"])) == PROCS * n
    assert len(t["outcomes"]) == PROCS * n
    wins, losses = t["bins"]["a"][9]
    assert (wins, losses) == (PROCS * n // 2, PROCS * n // 2)
    assert [p.name for p in tmp_path.iterdir() if p.name.endswith(".tmp")] == []


def test_ledger_resets_of_different_heads_do_not_overwrite_each_other(tmp_path, pool):
    path = tmp_path / "bernoulli.json"
    n = 40
    start = _soon()
    errors = pool.map(_reset_worker, [(path, w, n, start) for w in range(PROCS)])
    assert [e for es in errors for e in es] == []
    led = Ledger(path)
    for w in range(PROCS):
        assert led.table(f"h{w}")["model_version"] == f"v{n - 1}"
        for i in range(n - 1):               # every replaced table was archived under its version
            assert led.archived(f"h{w}", f"v{i}")["model_version"] == f"v{i}"


def test_ledger_sees_another_process_write(tmp_path):
    """Nothing is cached between calls: a second Ledger object (another process) is never overwritten by a stale
    copy."""
    path = tmp_path / "bernoulli.json"
    a, b = Ledger(path), Ledger(path)
    a.reset("h", Ledger.empty(CLASSES, "v1"))
    assert a.update("h", "d1", "a", 0.95, "a")["updated"]
    assert b.update("h", "d2", "a", 0.95, "a")["wins"] == 2
    assert a.update("h", "d3", "a", 0.95, "b")["wins"] == 2
    assert a.table("h")["online_trials"] == 3
    assert json.loads(path.read_text())["h"]["applied"] == ["d1", "d2", "d3"]      # the format is unchanged


# ----------------------------------------------------------------------------- JSON-lines logs

def _lines(path):
    raw = path.read_bytes()
    assert raw.endswith(b"\n")
    return raw[:-1].split(b"\n")


@pytest.mark.parametrize("pad", [0, 200_000])
def test_jsonl_appends_from_several_processes_stay_whole(tmp_path, pool, pad):
    path = tmp_path / "feedback.jsonl"
    n = 300 if pad == 0 else 12
    start = _soon()
    assert pool.map(_log_worker, [(path, w, n, pad, start) for w in range(PROCS)]) == [n] * PROCS
    lines = _lines(path)
    assert len(lines) == PROCS * n           # no blank line, nothing glued together
    rows = [json.loads(x) for x in lines]    # every line is whole
    assert {(r["w"], r["i"]) for r in rows} == {(w, i) for w in range(PROCS) for i in range(n)}
    for w in range(PROCS):                   # each writer's rows are in its own order
        assert [r["i"] for r in rows if r["w"] == w] == list(range(n))
    assert len(list(JsonlLog(path).rows())) == PROCS * n


def test_jsonl_torn_last_line_is_terminated_once(tmp_path, pool):
    """A crash left half a line. Writers that arrive together must terminate it once (the old code let each of
    them see the torn line and add its own newline) and lose no row to it."""
    for trial in range(25):
        path = tmp_path / f"feedback-{trial}.jsonl"
        path.write_bytes(b'{"w":-1,"i":0}\n{"torn":tr')
        start = time.time() + 0.01
        pool.map(_log_worker, [(path, w, 1, 0, start) for w in range(PROCS)], chunksize=1)
        lines = _lines(path)
        assert lines[1] == b'{"torn":tr'
        assert b"" not in lines              # terminated once, not once per writer
        assert len(lines) == 2 + PROCS
        assert len(list(JsonlLog(path).rows())) == 1 + PROCS


# ----------------------------------------------------------------------------- the registry

def test_registry_never_hands_out_a_version_twice(tmp_path, pool):
    home = tmp_path / "home"
    n = 8
    start = _soon()
    got = [v for vs in pool.map(_commit_worker, [(home, n, start) for _ in range(PROCS)]) for v in vs]
    assert [v for v in got if v.startswith("error")] == []
    assert len(set(got)) == PROCS * n
    reg = Registry(home).registry()
    assert set(reg["versions"]) == set(got)
    assert reg["next"] == PROCS * n + 1
    assert reg["latest"] == f"rlcd-0.1.{PROCS * n}"
    assert sorted(p.name for p in (home / "models").iterdir()) == sorted(got)
    assert [p.name for p in home.iterdir() if p.name.endswith(".tmp")] == []


def test_registry_history_keeps_every_promotion(tmp_path, pool):
    home = tmp_path / "home"
    reg = Registry(home)
    versions = [reg.commit({}, pack={"name": "p"}, feature_version="f1", encoder=None, policy={}) for _ in range(3)]
    rounds = 10
    before = len(reg.registry().get("history") or [])
    start = _soon()
    changed = before + sum(pool.map(_promote_worker, [(home, versions * rounds, start) for _ in range(PROCS)]))
    hist = Registry(home).registry()["history"]
    assert len(hist) == changed              # every change of `latest` is in the history, none overwritten
    for prev, cur in zip(hist, hist[1:], strict=False):
        assert cur["from"] == prev["to"]     # and the history is one line of succession


def test_registry_patch_commits_share_the_lock_and_the_history_with_trains(tmp_path, pool):
    """`rlcd patch` (commit_edit) in some processes while others train (commit): no version id is handed out twice,
    an edit is promoted only over the version it was judged against (checked inside the same lock as the write),
    and every move of `latest`, a promoted edit's included (kind "patch"), is in the history, one line of
    succession."""
    home = tmp_path / "home"
    first = Registry(home).commit({}, pack={"name": "p"}, feature_version="f1", encoder=None, policy={})
    n = 6
    start = _soon()
    jobs = [("edit" if w % 2 == 0 else "train", home, n, start) for w in range(PROCS)]
    results = pool.map(_train_or_edit_worker, jobs)
    assert [x for r in results for x in r if x.startswith("error")] == []
    edits = [json.loads(x) for r, job in zip(results, jobs, strict=True) if job[0] == "edit" for x in r]
    trains = [x for r, job in zip(results, jobs, strict=True) if job[0] == "train" for x in r]
    got = [first, *trains, *(e["version"] for e in edits)]
    assert len(set(got)) == len(got) == 1 + PROCS * n
    reg = Registry(home).registry()
    assert set(reg["versions"]) == set(got) and reg["next"] == len(got) + 1
    hist = reg["history"]
    for prev, cur in zip(hist, hist[1:], strict=False):
        assert cur["from"] == prev["to"]
    assert hist[-1]["to"] == reg["latest"]
    moved = {h["to"]: h for h in hist}
    assert {v for v, h in moved.items() if h["kind"] == "train"} == {first, *trains}
    for e in edits:
        if e["promoted"]:
            assert (moved[e["version"]]["kind"], moved[e["version"]]["from"]) == ("patch", e["base"])
        else:   # its base had moved on: kept, servable by name, and `latest` did not move
            assert e["version"] not in moved and reg["versions"][e["version"]]["edit"]["servable"] is True


def test_registry_sees_another_process_write(tmp_path):
    home = tmp_path / "home"
    a, b = Registry(home), Registry(home)
    kw = dict(pack={"name": "p"}, feature_version="f1", encoder=None, policy={})
    assert a.commit({}, **kw) == "rlcd-0.1.1"
    assert b.commit({}, **kw) == "rlcd-0.1.2"
    assert a.commit({}, **kw) == "rlcd-0.1.3"
    assert a.latest_id() == b.latest_id() == "rlcd-0.1.3"
    on_disk = json.loads((home / "registry.json").read_text())                      # the format is unchanged
    assert on_disk["next"] == 4 and sorted(on_disk["versions"]) == ["rlcd-0.1.1", "rlcd-0.1.2", "rlcd-0.1.3"]
