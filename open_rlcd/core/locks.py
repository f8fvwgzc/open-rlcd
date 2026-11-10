"""One writer at a time per file, across processes.

A server (`rlcd serve`) and the admin commands (`rlcd train`, `import`, `promote`, `rollback`) are separate
processes writing the same RLCD_HOME. Every read-modify-write of a file there happens inside a `FileLock`:

    self._file_lock = FileLock(path)          # locks the sidecar `<path>.lock`, never the data file
    with self._lock, self._file_lock:         # the object's own RLock first, then the file
        data = read(path)                     # read inside the lock: another process may have written
        ...
        write(path, data)                     # a unique temp file, then os.replace

The lock is an exclusive `flock` on a sidecar file, so the data file can be replaced atomically while it is held
(a lock on the data file itself would stay with the replaced inode). The sidecar is created on first use and never
removed: removing it would let two processes lock two different files of the same name. A lock dies with its
process, so a crash never leaves the home locked. Readers do not take it: they see the old file or the new one.

`FileLock` is re-entrant in a thread and exclusive between threads, like the `threading.RLock` it sits next to.
POSIX only (macOS, Linux): `flock` is advisory and local; do not put RLCD_HOME on a network share that ignores it.
"""
from __future__ import annotations

import os
import threading
from pathlib import Path

try:
    import fcntl
except ImportError:  # pragma: no cover - not a POSIX system
    fcntl = None  # type: ignore[assignment]


class LockUnavailable(RuntimeError):
    """This platform has no `fcntl.flock`: writing RLCD_HOME safely is not possible here."""


class FileLock:
    """Exclusive inter-process lock for the file at `path`, held through the sidecar `<path>.lock`."""

    def __init__(self, path: Path):
        path = Path(path)
        self.path = path.with_name(path.name + ".lock")
        self._thread = threading.RLock()
        self._depth = 0
        self._fd: int | None = None

    def _acquire(self) -> int:
        if fcntl is None:
            raise LockUnavailable(
                f"cannot lock {self.path}: this platform has no fcntl.flock. open-rlcd writes its home directory "
                "only on POSIX systems (macOS, Linux); on Windows run it under WSL.")
        flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_CLOEXEC", 0)
        try:
            fd = os.open(self.path, flags, 0o644)
        except FileNotFoundError:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            fd = os.open(self.path, flags, 0o644)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
        except BaseException:
            os.close(fd)
            raise
        return fd

    def __enter__(self) -> FileLock:
        self._thread.acquire()
        try:
            if self._depth == 0:
                self._fd = self._acquire()
            self._depth += 1
        except BaseException:
            self._thread.release()
            raise
        return self

    def __exit__(self, *exc) -> None:
        self._depth -= 1
        if self._depth == 0:
            fd, self._fd = self._fd, None
            if fd is not None:
                os.close(fd)             # closing the descriptor releases the lock
        self._thread.release()


def unique_tmp(path: Path) -> Path:
    """A temp file name next to `path` that no other process or thread uses: write it, then `os.replace` it."""
    path = Path(path)
    return path.with_name(f"{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")


def write_atomic(path: Path, text: str) -> None:
    """Replace `path` with `text` in one step: readers see the old file or the new one, never half of one."""
    tmp = unique_tmp(path)
    try:
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
