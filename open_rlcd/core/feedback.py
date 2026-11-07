"""Append-only logs under RLCD_HOME.

    feedback.jsonl    outcomes and corrections posted to /v1/feedback (the reinforcement loop). Each row becomes
                      an up-weighted training row at the next train; with a decision_id it also updates the
                      Beta-Bernoulli ledger at once.
    decisions.jsonl   what /v1/decide and /v1/evaluate answered, so feedback can refer to a decision_id.

Both are plain JSON lines (never pickles); a torn last line is skipped, never fatal. States are stored as sent
(they are re-featurised at the next train, so a feature-version change does not orphan them); set
`log_states: false` in rlcd.yaml to keep request contents out of decisions.jsonl. A feedback state is capped in
size (contract.MAX_FEEDBACK_STATE_BYTES) and the per-head counts /v1/heads shows are maintained incrementally
(only the bytes appended since the last count are read), so a long log does not slow the hot endpoints. The log
is never rotated by the service: archive it yourself after a train if it grows large.
"""
from __future__ import annotations

import json
import threading
import uuid
from collections.abc import Iterator
from pathlib import Path

from .training import now_iso


class JsonlLog:
    def __init__(self, path: Path):
        self.path = Path(path)
        self._lock = threading.RLock()

    def append(self, row: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock, open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, separators=(",", ":"), ensure_ascii=False, default=str) + "\n")

    def rows(self) -> Iterator[dict]:
        if not self.path.exists():
            return
        with open(self.path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError:
                        continue  # a torn last line must not break training


class FeedbackLog:
    def __init__(self, home: Path):
        self.feedback_log = JsonlLog(Path(home) / "feedback.jsonl")
        self.decision_log = JsonlLog(Path(home) / "decisions.jsonl")
        self._index: dict[str, dict] = {}
        self._counts: dict[str, int] = {}
        self._counted = 0                  # bytes of feedback.jsonl already counted
        self._count_id: tuple | None = None   # (device, inode) of the counted file: a replaced file is recounted
        self._count_lock = threading.Lock()

    def add_feedback(self, row: dict) -> dict:
        row = {"id": uuid.uuid4().hex, "received_at": now_iso(), **row}
        self.feedback_log.append(row)
        return row

    def feedback(self, head: str | None = None) -> list[dict]:
        return [r for r in self.feedback_log.rows() if head is None or r.get("head") == head]

    def feedback_counts(self) -> dict[str, int]:
        """Feedback rows per head. Incremental: reads only what was appended since the last call (a file that was
        replaced or truncated is counted again from the start); a torn last line waits for its newline."""
        path = self.feedback_log.path
        with self._count_lock:
            try:
                st = path.stat()
            except FileNotFoundError:
                self._counts, self._counted, self._count_id = {}, 0, None
                return {}
            ident = (st.st_dev, st.st_ino)
            if ident != self._count_id or st.st_size < self._counted:
                self._counts, self._counted, self._count_id = {}, 0, ident
            if st.st_size > self._counted:
                with self.feedback_log._lock, open(path, "rb") as f:
                    f.seek(self._counted)
                    pos = self._counted
                    for line in f:
                        if not line.endswith(b"\n"):   # count complete lines only
                            break
                        pos += len(line)
                        try:
                            h = json.loads(line).get("head", "?")
                        except (ValueError, AttributeError):
                            continue
                        self._counts[h] = self._counts.get(h, 0) + 1
                self._counted = pos
            return dict(self._counts)

    def add_decision(self, row: dict) -> str:
        did = uuid.uuid4().hex
        rec = {"decision_id": did, "created_at": now_iso(), **row}
        self.decision_log.append(rec)
        self._index[did] = rec
        return did

    def decision(self, decision_id: str) -> dict | None:
        if decision_id in self._index:
            return self._index[decision_id]
        found = None
        for r in self.decision_log.rows():
            if r.get("decision_id") == decision_id:
                found = r
        if found is not None:
            self._index[decision_id] = found
        return found
