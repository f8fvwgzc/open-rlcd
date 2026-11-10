"""Rollout: how much automation the operator allows, independent of what the model could vouch for.

The policy decides what the evidence allows (thresholds, holdout verdict, conformal promise, Bernoulli gate). The
rollout decides what the *operator* allows, and can only take automation away: a decision's tier is
min(policy tier, rollout cap). `act` is the only automatic tier, so every cap below is "no act here".

    rollout:
      mode: full              # shadow | assist | limited | full
      limited:                # mode limited: act only inside this slice, confirm everywhere else
        stakes: [low]         # stakes levels that may act (default: [low])
        heads: [route]        # optional: only these heads may act (default: any head)
      heads:                  # optional per-head modes (a feature flag per question): override `mode`
        needs_graph: shadow

    shadow    log every decision, automate nothing: decisions are capped at confirm and marked `shadow`, so the
              application keeps its existing path and compares (the guide's "log decisions, change nothing")
    assist    recommend to people: capped at confirm (a person confirms every action)
    limited   automate a low-risk slice: act only for the listed stakes levels (and heads)
    full      no rollout cap (the default: the policy alone decides)

The cap is never the abstain action: abstaining stays the model's own statement ("I cannot vouch for this"), so
abstention rates are not polluted by rollout, and a plan that answers still answers (with a person confirming).

Kill switch: automation can be switched off at run time without a model update or a restart, from the CLI
(`rlcd automation off --reason ...`) or the API (POST /v1/automation). It caps every decision at confirm while
decisions are still made, logged and returned (observation continues). The switch is a JSON file in RLCD_HOME
(`automation.json`, every change is also appended to `automation.jsonl`), re-read when it changes. The
environment variable RLCD_AUTOMATION=off forces it off: then the API cannot switch it back on.
"""
from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .contract import ContractError
from .locks import FileLock, write_atomic

MODES = ("shadow", "assist", "limited", "full")
ENV_SWITCH = "RLCD_AUTOMATION"


@dataclass(frozen=True)
class Rollout:
    mode: str = "full"
    limited_stakes: tuple[str, ...] = ("low",)
    limited_heads: tuple[str, ...] | None = None
    head_modes: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        for where, m in (("rollout.mode", self.mode), *((f"rollout.heads.{h}", m) for h, m in self.head_modes)):
            if m not in MODES:
                raise ValueError(f"{where} must be one of {list(MODES)}, got {m!r}")

    @classmethod
    def from_config(cls, d: dict | None) -> Rollout:
        if d is not None and not isinstance(d, dict):
            raise ValueError("rollout must be a mapping {mode, limited?, heads?}")
        d = dict(d or {})
        mode = d.get("mode", "full")
        if not isinstance(mode, str) or not mode:
            # YAML 1.1 reads an unquoted off / no / false as false: a mode that is not a name is an error, never
            # silently `full` (the one mode without a cap)
            raise ValueError(f"rollout.mode must be one of {list(MODES)}, got {mode!r} (quote it in rlcd.yaml)")
        lim = d.get("limited", {})
        if lim is None:
            lim = {}
        if not isinstance(lim, dict):
            raise ValueError("rollout.limited must be a mapping {stakes: [...], heads: [...]}")
        stakes = lim.get("stakes", ["low"])
        if stakes is None:          # an explicit null, like an empty list, automates nothing
            stakes = []
        heads = lim.get("heads")
        if isinstance(stakes, str):
            stakes = [stakes]
        if isinstance(heads, str):
            heads = [heads]
        if not isinstance(stakes, (list, tuple)) or (heads is not None and not isinstance(heads, (list, tuple))):
            raise ValueError("rollout.limited.stakes and rollout.limited.heads must be lists of names")
        per_head = d.get("heads") or {}
        if not isinstance(per_head, dict):
            raise ValueError("rollout.heads maps a head name to its mode")
        for h, m in per_head.items():
            if not isinstance(m, str):
                raise ValueError(f"rollout.heads.{h} must be one of {list(MODES)}, got {m!r} (quote it in rlcd.yaml)")
        # an explicit empty list automates nothing (as `stakes: []` does); only an absent `heads` means any head
        return cls(mode=mode, limited_stakes=tuple(str(s) for s in stakes),
                   limited_heads=tuple(str(h) for h in heads) if heads is not None else None,
                   head_modes=tuple(sorted((str(h), str(m)) for h, m in per_head.items())))

    def mode_for(self, head: str | None) -> str:
        return dict(self.head_modes).get(head, self.mode) if head is not None else self.mode

    def to_dict(self) -> dict:
        out: dict[str, Any] = {"mode": self.mode}
        if self.mode == "limited" or any(m == "limited" for _, m in self.head_modes):
            out["limited"] = {"stakes": list(self.limited_stakes),
                              "heads": list(self.limited_heads) if self.limited_heads is not None else None}
        if self.head_modes:
            out["heads"] = dict(self.head_modes)
        return out

    def cap(self, *, stakes: str | None, head: str | None) -> tuple[int, str | None]:
        """(strongest tier this rollout allows for a decision of `head` at `stakes`, reason or None)."""
        mode = self.mode_for(head)
        scope = f"head '{head}' is in rollout mode" if mode != self.mode else "rollout mode"
        if mode == "full":
            return 2, None
        if mode == "shadow":
            return 1, f"{scope} shadow: decisions are logged and returned, nothing is automated"
        if mode == "assist":
            return 1, f"{scope} assist: the decision is a recommendation, a person confirms every action"
        # limited
        if stakes not in self.limited_stakes:
            return 1, (f"rollout mode limited: only stakes {list(self.limited_stakes)} are automated, this decision "
                       f"is at {stakes!r} stakes")
        if self.limited_heads is not None and head not in self.limited_heads:
            return 1, (f"rollout mode limited: only heads {list(self.limited_heads)} are automated, not "
                       f"{head!r}")
        return 2, None


@dataclass
class SwitchState:
    enabled: bool = True
    reason: str | None = None
    by: str | None = None
    at: str | None = None
    forced_by_env: bool = False

    def to_dict(self) -> dict:
        return {"enabled": self.enabled, "reason": self.reason, "by": self.by, "at": self.at,
                "forced_by_env": self.forced_by_env}


class AutomationSwitch:
    """The run-time kill switch of one RLCD_HOME. Cheap to consult on every request: the file is re-read only when
    its modification time or size changes."""

    def __init__(self, home: Path):
        self.path = Path(home) / "automation.json"
        self.audit = Path(home) / "automation.jsonl"
        self._lock = threading.Lock()
        self._seen: tuple[int, int] | None = None
        self._cached = SwitchState()

    @staticmethod
    def env_forced_off() -> bool:
        return (os.environ.get(ENV_SWITCH) or "").strip().lower() in ("off", "0", "false", "disabled", "no")

    def state(self) -> SwitchState:
        forced = self.env_forced_off()
        with self._lock:
            try:
                st = self.path.stat()
                key = (st.st_ino, st.st_mtime_ns, st.st_size)   # set() renames a new file in: a new inode
            except FileNotFoundError:
                key = None
            if key != self._seen:
                self._seen = key
                s = SwitchState()
                if key is not None:
                    try:
                        d = json.loads(self.path.read_text(encoding="utf-8"))
                        enabled = d.get("enabled", True)
                        if not isinstance(enabled, bool):   # "false" / "off" / 0 must never read as on
                            raise ValueError("enabled must be true or false")
                        s = SwitchState(enabled=enabled, reason=d.get("reason"), by=d.get("by"), at=d.get("at"))
                    except (OSError, ValueError, AttributeError):
                        # an unreadable switch file fails safe: automation off, and the reason says why
                        s = SwitchState(enabled=False, reason="automation.json is unreadable: automation stays off "
                                                              "until it is rewritten (rlcd automation on|off)")
                self._cached = s
            cur = self._cached
        if forced:
            return SwitchState(enabled=False, reason=f"{ENV_SWITCH}=off in the server's environment",
                               by="environment", at=cur.at, forced_by_env=True)
        return SwitchState(enabled=cur.enabled, reason=cur.reason, by=cur.by, at=cur.at)

    def set(self, enabled: bool, *, reason: str | None = None, by: str | None = None, at: str) -> SwitchState:
        if enabled and self.env_forced_off():
            raise ContractError(f"automation is forced off by {ENV_SWITCH}=off in the server's environment: it can "
                                f"only be switched on by changing that variable", kind="automation_locked",
                                status=409)
        rec = {"enabled": bool(enabled), "reason": reason, "by": by, "at": at}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # the server and `rlcd automation` are separate processes: the state and its audit line change together
        with self._lock, FileLock(self.path):
            write_atomic(self.path, json.dumps(rec, ensure_ascii=False))
            with open(self.audit, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            self._seen = None
        return self.state()

    def history(self, limit: int = 50) -> list[dict]:
        if limit <= 0 or not self.audit.exists():
            return []
        rows = []
        for line in self.audit.read_text(encoding="utf-8").splitlines():
            try:
                rows.append(json.loads(line))
            except ValueError:
                continue
        return rows[-limit:]

    def cap(self) -> tuple[int, str | None]:
        s = self.state()
        if s.enabled:
            return 2, None
        why = f"automation is switched off ({s.reason})" if s.reason else "automation is switched off"
        return 1, f"{why}: decisions are capped at confirm until it is switched back on"


def effective_cap(rollout: Rollout, switch: AutomationSwitch | None, *, stakes: str | None,
                  head: str | None) -> tuple[int, list[str]]:
    """The strongest tier allowed for one decision and the reasons for every cap below act."""
    cap, reasons = 2, []
    c, why = rollout.cap(stakes=stakes, head=head)
    if c < cap:
        cap = c
    if why:
        reasons.append(why)
    if switch is not None:
        c, why = switch.cap()
        if c < cap:
            cap = c
        if why:
            reasons.append(why)
    return cap, reasons
