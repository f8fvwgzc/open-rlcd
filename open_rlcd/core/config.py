"""Project configuration: `rlcd.yaml` (written by `rlcd init`) plus the environment.

    name: my-business
    pack: documents                  # documents | tabular | a dotted path "my_pkg.my_pack:Pack"
    home: ./var                      # RLCD_HOME for this project (models, logs, labels, caches)
    pack_config: {...}               # pack-specific: encoder, data paths, connectors, label sources
    heads:                           # per-head overrides of the pack's defaults
      answerable: {stakes: high, conformal: 0.05, model: logreg, enabled: true}
    policy: {abstain: escalate, stakes: {medium: {act: 0.85, confirm: 0.65, min_accuracy: 0.75}}}
    train: {holdout_fraction: 0.2, n_splits: 5, ordering: auto, calibration: auto, feedback_weight: 5}
    labeling: {judge: claude-cli, model: null, batch: 20}
    serve: {host: 127.0.0.1, port: 8095, log_states: true, redact: true, receipts: decisions,
            review_capacity_per_day: 40}
    security: {allow_remote: false}
    rollout: {mode: full}            # shadow | assist | limited | full; limited: {stakes, heads}; heads: {head: mode}
    routes: {...}                    # route contracts of choice heads (core/routes.py)
    batteries: {...}                 # versioned question sets (core/battery.py)
    evaluate: {max_revisions: 2, output_scan: [secrets]}
    regressions: ./regressions.jsonl # regression cases for `rlcd regress` (default: regressions.jsonl in the project)
    owners: {pack: ..., heads: ..., policy: ..., review: ..., incidents: ...}   # read by `rlcd check`

Every key, its type and its default: docs/configuration.md. `rlcd config check` (core/config_check.py) reports
an unknown key, a wrong type or a value out of range before anything runs; `Project.check()` is the same call.

Environment: RLCD_HOME overrides `home`, RLCD_PROJECT points at a project directory or rlcd.yaml,
RLCD_API_KEY protects the API, RLCD_ALLOW_REMOTE=1 allows non-loopback connector / judge hosts.
Secrets (API keys, database passwords) are read from the environment, never from rlcd.yaml.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

CONFIG_NAME = "rlcd.yaml"


@dataclass
class Project:
    root: Path
    data: dict = field(default_factory=dict)
    path: Path | None = None
    home_override: Path | None = None    # set by Service(project, home=...): one home for registry and packs

    @property
    def name(self) -> str:
        return str(self.data.get("name") or self.root.name)

    @property
    def pack(self) -> str:
        return str(self.data.get("pack") or os.environ.get("RLCD_PACK") or "documents")

    @property
    def home(self) -> Path:
        if self.home_override is not None:
            return Path(self.home_override).expanduser().resolve()
        env = os.environ.get("RLCD_HOME")
        if env:
            return Path(env).expanduser().resolve()
        h = Path(str(self.data.get("home") or "var")).expanduser()
        return (h if h.is_absolute() else self.root / h).resolve()

    def section(self, name: str) -> dict:
        v = self.data.get(name)
        return dict(v) if isinstance(v, dict) else {}

    @property
    def pack_config(self) -> dict:
        return self.section("pack_config")

    @property
    def head_overrides(self) -> dict:
        """`heads:` of rlcd.yaml. A field written with no value (`model:`) is left out, so the pack's value stays, as
        if the line were not there (most keys read an empty value as their default; docs/configuration.md lists the
        main exceptions). `conformal:` is kept: an empty one is how a pack's promise is removed."""
        return {name: ({k: v for k, v in o.items() if v is not None or k == "conformal"} if isinstance(o, dict) else o)
                for name, o in self.section("heads").items()}

    @property
    def allow_remote(self) -> bool:
        return os.environ.get("RLCD_ALLOW_REMOTE") == "1" or bool(self.section("security").get("allow_remote"))

    def check(self, pack: Any = None) -> list[dict]:
        """Findings about this configuration, [{level: "error" | "warn", path, message}] (core/config_check.py)."""
        from .config_check import check_project

        return check_project(self, pack)

    def resolve_path(self, p: str | os.PathLike) -> Path:
        p = Path(p).expanduser()
        return (p if p.is_absolute() else self.root / p).resolve()

    def get(self, dotted: str, default: Any = None) -> Any:
        cur: Any = self.data
        for part in dotted.split("."):
            if not isinstance(cur, dict) or part not in cur:
                return default
            cur = cur[part]
        return cur


def find_config(start: Path | None = None) -> Path | None:
    """The rlcd.yaml to use: where RLCD_PROJECT points, else the nearest one in `start` (the current directory) or
    a parent; None when there is none. RLCD_PROJECT set to a path without an rlcd.yaml is an error, never "no
    project": a mistyped mount must not end up serving the default project (or the synthetic demo model)."""
    env = os.environ.get("RLCD_PROJECT")
    if env:
        p = Path(env).expanduser()
        p = p / CONFIG_NAME if p.is_dir() else p
        if not p.is_file():
            raise FileNotFoundError(f"RLCD_PROJECT={env} but there is no {CONFIG_NAME} at {p} (a mistyped path or a "
                                    f"missing mount?). Point RLCD_PROJECT at a project directory or its "
                                    f"{CONFIG_NAME}, or unset it")
        return p
    cur = (start or Path.cwd()).resolve()
    for d in (cur, *cur.parents):
        if (d / CONFIG_NAME).is_file():
            return d / CONFIG_NAME
    return None


def load_project(path: str | os.PathLike | None = None, overrides: dict | None = None) -> Project:
    """Load rlcd.yaml from `path` (a file or directory), RLCD_PROJECT, or the nearest parent directory. With no
    config at all, a default project rooted at the current directory (pack from RLCD_PACK, home ./var)."""
    cfg: Path | None
    if path is not None:
        p = Path(path).expanduser()
        cfg = p / CONFIG_NAME if p.is_dir() else p
        if not cfg.is_file():
            raise FileNotFoundError(f"no {CONFIG_NAME} at {cfg}")
    else:
        cfg = find_config()
    data: dict = {}
    if cfg is not None:
        loaded = yaml.safe_load(cfg.read_text(encoding="utf-8")) or {}
        if not isinstance(loaded, dict):
            raise ValueError(f"{cfg}: top level must be a mapping")
        data = loaded
    if overrides:
        data = _merge(data, overrides)
    root = cfg.parent.resolve() if cfg is not None else Path.cwd().resolve()
    return Project(root=root, data=data, path=cfg)


def _merge(a: dict, b: dict) -> dict:
    out = dict(a)
    for k, v in b.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def write_config(path: Path, data: dict, header: str = "") -> Path:
    path = Path(path)
    text = yaml.safe_dump(data, sort_keys=False, allow_unicode=True, width=110)
    path.write_text((header + "\n" if header else "") + text, encoding="utf-8")
    return path
