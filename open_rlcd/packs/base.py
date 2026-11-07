"""Domain packs: what a business plugs in.

A pack = YAML (the questions: heads, classes, views, defaults) + a little Python (a FeatureProvider that turns
request `state` into feature rows, and a way to get labelled rows). Built-in packs: `documents`, `tabular`.
A custom pack is referenced from rlcd.yaml by a dotted path, `pack: my_pkg.my_pack:Pack`
(that imports your code: only point it at code you trust). See docs/writing-a-pack.md.

Only rlcd.yaml (the project's own configuration) may name code to import. A bundle's `pack.name` is data: the
service maps it to the project's configured pack (by the pack's `.name`) or to a built-in pack, never to an import.

Every head also accepts a precomputed state `{x, feature_names, feature_version}`; the service checks it
against the bundle and refuses a mismatch with a 422.
"""
from __future__ import annotations

import importlib
import json
from collections.abc import Callable, Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np
import yaml

from ..core.config import Project
from ..core.contract import ContractError
from ..core.dataset import Dataset
from ..core.head import HeadSpec

BUILTIN = {
    "documents": "open_rlcd.packs.documents:DocumentsPack",
    "tabular": "open_rlcd.packs.tabular:TabularPack",
}
# what to install when a built-in pack's optional dependencies are missing
EXTRAS = {"documents": "documents"}


@runtime_checkable
class FeatureProvider(Protocol):
    feature_version: str

    def names(self, view: str) -> list[str]: ...

    def transform(self, view: str, states: list[Any]) -> np.ndarray: ...

    def groups(self, view: str) -> dict[str, str]: ...

    def describe(self, view: str) -> dict: ...

    def to_config(self) -> dict: ...


class Pack(Protocol):
    name: str
    version: str

    def heads(self, project: Project) -> dict[str, HeadSpec]: ...

    def provider(self, project: Project, config: dict | None = None) -> FeatureProvider: ...

    def datasets(self, project: Project, provider: FeatureProvider, heads: list[str],
                 feedback: list[dict] | None = None, feedback_weight: float = 5.0) -> dict[str, Dataset]: ...


class BasePack:
    """Shared plumbing: head specs from the pack's YAML merged with rlcd.yaml overrides, optional hooks."""

    name = "base"
    version = "0.1.0"
    yaml_path: Path | None = None
    abstain = "escalate"
    rank_head: str | None = None          # default head of /v1/rank

    def defaults(self) -> dict:
        if self.yaml_path is None:
            return {}
        return yaml.safe_load(Path(self.yaml_path).read_text(encoding="utf-8")) or {}

    def heads(self, project: Project) -> dict[str, HeadSpec]:
        base = dict(self.defaults().get("heads") or {})
        overrides = project.head_overrides
        out: dict[str, HeadSpec] = {}
        for name in [*base, *[n for n in overrides if n not in base]]:
            d = {**(base.get(name) or {}), **(overrides.get(name) or {})}
            if d.pop("enabled", True) is False:
                continue
            out[name] = spec_from_yaml(name, d)
        return out

    def policy_defaults(self) -> dict:
        return dict(self.defaults().get("policy") or {"abstain": self.abstain})

    def info(self, provider: FeatureProvider | None = None) -> dict:
        return {"name": self.name, "version": self.version,
                "config": provider.to_config() if provider is not None else {}}

    # optional hooks; packs override what they support
    def decide(self, service: Any, request: dict) -> dict:
        raise NotImplementedError

    def evaluate(self, service: Any, request: dict) -> dict:
        raise NotImplementedError

    def ingest(self, project: Project, args: dict) -> dict:
        raise ContractError(f"pack '{self.name}' has no ingest step")

    def label_tasks(self, project: Project, provider: FeatureProvider | None = None) -> Iterable[Any]:
        return []

    def synth(self, seed: int = 7, n: int = 1000) -> dict:
        raise NotImplementedError

    def supports(self, hook: str) -> bool:
        return getattr(type(self), hook, None) is not getattr(BasePack, hook, None)


def spec_from_yaml(name: str, d: dict) -> HeadSpec:
    d = dict(d)
    levels = d.pop("levels", None)
    classes = d.pop("classes", None)
    if d.get("type") == "score" and levels and not classes:
        # score levels given as descriptions: classes are their indices
        classes = [str(i) for i in range(len(levels))]
    known = set(HeadSpec.__dataclass_fields__)
    extra = sorted(set(d) - known)
    if extra:
        raise ValueError(f"head {name!r}: unknown fields {extra}; known: {sorted(known)}")
    return HeadSpec(name=name, classes=tuple(str(c) for c in (classes or ())),
                    levels=tuple(str(x) for x in (levels or ())), **{k: v for k, v in d.items() if k != "name"})


def load_pack(name: str) -> BasePack:
    """A built-in pack by name, or a custom one by dotted path `module:Class`."""
    target = BUILTIN.get(name, name)
    if ":" not in target:
        raise ValueError(f"unknown pack {name!r}; built-in packs: {sorted(BUILTIN)}, or give 'module:Class'")
    mod, _, cls = target.partition(":")
    try:
        return getattr(importlib.import_module(mod), cls)()
    except ImportError as e:
        extra = EXTRAS.get(name)
        hint = (f" Install the pack's extra: `uv sync --extra {extra}` (or `pip install 'open-rlcd[{extra}]'`)."
                if extra else "" if name in BUILTIN else " Check that the module is importable (PYTHONPATH).")
        raise ImportError(f"pack {name!r} needs a module that is not installed ({e}).{hint}") from e


# ----------------------------------------------------------------------------- shared dataset building

def epoch(ts: Any) -> int | None:
    """Epoch seconds from epoch seconds / milliseconds or ISO-8601. Naive timestamps are read as UTC, so the same
    label file gives the same time order (and holdout cut) on every machine."""
    if ts is None or ts == "" or isinstance(ts, bool):
        return None
    if isinstance(ts, (int, float)) or (isinstance(ts, str) and ts.strip().lstrip("-").replace(".", "", 1).isdigit()):
        v = float(ts)
        return int(v / 1000) if abs(v) > 1e11 else int(v)
    try:
        dt = datetime.fromisoformat(str(ts).strip().replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return int(dt.timestamp())


def precomputed(state: Any) -> bool:
    return isinstance(state, dict) and "x" in state and "feature_version" in state


def _feedback_vector(state: dict, names: list[str], feature_version: str) -> np.ndarray | None:
    """The feature row of a precomputed feedback state, or None when it belongs to another feature schema (it
    cannot be re-featurised: only the vector was sent)."""
    x = state.get("x")
    if state.get("feature_version") != feature_version or not isinstance(x, (list, tuple)) or len(x) != len(names):
        return None
    if state.get("feature_names") is not None and list(state["feature_names"]) != list(names):
        return None
    try:
        return np.array([np.nan if v is None else float(v) for v in x], np.float32)
    except (TypeError, ValueError):
        return None


def _state_key(state: Any) -> str | None:
    """A stable text key of a JSON state (sorted keys), or None when it cannot be serialised."""
    try:
        return json.dumps(state, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return None


def dataset_from_labels(spec: HeadSpec, provider: FeatureProvider, labelled: list[dict],
                        feedback: list[dict] | None = None, feedback_weight: float = 5.0,
                        notes: list[str] | None = None,
                        group_of: Callable[[Any], str | None] | None = None) -> Dataset:
    """Rows for one head from the label store (`LabelStore.labelled`) plus feedback rows for that head.

    Trusted labels (human / ground_truth / synthetic) calibrate and are scored; judge labels only train;
    feedback rows (production outcomes posted to /v1/feedback with a raw state, or with precomputed features of
    the provider's current feature version) train with weight x `feedback_weight`. Only the latest feedback row
    per decision_id counts. Precomputed features of another feature version cannot be re-featurised and are
    skipped (with a note).

    Groups. A feedback row carries the group of the item it is about, so the split keeps that item out of the
    holdout (otherwise its exact twin would train the served model while being scored): `group_of(state)` when
    the pack gives one (it is also used for labelled items without a group), else the group of the labelled item
    with an identical state (stable JSON) or, failing that, an identical feature row, when exactly one group has
    it; otherwise the row is its own group `feedback:<decision_id>`. When no labelled item has a real group, the
    dataset has no group key at all (`group=None`), so timestamped items get time ordering instead of one
    singleton group per row."""
    states, y, src, judge_y, groups, ts, ids, w = [], [], [], [], [], [], [], []
    ok: list[bool] = []   # may this row be scored in the holdout (False: picked by an active-learning queue)
    real_group = False    # does any labelled item carry a group of its own
    skipped = 0
    for r in labelled:
        it = r["item"]
        try:
            yi = spec.label_index(r["label"])
        except ValueError:
            skipped += 1
            continue
        jy = -1
        if r.get("judge_label") is not None:
            try:
                jy = spec.label_index(r["judge_label"])
            except ValueError:
                jy = -1
        g = it.group or (group_of(it.state) if group_of is not None else None)
        real_group = real_group or bool(g)
        states.append(it.state)
        y.append(yi)
        src.append(r["source"])
        ok.append(not str(r.get("note") or "").startswith("sample=queue"))
        judge_y.append(jy if r["source"] != "judge" else -1)
        groups.append(g or it.id)
        ts.append(-1 if it.t is None else int(it.t))
        ids.append(it.id)
        w.append(1.0)
    n_items = len(states)
    n_fb = replaced = foreign = 0
    names = provider.names(spec.view)
    vectors: dict[int, np.ndarray] = {}   # row position -> precomputed feature row ({x, feature_version} feedback)
    fallback: dict[int, str] = {}         # row position -> the feedback row's own group when it links to no item
    fb_rows = [r for r in feedback or [] if r.get("head") == spec.name and r.get("state") is not None]
    # a decision's outcome counts once, as in the Bernoulli ledger: a retried or corrected POST /v1/feedback for
    # the same decision_id replaces the earlier row instead of adding weight
    last = {str(r["decision_id"]): i for i, r in enumerate(fb_rows) if r.get("decision_id")}
    for i, r in enumerate(fb_rows):
        if r.get("decision_id") and last[str(r["decision_id"])] != i:
            replaced += 1
            continue
        try:
            yi = spec.label_index(r.get("label"))
        except ValueError:
            skipped += 1
            continue
        if precomputed(r["state"]):
            v = _feedback_vector(r["state"], names, provider.feature_version)
            if v is None:
                foreign += 1
                continue
            vectors[len(states)] = v
        states.append(r["state"])
        y.append(yi)
        src.append("feedback")
        ok.append(True)
        judge_y.append(-1)
        groups.append(None)   # resolved below, once the feature rows exist
        fallback[len(states) - 1] = f"feedback:{r.get('decision_id') or r.get('id')}"
        t = epoch(r.get("time") or r.get("received_at"))
        ts.append(-1 if t is None else t)
        ids.append(f"feedback:{r.get('id')}")
        w.append(float(r.get("weight") or 1.0) * float(feedback_weight))
        n_fb += 1
    notes = list(notes or [])
    if skipped:
        notes.append(f"{skipped} labels skipped (not a valid label for head '{spec.name}')")
    if replaced:
        notes.append(f"{replaced} feedback rows replaced by a later outcome posted for the same decision_id")
    if foreign:
        notes.append(f"{foreign} feedback rows with precomputed features of another feature version or schema "
                     f"were skipped (expected '{provider.feature_version}', {len(names)} features)")
    if n_fb:
        notes.append(f"{n_fb} feedback rows added to training (weight x {feedback_weight:g})")
    if not states:
        return Dataset(X=np.zeros((0, len(names)), np.float32), y=np.zeros(0, int),
                       feature_names=names, feature_version=provider.feature_version,
                       notes=notes)
    X = np.empty((len(states), len(names)), np.float32)
    raw = [i for i in range(len(states)) if i not in vectors]
    if raw:
        X[raw] = provider.transform(spec.view, [states[i] for i in raw])
    for i, v in vectors.items():
        X[i] = v
    linked = _link_feedback_groups(states, X, groups, n_items, fallback, group_of)
    if linked and real_group:
        notes.append(f"{linked} feedback rows are about a labelled item: they share its group, so that item is never "
                     f"scored in the holdout while its outcome trains the model")
    has_t = any(t >= 0 for t in ts)
    return Dataset(X=X, y=np.asarray(y), feature_names=provider.names(spec.view),
                   feature_version=provider.feature_version, ids=np.asarray(ids, dtype=object),
                   t=np.asarray(ts) if has_t else None,
                   group=np.asarray(groups, dtype=object) if real_group else None,
                   weight=np.asarray(w), source=np.asarray(src, dtype=object),
                   judge_y=np.asarray(judge_y) if any(j >= 0 for j in judge_y) else None,
                   feature_groups=provider.groups(spec.view), holdout_ok=np.asarray(ok, bool), notes=notes)


def _link_feedback_groups(states: list[Any], X: np.ndarray, groups: list, n_items: int, fallback: dict[int, str],
                          group_of: Callable[[Any], str | None] | None) -> int:
    """Fill in the group of each feedback row (positions >= n_items, group None) in place; returns how many were
    linked to a labelled item's group (by `group_of`, an identical state or an identical feature row)."""
    by_state: dict[str, set] = {}
    by_row: dict[bytes, set] = {}
    for i in range(n_items):
        k = _state_key(states[i])
        if k is not None:
            by_state.setdefault(k, set()).add(groups[i])
        by_row.setdefault(np.ascontiguousarray(X[i]).tobytes(), set()).add(groups[i])
    known = set(groups[:n_items])
    linked = 0
    for i in range(n_items, len(states)):
        g = group_of(states[i]) if group_of is not None else None
        if not g:
            k = _state_key(states[i])
            cand = by_state.get(k) if k is not None else None
            if not cand or len(cand) != 1:
                cand = by_row.get(np.ascontiguousarray(X[i]).tobytes())
            g = next(iter(cand)) if cand and len(cand) == 1 else None
        if g and g in known:
            linked += 1
        groups[i] = g or fallback[i]
    return linked
