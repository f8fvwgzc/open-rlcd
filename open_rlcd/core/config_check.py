"""Check rlcd.yaml before anything runs on it (`rlcd config check`, `rlcd config show`).

A typo in rlcd.yaml used to be ignored (`polcy:`, `train: {holdout_fracton: 0.3}`) or to fail far from its cause
(`rollout: {mode: shaddow}` at the first request). `check_project` reads the whole file once and returns findings,

    {"level": "error" | "warn", "path": "rollout.mode", "message": "must be one of ..., got 'shaddow' ..."}

    error   the file cannot mean what was written: an unknown key, a wrong type, a value outside its range, a
            section one of the validators below refuses. `rlcd train` and `rlcd serve` refuse to start on one.
    warn    it works, and deserves a look: a value that looks like a secret, a name nothing matches, a key that
            is ignored.

The keys come from `schema()`: one `Key` per key the code reads, with its type, default and allowed values
(docs/configuration.md lists the same keys; tests/test_config_check.py keeps the two equal). A pack declares its
own `pack_config` keys with `BasePack.config_keys()`; a custom pack that declares nothing is not checked there.

Where a section already has a validator, that code decides, so the check and the service cannot disagree:
`policy.PolicyConfig.from_dict` (thresholds, overrides), `rollout.Rollout.from_config`,
`routes.RouteBook.from_config`, `battery.load`, `redact.kinds_from_config` (serve.redact, evaluate.output_scan),
`review.max_revisions`, `regress.cases_path`, and the pack's own `heads()` / `check_config()`.

What this does not do: it does not open the documents folder, the query log or the training table's rows on its
own (the pack's `heads()` may read the table to learn a head's classes, as every request does); it does not
check a trained model against the file (`rlcd check` does); and the secret scan only knows well-known shapes and
key names (core/redact.py), so a bare random string under an innocent key is not noticed.
"""
from __future__ import annotations

import difflib
import functools
import math
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

from . import battery as BT
from . import policy as PL
from .calibration import METHODS
from .config import CONFIG_NAME, Project, _merge
from .contract import QUESTION_TYPES, ContractError
from .head import DECISIONS, MODELS, HeadSpec
from .redact import _secret_key, kinds_from_config, redact, redact_text
from .rollout import ENV_SWITCH, Rollout
from .rollout import MODES as ROLLOUT_MODES
from .routes import RouteBook
from .splits import MODES as ORDERINGS

LEVELS = ("error", "warn")
JUDGE_KINDS = ("claude-cli", "anthropic", "ollama", "openai")
OWNER_ROLES = ("pack", "heads", "policy", "review", "incidents")       # core/checklist.py OWNER_ROLES
RECEIPT_MODES = ("decisions", "all")                                   # core/service.py RECEIPT_MODES
BATTERY_FIELDS = ("version", "owner", "description", "changelog", "consumers", "questions")   # core/battery.py
REDACT_KINDS = ("secrets", "pii")
# environment variables open-rlcd reads; the second field says whether the value is a secret (never printed)
ENVIRONMENT = (("RLCD_PROJECT", False), ("RLCD_HOME", False), ("RLCD_PACK", False), ("RLCD_ALLOW_REMOTE", False),
               (ENV_SWITCH, False), ("RLCD_MAX_BODY_BYTES", False), ("RLCD_URL", False), ("RLCD_DEBUG", False),
               ("HF_HUB_OFFLINE", False), ("RLCD_API_KEY", True), ("RLCD_ADMIN_KEY", True),
               ("ANTHROPIC_API_KEY", True), ("OPENAI_API_KEY", True))
_ENV_REF = re.compile(r"^\$\{?[A-Za-z_][A-Za-z0-9_]*\}?$")
_BOOL_WORDS = "YAML reads an unquoted yes / no / on / off / true / false as a boolean: quote it"


# ----------------------------------------------------------------------------- the schema

@dataclass(frozen=True)
class Key:
    """One key of rlcd.yaml: what may be written there.

    kind      str | bool | int | number | names (a name or a list of names) | list | map | scalar | any
    default   what the code uses when the key is absent (documentation; None: no default)
    choices   the allowed values of an enum
    min, max  the range of a number (`open_min` / `open_max`: the bound itself is not allowed)
    nullable  an explicit `key:` / `key: null` means "use the default" (False: the code would crash on it)
    strict    a number written as a string ("0.9") is refused, because the code that reads it does not convert
    keys      map: the fixed keys inside
    each      map: every value under a free name (a head, a stakes level, a battery) has this shape; `label`
              is the placeholder the reference shows, e.g. heads.<head>.stakes
    unknown   what a key outside `keys` is: "error", "warn" (read by nothing), or "open" (the map belongs to the
              caller; only a near miss of a known key is reported, as a warning)
    """
    kind: str = "any"
    default: Any = None
    choices: tuple = ()
    min: float | None = None
    max: float | None = None
    open_min: bool = False
    open_max: bool = False
    nullable: bool = True
    strict: bool = False
    keys: dict[str, Key] | None = None
    each: Key | None = None
    label: str = "name"
    unknown: str = "error"

    def type_text(self) -> str:
        if self.choices:
            return " | ".join(str(c) for c in self.choices)
        text = {"str": "text", "bool": "true / false", "int": "whole number", "number": "number",
                "names": "name or list of names", "list": "list", "map": "mapping", "scalar": "text",
                "any": "any"}[self.kind]
        rng = self.range_text()
        return f"{text} {rng}" if rng else text

    def range_text(self) -> str:
        lo, hi = self.min, self.max
        if lo is None and hi is None:
            return ""
        if lo is not None and hi is not None:
            return f"{'(' if self.open_min else '['}{lo:g}, {hi:g}{')' if self.open_max else ']'}"
        if lo is not None:
            return f"{'>' if self.open_min else '>='} {lo:g}"
        return f"{'<' if self.open_max else '<='} {hi:g}"


def head_keys() -> dict[str, Key]:
    """The fields of one `heads.<head>` override: every HeadSpec field except the name, plus `enabled`."""
    typed = {
        "type": Key("str", choices=QUESTION_TYPES),
        "classes": Key("list"),
        "levels": Key("list"),
        "view": Key("str", default="default"),
        "model": Key("str", default="logreg", choices=MODELS),
        "decision": Key("str", default="argmax", choices=DECISIONS),
        "stakes": Key("str", default="medium"),
        "conformal": Key("number", min=0, max=0.5, open_min=True, open_max=True, strict=True),
        "description": Key("str", default=""),
        "input": Key("map"),
        "params": Key("map"),
        "positive": Key("any"),
    }
    out = {f: typed.get(f, Key("any")) for f in HeadSpec.__dataclass_fields__ if f != "name"}
    out["enabled"] = Key("bool", default=True)
    return out


@functools.cache
def schema() -> dict[str, Key]:
    """Every key of rlcd.yaml the code reads, outside `pack_config` (which belongs to the pack)."""
    p01 = {"min": 0, "max": 1, "strict": True, "nullable": False}
    level = Key("map", nullable=False, label="level", keys={
        "act": Key("number", **p01), "confirm": Key("number", **p01), "min_accuracy": Key("number", **p01)})
    contract = Key("map", label="option", unknown="open", keys={
        "enabled": Key("bool", default=True), "requires": Key("names"), "allowed_data": Key("names"),
        "latency_ms": Key("number", strict=True), "fallback": Key("str")})
    return {
        "name": Key("scalar"),
        "pack": Key("str", default="documents"),
        "home": Key("str", default="var"),
        "pack_config": Key("map"),
        "heads": Key("map", each=Key("map", label="head", keys=head_keys())),
        "policy": Key("map", keys={
            "abstain": Key("str", default="escalate"),
            "stakes": Key("map", each=level),
            "min_verify_rows": Key("int", default=30, min=0, strict=True, nullable=False),
            "gate_min_p_above": Key("number", default=0.9, min=0, max=1, strict=True, nullable=False),
            "miscalibration_p": Key("number", default=0.05, min=0, max=1, strict=True, nullable=False),
            "overrides": Key("map"),
        }),
        "train": Key("map", keys={
            # the ranges of holdout_fraction, n_splits, n_estimators and feedback_weight are the bounds POST /v1/train
            # enforces (core/contract.py TrainRequest), so the file and the API accept the same values
            "holdout_fraction": Key("number", default=0.2, min=0.05, max=0.5, nullable=False),
            "n_splits": Key("int", default=5, min=2, max=20, nullable=False),
            "ordering": Key("str", default="auto", choices=ORDERINGS, nullable=False),
            "gap": Key("int", default=0, min=0, nullable=False),
            "calibration": Key("str", default="auto", choices=("auto", *METHODS), nullable=False),
            "isotonic_min_per_class": Key("int", default=3000, min=1, nullable=False),
            "conformal_delta": Key("number", default=0.1, min=0, max=1, open_min=True, open_max=True, nullable=False),
            "n_estimators": Key("int", default=300, min=10, max=3000, nullable=False),
            "min_trusted": Key("int", default=30, min=1, nullable=False),
            "seed": Key("int", default=7, nullable=False),
            "feedback_weight": Key("number", default=5, min=0, max=1000, open_min=True, nullable=False),
        }),
        "labeling": Key("map", keys={
            "judge": Key("str", default="claude-cli"),
            "model": Key("str"),
            "batch": Key("int", default=20, min=1, nullable=False),
            "timeout": Key("number", default=300, min=0, open_min=True),
            "executable": Key("str", default="claude"),
            "base_url": Key("str", default="http://127.0.0.1:11434"),
        }),
        "serve": Key("map", keys={
            "host": Key("str", default="127.0.0.1", nullable=False),
            "port": Key("int", default=8095, min=0, max=65535, nullable=False),
            "log_states": Key("bool", default=True),
            "redact": Key("any", default=True),
            "receipts": Key("str", default="decisions", choices=RECEIPT_MODES),
            "review_capacity_per_day": Key("number", min=0, open_min=True, strict=True),
        }),
        "security": Key("map", keys={"allow_remote": Key("bool", default=False)}),
        "rollout": Key("map", keys={
            "mode": Key("str", default="full", choices=ROLLOUT_MODES, nullable=False),
            "limited": Key("map", keys={"stakes": Key("names", default=["low"]), "heads": Key("names")}),
            "heads": Key("map", each=Key("str", choices=ROLLOUT_MODES, nullable=False, label="head")),
        }),
        "routes": Key("map", each=Key("map", label="head", nullable=False, each=contract)),
        "batteries": Key("map", each=Key("map", label="battery", nullable=False,
                                         keys={f: Key("any") for f in BATTERY_FIELDS})),
        "evaluate": Key("map", keys={
            "max_revisions": Key("int", default=2, min=0, nullable=False),
            "output_scan": Key("any", default=["secrets"]),
        }),
        "regressions": Key("str", default="regressions.jsonl"),
        "owners": Key("map", unknown="warn", keys={r: Key("names") for r in OWNER_ROLES}),
    }


def flatten(keys: dict[str, Key], prefix: str = "") -> dict[str, Key]:
    """{dotted path: Key} for every key a person can write: `train.seed`, `heads.<head>.stakes`, ..."""
    out: dict[str, Key] = {}
    for name, key in keys.items():
        out.update(_flatten(key, f"{prefix}{name}"))
    return out


def _flatten(key: Key, path: str) -> dict[str, Key]:
    if key.keys is not None:
        return flatten(key.keys, path + ".")
    if key.each is not None:
        return _flatten(key.each, f"{path}.<{key.each.label}>")
    return {path: key}


def known_keys(pack: Any = None) -> list[str]:
    """Every documented key, sorted: the core keys, or (with a pack) the pack's `pack_config.*` keys."""
    if pack is None:
        return sorted(flatten(schema()))
    declared = _pack_keys(pack)
    return sorted(flatten(declared, "pack_config.")) if declared is not None else []


def reference(pack: Any = None) -> list[dict]:
    """[{key, type, default}] for docs/configuration.md, from the same schema the check uses."""
    keys = flatten(schema()) if pack is None else flatten(_pack_keys(pack) or {}, "pack_config.")
    return [{"key": k, "type": v.type_text(), "default": v.default} for k, v in keys.items()]


# ----------------------------------------------------------------------------- findings

def _finding(level: str, path: str, message: str) -> dict:
    return {"level": level, "path": path, "message": message}


def errors(findings: list[dict]) -> list[dict]:
    return [f for f in findings if f["level"] == "error"]


def warnings(findings: list[dict]) -> list[dict]:
    return [f for f in findings if f["level"] == "warn"]


def format_findings(findings: list[dict]) -> list[str]:
    """One line per finding: `error  rollout.mode: must be one of ...` (errors first, in file order)."""
    ordered = errors(findings) + warnings(findings)
    return [f"{f['level']:<5}  {f['path']}: {f['message']}" for f in ordered]


def summary(findings: list[dict]) -> str:
    e, w = len(errors(findings)), len(warnings(findings))
    if not e and not w:
        return "no problems found"
    return f"{e} error{'' if e == 1 else 's'}, {w} warning{'' if w == 1 else 's'}"


def _has_error(out: list[dict], prefix: str) -> bool:
    return any(f["level"] == "error" and (f["path"] == prefix or f["path"].startswith(prefix + "."))
               for f in out)


def _suggest(word: Any, options: Any, cutoff: float = 0.6) -> str | None:
    hit = difflib.get_close_matches(str(word), [str(o) for o in options], n=1, cutoff=cutoff)
    return hit[0] if hit else None


def _one_of(value: Any, options: Any) -> str:
    text = f"must be one of {', '.join(str(o) for o in options)}; got {value!r}"
    if isinstance(value, bool):
        return f"{text} ({_BOOL_WORDS})"
    hint = _suggest(value, options)
    return f"{text}. Did you mean {hint!r}?" if hint else text


# ----------------------------------------------------------------------------- the generic walk

def _number(v: Any, strict: bool) -> float | int | None:
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return None if isinstance(v, float) and not math.isfinite(v) else v
    if not strict and isinstance(v, str):
        try:
            n = float(v.strip())
        except ValueError:
            return None
        return (int(n) if n.is_integer() else n) if math.isfinite(n) else None
    return None


def _names_ok(v: Any) -> bool:
    return isinstance(v, str) or (isinstance(v, (list, tuple)) and all(isinstance(x, str) for x in v))


def _check_keys(value: dict, keys: dict[str, Key], path: str, out: list[dict], unknown: str = "error",
                elsewhere: dict[str, str] | None = None) -> None:
    for k, v in value.items():
        name = str(k)
        p = f"{path}.{name}" if path else name
        if name in keys and isinstance(k, str):
            _check_value(v, keys[name], p, out)
            continue
        if unknown == "open":
            hint = _suggest(name, keys, cutoff=0.8)
            if hint:
                out.append(_finding("warn", p, f"is not a key open-rlcd reads here (it is passed through as "
                                               f"written). Did you mean {hint!r}?"))
            continue
        hint = _suggest(name, keys)
        if hint:
            tail = f". Did you mean {hint!r}?"
        elif elsewhere and name in elsewhere:
            tail = f". Did you mean {elsewhere[name]!r}? (it belongs one level down)"
        else:
            tail = f"; the keys here are: {', '.join(keys)}"
        if isinstance(k, bool):
            tail += f" ({_BOOL_WORDS})"
        if unknown == "warn":
            out.append(_finding("warn", p, f"nothing reads this key{tail}"))
        else:
            out.append(_finding("error", p, f"unknown key{tail}"))


def _check_value(v: Any, key: Key, path: str, out: list[dict]) -> None:
    def err(message: str) -> None:
        out.append(_finding("error", path, message))

    if v is None:
        if not key.nullable:
            err("is empty: give it a value, or remove the line to get the default"
                + (f" ({key.default!r})" if key.default is not None else ""))
        return
    kind = key.kind
    if kind == "map" or (kind == "any" and isinstance(v, dict) and (key.keys is not None or key.each is not None)):
        if not isinstance(v, dict):
            err(f"must be a mapping (indented `key: value` lines), got {_shown(v)}")
            return
        if key.keys is not None:
            _check_keys(v, key.keys, path, out, key.unknown)
        elif key.each is not None:
            for name, sub in v.items():
                _check_value(sub, key.each, f"{path}.{name}", out)
        return
    if kind == "bool":
        if not isinstance(v, bool):
            err(f"must be true or false (without quotes), got {_shown(v)}")
        return
    if kind in ("int", "number"):
        n = _number(v, key.strict)
        if n is None:
            what = "a whole number" if kind == "int" else "a number"
            err(f"must be {what}, got {_shown(v)}" + (" (write it without quotes)" if isinstance(v, str) else ""))
            return
        if kind == "int" and not float(n).is_integer():
            err(f"must be a whole number, got {n!r}")
            return
        lo, hi = key.min, key.max
        below = lo is not None and (n <= lo if key.open_min else n < lo)
        above = hi is not None and (n >= hi if key.open_max else n > hi)
        if below or above:
            rng = key.range_text()
            err(f"must be {'in ' if lo is not None and hi is not None else ''}{rng}, got {n!r}")
        return
    if kind == "names":
        if not _names_ok(v):
            err(f"must be a name or a list of names, got {_shown(v)}")
        return
    if kind == "list":
        if not isinstance(v, (list, tuple)):
            err(f"must be a list, got {_shown(v)}")
        return
    if kind == "scalar":
        if isinstance(v, (dict, list)):
            err(f"must be a single value, got {_shown(v)}")
        return
    if kind == "str":
        if key.choices:
            if not isinstance(v, str) or v not in key.choices:
                err(_one_of(v, key.choices))
            return
        if not isinstance(v, str):
            err(f"must be text, got {_shown(v)}" + (f" ({_BOOL_WORDS})" if isinstance(v, bool) else ""))


def _shown(v: Any) -> str:
    """A value for an error message: its type for containers, its repr (cut) for scalars."""
    if isinstance(v, dict):
        return "a mapping"
    if isinstance(v, (list, tuple)):
        return "a list"
    text = repr(v)
    return text if len(text) <= 40 else text[:37] + "..."


# ----------------------------------------------------------------------------- packs

def _pack_keys(pack: Any) -> dict[str, Key] | None:
    """What a pack declares with `config_keys()`: {key: Key}; a value that is not a Key means "known, any value"."""
    fn = getattr(pack, "config_keys", None)
    declared = fn() if callable(fn) else None
    if declared is None:
        return None
    if not isinstance(declared, dict):
        declared = {str(k): None for k in declared}
    return {str(k): v if isinstance(v, Key) else Key("any") for k, v in declared.items()}


def _check_pack_name(project: Project, out: list[dict]) -> None:
    from ..packs.base import BUILTIN

    raw = project.data.get("pack")
    if raw is None or not isinstance(raw, str):
        return
    if raw in BUILTIN:
        return
    mod, sep, cls = raw.partition(":")
    if sep and mod.strip() and cls.strip():
        return
    hint = _suggest(raw, BUILTIN)
    out.append(_finding("error", "pack", f"unknown pack {raw!r}: use a built-in pack ({', '.join(sorted(BUILTIN))}) "
                                         f"or your own as module:Class"
                                         + (f". Did you mean {hint!r}?" if hint else "")))


def _load_pack(project: Project, out: list[dict]) -> Any:
    from ..packs.base import load_pack

    raw = project.data.get("pack")
    if _has_error(out, "pack") or (raw is not None and not isinstance(raw, str)):
        return None
    try:
        return load_pack(project.pack, root=project.root)
    except Exception as e:
        out.append(_finding("warn", "pack", f"the pack could not be loaded, so pack_config and heads were not "
                                            f"checked against it ({type(e).__name__}: {str(e)[:300]})"))
        return None


# ----------------------------------------------------------------------------- sections

def _policy(project: Project, pack: Any, out: list[dict]) -> list[str]:
    """Checks `policy`; returns the names of the stakes levels the project ends up with."""
    defaults = PL.default_stakes()
    pack_policy: dict = {}
    if pack is not None and callable(getattr(pack, "policy_defaults", None)):
        try:
            got = pack.policy_defaults()
            pack_policy = dict(got) if isinstance(got, dict) else {}
        except Exception:
            pack_policy = {}
    section = project.section("policy")
    merged = _merge(pack_policy, section)
    stakes = merged.get("stakes") if isinstance(merged.get("stakes"), dict) else {}
    names = list(dict.fromkeys([*defaults, *(str(n) for n in stakes)]))
    abstain = section.get("abstain")
    if isinstance(abstain, str) and (not abstain.strip() or abstain in PL.TIERS[1:]):
        out.append(_finding("error", "policy.abstain",
                            f"names the abstaining action (escalate, hold, ...); it cannot be {abstain!r}: a caller "
                            f"would read an abstention as {abstain or 'nothing'}"))
    if _has_error(out, "policy.stakes"):
        return names
    for name, lv in stakes.items():
        if not isinstance(lv, dict):
            continue
        full = {**(asdict(defaults[name]) if name in defaults else {}), **lv}
        missing = [k for k in ("act", "confirm", "min_accuracy") if k not in full]
        if missing:
            out.append(_finding("error", f"policy.stakes.{name}", f"a new stakes level needs act, confirm and "
                                                                  f"min_accuracy; missing: {', '.join(missing)}"))
        elif full["confirm"] > full["act"]:
            out.append(_finding("error", f"policy.stakes.{name}",
                                f"confirm ({full['confirm']:g}) must not be above act ({full['act']:g}): a decision "
                                f"sure enough to act is also sure enough to confirm"))
    if _has_error(out, "policy"):
        return names
    try:
        PL.PolicyConfig.from_dict(merged)
    except (ValueError, TypeError) as e:
        msg = str(e)
        where = "policy.overrides" if msg.startswith("policy.overrides") else "policy"
        out.append(_finding("error", where, msg))
    return names


def _heads(project: Project, levels: list[str], out: list[dict]) -> None:
    """What the generic walk cannot know about `heads`: the stakes levels this project's policy defines."""
    raw = project.data.get("heads")
    if not isinstance(raw, dict):
        return
    for name, o in raw.items():
        if not isinstance(o, dict):
            continue
        stakes = o.get("stakes")
        if isinstance(stakes, str) and stakes not in levels:
            out.append(_finding("error", f"heads.{name}.stakes", _one_of(stakes, levels)))


def _new_heads(project: Project, pack: Any, out: list[dict]) -> None:
    """A name under `heads:` that the pack does not define is a new head and needs a `type`; far more often it
    is a misspelled name. Only for packs that build their heads the stock way (BasePack.heads)."""
    from ..packs.base import BasePack

    raw = project.data.get("heads")
    if not isinstance(raw, dict) or not isinstance(pack, BasePack) or type(pack).heads is not BasePack.heads:
        return
    try:
        base = dict(pack.defaults().get("heads") or {})
    except Exception:
        return
    for name, o in raw.items():
        if name in base or (isinstance(o, dict) and o.get("type") is not None) or _has_error(out, f"heads.{name}"):
            continue
        hint = _suggest(name, base)
        out.append(_finding("error", f"heads.{name}",
                            f"is not a head of the {pack.name} pack ({', '.join(base) or 'it defines none'})"
                            + (f". Did you mean {hint!r}?" if hint else "")
                            + " A head of your own needs at least `type`"))


def _head_schema(extra: set[str]) -> Key:
    keys = head_keys()
    keys.update({k: Key("any") for k in extra if k not in keys})
    return Key("map", label="head", keys=keys)


def _rollout(project: Project, levels: list[str], out: list[dict]) -> Rollout | None:
    raw = project.data.get("rollout")
    if raw is None or _has_error(out, "rollout"):
        return None
    try:
        ro = Rollout.from_config(raw)
    except (ValueError, TypeError) as e:
        out.append(_finding("error", "rollout", str(e)))
        return None
    limited_used = ro.mode == "limited" or any(m == "limited" for _, m in ro.head_modes)
    if isinstance(raw, dict) and raw.get("limited") is not None and not limited_used:
        out.append(_finding("warn", "rollout.limited", f"is only read in mode limited; rollout.mode is {ro.mode!r}, "
                                                       f"so this slice changes nothing"))
    if limited_used:
        for s in ro.limited_stakes:
            if s not in levels:
                hint = _suggest(s, levels)
                out.append(_finding("warn", "rollout.limited.stakes",
                                    f"{s!r} is not a stakes level of this project ({', '.join(levels)}), so nothing "
                                    f"is automated for it" + (f". Did you mean {hint!r}?" if hint else "")))
        if not ro.limited_stakes:
            out.append(_finding("warn", "rollout.limited.stakes", "is empty: mode limited then automates nothing"))
    return ro


def _routes(project: Project, out: list[dict]) -> RouteBook | None:
    raw = project.data.get("routes")
    if raw is None or _has_error(out, "routes"):
        return None
    try:
        return RouteBook.from_config(raw)
    except (ValueError, TypeError) as e:
        msg = str(e)
        m = re.match(r"(routes(?:\.[^\s:]+)*)", msg)
        out.append(_finding("error", m.group(1) if m else "routes", msg))
        return None


def _batteries(project: Project, pack: Any, out: list[dict]) -> None:
    raw = project.data.get("batteries")
    if raw is None or _has_error(out, "batteries"):
        return
    defaults: Any = None
    if pack is not None and callable(getattr(pack, "defaults", None)):
        try:
            defaults = (pack.defaults() or {}).get("batteries")
        except Exception:
            defaults = None
    try:
        BT.load(defaults if isinstance(defaults, dict) else None, raw)
    except (ValueError, TypeError) as e:
        out.append(_finding("error", "batteries", str(e)))
    except ContractError as e:
        out.append(_finding("error", "batteries", e.message))


def _kinds(project: Project, dotted: str, out: list[dict]) -> None:
    """serve.redact and evaluate.output_scan share one reader: true, false, or a list of secrets / pii."""
    value = project.get(dotted)
    section = dotted.split(".")[0]
    if value is None or not isinstance(project.data.get(section), dict):
        return
    try:
        kinds_from_config(value)
    except (ValueError, TypeError):
        bad = value if isinstance(value, (list, tuple)) else [value]
        hints = [h for h in (_suggest(b, REDACT_KINDS) for b in bad if b not in REDACT_KINDS) if h]
        got = repr(value) if isinstance(value, list) and len(repr(value)) <= 60 else _shown(value)
        out.append(_finding("error", dotted, f"must be true, false or a list of {' / '.join(REDACT_KINDS)}; got {got}"
                                             + (f". Did you mean {hints[0]!r}?" if hints else "")))


def _labeling(project: Project, out: list[dict]) -> None:
    lab = project.section("labeling")
    judge = lab.get("judge")
    if not isinstance(judge, str) or _has_error(out, "labeling.judge"):
        return
    kind, _, arg = judge.strip().partition(":")
    kind, arg = kind.strip().lower(), arg.strip()
    forms = "claude-cli[:model], anthropic[:model], ollama:<model>, openai:<base_url>|<model>"
    if kind not in JUDGE_KINDS:
        hint = _suggest(kind, JUDGE_KINDS)
        out.append(_finding("error", "labeling.judge", f"unknown judge {judge!r}; use one of {forms}"
                                                       + (f". Did you mean {hint!r}?" if hint else "")))
    elif kind == "ollama" and not arg and not lab.get("model"):
        out.append(_finding("error", "labeling.judge", "ollama needs a model: write ollama:<model>, or set "
                                                       "labeling.model"))
    elif kind == "openai":
        base, sep, model = (arg or (str(lab.get("model") or ""))).rpartition("|")
        if not sep or not base.strip() or not model.strip():
            out.append(_finding("error", "labeling.judge", "openai needs a base URL and a model: "
                                                           "'openai:http://127.0.0.1:8080/v1|<model>'"))


def _evaluate(project: Project, out: list[dict]) -> None:
    from .review import max_revisions

    if not isinstance(project.data.get("evaluate"), dict) or _has_error(out, "evaluate.max_revisions"):
        return
    try:
        max_revisions(project)
    except ContractError as e:
        out.append(_finding("error", "evaluate.max_revisions", e.message))


def _regressions(project: Project, out: list[dict]) -> None:
    if project.data.get("regressions") is None or _has_error(out, "regressions"):
        return
    from .regress import cases_path

    try:
        cases_path(project)
    except ContractError as e:
        out.append(_finding("error", "regressions", e.message))


def _resolve_heads(project: Project, pack: Any, levels: list[str], out: list[dict]) -> dict | None:
    """The heads the pack builds from this file (its defaults merged with `heads:`), or None when it cannot."""
    if pack is None or any(_has_error(out, p) for p in ("heads", "pack_config", "pack")):
        return None
    check = getattr(pack, "check_config", None)
    if callable(check):
        try:
            found = [f for f in (check(project) or []) if isinstance(f, dict) and f.get("level") in LEVELS]
        except Exception as e:
            found = [_finding("warn", "pack_config", f"the pack's own check failed ({type(e).__name__}: "
                                                     f"{str(e)[:300]})")]
        out.extend(_finding(f["level"], str(f.get("path") or "pack_config"), str(f.get("message") or ""))
                   for f in found)
        if errors(found):
            return None
    try:
        specs = dict(pack.heads(project))
    except ContractError as e:
        out.append(_finding("error", "heads", e.message))
        return None
    except (ValueError, TypeError) as e:
        out.append(_finding("error", "heads", str(e)))
        return None
    except Exception as e:
        out.append(_finding("warn", "heads", f"the pack could not build its heads from this file "
                                             f"({type(e).__name__}: {str(e)[:300]})"))
        return None
    overrides = project.head_overrides
    for name, spec in specs.items():
        stakes = getattr(spec, "stakes", None)
        o = overrides.get(name)
        if stakes not in levels and not (isinstance(o, dict) and "stakes" in o):
            out.append(_finding("error", f"heads.{name}", f"the head's stakes {_one_of(stakes, levels)}"))
    return specs


def _switched_off(project: Project, pack: Any, specs: dict | None, out: list[dict]) -> set[str]:
    """The heads the pack switches off for this file (`packs.base.switched_off`, the list the service and `rlcd
    check` read): never served, so their policy overrides, rollout modes and route contracts are kept unchecked.
    Only when the pack's heads are known (`specs`)."""
    from ..packs.base import BasePack, switched_off

    if specs is None:
        return set()
    try:
        return switched_off(pack, project)
    except Exception as e:
        out.append(_finding("warn", "heads", f"the pack could not say which heads this file switches off "
                                             f"({type(e).__name__}: {str(e)[:300]})"))
        return BasePack.switched_off(pack, project)


def _overrides(project: Project, specs: dict | None, off: set[str], out: list[dict]) -> None:
    """`policy.overrides.<head>.<class>`: the head must be one of the project's and the class one the head can
    commit to. An override under a name nothing matches is silently never applied, so the stricter threshold
    somebody wrote for `refund_approved` would simply not exist: an error, with a suggestion. Only when the
    pack's heads are known (`specs`); a head the pack switches off (`off`: `heads.<head>.enabled: false`, or a
    tabular target switched off in `pack_config`) keeps its overrides unchecked."""
    raw = project.section("policy").get("overrides")
    if not specs or not isinstance(raw, dict) or _has_error(out, "policy"):
        return          # no heads resolved (e.g. the tabular pack before its data file exists): nothing to compare with
    for head, by_class in raw.items():
        name = str(head)
        if name in off:
            continue
        spec = specs.get(name)
        if spec is None:
            hint = _suggest(name, specs)
            out.append(_finding("error", f"policy.overrides.{name}",
                                f"is not a head of this project ({', '.join(specs) or 'it has none'}), so these "
                                f"thresholds would never apply" + (f". Did you mean {hint!r}?" if hint else "")))
            continue
        if not isinstance(by_class, dict):
            continue
        labels = [str(x) for x in spec.labels]
        commits = labels[1:] if spec.decision == "gate" else labels
        for cls in by_class:
            c = ("true" if cls else "false") if isinstance(cls, bool) else str(cls)   # as policy._check_overrides
            if c not in labels:
                hint = _suggest(c, labels)
                out.append(_finding("error", f"policy.overrides.{name}.{c}",
                                    f"is not a class of head {name!r} ({', '.join(labels)}), so these thresholds "
                                    f"would never apply" + (f". Did you mean {hint!r}?" if hint else "")
                                    + (". A noul head's classes are written \"true\" and \"false\", whatever the "
                                       "label values in the data are" if spec.type == "noul" else "")))
            elif c not in commits:
                out.append(_finding("warn", f"policy.overrides.{name}.{c}",
                                    f"head {name!r} is a gate head: its decisions only ever commit to "
                                    f"{commits[0]!r}, so thresholds for {c!r} are never used"))


def _cross(project: Project, specs: dict | None, off: set[str], ro: Rollout | None, book: RouteBook | None,
           out: list[dict]) -> None:
    """Names that must match a head: warnings, because a trained model may still carry a head the file no longer
    lists (`rlcd check` compares routes against the model). A head the pack switches off (`off`) is a head of the
    project, never served: its names are kept and not compared with it. Route contracts on a head of the project
    are stricter: one under an option the head does not have, or on a head that is not a choice head, fails `rlcd
    check` (routes.contracts), and RouteBook.check_head refuses every /v1/decide that applies route contracts to
    the head (500 bad_config). The generic decide applies them to every head it decides; the documents pack's
    decide only to response_type and route. So that is an error."""
    if specs is None:
        return
    known = set(specs) | off

    def unknown(path: str, name: str, what: str) -> None:
        hint = _suggest(name, known)
        out.append(_finding("warn", path, f"{what} {name!r}, which is not a head of this project "
                                          f"({', '.join(sorted(known)) or 'it has none'})"
                                          + (f". Did you mean {hint!r}?" if hint else "")))
    if ro is not None:
        for h, _ in ro.head_modes:
            if h not in known:
                unknown(f"rollout.heads.{h}", h, "sets a mode for")
        for h in ro.limited_heads or ():
            if h not in known:
                unknown("rollout.limited.heads", h, "lists")
    if book is not None:
        for head, options in book.routes.items():
            if head not in known:
                unknown(f"routes.{head}", head, "has contracts for")
                continue
            spec = specs.get(head)
            if spec is None:
                continue
            refused = (f"`rlcd check` fails routes.contracts, and a /v1/decide that applies route contracts to head "
                       f"{head!r} answers 500 bad_config")
            if spec.type != "choice":
                out.append(_finding("error", f"routes.{head}", f"head {head!r} is a {spec.type} head; contracts "
                                                                f"belong to the options of a choice head, so remove "
                                                                f"them. Until then {refused}"))
                continue
            for opt in options:
                if opt not in spec.classes:
                    hint = _suggest(opt, spec.classes)
                    out.append(_finding("error", f"routes.{head}.{opt}",
                                        f"is not an option of head {head!r} ({', '.join(spec.classes)}), so "
                                        f"{refused}" + (f". Did you mean {hint!r}?" if hint else "")))


def _secrets(data: dict, out: list[dict]) -> None:
    """Values that look like secrets. The value itself is never put into the message."""
    advice = "secrets belong in the environment (see docs/configuration.md), never in rlcd.yaml"

    def text(v: Any, path: str, named: bool) -> None:
        if not isinstance(v, str) or not v.strip() or _ENV_REF.match(v.strip()):
            return
        if named:
            out.append(_finding("warn", path, f"the key names a secret and has a value written into the file: "
                                              f"{advice}"))
            return
        _, found = redact_text(v, ("secrets",), skip=("assignment",))
        if found:
            out.append(_finding("warn", path, f"contains what looks like a secret ({', '.join(found)}): {advice}"))

    def walk(v: Any, path: str, depth: int) -> None:
        if depth > 32:
            return
        if isinstance(v, dict):
            for k, x in v.items():
                p = f"{path}.{k}" if path else str(k)
                if isinstance(x, (dict, list)):
                    if isinstance(x, list) and _secret_key(k):
                        for i, item in enumerate(x):
                            text(item, f"{p}[{i}]", True)
                    else:
                        walk(x, p, depth + 1)
                else:
                    text(x, p, _secret_key(k))
        elif isinstance(v, list):
            for i, x in enumerate(v):
                if isinstance(x, (dict, list)):
                    walk(x, f"{path}[{i}]", depth + 1)
                else:
                    text(x, f"{path}[{i}]", False)

    walk(data, "", 0)


def duplicate_keys(text: str) -> list[str]:
    """Dotted paths of keys written twice in one mapping (YAML keeps the last and drops the first silently)."""
    try:
        root = yaml.compose(text, Loader=yaml.SafeLoader)
    except yaml.YAMLError:
        return []
    found: list[str] = []

    def walk(node: Any, path: str) -> None:
        if isinstance(node, yaml.MappingNode):
            seen: set[str] = set()
            for k, v in node.value:
                name = str(getattr(k, "value", ""))
                p = f"{path}.{name}" if path else name
                if isinstance(k, yaml.ScalarNode):
                    if name in seen and name != "<<":
                        found.append(p)
                    seen.add(name)
                walk(v, p)
        elif isinstance(node, yaml.SequenceNode):
            for i, v in enumerate(node.value):
                walk(v, f"{path}[{i}]")

    walk(root, "")
    return found


# ----------------------------------------------------------------------------- entry points

def check_project(project: Project, pack: Any = None) -> list[dict]:
    """Every finding for a loaded project. `pack`: the pack instance to check `pack_config` and `heads` against
    (default: the one `pack:` names, loaded here; a pack that cannot be loaded is a warning)."""
    out: list[dict] = []
    data = project.data
    if not isinstance(data, dict):
        return [_finding("error", "(file)", "the top level of rlcd.yaml must be a mapping (`key: value` lines)")]
    if project.path is not None:
        try:
            dups = duplicate_keys(Path(project.path).read_text(encoding="utf-8"))
        except OSError:
            dups = []
        out.extend(_finding("error", p, "is written twice in the same mapping: YAML keeps the last one and drops "
                                        "the other without a word. Merge them") for p in dups)
    _check_pack_name(project, out)
    if pack is None:
        pack = _load_pack(project, out)
    extra: set[str] = set()
    fn = getattr(pack, "head_override_keys", None)
    if callable(fn):
        try:
            extra = {str(k) for k in fn() or ()}
        except Exception:
            extra = set()
    keys = dict(schema())
    keys["heads"] = Key("map", each=_head_schema(extra))
    sections = {leaf: f"{name}.{leaf}" for name, key in keys.items() if key.keys for leaf in key.keys}
    _check_keys(data, keys, "", out, elsewhere=sections)

    declared = _pack_keys(pack) if pack is not None else None
    if declared is not None and isinstance(data.get("pack_config"), dict):
        _check_keys(data["pack_config"], declared, "pack_config", out)
    for name, o in (data.get("heads") or {}).items() if isinstance(data.get("heads"), dict) else ():
        if isinstance(o, dict) and "name" in o:
            # spec_from_yaml drops it: the head's name is its key
            out[:] = [f for f in out if f["path"] != f"heads.{name}.name"]
            out.append(_finding("warn", f"heads.{name}.name", "is ignored: a head's name is its key under `heads:`"))

    levels = _policy(project, pack, out)
    _heads(project, levels, out)
    _new_heads(project, pack, out)
    ro = _rollout(project, levels, out)
    book = _routes(project, out)
    _batteries(project, pack, out)
    _kinds(project, "serve.redact", out)
    _kinds(project, "evaluate.output_scan", out)
    _labeling(project, out)
    _evaluate(project, out)
    _regressions(project, out)
    specs = _resolve_heads(project, pack, levels, out)
    off = _switched_off(project, pack, specs, out)
    _overrides(project, specs, off, out)
    _cross(project, specs, off, ro, book, out)
    _secrets(data, out)
    return out


def check_file(path: str | os.PathLike | None = None) -> tuple[Project | None, list[dict]]:
    """Load rlcd.yaml as `load_project` does and check it. A file that cannot be read as YAML is one error
    finding (and no project) instead of an exception."""
    from .config import load_project

    try:
        project = load_project(path)
    except yaml.YAMLError as e:
        mark = getattr(e, "problem_mark", None)
        where = f"line {mark.line + 1}, column {mark.column + 1}: " if mark is not None else ""
        problem = getattr(e, "problem", None) or str(e).splitlines()[0]
        return None, [_finding("error", "(file)", f"not valid YAML: {where}{problem}")]
    except ValueError as e:
        return None, [_finding("error", "(file)", str(e))]
    return project, check_project(project)


# ----------------------------------------------------------------------------- the effective configuration

def _with_defaults(key: Key, value: Any) -> Any:
    if key.keys is None:
        return key.default if value is None else value
    value = value if isinstance(value, dict) else {}
    out = {name: _with_defaults(sub, value.get(name)) for name, sub in key.keys.items()}
    out.update({k: v for k, v in value.items() if k not in out})
    return out


def environment() -> dict:
    """The environment variables open-rlcd reads: the value of a setting, `set` / `not set` for a secret."""
    out: dict[str, Any] = {}
    for name, secret in ENVIRONMENT:
        v = os.environ.get(name)
        out[name] = ("set" if v else "not set") if secret else v
    return out


def effective(project: Project, pack: Any = None) -> dict:
    """What the project resolves to: the file, the pack's defaults and the code's defaults merged, as
    `rlcd config show` prints it. A part that cannot be resolved carries {"error": ...} instead of raising.
    `pack_config`, `routes` and `owners` are echoed as written, with anything that looks like a secret redacted."""
    data = project.data if isinstance(project.data, dict) else {}
    s = schema()
    notes: list[dict] = []
    if pack is None:
        pack = _load_pack(project, notes)
    env_home = os.environ.get("RLCD_HOME")
    home_from = ("--home" if project.home_override is not None else "RLCD_HOME" if env_home
                 else "rlcd.yaml" if data.get("home") else "default")
    pack_from = "rlcd.yaml" if data.get("pack") else "RLCD_PACK" if os.environ.get("RLCD_PACK") else "default"
    out: dict[str, Any] = {
        "project": {"name": project.name, "config": str(project.path) if project.path else None,
                    "root": str(project.root), "home": str(project.home), "home_from": home_from},
        "pack": {"name": project.pack, "from": pack_from, "class": type(pack).__name__ if pack is not None else None,
                 "version": getattr(pack, "version", None)},
        "pack_config": redact(project.pack_config)[0],
    }
    if pack is None:
        out["pack"]["error"] = notes[0]["message"] if notes else "not loaded"

    def attempt(fn: Any) -> Any:
        try:
            return fn()
        except ContractError as e:
            return {"error": e.message}
        except Exception as e:
            return {"error": f"{type(e).__name__}: {e}"}

    def heads() -> dict:
        from ..packs.base import switched_off

        if pack is None:
            return {"error": "the pack is not loaded"}
        overrides = project.head_overrides
        res: dict[str, Any] = {}
        for name, spec in pack.heads(project).items():
            d = {k: (list(v) if isinstance(v, tuple) else v) for k, v in asdict(spec).items()
                 if k != "name" and v not in (None, "", (), {})}
            o = overrides.get(name)
            d["from_rlcd_yaml"] = sorted(o) if isinstance(o, dict) else []
            res[name] = d
        for name in sorted(switched_off(pack, project)):     # what the service does not serve, trained or not
            res[name] = {"enabled": False}
        return res

    def policy() -> dict:
        defaults = pack.policy_defaults() if pack is not None else {}
        return PL.PolicyConfig.from_dict(_merge(defaults, project.section("policy"))).to_dict()

    out["heads"] = attempt(heads)
    out["policy"] = attempt(policy)
    out["rollout"] = attempt(lambda: Rollout.from_config(data.get("rollout")).to_dict())
    for name in ("train", "labeling", "serve", "security", "evaluate"):
        out[name] = _with_defaults(s[name], data.get(name))

    def batteries() -> dict:
        defaults = (pack.defaults() or {}).get("batteries") if pack is not None else None
        return {n: {"version": b.version, "from": b.source, "owner": b.owner, "questions": list(b.questions_raw)}
                for n, b in BT.load(defaults, data.get("batteries")).items()}

    out["batteries"] = attempt(batteries)
    out["routes"] = redact(data.get("routes") or {})[0]

    def regressions() -> str:
        from .regress import cases_path

        return str(cases_path(project))

    out["regressions"] = attempt(regressions)
    out["owners"] = redact(data.get("owners") or {})[0]
    out["environment"] = environment()
    return out


__all__ = ["CONFIG_NAME", "Key", "check_file", "check_project", "duplicate_keys", "effective", "environment",
           "errors", "flatten", "format_findings", "head_keys", "known_keys", "reference", "schema", "summary",
           "warnings"]
