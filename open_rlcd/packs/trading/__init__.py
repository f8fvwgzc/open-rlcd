"""Trading adapter pack: talks to the View Engine quant sidecar over HTTP. No trading logic is copied here.

The sidecar owns the features (`GET /features`) and the labelled history (`GET /dataset`); this pack only maps
them onto open-rlcd heads and lets the core do what it does for every pack: time-ordered purged folds,
calibration on ground-truth rows, an untouched holdout, a skill test, and act / confirm / hold.

    pack: trading
    pack_config:
      quant_url: http://127.0.0.1:8090      # local hosts only unless security.allow_remote / RLCD_ALLOW_REMOTE=1
      symbols: [XAUUSD, EURUSD]
      intervals: [15m]
      sl_pips: 20
      tp_pips: 50
      tp2_pips: 100
      horizon: 48
      max_rows: 20000                       # JSON export cap per symbol
      pips: {XAUUSD: [0.1]}                 # optional pip sizes per symbol (each one is a segment)

Head `setup:<interval>` (choice buy / sell / hold) answers "from this candle's close, which action reached its
target before its stop within the horizon", exactly as the sidecar labels it. Serve-time state is either
`{symbol, interval?, as_of?, pip?}` (features are fetched from the sidecar) or a precomputed
`{x, feature_names, feature_version}`. Analysis tool, not financial advice.

Every key: docs/configuration.md (`pack_config` of the trading pack). `rlcd config check` checks them
(`config_keys`, `check_config`); a key written with no value is its default, as when it is missing.
"""
from __future__ import annotations

import copy
import difflib
import io
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import httpx
import numpy as np

from ...core.config import Project
from ...core.contract import ContractError
from ...core.dataset import Dataset, concat
from ...core.head import HeadSpec
from ..base import BasePack, epoch, spec_from_yaml

INTERVAL_SECONDS = {"1m": 60, "5m": 300, "15m": 900, "30m": 1800, "1h": 3600, "4h": 14400, "1d": 86400}
CLASSES = ("buy", "sell", "hold")
WIRE_NPZ = {1: 0, 2: 1, 0: 2}   # sidecar npz `best`: 0 hold, 1 buy, 2 sell -> our order buy, sell, hold
LOOPBACK = {"127.0.0.1", "localhost", "::1"}
# what `_cfg` uses for a pack_config key that is missing or written with no value
DEFAULTS = {"quant_url": "http://127.0.0.1:8090", "symbols": ["XAUUSD"], "intervals": ["15m"], "sl_pips": 20,
            "tp_pips": 50, "tp2_pips": 100, "horizon": 48, "max_rows": 20000}
# the head fields a `heads.<head>` override cannot change: the question is the sidecar's (what it labels, in this
# class order, on which candles)
FIXED = ("type", "classes", "view")


def _hint(word: Any, options: Iterable[str]) -> str:
    hit = difflib.get_close_matches(str(word), list(options), n=1, cutoff=0.6)
    return f". Did you mean {hit[0]!r}?" if hit else ""


def _check_url(url: str, allow_remote: bool) -> str:
    try:
        from ...connectors.security import check_host

        check_host(url, allow_remote)
    except ImportError:  # pragma: no cover - connectors package always ships; keep a local fallback
        host = httpx.URL(url).host
        if host not in LOOPBACK and not allow_remote:
            raise ValueError(f"quant_url host {host!r} is not local; set security.allow_remote or "
                             f"RLCD_ALLOW_REMOTE=1") from None
    return url.rstrip("/")


class SidecarClient:
    def __init__(self, url: str, transport: httpx.BaseTransport | None = None, timeout: float = 120.0):
        self.url = url.rstrip("/")
        self._http = httpx.Client(base_url=self.url, transport=transport, timeout=timeout)

    def _get(self, path: str, params: dict) -> httpx.Response:
        params = {k: v for k, v in params.items() if v is not None}
        try:
            r = self._http.get(path, params=params)
        except httpx.HTTPError as e:
            raise ContractError(f"quant sidecar unreachable at {self.url}{path}: {type(e).__name__}",
                                kind="connector_unavailable", status=503) from e
        if r.status_code >= 500:
            raise ContractError(f"quant sidecar {path} failed with {r.status_code}", kind="connector_unavailable",
                                status=503)
        if r.status_code >= 400:
            try:
                msg = r.json().get("error") or r.text
            except ValueError:
                msg = r.text
            raise ContractError(f"quant sidecar {path}: {str(msg)[:300]}", kind="quant_rejected",
                                quant_status=r.status_code)
        return r

    def features(self, symbol: str, interval: str, **kw: Any) -> dict:
        return self._get("/features", {"symbol": symbol, "interval": interval, **kw}).json()

    def dataset(self, symbol: str, interval: str, **kw: Any) -> dict:
        """Rows of one export as arrays: X, times, best (our class order, -1 unlabelled), meta."""
        r = self._get("/dataset", {"symbol": symbol, "interval": interval, **kw})
        if "json" not in r.headers.get("content-type", "json"):
            z = np.load(io.BytesIO(r.content), allow_pickle=False)   # never unpickle a download
            meta = json.loads(str(z["meta"][()]))
            wire = z["best"].astype(int)
            best = np.array([WIRE_NPZ.get(int(b), -1) for b in wire])
            return {"X": np.asarray(z["X"], np.float32), "times": np.asarray(z["times"], np.int64), "best": best,
                    "meta": meta}
        d = r.json()
        names = {c: i for i, c in enumerate(CLASSES)}
        best = np.array([names.get(b, -1) for b in (d.get("y") or {}).get("best") or []])
        times = np.array([epoch(t) for t in d.get("times") or []], np.int64)
        X = np.array([[np.nan if v is None else float(v) for v in row] for row in d.get("X") or []], np.float32)
        meta = {k: v for k, v in d.items() if k not in ("X", "y", "times")}
        return {"X": X.reshape(len(times), -1), "times": times, "best": best, "meta": meta}


class TradingProvider:
    """Features come from the sidecar; this provider only fetches and checks them."""

    def __init__(self, client: SidecarClient, config: dict):
        self.client, self.cfg = client, dict(config)

    @property
    def feature_version(self) -> str:
        return str(self.cfg.get("feature_version") or "unknown")

    def names(self, view: str) -> list[str]:
        return list(self.cfg.get("feature_names") or [])

    def groups(self, view: str) -> dict[str, str]:
        return dict(self.cfg.get("feature_groups") or {})

    def describe(self, view: str) -> dict:
        return {"state": "{symbol, interval?, as_of?, pip?} (features fetched from the quant sidecar) or a "
                         "precomputed {x, feature_names, feature_version}", "interval": view}

    def to_config(self) -> dict:
        return dict(self.cfg)

    def transform(self, view: str, states: list[Any]) -> np.ndarray:
        rows = []
        p = self.cfg.get("params") or {}
        for s in states:
            if not isinstance(s, dict) or not s.get("symbol"):
                raise ContractError("trading heads need state {symbol, interval?, as_of?, pip?} or "
                                    "{x, feature_names, feature_version}")
            if s.get("interval") not in (None, view):
                raise ContractError(f"state.interval is {s.get('interval')!r} but this head is the {view} head")
            f = self.client.features(str(s["symbol"]), view, as_of=s.get("as_of"), pip=s.get("pip"),
                                     sl_pips=p.get("sl_pips"), tp_pips=p.get("tp_pips"), tp2_pips=p.get("tp2_pips"),
                                     horizon=p.get("horizon"))
            if f.get("feature_version") != self.feature_version:
                raise ContractError(f"the sidecar now serves feature version {f.get('feature_version')!r}; this model "
                                    f"was trained on {self.feature_version!r}. Retrain.",
                                    kind="feature_version_mismatch", expected=self.feature_version,
                                    got=f.get("feature_version"))
            if list(f.get("feature_names") or []) != self.names(view):
                raise ContractError("the sidecar's feature_names differ from the trained ones", kind="feature_names_mismatch")
            rows.append([np.nan if v is None else float(v) for v in f.get("x") or []])
        return np.asarray(rows, np.float32).reshape(len(states), len(self.names(view)))


class TradingPack(BasePack):
    name = "trading"
    version = "0.1.0"
    yaml_path = Path(__file__).with_name("pack.yaml")
    abstain = "hold"

    def __init__(self, transport: httpx.BaseTransport | None = None):
        self.transport = transport

    def _cfg(self, project: Project) -> dict:
        """pack_config over DEFAULTS. A key written with no value (`horizon:`) is its default, as when it is missing:
        the convention of every rlcd.yaml key (docs/configuration.md), and what `config_keys` declares."""
        c = copy.deepcopy(DEFAULTS)
        c.update({k: v for k, v in project.pack_config.items() if v is not None})
        return c

    # ------------------------------------------------------------------ rlcd.yaml validation (core/config_check.py)
    def config_keys(self) -> dict:
        """The `pack_config` keys this pack reads, for `rlcd config check`. Where each is read: quant_url ->
        `client` (and `_check_url`: loopback only unless remote hosts are allowed); symbols, pips, max_rows ->
        `datasets`; intervals -> `heads`; sl_pips, tp_pips, tp2_pips, horizon -> `provider` (sent to the sidecar
        with every /dataset and /features call, stored with the trained version) and `datasets` (horizon: the
        label window)."""
        from ...core.config_check import Key

        pips = {"min": 0, "open_min": True}
        return {
            "quant_url": Key("str", default=DEFAULTS["quant_url"]),
            "symbols": Key("list", default=DEFAULTS["symbols"]),
            "intervals": Key("list", default=DEFAULTS["intervals"]),
            "sl_pips": Key("number", default=DEFAULTS["sl_pips"], **pips),
            "tp_pips": Key("number", default=DEFAULTS["tp_pips"], **pips),
            "tp2_pips": Key("number", default=DEFAULTS["tp2_pips"], **pips),
            "horizon": Key("int", default=DEFAULTS["horizon"], min=1),
            "max_rows": Key("int", default=DEFAULTS["max_rows"], min=1),
            "pips": Key("map", each=Key("list", label="symbol")),
        }

    def check_config(self, project: Project) -> list[dict]:
        """What the key list cannot say (no network): an empty or unsupported `intervals`, an empty `symbols`, a
        `quant_url` that is not an http(s) URL, pip sizes that are not positive numbers or belong to no listed
        symbol, and a `heads:` entry that is not one of this pack's heads or changes the question itself (FIXED).
        Runs after the key and type checks passed."""
        cfg = self._cfg(project)
        out: list[dict] = []

        def add(path: str, message: str, level: str = "error") -> None:
            out.append({"level": level, "path": path, "message": message})

        intervals, symbols = list(cfg["intervals"]), list(cfg["symbols"])
        if not intervals:
            add("pack_config.intervals", "is empty: list at least one interval (each one is a setup:<interval> head)")
        for iv in intervals:
            if iv not in INTERVAL_SECONDS:
                add("pack_config.intervals", f"{iv!r} is not an interval this pack supports "
                                             f"({', '.join(INTERVAL_SECONDS)}){_hint(iv, INTERVAL_SECONDS)}")
        if not symbols:
            add("pack_config.symbols", "is empty: list at least one symbol the quant sidecar exports")
        elif not all(isinstance(s, str) and s.strip() for s in symbols):
            add("pack_config.symbols", "must be a list of symbol names (text)")
        url = urlsplit(str(cfg["quant_url"]).strip())
        if url.scheme not in ("http", "https") or not url.hostname:
            add("pack_config.quant_url", f"must be an http:// or https:// URL with a host, got {cfg['quant_url']!r}")
        for symbol, sizes in (cfg.get("pips") or {}).items():
            if any(isinstance(p, bool) or not isinstance(p, (int, float)) or p <= 0 for p in sizes or ()):
                add(f"pack_config.pips.{symbol}", "must be a list of pip sizes (numbers > 0)")
            elif symbol not in symbols:
                add(f"pack_config.pips.{symbol}", f"{symbol!r} is not in pack_config.symbols, so these pip sizes are "
                                                  f"never used", "warn")
        names = {f"setup:{iv}": iv for iv in intervals if iv in INTERVAL_SECONDS}
        for name, o in project.head_overrides.items():
            if not isinstance(o, dict):
                continue
            if name not in names:
                if o.get("enabled", True) is not False:      # switching off a head that does not exist changes nothing
                    add(f"heads.{name}", f"is not a head of the trading pack ({', '.join(names) or 'it has none'}): "
                                         f"its heads are setup:<interval>, one per pack_config.intervals"
                                         f"{_hint(name, names)}")
                continue
            for field, message in self._fixed_changed(names[name], o):
                add(f"heads.{name}.{field}", message)
        return out

    def switched_off(self, project: Project, trained: Iterable[str] = ()) -> set[str]:
        """BasePack's (`heads.<head>.enabled: false`), plus every trained head whose interval is no longer in
        `pack_config.intervals`: that list names the heads this pack serves, so removing an interval stops serving
        its head at once, as `enabled: false` does, while a train keeps carrying the trained head over (same
        feature version): listing the interval again serves it again without a retrain. Reads rlcd.yaml only."""
        intervals = self._cfg(project)["intervals"]
        listed = {f"setup:{iv}" for iv in intervals} if isinstance(intervals, (list, tuple)) else set()
        return super().switched_off(project, trained) | {str(n) for n in trained if n not in listed}

    def client(self, project: Project, url: str | None = None) -> SidecarClient:
        try:
            u = _check_url(url or self._cfg(project)["quant_url"], project.allow_remote)
        except ValueError as e:
            raise ContractError(str(e)) from None
        return SidecarClient(u, self.transport)

    @staticmethod
    def _default_head(iv: str) -> dict:
        return {"type": "choice", "classes": list(CLASSES), "view": iv, "model": "auto", "stakes": "medium",
                "description": f"From this {iv} candle's close, which action reached its target before its stop "
                               f"within the horizon (labelled by the quant sidecar)?"}

    def _fixed_changed(self, iv: str, override: dict) -> list[tuple[str, str]]:
        """[(field, message)] for each FIXED field `override` sets to something other than the pack's value."""
        base = self._default_head(iv)
        out = []
        for field in FIXED:
            if field not in override:
                continue
            value = override[field]
            if field == "classes":
                same = isinstance(value, (list, tuple)) and [str(c) for c in value] == base["classes"]
            else:
                same = str(value) == base[field]
            if not same:
                out.append((field, f"cannot be changed in the trading pack: the quant sidecar labels this head as a "
                                   f"{base['type']} head with classes {', '.join(CLASSES)} (in this order) on the "
                                   f"{iv} candles; got {value!r}"))
        return out

    def heads(self, project: Project) -> dict[str, HeadSpec]:
        """One `setup:<interval>` head per interval of `pack_config.intervals`, with its `heads.<head>` fields from
        rlcd.yaml merged in (the override wins field by field, as in every pack). `enabled: false` switches the head
        off: it is not trained and not served. A FIXED field set to another value is refused."""
        out = {}
        overrides = project.head_overrides
        for iv in self._cfg(project)["intervals"]:
            if iv not in INTERVAL_SECONDS:
                raise ValueError(f"interval {iv!r} not supported; choose from {list(INTERVAL_SECONDS)}")
            name = f"setup:{iv}"
            o = overrides.get(name)
            o = o if isinstance(o, dict) else {}
            if o.get("enabled", True) is False:
                continue
            changed = self._fixed_changed(iv, o)
            if changed:
                raise ValueError(f"head {name!r}: {changed[0][0]} {changed[0][1]}")
            d = {**self._default_head(iv), **{k: v for k, v in o.items() if k != "enabled"}}
            out[name] = spec_from_yaml(name, d)
        return out

    def provider(self, project: Project, config: dict | None = None) -> TradingProvider:
        cfg = self._cfg(project)
        conf = dict(config) if config else {"params": {k: cfg[k] for k in ("sl_pips", "tp_pips", "tp2_pips",
                                                                        "horizon")}}
        return TradingProvider(self.client(project), conf)

    def datasets(self, project: Project, provider: TradingProvider, heads: list[str],
                 feedback: list[dict] | None = None, feedback_weight: float = 5.0) -> dict[str, Dataset]:
        cfg = self._cfg(project)
        specs = self.heads(project)
        p = provider.cfg.setdefault("params", {})
        out: dict[str, Dataset] = {}
        for name in heads:
            iv = specs[name].view
            parts, notes = [], []
            for symbol in cfg["symbols"]:
                for pip in (cfg.get("pips") or {}).get(symbol) or [None]:
                    try:
                        d = provider.client.dataset(symbol, iv, sl_pips=p.get("sl_pips"), tp_pips=p.get("tp_pips"),
                                                    tp2_pips=p.get("tp2_pips"), horizon=p.get("horizon"), pip=pip,
                                                    max_rows=cfg.get("max_rows"))
                    except ContractError as e:
                        if e.status == 503:
                            raise
                        notes.append(f"skipped {symbol}: {e.message}")
                        continue
                    meta = d["meta"]
                    if not provider.cfg.get("feature_version"):
                        provider.cfg.update(feature_version=meta.get("feature_version"),
                                            feature_names=list(meta.get("feature_names") or []),
                                            feature_groups=meta.get("feature_groups") or {})
                    if meta.get("feature_version") != provider.feature_version or \
                            list(meta.get("feature_names") or []) != provider.names(iv):
                        notes.append(f"skipped {symbol}: feature version {meta.get('feature_version')} differs")
                        continue
                    keep = d["best"] >= 0
                    if not keep.any():
                        notes.append(f"skipped {symbol}: no labelled rows")
                        continue
                    # label window of row i = the next `horizon` candles of this export (past its end:
                    # extrapolated), exactly what the sidecar labelled; weekend gaps are therefore respected
                    times, step, H = d["times"], INTERVAL_SECONDS[iv], int(p.get("horizon") or 48)
                    idx = np.arange(len(times)) + H
                    t_end_all = times[np.minimum(idx, len(times) - 1)] + np.maximum(0, idx - (len(times) - 1)) * step
                    t = times[keep]
                    seg = f"{meta.get('symbol', symbol)}@{float(meta.get('pip') or pip or 0):g}"
                    parts.append(Dataset(X=d["X"][keep], y=d["best"][keep], feature_names=provider.names(iv),
                                         feature_version=provider.feature_version,
                                         ids=np.array([f"{seg}:{x}" for x in t], dtype=object), t=t,
                                         t_end=t_end_all[keep],
                                         source=np.full(int(keep.sum()), "ground_truth", dtype=object),
                                         segment=np.full(int(keep.sum()), seg, dtype=object),
                                         feature_groups=provider.groups(iv)))
            fb = [r for r in feedback or [] if r.get("head") == name and isinstance(r.get("state"), dict)
                  and "x" in r["state"] and r["state"].get("feature_version") == provider.feature_version]
            if fb:
                X = np.array([[np.nan if v is None else float(v) for v in r["state"]["x"]] for r in fb], np.float32)
                y = np.array([CLASSES.index(r["label"]) for r in fb])
                t = np.array([epoch(r.get("time") or r.get("received_at")) or -1 for r in fb], np.int64)
                parts.append(Dataset(X=X, y=y, feature_names=provider.names(iv),
                                     feature_version=provider.feature_version,
                                     ids=np.array([f"feedback:{r.get('id')}" for r in fb], dtype=object), t=t,
                                     t_end=np.where(t >= 0, t + int(p.get("horizon") or 48) * INTERVAL_SECONDS[iv], t),
                                     weight=np.array([float(r.get("weight") or 1) * feedback_weight for r in fb]),
                                     source=np.full(len(fb), "feedback", dtype=object),
                                     segment=np.full(len(fb), "feedback", dtype=object)))
                notes.append(f"{len(fb)} feedback rows added to training (weight x {feedback_weight:g})")
            if parts:
                ds = concat(parts)
                order = np.argsort(ds.t, kind="stable")
                ds = ds.subset(order)
                ds.notes = notes
                out[name] = ds
        return out

