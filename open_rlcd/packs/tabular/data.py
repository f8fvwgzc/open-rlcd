"""The tabular pack's input side: reading the training table and turning `pack_config` into typed settings.

    pack_config:
      data: data/churn.csv            # .csv / .tsv / .txt (optionally .gz), or .parquet (needs the 'parquet' extra)
      id: row_id                      # optional: row ids (reported, never a feature)
      group: customer_id              # optional: rows of one group never straddle train / calibration / holdout
      time: month                     # optional: ISO-8601 or epoch seconds; enables time ordering
      time_format: "%d/%m/%Y %I:%M %p"   # optional: a strptime format for `time` values in another notation
      segment: channel                # optional: per-segment base rates and holdout slices
      features: auto                  # or a list of columns; auto = every column except label/id/group/time/segment
                                      #   (empty and identifier-like columns are left out, with a note)
      exclude: [review_level]         # columns never used as features, whatever `features` says: list every
                                      #   column that is only known after the decision (it would leak the answer)
      categorical: auto               # columns whose values are not all numbers; a list adds columns to
                                      #   treat as categorical even if numeric (zip codes, plan ids)
      text: [note]                    # free-text columns: hashed word and character n-gram features (provider.py);
                                      #   nothing of the text is stored in the bundle
      text_buckets: 256               # hashed features per text column
      max_categories: 50              # one-hot vocabulary size per categorical column (plus __other__, __missing__)
      hash_categories: false          # true: bundles store salted SHA-256 hashes of categorical values, not the
                                      #   raw values (use it when a categorical column could hold personal data)
      delimiter: auto                 # auto = the most frequent of , ; TAB | in the header line
      na_values: ["", NA, N/A, NaN, null, "<NA>", "#N/A"]      # case-insensitive
      label_source: ground_truth      # provenance of the file's labels: ground_truth | human | synthetic | judge
      targets:
        churned: {label: churned, positive: "yes", stakes: medium}
        next_plan: {label: plan_tier_next, type: choice, classes: [downgrade, same, upgrade]}
        satisfaction: {label: satisfaction, type: score, levels: [very unhappy, unhappy, happy, very happy]}

    Shorthand for one head: `label: churned` (+ optional `question: will_churn`, default head name `outcome`,
    and any target key such as `positive` or `type` at the top level).

What this module does not do: it never fetches anything over the network (`data` must be a local file), never
guesses a semantic order for score labels (non-numeric levels need `levels` or `classes`), and treats a label
value outside a choice / score head's classes as "skip this row" (counted in the dataset notes), not as an error.
"""
from __future__ import annotations

import csv
import gzip
import math
import re
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

from ...core.contract import MAX_CHOICE_OPTIONS, MAX_SCORE_LEVELS, MIN_SCORE_LEVELS
from ...core.dataset import SOURCES

DEFAULT_NA = ("", "na", "n/a", "nan", "null", "<na>", "#n/a")
POSITIVE_WORDS = ("true", "yes", "1", "positive", "churn")
TRUE_SPELLINGS = frozenset({"true", "yes", "y", "t", "1", "on"})
FALSE_SPELLINGS = frozenset({"false", "no", "n", "f", "0", "off"})
DELIMITERS = (",", ";", "\t", "|")
CSV_SUFFIXES = (".csv", ".tsv", ".txt")
PARQUET_SUFFIXES = (".parquet", ".pq")

TARGET_KEYS = ("label", "type", "positive", "classes", "levels")
SPEC_KEYS = ("model", "stakes", "decision", "conformal", "description", "params", "input", "enabled")
BASE_KEYS = ("data", "id", "group", "time", "segment", "features", "categorical", "max_categories", "targets",
             "delimiter", "encoding", "na_values", "label_source", "question", "hash_categories", "exclude", "text",
             "text_buckets", "time_format")
DEFAULT_TEXT_BUCKETS = 256
MIN_TEXT_BUCKETS, MAX_TEXT_BUCKETS = 8, 4096
# general keys other parts of open-rlcd may put into any pack_config; tolerated, unused here
TOLERATED_KEYS = ("allow_remote",)
_INT_FLOAT = re.compile(r"^[+-]?\d+\.0+$")


class TabularConfigError(ValueError):
    """rlcd.yaml pack_config (or the data file it points at) cannot be used as written."""


class DataUnavailable(TabularConfigError):
    """The data file is missing or unreadable (heads that need it to infer their classes are skipped)."""


# ----------------------------------------------------------------------------- cell values

def is_missing(v: Any, na: frozenset[str]) -> bool:
    if v is None:
        return True
    if isinstance(v, float):
        return math.isnan(v)
    if isinstance(v, str):
        return v.strip().lower() in na
    return False


def canon(v: Any, na: frozenset[str]) -> str | None:
    """A cell (CSV text, Parquet scalar or JSON value) as the string a vocabulary / label set is keyed by.

    The same rules run at train and at serve time: strip whitespace, NA tokens -> None, booleans -> "true" /
    "false", integral numbers -> "3" whether written 3, 3.0 or "3.0". Leading zeros are kept ("02139")."""
    if is_missing(v, na):
        return None
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        return str(int(v)) if v.is_integer() else repr(v)
    s = str(v).strip()
    if _INT_FLOAT.match(s):
        s = s.split(".", 1)[0]
        s = s[1:] if s.startswith("+") else s
    return s


def to_float(v: Any, na: frozenset[str]) -> float:
    """A numeric cell as float; NaN when missing, non-finite or unparseable."""
    if is_missing(v, na):
        return math.nan
    if isinstance(v, bool):
        return 1.0 if v else 0.0
    if isinstance(v, (int, float, Decimal)):
        f = float(v)
    else:
        try:
            f = float(str(v).strip())
        except ValueError:
            return math.nan
    return f if math.isfinite(f) else math.nan


def parse_time(v: Any, na: frozenset[str], fmt: str | None = None) -> int | None:
    """Epoch seconds from ISO-8601 text / a datetime, or from epoch seconds (milliseconds are recognised by size).
    Naive timestamps are read as UTC so a row's time never depends on the machine's time zone.

    `fmt` (pack_config.time_format): a `strptime` format tried first on text values, so "20250106" with
    "%Y%m%d" is a date and not an epoch; a value that does not match it is still read as ISO-8601 / epoch
    seconds (a feedback row's state may carry the time in either notation)."""
    if is_missing(v, na):
        return None
    if isinstance(v, datetime):
        return int((v if v.tzinfo else v.replace(tzinfo=UTC)).timestamp())
    if isinstance(v, date):
        return int(datetime(v.year, v.month, v.day, tzinfo=UTC).timestamp())
    if fmt and not isinstance(v, bool):
        try:
            dt = datetime.strptime(canon(v, na) or "", fmt)
        except ValueError:
            pass
        else:
            return int((dt if dt.tzinfo else dt.replace(tzinfo=UTC)).timestamp())
    f = to_float(v, na) if not isinstance(v, bool) else math.nan
    if not math.isnan(f):
        return int(f / 1000) if abs(f) > 1e11 else int(f)
    try:
        dt = datetime.fromisoformat(str(v).strip().replace("Z", "+00:00"))
    except ValueError:
        return None
    return int((dt if dt.tzinfo else dt.replace(tzinfo=UTC)).timestamp())


# ----------------------------------------------------------------------------- the table

@dataclass
class Table:
    """A table read column-wise. CSV cells are strings; Parquet cells are Python scalars (or None)."""
    path: Path
    columns: list[str]
    data: dict[str, list[Any]]
    delimiter: str | None = None

    @property
    def n(self) -> int:
        return len(self.data[self.columns[0]]) if self.columns else 0

    def rows(self, columns: list[str] | None = None) -> list[dict[str, Any]]:
        cols = columns or self.columns
        arrs = [self.data[c] for c in cols]
        return [dict(zip(cols, vals, strict=True)) for vals in zip(*arrs, strict=True)]

    def require(self, columns: list[str], what: str) -> None:
        missing = [c for c in columns if c not in self.data]
        if missing:
            raise TabularConfigError(f"{what}: column(s) {missing} are not in {self.path.name} "
                                     f"(columns: {self.columns})")


def _sniff(header_line: str, suffix: str) -> str:
    counts = {d: header_line.count(d) for d in DELIMITERS}
    best = max(DELIMITERS, key=lambda d: counts[d])
    if counts[best] == 0:
        return "\t" if suffix == ".tsv" else ","
    return best


def _open_text(path: Path, encoding: str):
    if path.suffix.lower() == ".gz":
        return gzip.open(path, "rt", encoding=encoding, newline="")
    return open(path, encoding=encoding, newline="")


def _base_suffix(path: Path) -> str:
    s = [x.lower() for x in path.suffixes]
    if s and s[-1] == ".gz":
        s = s[:-1]
    return s[-1] if s else ""


def read_csv(path: Path, delimiter: str = "auto", encoding: str = "utf-8-sig") -> Table:
    """Read a delimited text file with the stdlib csv module (header row, standard quoting). A BOM is dropped
    with the default encoding. A row with the wrong number of fields is an error naming its line: that is
    almost always a wrong delimiter, and silently dropping rows would bias the data."""
    suffix = _base_suffix(path)
    with _open_text(path, encoding) as f:
        first = f.readline()
    if not first.strip():
        raise TabularConfigError(f"{path}: empty file or empty header line")
    delim = _sniff(first, suffix) if delimiter == "auto" else delimiter
    if len(delim) != 1:
        raise TabularConfigError(f"delimiter must be one character or 'auto', got {delimiter!r}")
    with _open_text(path, encoding) as f:
        reader = csv.reader(f, delimiter=delim)
        header = [h.strip() for h in next(reader)]
        _check_header(path, header)
        cols: list[list[str]] = [[] for _ in header]
        for rec in reader:
            if not rec or all(not c.strip() for c in rec):
                continue
            if len(rec) != len(header):
                raise TabularConfigError(f"{path}: line {reader.line_num} has {len(rec)} fields, the header has "
                                         f"{len(header)} (delimiter {delim!r}; set pack_config.delimiter if wrong)")
            for j, c in enumerate(rec):
                cols[j].append(c)
    return Table(path=path, columns=header, data=dict(zip(header, cols, strict=True)), delimiter=delim)


def _check_header(path: Path, header: list[str]) -> None:
    if any(not h for h in header):
        raise TabularConfigError(f"{path}: the header has an empty column name: {header}")
    dup = sorted({h for h in header if header.count(h) > 1})
    if dup:
        raise TabularConfigError(f"{path}: duplicate column names {dup}")


def _py(v: Any) -> Any:
    if v is None or isinstance(v, (bool, int, float, str)):
        return v
    if isinstance(v, Decimal):
        return float(v)
    if isinstance(v, (datetime, date)):
        return v.isoformat()
    if isinstance(v, bytes):
        return v.decode("utf-8", errors="replace")
    return str(v)


def read_parquet(path: Path) -> Table:
    """Read a Parquet file through pyarrow (imported here, only when a Parquet file is configured)."""
    try:
        import pyarrow.parquet as pq  # noqa: PLC0415
    except ImportError as e:
        raise ImportError("reading Parquet needs pyarrow: install the 'parquet' extra (`uv sync --extra parquet` "
                          "or `pip install 'open-rlcd[parquet]'`), or export the table to CSV") from e
    d = pq.read_table(path).to_pydict()
    header = [str(c) for c in d]
    _check_header(path, header)
    return Table(path=path, columns=header, data={str(k): [_py(v) for v in vals] for k, vals in d.items()})


def read_table(path: Path, delimiter: str = "auto", encoding: str = "utf-8-sig") -> Table:
    path = Path(path)
    if not path.is_file():
        raise DataUnavailable(f"tabular data file not found: {path}")
    suffix = _base_suffix(path)
    if suffix in PARQUET_SUFFIXES:
        t = read_parquet(path)
    elif suffix in CSV_SUFFIXES:
        t = read_csv(path, delimiter, encoding)
    else:
        raise TabularConfigError(f"{path.name}: unsupported file type {suffix!r}; use .csv/.tsv/.txt (optionally "
                                 f".gz) or .parquet")
    if not t.columns:
        raise TabularConfigError(f"{path}: no columns")
    return t


# ----------------------------------------------------------------------------- pack_config

@dataclass(frozen=True)
class TabularConfig:
    data: Path
    id: str | None = None
    group: str | None = None
    time: str | None = None
    segment: str | None = None
    time_format: str | None = None                 # strptime format of the time column (None: ISO-8601 / epoch)
    features: tuple[str, ...] | None = None        # None = auto
    exclude: tuple[str, ...] = ()                  # never features (known only after the decision, or unwanted)
    categorical: tuple[str, ...] | None = None     # None = auto
    text: tuple[str, ...] = ()                     # free-text columns -> hashed n-gram features
    text_buckets: int = DEFAULT_TEXT_BUCKETS       # hashed features per text column
    max_categories: int = 50
    hash_categories: bool = False                  # vocabularies as salted SHA-256 hashes (see provider.py)
    delimiter: str = "auto"
    encoding: str = "utf-8-sig"
    na_values: frozenset[str] = frozenset(DEFAULT_NA)
    label_source: str = "ground_truth"
    targets: dict[str, dict] = field(default_factory=dict)

    @property
    def meta_columns(self) -> list[str]:
        return [c for c in (self.id, self.group, self.time, self.segment) if c]

    def signature(self) -> tuple:
        return (str(self.data), self.delimiter, self.encoding)


def _col(v: Any, key: str) -> str | None:
    if v is None or v == "":
        return None
    if not isinstance(v, str):
        raise TabularConfigError(f"pack_config.{key} must be a column name, got {v!r}")
    return v


def _cols(v: Any, key: str) -> tuple[str, ...] | None:
    if v is None or v == "auto":
        return None
    if not isinstance(v, list) or not all(isinstance(c, str) and c for c in v):
        raise TabularConfigError(f"pack_config.{key} must be 'auto' or a list of column names")
    return tuple(v)


def _col_list(v: Any, key: str) -> tuple[str, ...]:
    if v is None:
        return ()
    if not isinstance(v, list) or not all(isinstance(c, str) and c for c in v):
        raise TabularConfigError(f"pack_config.{key} must be a list of column names")
    if len(set(v)) != len(v):
        raise TabularConfigError(f"pack_config.{key} lists a column twice: {v}")
    return tuple(v)


def parse_config(pc: dict, resolve_path: Any) -> TabularConfig:
    """Validate rlcd.yaml `pack_config` for the tabular pack. Unknown keys are errors (a typo such as
    `featurs` must not silently train on every column)."""
    pc = dict(pc or {})
    if not pc.get("data"):
        raise TabularConfigError("pack_config.data is required: the path of the training table (.csv or .parquet)")
    data = str(pc["data"])
    if re.match(r"^[a-z][a-z0-9+.-]*://", data, re.I):
        raise TabularConfigError(f"pack_config.data must be a local file path, got {data!r}: download the file "
                                 f"first (the tabular pack never fetches over the network)")
    shorthand = "label" in pc
    if shorthand and pc.get("targets"):
        raise TabularConfigError("pack_config has both `label` (one-head shorthand) and `targets`: use one")
    allowed = set(BASE_KEYS) | set(TOLERATED_KEYS) | ((set(TARGET_KEYS) | set(SPEC_KEYS)) if shorthand else set())
    unknown = sorted(set(pc) - allowed)
    if unknown:
        raise TabularConfigError(f"pack_config: unknown keys {unknown}; known: {sorted(allowed)}")
    if shorthand:
        name = str(pc.get("question") or "outcome")
        targets = {name: {k: pc[k] for k in (*TARGET_KEYS, *SPEC_KEYS) if k in pc}}
    else:
        raw = pc.get("targets")
        if not isinstance(raw, dict) or not raw:
            raise TabularConfigError("pack_config needs `targets: {head_name: {label: column, ...}}` (or the "
                                     "one-head shorthand `label: column`)")
        targets = {}
        for name, t in raw.items():
            if isinstance(t, str):
                t = {"label": t}
            if not isinstance(t, dict):
                raise TabularConfigError(f"pack_config.targets.{name} must be a mapping with at least `label`")
            targets[str(name)] = dict(t)
    for name, t in targets.items():
        bad = sorted(set(t) - set(TARGET_KEYS) - set(SPEC_KEYS))
        if bad:
            raise TabularConfigError(f"target {name!r}: unknown keys {bad}; known: {sorted({*TARGET_KEYS, *SPEC_KEYS})}")
    mc = pc.get("max_categories", 50)
    if not isinstance(mc, int) or isinstance(mc, bool) or not 1 <= mc <= 10000:
        raise TabularConfigError(f"pack_config.max_categories must be an integer in 1..10000, got {mc!r}")
    hc = pc.get("hash_categories", False)
    if hc is None:
        hc = False
    if not isinstance(hc, bool):
        raise TabularConfigError(f"pack_config.hash_categories must be true or false, got {hc!r}")
    src = str(pc.get("label_source") or "ground_truth")
    if src not in SOURCES or src == "feedback":
        raise TabularConfigError(f"pack_config.label_source must be one of {[s for s in SOURCES if s != 'feedback']}")
    na = pc.get("na_values", list(DEFAULT_NA))
    if not isinstance(na, list):
        raise TabularConfigError("pack_config.na_values must be a list of strings")
    features = _cols(pc.get("features"), "features")
    categorical = _cols(pc.get("categorical"), "categorical")
    exclude = _col_list(pc.get("exclude"), "exclude")
    text = _col_list(pc.get("text"), "text")
    both = sorted(set(features or ()) & set(exclude))
    if both:
        raise TabularConfigError(f"pack_config.features and pack_config.exclude both list {both}: a column is "
                                 f"either a feature or excluded")
    both = sorted(set(text) & set(exclude))
    if both:
        raise TabularConfigError(f"pack_config.text and pack_config.exclude both list {both}: a column is either "
                                 f"a text feature or excluded")
    both = sorted(set(text) & set(categorical or ()))
    if both:
        raise TabularConfigError(f"pack_config.text and pack_config.categorical both list {both}: a column is "
                                 f"encoded one way")
    tb = pc.get("text_buckets", DEFAULT_TEXT_BUCKETS)
    if tb is None:
        tb = DEFAULT_TEXT_BUCKETS
    if not isinstance(tb, int) or isinstance(tb, bool) or not MIN_TEXT_BUCKETS <= tb <= MAX_TEXT_BUCKETS:
        raise TabularConfigError(f"pack_config.text_buckets must be an integer in {MIN_TEXT_BUCKETS}.."
                                 f"{MAX_TEXT_BUCKETS}, got {tb!r}")
    tf = pc.get("time_format")
    if tf is not None:
        if not isinstance(tf, str) or "%" not in tf:
            raise TabularConfigError(f"pack_config.time_format must be a strptime format such as "
                                     f"'%d/%m/%Y %H:%M', got {tf!r}")
        try:
            datetime.strptime(datetime(2001, 2, 3, 4, 5, 6, tzinfo=UTC).strftime(tf), tf)
        except ValueError as e:
            raise TabularConfigError(f"pack_config.time_format {tf!r} is not a usable strptime format ({e})") from None
        if not pc.get("time"):
            raise TabularConfigError("pack_config.time_format is set but pack_config.time (the column) is not")
    return TabularConfig(
        data=Path(resolve_path(data)), id=_col(pc.get("id"), "id"), group=_col(pc.get("group"), "group"),
        time=_col(pc.get("time"), "time"), segment=_col(pc.get("segment"), "segment"), time_format=tf,
        features=features, exclude=exclude, categorical=categorical, text=text, text_buckets=tb,
        max_categories=mc, hash_categories=hc, delimiter=str(pc.get("delimiter") or "auto"),
        encoding=str(pc.get("encoding") or "utf-8-sig"),
        na_values=frozenset(str(x).strip().lower() for x in na), label_source=src, targets=targets)


def config_keys() -> dict:
    """The `pack_config` keys of the tabular pack for `rlcd config check` (core/config_check.py): BASE_KEYS and
    TOLERATED_KEYS, plus the one-head shorthand (TARGET_KEYS and SPEC_KEYS at the top level, next to `label`).
    `parse_config` stays the judge of what a combination means; this only names the keys and their types."""
    from ...core.config_check import Key, head_keys

    head = head_keys()
    target = {"label": Key("str"), **{k: head[k] for k in (*TARGET_KEYS, *SPEC_KEYS) if k != "label"},
              "model": Key("str", default="auto", choices=head["model"].choices)}
    base = {
        "data": Key("str", nullable=False),
        "id": Key("str"), "group": Key("str"), "time": Key("str"), "segment": Key("str"),
        "features": Key("any", default="auto"), "categorical": Key("any", default="auto"),
        "max_categories": Key("int", default=50, min=1, max=10000, strict=True, nullable=False),
        "targets": Key("map", each=Key("any", label="head", keys=target)),
        "delimiter": Key("str", default="auto"),
        "encoding": Key("str", default="utf-8-sig"),
        "na_values": Key("list", default=list(DEFAULT_NA), nullable=False),
        "label_source": Key("str", default="ground_truth", choices=tuple(s for s in SOURCES if s != "feedback")),
        "question": Key("str", default="outcome"),
        "hash_categories": Key("bool", default=False),
        "exclude": Key("list", default=[]),
        "text": Key("list", default=[]),
        "text_buckets": Key("int", default=DEFAULT_TEXT_BUCKETS, min=MIN_TEXT_BUCKETS, max=MAX_TEXT_BUCKETS,
                            strict=True),
        "time_format": Key("str"),
        "allow_remote": Key("bool", default=False),
    }
    return {**base, **target}


def merged_targets(cfg: TabularConfig, overrides: dict) -> dict[str, dict]:
    """pack_config targets merged with rlcd.yaml `heads:` overrides (the override wins field by field). An
    override that names a head without a target must bring its own `label`. Disabled heads stay in the result
    (with enabled: false) because their label columns must still be kept out of the features."""
    out: dict[str, dict] = {}
    for name in [*cfg.targets, *[n for n in overrides if n not in cfg.targets]]:
        d = {**(cfg.targets.get(name) or {}), **(overrides.get(name) or {})}
        if "view" in d and d["view"] != "row":
            raise TabularConfigError(f"head {name!r}: the tabular pack has one feature view, 'row'")
        d.pop("view", None)
        if not d.get("label"):
            if d.get("enabled", True) is False:
                continue
            raise TabularConfigError(f"head {name!r} (rlcd.yaml heads) has no label column: add it to "
                                     f"pack_config.targets or give it `label: <column>`")
        out[name] = d
    return out


def label_columns(targets: dict[str, dict]) -> set[str]:
    return {str(t["label"]) for t in targets.values()}


# ----------------------------------------------------------------------------- targets -> heads

@dataclass(frozen=True)
class Target:
    """One head's label: which column, how its values map to the head's classes, and the HeadSpec fields."""
    head: str
    label: str
    type: str
    positive: str | None                  # noul: the value that means true
    mapping: dict[str, int]               # choice / score: canonical value -> class index
    spec: dict                            # fields for packs.base.spec_from_yaml
    how: str                              # how type / positive / classes were chosen (shown in the description)
    notes: tuple[str, ...] = ()           # what the reader should check (goes into the head's dataset notes)

    def index(self, raw: Any, na: frozenset[str]) -> int | None:
        """Class index of a label cell; None when missing or not one of the head's classes."""
        v = canon(raw, na)
        if v is None:
            return None
        if self.type == "noul":
            return int(v == self.positive)
        return self.mapping.get(v)


def _sorted_values(values: set[str]) -> list[str]:
    def key(s: str):
        try:
            return (0, float(s), s)
        except ValueError:
            return (1, 0.0, s)
    return sorted(values, key=key)


def _distinct(table: Table | None, column: str, na: frozenset[str]) -> set[str]:
    if table is None:
        raise DataUnavailable("data file unavailable")
    table.require([column], "label")
    return {v for v in (canon(x, na) for x in table.data[column]) if v is not None}


def _match(v: Any, values: set[str] | None, strict: bool, where: str) -> str:
    """A label value written in rlcd.yaml, matched to the column's values. YAML reads `yes` / `no` / `on` as
    booleans, so a boolean is matched to the column's own spelling of true / false; otherwise an exact match,
    then a unique case-insensitive one. `strict`: the value must occur in the column (when it was read)."""
    if isinstance(v, bool):
        words = TRUE_SPELLINGS if v else FALSE_SPELLINGS
        hit = sorted(x for x in (values or ()) if x.lower() in words)
        s = hit[0] if len(hit) == 1 else ("true" if v else "false")
    else:
        s = canon(v, frozenset()) or str(v)
    if values is None or s in values:
        return s
    ci = [x for x in values if x.lower() == s.lower()]
    if len(ci) == 1:
        return ci[0]
    if strict:
        raise TabularConfigError(f"{where}: value {s!r} does not occur in the column (values: {sorted(values)[:20]})")
    return s


def resolve_target(name: str, d: dict, table: Table | None, na: frozenset[str]) -> Target:
    """Decide a head's type and classes from its target config, reading the label column for what is not given
    (and to match configured values to the column's spelling). Raises DataUnavailable when inference is needed
    and there is no table."""
    label = str(d["label"])
    qtype = d.get("type")
    if qtype is not None and qtype not in ("noul", "choice", "score"):
        raise TabularConfigError(f"target {name!r}: type must be noul, choice or score, got {qtype!r}")
    values: set[str] | None = _distinct(table, label, na) if table is not None else None
    where = f"target {name!r} (column {label!r})"
    classes = [_match(c, values, False, where) for c in d["classes"]] if d.get("classes") else None
    levels = [str(x) for x in d["levels"]] if d.get("levels") else None
    how: list[str] = []
    notes: list[str] = []
    if qtype is None:
        if classes:
            qtype = "choice"
        else:
            values = _distinct(table, label, na)
            if len(values) < 2:
                raise TabularConfigError(f"{where}: {len(values)} distinct value(s); a head needs at least two")
            qtype = "noul" if len(values) == 2 else "choice"
            how.append(f"type {qtype} inferred from {len(values)} distinct values")
    spec = {k: d[k] for k in SPEC_KEYS if k in d and k not in ("enabled", "description")}
    spec.setdefault("model", "auto")
    spec.update(type=qtype, view="row")
    mapping: dict[str, int] = {}
    positive = None
    if qtype == "noul":
        if d.get("positive") is not None:
            positive = _match(d["positive"], values, True, where)
            how.append(f"true means {label} = {positive!r} (set in rlcd.yaml); every other value is false")
        else:
            values = _distinct(table, label, na)
            if len(values) > 2:
                raise TabularConfigError(f"{where}: {len(values)} values; a noul head over it needs "
                                         f"`positive: <value meaning true>`")
            word = next((v for w in POSITIVE_WORDS for v in sorted(values) if v.lower() == w), None)
            if word is not None:
                positive = word
                how.append(f"true means {label} = {positive!r} (chosen by default: a conventional 'true' value; "
                           f"set `positive` to change it)")
            else:
                positive = max(values) if values else None
                if positive is not None:
                    other = min(values)
                    notes.append(f"true means {label} = {positive!r}, false means {label} = {other!r}: rlcd.yaml does "
                                 f"not say which value is the positive class, so the alphabetically later one "
                                 f"was taken. If {other!r} is the outcome you ask about, set `positive: {other}` "
                                 f"for head {name!r}")
                how.append(f"true means {label} = {positive!r} (chosen by default as the lexicographically larger "
                           f"of {sorted(values)}: set `positive` in rlcd.yaml if that is wrong)")
        if positive is None:
            raise TabularConfigError(f"target {name!r}: cannot tell which value of {label!r} means true")
        spec["positive"] = positive
    elif qtype == "choice":
        if not classes:
            values = _distinct(table, label, na)
            classes = _sorted_values(values)
            how.append(f"classes {classes} read from the column (sorted)")
        if not 2 <= len(classes) <= MAX_CHOICE_OPTIONS:
            raise TabularConfigError(f"target {name!r}: a choice head needs 2 to {MAX_CHOICE_OPTIONS} classes, "
                                     f"column {label!r} gives {len(classes)}; set `type` / `classes`")
        mapping = {c: i for i, c in enumerate(classes)}
        spec["classes"] = classes
    else:  # score: ordered levels
        if classes:
            mapping = {c: i for i, c in enumerate(classes)}
            spec["classes"] = classes
            if levels:
                spec["levels"] = levels
        elif levels:
            mapping = {**{str(i): i for i in range(len(levels))}, **{lv: i for i, lv in enumerate(levels)}}
            spec["levels"] = levels
            how.append("labels are the level names or their 0-based indices")
        else:
            values = _distinct(table, label, na)
            nums = [v for v in values if re.fullmatch(r"[+-]?\d+", v)]
            if len(nums) != len(values):
                raise TabularConfigError(f"target {name!r}: score labels {sorted(values)[:10]} are not all integers; "
                                         f"give the order with `levels` (names in order) or `classes`")
            order = sorted(values, key=int)
            mapping = {v: i for i, v in enumerate(order)}
            spec["classes"] = order
            how.append(f"levels {order} read from the column (numeric order)")
        n = len(spec.get("classes") or spec.get("levels") or [])
        if not MIN_SCORE_LEVELS <= n <= MAX_SCORE_LEVELS:
            raise TabularConfigError(f"target {name!r}: a score head needs {MIN_SCORE_LEVELS} to {MAX_SCORE_LEVELS} "
                                     f"levels, got {n}")
    text = f"from column {label!r}"
    if qtype == "noul":
        text = f"P({label} = {positive!r}) from column {label!r}"
    detail = "; ".join([text, *how])
    desc = str(d.get("description") or "").strip()
    spec["description"] = f"{desc} [{detail}]" if desc else detail
    return Target(head=name, label=label, type=qtype, positive=positive, mapping=mapping, spec=spec,
                  how="; ".join(how), notes=tuple(notes))
