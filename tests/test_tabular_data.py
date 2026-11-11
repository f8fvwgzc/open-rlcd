"""Tabular pack, no training: reading files, pack_config, label inference, the feature schema, dataset rows."""
from __future__ import annotations

import gzip
import json
import math
import re
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest

from open_rlcd.core import splits as S
from open_rlcd.core.config import Project
from open_rlcd.core.contract import ContractError
from open_rlcd.core.service import Service
from open_rlcd.packs.base import load_pack, switched_off
from open_rlcd.packs.tabular import (
    TabularConfigError,
    TabularPack,
    TabularProvider,
    generate,
    read_table,
    synthetic_pack_config,
    true_probabilities,
    write_synthetic_csv,
)
from open_rlcd.packs.tabular.data import DEFAULT_NA, DataUnavailable, canon, parse_time

NA = frozenset(DEFAULT_NA)


def project(tmp_path: Path, pack_config: dict, **extra) -> Project:
    return Project(root=tmp_path, data={"pack": "tabular", "pack_config": pack_config, **extra})


@pytest.fixture
def synth(tmp_path):
    path = tmp_path / "churn.csv"
    write_synthetic_csv(path, n=600, seed=3)
    return path


def write(path: Path, text: str, encoding: str = "utf-8") -> Path:
    path.write_text(text, encoding=encoding)
    return path


# ----------------------------------------------------------------------------- reading files

def test_read_csv_semicolons_quoting_and_bom(tmp_path):
    p = write(tmp_path / "bank.csv", '﻿"age";"job";"note";"y"\n30;"admin.";"a;b ""quoted""";"no"\n\n'
                                     '41;"blue-collar";"";"yes"\n')
    t = read_table(p)
    assert t.columns == ["age", "job", "note", "y"]          # BOM dropped, delimiter sniffed
    assert t.delimiter == ";"
    assert t.data["note"] == ['a;b "quoted"', ""]
    assert t.n == 2                                          # blank line skipped


def test_read_csv_tsv_and_gzip(tmp_path):
    t = read_table(write(tmp_path / "x.tsv", "a\tb\n1\t2\n"))
    assert t.data == {"a": ["1"], "b": ["2"]}
    gz = tmp_path / "x.csv.gz"
    with gzip.open(gz, "wt", encoding="utf-8") as f:
        f.write("a,b\n1,2\n3,4\n")
    assert read_table(gz).data["b"] == ["2", "4"]


def test_read_csv_ragged_row_names_the_line(tmp_path):
    p = write(tmp_path / "bad.csv", "a,b\n1,2\n3,4,5\n")
    with pytest.raises(TabularConfigError, match="line 3 has 3 fields"):
        read_table(p)


def test_read_csv_rejects_duplicate_header_and_unknown_suffix(tmp_path):
    with pytest.raises(TabularConfigError, match="duplicate"):
        read_table(write(tmp_path / "d.csv", "a,a\n1,2\n"))
    with pytest.raises(TabularConfigError, match="unsupported file type"):
        read_table(write(tmp_path / "d.xlsx", "x"))
    with pytest.raises(DataUnavailable):
        read_table(tmp_path / "nope.csv")


def test_parquet_reads_or_names_the_extra(tmp_path):
    p = tmp_path / "t.parquet"
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError:
        p.write_bytes(b"PAR1")
        with pytest.raises(ImportError, match="'parquet' extra"):
            read_table(p)
        return
    pq.write_table(pa.table({"x": [1.5, None], "c": ["a", "b"], "y": [True, False]}), p)
    t = read_table(p)
    assert t.data["x"] == [1.5, None] and t.data["y"] == [True, False]


def test_cell_canonicalisation_and_times():
    assert canon("1.0", NA) == "1" and canon(3.0, NA) == "3" and canon(3, NA) == "3"
    assert canon("02139", NA) == "02139"                     # leading zeros kept
    assert canon(True, NA) == "true" and canon(" a ", NA) == "a"
    assert canon("NA", NA) is None and canon("", NA) is None and canon(float("nan"), NA) is None
    jan = int(datetime(2024, 1, 31, tzinfo=UTC).timestamp())
    assert parse_time("2024-01-31", NA) == jan               # naive = UTC, never the machine's zone
    assert parse_time("2024-01-31T00:00:00Z", NA) == jan
    assert parse_time(str(jan), NA) == jan and parse_time(jan * 1000, NA) == jan
    assert parse_time("not a time", NA) is None and parse_time("", NA) is None


# ----------------------------------------------------------------------------- pack_config and heads

def test_shorthand_and_question_name(tmp_path):
    p = write(tmp_path / "t.csv", "x,y\n1,yes\n2,no\n")
    pack = TabularPack()
    assert list(pack.heads(project(tmp_path, {"data": "t.csv", "label": "y"}))) == ["outcome"]
    heads = pack.heads(project(tmp_path, {"data": str(p), "label": "y", "question": "will_buy", "stakes": "high"}))
    assert heads["will_buy"].stakes == "high" and heads["will_buy"].view == "row"
    assert heads["will_buy"].model == "auto"


@pytest.mark.parametrize("pc, match", [
    ({"data": "t.csv", "label": "y", "featurs": "auto"}, "unknown keys"),
    ({"data": "t.csv", "label": "y", "targets": {"a": {"label": "y"}}}, "both"),
    ({"data": "https://example.com/t.csv", "label": "y"}, "local file"),
    ({"data": "t.csv"}, "targets"),
    ({"label": "y"}, "data is required"),
    ({"data": "t.csv", "targets": {"a": {"label": "y", "colour": 1}}}, "unknown keys"),
    ({"data": "t.csv", "label": "y", "label_source": "feedback"}, "label_source"),
])
def test_config_errors(tmp_path, pc, match):
    write(tmp_path / "t.csv", "x,y\n1,yes\n2,no\n")
    with pytest.raises(TabularConfigError, match=match):
        TabularPack().heads(project(tmp_path, pc))


def test_noul_positive_defaults_and_says_which(tmp_path):
    write(tmp_path / "t.csv", "x,y,z,w\n1,yes,approved,Y\n2,no,rejected,N\n3,no,rejected,N\n")
    pack = TabularPack()
    h = pack.heads(project(tmp_path, {"data": "t.csv", "targets": {"a": "y", "b": "z"}}))
    assert h["a"].positive == "yes" and "conventional" in h["a"].description
    assert h["b"].positive == "rejected" and "lexicographically larger" in h["b"].description
    # YAML reads `positive: yes` as True: it is matched to the column's own spelling
    h = pack.heads(project(tmp_path, {"data": "t.csv", "targets": {"c": {"label": "w", "positive": True}}}))
    assert h["c"].positive == "Y"
    with pytest.raises(TabularConfigError, match="does not occur"):
        pack.heads(project(tmp_path, {"data": "t.csv", "targets": {"d": {"label": "z", "positive": "maybe"}}}))


def test_choice_and_score_inference(tmp_path):
    write(tmp_path / "t.csv", "x,c,s,v\n1,b,2,low\n2,a,0,high\n3,c,1,mid\n4,a,3,low\n")
    pack = TabularPack()
    h = pack.heads(project(tmp_path, {"data": "t.csv", "targets": {
        "c": "c", "s": {"label": "s", "type": "score"},
        "v": {"label": "v", "type": "score", "classes": ["low", "mid", "high"]},
        "n": {"label": "v", "type": "score", "levels": ["low", "mid", "high"]}}}))
    assert h["c"].type == "choice" and h["c"].classes == ("a", "b", "c")
    assert h["s"].type == "score" and h["s"].classes == ("0", "1", "2", "3")
    assert h["v"].classes == ("low", "mid", "high")
    assert h["n"].classes == ("0", "1", "2") and h["n"].legend == ("low", "mid", "high")
    with pytest.raises(TabularConfigError, match="not all integers"):
        pack.heads(project(tmp_path, {"data": "t.csv", "targets": {"v": {"label": "v", "type": "score"}}}))
    with pytest.raises(TabularConfigError, match="positive"):
        pack.heads(project(tmp_path, {"data": "t.csv", "targets": {"c": {"label": "c", "type": "noul"}}}))


def test_head_overrides_merge_disable_and_add(tmp_path, synth):
    pc = synthetic_pack_config(synth)
    heads = {"churned": {"stakes": "high", "model": "logreg"}, "satisfaction": {"enabled": False},
             "big_spender": {"label": "plan", "positive": "premium"}}
    pr = project(tmp_path, pc, heads=heads)
    pack = TabularPack()
    h = pack.heads(pr)
    assert set(h) == {"churned", "plan_tier_next", "big_spender"}
    assert h["churned"].stakes == "high" and h["churned"].model == "logreg"
    names = pack.provider(pr).names("row")
    # a disabled head's label column is still an outcome, never a feature; a new head's label neither
    assert not any(n.startswith(("satisfaction", "plan=", "churned", "plan_tier_next")) for n in names)
    with pytest.raises(TabularConfigError, match="no label column"):
        pack.heads(project(tmp_path, pc, heads={"ghost": {"stakes": "low"}}))


def test_heads_without_the_data_file(tmp_path):
    pc = {"data": "missing.csv", "targets": {"known": {"label": "y", "type": "noul", "positive": "yes"},
                                             "inferred": {"label": "z"}}}
    pack = TabularPack()
    assert list(pack.heads(project(tmp_path, pc))) == ["known"]
    with pytest.raises(DataUnavailable, match="not found"):
        pack.provider(project(tmp_path, pc))


def test_switched_off_names_every_head_the_project_does_not_serve(tmp_path, synth):
    """`TabularPack.switched_off`, the one list the service, `rlcd check` and config check read: a target switched
    off in pack_config or under `heads:` (the override wins field by field, as in `merged_targets`), a head `heads:`
    switches off, and a trained head that is no longer a target. `heads()` serves none of them, and the list needs
    rlcd.yaml only, not the data file."""
    pack = TabularPack()
    pc = synthetic_pack_config(synth)
    pc["targets"]["plan_tier_next"]["enabled"] = False
    pr = project(tmp_path, pc)
    assert set(pack.heads(pr)) == {"churned", "satisfaction"} and pack.switched_off(pr) == {"plan_tier_next"}
    trained = ["churned", "plan_tier_next", "satisfaction", "renewed"]       # "renewed": a target since removed
    assert pack.switched_off(pr, trained) == {"plan_tier_next", "renewed"}
    heads = {"churned": {"enabled": False}, "plan_tier_next": {"enabled": True}, "ghost": {"enabled": False}}
    pr = project(tmp_path, pc, heads=heads)
    assert set(pack.heads(pr)) == {"plan_tier_next", "satisfaction"}
    assert pack.switched_off(pr, trained) == {"churned", "ghost", "renewed"}
    # a pack without the hook (not a BasePack) gets BasePack's rule: what `heads:` switches off
    assert switched_off(object(), pr, trained) == {"churned", "ghost"}
    # the shorthand's `enabled` switches its one head off; a head trained under the shorthand's old name is gone
    one = project(tmp_path, {"data": str(synth), "label": "churned", "question": "churn", "enabled": False})
    assert pack.heads(one) == {} and pack.switched_off(one, ["churn", "outcome"]) == {"churn", "outcome"}
    synth.unlink()
    pr = project(tmp_path, pc)
    assert pack.switched_off(pr, trained) == {"plan_tier_next", "renewed"}
    assert list(pack.heads(pr)) == ["satisfaction"]       # churned's classes come from the file: back with it


def test_policy_holds_by_default_and_rlcd_yaml_overrides(tmp_path, synth):
    assert TabularPack().policy_defaults() == {"abstain": "hold"}
    assert isinstance(load_pack("tabular"), TabularPack)
    svc = Service(project(tmp_path, synthetic_pack_config(synth)), home=tmp_path / "var")
    assert svc.policy_config().abstain == "hold"
    svc = Service(project(tmp_path, synthetic_pack_config(synth), policy={"abstain": "escalate"}),
                  home=tmp_path / "var2")
    assert svc.policy_config().abstain == "escalate"


# ----------------------------------------------------------------------------- the provider

def test_provider_schema(tmp_path, synth):
    pr = project(tmp_path, synthetic_pack_config(synth))
    prov = TabularPack().provider(pr)
    names = prov.names("row")
    assert "tenure_months" in names and "monthly_charges" in names
    assert {"plan=basic", "plan=__other__", "plan=__missing__", "paperless=yes"} <= set(names)
    for col in ("row_id", "customer_id", "month", "channel", "churned", "plan_tier_next", "satisfaction"):
        assert not any(n == col or n.startswith(col + "=") for n in names)
    assert re.fullmatch(r"tabular-v1:[0-9a-f]{12}", prov.feature_version)
    assert prov.groups("row")["plan=premium"] == "plan"
    d = prov.describe("row")
    assert d["columns"]["plan"]["type"] == "category" and d["columns"]["tenure_months"]["type"] == "number"
    assert set(d["required"]) == {"tenure_months", "monthly_charges", "support_calls", "plan", "contract",
                                  "region", "paperless"}
    with pytest.raises(ContractError, match="one feature view"):
        prov.names("query")


def state(**kw) -> dict:
    base = {"tenure_months": 12, "monthly_charges": 60.0, "support_calls": 1, "plan": "basic",
            "contract": "one-year", "region": "north", "paperless": "no"}
    return {**base, **kw}


def test_transform_unseen_missing_and_unparseable(tmp_path, synth):
    prov = TabularPack().provider(project(tmp_path, synthetic_pack_config(synth)))
    names = prov.names("row")
    x = prov.transform("row", [state(plan="enterprise", region=None, monthly_charges="abc", support_calls="3")])[0]
    col = dict(zip(names, x.tolist(), strict=True))
    assert col["plan=__other__"] == 1 and col["plan=basic"] == 0
    assert col["region=__missing__"] == 1
    assert math.isnan(col["monthly_charges"]) and col["support_calls"] == 3.0
    # JSON numbers and CSV strings give the same row
    a = prov.transform("row", [state(tenure_months="12", monthly_charges="60.0")])
    b = prov.transform("row", [state()])
    assert np.array_equal(a, b)


def test_transform_refuses_missing_columns(tmp_path, synth):
    prov = TabularPack().provider(project(tmp_path, synthetic_pack_config(synth)))
    s = state()
    del s["plan"], s["tenure_months"]
    with pytest.raises(ContractError) as e:
        prov.transform("row", [s])
    assert e.value.status == 422 and set(e.value.details["missing_columns"]) == {"plan", "tenure_months"}
    with pytest.raises(ContractError, match="object"):
        prov.transform("row", [[1, 2, 3]])


def test_schema_is_stable_when_only_frequencies_shift(tmp_path):
    write(tmp_path / "t.csv", "c,x,y\na,1,yes\nb,2,no\nb,3,no\n")
    pr = project(tmp_path, {"data": "t.csv", "label": "y"})
    fv = TabularPack().provider(pr).feature_version
    write(tmp_path / "t.csv", "c,x,y\na,1,yes\nb,2,no\nb,3,no\na,4,yes\na,5,no\n")   # 'a' is now the most frequent
    prov = TabularPack().provider(pr)
    assert prov.feature_version == fv
    x = prov.transform("row", [{"c": "A", "x": 1}])[0]                                  # case-only difference
    assert dict(zip(prov.names("row"), x.tolist(), strict=True))["c=a"] == 1


def test_provider_config_round_trip(tmp_path, synth):
    prov = TabularPack().provider(project(tmp_path, synthetic_pack_config(synth)))
    cfg = json.loads(json.dumps(prov.to_config()))           # what the bundle stores
    again = TabularPack().provider(project(tmp_path, {"data": "gone.csv", "label": "y"}), cfg)
    assert again.feature_version == prov.feature_version and again.names("row") == prov.names("row")
    rows = [state(), state(plan="premium", region="east"), state(monthly_charges=None)]
    assert np.array_equal(again.transform("row", rows), prov.transform("row", rows), equal_nan=True)
    cfg["columns"][0]["name"] = "renamed"
    with pytest.raises(TabularConfigError, match="modified"):
        TabularProvider.from_config(cfg)


def test_vocabulary_cap_and_identifier_columns(tmp_path):
    rng = np.random.default_rng(0)
    lines = ["city,email,amount,y"]
    for i in range(300):
        lines.append(f"c{int(rng.integers(0, 10))},user{i}@example.com,{rng.normal():.3f},{'yes' if i % 3 else 'no'}")
    write(tmp_path / "t.csv", "\n".join(lines) + "\n")
    pr = project(tmp_path, {"data": "t.csv", "label": "y", "max_categories": 3})
    prov = TabularPack().provider(pr)
    names = prov.names("row")
    assert len([n for n in names if n.startswith("city=")]) == 5      # top 3 + __other__ + __missing__
    assert not any(n.startswith("email") for n in names)               # identifier-like: left out
    assert any("identifier" in n for n in prov.notes) and any("most frequent" in n for n in prov.notes)
    # listed explicitly, it is used; a numeric-looking column listed as categorical is one-hot encoded
    pr = project(tmp_path, {"data": "t.csv", "label": "y", "features": ["email", "amount", "city"],
                            "categorical": ["amount"], "max_categories": 3})
    names = TabularPack().provider(pr).names("row")
    assert any(n.startswith("email=") for n in names) and "amount" not in names
    with pytest.raises(TabularConfigError, match="leaks"):
        TabularPack().provider(project(tmp_path, {"data": "t.csv", "label": "y", "features": ["y", "city"]}))


# ----------------------------------------------------------------------------- dataset rows

def test_datasets_rows_labels_and_keys(tmp_path):
    write(tmp_path / "t.csv", "id,cust,t,seg,x,c,y\n"
                              "a,k1,2024-01-01,s1,1.0,u,yes\n"
                              "b,k1,2024-02-01,s1,2.0,v,no\n"
                              "c,k2,2024-03-01,,3.0,u,\n"
                              "d,,2024-04-01,s2,4.0,w,yes\n")
    pc = {"data": "t.csv", "id": "id", "group": "cust", "time": "t", "segment": "seg", "label": "y",
          "label_source": "human"}
    pr = project(tmp_path, pc)
    pack = TabularPack()
    prov = pack.provider(pr)
    assert prov.required() == ["x", "c"]
    ds = pack.datasets(pr, prov, ["outcome"])["outcome"]
    assert list(ds.ids) == ["a", "b", "d"] and list(ds.y) == [1, 0, 1]
    assert list(ds.group) == ["k1", "k1", "row:3"] and list(ds.segment) == ["s1", "s1", "s2"]
    assert ds.t[0] == int(datetime(2024, 1, 1, tzinfo=UTC).timestamp())
    assert set(ds.source) == {"human"} and ds.trusted.all()
    assert any("1 rows have no label" in n for n in ds.notes)
    assert ds.feature_groups["c=u"] == "c"


def test_time_only_table_orders_by_time(tmp_path):
    path = tmp_path / "churn.csv"
    write_synthetic_csv(path, n=400, seed=1)
    pc = {**synthetic_pack_config(path), "group": None}
    pr = project(tmp_path, pc)
    pack = TabularPack()
    ds = pack.datasets(pr, pack.provider(pr), ["churned"])["churned"]
    assert ds.group is None and ds.t is not None and S.resolve_mode(ds) == "time"


def test_group_ordering_keeps_a_customer_on_one_side(tmp_path):
    path = tmp_path / "churn.csv"
    write_synthetic_csv(path, n=2000, seed=5)
    pr = project(tmp_path, synthetic_pack_config(path))
    pack = TabularPack()
    ds = pack.datasets(pr, pack.provider(pr), ["churned"])["churned"]
    split = S.holdout_split(ds, 0.2, "auto", 7)
    assert split.mode == "group"
    dev, hold = set(ds.group[split.dev]), set(ds.group[split.holdout])
    assert hold and dev and not dev & hold
    first = {}
    for g, t in zip(ds.group, ds.t, strict=True):
        first[g] = min(first.get(g, t), t)
    # the holdout is the customers who appeared last
    assert min(first[g] for g in hold) >= max(first[g] for g in dev)


# ----------------------------------------------------------------------------- synthetic data

def test_synthetic_data_is_deterministic_and_calibrated_by_construction(tmp_path):
    rows, truth = generate(3000, seed=11)
    again, _ = generate(3000, seed=11)
    assert rows == again
    assert len({r["customer_id"] for r in rows}) < len(rows)              # several rows per customer
    y = np.array([r["churned"] == "yes" for r in rows])
    assert abs(y.mean() - truth["churned"].mean()) < 0.03
    assert np.allclose(truth["plan_tier_next"].sum(1), 1) and truth["satisfaction"].shape == (3000, 4)
    noise = true_probabilities(rows, noise=True)
    assert np.ptp(noise["churned"]) == 0                                   # independent of the features
    info = TabularPack().synth(seed=2, n=50, path=tmp_path / "s.csv")
    assert info["rows"] == 50 and info["pack_config"]["data"] == str(tmp_path / "s.csv")


# ----------------------------------------------------------------------------- hashed vocabularies (F23)

def _homed(root: Path, pack_config: dict) -> Project:
    pr = project(root, pack_config)
    pr.home_override = root / "var"          # its own home (RLCD_HOME is shared by the whole test session)
    return pr


def _with_emails(tmp_path: Path, n: int = 600) -> Path:
    """The synthetic churn table plus a personal-data column that repeats per customer (not identifier-like)."""
    import csv

    raw = tmp_path / "raw.csv"
    write_synthetic_csv(raw, n=n, seed=3)
    rows = list(csv.DictReader(raw.open()))
    for r in rows:
        r["contact_email"] = f"{r['customer_id']}.person@example.com"
    out = tmp_path / "customers.csv"
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return out


def test_hashed_vocabularies_keep_raw_values_out_and_transform_identically(tmp_path):
    from open_rlcd.packs.tabular.provider import category_hash

    data = _with_emails(tmp_path)
    pc = synthetic_pack_config(data)
    raw = TabularPack().provider(_homed(tmp_path / "a", pc))
    hashed = TabularPack().provider(_homed(tmp_path / "b", {**pc, "hash_categories": True}))
    cfg = json.dumps(hashed.to_config())
    names = hashed.names("row")
    assert "person@example.com" not in cfg and "basic" not in cfg and "premium" not in cfg
    assert not any("example.com" in n or n == "plan=basic" for n in names)
    assert "plan=__other__" in names and "plan=__missing__" in names and "tenure_months" in names
    assert any(n.startswith("contact_email=h:") for n in names)
    assert "person@example.com" in json.dumps(raw.to_config())         # raw mode: unchanged (and warned about)
    assert hashed.data_card()["tabular"]["categorical_values"] == "salted sha256 hashes"
    assert any("salted SHA-256" in n for n in hashed.notes) and any("hash_categories" in n for n in raw.notes)
    # same rows -> the same one-hot positions, column by column (hashed blocks are ordered by hash)
    first = next(c for c in raw.columns if c.name == "contact_email").vocabulary[0]
    rows = [state(contact_email=first), state(plan="PREMIUM", region=None, contact_email=first.upper()),
            state(plan="enterprise", contact_email="nobody@example.com"), state(paperless="Yes", contact_email=None)]
    xr, xh = raw.transform("row", rows), hashed.transform("row", rows)
    rn, hn = raw.names("row"), hashed.names("row")
    for name in rn:
        col, sep, val = name.partition("=")
        if sep and val not in ("__other__", "__missing__"):
            name_h = f"{col}={category_hash(hashed.salt, val)}"
        else:
            name_h = name
        assert np.array_equal(xr[:, rn.index(name)], xh[:, hn.index(name_h)], equal_nan=True), name
    assert len(rn) == len(hn)
    # what /v1/heads shows: no category values for a hashed column
    d = hashed.describe("row")["columns"]["plan"]
    assert d["categories"] == [] and d["hashed"] is True and d["more_categories"] == 3


def test_hashed_provider_round_trip_and_tamper_checks(tmp_path):
    data = _with_emails(tmp_path)
    pr = _homed(tmp_path, {**synthetic_pack_config(data), "hash_categories": True})
    prov = TabularPack().provider(pr)
    cfg = json.loads(json.dumps(prov.to_config()))                      # what the bundle stores
    again = TabularProvider.from_config(cfg)
    assert again.feature_version == prov.feature_version and again.names("row") == prov.names("row")
    rows = [state(contact_email="C00003.person@example.com"), state(plan="Premium", contact_email=None)]
    assert np.array_equal(again.transform("row", rows), prov.transform("row", rows), equal_nan=True)
    bad = json.loads(json.dumps(cfg))
    bad["category_salt"] = "0" * 32                                     # another salt: other hashes
    with pytest.raises(TabularConfigError, match="modified"):
        TabularProvider.from_config(bad)
    bad = json.loads(json.dumps(cfg))
    del bad["category_salt"]
    with pytest.raises(TabularConfigError, match="salt"):
        TabularProvider.from_config(bad)
    # one salt per project home: a retrain keeps the schema (heads it does not retrain carry over) ...
    assert TabularPack().provider(pr).feature_version == prov.feature_version
    salt_file = pr.home / "tabular" / "category_salt"
    assert salt_file.read_text().strip() == prov.salt and (salt_file.stat().st_mode & 0o077) == 0
    # ... and another home hashes differently
    other = TabularPack().provider(_homed(tmp_path / "other", {**synthetic_pack_config(data),
                                                                 "hash_categories": True}))
    assert other.salt != prov.salt and other.feature_version != prov.feature_version


def test_raw_schema_hash_is_unchanged_by_the_option(tmp_path, synth):
    """Bundles trained before hash_categories existed must keep validating: raw mode hashes the same schema."""
    import hashlib

    prov = TabularPack().provider(project(tmp_path, synthetic_pack_config(synth)))
    old_schema = {"columns": [{"name": c.name, "kind": c.kind, **({"vocabulary": list(c.vocabulary)}
                                                                  if c.kind == "categorical" else {})}
                              for c in prov.columns], "na_values": sorted(prov.na_values)}
    text = json.dumps(old_schema, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    assert prov.feature_version == f"tabular-v1:{hashlib.sha256(text.encode()).hexdigest()[:12]}"
    assert "category_salt" not in prov.to_config()


def test_hash_categories_must_be_a_boolean(tmp_path, synth):
    with pytest.raises(TabularConfigError, match="hash_categories"):
        TabularPack().provider(project(tmp_path, {**synthetic_pack_config(synth), "hash_categories": "yes"}))
