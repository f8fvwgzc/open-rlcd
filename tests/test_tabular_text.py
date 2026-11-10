"""Tabular pack: free-text columns (`pack_config.text`) as stateless hashed n-gram features, `exclude`, the
leakage note, `time_format`, serve-time input warnings. Offline; logistic regression."""
from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from open_rlcd.core.config import Project
from open_rlcd.core.contract import ContractError, DecideRequest, SystemOneRequest, TrainRequest
from open_rlcd.core.service import Service
from open_rlcd.packs.tabular import TabularConfigError, TabularPack, TabularProvider
from open_rlcd.packs.tabular.pack import leak_notes

GOOD = ("client visit", "team offsite", "licence renewal", "conference fee", "train to customer site",
        "顧客訪問の交通費", "үйлчлүүлэгчтэй уулзалт")
BAD = ("personal gift", "weekend minibar", "no receipt attached", "upgrade to first class", "birthday party drinks",
       "個人的な贈り物", "хувийн бэлэг")
FILLER = ("for", "the", "per", "see", "ref", "invoice", "project", "team", "tokyo", "ulaanbaatar")


def claims(path: Path, n: int = 1200, seed: int = 5, leak: bool = False) -> Path:
    """Expense claims whose outcome depends on the amount and on a phrase inside an otherwise unique note."""
    r = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        bad = r.random() < 0.4
        phrase = (BAD if bad else GOOD)[int(r.integers(7))]
        note = f"{FILLER[int(r.integers(10))]} {phrase} {FILLER[int(r.integers(10))]} {int(r.integers(1000, 9999))}"
        amount = float(np.exp(r.normal(4.5, 1.0)))
        p = 1 / (1 + np.exp(-(2.2 - 4.0 * bad - 0.5 * (np.log(amount) - 4.5))))
        ok = r.random() < p
        row = {"claim_id": f"C{i:05d}", "amount": f"{amount:.2f}", "category": ("travel", "meals", "other")[i % 3],
               "note": note, "coin": ("heads", "tails")[int(r.integers(2))],
               "outcome": "approved" if ok else "rejected"}
        if leak:   # filled in by the reviewer afterwards
            row["rejection_reason"] = "" if ok else ("policy", "no receipt", "over limit")[int(r.integers(3))]
            row["refund"] = f"{amount:.2f}" if ok else "0"
        rows.append(row)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return path


def service(tmp: Path, pack_config: dict) -> Service:
    pc = {"id": "claim_id", "label": "outcome", "positive": "approved", "model": "logreg", **pack_config}
    return Service(Project(root=tmp, data={"pack": "tabular", "pack_config": pc}), home=tmp / "var")


def train(svc: Service) -> dict:
    job = svc.trainer.start(TrainRequest(), wait=True)
    assert job["status"] == "succeeded", job.get("error")
    return job


def provider(tmp: Path, pack_config: dict) -> TabularProvider:
    pr = Project(root=tmp, data={"pack": "tabular", "pack_config": {"label": "outcome", **pack_config}})
    return TabularPack().provider(pr)


@pytest.fixture(scope="module")
def data(tmp_path_factory) -> Path:
    return claims(tmp_path_factory.mktemp("claims") / "claims.csv")


@pytest.fixture(scope="module")
def trained(tmp_path_factory, data):
    tmp = tmp_path_factory.mktemp("text")
    svc = service(tmp, {"data": str(data), "features": ["amount", "category"], "text": ["note"]})
    return {"svc": svc, "job": train(svc), "tmp": tmp}


# ----------------------------------------------------------------------------- text features

def test_text_features_carry_the_signal_one_hot_does_not(tmp_path, data, trained):
    with_text = trained["job"]["heads"]["outcome"]["metrics"]
    svc = service(tmp_path, {"data": str(data), "features": ["amount", "category", "note"]})
    job = train(svc)
    one_hot = job["heads"]["outcome"]["metrics"]
    assert with_text["has_skill"] and with_text["brier_skill_score"] > one_hot["brier_skill_score"] + 0.10
    assert with_text["log_loss"] < one_hot["log_loss"] - 0.05
    # the one-hot run says what is wrong with it, and that the strings are in the bundle
    notes = " ".join(job["heads"]["outcome"]["notes"])
    assert "holds mostly unique long strings" in notes and "pack_config.text" in notes
    assert "pack_config.exclude" in notes and "stored word for word in every bundle" in notes


def test_nothing_of_the_text_is_stored(trained, data):
    svc = trained["svc"]
    version = svc.resolve(None)
    cfg = svc.provider(version).to_config()
    col = next(c for c in cfg["columns"] if c["name"] == "note")
    assert col == {"name": "note", "kind": "text", "buckets": 256,
                   "encoder": {"kind": "hashing", "dim": 256, "seed": 0, "version": 1, "ngrams": [2, 4],
                               "norm": "nfkc+casefold"}}
    blob = json.dumps(svc.bundle(version).meta, ensure_ascii=False) + json.dumps(cfg, ensure_ascii=False)
    for phrase in (*GOOD, *BAD):
        assert phrase not in blob
    names = svc.provider(version).names("row")
    assert "text:note#0" in names and "text:note#255" in names and len(names) == 1 + 5 + 256
    assert set(svc.provider(version).groups("row").values()) == {"amount", "category", "note"}
    assert svc.provider(version).describe("row")["columns"]["note"]["type"] == "text"


def test_text_rows_are_normalised_and_language_independent(tmp_path, data):
    prov = provider(tmp_path, {"data": str(data), "features": ["amount"], "text": ["note"], "text_buckets": 64})
    states = [{"amount": 1, "note": "Client visit, Tokyo"}, {"amount": 1, "note": "client VISIT  tokyo"},
              {"amount": 1, "note": "顧客訪問の交通費"}, {"amount": 1, "note": "顧客訪問の宿泊費"},
              {"amount": 1, "note": "үйлчлүүлэгчтэй уулзалт"}, {"amount": 1, "note": None},
              {"amount": 1, "note": "   "}]
    X = prov.transform("row", states)[:, 1:]
    assert X.shape == (7, 64) and X.dtype == np.float32
    assert np.allclose(np.linalg.norm(X[:5], axis=1), 1.0, atol=1e-5)          # L2-normalised
    assert np.allclose(X[0], X[1])                                             # case, punctuation, spacing
    assert not X[5].any() and not X[6].any()                                   # missing text: the zero vector
    cos = X[:5] @ X[:5].T
    assert cos[2, 3] > 0.4 > abs(cos[2, 0])       # Japanese without spaces: shared character n-grams count
    assert abs(cos[4, 0]) < 0.4 and X[4].any()    # Mongolian Cyrillic is hashed like any other script
    # sublinear term frequency: repeating a word ten times does not make it ten times as heavy
    a = prov.transform("row", [{"amount": 1, "note": "taxi receipt"}])[0, 1:]
    b = prov.transform("row", [{"amount": 1, "note": "taxi " * 10 + "receipt"}])[0, 1:]
    assert float(a @ b) > 0.75


def test_hashing_is_the_same_in_another_process(tmp_path, data):
    """No Python hash(): a process with another PYTHONHASHSEED builds the same rows, and the bucket of a known
    n-gram is pinned so a change of the hashing cannot go unnoticed."""
    prov = provider(tmp_path, {"data": str(data), "features": ["amount"], "text": ["note"]})
    states = [{"amount": 3, "note": t} for t in ("weekend minibar 4411", "顧客訪問の交通費", "хувийн бэлэг")]
    here = prov.transform("row", states)
    code = ("import json,sys,numpy as np\n"
            "from open_rlcd.packs.tabular import TabularProvider\n"
            "c=json.load(open(sys.argv[1]));s=json.load(open(sys.argv[2]))\n"
            "np.save(sys.argv[3], TabularProvider.from_config(c).transform('row', s))\n")
    (tmp_path / "cfg.json").write_text(json.dumps(prov.to_config()), encoding="utf-8")
    (tmp_path / "states.json").write_text(json.dumps(states), encoding="utf-8")
    for seed in ("1", "4242"):
        subprocess.run([sys.executable, "-c", code, str(tmp_path / "cfg.json"), str(tmp_path / "states.json"),
                        str(tmp_path / f"x{seed}.npy")], check=True, env={"PYTHONHASHSEED": seed, "PATH": ""},
                       cwd=Path(__file__).resolve().parents[1])
        assert np.array_equal(np.load(tmp_path / f"x{seed}.npy"), here)
    one = prov.transform("row", [{"amount": 3, "note": "a"}])[0, 1:]
    assert np.flatnonzero(one).tolist() == PINNED_BUCKETS_OF_A


PINNED_BUCKETS_OF_A = [17, 123, 129, 234]    # "a": the word, and the n-grams "<a", "a>", "<a>"


def test_text_schema_round_trips_and_changes_the_feature_version(tmp_path, data, trained):
    plain = provider(tmp_path / "a", {"data": str(data), "features": ["amount", "category"]})
    text = provider(tmp_path / "b", {"data": str(data), "features": ["amount", "category"], "text": ["note"]})
    wide = provider(tmp_path / "c", {"data": str(data), "features": ["amount", "category"], "text": ["note"],
                                     "text_buckets": 64})
    assert len({plain.feature_version, text.feature_version, wide.feature_version}) == 3
    again = TabularProvider.from_config(json.loads(json.dumps(text.to_config())))
    assert again.feature_version == text.feature_version and again.names("row") == text.names("row")
    s = [{"amount": "12.5", "category": "meals", "note": "weekend minibar"}]
    assert np.array_equal(again.transform("row", s), text.transform("row", s))
    # a config whose text hashing is not this build's is refused, not silently re-hashed
    bad = json.loads(json.dumps(text.to_config()))
    bad["columns"][-1]["encoder"]["ngrams"] = [1, 3]
    with pytest.raises(TabularConfigError, match="hashes with"):
        TabularProvider.from_config(bad)
    # a model trained without the text column refuses precomputed rows of the schema with it (and vice versa)
    svc = trained["svc"]
    x = plain.transform("row", [{"amount": 1, "category": "meals"}])[0].tolist()
    with pytest.raises(ContractError) as e:
        svc.systemone(SystemOneRequest(state={"x": x, "feature_version": plain.feature_version},
                                       questions={"q": {"head": "outcome", "type": "noul"}}))
    assert e.value.kind == "feature_version_mismatch"
    with pytest.raises(ContractError, match="missing column"):
        svc.systemone(SystemOneRequest(state={"amount": 1, "category": "meals"},
                                       questions={"q": {"head": "outcome", "type": "noul"}}))


def test_bundle_export_import_gives_identical_probabilities(tmp_path, trained):
    svc = trained["svc"]
    version = svc.resolve(None)
    states = [{"amount": 80, "category": "meals", "note": "weekend minibar 1234"},
              {"amount": 80, "category": "meals", "note": "ref client visit tokyo 9876"},
              {"amount": None, "category": "travel", "note": "個人的な贈り物"},
              {"amount": 15.5, "category": "other", "note": None}]
    spec = svc.spec("outcome", version)
    before, calibrated = svc.probabilities(version, spec, states)
    yes = spec.labels.index("true")
    assert calibrated and before[0, yes] < before[1, yes]      # P(approved): the minibar is the worse claim
    out = svc.registry.export(version, tmp_path / "bundle")
    for f in Path(out).rglob("*"):
        if f.is_file() and f.suffix in (".json", ".md"):
            assert "minibar" not in f.read_text(encoding="utf-8"), f
    # another machine: no training file there, only the imported bundle
    other = Service(Project(root=tmp_path / "other", data={"pack": "tabular", "pack_config": {
        "data": "not-here.csv", "label": "outcome"}}), home=tmp_path / "other" / "var")
    imported = other.registry.import_(Path(out))
    after, calibrated = other.probabilities(imported, other.spec("outcome", imported), states)
    assert calibrated and np.array_equal(before, after)
    assert other.provider(imported).to_config() == svc.provider(version).to_config()


def test_reasons_name_the_text_column(trained):
    d = trained["svc"].generic_decide(DecideRequest(
        state={"amount": 60, "category": "meals", "note": "weekend minibar birthday party drinks"}))
    r = d["heads"]["outcome"]
    assert any(str(x.get("feature", "")).startswith("text:note#") for x in r["reasons"]), r["reasons"]
    assert "note" in json.dumps(r["reason_groups"])


def test_text_config_errors(tmp_path, data):
    base = {"data": str(data)}
    for extra, match in (({"text": "note"}, "pack_config.text must be a list"),
                         ({"text": ["note"], "exclude": ["note"]}, "text and pack_config.exclude both list"),
                         ({"text": ["note"], "categorical": ["note"]}, "encoded one way"),
                         ({"text": ["note"], "text_buckets": 4}, "text_buckets must be an integer in 8..4096"),
                         ({"text": ["nope"]}, "pack_config.text: column"),
                         ({"text": ["outcome"]}, "label / id / group / time / segment"),
                         ({"features": ["amount", "coin"], "exclude": ["coin"]}, "both list"),
                         ({"exclude": ["nope"]}, "pack_config.exclude: column"),
                         ({"exclude": "coin"}, "pack_config.exclude must be a list"),
                         ({"time_format": "%d/%m/%Y"}, "pack_config.time"),
                         ({"time": "amount", "time_format": "dd/mm/yyyy"}, "strptime format")):
        with pytest.raises(TabularConfigError, match=match):
            provider(tmp_path, {**base, **extra})


def test_a_listed_text_column_may_also_be_in_features(tmp_path, data):
    a = provider(tmp_path, {"data": str(data), "features": ["amount", "note"], "text": ["note"]})
    b = provider(tmp_path, {"data": str(data), "features": ["amount"], "text": ["note"]})
    assert a.feature_version == b.feature_version and [c.kind for c in a.columns] == ["numeric", "text"]
    auto = provider(tmp_path, {"data": str(data), "id": "claim_id", "text": ["note"]})
    assert [c.name for c in auto.columns] == ["amount", "category", "coin", "note"]


# ----------------------------------------------------------------------------- exclude and the leakage note

@pytest.fixture(scope="module")
def leaky(tmp_path_factory) -> Path:
    return claims(tmp_path_factory.mktemp("leaky") / "claims.csv", leak=True)


def _notes(tmp: Path, pack_config: dict) -> list[str]:
    pr = Project(root=tmp, data={"pack": "tabular", "pack_config": {"label": "outcome", "positive": "approved",
                                                                    **pack_config}})
    pack = TabularPack()
    return pack.datasets(pr, pack.provider(pr), ["outcome"])["outcome"].notes


def test_exclude_keeps_columns_out_of_auto_features(tmp_path, leaky):
    cfg = {"data": str(leaky), "id": "claim_id"}
    assert {"rejection_reason", "refund"} <= set(provider(tmp_path, cfg).required())
    prov = provider(tmp_path, {**cfg, "exclude": ["rejection_reason", "refund", "note"]})
    assert prov.required() == ["amount", "category", "coin"]
    # without options the schema (and so the feature_version of existing bundles) is what it was
    assert provider(tmp_path, {**cfg, "exclude": []}).feature_version == provider(tmp_path, cfg).feature_version


def test_leak_note_names_columns_that_determine_the_label(tmp_path, leaky):
    notes = [n for n in _notes(tmp_path, {"data": str(leaky), "id": "claim_id"}) if "possible label leak" in n]
    assert len(notes) == 2, notes
    reason = next(n for n in notes if "'rejection_reason'" in n)
    assert "100.0 % of 1200 rows" in reason and "pack_config.exclude" in reason
    refund = next(n for n in notes if "'refund'" in n)
    assert "rank AUC 1.000" in refund and "'approved'" in refund
    # excluded: nothing left to say; honest features (amount, category, a coin) never raise the note
    assert not [n for n in _notes(tmp_path, {"data": str(leaky), "id": "claim_id",
                                             "exclude": ["rejection_reason", "refund"]}) if "leak" in n]


def test_leak_note_false_alarms(tmp_path):
    """Strong but honest features, rare-valued columns and missingness that is not the label stay silent; a
    numeric column that is only filled in for one class is named."""
    from open_rlcd.packs.base import spec_from_yaml
    from open_rlcd.packs.tabular.data import read_table, resolve_target

    r = np.random.default_rng(0)
    n = 3000
    z = r.normal(size=n)
    y = (z + r.normal(scale=0.35, size=n) > 0.8)                    # a strong honest signal: AUC about 0.97
    many = [f"v{int(v)}" for v in r.integers(0, 900, n)]            # about 3 rows per value: pure cells by chance
    strong_cat = np.where(r.random(n) < 0.93, np.where(y, "a", "b"), np.where(y, "b", "a"))   # 93 % aligned
    part = ["" if (m and r.random() < 0.5) else "1.5" for m in y]   # missing only among positives, but half
    paid = [f"{abs(v):.2f}" if yes else "" for v, yes in zip(z, y, strict=True)]              # only when y
    path = tmp_path / "t.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["z", "many", "strong_cat", "part", "paid", "y"])
        for i in range(n):
            w.writerow([f"{z[i]:.4f}", many[i], strong_cat[i], part[i], paid[i], "yes" if y[i] else "no"])
    pr = Project(root=tmp_path, data={"pack": "tabular", "pack_config": {
        "data": str(path), "label": "y", "max_categories": 1000, "categorical": ["many"]}})
    pack = TabularPack()
    prov = pack.provider(pr)
    assert prov.required() == ["z", "many", "strong_cat", "part", "paid"]
    leaks = [x for x in pack.datasets(pr, prov, ["outcome"])["outcome"].notes if "possible label leak" in x]
    assert len(leaks) == 1 and "'paid'" in leaks[0] and "whether it is filled in at all" in leaks[0]
    # too little data to tell: silent even for a copy of the label
    table = read_table(path)
    cfg, merged = pack.targets(pr)
    target = resolve_target("outcome", merged["outcome"], table, cfg.na_values)
    spec = spec_from_yaml("outcome", target.spec)
    keep = np.arange(150)
    y150 = np.asarray([target.index(v, cfg.na_values) for v in table.data["y"][:150]])
    assert leak_notes(spec, target, table, prov, keep, y150, cfg.na_values) == []


def test_leak_note_is_silent_on_the_synthetic_tables(tmp_path):
    from open_rlcd.packs.tabular import synthetic_pack_config, write_synthetic_csv

    for signal in (1.0, 3.0):
        path = tmp_path / f"churn{signal}.csv"
        write_synthetic_csv(path, n=3000, seed=2, signal=signal)
        pr = Project(root=tmp_path, data={"pack": "tabular", "pack_config": synthetic_pack_config(path)})
        pack = TabularPack()
        for ds in pack.datasets(pr, pack.provider(pr), ["churned", "plan_tier_next", "satisfaction"]).values():
            assert not [x for x in ds.notes if "leak" in x], ds.notes


# ----------------------------------------------------------------------------- free text as a category

def test_listed_categorical_free_text_gets_a_note_not_an_error(tmp_path, data):
    """A free-text column the config lists as categorical used to be dropped by `features: auto` and then
    refused as "not a feature column"; it is kept, with the note that says what to do instead."""
    prov = provider(tmp_path, {"data": str(data), "id": "claim_id", "categorical": ["note"]})
    assert "note" in prov.required()
    note = next(n for n in prov.notes if "mostly unique long strings" in n)
    assert "'note'" in note and "pack_config.text" in note and "pack_config.exclude" in note
    assert "50 most frequent values are stored word for word" in note
    hashed = provider(tmp_path, {"data": str(data), "id": "claim_id", "categorical": ["note"],
                                 "hash_categories": True})
    note = next(n for n in hashed.notes if "mostly unique long strings" in n)
    assert "stored only as salted hashes" in note and "word for word" not in note
    # short codes and real categories are not free text
    assert not [n for n in provider(tmp_path, {"data": str(data), "features": ["category", "claim_id"]}).notes
                if "long strings" in n]
    # auto still leaves it out, and now says where free text belongs
    auto = provider(tmp_path, {"data": str(data), "id": "claim_id"})
    assert "note" not in auto.required() and any("pack_config.text" in n for n in auto.notes)


# ----------------------------------------------------------------------------- serve-time inputs

def test_input_warnings_name_non_numeric_and_unseen_values(tmp_path, data):
    prov = provider(tmp_path, {"data": str(data), "features": ["amount", "category", "coin"], "text": ["note"]})
    ok = {"amount": "12.50", "category": "Meals", "coin": "heads", "note": "anything at all"}
    assert prov.input_warnings("row", [ok, {**ok, "amount": None, "category": None}]) == []
    w = prov.input_warnings("row", [{**ok, "amount": "lots"}])
    assert len(w) == 1 and "'amount'" in w[0] and "'lots' is not a number" in w[0] and "read as missing" in w[0]
    w = prov.input_warnings("row", [{**ok, "category": "yacht"}])
    assert len(w) == 1 and "'category'" in w[0] and "'yacht'" in w[0] and "__other__" in w[0]
    assert "which no training row had" in w[0]
    assert "non-finite" in prov.input_warnings("row", [{**ok, "amount": "1e999"}])[0]
    # the same after a bundle round trip; a config from before the counts were stored still warns
    cfg = json.loads(json.dumps(prov.to_config()))
    assert TabularProvider.from_config(cfg).input_warnings("row", [{**ok, "category": "yacht"}]) == w
    del cfg["source"]["other_rows"]
    old = TabularProvider.from_config(cfg).input_warnings("row", [{**ok, "category": "yacht"}])
    assert len(old) == 1 and "encoded as __other__" in old[0]
    # hashed vocabularies: the value is not repeated
    hashed = provider(tmp_path, {"data": str(data), "features": ["amount", "category"], "hash_categories": True})
    w = hashed.input_warnings("row", [{"amount": 1, "category": "yacht"}])
    assert len(w) == 1 and "yacht" not in w[0]
    # a column where rare values are ordinary (most rows are __other__) does not warn on every request
    busy = provider(tmp_path, {"data": str(data), "features": ["amount", "claim_id"], "max_categories": 5})
    assert busy.input_warnings("row", [{"amount": 1, "claim_id": "C99999"}]) == []
    # and the features are what they were: the warning changes nothing about the row
    assert np.isnan(prov.transform("row", [{**ok, "amount": "lots"}])[0, 0])


# ----------------------------------------------------------------------------- time formats

def test_time_format_reads_other_notations_as_utc(tmp_path):
    from open_rlcd.packs.tabular.data import DEFAULT_NA, parse_time

    na = frozenset(DEFAULT_NA)
    fmt = "%d/%m/%Y %I:%M %p"
    assert parse_time("06/01/2025 02:15 PM", na, fmt) == 1736172900            # 2025-01-06T14:15:00Z
    assert parse_time("2025-01-06T14:15:00Z", na, fmt) == 1736172900           # ISO still reads
    assert parse_time("20250106", na, "%Y%m%d") == 1736121600                  # not an epoch of 20 million
    assert parse_time("20250106", na) == 20250106
    assert parse_time("yesterday", na, fmt) is None
    path = tmp_path / "t.csv"
    rows = [("06/01/2025 08:44 AM", 1, "a"), ("07/01/2025 11:12 PM", 2, "b"), ("", 3, "a"), ("soon", 4, "b")] * 30
    path.write_text("when,x,y\n" + "\n".join(f"{t},{x},{y}" for t, x, y in rows) + "\n", encoding="utf-8")
    pack = TabularPack()
    pr = Project(root=tmp_path, data={"pack": "tabular", "pack_config": {"data": str(path), "label": "y",
                                                                         "positive": "a", "time": "when"}})
    with pytest.raises(TabularConfigError) as e:
        pack.datasets(pr, pack.provider(pr), ["outcome"])
    msg = str(e.value)
    assert "'06/01/2025 08:44 AM' and '07/01/2025 11:12 PM'" in msg and "pack_config.time_format" in msg
    pr.data["pack_config"]["time_format"] = fmt
    ds = pack.datasets(pr, pack.provider(pr), ["outcome"])["outcome"]
    assert ds.t[0] == 1736153040 and ds.t[1] == 1736291520 and ds.t[2] == -1
    note = next(n for n in ds.notes if "no parseable time" in n)
    assert note.startswith("60 of 120 rows (50.0 %) have no parseable time in 'when': they only ever train")
    pr.data["pack_config"]["time_format"] = "%Y.%m.%d"
    with pytest.raises(TabularConfigError, match="no value matches pack_config.time_format '%Y.%m.%d'"):
        pack.datasets(pr, pack.provider(pr), ["outcome"])


# ----------------------------------------------------------------------------- the positive class

def test_default_positive_class_is_a_visible_note(tmp_path, data):
    pr = Project(root=tmp_path, data={"pack": "tabular", "pack_config": {"data": str(data), "label": "outcome",
                                                                         "features": ["amount"]}})
    pack = TabularPack()
    ds = pack.datasets(pr, pack.provider(pr), ["outcome"])["outcome"]
    note = next(n for n in ds.notes if "positive" in n)
    assert "true means outcome = 'rejected', false means outcome = 'approved'" in note
    assert "`positive: approved`" in note and "'outcome'" in note
    # said in rlcd.yaml, or a conventional yes / no column: nothing to point out
    pr.data["pack_config"]["positive"] = "approved"
    assert not [n for n in pack.datasets(pr, pack.provider(pr), ["outcome"])["outcome"].notes if "positive" in n]
    pr2 = Project(root=tmp_path, data={"pack": "tabular", "pack_config": {"data": str(data), "label": "coin",
                                                                          "features": ["amount"]}})
    assert any("`positive: heads`" in n for n in pack.datasets(pr2, pack.provider(pr2), ["outcome"])["outcome"].notes)


# ----------------------------------------------------------------------------- rlcd init and config check

def test_init_says_where_outcomes_and_after_the_fact_columns_go(tmp_path, capsys):
    import yaml

    from open_rlcd.cli import main
    from open_rlcd.core import config_check as CC
    from open_rlcd.core.config import load_project

    assert main(["init", str(tmp_path / "t"), "--pack", "tabular"]) == 0
    out = capsys.readouterr().out
    assert "list every column that is only known after the decision under `exclude:`" in out
    assert "pack_config.targets" in out and "rlcd config check" in out and "docs/tabular-quickstart.md" in out
    text = (tmp_path / "t" / "rlcd.yaml").read_text(encoding="utf-8")
    assert "list every column that is only known after the decision under `exclude:`" in text
    assert "# targets:" in text and text.count("{label:") == 2          # a commented example with two outcomes
    data = yaml.safe_load(text)                                         # the comments leave the YAML intact
    assert data["pack_config"]["exclude"] == [] and data["pack_config"]["text"] == []
    assert data["policy"] == {"abstain": "hold"}
    project = load_project(tmp_path / "t")
    assert CC.check_project(project) == []
    # the commented example is itself a valid config once uncommented
    block = text[text.index("  # targets:"):].split("  #\n")[0]
    example = yaml.safe_load("\n".join(ln.replace("  # ", "", 1) for ln in block.splitlines()))
    assert set(example["targets"]) == {"approved", "payout_delay"}
    data["pack_config"].pop("label")
    data["pack_config"].update(example, exclude=["review_level"])
    assert TabularPack().check_config(Project(root=tmp_path, data=data)) == []


def test_config_check_knows_the_new_keys(tmp_path):
    from open_rlcd.core import config_check as CC

    def findings(pc: dict) -> list[str]:
        project = Project(root=tmp_path, data={"pack": "tabular", "pack_config": {"data": "d.csv", "label": "y", **pc}})
        return [f"{f['path']}: {f['message']}" for f in CC.check_project(project)]

    assert findings({"exclude": ["a"], "text": ["b"], "text_buckets": 64, "time": "t", "time_format": "%Y%m%d"}) == []
    assert any("pack_config.text_buckets" in f for f in findings({"text_buckets": 4}))
    assert any("Did you mean 'exclude'?" in f for f in findings({"exclud": ["a"]}))
    assert any("both list ['a']" in f for f in findings({"features": ["a", "b"], "exclude": ["a"]}))
