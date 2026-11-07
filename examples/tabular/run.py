"""Train the tabular pack on the UCI Bank Marketing dataset and print each head's holdout summary.

    uv run python examples/tabular/run.py                    # download (about 1 MB), train, report
    uv run python examples/tabular/run.py --ordering row     # random-like holdout instead of the latest calls
    uv run python examples/tabular/run.py --zip ~/Downloads/bank+marketing.zip   # use an archive you already have

Data: Moro, S., Rita, P., & Cortez, P. (2014). Bank Marketing. UCI Machine Learning Repository.
https://doi.org/10.24432/C5K306 - licensed CC BY 4.0. The archive is downloaded at run time into --dir (default
examples/tabular/var/, which the repository's `var/` rule keeps out of git); it is not redistributed here.

What this script does: extracts bank-full.csv (45,211 phone calls, May 2008 - Nov 2010, in date order),
reconstructs each call's date from that order and its month / day columns, writes rlcd.yaml for the tabular
pack (one noul head, `subscribed`), trains through core.service.Service, and prints the holdout summary.
What it does not do: it does not tune anything for this dataset, and it leaves out `duration` (the call length
is only known after the call, when the outcome is known too: the dataset's authors say it must be discarded
for a realistic model). It is never run by the test suite.
"""
from __future__ import annotations

import argparse
import csv
import io
import sys
import zipfile
from pathlib import Path

URL = "https://archive.ics.uci.edu/static/public/222/bank+marketing.zip"
MAX_DOWNLOAD = 100 << 20            # bytes; the archive is about 1 MB
MAX_MEMBER = 200 << 20
MONTHS = ("jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec")
FEATURES = ["age", "job", "marital", "education", "default", "balance", "housing", "loan", "contact", "day",
            "month", "campaign", "pdays", "previous", "poutcome"]           # every column but duration and y


def download(dest: Path) -> Path:
    import httpx

    print(f"downloading {URL}")
    tmp = dest.with_suffix(".part")
    size = 0
    with httpx.stream("GET", URL, follow_redirects=True, timeout=60.0) as r:
        r.raise_for_status()
        with open(tmp, "wb") as f:
            for chunk in r.iter_bytes():
                size += len(chunk)
                if size > MAX_DOWNLOAD:
                    raise SystemExit(f"download larger than {MAX_DOWNLOAD >> 20} MB: refusing")
                f.write(chunk)
    tmp.replace(dest)
    return dest


def _find(z: zipfile.ZipFile, name: str, depth: int = 0) -> bytes | None:
    """The bytes of member `name`, looking inside nested zip archives (the UCI archive nests bank.zip).
    Members are read in memory; nothing is extracted to a path taken from the archive."""
    for info in z.infolist():
        if info.file_size > MAX_MEMBER:
            continue
        base = info.filename.rsplit("/", 1)[-1]
        if base == name:
            return z.read(info)
        if base.endswith(".zip") and depth < 2:
            with zipfile.ZipFile(io.BytesIO(z.read(info))) as inner:
                found = _find(inner, name, depth + 1)
                if found is not None:
                    return found
    return None


def extract_bank_full(archive: Path, dest: Path) -> Path:
    with zipfile.ZipFile(archive) as z:
        data = _find(z, "bank-full.csv")
    if data is None:
        raise SystemExit(f"{archive} has no bank-full.csv (also looked inside nested zip files)")
    dest.write_bytes(data)
    return dest


def with_dates(src: Path, dest: Path) -> int:
    """Copy bank-full.csv (';'-separated) to a ','-separated file with an ISO `date` column. The rows are in
    date order from May 2008, so the year advances whenever the month wraps around (a drop of more than six
    months, so a stray row out of order cannot add a year). Returns the number of rows."""
    with open(src, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f, delimiter=";"))
    if not rows or not {"y", "month", "day", *FEATURES} <= set(rows[0]):
        raise SystemExit(f"{src}: unexpected columns {list(rows[0]) if rows else []}")
    year, prev = 2008, None
    for r in rows:
        m = MONTHS.index(r["month"].strip().lower()) + 1
        if prev is not None and m < prev - 6:
            year += 1
        prev = m
        r["date"] = f"{year:04d}-{m:02d}-{int(r['day']):02d}"
    if year != 2010:
        print(f"note: date reconstruction ended in {year}, not 2010; the time order is still the file's order")
    with open(dest, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["date", *[c for c in rows[0] if c != "date"]])
        w.writeheader()
        w.writerows(rows)
    return len(rows)


def write_project(work: Path, ordering: str) -> Path:
    from open_rlcd.core.config import write_config

    pack_config = {"data": "bank-dated.csv", "features": FEATURES,
                   "targets": {"subscribed": {"label": "y", "positive": "yes", "stakes": "medium",
                                              "description": "will the client subscribe to a term deposit?"}}}
    if ordering == "time":
        pack_config["time"] = "date"
    data = {"name": "bank-marketing", "pack": "tabular", "home": "home", "pack_config": pack_config,
            "train": {"holdout_fraction": 0.2, "n_splits": 5}}
    return write_config(work / "rlcd.yaml", data, header="# written by examples/tabular/run.py")


def report(job: dict) -> None:
    print(f"\njob {job['job_id']}: {job['status']}  model {job.get('model')}")
    if job.get("error"):
        print("error:", job["error"])
    for name, h in job["heads"].items():
        print(f"\nhead {name}: {h['status']}")
        if h["status"] != "trained":
            continue
        m, d = h["metrics"], h["data"]
        print(f"  holdout rows       {m['n_holdout']} ({d['ordering']} ordering; trained on "
              f"{d['development_rows']} rows)")
        print(f"  log loss           {m['log_loss']:.4f}   baseline {m['baseline_log_loss']:.4f} (base rate)")
        print(f"  Brier              {m['brier']:.4f}   baseline {m['baseline_brier']:.4f}   "
              f"Brier skill {m['brier_skill_score']:+.4f}")
        print(f"  ECE                {m['ece']:.4f}   accuracy {m['accuracy']:.4f} (baseline {m['baseline_accuracy']:.4f})")
        print(f"  has_skill          {m['has_skill']} (one-sided 95% lower bound of the Brier gain: "
              f"{m['brier_gain_lower_95']})")
        for stakes, v in h["verdict"].items():
            print(f"  verdict {stakes:<7}    {v['max_tier']}: {v['reason']}")
        for n in h.get("notes") or []:
            print(f"  note: {n}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dir", type=Path, default=Path(__file__).resolve().parent / "var",
                    help="working folder for the data, rlcd.yaml and the trained model (default: %(default)s)")
    ap.add_argument("--zip", type=Path, default=None, help="use this bank+marketing.zip instead of downloading")
    ap.add_argument("--ordering", choices=("time", "row"), default="time",
                    help="time: the holdout is the latest 20%% of calls (how a deployed model meets the future); "
                         "row: a seeded shuffle (rows treated as independent)")
    args = ap.parse_args(argv)
    work = args.dir.expanduser().resolve()
    work.mkdir(parents=True, exist_ok=True)
    full = work / "bank-full.csv"
    if not full.exists():
        archive = args.zip.expanduser() if args.zip else work / "bank+marketing.zip"
        if not archive.exists():
            download(archive)
        extract_bank_full(archive, full)
    n = with_dates(full, work / "bank-dated.csv")
    cfg = write_project(work, args.ordering)
    print(f"{n} rows; project written to {cfg}")

    from open_rlcd.core.config import load_project
    from open_rlcd.core.contract import DecideRequest
    from open_rlcd.core.service import Service

    project = load_project(cfg)
    svc = Service(project, home=project.home)  # explicit: ignore any RLCD_HOME set in the shell for this demo
    print("training (LightGBM when importable, else logistic regression) ...")
    job = svc.trainer.start(wait=True)
    report(job)
    if job.get("model"):
        with open(work / "bank-dated.csv", encoding="utf-8", newline="") as f:
            last = list(csv.DictReader(f))[-1]
        state = {c: last[c] for c in FEATURES}
        d = svc.generic_decide(DecideRequest(state=state, heads=["subscribed"]))
        r = d["heads"]["subscribed"]
        dec = r["decision"]
        print(f"\nexample decision for the last call in the file: P(subscribed) = {r['answer']['noul']}; the head "
              f"commits to subscribed = {dec['class']} (p = {dec['p']}), action {d['action']}")
        for reason in r.get("reasons") or []:
            print(f"  {reason.get('feature')} = {reason.get('value')}: {reason.get('effect')}")
        for w in d["warnings"]:
            print(f"  warning: {w}")
        print(f"\nmodel bundle: {svc.home / 'models' / job['model']}")
        print(f"serve it: RLCD_PROJECT={cfg} rlcd serve   (or create_app() from open_rlcd.serve.api)")
    return 0 if job["status"] in ("succeeded", "partial") else 1


if __name__ == "__main__":
    sys.exit(main())
