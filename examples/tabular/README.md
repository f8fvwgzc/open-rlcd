# Tabular example: UCI Bank Marketing

A calibrated yes/no head on a real, public table: will a client contacted by phone subscribe to a term
deposit? It shows the whole tabular-pack loop on data the code was not written for: read a CSV, train a
calibrated head, publish how its probabilities held up on calls it never saw, and decide `act / confirm / hold`.

New to the tabular pack? Start with [docs/tabular-quickstart.md](../../docs/tabular-quickstart.md): your own CSV
to a served, calibrated decision in one page (`rlcd init --pack tabular`, outcomes and `exclude`, text columns,
reading `rlcd train`, `/v1/systemone`, `/v1/feedback`). This example is the same loop as one script on a public
dataset.

```bash
uv run python examples/tabular/run.py                  # downloads the archive (about 1 MB) on first run
uv run python examples/tabular/run.py --ordering row   # holdout = a seeded shuffle instead of the latest calls
uv run python examples/tabular/run.py --zip ~/Downloads/bank+marketing.zip   # already downloaded / offline
```

Everything lands in `examples/tabular/var/` (kept out of git by the repository's `var/` rule): the archive,
`bank-full.csv`, `bank-dated.csv`, the generated `rlcd.yaml` and the trained model under `home/`.

## Data

Moro, S., Rita, P., & Cortez, P. (2014). *Bank Marketing* [Dataset]. UCI Machine Learning Repository.
<https://doi.org/10.24432/C5K306>. Licensed [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
The script downloads <https://archive.ics.uci.edu/static/public/222/bank+marketing.zip> at run time; the data is
not redistributed in this repository. `bank-full.csv` holds 45,211 calls from May 2008 to November 2010, in
date order, `;`-separated.

## What the script does

1. Downloads the archive (or uses `--zip`) and reads `bank-full.csv` out of the nested `bank.zip` in memory
   (nothing is extracted to a path named by the archive).
2. Writes `bank-dated.csv` with an ISO `date` column. The file has `month` and `day` but no year; because the
   rows are in date order from May 2008, the year is advanced each time the month wraps around.
3. Writes `rlcd.yaml`:

   ```yaml
   pack: tabular
   pack_config:
     data: bank-dated.csv
     time: date                       # omitted with --ordering row
     features: [age, job, marital, education, default, balance, housing, loan, contact, day, month,
                campaign, pdays, previous, poutcome]
     targets:
       subscribed: {label: y, positive: "yes", stakes: medium}
   ```

4. Trains through `open_rlcd.core.service.Service` (LightGBM when importable, else logistic regression) and
   prints, per head: holdout log loss against the base-rate forecaster, Brier and Brier skill score, ECE,
   `has_skill` (block-bootstrap lower bound of the Brier gain), and the verdict per stakes level.
5. Runs one decision on the last call in the file and prints its reasons.

## Read the numbers honestly

* `duration` (call length) is **left out**: it is only known once the call is over, when the outcome is known
  too (with `features: auto` you would list it under `pack_config.exclude`). The dataset's own documentation says to discard it for a realistic model; with it, any model looks far
  better than it can be in use.
* With the default `--ordering time` the holdout is the **latest 20 % of calls**, which is how a deployed
  model meets the future. The calls span the 2008-2010 financial crisis, so the latest months need not look
  like the earlier ones; if the probabilities stop holding on them, the holdout metrics and the verdict say
  so. `--ordering row` shows what a shuffled split reports; compare the two before trusting either.
* The base rate is low (about 12 % subscribe), so a high accuracy alone means little: compare log loss and
  Brier with the baseline, and read the reliability of the probabilities (ECE), not the hit rate.
* Nothing here is tuned for this dataset. The test suite never runs this script.
