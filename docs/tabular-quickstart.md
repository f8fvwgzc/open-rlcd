# Tabular quickstart: from a CSV to a served, calibrated decision

One page, one hour, one laptop CPU. You start with a table of past cases whose outcomes you know (claims
approved or rejected, customers who churned, leads that converted, payments that were fraud). You end with a
service that answers "how likely is this outcome for this new row?" with a probability that held up on rows it
never saw, and turns it into `act` / `confirm` / `hold`.

Every output below is real, from a synthetic table of 4,000 expense claims. Your numbers will differ.

## 0. Install (5 minutes)

```bash
brew install uv libomp       # libomp: for LightGBM; without it, `model: auto` trains logistic regression (with a note)
uv tool install "open-rlcd @ git+https://github.com/f8fvwgzc/open-rlcd"
rlcd --version
```

From a clone instead: `uv sync`, then prefix every command with `uv run`. Parquet files need the `parquet` extra.

## 1. What the table must look like

One row per past case, as it looked **when the decision had to be made**, plus what happened afterwards:

```
claim_id,employee_id,submitted_at,amount,category,department,payment_method,days_since_expense,has_receipt,note,outcome,rejection_reason,review_level,payout_delay,coin
C00000,E0081,2025-01-06T08:44:00Z,387.51,travel,ops,transfer,36,yes,osaka no receipt attached project 1633,approved,,audit,2,tails
C00001,E0209,2025-01-06T11:41:00Z,226.90,training,engineering,cash,30,no,osaka safety training course the 2840,rejected,over limit,audit,1,heads
```

| kind of column | here | why it matters |
|---|---|---|
| outcome (label) | `outcome`, `payout_delay` | What you want predicted. One head per outcome column. |
| id | `claim_id` | Reported with results; never a feature. |
| group | `employee_id` | All rows of one group land on the same side of the holdout. Without it, the model is scored on people it has already seen and looks better than it is. |
| time | `submitted_at` | The holdout becomes the most recent cases, which is how a deployed model meets the future. With a group as well, it becomes the groups that appear latest, with all their rows (step 2). ISO 8601 or epoch seconds; other notations need `time_format` (e.g. `"%d/%m/%Y %I:%M %p"`). |
| known only afterwards | `rejection_reason`, `review_level` | Must **not** be features: they are filled in once the outcome is known and give it away. |
| everything else | `amount`, `category`, `note`, ... | Features. Numbers stay numbers, other values become categories, empty cells are missing. |

A head needs about 45 labelled rows before it trains at all: at least 30, with 10 or more landing in the 20 %
holdout and enough left over for the calibration folds. With `group`, whole groups move, so the count depends
on the group sizes. A holdout you can read needs a few thousand.

## 2. Create the project and describe the table (10 minutes)

```bash
rlcd init claims --pack tabular && cd claims
cp ~/path/to/your/table.csv ./data.csv
```

`rlcd init` writes `rlcd.yaml` with the one-outcome shorthand (`label: outcome`, `features: auto`). **With
`features: auto` every column you do not name is a feature**, and the pack cannot know which columns are only
known after the decision. Trained as written, this table gives:

```
head              status     holdout  logloss    base     BSS    ECE  skill verdict(medium)
outcome           trained        800    0.003   0.642   1.000  0.003  yes   act
  outcome: possible label leak: feature column 'rejection_reason' on its own almost determines 'outcome' (its value alone gives the right label for 100.0 % of 4000 rows; always guessing not 'rejected' gives 64.8 %). If it is only…
```

A Brier skill of 1.000 is not a good model, it is a leak. The note catches a column that nearly decides the label
alone; it does not catch a weaker one (`review_level` here), so do not rely on it. Edit `pack_config`:

```yaml
pack_config:
  data: ./data.csv
  id: claim_id
  group: employee_id          # one employee's claims stay on one side of the holdout
  time: submitted_at          # orders the employees: the holdout is those who appear latest, with all their claims
  features: auto
  exclude: [rejection_reason, review_level]   # filled in by the reviewer, after the decision
  text: [note]
  categorical: auto
  hash_categories: false
  targets:
    approved: {label: outcome, positive: approved, stakes: medium}
    payout_delay: {label: payout_delay, type: score}
    lucky: {label: coin, positive: heads}
```

* **`group` and `time` together.** With both set, the default `train.ordering: auto` splits by group: the
  holdout is the employees whose first claim came latest, added until they hold 20 % of the claims, with all
  their claims. That keeps each employee on one side, but the holdout is not "the most recent claims": recent
  claims of other employees train.
  `train: {ordering: time}` makes the holdout the latest 20 % of claims instead, and then one employee's claims
  can land on both sides. This example keeps the group split.
* **One outcome or several.** `label: outcome` is one head. For more, delete `label` and list them under
  `targets`: each is a head with its own model, and every declared label column is kept out of the features
  of all heads. (`lucky` is a coin flip, here to show what "no signal" looks like.)
* **`type`**: two distinct values make a yes/no head (`noul`), more make a `choice` (one of N); `score`
  (ordered levels, here 0-3) must be asked for.
* **`positive`**: the label value that means *true*. Say it. Without it, `yes` / `true` / `1` is taken when
  present, else the alphabetically later value (here `rejected`, the opposite of what you would guess), with a note.
* **`stakes`**: `low`, `medium` or `high`: how sure the model must be before it acts on its own (act at
  p >= 0.7 / 0.85 / 0.95 by default).
* **`exclude`**: every column that is only known after the decision. Never a feature, whatever `features` says.
* **`text`**: free-text columns. Each becomes 256 hashed word and character n-gram features; no vocabulary is
  stored, so nothing of the text ends up in the model bundle, and it works for languages without spaces. Do not
  list free text under `features`: that one-hot encodes whole strings, which carries nothing.
* **`hash_categories`**: with `false`, the most frequent values of each categorical column are stored word
  for word in every bundle. Set `true` when a categorical column could hold personal data.

```bash
rlcd config check
```

```
…/claims/rlcd.yaml: no problems found
```

It names a misspelt key (`featurs` → "Did you mean 'features'?") or a wrong type before anything trains. Every
key is in [configuration.md](configuration.md).

## 3. Train and read the table (1 minute)

```bash
rlcd train
```

```
model rlcd-0.1.1 — job succeeded
head              status     holdout  logloss    base     BSS    ECE  skill verdict(medium)
approved          trained        801    0.511   0.671   0.289  0.013  yes   act
payout_delay      trained        801    1.213   1.363   0.099  0.032  yes   confirm
lucky             trained        801    0.694   0.694   0.001  0.022  NO    hold

lucky: no skill over the base rate on the holdout — it will abstain. the holdout shows no skill over the base rate: every decision abstains until a retrain shows skill

Data notes:
  all heads: text columns ['note']: 256 hashed word / character n-gram features each; no vocabulary is stored, so no text from the file is in the bundle
  ...
```

* **holdout**: rows the model never trained or calibrated on (here the employees who appear latest, holding
  20 % of the claims).
* **logloss / base**: the model's log loss next to that of always answering the base rate. Lower than `base`
  is the point; equal means it learned nothing.
* **BSS** (Brier skill score): 0 = no better than the base rate, 1 = perfect. Honest tables give 0.05-0.5.
  Close to 1 means a leak (step 2).
* **ECE**: how far the probabilities are from what happened; 0.013 means "90 %" came true about 90 % of the time.
* **skill NO → `hold`**: the holdout shows nothing better than the base rate, so the head abstains on every
  request and says why. That is the system working: `lucky` is a coin. Drop the head, or give it better columns.
* **verdict**: per stakes level, what the holdout backs: `act`, or `confirm` when the evidence for acting
  alone is not there (`payout_delay`: too few confident holdout rows).
* **Data notes**: read them all once. They say what was left out, skipped, guessed or stored.

```bash
rlcd report          # MODEL_CARD.md + report.html (reliability diagrams) under var/reports/<version>/
rlcd check           # the pre-launch checklist
```

```
PASS  skill.approved         Head approved shows skill on its holdout
WARN  skill.lucky            Head lucky shows skill on its holdout
no check fails (12 pass, 4 warn, 0 fail, 16 info); the warnings are for the owners to judge before launch
```

## 4. Decide on a new row

```bash
rlcd decide --heads approved --state '{"amount": 62.40, "category": "travel", "department": "sales",
  "payment_method": "card", "days_since_expense": 4, "has_receipt": "yes",
  "note": "train to customer site ref 4411"}'
```

```
"action": "act",
"heads": {"approved": {
  "answer":   {"type": "noul", "noul": 0.914889, "calibrated": true, "skill": true},
  "decision": {"class": "true", "p": 0.914889, "tier": "act", "thresholds": {"act": 0.85, "confirm": 0.65, ...}},
  "reasons":  [{"feature": "amount", "value": 62.4, "effect": "raises P(true)"}, ...,
               {"feature": "text:note#42", "value": 0.0974, "effect": "lowers P(true)"}],
  "reason_groups": {"note": 0.5743, "amount": 0.131, "days_since_expense": 0.1234, ...}, ...}}
```

Each head's answer, decision and reasons sit under `heads.<head>`; `action` is the decision as a whole.

The state is one row as JSON: every feature column must be present (`null` for a missing value); `rlcd serve`
lists them at `GET /v1/heads`. Never act on `p` alone when `skill` is false.

Over HTTP, `rlcd serve --port 8095`, then ask any number of typed questions about one row:

```bash
curl -s localhost:8095/v1/systemone -H 'content-type: application/json' -d '{
  "state": {"amount": 62.40, "category": "travel", "department": "sales", "payment_method": "card",
            "days_since_expense": 4, "has_receipt": "yes", "note": "train to customer site ref 4411"},
  "questions": {"approve": {"head": "approved", "type": "noul"},
                "delay": {"head": "payout_delay", "type": "score"},
                "lucky": {"head": "lucky", "type": "noul"}}}'
```

```
"answers": {
  "approve": {"type": "noul", "noul": 0.914889, "calibrated": true, "skill": true},
  "delay":   {"type": "score", "score": 0, "probabilities": {"0": 0.472689, "1": 0.285513, "2": 0.155539, "3": 0.086258}, ...},
  "lucky":   {"type": "noul", "noul": 0.516736, "calibrated": true, "skill": false}},
"warnings": ["head 'lucky': the holdout shows no skill over the base rate: every decision abstains until a retrain shows skill"]
```

`/v1/systemone` returns probabilities; `POST /v1/decide` (`{"state": {...}, "heads": ["approved"]}`) also applies
the policy and returns a `decision_id`.

## 5. Post what really happened

```bash
curl -s localhost:8095/v1/feedback -H 'content-type: application/json' -d '{
  "decision_id": "fb3e60c8c4f74961beb840272c74b2fe", "label": false, "source": "finance-review"}'
```

```
{"stored": true, "head": "approved", "label": "false", "feedback_rows": 1,
 "bernoulli": {"updated": true, "class": "false", "bin": 8, "wins": 33, "losses": 10, "correct": true},
 "note": "used as an up-weighted training row at the next train"}
```

* `label`: for a yes/no head `true` / `false` (was it the `positive` value?); for choice and score heads the
  class. For a decision that involved several heads, add `"head": "<name>"`.
* `source` is **required** (a 422 without it): 1 to 120 characters of your choosing that say where the outcome
  came from (`finance-review`, `crm`, `chargeback-file`). It is stored with the row for audits; there is no fixed
  list. (The fixed list `ground_truth | human | synthetic | judge` belongs to `pack_config.label_source`, which
  describes the labels in the training file.)
* Posting the same `decision_id` again is a retry or a correction: the latest outcome counts, once.

**Why some decisions say `confirm` although the model is sure.** A verdict of `act` is necessary, not
sufficient. Each live decision also passes a gate that counts how often predictions *like this one* (same
class, same probability band) were right, starting from the holdout rows:

```
"tier": "confirm",
"heads": {"approved": {"decision": {"tier": "confirm", "uncapped_tier": "act", ...}, ...}},
"warnings": ["head 'approved': Bernoulli gate: predictions like this one (P(false) in 0.8-0.9) were right 32 of 42 times; P(accuracy > 0.75) = 0.523228 is below 0.9, so act becomes confirm"]
```

Bands the holdout already supports act from the first request (the 0.91 approval above). The others start at
`confirm`, where a person decides, and every outcome you post moves the count (`wins: 33` above). `rlcd train`
prints how many correct outcomes each band still needs. So automation is earned band by band, from your own
outcomes, and lost again if they turn out wrong.

## 6. Retrain

```bash
rlcd train
```

```
model rlcd-0.1.2 — job succeeded
  approved: 1 feedback rows replaced by a later outcome posted for the same decision_id
  approved: 1 feedback rows added to training (weight x 5)
```

Feedback rows train the next model (up-weighted) but are never scored: the holdout stays the file's own rows.
For new history, append rows to the CSV and train again. `rlcd rollback` returns to the previous version.

## Troubleshooting

| message | meaning / fix |
|---|---|
| `possible label leak: feature column …` | one column nearly decides the label alone: if it is only known afterwards, add it to `exclude` |
| `time column …: no value parses as ISO-8601 or epoch seconds; the column holds values like …` | set `pack_config.time_format` to the values' `strptime` format |
| `categorical column … holds mostly unique long strings` | free text or an id was one-hot encoded: move it to `text` or `exclude` |
| `true means … (the alphabetically later one was taken)` | set `positive` for that head |
| `a choice head needs 2 to 255 classes, column … gives 4000` | `label` points at a free-text or id column, not an outcome |
| `state is missing column(s) […]` (422) | send every feature column; `null` for a missing value |
| `the request lacks … input feature(s)` | a value was missing or not a number where training rows had one; the decision is capped at `confirm` |
| `LightGBM cannot be imported … libomp` | `brew install libomp`, or `model: logreg` for the head |
| a head always says `hold` | read `skill` and the verdict in `rlcd report`: that is the honest answer for now |

## Next

* [configuration.md](configuration.md): every `rlcd.yaml` key, including thresholds per stakes level.
* [adoption.md](adoption.md): introducing it in an organisation: shadow first, then assist, limited, full.
* [deployment.md](deployment.md): running the service outside your laptop (API keys, containers, backups).
