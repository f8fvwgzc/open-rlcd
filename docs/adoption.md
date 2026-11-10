# Adopting open-rlcd in a company

The other pages take one team on one laptop to a trained model. This one is for a business lead and their
engineer reading together: which decision to start with, who owns what, who sets the risk thresholds, what the
tool does with personal data, and what has to be true before each step of automation. About 15 minutes.

For every `rlcd.yaml` key see [configuration.md](configuration.md); for Docker, HTTPS, backups and monitoring see
[deployment.md](deployment.md); for the request path around an LLM see [control-layer.md](control-layer.md).

## 1. Is this for you

open-rlcd answers one narrow question per request with a probability that was checked against outcomes it never
saw, and turns it into `act` (do it automatically), `confirm` (a person confirms) or abstain (`escalate`; the
tabular pack calls it `hold`). It never writes text and never authorises anything by itself.

| your decision looks like | pack | what you bring |
|---|---|---|
| requests in front of a document store or an LLM: what kind of answer, which evidence, is it enough, is the draft grounded | `documents` | a folder of documents, a log of real questions, a few hundred labels you check yourself |
| one row of fields in, one outcome out: churn, fraud, lead scoring, approvals, ticket priority from structured fields | `tabular` | a CSV / Parquet file with one row per past decision and a column that holds what really happened |
| anything else: free text plus fields, several tables, your own features | your own pack ([writing-a-pack.md](writing-a-pack.md)) | Python that turns a request into a feature row and yields labelled rows |

It is the wrong tool when:

* **You have no outcomes and cannot label any.** A head needs at least 30 trusted rows to train at all, and
  usually several hundred before automatic action can be verified (section 4).
* **The decision is rare or one-off.** Calibration is a statement about frequencies; a decision made a few times
  a month never accumulates the evidence.
* **An exact rule decides it.** Limits, eligibility, permissions and allowlists belong in your code, before any
  head is asked.
* **You need generated text**, or a model that reads raw documents without a feature step. It is not an LLM.
* **The data changes faster than you can label it.** Calibration holds for data like the training data.
* **You need joins or time-window aggregates done for you.** The tabular pack reads one local file with one row per
  decision; it does not fetch, join or deduplicate.

## 2. Choosing the first decision

Pick one decision that is:

* **narrow** — one question with a fixed set of answers, not "handle the ticket";
* **frequent** — enough volume that a few hundred labelled rows already exist or can be made in a week;
* **measurable** — someone can say afterwards whether the answer was right, and you can record it;
* **recoverable** — a wrong automatic action can be undone, at least in the first slice;
* **already made by people** — so there is a baseline to compare with in shadow mode.

Write it as a typed question:

| type | answers | use it when |
|---|---|---|
| `noul` | yes / no, as P(yes) | one condition; say what yes means (`positive`) |
| `choice` | one of N options | routing; include an option for "none of these" (`other`, `out_of_scope`, ...) — `rlcd check` warns without one |
| `score` | an ordered level 0..K | severity, priority; describe each level by what a reviewer can observe, not "low / medium / high" |

A noul whose yes means "proceed" can be a **gate** (`decision: gate`): it commits only to yes, and can carry a
promised error rate (`conformal`, section 5).

Three worked mappings:

| | support-ticket triage | expense approval | lead scoring |
|---|---|---|---|
| question | Which queue does this ticket belong to? | Can this claim be paid without manual review? | Will this lead become a customer within 90 days? |
| type | `choice`: billing, technical, account, other | `noul`, `decision: gate`, `conformal: 0.05` | `noul` |
| label | the queue that finally resolved it | the reviewer's final decision on past claims | the CRM outcome after the window closed |
| stakes | `low`: a misroute costs one handover | `high`: a wrong yes pays money out | `medium`: a wrong yes wastes a sales call, a wrong no loses a deal |
| `act` | route automatically | pay without review — after your own limit and policy checks | assign to sales now |
| `confirm` | suggest the queue, an agent clicks | a reviewer sees the recommendation and its reasons | goes to a qualification queue |
| abstain | general queue, as today | full manual review, as today | stays in nurture, as today |
| pack | `tabular` if the fields carry it; your own pack for the ticket text | `tabular` | `tabular` |

Two things that decide whether the first project works:

* **The label must be known for the rows you train on.** Approvals have an outcome only for what was approved;
  leads only for those someone called. The model learns from, and is checked against, the rows that have a label.
* **Features must be known at decision time.** The tabular example leaves out call duration because it is only
  known once the outcome is. A column that leaks the outcome makes every number look better than it will be.

`act` means "eligible for your next gate", not "approved": identity, limits and user confirmation stay in your code.

## 3. Roles

`rlcd check` reads `owners:` in rlcd.yaml and warns unless these five keys each name a person or team:

```yaml
owners:
  pack: data-platform           # the state builder: pack, features, data
  heads: support-ops            # the question contract: heads and batteries
  policy: head-of-support       # thresholds, rollout and route contracts
  review: support-team-leads    # the review queue
  incidents: on-call-platform   # incident response and the kill switch
```

| activity | `pack` | `heads` | `policy` | `review` | `incidents` |
|---|---|---|---|---|---|
| data file / corpus, features, `feature_version` | **A/R** | C | I | – | I |
| question wording, options, level descriptions, battery versions | C | **A/R** | C | C | – |
| labelling plan and audit sessions | C | **A** | I | **R** | – |
| thresholds per stakes level, `policy.overrides`, conformal targets | C | C | **A/R** | C | I |
| rollout mode and the slice that may act | I | C | **A/R** | C | C |
| working the `confirm` queue, posting feedback | – | C | I | **A/R** | – |
| `serve.review_capacity_per_day` | – | – | C | **A/R** | I |
| retrain, replay, regress, promote | **R** | C | **A** | I | I |
| kill switch, rollback, regression case after an incident | C | C | I | I | **A/R** |

R does the work, A answers for it, C is consulted, I is told. `owners` has no key for labelling: put it with
`heads` (who define what a label means) and have `review` do it, as above, or name it in your own process.

The business lead is usually `policy`; the engineer usually `pack`. `policy` and `incidents` should not be only
the same single person: someone must be able to switch automation off when the threshold owner is away.

What the tool enforces of this: nothing beyond the warning. `owners` is a list of names for `rlcd check`; it grants
no permission. Access is two shared keys: `RLCD_API_KEY` for every route but `/health` and `/ready`, and
`RLCD_ADMIN_KEY` for `POST /v1/train`, `POST /v1/automation` and `POST /v1/receipts/{id}/confirm`.

## 4. Labelling plan

**How many rows, and why.** The numbers come from three defaults: a head needs 30 trusted rows to train
(`min_trusted`), the holdout is 20 % of the trusted rows (`train.holdout_fraction`: the latest 20 % when rows
have a time column, or the groups that appear latest when they also have a group; otherwise a fixed hashed 20 %),
and `act` is verified only with at least 30 holdout rows and 30 holdout rows the policy would act on
(`policy.min_verify_rows`). For a gate head (`decision: gate`) those are the rows with P(yes) at or above `act`.

| trusted rows per head | what you get |
|---|---|
| under 30 | the head does not train |
| 30 – ~150 | calibrated, but the holdout is under 30 rows: decisions are capped at `confirm` |
| ~150 | a 30-row holdout: the floor below which `act` is impossible. `act` needs every one of those 30 rows to clear the act threshold, so expect `confirm` |
| ~300 (recommended) | a holdout of about 60 rows and usable reliability bins; `act` needs half of them to clear the threshold |

So the trusted rows `act` needs are about 30 / (holdout fraction × the share of holdout rows that clear the act
threshold): with 20 % and a third of the rows clearing it, about 450. High stakes need more: at `act: 0.95`
fewer rows clear the threshold. The report's verdict lines give the act-tier row count per head and stakes level
(also `policy.holdout.<stakes>.act.n` in `calibration_<head>.json`).

**Where labels come from.**

| pack | trusted labels | weak labels |
|---|---|---|
| `tabular` | the outcome column of your file (`label_source: ground_truth` by default). No labelling session: the work is making the column correct | none; `rlcd label` has nothing to label for this pack |
| `documents` | `rlcd label audit`: a person checks rows in the terminal | `rlcd label bootstrap`: an LLM judge labels the items built from each query (see below) |
| your own | `source: ground_truth` rows, or the audit if the pack implements `label_tasks` | the judge, if the pack implements `label_tasks` |

What a documents query log yields: every query gives items for `response_type`, `route` and `needs_graph`.
`chunk_relevant` and `answerable` need retrieved chunks, so an ingested corpus (`rlcd ingest`), or `evidence`
logged with the query. `grounded`, `complete` and `risk_tier` get an item only for a `queries.jsonl` row that
carries the drafted `answer`. Logged `evidence` is optional there: without it the top retrieved chunks stand in.
A plain question log therefore yields no `grounded` labels at all.

**LLM judge, then human audit** (documents and packs with `label_tasks`):

```bash
rlcd label bootstrap --judge ollama:<model>        # claude-cli | anthropic need the remote opt-in (section 6)
rlcd label audit --heads answerable,grounded --n 600 --annotator maria
rlcd label status                                  # counts per head and source
rlcd train && rlcd report
```

* Judge labels only **train** the base model. Calibrators are fitted, and every published number is measured, on
  trusted rows only. A weak judge costs skill, not honesty.
* The audit shows the judge's answer; Enter accepts it. It samples in proportion to the judge's labels and
  confidence, so stopping early still leaves a representative sample. It is resumable.
* `--n` is questions for the session, spread over the selected heads: audit the heads that gate actions first.
* **Judge agreement** is printed at the end of an audit and in the report: per head, how often the judge's label
  equals the human's on audited items (accuracy, Cohen's kappa, a confusion table). Low agreement means the base
  model learned from noisy labels: expect less skill, and audit more or change the judge. There is no built-in
  pass mark; the head's own skill test is the gate.

**Active learning**, after the first train:

```bash
rlcd label next --head answerable --n 50     # the rows the model is least sure of, or disagrees with the judge on
rlcd label audit --queue
rlcd train && rlcd report
```

Queue rows train and calibrate but stay out of the holdout, because they were picked for being hard. Keep the
plain audit as the backbone and use the queue on top.

**How long.** The quickstart's pace is about 300 audited questions in 60–90 minutes; the bootstrap runs 20–60
minutes unattended; training takes seconds to minutes. Plan one working day of audit for a first documents model
(two gate heads at 300 rows each), by people who know the business answer — the `review` owners, not the engineer
alone. All audit labels count the same: the tool records the annotator's name but does not compare annotators.

## 5. Setting risk thresholds

A threshold is a statement about acceptable error: with calibrated probabilities, acting at p ≥ 0.85 is wrong about
15 % of the time or less. Each stakes level has three numbers:

| key | meaning in business terms | low | medium | high |
|---|---|---|---|---|
| `act` | act automatically at or above this probability | 0.70 | 0.85 | 0.95 |
| `confirm` | below `act` but at or above this: a person confirms; below it: abstain | 0.50 | 0.65 | 0.80 |
| `min_accuracy` | the track record `act` needs: predictions like this one (same head, class and probability band; holdout plus feedback) must have been right at least this often, with 90 % certainty (`gate_min_p_above: 0.9`) | 0.60 | 0.75 | 0.90 |

The defaults above are in `core/policy.py`. A head has a default level (`heads.<name>.stakes`; the pack's
choice, `medium` when it makes none); a request can send `stakes` for one decision.

The evidence then caps what the thresholds allow. No `act` threshold, however low, lifts these caps:

* no skill over the base rate on the holdout → the head abstains on every request;
* fewer than `min_verify_rows` (30) holdout rows, or act-tier holdout rows, → at most `confirm`;
* act-tier holdout rows that came true less often than predicted (`miscalibration_p: 0.05`) → at most `confirm`;
* a request missing inputs that almost every training row had → at most `confirm`.

**Conformal promise** (gate heads): `heads.<name>.conformal: 0.05` means "when I say yes, I am wrong at most 5 %
of the time". The promise is tested on the holdout. If it cannot be made with the data you have, or the holdout
breaks it, the head is capped at `confirm` and says so. Choose the number as the share of wrong automatic yeses
the business accepts.

**Asymmetric costs.** When one wrong answer costs more than the other, raise the bar for that class only:

```yaml
policy:
  stakes:
    high: {act: 0.97, confirm: 0.85, min_accuracy: 0.92}
  overrides:
    payable: {"true": {"*": {act: 0.98}}}          # head -> committed class -> stakes level or "*"
```

**Who signs off.** The `policy` owner, after reading `rlcd report` with the `review` owner: for each head and
stakes level, the holdout's act-tier accuracy with its interval (the verdict lines of `MODEL_CARD.md`), and how
many decisions would land in `confirm` per day at these thresholds. The report has no per-day number. Estimate
it from `calibration_<head>.json`, which `rlcd report` writes next to it. `policy.holdout.<stakes>.tiers` counts
holdout rows per tier before any cap. The confirm share is `confirm`, plus `act` where
`policy.verdict.<stakes>.max_tier` is `confirm`, divided by `policy.holdout.rows`. Multiply it by your daily
volume, and expect more at first: the Bernoulli gate starts some cells at `confirm`. Once shadow traffic flows,
count it there: `rollout.proposed.confirm` in `rlcd metrics`, divided by the window's days.
(`review_load.confirm_per_day` counts every capped act as confirm, so it fits only modes `limited` and `full`.)
The thresholds in force travel with every decision and receipt (`thresholds`, `policy_digest`), so what was
signed off can be compared with what was served. The tool keeps no approval record:
commit rlcd.yaml and let the change review be the sign-off.

**How to change them.**

1. Edit `policy:` (or `heads.<name>.stakes` / `conformal`) in rlcd.yaml.
2. `rlcd train`. A version keeps the thresholds it was verified with; new ones are served only by a version whose
   holdout verified them. Until then `rlcd check` shows `thresholds.config` as a warning.
3. `rlcd report`, `rlcd check`, then the retrain steps in section 8.

Do not lower thresholds to shrink the review queue: that automates decisions the evidence did not cover. Staff the
queue or narrow the traffic.

## 6. Personal data and retention

What follows is what the code does. Whether that satisfies your legal and contractual duties is your judgement.

**What is written where** (all under the project's `home`, `./var` by default, unless noted):

| file | holds | redacted | removed by |
|---|---|---|---|
| `decisions.jsonl` | every decide / evaluate: the state each head judged, probabilities, action, receipt. Also every systemone call that sets `receipt: true` (every one under `serve.receipts: all`), with each question's input state; rank calls under the same switch add candidate ids and probabilities, no states | yes (`serve.redact`) | `rlcd receipts prune --before DATE` drops states, keeps digests |
| `feedback.jsonl` | labels posted to `/v1/feedback`, with their own copy of the state, notes and tags | yes | you |
| `outcomes.jsonl` | what you post to `/v1/outcome` | notes and tags | you |
| `confirmations.jsonl` | who confirmed which decision (`by`, a name the caller sends) | notes | you |
| `labels/items.jsonl`, `judge.jsonl`, `human.jsonl` | every item to label with its full state, judge labels and rationales, human labels with annotator names | **no** | you |
| `regressions.jsonl` (project directory, meant to be committed) | the state of each regression case, as it was logged | as logged | you — review it before committing |
| documents pack: chunks, vector index, entity graph, embedding cache | your documents | no | you |
| the training table (tabular) | wherever you keep it; the pack only reads it | – | you |

None of these logs is rotated by the service.

**The switches.**

* `serve.log_states` (default `true`): `false` keeps only state digests in `decisions.jsonl`. The cost: feedback
  must then send the state itself, logged decisions cannot be replayed, and a regression case cannot be made from a
  decision.
* `serve.redact` (default `true` = secrets): masks API keys, tokens, private keys, passwords in URLs and values
  stored under keys such as `password` or `token`. `serve.redact: [secrets, pii]` also masks e-mail addresses and
  card numbers that pass the Luhn check. That is all `pii` covers: names, phone numbers, addresses, national ids
  and every non-string value (an income, an age) are logged as sent. Redaction is pattern-based and applies to
  what the server logs — not to the label store, and not to what a judge is sent.
* `rlcd receipts prune --before 2026-10-01`: drops logged states older than the date from `decisions.jsonl` and
  keeps each state's digest, so the chain still verifies. Feedback can no longer train from a pruned decision. Run
  it while the server is idle. It does not touch `feedback.jsonl`, the label store or `regressions.jsonl`.
* `pack_config.hash_categories` (tabular, default `false`): a bundle stores the vocabulary of each categorical
  column — the up to `max_categories` (50) most frequent raw values. With `true` it stores salted SHA-256 hashes
  instead. The salt is in the bundle, so a guessed value can still be tested: it stops reading values out of a
  bundle, not recovering a small set such as country or plan. Leaving a personal column out of `features` is
  stronger.
* `evaluate.output_scan` (documents): a drafted answer containing a secret shape — with `[secrets, pii]` also an
  e-mail address or card number — is rejected whatever the heads say.

The leanest setup is a state builder that sends only the fields the question needs. Everything above is a net
behind that.

**What a bundle contains.** `rlcd export ./bundle` writes `model.json`, weights (safetensors, LightGBM text),
calibrators, holdout reports and `MODEL_CARD.md`. It contains no training rows, no labels, no logs and no code.
It does contain what the model learned from them, the feature names, and for tabular the vocabularies above.
Bundles stay in `home` unless you run `rlcd export` or `rlcd push --hf REPO` (`--private` refuses a repo that is
already public).

**Where labelling sends data.** `rlcd label bootstrap` sends item texts — queries, document chunks, drafted
answers, unredacted — to the judge:

| judge | data goes to | gated by `security.allow_remote` |
|---|---|---|
| `ollama:<model>` | a local Ollama server on 127.0.0.1 | yes: another host is refused unless allowed |
| `openai:<url>\|<model>` | the URL you give | yes: a non-loopback host is refused unless allowed |
| `anthropic` | the Anthropic API | yes: refused unless allowed |
| `claude-cli` (the default in a new rlcd.yaml) | Anthropic, through Claude Code | yes: refused unless allowed |

`security.allow_remote: true` (or `RLCD_ALLOW_REMOTE=1`) lets connectors and judges send data off this machine;
`rlcd check` warns while it is on. A new rlcd.yaml has it off, so `rlcd label bootstrap` with the default judge
stops with one line until someone decides. The setting is one switch for every remote host: once it is on, any
judge and any connector host is allowed, so prefer `RLCD_ALLOW_REMOTE=1` in the shell of the labelling run over
`true` in rlcd.yaml. If labelling data must not leave your machines, leave it off and set `labeling.judge` to an
`ollama:` or loopback `openai:` judge. Serving and training call no LLM.

## 7. Rollout with exit criteria

`rollout.mode` in rlcd.yaml says how much automation the operator allows. It can only take automation away: a
decision's tier is the weaker of what the policy says and what the rollout allows, and the cap is always
`confirm`, never abstain. `rlcd init` writes `mode: full`; change it before the first request. The server reads
rlcd.yaml at start, so a mode change needs a restart of `rlcd serve`.

| mode | what the server does | what your application does |
|---|---|---|
| `shadow` | decides, logs, returns; every decision capped at `confirm` and marked shadow | keeps its existing path; sends that path as `baseline` so the receipt holds both |
| `assist` | the same cap | shows the decision as a recommendation; a person confirms every action |
| `limited` | `act` only for the stakes levels in `rollout.limited.stakes` (default `[low]`) and, if set, the heads in `rollout.limited.heads`; `confirm` elsewhere | automates the slice |
| `full` | no rollout cap; the policy and the evidence decide | automates whatever reaches `act` |

`rollout.heads: {<head>: shadow}` holds one head back whatever the global mode.

In every mode the record keeps what the policy would have done without the cap (`policy_act` in the metrics), so
shadow traffic already measures full automation.

**What to measure.** `rlcd metrics --since DATE` (or `GET /v1/metrics`), `rlcd review` (or `GET /v1/review`),
`rlcd replay`, `rlcd regress`, `rlcd check`. The accuracy numbers exist only for decisions that received feedback,
and stand for all traffic only if you post feedback for a random sample — not just for disputed cases.

**Before shadow**

* `rlcd check` shows no `fail`, and each warning has been read by its owner.
* The heads you plan to automate have skill, and a verdict of `act` at their stakes level in `rlcd report`.
* `owners` is complete; `RLCD_API_KEY` and `RLCD_ADMIN_KEY` are set; the application treats an error or timeout as
  "no decision", never as approval.

**Shadow → assist**

* Feedback is flowing: at least 30 labelled decisions that the policy put at `act`, per head and stakes level you
  want to automate (below 30 the drift test does not run).
* `drift_suspected` is false for those heads, and the error rate of what the policy would have automated,
  `1 − heads.<head>.feedback.policy_act.<stakes>.accuracy` in `rlcd metrics`, is no higher than the act threshold
  allows (at `act: 0.85`, about 15 %) and than the business accepted in section 5. `false_automation` cannot
  show this yet: in shadow and assist nothing acts, so it stays empty, and its `count: 0` means "nothing
  measured", not "no errors".
* The model's proposed actions have been compared with `baseline`, and the disagreements read by the `review`
  owner.

**Assist → limited**

* `serve.review_capacity_per_day` is set and `review_over_capacity` is false: reviewers keep up with `confirm`.
* `review_agreement` — how often reviewers found a `confirm` decision right — is known and acceptable to `policy`.
* The kill switch and `rlcd rollback` have each been tried once (`rlcd check` asks for the rehearsal).
* The first slice is chosen: lowest stakes, a head whose wrong action can be undone.

**Limited → wider slice → full**

* Inside the slice: 30 or more labelled automatic decisions per stakes level, `drift_suspected` false,
  `false_automation` within the threshold.
* For the next slice (a stakes level, a head): the same evidence from its `policy_act` decisions while it was
  still capped.
* `rlcd regress` passes; every incident so far left a regression case.

**Moving back** — any one of these:

| sign | first step |
|---|---|
| a wrong automatic action with real cost, or anything you do not understand | `rlcd automation off --reason "..."` |
| `drift_suspected` for a head | `rollout.heads: {<head>: assist}`, label a random sample, retrain |
| `false_automation` above what the threshold allows | narrow `rollout.limited`, or go back one mode |
| `review_over_capacity` | staff the queue or narrow the traffic; do not lower thresholds |
| a new version behaves worse (`rlcd replay`, a failing `rlcd regress`) | `rlcd rollback` |

**Kill switch.** `rlcd automation off --reason "incident 42" --by maria` (or `POST /v1/automation
{"enabled": false}`) caps every decision at `confirm` at once, with no restart; decisions are still made, logged
and returned. `rlcd automation status` shows it, `rlcd automation on` lifts it, and every change is appended to
`automation.jsonl` in `home`. `RLCD_AUTOMATION=off` in the server's environment forces it off so the API cannot
switch it back. With `RLCD_ADMIN_KEY` set, a caller holding only the decision key cannot switch it either.

**Rollback.** `rlcd rollback --reason "..."` points `rlcd-latest` at the previous version (`--to VERSION` for a
specific one; `rlcd promote VERSION` to move forward again). No retrain: the version gates on its own holdout
evidence and thresholds. It needs an earlier version to exist — `rlcd check` warns when there is none.

Rollout and the kill switch cap automation. They do not make a head more accurate.

## 8. Running it week to week

**Feedback.** For each reviewed or resolved decision, post what was true:

```bash
curl -s localhost:8095/v1/feedback -H "Authorization: Bearer $RLCD_API_KEY" -H 'content-type: application/json' \
  -d '{"decision_id": "...", "head": "payable", "label": "false", "source": "expense-review"}'
```

`source` is required: a name for who or what reported the outcome.

It updates the track record behind the `min_accuracy` gate at once, and becomes a training row at the next train,
weighted by `train.feedback_weight` (5). Feedback rows train; they do not calibrate and are not scored. A later
row for the same decision and head corrects the earlier one. `POST /v1/outcome` records task success, cost and
latency for the metrics and never trains. There is no command-line or screen for posting feedback: the reviewer's
tool is yours to build on `GET /v1/review` and `POST /v1/feedback`.

**Review queue.** `rlcd review` lists `confirm` and abstained decisions without feedback, oldest first, with the
state, uncertainty and reasons. An item leaves the queue when feedback is posted. A person confirms a `confirm`
decision with `rlcd receipts confirm ID --by NAME` or `POST /v1/receipts/{id}/confirm`.

**Review capacity.** Set `serve.review_capacity_per_day` to what the reviewers can really do. `rlcd metrics`
reports `review_over_capacity`; `rlcd check` compares the last 7 days and names the days above capacity.
Abstentions are counted apart: if they also go to people, add them.

**Retraining.** Nothing is scheduled: retrain when enough feedback or fresh labels have accumulated, when drift is
suspected, or when thresholds change. A monthly slot on the `pack` owner's calendar is a reasonable start.
`rlcd train` makes the new version `rlcd-latest` immediately, so either:

* train in place, then `rlcd replay` (compares with the version that was latest before), `rlcd regress`,
  `rlcd check`, `rlcd report` — and `rlcd rollback` if it is worse; or
* train on a copy of the project, `rlcd export ./bundle`, then on the serving home
  `rlcd import ./bundle --keep-latest`, `rlcd replay --base rlcd-latest --candidate VERSION`,
  `rlcd regress --model VERSION`, and `rlcd promote VERSION` when the `policy` owner agrees.

New trusted rows are what move a verdict from `confirm` to `act`: feedback alone improves the base model, not the
holdout. For tabular, append resolved rows to the data file; for documents, audit fresh queries.

**Drift signs.**

* `drift_suspected` in the metrics (needs 30 labelled policy-act decisions; it flags only a clear drop);
* the abstain or review share rising, or the mix of `uncertainty.sources` changing;
* online reliability moving away from the holdout (`vs_holdout`);
* a change you know about: a new product, document type, language, customer segment or form field.

A flag is a reason to label a random sample and retrain, or to roll back. It does not measure how large the drop is.

**Regression cases.** Every incident and every instructive review correction leaves one:

```bash
rlcd case add --from-decision ID --head payable --expect class=false --incident INC-42
rlcd regress            # exit 1 if any case fails; run it in CI and before every promote
```

Feedback posted with `tags: ["regression"]` adds one too. Cases live in `regressions.jsonl` in the project and
carry the logged state: review before committing.

**Incident steps.**

1. `rlcd automation off --reason "INC-42: ..."`. Decisions keep flowing to people.
2. Find the decisions: `rlcd receipts show ID`, `rlcd metrics --since DATE --head H`, `rlcd review --head H`.
3. If a new version caused it, `rlcd rollback --reason "INC-42"`.
4. `rlcd receipts verify` if the log itself is in doubt.
5. Add a regression case for each wrong decision; post the correct labels as feedback.
6. Fix the cause — labels, a feature, a threshold, a rule that belongs in code — retrain, `rlcd replay`,
   `rlcd regress`, `rlcd check`.
7. `rlcd automation on --reason "INC-42 closed"`, at the rollout mode the `policy` owner chooses.

## 9. Several businesses or teams in one company

One project per business or decision domain. A project is one `rlcd.yaml` and one `home`; a server serves exactly
one. Run one `rlcd serve` per project, each on its own port (`serve.port`) with its own keys.

```
company-rlcd/
  support-triage/    rlcd.yaml   var/
  expense-approval/  rlcd.yaml   var/
  sales-leads/       rlcd.yaml   var/
```

Commands pick the project from `--project DIR` (written before the command: `rlcd --project DIR metrics`), else
`RLCD_PROJECT`, else the nearest `rlcd.yaml` above the current folder.

| may be shared | must not be shared |
|---|---|
| the install (`uv tool install ...`), pinned to one version | `home`: labels, logs, receipts, bundles, the version registry, the kill switch |
| a custom pack's code, as a Python package named in each rlcd.yaml (`pack: my_pkg.my_pack:Pack`) | `policy` thresholds and stakes levels: each business accepts its own error rates |
| this adoption process and the incident steps | a trained bundle: it learned one business's data and was verified on one business's holdout |
| the people, where one team really owns two domains | `owners`, `RLCD_API_KEY` and `RLCD_ADMIN_KEY` |

Four things to watch:

* `RLCD_HOME` in the environment overrides `home` in every rlcd.yaml that process loads. Set globally on a machine
  with two projects, it makes them share one home — one registry, one kill switch. Leave it unset, or set it per
  service.
* `RLCD_PROJECT` in the environment outranks the folder you are in. Set globally, it sends every command run in
  any project folder to that one project, `rlcd automation off` included. Leave it unset, or set it per service.
* The kill switch is per home. Switching automation off for one business does not touch another; a company-wide
  stop is one command per project.
* Importing another team's bundle (`rlcd import`) brings their thresholds and their holdout verdicts. It says
  nothing about your traffic until you have fed it your own outcomes in shadow mode.

When two teams ask the same question about different populations, start with two projects on the same pack.
Merge only if the reports show that one model serves both — the tabular `segment` key gives per-segment holdout
slices for that comparison.

## 10. Checklist

**Decide**
- [ ] One decision chosen: narrow, frequent, measurable, recoverable, made by people today
- [ ] Written as one typed question; yes is defined, a no-match option exists, levels are observable
- [ ] The label and where it comes from are named; features are known at decision time
- [ ] `act`, `confirm` and abstain each mapped to a concrete step in the business

**People**
- [ ] `owners:` names `pack`, `heads`, `policy`, `review`, `incidents`
- [ ] `policy` and `incidents` are not one single person
- [ ] Reviewers named and `serve.review_capacity_per_day` set

**Data**
- [ ] The state carries only the fields the question needs
- [ ] `serve.log_states` and `serve.redact` chosen on purpose; `hash_categories` for tabular
- [ ] The judge is one your data may be sent to; `security.allow_remote` is off unless decided
- [ ] A retention date and a schedule for `rlcd receipts prune`, and for the files it does not cover
- [ ] `RLCD_API_KEY` and `RLCD_ADMIN_KEY` set, held by different callers

**Model**
- [ ] Per head that gates an action, enough trusted rows for 30 act-tier holdout rows at its stakes level (about
      300 to start; below 150 `act` is impossible)
- [ ] `rlcd train`, `rlcd report` read by the `policy` and `review` owners
- [ ] Stakes level per head and thresholds signed off by `policy`; rlcd.yaml committed
- [ ] `rlcd check` shows no fail; every warning has an owner's answer

**Rollout**
- [ ] `rollout.mode: shadow` before the first request; the application sends `baseline`
- [ ] Feedback posted for a random sample of decisions
- [ ] Exit criteria of section 7 met before each step up; the step recorded in the rlcd.yaml history
- [ ] Kill switch and `rlcd rollback` rehearsed
- [ ] Application fails closed on errors and calls `POST /v1/receipts/{id}/check` before side effects

**Every week**
- [ ] `rlcd metrics`: `false_automation`, `drift_suspected`, `review_over_capacity`
- [ ] `rlcd review`: the queue is being worked
- [ ] Incidents left regression cases; `rlcd regress` passes
- [ ] Retrain slot kept; new version replayed and checked before it serves
