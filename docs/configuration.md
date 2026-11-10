# Configuration reference: every key of `rlcd.yaml`

One file configures a project: `rlcd.yaml`, written by `rlcd init NAME --pack documents|tabular` and found by
looking in the current folder and its parents (or where `--project` / `RLCD_PROJECT` points). Everything in it
is optional, but the file itself is not: `rlcd train`, `rlcd ingest` and `rlcd label ...` stop with one line
(exit code 2, nothing read, written or downloaded) when there is no `rlcd.yaml`; the commands that work on a
model home (`decide`, `ask`, `check`, `import`, `promote`, `metrics`, ...) also run without a project when the
home is named with `--home` or `RLCD_HOME` (or `./var` already holds a model); and `rlcd serve` without a
project serves the bundled synthetic demo model. `RLCD_PROJECT` pointing at a path without an `rlcd.yaml` is an
error for every command that looks for a project, never "no project".

Two commands tell you whether the file means what you think:

```bash
rlcd config check     # every key and value checked; exit 1 when something is wrong
rlcd config show      # what the project resolves to: home, pack, heads, thresholds, rollout
```

`rlcd config check` prints one line per finding, `level  path: message`:

```
error  rollout.mode: must be one of shadow, assist, limited, full; got 'shaddow'. Did you mean 'shadow'?
error  train.holdout_fracton: unknown key. Did you mean 'holdout_fraction'?
warn   routes.response_type.direct_answer.api_key: the key names a secret and has a value written into the file: secrets belong in the environment ...
```

* **error**: the file cannot mean what was written (an unknown key, a wrong type, a value out of range, a route
  contract the service would refuse). `rlcd train` and `rlcd serve` print the findings and refuse to start (exit
  code 2) until it is fixed. `rlcd check` reports it as the failing item `config.valid`. `rlcd decide` and
  `rlcd ask` print the findings to stderr and still answer (they are read-only), so a misspelled `rolout:` is not
  silently ignored there either.
* **warn**: it runs, and deserves a look (a value that looks like a secret, a name that matches no head, a key
  nothing reads). Warnings never stop a command.

`--json` prints `{config, ok, errors, warnings, findings: [{level, path, message}]}` for CI.

What the check does not do: it does not look at your documents, your query log or a trained model (`rlcd check`
compares the model with the file), and its secret scan knows well-known key shapes and key names only. A file
without findings is a file without typos, not a good configuration.

The tables below are the keys the code reads. They are kept equal to the checker's own list by a test
(`tests/test_config_check.py`), so a key that is here is accepted and a key that is not is reported.
"default" is what you get when the key is absent.

A key written with no value (`timeout:`, or `timeout: null`) usually means the default too. The exceptions:

* **An empty value is an error** for every `train.*` key, `policy.min_verify_rows`, `policy.gate_min_p_above`,
  `policy.miscalibration_p`, `policy.stakes.<level>` and its `act`, `confirm` and `min_accuracy`,
  `labeling.batch`, `serve.host`, `serve.port`, `rollout.mode`, `rollout.heads.<head>`, `routes.<head>`,
  `batteries.<battery>`, `evaluate.max_revisions`, and in the tabular pack `pack_config.data`,
  `pack_config.max_categories` and `pack_config.na_values`. `rlcd train` and `rlcd serve` refuse to start until
  the key has a value or the line is gone.
* **Under `heads.<head>`, an empty field does not bring back the pack's value: it replaces it.** `type:`,
  `model:` and `decision:` are errors. `conformal:` (like `conformal: null`) removes the pack's promise, such as
  the 0.05 on `answerable` and `grounded` in the documents pack. `stakes:` passes the check, but `rlcd train` then
  stops on that head ("stakes None is not a level the policy defines"). To keep the pack's value, remove the line.
* **`rollout.limited.stakes:` with no value means no level**, not the default `[low]`: mode `limited` then
  automates nothing (the check warns).

## Top level

| key | type | default | what it does |
|---|---|---|---|
| `name` | text | the folder's name | The project's name, shown by `rlcd config show`. Nothing else reads it. |
| `pack` | `documents`, `tabular`, or `module:Class` | `documents` (or `RLCD_PACK`) | Which pack answers the questions. `module:Class` imports your own pack: only name code you trust. |
| `home` | path | `var` | Where models, logs, labels and caches are written, relative to the folder of `rlcd.yaml`. `RLCD_HOME` and `--home` override it. |
| `pack_config` | mapping | empty | The pack's own settings: see [the documents pack](#pack_config-of-the-documents-pack) and [the tabular pack](#pack_config-of-the-tabular-pack). A custom pack that declares no keys is not checked here. |
| `regressions` | path | `regressions.jsonl` | The regression cases file for `rlcd regress`, relative to the folder of `rlcd.yaml`. |

## `heads`: change a question the pack defines

`heads:` maps a head name to the fields you override; every other field keeps the pack's value (`rlcd config
show` prints the merged result and which fields came from your file). A name the pack does not have adds a
head. In the documents pack it needs at least `type` and `view` (`query`, `query_chunk`, `query_evidence` or
`query_evidence_answer`), and `classes` or `levels` for a choice or score head; the check asks only for `type`,
and a head without `view` cannot train. In the tabular pack it needs `label`: the type is inferred from the
column.

| key | type | default | what it does |
|---|---|---|---|
| `heads.<head>.enabled` | true / false | true | `false` switches the head off: it is not trained and not served, without a retrain. A battery that asks the head then fails as a whole (422 `unknown_head`, and `rlcd check` fails `batteries.bound`): give the battery a version without that question (worked example 1). |
| `heads.<head>.stakes` | a stakes level | the pack's (`medium`; documents `grounded`: `high`) | Which thresholds of `policy.stakes` its decisions use when the request names none. Must be a level the policy has. |
| `heads.<head>.conformal` | number in (0, 0.5) | the pack's (documents: 0.05 on `answerable` and `grounded`, none on the others; tabular: none) | A promised error rate. For a `gate` head, the error among its yeses: the threshold is chosen on the calibration blocks and checked on the holdout. Too few calibration rows (45 at 0.05 with the default `train.conformal_delta`) or a holdout that breaks the promise caps the head at confirm. For other heads, the miscoverage of a prediction set returned with each answer; it never caps a decision. |
| `heads.<head>.model` | `logreg`, `lightgbm`, `auto` | the pack's (`logreg`) | The base model. `auto` is LightGBM when it can be imported, else logistic regression. |
| `heads.<head>.decision` | `argmax`, `gate` | the pack's (`argmax`) | `gate` (noul heads only): the decision is about `true` ("proceed"), never about the more likely class. |
| `heads.<head>.type` | `noul`, `choice`, `score` | the pack's | The question type. Changing it changes the question: retrain. |
| `heads.<head>.classes` | list | the pack's | The options of a choice head (2 to 255) or the level names of a score head (2 to 10). A noul head has none. |
| `heads.<head>.levels` | list | the pack's | One description per class, shown to labellers and in the legend. For a score head given only `levels`, the classes are `0`, `1`, ... |
| `heads.<head>.positive` | text | the pack's | What `true` means for a noul head (in the tabular pack: the label value that means true). |
| `heads.<head>.description` | text | the pack's | The question as a labeller and an LLM judge read it. |
| `heads.<head>.view` | text | the pack's | Which feature view of the pack's provider the head reads. The tabular pack has one, `row`. |
| `heads.<head>.params` | mapping | empty | Parameters passed to the base model (for example LightGBM's `n_estimators`). |
| `heads.<head>.input` | mapping | empty | The head's input declaration (which state fields it needs), where the pack uses one. |

## `policy`: from a probability to act / confirm / abstain

A decision acts when its calibrated probability is at least `act`, asks for confirmation at `confirm`, and
abstains below. Thresholds are per stakes level; a version keeps the thresholds it was verified with until a
retrain verifies new ones (`rlcd check`, item `thresholds.config`, says when the file and the model differ).

| key | type | default | what it does |
|---|---|---|---|
| `policy.abstain` | text | the pack's (`escalate`; tabular: `hold`) | The name of the abstaining action in every response. Not `act` or `confirm`. |
| `policy.stakes.<level>.act` | number in [0, 1] | low 0.70, medium 0.85, high 0.95 | The probability at which a decision is automated. |
| `policy.stakes.<level>.confirm` | number in [0, 1] | low 0.50, medium 0.65, high 0.80 | The probability at which a person (or a cheaper check) confirms. Never above `act`. |
| `policy.stakes.<level>.min_accuracy` | number in [0, 1] | low 0.60, medium 0.75, high 0.90 | The accuracy that predictions like this one must have shown (holdout plus scored feedback) before `act` is allowed. |
| `policy.min_verify_rows` | whole number >= 0 | 30 | Trusted holdout rows (and act-tier rows) needed before `act` counts as verified: with fewer, the head's verdict is capped at confirm. See [below](#when-does-a-decision-act). |
| `policy.gate_min_p_above` | number in [0, 1] | 0.9 | The Bernoulli gate: how sure the evidence must be that predictions like this one are right at least `min_accuracy` of the time before `act` is allowed. Higher means more correct outcomes are needed. See [below](#when-does-a-decision-act). |
| `policy.miscalibration_p` | number in [0, 1] | 0.05 | Significance level of the test "act-tier rows came true less often than predicted"; when it fires the head is capped at confirm. |
| `policy.overrides` | mapping | empty | Different thresholds for one head and one committed class: `{head: {class: {level or "*": {act, confirm, min_accuracy}}}}`. The head must be a head of the project and the class one of its classes (`true` / `false` for a noul head): anything else is an error, because the thresholds would never apply. |

You can redefine `low`, `medium`, `high` partly (the other numbers keep their defaults) or add a level of your
own, which then needs all three numbers.

### When does a decision act?

A probability above `act` is necessary, not sufficient. Two more things must hold, and both are evidence, not
settings you can talk the model into:

1. **The holdout verdict** (`rlcd train` prints it as `verdict(medium)`, `rlcd check` per level). `act` is
   verified only when the head showed skill and at least `policy.min_verify_rows` trusted holdout rows (and as
   many act-tier rows among them) were scored; otherwise the verdict is `confirm` (or, for a head that shows no
   skill, the abstain action) and no live decision acts.
2. **The Bernoulli gate.** Every decision is a trial: the committed class came true or it did not. Counts are
   kept per head, committed class and probability bin (0.8-0.9, 0.9-1.0, ...), start from the holdout rows of
   the version being served, and grow with every outcome posted to `/v1/feedback` with a `decision_id`. `act`
   needs the accuracy posterior of that cell (a uniform prior plus the counts) to put at least
   `policy.gate_min_p_above` of its mass above the level's `min_accuracy`. A cell the holdout filled well
   passes at once. A cell it never reached (a class or a probability range that was rare on the holdout) starts
   at `confirm`, with the warning "predictions like this one were right 0 of 0 times", and moves to `act` after
   enough correct outcomes: with the defaults, 4 in a row at low stakes, 8 at medium, 21 at high (one wrong
   outcome costs several more). So autonomy is earned from posted outcomes, cell by cell.

`rlcd train` and `rlcd check` (item `skill.<head>`) print, for each head whose verdict is `act`, how many of
those cells already pass and how many correct outcomes each of the others needs, computed with your thresholds
and the gate itself. Raising `gate_min_p_above` or `min_accuracy` raises those numbers; lowering them lets a
cell act on thinner evidence. `GET /v1/bernoulli?head=` shows the table. (`bernoulli.decisions_needed` in a
decision is a different number: the sample size a statistical test would need to tell this probability from
`min_accuracy`.)

## `train`

Command-line flags of `rlcd train` and fields of `POST /v1/train` win over these.

| key | type | default | what it does |
|---|---|---|---|
| `train.holdout_fraction` | number in [0.05, 0.5] | 0.2 | Share of the trusted rows kept out of training and scored once. |
| `train.n_splits` | whole number in [2, 20] | 5 | Calibration folds (fewer are used when the data is too small). |
| `train.ordering` | `auto`, `time`, `group`, `row` | `auto` | How rows are split. `auto`: by group when rows have groups, else by time when they have timestamps, else by row. |
| `train.gap` | whole number >= 0 | 0 | Time ordering: timestamps left out before each calibration block. |
| `train.calibration` | `auto`, `sigmoid`, `isotonic`, `temperature` | `auto` | The calibrator. `auto` picks isotonic only with enough rows per class, else sigmoid. |
| `train.isotonic_min_per_class` | whole number >= 1 | 3000 | Trusted rows per class in every calibration block before `auto` picks isotonic. |
| `train.conformal_delta` | number in (0, 1) | 0.1 | A gate head's `conformal` promise holds with probability at least 1 minus this. Not used for other heads. |
| `train.n_estimators` | whole number in [10, 3000] | 300 | LightGBM trees, unless the head's `params` set them. |
| `train.min_trusted` | whole number >= 1 | 30 | Fewest trusted rows a head needs to be trained at all. |
| `train.seed` | whole number | 7 | Random seed of the splits and the models. |
| `train.feedback_weight` | number in (0, 1000] | 5 | How much more a feedback row weighs than a labelled row when training the base models. |

`holdout_fraction`, `n_splits`, `n_estimators` and `feedback_weight` have the same ranges here as in
`POST /v1/train`, so the file and the API accept the same values.

## `labeling`: the LLM judge of `rlcd label bootstrap`

| key | type | default | what it does |
|---|---|---|---|
| `labeling.judge` | text | `claude-cli` | `claude-cli[:model]`, `anthropic[:model]`, `ollama:<model>` or `openai:<base_url>\|<model>`. `--judge` overrides it. `claude-cli` and `anthropic` send item text to Anthropic and are refused unless `security.allow_remote` is on. |
| `labeling.model` | text | none | The model, when `labeling.judge` names none: `judge: ollama` with `model: llama3` is `ollama:llama3`. Not read when `--judge` is given. |
| `labeling.batch` | whole number >= 1 | 20 | Items per judge call. `--batch` overrides it. |
| `labeling.timeout` | number > 0 | 300 | Seconds before one judge call is given up. |
| `labeling.executable` | text | `claude` | The Claude Code command the `claude-cli` judge runs. |
| `labeling.base_url` | text | `http://127.0.0.1:11434` | Where the `ollama` judge finds Ollama. A host that is not loopback also needs remote hosts allowed (below). |

Judge API keys are read from the environment only (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`).

## `serve`: the HTTP API and what it logs

| key | type | default | what it does |
|---|---|---|---|
| `serve.host` | text | `127.0.0.1` | The interface `rlcd serve` listens on. On anything but loopback `rlcd serve` refuses to start without `RLCD_API_KEY` (exit code 2; `--insecure-no-auth` overrides, `rlcd check` fails without the key). `--host` overrides it. |
| `serve.port` | whole number in [0, 65535] | 8095 | The port. `--port` overrides it. |
| `serve.log_states` | true / false | true | Keep request states (redacted) in the decision log, so feedback can train from a `decision_id` and decisions can be replayed. `false` keeps digests only. |
| `serve.redact` | true, false, or a list of `secrets` / `pii` | true (`secrets`) | What is masked before anything is logged. `[secrets, pii]` also masks e-mail addresses and card numbers. |
| `serve.receipts` | `decisions`, `all` | `decisions` | `decisions`: decide and evaluate calls are logged, a systemone or rank call only when it asks for a receipt. `all`: every call. |
| `serve.review_capacity_per_day` | number > 0 | none | Confirm decisions your reviewers can take per day; `rlcd check` and the metrics compare the real load with it. |

## `security`

| key | type | default | what it does |
|---|---|---|---|
| `security.allow_remote` | true / false | false | Let connectors and LLM judges talk to hosts that are not loopback, including the `claude-cli` and `anthropic` judges. Same as `RLCD_ALLOW_REMOTE=1`. |

## `rollout`: how much automation the operator allows

The rollout can only take automation away from what the policy allows.

| key | type | default | what it does |
|---|---|---|---|
| `rollout.mode` | `shadow`, `assist`, `limited`, `full` | `full` | `shadow`: decide and log, automate nothing. `assist`: a person confirms every action. `limited`: act only inside the slice below. `full`: the policy alone decides. There is no `off` mode: to automate nothing use `shadow`, or the kill switch (`rlcd automation off`, `RLCD_AUTOMATION=off`). A bare `off` or `no` is read by YAML as false and refused like any other value. |
| `rollout.limited.stakes` | name or list of names | `[low]` | Mode `limited`: the stakes levels that may act. |
| `rollout.limited.heads` | name or list of names | any head | Mode `limited`: only these heads may act. |
| `rollout.heads.<head>` | `shadow`, `assist`, `limited`, `full` | `rollout.mode` | A mode for one head, overriding `rollout.mode`. |

## `routes`: what an option of a choice head commits you to

`routes.<head>.<option>` is the contract of one option. The five keys below are read by the deterministic
filter that runs before the head chooses; every other key in a contract is yours and is returned untouched with
the decision (so the check only warns when one looks like a misspelling of these five).

A contract under an option the head does not have (a misspelt option), or on a head that is not a `choice` head,
is an error: the service would refuse every decision that consults that head (500 `bad_config`). A contract
under a name that is not a head of the project is only a warning, since no decision consults it, but `rlcd check`
fails `routes.contracts` until the model has that head or the contract is gone.

| key | type | default | what it does |
|---|---|---|---|
| `routes.<head>.<option>.enabled` | true / false | true | `false`: the option is never chosen. |
| `routes.<head>.<option>.requires` | name or list of names | none | Capabilities the option needs; it is removed when the request's `capabilities` lack one. |
| `routes.<head>.<option>.allowed_data` | name or list of names | any | Data classes the option may receive; it is removed when the request's `sensitivity` is not one of them. |
| `routes.<head>.<option>.latency_ms` | number | none | Expected latency; the option is removed when it is above the request's `latency_budget_ms`. |
| `routes.<head>.<option>.fallback` | another option of the head | none | What to run when this option fails. It must have a contract too, and a chain of fallbacks must end. |

## `batteries`: named, versioned question sets

A battery in `rlcd.yaml` replaces the pack's battery of the same name.

| key | type | default | what it does |
|---|---|---|---|
| `batteries.<battery>.version` | text, in quotes | required | Change it whenever a question's meaning changes. Quote it: YAML reads a bare `1.10` as the number 1.1. |
| `batteries.<battery>.questions` | mapping | required | `{question id: {type, head, ...}}`, the same question objects `/v1/systemone` takes. |
| `batteries.<battery>.owner` | text | none | Who owns the question set. |
| `batteries.<battery>.description` | text | empty | When the battery is asked, on what state. |
| `batteries.<battery>.changelog` | list of text | empty | One entry per version. |
| `batteries.<battery>.consumers` | mapping | empty | `{question id: who reads its answer}`; only ids the battery asks. |

## `evaluate`

| key | type | default | what it does |
|---|---|---|---|
| `evaluate.max_revisions` | whole number >= 0 | 2 | How many revisions of a drafted answer are judged before evaluate says "escalate or narrow the task". |
| `evaluate.output_scan` | true, false, or a list of `secrets` / `pii` | true (`secrets`) | A drafted answer that contains one of these never ships, whatever the heads say. |

## `owners`

Read by `rlcd check` (item `owners`), which warns until all five are named. Each is a person or team, or a list.

| key | type | default | what it does |
|---|---|---|---|
| `owners.pack` | name or list of names | none | Who owns the state builder: pack, features, data. |
| `owners.heads` | name or list of names | none | Who owns the question contract: heads and batteries. |
| `owners.policy` | name or list of names | none | Who owns thresholds, rollout and route contracts. |
| `owners.review` | name or list of names | none | Who staffs the review queue. |
| `owners.incidents` | name or list of names | none | Who responds to incidents and holds the kill switch. |

## `pack_config` of the documents pack

| key | type | default | what it does |
|---|---|---|---|
| `pack_config.encoder.kind` | `onnx`, `hashing` | `onnx` | The text encoder. `onnx` downloads a model once (needs the `documents` extra); `hashing` needs nothing and gives weaker features. Changing the encoder means re-ingesting and retraining. |
| `pack_config.encoder.model_id` | text | `intfloat/multilingual-e5-small` | onnx: the Hugging Face model. |
| `pack_config.encoder.file` | text | `onnx/model_qint8_avx512_vnni.onnx` | onnx: the model file inside that repository. |
| `pack_config.encoder.max_length` | whole number >= 1 | 512 | onnx: tokens read per text. |
| `pack_config.encoder.batch_size` | whole number >= 1 | 32 | onnx: texts encoded per batch. |
| `pack_config.encoder.quantize_locally` | true / false | false | onnx: quantize the model on this machine instead of downloading a quantized file. |
| `pack_config.encoder.dim` | whole number >= 1 | 384 | hashing: vector size. |
| `pack_config.encoder.seed` | whole number | 0 | hashing: seed of the hash. |
| `pack_config.encoder.sha256` | text | none | Refuse to run when the encoder built here has another digest. |
| `pack_config.encoder.cache` | true / false | true | Cache embeddings under `<home>/cache/embeddings`. |
| `pack_config.corpus` | path | none | The folder `rlcd ingest` reads when no path is given. |
| `pack_config.queries` | path | none | The query log `rlcd label bootstrap` labels (JSON lines, a JSON array, or CSV with a `query` column). |
| `pack_config.chunking.max_chars` | whole number >= 1 | 1200 | Chunk size in characters. The chunker raises a value below 50 to 50. |
| `pack_config.chunking.overlap` | whole number >= 0 | 150 | Characters shared by neighbouring chunks, at most a third of `max_chars`: the chunker lowers a larger value to that. |
| `pack_config.vector.kind` | `local` | `local` | The vector store. Only the local one exists. |
| `pack_config.vector.backend` | `auto`, `faiss`, `hnswlib`, `numpy` | `auto` | The index of the local vector store. `auto`: faiss, else hnswlib, else numpy, whichever is installed. |
| `pack_config.graph.kind` | `local` | `local` | The entity graph. Only the local one exists. |
| `pack_config.graph.terms` | list of text | empty | Entity names to recognise (needed for names the tokenizer cannot find, such as kanji names). |
| `pack_config.graph.max_fanout` | whole number >= 1 | 200 | An entity mentioned by more than this many chunks is always a hub (never followed), whatever `hub_share` gives. It does not cut the chunks of an entity that is followed. |
| `pack_config.graph.hub_share` | number in [0, 1] | 0.05 | An entity mentioned in more than this share of all chunks is a hub and is not followed. |
| `pack_config.graph.hub_min` | whole number >= 0 | 10 | An entity mentioned by this many chunks or fewer is never a hub. |
| `pack_config.retrieve_k` | whole number >= 1 | 5 | Top-k of each retriever at decide time (fused, up to 3k candidates). |
| `pack_config.label_top_k` | whole number >= 1 | 5 | Chunks retrieved per query when label tasks are built. |
| `pack_config.rrf_k` | number > 0 | 60 | The constant of reciprocal rank fusion. |
| `pack_config.batch_size` | whole number >= 1 | 64 | Chunks embedded per batch at ingest. |
| `pack_config.on_encoder_change` | `error`, `reembed` | `error` | What opening a corpus built with another encoder does: refuse, or embed it again. |
| `pack_config.feature_cache` | whole number >= 0 | 1024 | Featurised states kept in memory (0: off). |
| `pack_config.max_evidence_age_days` | number >= 0 | none | Evidence older than this (by its `updated_at`) caps `answerable` at confirm. |
| `pack_config.require_evidence_dates` | true / false | false | Evidence without `updated_at` caps `answerable` at confirm. |
| `pack_config.allow_remote` | true / false | false | Same as `security.allow_remote`. |

## `pack_config` of the tabular pack

One head per label column. Either list the heads under `targets`, or use the one-head shorthand: `label` (and
any of the target keys) directly under `pack_config`. Not both.

| key | type | default | what it does |
|---|---|---|---|
| `pack_config.data` | path | required | The training table: `.csv`, `.tsv`, `.txt` (optionally `.gz`) or `.parquet`. A local file; nothing is fetched. |
| `pack_config.id` | column | none | Row ids: reported, never a feature. |
| `pack_config.group` | column | none | With `train.ordering: group` (or `auto`, which picks group whenever this column is set), rows of one group never straddle train, calibration and holdout. Under `ordering: time` the holdout is one time cut that ignores groups. |
| `pack_config.time` | column | none | ISO 8601 or epoch seconds; lets the holdout be the most recent period (with a `group` and ordering `auto`: the groups whose first row came latest, with all their rows). |
| `pack_config.time_format` | text | none | A Python `strptime` format for `time` values in another notation, e.g. `"%d/%m/%Y %I:%M %p"` for `31/01/2025 02:15 PM`. Times without a zone are read as UTC. Values that do not match it are still tried as ISO 8601 / epoch seconds. |
| `pack_config.segment` | column | none | Per-segment base rates and holdout slices. |
| `pack_config.features` | `auto` or a list of columns | `auto` | `auto`: every column except label, id, group, time, segment and the `exclude` and `text` columns (empty and identifier-like columns are left out, with a note). `auto` cannot know which other columns are only filled in after the decision: list those under `exclude`. |
| `pack_config.exclude` | list of columns | empty | Columns that are never features, whatever `features` says: list every column that is only known after the decision (an outcome you do not model, a review level, a closing date). A column in both an explicit `features` list and `exclude` is an error, and so is a column that is not in the file. Training adds a note when one feature on its own almost determines a label. |
| `pack_config.text` | list of columns | empty | Free-text columns. Each becomes `text_buckets` hashed word and character n-gram features (`text:<column>#<i>`): no vocabulary is learned or stored, so nothing of the text is in the bundle, and it works without spaces between words (English, Japanese, Mongolian). It matches shared words and character runs, not meaning. Changes the model's `feature_version`. |
| `pack_config.text_buckets` | whole number in [8, 4096] | 256 | Hashed features per text column. More buckets: fewer collisions, more features to fit. |
| `pack_config.categorical` | `auto` or a list of columns | `auto` | `auto`: columns whose values are not all numbers. A list adds columns to treat as categories even if numeric (zip codes, plan ids). |
| `pack_config.max_categories` | whole number in [1, 10000] | 50 | One-hot vocabulary size per categorical column. |
| `pack_config.hash_categories` | true / false | false | Store salted hashes of categorical values in the bundle instead of the raw values. Use it when a column could hold personal data. With false, the most frequent values of every categorical column are stored word for word in every bundle. |
| `pack_config.delimiter` | text | `auto` | `auto`: the most frequent of `,` `;` tab `\|` in the header line. |
| `pack_config.encoding` | text | `utf-8-sig` | The file's text encoding. |
| `pack_config.na_values` | list of text | `""`, `NA`, `N/A`, `NaN`, `null`, `<NA>`, `#N/A` | Cells read as missing (case-insensitive). |
| `pack_config.label_source` | `human`, `ground_truth`, `synthetic`, `judge` | `ground_truth` | Where the file's labels come from. `judge` labels train and never calibrate. |
| `pack_config.allow_remote` | true / false | false | Same as `security.allow_remote`. The tabular pack itself never uses the network. |
| `pack_config.targets.<head>.label` | column | required | The label column of the head. `targets: {churned: churned}` is short for `{churned: {label: churned}}`. |
| `pack_config.targets.<head>.type` | `noul`, `choice`, `score` | inferred | 2 distinct values: noul; more: choice. A score head must be asked for. |
| `pack_config.targets.<head>.positive` | a label value | inferred | noul: the value that means true. Without it a conventional word (`true`, `yes`, `1`, ...) is taken, else the larger value: say it when in doubt. |
| `pack_config.targets.<head>.classes` | list | read from the column | The options of a choice head, or the levels of a score head in order. Rows with another value are skipped. |
| `pack_config.targets.<head>.levels` | list | none | Score: level names in order (labels are the names or their 0-based index). |
| `pack_config.targets.<head>.model` | `logreg`, `lightgbm`, `auto` | `auto` | As `heads.<head>.model`. |
| `pack_config.targets.<head>.stakes` | a stakes level | `medium` | As `heads.<head>.stakes`. |
| `pack_config.targets.<head>.decision` | `argmax`, `gate` | `argmax` | As `heads.<head>.decision`. |
| `pack_config.targets.<head>.conformal` | number in (0, 0.5) | none | As `heads.<head>.conformal`. |
| `pack_config.targets.<head>.description` | text | empty | As `heads.<head>.description`; how the type and the classes were chosen is appended. |
| `pack_config.targets.<head>.params` | mapping | empty | As `heads.<head>.params`. |
| `pack_config.targets.<head>.input` | mapping | empty | As `heads.<head>.input`. |
| `pack_config.targets.<head>.enabled` | true / false | true | `false`: no head, and its label column still stays out of the features. |
| `pack_config.label` | column | none | Shorthand: the label column of the only head. |
| `pack_config.question` | text | `outcome` | Shorthand: the name of that head. |
| `pack_config.type` | `noul`, `choice`, `score` | inferred | Shorthand for `targets.<head>.type`. |
| `pack_config.positive` | a label value | inferred | Shorthand for `targets.<head>.positive`. |
| `pack_config.classes` | list | read from the column | Shorthand for `targets.<head>.classes`. |
| `pack_config.levels` | list | none | Shorthand for `targets.<head>.levels`. |
| `pack_config.model` | `logreg`, `lightgbm`, `auto` | `auto` | Shorthand for `targets.<head>.model`. |
| `pack_config.stakes` | a stakes level | `medium` | Shorthand for `targets.<head>.stakes`. |
| `pack_config.decision` | `argmax`, `gate` | `argmax` | Shorthand for `targets.<head>.decision`. |
| `pack_config.conformal` | number in (0, 0.5) | none | Shorthand for `targets.<head>.conformal`. |
| `pack_config.description` | text | empty | Shorthand for `targets.<head>.description`. |
| `pack_config.params` | mapping | empty | Shorthand for `targets.<head>.params`. |
| `pack_config.input` | mapping | empty | Shorthand for `targets.<head>.input`. |
| `pack_config.enabled` | true / false | true | Shorthand for `targets.<head>.enabled`. |

In the tabular pack a `heads.<head>` override may also set `label`.

## A pack of your own

A custom pack (`pack: my_pkg.my_pack:Pack`) is not checked inside `pack_config` until it says which keys it
reads. Add `config_keys()` to the pack and `rlcd config check` treats your keys like the built-in ones:

```python
from open_rlcd.core.config_check import Key

class Pack(BasePack):
    def config_keys(self):
        return {"dsn_env": Key("str"), "window_days": Key("int", default=30, min=1),
                "mode": Key("str", default="fast", choices=("fast", "exact"))}
```

`check_config(project)` may return more findings (`[{level, path, message}]`) for what a key list cannot say,
and `head_override_keys()` names extra fields your pack accepts under `heads.<head>`.

## Environment variables

Secrets are read from the environment only. A value in `rlcd.yaml` that looks like one (a key named
`password`, `token`, `api_key`, ...; a string shaped like a known key) is a warning from `rlcd config check`.
`rlcd config show` prints whether a secret is set, never its value.

| variable | what it does |
|---|---|
| `RLCD_PROJECT` | A project folder or an `rlcd.yaml` to use instead of searching from the current folder. A path without an `rlcd.yaml` is an error (a mistyped mount never means "no project"). |
| `RLCD_HOME` | The artifacts folder; overrides `home`. |
| `RLCD_PACK` | The pack when `rlcd.yaml` names none (or there is no `rlcd.yaml`). |
| `RLCD_API_KEY` | Secret. When set, every route except `/health` and `/ready` needs `Authorization: Bearer <key>`; the Python client sends it. `rlcd serve` on a host that is not loopback refuses to start without it. |
| `RLCD_ADMIN_KEY` | Secret. When set, training, the kill switch and confirmations (`POST /v1/train`, `POST /v1/automation`, `POST /v1/receipts/{id}/confirm`) need this key instead. It protects only those three routes: without `RLCD_API_KEY` everything else stays open. |
| `RLCD_ALLOW_REMOTE` | `1` lets connectors and judges reach hosts that are not loopback (as `security.allow_remote`). |
| `RLCD_AUTOMATION` | `off` forces the kill switch off: every decision is capped at confirm and the API cannot switch it back on. |
| `RLCD_MAX_BODY_BYTES` | Largest request body the API accepts (default 16 MiB). |
| `RLCD_URL` | Where the Python client finds the service (default `http://127.0.0.1:8095`). |
| `RLCD_DEBUG` | `1` makes the `rlcd` command print a full traceback instead of one line. |
| `ANTHROPIC_API_KEY` | Secret. The key of the `anthropic` judge. |
| `OPENAI_API_KEY` | Secret. Sent by the `openai:` judge when set. |
| `HF_HUB_OFFLINE` | `1`: the ONNX encoder downloads nothing and fails if its files are not already in `<home>/encoders`. |

## When does a change take effect?

Every `rlcd` command reads `rlcd.yaml` when it starts. A running `rlcd serve` read it once, when it started, and
does not read it again: there is no hot reload. And a trained version carries part of the configuration inside
its bundle, which only a retrain replaces. So the same edit can apply at once in `rlcd decide` and not at all in
the server next to it. `GET /health` shows what the server runs with (`model`, `rollout`, `automation`,
`ready`).

| you change | `rlcd` commands (`decide`, `ask`, `check`, ...) | a running `rlcd serve` |
|---|---|---|
| `policy.*` (thresholds, `abstain`, `min_verify_rows`, `gate_min_p_above`, `overrides`) | after `rlcd train`: a version serves the policy stored in its bundle, the one its holdout verified | after a retrain that read the edited file: `rlcd train`, or a restart followed by `POST /v1/train` (the server trains with the file it read at start). The new version is then served without another restart |
| `heads.<head>.stakes` and the other head fields (`type`, `classes`, `model`, `decision`, `conformal`, ...) | after `rlcd train`: a trained head is served as it was trained | after `rlcd train`, as above |
| `heads.<head>.enabled` | at once | restart |
| `train.*` | the next `rlcd train` | restart, then the next `POST /v1/train` |
| `pack_config`: what features are built from (tabular: the table and its columns; documents: the encoder, chunking) | after `rlcd train` (documents: `rlcd ingest` first): the trained schema and encoder are stored in the bundle | after `rlcd train`, as above |
| `pack_config`: decide-time settings of the documents pack (`retrieve_k`, `max_evidence_age_days`, `require_evidence_dates`, `feature_cache`, and what opening the corpus reads: `rrf_k`, `vector.backend`, `graph.terms`, `graph.max_fanout`, `graph.hub_share`, `graph.hub_min`, `on_encoder_change`; new `graph.terms` link chunks only after a re-ingest) | at once | restart |
| `rollout.*`, `routes`, `batteries`, `evaluate.*` | at once | restart |
| `serve.log_states`, `serve.redact`, `serve.receipts`, `serve.review_capacity_per_day`, `security.allow_remote`, `regressions`, `owners` | at once | restart |
| `serve.host`, `serve.port`, `home`, `pack` | the next command | restart |
| `labeling.*` | the next `rlcd label ...` | not read by the server |
| environment variables (`RLCD_API_KEY`, `RLCD_ADMIN_KEY`, `RLCD_AUTOMATION`, ...) | the next command | restart |

Not in the file, and seen by a running server on its next request without a restart: the kill switch (`rlcd
automation off`, `POST /v1/automation`), which version `rlcd-latest` points at (`rlcd promote`, `rlcd rollback`,
`rlcd import`, a finished train), and the Bernoulli ledger (every `/v1/feedback`).

When `rlcd.yaml`'s `policy` (or a head's `stakes` or `conformal`) is not what the served version uses, `rlcd
decide`, `rlcd ask` and `rlcd serve` (at start) print one line to stderr saying so, and `rlcd check` reports it
as `thresholds.config`. A version trained before `rlcd config check` existed whose stored policy is unusable
(`abstain: act`) refuses to decide (422 `bad_bundle`) until it is retrained or rolled back.

## Worked examples

### Documents: the gate in front of a RAG system

```yaml
name: support-search
pack: documents
home: ./var
pack_config:
  encoder: {kind: onnx, model_id: intfloat/multilingual-e5-small}
  corpus: ./data
  queries: ./queries.jsonl
  chunking: {max_chars: 1200, overlap: 150}
  max_evidence_age_days: 365      # year-old evidence is confirmed by a person, not acted on
heads:
  answerable: {stakes: medium, conformal: 0.05}   # promise: at most 5% of the yeses are wrong
  grounded: {stakes: high}
  needs_graph: {enabled: false}                    # this deployment has no entity graph
batteries:
  preflight:                       # the pack's preflight also asks needs_graph, which is off here
    version: "1-nograph"
    owner: search-team
    changelog: ["1-nograph: the pack's preflight without multi_hop (needs_graph is off)"]
    questions:
      kind: {type: choice, head: response_type}
      route: {type: choice, head: route}
labeling: {judge: claude-cli, batch: 20}
serve: {host: 127.0.0.1, port: 8095, redact: [secrets, pii]}
rollout: {mode: shadow}            # log every decision, automate nothing, until `rlcd check` is clean
owners: {pack: search-team, heads: search-team, policy: risk, review: support, incidents: on-call}
```

### Tabular: two questions about one table

```yaml
name: churn
pack: tabular
pack_config:
  data: data/customers.csv
  id: row_id
  group: customer_id               # with ordering auto (below), a customer's rows stay on one side of the holdout
  time: month                      # orders the groups: the holdout is the customers who appear latest
  hash_categories: true            # the bundle stores hashes of categorical values, not the values
  targets:
    churned: {label: churned, positive: "yes", stakes: medium}
    next_plan: {label: plan_tier_next, type: choice, classes: [downgrade, same, upgrade], stakes: low}
train: {holdout_fraction: 0.2, n_splits: 5, ordering: auto}   # `time`: the latest months, customers on both sides
policy: {abstain: hold}
rollout:
  mode: limited
  limited: {stakes: [low]}         # only next_plan may act; churned decisions are confirmed by a person
```

### A strict policy for high stakes

```yaml
pack: documents
heads:
  answerable: {stakes: critical, conformal: 0.02}
  grounded: {stakes: critical}
policy:
  abstain: escalate
  min_verify_rows: 100             # act is not verified on fewer holdout rows
  gate_min_p_above: 0.95
  stakes:
    high: {act: 0.97, confirm: 0.9, min_accuracy: 0.95}
    critical: {act: 0.99, confirm: 0.95, min_accuracy: 0.98}   # a level of your own needs all three
  overrides:
    response_type:
      out_of_scope: {"*": {act: 0.99}}     # refusing a customer needs more than answering one
rollout: {mode: assist}            # a person confirms every action
serve: {review_capacity_per_day: 40}
evaluate: {max_revisions: 1, output_scan: [secrets, pii]}
```

Stricter thresholds do not make the model better: they move decisions from act to confirm and abstain, and
`serve.review_capacity_per_day` is there so `rlcd check` tells you when that is more than your reviewers can
take. A head needs enough trusted holdout rows to be verified at a threshold at all. `rlcd train` prints only
the verdict at `medium`; `rlcd check` (item `skill.<head>`) and `rlcd calibration <head>` print it per stakes
level, `critical` included.
