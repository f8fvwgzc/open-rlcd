# Weight edits (experimental)

`rlcd patch` lets a team try a direct edit of a trained model's weights, made by hand, by a script, or by an LLM
doing "weight-space programming", without ever letting the edited model serve on faith. The edit becomes a
**candidate version**. It is recalibrated on trusted rows and scored once on the base version's untouched holdout,
through the same code path a freshly trained head goes through. It becomes `rlcd-latest` only if that holdout
supports it.

## Why: an edit is a claim, and the holdout decides

Weight-space programming is having a moment. One reported example, from MIT CSAIL's "Sparks of Weight-Space
Programming" (November 2026): an LLM edited about 0.016 % of a 0.8B-parameter model's weights without gradient
descent, and a maths benchmark went from 7.9 % to 48.1 %. Critics say the edit mostly fixed repetition loops rather
than adding an ability, and the result had not been independently replicated when this was written. Whatever the
outcome there, the lesson for a decision service is the same. An edit is a hypothesis about the model ("this
coefficient encodes a shortcut", "this direction causes the failure"), and open-rlcd already has a referee for
hypotheses about probabilities: calibration on trusted rows, then one scoring on a holdout the model never saw.
So an edit is held to exactly the standard a training run is held to, and nothing less.

## Two kinds of edit

| | head edit (`--delta`) | encoder edit (`--encoder`) |
|---|---|---|
| input | a `.safetensors` file of **additive deltas** to one head's logistic-regression arrays, named `member_<k>.coef`, `.intercept`, `.mean`, `.scale`, `.fill` (the arrays of `member_<k>.safetensors`) | a replacement ONNX text encoder (documents pack) |
| what is kept | the base models with the delta applied | nothing: a new encoder sha256 is a new `feature_version`, and no head trained on the old features is valid on the new ones |
| what is redone | every member's calibrator, refitted on its own fold's **trusted calibration block** | every head of the base version is **retrained** (`Trainer.train_heads`) with the base's head specs and split settings |
| scored on | the base head's holdout rows | the same holdout rows as each base head, paired row by row |

```bash
# zero the coefficients of a feature you believe is a shortcut, in every member, then let the holdout judge
rlcd patch --head answerable --delta no-leak.safetensors \
    --author "LLM: <model>" --tuned-on development --description "drop the leaked feature"
rlcd patch --encoder e5-small-edited-int8.onnx --author "script:steer.py" --tuned-on none
```

A delta file may also describe itself in its safetensors metadata: `author`, `tuned_on`, `tuned_on_ids` (a JSON
list), `description`, `head`. If the file and the command disagree, the edit is refused. Exit status: `0` the
candidate was promoted to `rlcd-latest`, `3` it was kept as a non-latest version, `1` it was refused (nothing was
scored or saved).

## The gate

1. **Re-derive the base's holdout.** The base head's split settings come from its report (fraction, ordering, seed,
   folds), and the holdout is rebuilt from today's rows. It must be the one the base was scored on: same row-id
   hash, and the base's published Brier and log loss must come out again (for a head edit, also the same
   calibration blocks). If the labelled rows have changed since then, the edit is refused. Retrain instead, or
   edit the retrained version.
2. **Refuse holdout tuning** (next section).
3. **Recalibrate or retrain, then score once** with `training.finish_head`: the report, block-bootstrap skill test,
   conformal threshold and its holdout check, per-stakes policy verdict, and Bernoulli table.
4. **Paired test against the base.** For each changed head, take the per-row Brier gain and log-loss gain of the
   candidate over the base on the same holdout rows, sum them per block (whole groups, or contiguous time blocks),
   and resample blocks with replacement. Both one-sided lower bounds must be > 0. Pairing removes the difficulty
   both models share, so the test sees only how they differ.
5. **No weaker verdict.** For every stakes level, the candidate's verdict (abstain < confirm < act) must be at
   least the base's.
6. **Promote, or keep with the reason.** The candidate becomes `rlcd-latest` only if every changed head passes
   **and** the base is still `rlcd-latest` (an edit is judged against its base, not against whatever is newer).
   That check and the save happen inside the registry's inter-process lock, so a `rlcd train`, `rlcd import` or
   `rlcd promote` running beside it cannot slip in between. A candidate that passed but whose base has moved on
   is kept and can be served by name. A candidate that failed is kept for inspection (`rlcd report --model <id>`,
   `/v1/calibration`) and **never serves**: every answering path refuses it, `rlcd promote` and `rlcd rollback`
   (with or without `--to`) never make it latest, and `rlcd import` of an exported copy keeps it out of service.

An encoder edit changes every head's inputs, so every head must pass. To fix one head, edit that head.

## The holdout judges; it never chooses

If an edit is chosen while looking at holdout rows, the gate turns into a fit and its promise means nothing.

* Every edit must say what it was chosen or tuned on: `--tuned-on none | development | holdout`. `holdout` is
  refused. It can also list the row ids it looked at (`--tuned-on-ids ids.txt`). If any of them is a holdout row
  of the base, the edit is refused before anything is scored. This is a declaration, not a proof: the edit's
  author has to be honest about it, and the provenance records what they declared.
* Each scoring uses up some of the holdout's credibility. The k-th candidate scored on the same holdout (same
  head, same row ids) is tested at a one-sided level of 0.05 / k, and the attempt number is published with the
  result. This slows the search but does not make an unlimited one safe. After many attempts, the remedy is fresh
  trusted labels, which give a new holdout.

## Provenance

`model.json` (`edit`), each changed head's hash-checked `head.json` (`edit`), `registry.json` and `MODEL_CARD.md`
all record:

* the edit's sha256;
* who or what produced it;
* what it was tuned on (listed ids as a count and a hash);
* the number of parameters touched and the total (for ONNX, counted by diffing the graph initializers);
* the base version and the holdout-id hash;
* the paired-test result and the attempt number;
* the verdicts before and after;
* the outcome and its reason.

The policy caps of an edited head come from its `head.json`, the same as for any head.

## After an edit

A promoted edit is a version like any other ([control layer](control-layer.md)):

* **History and rollback.** Its promotion is recorded in `registry.json` `history` (kind `patch`), like a train or
  an import, so `rlcd rollback` returns to the version it edited. Each head gets its own Bernoulli evidence back:
  the edited head the base's table, outcomes scored for the base included, and a carried-over head keeps its
  table. A candidate kept out of service moves nothing and adds nothing to the history.
* **Checks.** `rlcd check` runs the pre-launch checklist on it: `rollback.ready` names the version `rlcd rollback`
  would return to, and `regressions.cases` runs the project's regression cases through the edited heads.
  `rlcd replay` compares it with its base on logged decisions.
* **Keep training on an edited encoder.** An encoder edit is served from `encoders/edited/<sha256>.onnx` in the
  home, but `rlcd train` builds features from `pack_config.encoder` in rlcd.yaml. Until rlcd.yaml points there
  too (`path: encoders/edited/<sha256>.onnx`, `sha256: <sha256>`; the path must stay inside the home), the next
  train is on the original encoder again: a new feature version, so no edited head is carried over.

## Honest limits

* **An edit can remove an obstacle. It cannot add an ability.** A head is a linear map over frozen features, so an
  edit can only re-weight what the features already carry. Dropping a leaked or spurious feature, or undoing a
  bad bias, can help. No edit makes the encoder see something it does not encode. (The repetition-loop reading of
  the CSAIL result is the same kind of story: an obstacle was removed.)
* **Distributed errors are poor targets.** When an error is spread thinly over many weights (biased labels, a
  slightly wrong direction learned from all features at once), no sparse edit locates it. Better labels and a
  retrain do. Pure rescaling is pointless too: the calibrators are refitted, so an edit that only scales or
  shifts a logit is absorbed by design.
* **Edit quantized encoders in fp32 and re-quantize.** The default encoder is int8: each weight is a small integer
  times a per-tensor or per-channel scale. Nudging an int8 value moves it by a whole quantization step, and
  smaller changes vanish. Changing a scale moves an entire channel. Edit the fp32 model (`onnx/model.onnx`),
  re-quantize it (`onnxruntime.quantization.quantize_dynamic`), and submit the int8 file: the submitted file is the
  one that is hashed, scored and served.
* **Reliable is not large.** A lower bound above 0 says the gain is real on rows like the holdout, not that it is
  worth having. Read the size of the gain in the report.
* **Retraining is noisy.** An encoder edit compares retrained heads to the base heads, so part of any difference is
  retraining variance. Requiring every head to pass keeps that noise from promoting an edit.
* **Scope.** LightGBM members are not editable here. After an encoder edit, re-ingest the local corpus (or set
  `pack_config.on_encoder_change: reembed`) before `/v1/decide` retrieves with the new version. Clients that send
  precomputed feature vectors must send the new feature version.
* **Not a security boundary.** Anyone who can write to `RLCD_HOME` can change `registry.json` and `model.json`,
  which are not hashed. `head.json` is hashed. The gate protects against adopting an edit on faith, not against an
  attacker who already controls the model directory.
