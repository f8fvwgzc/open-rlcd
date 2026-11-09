# The wire contract

Open RLCD speaks the evaluation contract of Typesafe's "System One" documentation (one evaluation endpoint,
three question types, the same confidence formulas) with the same additive extensions as the View Engine RLCD
service it was extracted from. **A client of the View Engine RLCD `/v1/systemone` works unchanged**: same
request, same answer objects, same `usage`, same `warnings`, same errors, same model ids.

Default address `127.0.0.1:8095`. Errors are `{"error": "...", "kind": "...", ...details}`:

| status | kind | when |
|---|---|---|
| 401 | `unauthorized` | only when `RLCD_API_KEY` is set and the request lacks `Authorization: Bearer <key>` (or `x-api-key`); `/health` is always open |
| 404 / 405 | `not_found` / `method_not_allowed`, `unknown_job` | unknown route or method; unknown training job |
| 409 | `busy` | a training job is already running (`job_id` given) |
| 413 | `too_large` | request body larger than `RLCD_MAX_BODY_BYTES` (default 16 MiB) |
| 422 | `validation` | malformed body, limits exceeded, criteria that do not match the head, unknown `stakes` (lists the configured levels) |
| 422 | `unknown_head` | lists `available_heads` |
| 422 | `unknown_model` | lists `available` (`rlcd-latest` + versions) |
| 422 | `feature_version_mismatch`, `feature_names_mismatch`, `bad_feature_vector` | a precomputed `{x, feature_version}` state that does not fit the trained head |
| 422 | `encoder_mismatch`, `feature_schema_mismatch` | the configured encoder / feature schema is not the one the bundle was trained with |
| 422 | `bad_bundle` | a bundle names a pack that is neither the project's pack nor a built-in one (bundles never name code to import) |
| 422 | `untrained`, `unknown_decision`, `unsupported` | report of an untrained head, feedback for an unknown `decision_id`, an endpoint the pack does not offer |
| 503 | `connector_unavailable`, `encoder_unavailable` | the corpus / a vector or graph store cannot be opened or searched; the encoder cannot be loaded (messages never contain paths or DSNs; details are in the server log) |
| 500 | `internal` | the exception type and a generic message; details are in the server log |
| 422 | `missing_evidence` | a question's `path` / `inputs` does not resolve in the state, or a `strict` request lacks inputs almost every training row had (`missing`, `missing_groups`) |
| 422 | `criteria_order` | a score question's `criteria` list the head's levels in another order (levels are indexed as the head defines them) |
| 422 | `unknown_battery`, `battery_version_mismatch` | a battery that does not exist (lists `available`); a pinned battery version that is not the current one (`expected`, `current`, `changelog`) |
| 404 | `unknown_decision` | `GET /v1/receipts/{id}` and its check for an unknown decision id (also 422 for an unknown `decision_id` in feedback, outcome, `plan` or `revision.of`) |
| 422 | `no_receipt` | a receipt check on a decision logged before receipts existed |
| 409 | `automation_locked` | switching automation on while `RLCD_AUTOMATION=off` is set in the server's environment |
| 500 | `bad_config` | an invalid control-layer section in rlcd.yaml (`rollout`, `routes`, `batteries`, `serve.receipts`, `serve.redact`, `evaluate.max_revisions`, `evaluate.output_scan`) |
| 422 | `state_mismatch` | a confirmation whose `state_digest` is not the digest of the state the decision judged |

Every error body also carries `retryable`: `true` only for 409 `busy`, 429 and 503, which come with a
`Retry-After` header (seconds). Retry those with bounded backoff; never retry anything else unchanged — a 422 needs
a different request. **Failure behaviour for callers**: an error, a timeout or an unreachable service means *the
review is unavailable*, never approval: fall back to proposal-only or a person (the Python client's
`fail_closed` does exactly that).

With `RLCD_ADMIN_KEY` set, `POST /v1/train`, `POST /v1/automation` and `POST /v1/receipts/{id}/confirm` need that key (`Authorization: Bearer`),
so a caller holding only `RLCD_API_KEY` can neither retrain nor switch automation back on; the admin key is also
accepted wherever `RLCD_API_KEY` is.

## `POST /v1/systemone`

```json
{"model": "rlcd-latest",
 "state": {"query": "Who leads the team that owns the K-200 scanner?"},
 "questions": {
   "kind":  {"type": "choice", "head": "response_type"},
   "graph": {"type": "noul", "head": "needs_graph", "criteria": {"true": "multi-hop", "false": "single lookup"}},
   "route": {"type": "choice", "instructions": "ignored: the head defines the question"}}}
```

```json
{"model": "rlcd-0.1.3",
 "answers": {
   "kind":  {"type": "choice", "choice": "direct_answer", "probabilities": {"direct_answer": 0.81, "...": 0.01},
             "confidence": 0.78, "head": "response_type", "calibrated": true},
   "graph": {"type": "noul", "noul": 0.93, "confidence": 0.86, "head": "needs_graph", "calibrated": true},
   "route": {"type": "choice", "choice": "graph", "probabilities": {"vector": 0.04, "graph": 0.9, "hybrid": 0.05,
             "none": 0.01}, "confidence": 0.87, "head": "route", "calibrated": true}},
 "usage": {"input_tokens": 0, "output_tokens": 0, "questions": 3, "latency_ms": 6.1}}
```

Answers: noul `{type, noul, confidence, head, calibrated}`, choice `{type, choice, probabilities, confidence,
head, calibrated}`, score `{type, score, legend, probabilities, confidence, head, calibrated}` (levels indexed
from 0). Limits: at most 255 choice options, 2 to 10 score levels, 64 questions, 200 rank candidates.

Confidence: noul `|2p - 1|`; choice `(p_max - 1/n) / (1 - 1/n)`; score `max(0, 1 - sum_i p_i |i - m| / MAD_unif)`.
Confidence describes the shape of a distribution, not whether it is right: whether probabilities can be trusted
is what `/v1/calibration` publishes. **Do not gate actions on `confidence` alone**: a head whose holdout showed no
skill still answers (`calibrated: true`, base-rate probabilities that may look confident on imbalanced classes),
with `skill: false` on the answer and a top-level warning. Gate on `skill` too, or use `POST /v1/decide`, which
applies the holdout verdict and the Bernoulli gate for you.

The answer key set of the View Engine RLCD service is kept; `skill` is the only added key. Clients that parse
answers as JSON objects (as the View Engine client does) are unaffected.

### Where Open RLCD deviates from or extends the reference (identical to View Engine RLCD)

| field | where | meaning |
|---|---|---|
| `head` | question | which head answers it; defaults to the question id. Unknown head -> 422 listing the heads |
| `criteria` | question | optional for a choice (the head fixes the options); if given, its keys must equal the head's classes; for a score, its length must equal the head's levels |
| `instructions` | question | accepted and ignored: there is no language model |
| `model` | request | optional, defaults to `rlcd-latest`; versions are `rlcd-0.1.N`, `rlcd-0.1.0` = untrained |
| `confidence` | noul answer | `|2p - 1|` (the reference documents the formula but does not return it) |
| `head`, `calibrated` | every answer | the head that answered; `false` = untrained head, the answer is the uninformed prior |
| `skill` | every answer | open-rlcd: whether the head's trusted holdout showed reliable skill over the base rate. With `skill: false` the probabilities are still calibrated, but they are base-rate knowledge |
| `usage.questions`, `usage.latency_ms` | response | token counts are always 0 |
| `warnings` | response | present when a head is untrained or its holdout showed no skill |

State: whatever the head's feature view expects (`GET /v1/heads` documents each `input`), or a precomputed
feature vector `{x, feature_names, feature_version}`; a vector of another `feature_version` is refused with 422.

### Open RLCD additions (new endpoints and fields, all additive)

| endpoint / field | meaning |
|---|---|
| `POST /v1/rank` | `{model?, state?, candidates: [{id, state}], question?, option?}` -> candidates sorted by the positive probability. `question` defaults to the pack's ranking head (documents: `chunk_relevant`); per-candidate state is merged over the shared state. |
| `POST /v1/decide` | heads -> policy -> `act / confirm / escalate` (the tabular pack names it `hold`), with `thresholds`, `max_tier`, the Bernoulli `gate`, `reasons`, `warnings` and a `decision_id`. The documents pack returns a plan: `response_type`, `route`, `needs_graph`, `use_chunks` (candidates with P(relevant) >= `min_relevance`, at most `top_k`), `call_llm` (true exactly when `next` is `answer`), `next` (`answer`, `ask_clarifying_question`, `escalate_to_human` — also whenever response_type abstains —, `refuse_out_of_scope`, `retrieve_more` — when no evidence passes), `chain` and `reason`. Without `now` in the state, document ages are measured from the request time. |
| `POST /v1/evaluate` | evaluator mode (documents): judge a drafted answer — `grounded`, `complete`, `risk_tier` -> ship / review / reject as act / confirm / escalate. |
| `POST /v1/feedback` | `{decision_id?, head?, state? \| x + feature_version + feature_names, label, weight?, source}`: updates the Beta-Bernoulli ledger at once (with a `decision_id`) and becomes an up-weighted training row at the next train (weight x `train.feedback_weight`). `x` must match the current model's feature schema (422 otherwise). A second outcome for the same `decision_id` with another label is a correction: it replaces the first in the ledger (`bernoulli.corrected: true`) and in training. After a partial retrain, carried-over heads keep their ledger, so outcomes of earlier decisions still count. A state is capped at 2 MiB of JSON and `x` at 100,000 values; `feedback.jsonl` is append-only and never rotated by the service (archive it after a train if it grows large). |
| `POST /v1/train`, `GET /v1/train/{job}` | background training job (one at a time). |
| `GET /v1/calibration?head=&model=&segment=` | the trusted-holdout report: log loss, Brier, Brier skill vs base rate, reliability bins, ECE, block-bootstrap skill test, policy results per stakes level, verdict, conformal threshold and its holdout check, LLM-judge agreement with humans. |
| `GET /v1/bernoulli?head=&stakes=` | the posterior table per (committed class, probability bin). |
| `stakes` | decide / evaluate request: any level the policy defines (default `low`, `medium`, `high`; thresholds are configuration and returned with every decision). Unknown level -> 422. |
| input guard | a request that lacks inputs almost every training row had (missing or non-finite values) gets a warning naming them, and that head's decision is capped at `confirm` (answers stay `calibrated: true`). |
| `/v1/models`, `/v1/heads` | every head of each version with `trained`, `n_train`, `trained_at`, `model_version`, `metrics`, `verdict`; `/v1/heads` always has `n_train` and `positive` (null when unknown). |
| `explore` | decide / evaluate request: Thompson sampling decides the Bernoulli gate; labelled `exploration: true`. Off by default. |

## `/v1/decide` decision object (per head)

```json
{"head": "answerable", "decision": "gate", "stakes": "medium", "class": "true", "commit": "true", "p": 0.91,
 "tier": "act", "uncapped_tier": "act", "max_tier": "act",
 "thresholds": {"act": 0.85, "confirm": 0.65, "min_accuracy": 0.75, "conformal": 0.88},
 "bernoulli": {"alpha": 41, "beta": 3, "n": 42, "mean": 0.93, "ci95": [0.83, 0.98], "required": 0.75,
               "p_above_required": 0.999, "wins": 40, "losses": 2, "bin": 9, "decisions_needed": 17},
 "gate": {"mode": "posterior", "threshold": 0.9, "required_accuracy": 0.75, "passed": true},
 "calibrated": true, "warnings": []}
```

`tier` is never stronger than `max_tier` (the head's holdout verdict at that stakes level) and `act` needs the
Bernoulli gate to pass. A gate head (`answerable`, `grounded`) commits to `true` only when its tier is above the
abstain action.

## The control layer (additive; see [control-layer.md](control-layer.md))

Request fields (every one optional; without them a request is answered exactly as before):

| field | where | meaning |
|---|---|---|
| `path`, `inputs` | question | where the head's inputs live in a shared, named state: `path` = a sub-object, `inputs` = `{field the head reads: "dotted.path"}` (integer segments index arrays). Unresolvable -> 422 `missing_evidence` |
| `allowed` | choice question; systemone `{question id: [options]}` (narrows, also a battery's questions); decide `{head: [options]}` | options deterministic code kept. The answer lists only those, with calibrated probabilities **not renormalised**, plus `allowed` and `excluded_mass`; the decision commits to the best allowed option with its own probability |
| `battery` | systemone (instead of `questions`) | a named, versioned question set, optionally pinned `name@version`; the response adds `battery: {name, version, digest}` |
| `strict` | systemone, decide, evaluate | refuse (422 `missing_evidence`) a state that lacks inputs almost every training row had, instead of capping the decision at confirm |
| `receipt` | systemone, rank | record this call as a receipt and return it (`serve.receipts: all` does it for every call) |
| `capabilities`, `sensitivity`, `latency_budget_ms` | decide | filter the route contracts of rlcd.yaml `routes:` before the semantic choice |
| `baseline`, `artifact`, `checks`, `tags`, `revision` | decide / evaluate | kept in the receipt's `audit`; `artifact`, `checks` and `revision` also take part in evaluate's decision table |
| `return_evidence` | decide (documents) | return the kept chunks `{id, doc_id, title, text, updated_at, authority, score, rank}` |
| `plan`, `citations` | evaluate (documents) | bind the review to the evidence of a decide plan (and use its response_type contract's `max_retries` as the revision budget); the chunk ids the answer cites (outside the evidence -> reject; `null` -> a warning) |
| `tags`, `note`, `missing_evidence` | feedback | review annotations; `tags: ["regression"]` turns the corrected case into a regression fixture |

Response fields: decide / evaluate add `receipt: {id, digest, prev, state_digest, contract_digest,
questions_digest}` and `rollout: {mode, automation, capped, proposed_action}`; each head result adds
`uncertainty: {sources, margin, entropy, top2}` and, when route contracts apply, `considered` and `contract`; a
head decision adds `rollout: {mode, automation, cap, capped, proposed_tier?, reason?}` (and `caps` when a
deterministic rule lowered it); the documents plan adds `contracts`; evaluate adds `next` (`ship`,
`human_review`, `revise`, `retry_generation`, `escalate_to_human`), `decided_by` (`checks`, `artifact`,
`citations`, `output_scan`, `revision`, `heads`), `revise` and `revision`. `/health` adds `rollout` and `automation`.

New endpoints:

| endpoint | meaning |
|---|---|
| `GET /v1/batteries?model=` | the batteries with version, owner, changelog, consumers, digest, their questions and any problem (missing / untrained head) |
| `GET /v1/receipts/{id}` | one decision record (with logged states when `serve.log_states`) |
| `POST /v1/receipts/{id}/check` | `{state | state_digest, max_age_s?}` -> `{current, stale_reasons, checks}`: does the decision still approve acting on this state now? A `confirm` decision is current only after a person confirmed it; an abstained decision (escalate / hold, a rejected answer) approves nothing and is never current |
| `POST /v1/receipts/{id}/confirm` | `{by, note?, state_digest?}`: a person confirmed a confirm-tier decision (kept in `confirmations.jsonl`; a `state_digest` that is not the receipt's is a 422 `state_mismatch`; with `RLCD_ADMIN_KEY` set it needs that key) |
| `GET` / `POST /v1/automation` | the kill switch `{enabled, reason?, by?}`, the rollout mode and the switch history |
| `POST /v1/outcome` | a decision-level outcome `{decision_id, task_success?, route_taken?, provider?, fallback?, latency_ms?, cost?, tokens?, retries: {transport, workflow}?, verified_by?, note?, tags?}`; joined by the metrics, never trains |
| `GET /v1/metrics?since=&until=&model=&kind=&head=` | the operations report: action mix (automatic action, review, abstention), rollout caps, review load against `serve.review_capacity_per_day`, per head the tier / class mix, entropy, uncertainty sources and — joined with feedback — accuracy, false automation, false rejection, review agreement and online reliability next to the holdout (`vs_holdout`, `drift_suspected`), binding links, plan and verdict distributions, outcomes (task success, routes, fallbacks, cost, latency, transport vs workflow retries), latency percentiles, notes on what the numbers can and cannot say. `since` inclusive, `until` exclusive (ISO 8601) |
| `GET /v1/review?limit=1..1000&head=&kind=&include_abstain=&oldest_first=` | confirm and abstain decisions without feedback, oldest first, with what a reviewer needs (heads, uncertainty, state, warnings, why) |
| `GET /v1/check?model=` | the pre-launch checklist `{ok, model, checks: [{id, status, title, detail}], counts}`; `ok` is false only when a check fails (ids in `core/checklist.py`) |

`POST /v1/feedback` with `tags: ["regression"]` also adds a regression case (`regression_case: {id, stored: true}`
or `{stored: false, reason}`) to the project's regressions file (rlcd.yaml `regressions:`, default
`./regressions.jsonl`), which `rlcd regress` runs against every version (formats in `core/regress.py`).
