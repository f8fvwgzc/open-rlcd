# The control layer: open-rlcd around an LLM

An independent field guide, *How to use Jev with LLMs* (about TypeSafe's "System One" decision model, whose wire
contract open-rlcd speaks), describes how a typed decision model belongs next to a generative model:

> Let the LLM generate. Let the decision model answer narrow typed questions. Let deterministic code own
> authority. Let measured outcomes decide whether the system earns more autonomy.

Open RLCD is an open-weight decision model of that kind. This page maps the guide's ten-step playbook onto what
open-rlcd provides, with the request fields, endpoints and commands for each step, and says plainly what stays
your application's job. Everything here is additive: a request without the new fields is answered exactly as
before, and a View Engine RLCD `/v1/systemone` client works unchanged.

## The request path

```
request
  -> your deterministic validation: schema, identity, capability, allowlists, data egress
  -> POST /v1/decide            allowed options and route contracts filter first, then the typed choices
                                (response type, retrieval route), chunk relevance, the evidence gate
  -> your code                  branches on the action: act goes on, confirm waits for a person (POST
                                /v1/receipts/{id}/confirm), anything else stops; plan.call_llm false means no
                                generation (plan.next says what to do instead)
  -> POST /v1/receipts/{id}/check   the plan's approval still holds for this state, right before its context
                                leaves for the LLM
  -> your code                  assembles context from plan.use_chunks (or return_evidence) and picks the LLM from
                                plan.contracts
  -> the LLM                    writes
  -> your deterministic checks  parsers, schemas, tests, allowlists on the output
  -> POST /v1/evaluate          your check results, the artifact status and citations first, then grounded /
                                complete / risk_tier; next = ship | human_review | revise | retry_generation |
                                escalate_to_human
  -> POST /v1/receipts/{id}/check   the evaluation's receipt, right before the answer ships or the side effect runs
  -> your code                  executes with least privilege, or asks a person
  -> POST /v1/feedback, /v1/outcome   what really happened (labels per head; task success, route, cost)
```

Which step is "the action" is yours to say: when sending context to a provider is itself the risk (data egress),
gate the LLM call on the plan's receipt as above; when only shipping the answer is, the evaluation's receipt is the
gate. Read-only, low-risk flows may need only one side of this. A decision call that cannot change routing,
context or review is only latency: `GET /v1/metrics` shows which heads ever bind a decision.
`examples/control-layer/run.py` walks this path end to end.

## 1. Draw the boundary

Exact rules stay in your code; open-rlcd answers bounded judgments over messy text (noul = P(yes), choice = one
of N, score = an ordered level) with calibrated probabilities; the LLM writes. open-rlcd never writes prose and
never authorises a side effect: `act` means *eligible for your next policy gate*, never "approved". Identity,
permissions, rate limits and user confirmation are checks your code makes after it.

## 2. Build state: named fields, nothing guessed

Send one structured state and let each question say which part its head reads:

```json
{"state": {"request": {"text": "Refund my duplicate charge"},
           "retrieval": {"passages": [{"id": "p1", "text": "..."}]},
           "llm_proposal": {"text": "We refunded 39 USD."}},
 "questions": {
   "kind":     {"type": "choice", "head": "response_type", "inputs": {"query": "request.text"}},
   "enough":   {"type": "noul", "head": "answerable",
                "inputs": {"query": "request.text", "evidence": "retrieval.passages"}},
   "grounded": {"type": "noul", "head": "grounded",
                "inputs": {"query": "request.text", "evidence": "retrieval.passages",
                           "answer": "llm_proposal.text"}}}}
```

* `inputs` maps each field the head reads to a dotted path (`items.0.text` indexes arrays); `path` picks a whole
  sub-object. A path that does not resolve is a **422 `missing_evidence`**: a head never judges a guessed field.
* A state that lacks features almost every training row had is capped at `confirm` on decide and evaluate and
  answered with a warning on systemone; `strict: true` (all three) refuses it instead (422). The guard sees only
  features that come out missing: an optional field you leave out can read as an ordinary value (on systemone and
  rank the documents pack reads absent `candidates` as zero candidates; decide retrieves them, or warns when no
  corpus is ingested), so send every field the head's view lists, `now` included.
* Treat everything in the state as untrusted data: heads score features, they do not follow instructions found in
  a document.
* Secrets never belong in a state. As a safety net, logged states are redacted (`serve.redact`, on by default:
  API keys, tokens, private keys, passwords in URLs; `[secrets, pii]` also masks e-mails and card numbers).
* Preview what heads answer on a state: `rlcd ask --state @state.json --heads answerable` (a System One call, no
  HTTP; the state is the head's own view, e.g. `{"query": ..., "evidence": [...]}`); `rlcd ask --query "..."` asks
  the `preflight` battery.

## 3–5. Choose the primitive, decompose, batch: batteries

A battery is a named, versioned question set for one decision point, with an owner, a changelog and the consumers
of each answer. The documents pack ships `preflight` (response_type, route, needs_graph), `evidence`
(answerable) and `postflight` (grounded, complete, risk_tier); `rlcd.yaml` `batteries:` adds or replaces them.

```bash
curl -s localhost:8095/v1/systemone -H 'content-type: application/json' -d '{"battery": "preflight@1", "state": {"query": "..."}}'
```

The response carries `battery: {name, version, digest}`. A pinned version that is no longer current is a 422
`battery_version_mismatch` (your thresholds were chosen for the old meaning). The digest covers what can change
an answer (types, heads, criteria, allowed options, input paths), not `instructions`. `GET /v1/batteries` lists
them with problems (a question whose head is missing or untrained); `rlcd check` fails when receipts recorded the
same battery version with another digest — a change shipped without a version bump. It can only compare calls
that left a receipt: with the default `serve.receipts: decisions`, a `/v1/systemone` call leaves one only when it
sends `"receipt": true`, so set `serve.receipts: all` (or send it) for the check to see your battery traffic.

All questions on one state are answered in one call, each by its own head; ask the speculative ones too and let
your code ignore what it does not need.

## 6. Route before generation: allowed options and route contracts

Deterministic filters run before the semantic choice, so a head only chooses among options your policy allows:

* `allowed` on a choice question, `allowed: {question id: [options]}` on `/v1/systemone` (it narrows a battery's
  questions too) or `allowed: {head: [options]}` on `/v1/decide`. The answer lists
  the allowed options with their calibrated probabilities **not renormalised**, plus `excluded_mass` (the
  probability the head put on removed options). The decision commits to the most probable allowed option with its
  own probability, so thresholds keep meaning error rates; a large `excluded_mass` warns that the right answer
  was probably removed.
* Route contracts in `rlcd.yaml`: a choice selects a contract, not a naked name.

```yaml
routes:
  response_type:
    direct_answer:  {llm: small, context_budget_tokens: 2000, max_retries: 1, verify: postflight,
                     fallback: escalate_human}
    summarize:      {llm: large, requires: [llm_large], allowed_data: [public, internal], latency_ms: 4000,
                     verify: postflight, fallback: escalate_human}
    escalate_human: {human: true}
  route:
    graph:  {requires: [graph], fallback: hybrid}
    hybrid: {requires: [vector, graph], fallback: vector}
    vector: {requires: [vector]}
```

A request sends what is true right now: `capabilities` (available stores, models, tools), `sensitivity` (the
data class of this state) and `latency_budget_ms`. Options whose contract needs a missing capability, may not
receive that data class or is too slow are removed before the head chooses; `heads.<h>.considered` gives the
reason for every option, `plan.contracts` the chosen contract and its first usable fallback (a fallback must pass
the same filters: sensitive data never goes to a route it was kept from). When every option is removed the head
commits to nothing and the plan escalates.

`/v1/decide` with `return_evidence: true` returns the kept chunks, so the LLM context is assembled from exactly
the evidence the plan accepted. `baseline` (what your existing path would have done) is kept in the receipt, so
shadow traffic can be compared with the counterfactual.

## 7. Verify after generation: the evaluate decision table

`POST /v1/evaluate` judges a drafted answer with explicit precedence (`decided_by` says which rule decided):

1. malformed state → 422, nothing is judged;
2. `checks` — your deterministic results (`{"schema": true, "urls": {"passed": false, "detail": "..."}}`): any
   failure rejects without asking a head; a favourable semantic answer never overrides an exact check;
3. `artifact: {status: complete | partial | none, finish_reason, model, provider}` — a half answer never ships
   (`next: retry_generation`);
4. `citations` — chunk ids the answer cites that are not in the evidence reject it;
5. output screening — an answer that contains a secret by its shape (a private key block, an API key or token
   format, a JWT, a bearer token, a password in a URL; with `evaluate.output_scan: [secrets, pii]` also an e-mail
   address or a card number) never ships (`decided_by: output_scan`); `evaluate.output_scan: false` turns it off.
   Text like "Password: minimum twelve characters" is not a secret and passes;
6. the heads: grounded (gate), complete (only level 3 ships), risk_tier (sets the stakes when you do not).

Evidence marked `generated: true` caps the verdict at review (support by generated text is not support).
`plan: <decision_id of the /v1/decide plan>` binds the review to the evidence the plan selected: other or changed
chunks cap it at review.

When a review fails, `revise` says which property failed (with its meaning), the evidence ids to stay within and
the attempts left. Send the next draft with `revision: {of: <decision_id>}`; the loop is finite: past
`evaluate.max_revisions` (default 2; with `plan`, the chosen response_type contract's `max_retries`; a revision
inherits the plan of the evaluation it revises), when the same property fails twice, or when the state did not
change at all, the answer is rejected (`decided_by: revision`) and you escalate or narrow the task. A stopped loop
outranks every earlier rule: its decision is `revision`, the earlier rule's reason stays in `warnings`.

## 8. Gate by risk

* Three bands per stakes level: `act` (p ≥ act), `confirm` (p ≥ confirm), the abstain action otherwise; thresholds
  are configuration, returned with every decision, and capped by evidence: no holdout skill → abstain; acting not
  verified on the holdout → confirm; the Bernoulli track record of similar predictions too short → confirm.
* Asymmetric thresholds per head and committed class: `policy.overrides.<head>.<class>.<stakes | "*">`.
* Every head decision says why it is uncertain (`uncertainty.sources`: missing_evidence, options_removed,
  ambiguous, diffuse, below_threshold, no_skill, unverified, insufficient_track_record, untrained), so the fix can
  target the cause: more evidence, a better option set, more labels, or time.
* Probability is not permission: an `act` answer to a read-only search and an `act` answer to a deletion are
  different business decisions; put the second behind your own confirmation.

## 9. Record receipts

Every `/v1/decide` and `/v1/evaluate` (and, with `serve.receipts: all` or `"receipt": true`, `/v1/systemone` and
`/v1/rank`) writes a receipt to `decisions.jsonl`: the state digest, the question contract (battery or heads and
parameters), model, pack, feature version, policy digest, stakes, rollout, full distributions, tiers, the action
and plan or verdict, latency, warnings and what you asked to keep (`baseline`, `artifact`, `checks`, `revision`,
`tags`). `state_digest` and `contract_digest` identify the state as sent and the question contract it was asked
under (for `/v1/evaluate` the contract includes the check results, the artifact status, the citations, the plan
and the evaluation it revises); they are not a cache key for a decide / evaluate result, whose tier also depends
on the rollout, the kill switch, the live Bernoulli table, `explore`, the clock and the corpus — never cache those.

Receipts are hash-chained, and so are confirmations (`confirmations.jsonl`): `rlcd receipts verify` detects an
edited, inserted, deleted or reordered line in either and prints both head digests — anchor them elsewhere to
detect truncation. `GET /v1/receipts/{id}` returns one.

**Before a side effect**, check the approval still holds:

```bash
curl -s localhost:8095/v1/receipts/$ID/check -H 'content-type: application/json' -d '{"state": {...the state you are about to act on...}, "max_age_s": 300}'
```

`current` is false (with `stale_reasons`) when the state changed (a file, policy text, tool argument or answer),
the `rlcd-latest` alias moved, the policy or battery version changed, automation was switched off or capped
since an `act` decision, the receipt is older than `max_age_s`, or the decision is at `confirm` and nobody has
confirmed it yet, or when the decision abstained (it approves no action). A person confirms with
`POST /v1/receipts/{id}/confirm {"by": "..."}` (or `rlcd receipts confirm ID --by NAME`); with `RLCD_ADMIN_KEY` set
that route needs the admin key, so a caller holding only the decision key cannot confirm its own decisions
(without one, anyone who can call the API can; `rlcd check` warns once the log holds confirm decisions). The
confirmation belongs to that receipt and its state: a check counts it only when it names the state the decision
judged and matches its own digest.

Retention: `rlcd receipts prune --before 2026-10-01` drops the logged states of `decisions.jsonl` (also of lines
written before receipts existed) but keeps their digests, so the chain still verifies; feedback can no longer train
from a pruned decision. `feedback.jsonl` and `regressions.jsonl` keep their own copies of states: prune or archive
them yourself.

## 10. Roll out gradually

| stage | open-rlcd |
|---|---|
| offline | `rlcd replay --base A --candidate B`: logged decisions re-run under both versions (as trained: rollout full, kill switch on, each version's own holdout gate), with class and tier transitions, probability-band shifts and accuracy on rows with outcomes; `rlcd regress`: regression cases every version must pass (`rlcd case add --from-decision ID --head H --expect class=X`, or feedback tagged `regression`; kept in `regressions.jsonl` in the project, meant to be committed after review) |
| shadow | `rollout.mode: shadow`: decisions are made, logged and returned, nothing is automated (capped at confirm, marked) — compare with your `baseline` |
| assist | `rollout.mode: assist`: every decision is a recommendation a person confirms |
| limited | `rollout.mode: limited` + `limited: {stakes: [low], heads: [...]}`: act only inside the slice |
| expand | widen the slice, then `full`, as `GET /v1/metrics` shows the evidence |

Per-head modes (`rollout.heads: {needs_graph: shadow}`) put one question behind a flag; `heads.<name>.enabled:
false` in rlcd.yaml stops serving a head altogether (the trained head stays in the bundle, so enabling it again
needs no retrain). Rollout only ever takes
automation away: it never lifts a cap the evidence imposed, and it caps at `confirm`, never at the abstain action,
so abstention keeps meaning "the model cannot vouch for this".

* **Kill switch**: `rlcd automation off --reason "incident 42"` or `POST /v1/automation {"enabled": false}`:
  every decision is capped at confirm at once, while decisions are still made, logged and returned. With
  `RLCD_AUTOMATION=off` in the server's environment the API cannot switch it back on. With `RLCD_ADMIN_KEY` set,
  training, the switch and confirmations need that key, so a decision caller can neither retrain, re-enable
  automation nor confirm its own decisions.
* **Rollback**: `rlcd rollback` (or `rlcd promote VERSION`) points `rlcd-latest` at another version — no retrain,
  no code change; the version's heads gate on their own evidence again. `--by NAME` and `--reason` go into the
  registry history (`--by` defaults to `$USER`, else `unknown`).
* **Measure**: `GET /v1/metrics` / `rlcd metrics`: automatic-action, review and abstention rates, false automation
  and false rejection from feedback, online reliability next to the holdout, uncertainty sources, binding links
  (`binding.latency_only_candidates`: heads that never change an action), task success, cost and retries from
  `/v1/outcome`, latency percentiles, review load against `serve.review_capacity_per_day`. `drift_suspected` is set
  only with at least 30 labelled act decisions at a stakes level whose Wilson 95 % upper bound lies below the
  holdout's act-tier lower bound. Outcomes are only representative of all decisions when you post them for a
  random sample; the report says so.
* **Review queue**: `GET /v1/review` / `rlcd review` lists confirm and abstain decisions without feedback
  (`POST /v1/feedback`; an outcome or a confirmation does not remove an item), with the state, uncertainty and
  reasons a reviewer needs. Feedback with `tags: ["regression"]` turns the corrected
  case into a regression fixture; `missing_evidence` and `note` record what the reviewer found.
* **Before launch**: `rlcd check` (or `GET /v1/check`; `--json` for CI, exit code 1 when an item fails) runs the
  guide's pre-launch checklist mechanically: an `rlcd.yaml` without errors (item `config.valid`, the findings of
  `rlcd config check`; every key is in [configuration.md](configuration.md)), heads trained with skill and
  verified thresholds, no-match options, observable score anchors, security settings, redaction, a verifying
  receipt chain, a rollback target, the rollout and kill-switch state, battery versions, regression cases, review
  capacity and named owners (`owners:` in rlcd.yaml).

## The Python client

```python
from open_rlcd.client import Choice, Client, Noul, Score

rlcd = Client("http://127.0.0.1:8095", fail_closed="abstain")
r = rlcd.system_one(state={"query": "Our checkout is down"}, questions={
    "kind": Choice(head="response_type", allowed=["direct_answer", "escalate_human"]),
    "multi_hop": Noul(head="needs_graph")})
if r.answers["kind"].band() == "high":       # "low" whenever the head is untrained or shows no skill
    ...
d = rlcd.decide(state={"query": "...", "candidates": [...]}, capabilities=["vector"])
if d.allow:                                   # from the action only: act -> allow, confirm -> ask, else deny
    ...
```

Errors are separate states (`AuthError`, `ValidationError` with its `kind`, `Unavailable`, `TransportError`,
`MalformedResponse`, `MissingAnswer`); only transport failures and responses marked `retryable` are retried, with
bounded backoff; an answer is never invented. With `fail_closed`, a decide / evaluate that cannot be answered
returns a synthetic decision marked `fallback: true` that abstains (or asks) — never one that acts.

## What stays your job

* identity, capabilities, permissions, user confirmation, executing with least privilege;
* the LLM calls themselves, their prompts, and what your route contracts mean (open-rlcd stores and returns your
  contract fields, it does not call providers);
* deterministic validators — open-rlcd records their results and lets them win, it does not run them;
* staffing the review queue for the volume your thresholds create.

## Honest limits

* Receipts and confirmations are tamper-evident, not tamper-proof: anyone who can write `RLCD_HOME` can rewrite a
  whole chain, or append a confirmation that verifies; anchor both head digests outside it, and set
  `RLCD_ADMIN_KEY` so confirmations arrive through the API from whoever holds that key.
* Redaction is pattern-based: a bare high-entropy string with no recognisable shape is kept.
* Removing options does not retrain anything: when the head's mass sits on removed options, no allowed option is
  likely to be right, and `excluded_mass` says so.
* Shadow, assist, limited and the kill switch cap automation; they cannot make a head more accurate.
