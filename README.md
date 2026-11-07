# Open RLCD

**A small decision model that answers typed questions with calibrated probabilities — and says so when it
can't.** Train it on a laptop CPU in an afternoon, on your own data, and put it in front of (and behind) an LLM
as the fast, measurable router, gatekeeper and evaluator.

Open RLCD (Reinforcement Learning Calibrated Decision) answers three kinds of question:

| type | answers | example |
|---|---|---|
| `noul` | yes / no, as P(yes) | Is the retrieved evidence enough to answer? |
| `choice` | one of N options | Which response does this request need: direct answer, extraction, summary, …? |
| `score` | an ordered level 0..K | How risky is this drafted answer, 0–3? |

…and turns the answers into **`act` / `confirm` / `escalate`** through an explicit policy with
stakes-dependent thresholds. It never writes prose: the expensive LLM only writes text after the cheap model has
decided *what*, *whether* and *from which evidence*.

```bash
git clone https://github.com/f8fvwgzc/open-rlcd && cd open-rlcd
uv sync                      # core: CPU only, no torch
uv run rlcd serve            # serves the bundled demo model (synthetic data) on 127.0.0.1:8095
```

```bash
curl -s localhost:8095/v1/systemone -H 'content-type: application/json' -d '{
  "state": {"query": "Who leads the team that owns the K-200 scanner?"},
  "questions": {"kind": {"type": "choice", "head": "response_type"},
                "multi_hop": {"type": "noul", "head": "needs_graph"}}}'
```

Start here:

| you have | read |
|---|---|
| a folder of documents and questions people ask about them | [docs/teammate-quickstart.md](docs/teammate-quickstart.md) |
| a table of past cases with an outcome column (churn, fraud, approvals, leads) | [docs/tabular-quickstart.md](docs/tabular-quickstart.md) |
| another kind of decision (ticket triage, text → category, …) | [docs/writing-a-pack.md](docs/writing-a-pack.md) and [examples/custom-pack](examples/custom-pack) |
| a company to roll it out in: who labels, who sets thresholds, personal data, rollout criteria | [docs/adoption.md](docs/adoption.md) |
| a server to run it on: Docker, HTTPS, backups, upgrades, monitoring | [docs/deployment.md](docs/deployment.md) |
| an `rlcd.yaml` to write | [docs/configuration.md](docs/configuration.md) — every key; `rlcd config check` validates the file |

## What it promises — and how it checks

* **A probability of 0.8 comes true about 80 % of the time on data the model never saw.** Every head is scored
  once on an untouched holdout and the result is published at `/v1/calibration`: log loss and Brier next to the
  base-rate forecaster, Brier skill score, reliability bins, ECE, and a block-bootstrap skill test.
* **No skill, no decision.** If the holdout shows no reliable gain over the base rate, the head abstains
  (`escalate`) on every request and says why. Noise is never dressed up as a signal.
* **Probabilities stay frequencies.** No class weights, no resampling.
* **Weak labels train, humans calibrate.** An LLM judge can label thousands of rows cheaply; the calibrators are
  fitted, and every published number is measured, on human-verified (or ground-truth) rows only — and the
  judge's own agreement with humans is reported next to them.
* **Promised error rates are tested.** Gate heads (`answerable`, `grounded`) can promise "when I say yes, I'm
  wrong at most 5 % of the time" via split-conformal abstention; the promise is checked on the holdout and
  published. If it can't be made with the data you have, the head is capped at `confirm` and says so.
* **Every answer carries** `head`, `calibrated`, `confidence` and `skill`; every decision also carries `reasons`
  (exact feature attributions for linear heads, TreeSHAP for LightGBM), the thresholds it used and its warnings.

## What it is not

* **Not an LLM.** No text generation, no chat. Free-form `instructions` are accepted and ignored.
* **No nanosecond magic.** Heads take microseconds to well under a millisecond per row on a CPU; the text encoder
  is the dominant cost (single-digit to tens of milliseconds per query on CPU with ONNX int8); ANN search over
  millions of vectors in a dedicated vector database is typically low milliseconds. `rlcd bench` prints the real
  p50/p95/p99 on your machine — see the measured numbers below.
* **Not better than its labels.** LLM-judge labels carry the judge's mistakes into the base model; that is why
  calibration uses human labels only. With few human labels the holdout is small and the intervals are wide —
  read them.
* **Not a guarantee under drift.** Calibration holds for data like the training data. New document types, a new
  product line, a new language: audit fresh rows and retrain.
* **Synthetic numbers are a sanity check.** The demo bundle and the test suite run on synthetic data; their
  metrics show that the pipeline is honest, not how good a model will be on your traffic.

## Around an LLM: the control layer

open-rlcd follows the architecture of the field guide *How to use Jev with LLMs* (TypeSafe's System One): code
owns authority, the decision model answers narrow typed questions, the LLM generates, and measured outcomes earn
autonomy. [docs/control-layer.md](docs/control-layer.md) maps its ten-step playbook to open-rlcd:

* **named state, fail closed** — questions read paths of one shared state (`inputs`); a missing field is a 422,
  never a guess (`strict` also refuses missing training inputs); logged states are redacted;
* **versioned batteries** — `{"battery": "preflight@1"}` instead of a question list, with owner, changelog, digest;
* **route before generation** — `allowed` options and route contracts (`routes:` in rlcd.yaml, filtered by the
  request's `capabilities`, `sensitivity`, `latency_budget_ms`) run before the semantic choice; a choice returns a
  contract and its fallback;
* **verify after generation** — evaluate's decision table: your deterministic `checks` and the `artifact` status
  win over any head, `citations` and the plan's evidence are bound, and a revision loop stops after a budget;
* **receipts** — every decision is a hash-chained receipt (state, question and policy digests, distributions,
  route, action); `POST /v1/receipts/{id}/check` says whether an approval still holds right before a side effect;
* **earn autonomy** — rollout modes shadow / assist / limited / full, a run-time kill switch, `rlcd rollback`,
  replay and regression cases, `/v1/metrics` (false automation, online reliability, review load), a review queue,
  `rlcd check` for the pre-launch checklist, and a Python client (`open_rlcd.client`).

## The enterprise-documents pack

For each incoming request in front of a document store — the local corpus `rlcd ingest` builds (a FAISS / numpy
vector index and a NetworkX entity graph), the same corpus with its vectors in Qdrant or pgvector and its graph in
Neo4j, or a Qdrant collection, pgvector table or Neo4j graph you already run (read-only attach mode, no copy of
your corpus) — or in front of your own retriever (pass the candidates in the request):

| head | type | decides |
|---|---|---|
| `response_type` | choice | direct_answer, extract_fields, summarize, compare, list, clarify_question, escalate_human, out_of_scope |
| `route` | choice | vector, graph, hybrid, none |
| `needs_graph` | noul | is multi-hop graph traversal needed |
| `chunk_relevant` | noul | which retrieved chunks matter (`/v1/rank` re-ranks the ANN candidates) |
| `answerable` | noul, gate | is the evidence enough to answer — otherwise ask, escalate or refuse |
| `grounded` | noul, gate | evaluator: is every claim of a drafted answer supported by the evidence |
| `complete` | score 0–3 | evaluator: how much of the question the answer covers |
| `risk_tier` | score 0–3 | evaluator: how much damage a wrong answer could do |

`POST /v1/decide` returns a plan (`response_type`, `route`, `use_chunks`, `call_llm`, `next`) with one overall
`act / confirm / escalate` that is never stronger than its weakest gate; `POST /v1/evaluate` judges a drafted
answer: ship (`act`), human review (`confirm`) or reject (`escalate`).

Features: a frozen open multilingual encoder (default **intfloat/multilingual-e5-small**, MIT, ~100 languages
including English, Japanese and Mongolian) through ONNX Runtime int8 on CPU, embeddings cached by content
hash; lexical signals (BM25, term coverage, number support); retrieval signals (ANN scores, rank gaps, score
entropy, source agreement, freshness, authority, graph path length and degree). Heads are logistic regression or
LightGBM on top: they train in seconds to minutes. Never fine-tunes a transformer by default.

Other packs: **tabular** (CSV / Parquet: churn, fraud, lead scoring, approvals; categorical values travel with
the model — set `hash_categories: true` for columns that could hold personal data; free-text columns go under
`text:` and are hashed, so no text is stored in the bundle; columns only known after the decision go under
`exclude:`). A thin **trading** adapter talks to the View Engine quant sidecar's `/dataset` and `/features` over
HTTP (no trading logic inside; analysis tool, not financial advice). Writing your own:
[docs/writing-a-pack.md](docs/writing-a-pack.md), with a complete ticket-triage pack in
[examples/custom-pack](examples/custom-pack).

## Train on your own data

```bash
uv sync --extra documents --extra labeling
uv run rlcd init my-business --pack documents && cd my-business
uv run rlcd config check                           # after every edit of rlcd.yaml: typos and bad values, by key
uv run rlcd ingest ./data                          # your folder of .md / .txt / .html / .pdf / .json
# write queries.jsonl: real questions people ask, one {"query": ..., "time": ..., "group": ..., "answer": ...,
# "evidence": [...]} per line: the answer your LLM drafted and what it was shown (grounded needs the answer)
export RLCD_ALLOW_REMOTE=1                         # claude-cli / anthropic (item text goes to Anthropic) or a judge off loopback
uv run rlcd label bootstrap --judge claude-cli     # or: anthropic | ollama:<model> (stays local) | openai:<url>|<model>
uv run rlcd label audit --heads answerable,grounded --n 600   # your labels: these calibrate the model
uv run rlcd train                                  # all heads; prints the calibration summary
uv run rlcd report                                 # MODEL_CARD.md + an HTML reliability page
uv run rlcd label next --head answerable           # what to audit next; retrain, watch the curve tighten
uv run rlcd serve --port 8095
uv run rlcd bench
uv run rlcd export ./bundle                        # open weights: JSON + safetensors + LightGBM text
```

Millions of documents already in a vector database? Attach it instead of ingesting (`uv sync --extra qdrant`,
`--extra pgvector` or `--extra neo4j`):

```yaml
pack_config:
  encoder: {kind: onnx, model_id: intfloat/multilingual-e5-small}   # must be the model the collection was embedded with
  vector: {kind: qdrant, mode: attach, url: http://localhost:6333, collection: kb, api_key_env: QDRANT_API_KEY,
           text_field: body, doc_id_field: source_id, title_field: title, updated_at_field: modified}
  # or: {kind: pgvector, mode: attach, dsn_env: KB_DSN, table: public.kb_chunks, vector_field: embedding, ...}
  # graph: {kind: neo4j, mode: attach, uri: bolt://localhost:7687, password_env: NEO4J_PASSWORD,
  #         expand_cypher: "<read-only Cypher: $keys -> chunk ids with path_length>"}
```

Attach mode never writes to your store. Query vectors come from open-rlcd's encoder, so the collection must have
been embedded with the same model (a different vector size is refused; a different model of the same size cannot
be detected): otherwise pass `candidates` in the request. To let open-rlcd write its own chunks to a database
instead of the local index: `rlcd ingest ./data --qdrant http://localhost:6333 --collection docs --neo4j
bolt://localhost:7687` (index mode). Every connector key is in [docs/configuration.md](docs/configuration.md),
and `rlcd config check` checks them; a walk-through is in [docs/teammate-quickstart.md](docs/teammate-quickstart.md).

How many human labels? A head needs about 45 trusted rows to train (at least 30, enough that 10 land in its 20 %
holdout, and enough for the calibration folds). `act` is verified only with 30 holdout rows the policy would act on,
so usually **several hundred trusted rows per head**: about 300 if half the holdout clears the act threshold, 450 if
a third does ([docs/adoption.md](docs/adoption.md), section 4). In between, decisions are calibrated but capped at
`confirm`, and the report says exactly what is missing. Audit the heads that gate actions first.

Production outcomes posted to `/v1/feedback` update the Beta-Bernoulli posteriors at once and become
up-weighted training rows at the next train. That is the whole "reinforcement" — described as modestly as it is.

A holdout verdict of `act` is permission, not a promise: a live decision acts only where the evidence for
predictions like it (holdout rows plus posted outcomes) is strong enough, and is `confirm` until then.
`rlcd train` and `rlcd check` print, per head, which probability bands already act and how many correct
outcomes the others still need.

## What the synthetic demo shows (and doesn't)

`rlcd demo --encoder onnx` builds a fictional company corpus in English, Japanese and Mongolian with 900 queries
and ground truth for every head, trains all eight heads and reports on template groups the heads never saw:

| head | holdout rows | log loss (base rate) | Brier skill | ECE | verdict at medium stakes |
|---|---|---|---|---|---|
| response_type | 182 | 0.27 (1.91) | 0.84 | 0.06 | act |
| route | 182 | 0.24 (1.23) | 0.81 | 0.05 | act |
| needs_graph | 182 | 0.16 (0.43) | 0.60 | 0.03 | **escalate**: the gain is not reliably positive across 56 template groups |
| chunk_relevant | 747 | 0.21 (0.62) | 0.70 | 0.03 | act |
| answerable | 277 | 0.39 (0.70) | 0.51 | 0.03 | confirm: the 5 % conformal promise cannot be made yet |
| grounded | 322 | 0.15 (0.69) | 0.84 | 0.05 | confirm |
| complete | 322 | 0.48 (0.88) | 0.41 | 0.05 | act |
| risk_tier | 322 | 0.23 (1.23) | 0.81 | 0.03 | act |

These are synthetic numbers: they show that the pipeline is honest (a head with a Brier skill of 0.60 still
abstains when the evidence for its skill is not solid), not how good a model will be on your documents. The
bundle shipped in `bundles/demo-documents` is the same pipeline with the offline hashing encoder, so it works
without a download; its heads are weaker.

## Measured latency (one machine, not a promise)

`rlcd bench` on an Apple M2 Pro (10 cores), Python 3.12, single requests, the synthetic documents demo trained
with multilingual-e5-small (int8 ONNX):

| stage | p50 | p95 |
|---|---|---|
| one head on one prepared feature row (logistic regression, 5 calibrated members) | 0.06–0.20 ms | 0.07–0.22 ms |
| build a feature row (query, chunk, evidence or answer view), embeddings cached | 0.01–0.02 ms | 0.01–0.03 ms |
| embed one query (e5-small int8 ONNX, CPU) | 3.7 ms | 5.6 ms |
| embed 32 passages | 325 ms | 430 ms |
| top-10 search in the local index (50 chunks) / 2-hop entity expansion | 0.015 ms / 0.26 ms | 0.016 ms / 0.27 ms |
| whole `/v1/decide` for a **new** query (query and candidate chunks embedded) | 52 ms | 65 ms |
| whole `/v1/decide` for a repeated query (caches warm) | 3.6 ms | 4.1 ms |

The encoder dominates; the decision heads are a rounding error. Search over millions of vectors depends on your
vector database and index settings — measure it with your own store.

## API

`POST /v1/systemone` (System One-compatible: a View Engine RLCD client works unchanged), `/v1/rank`,
`/v1/decide`, `/v1/evaluate`, `/v1/feedback`, `/v1/outcome`, `/v1/train` (+ `GET /v1/train/{job}`),
`/v1/receipts/{id}/check`, `/v1/automation`; `GET /health`, `/ready`, `/v1/models`, `/v1/heads`, `/v1/batteries`,
`/v1/calibration`, `/v1/bernoulli`, `/v1/receipts/{id}`, `/v1/metrics`, `/v1/review`. Full contract and the
extensions table: [docs/contract.md](docs/contract.md).

## Open-weight bundles

A trained model is a directory: `model.json` (version, heads, classes, feature version, encoder id + sha256,
thresholds, data card), LightGBM native text models, calibrators as JSON, linear weights as `safetensors`, and a
generated `MODEL_CARD.md`. **No pickle or joblib anywhere: loading a bundle never executes code.** The repository
ships one demo bundle trained only on synthetic data (`bundles/demo-documents`); bundles trained on your data
stay on your machine unless you choose to `rlcd export` or `rlcd push --hf` them.

## Security

`RLCD_API_KEY` protects every route but `/health` and `/ready`; with `RLCD_ADMIN_KEY` set, training, the kill
switch and confirming a decision need that key instead (without it, a decision caller can confirm its own
confirm decisions). `rlcd serve` refuses to listen on a non-loopback address without `RLCD_API_KEY`.
Bundles load only from `RLCD_HOME` and contain no code. Logged states, notes and reasons are redacted of
secrets (`serve.redact`), and `rlcd receipts prune` drops old states while keeping the receipt chain
verifiable. Connectors and judges
refuse non-loopback hosts unless you allow them (`security.allow_remote: true` or `RLCD_ALLOW_REMOTE=1`). That
includes the `claude-cli` and `anthropic` judges, which send the text of the items being labelled to Anthropic:
`rlcd label bootstrap` refuses them until you opt in. Choose a local judge (`ollama:<model>`) for data that must
not leave your machines. Secrets come from environment variables only: connector configs name the
variable (`api_key_env`, `password_env`, `dsn_env`), and a password, key or DSN written into rlcd.yaml
is refused.

## Run it on a server

```bash
RLCD_API_KEY=... RLCD_ADMIN_KEY=... docker compose up -d --build   # HTTPS through Caddy; the API port is not published
```

One serving process per `RLCD_HOME`; the server and the admin commands (`rlcd import`, `promote`, `rollback`,
`train`) can share a home safely. [docs/deployment.md](docs/deployment.md) covers storage, backup and restore,
moving a model from a laptop to the server, upgrades and monitoring.

## Examples

* [examples/documents](examples/documents): build the synthetic company corpus, train, decide and evaluate
  (optionally retrieve from your own folder of documents).
* [examples/tabular](examples/tabular): the UCI Bank Marketing dataset (CC BY 4.0, downloaded at run time),
  trained and reported.
* [examples/custom-pack](examples/custom-pack): a pack of your own in about 110 lines — support-ticket triage
  (text → queue, urgent yes/no) in English, Japanese and Mongolian, trained, served and exported.
* [examples/control-layer](examples/control-layer/run.py): one request through the whole control layer, offline,
  with the demo bundle, the Python client and a stand-in LLM (`uv run python examples/control-layer/run.py`).

## Design and provenance

[docs/design.md](docs/design.md) explains what was reused from the reference RLCD service, what was generalised and
what is new. MIT licensed; see [NOTICE](NOTICE) for credits.
