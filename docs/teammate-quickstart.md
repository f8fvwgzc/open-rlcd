# Teammate quickstart: zero to a calibrated model on your documents

One page, one afternoon, one laptop CPU. You end with a model that routes requests, picks the chunks that
matter, says whether the evidence is enough, judges drafted answers — and an honest report of how often its
probabilities came true on rows it never saw.

## 0. Install (5 minutes)

```bash
brew install uv          # Python tooling; Python 3.12 is fetched by uv
brew install libomp      # only needed for LightGBM heads (the documents pack uses logistic regression)
uv tool install "open-rlcd[documents,labeling] @ git+https://github.com/f8fvwgzc/open-rlcd"
rlcd --version
```

Or from a clone: `git clone … && cd open-rlcd && uv sync --extra documents --extra labeling`, then prefix every
command below with `uv run`.

For the LLM judge you need one of: [Claude Code](https://claude.com/claude-code) logged in (`claude` on your PATH,
uses your subscription), an `ANTHROPIC_API_KEY` in your environment, or a local model in
[Ollama](https://ollama.com).

## 1. Create a project and ingest your documents (10 minutes)

```bash
rlcd init acme --pack documents && cd acme
cp -r ~/path/to/your/docs ./data          # .md .txt .html .pdf .json/.jsonl, any folder structure
rlcd ingest ./data
```

The first ingest downloads the encoder once (intfloat/multilingual-e5-small, int8 ONNX, ~135 MB, MIT license)
into `var/encoders/`. It handles English, Japanese and Mongolian (and ~100 more languages). Everything else stays
in `var/` on your machine: chunks, the vector index, the entity graph, embeddings cache. `var/` is git-ignored.

No network? Set `pack_config.encoder: {kind: hashing, dim: 256}` in `rlcd.yaml`: no download, weaker features.

## 2. Collect real questions (15 minutes)

Write `queries.jsonl` — the questions people actually ask (support tickets, chat logs, search logs), one per line:

```json
{"query": "What is the return window for the K-200?", "time": "2026-09-14T10:22:00Z", "group": "ticket-8812"}
{"query": "K-200の保証期間は？", "time": "2026-09-15T08:01:00Z", "group": "ticket-8840"}
```

`group` keeps related rows (one ticket, one customer, one session) on the same side of the holdout; `time` lets
the holdout be the most recent period. A few thousand queries is plenty; a few hundred works.

## 3. Weak labels from an LLM judge (20–60 minutes, unattended)

```bash
export RLCD_ALLOW_REMOTE=1                                   # claude-cli and anthropic send item text to Anthropic
rlcd label bootstrap --judge claude-cli:claude-haiku-5-5     # or: anthropic | ollama:qwen2.5:7b-instruct
```

`claude-cli` and `anthropic` are refused without that opt-in (or `security: {allow_remote: true}` in rlcd.yaml),
because the queries and document chunks being labelled leave your machine. `ollama:<model>` needs no opt-in:
it runs locally.

The judge labels every head for every query: response type, route, multi-hop, which retrieved chunks are
relevant, whether the evidence is enough. It is resumable (rerun it after an interruption) and never touches
human labels. Judge labels only **train** the models; they never calibrate them.

## 4. Check ~300 rows yourself (60–90 minutes) — the important part

```bash
rlcd label audit --n 300
```

You see the item, the question, the judge's answer and its confidence. **Enter** accepts the judge's label, a
number or name picks another, `s` skips, `b` goes back, `q` quits (resume any time). These human labels are what
the calibrators are fitted on and what every published number is measured on. At the end you see how often the
judge agreed with you.

## 5. Train and read the report (5 minutes)

```bash
rlcd train
rlcd report          # MODEL_CARD.md + an HTML reliability page opens in your browser
```

For each head: holdout log loss and Brier next to the base rate, Brier skill, ECE, a block-bootstrap skill
test, and a verdict per stakes level:

* **skill: NO** → the head abstains (`escalate`) on every request and says why. That is the system working: it
  found no reliable signal in your data. Add trusted labels, improve the query log, or drop the head.
* **verdict `confirm`** → useful, but the evidence for acting automatically isn't there yet (too few holdout
  rows, a conformal promise it can't make, an act tier that came true less often than predicted).
* **verdict `act`** → the holdout backs automatic action at that stakes level.

On the reliability diagrams, calibrated means the dots sit on the diagonal.

## 6. Tighten it (repeat as often as you like)

```bash
rlcd label next --head answerable      # the most informative rows to check next
rlcd label audit --queue
rlcd train && rlcd report
```

Each round, watch the reliability curve and the verdicts. Rows picked this way train and calibrate, but are kept
out of the holdout (they are not a representative sample).

## 7. Serve it

```bash
rlcd serve --port 8095
```

```bash
curl -s localhost:8095/v1/decide -H 'content-type: application/json' -d '{
  "state": {"query": "What is the return window for the K-200?"}, "stakes": "medium"}'
```

The plan says what to do before you spend an LLM call: `response_type`, `route`, which chunks to use
(`use_chunks`), whether to call the LLM (`call_llm`) and what to do otherwise (`next`: ask a clarifying question,
escalate, refuse, retrieve more). Bring your own retriever? Pass `candidates: [{id, text, score?}]` in `state` —
and send the same fields you trained with (the response warns, and caps the decision at `confirm`, when a request
lacks inputs the model was calibrated on). Send `now` (the request time) if your label log had times; without it
`/v1/decide` uses the current time. Each answer carries `skill`: never act on `confidence` alone when it is false. After the LLM writes, `POST /v1/evaluate` with `{query, evidence, answer}` says ship, review or
reject. Post real outcomes to `/v1/feedback` with the `decision_id`: the Bernoulli posteriors update at once and
the rows train the next model.

Protect it when it leaves your laptop: `export RLCD_API_KEY=…` and send `Authorization: Bearer …`.

## 8. Measure and share

```bash
rlcd bench                   # p50 / p95 / p99 per stage on your machine (new vs repeated queries)
rlcd export ./acme-bundle    # open weights: JSON + safetensors, no pickles, with MODEL_CARD.md
```

A bundle contains no training data, but its heads learned from your data: share it only if that is allowed.

## Troubleshooting

| message | meaning / fix |
|---|---|
| `LightGBM cannot be imported … libomp` | `brew install libomp`, or set `model: logreg` for that head |
| `… trusted rows; at least 30 are needed` | audit more rows (`rlcd label audit`) |
| `feature_version mismatch` | the bundle was trained with another encoder or feature set: retrain, or switch the encoder back |
| `the request lacks … input feature(s)` | your request omits fields the model was trained with (e.g. candidate scores): send them |
| `encoder unavailable … HF_HUB_OFFLINE` | run once online so the encoder downloads, or use the hashing encoder |
| a head always says `escalate` | read its `skill` and `verdict` in `rlcd report`: that is the honest answer for now |

## Next: put it in front of (and behind) your LLM

Read [control-layer.md](control-layer.md). The short version: run `rlcd check` before launch; start with
`rollout: {mode: shadow}` in rlcd.yaml and send your existing route as `baseline`; call
`POST /v1/receipts/{id}/check` right before any side effect; post what happened to `/v1/feedback` (per head) and
`/v1/outcome` (task success, cost); set `serve.review_capacity_per_day` and watch `rlcd metrics` and `rlcd review`; move to `assist`, then `limited`, then `full` as the
evidence shows it is safe. `rlcd automation off --reason ...` stops all automation at once if something goes wrong.
