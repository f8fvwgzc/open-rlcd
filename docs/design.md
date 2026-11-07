# Open RLCD — design

Open RLCD extracts RLCD (Reinforcement Learning Calibrated Decision, the decision service of View Engine) into
a standalone, domain-agnostic toolkit. A trained model answers **typed questions** (noul = yes/no, choice = one
of N, score = ordered levels) with **calibrated probabilities**, turns them into `act / confirm / escalate`
through an explicit policy, and publishes how well its probabilities held up on rows it never saw. It never
writes prose: it is the cheap, measurable router / gatekeeper / evaluator around an LLM.

This document says what is reused from the reference as is, what is generalised, and what is new, and fixes
the interfaces the packages build against.

## 1. Reference: what RLCD is today

`ViewEngine/backend/rlcd` (MIT, same author), about 4.5k lines:

| module | what it does |
|---|---|
| `contract.py` | System One wire contract: `{model, state, questions}` -> `{model, answers, usage}`; noul / choice / score; limits (255 options, 2-10 levels, 64 questions, 200 candidates); `ContractError` -> 422 `{error, kind, ...}`; additive extensions (`head`, `calibrated`, noul `confidence`, `warnings`). |
| `confidence.py` | The three confidence formulas: noul `abs(2p-1)`, choice `(p_max-1/n)/(1-1/n)`, score `max(0, 1 - sum p_i abs(i-m) / MAD_unif)`. |
| `metrics.py` | Holdout metrics: Brier, log loss, ECE, reliability bins, baseline = base-rate forecaster, Brier skill score, block skill test (one-sided t bound on per-block Brier gain), Wilson interval. |
| `bernoulli.py` | Each decision as a Bernoulli trial: Beta-Bernoulli posterior per (head, side, probability bin), exact binomial test, sample size, Thompson draw, Kelly sizing, a durable JSON ledger updated by `/v1/feedback`. |
| `policy.py` | Trading policy: breakeven, expected R, `act / confirm / hold` gates, threshold tuning on validation rows, `verify` (holdout evidence caps the strongest tier), composite score. |
| `heads/` | `HeadSpec`; intent heads (TF-IDF + logistic regression in `CalibratedClassifierCV(sigmoid, cv=5)`); setup and pattern heads (LightGBM in `CalibratedClassifierCV(ensemble=True)` over time-ordered purged folds, TreeSHAP reasons). |
| `training.py` | Time-ordered split: last 20 % of time = holdout, label-window purge, model A (tune) / model B (served, scored once on holdout), per-segment and per-year slices, feedback rows up-weighted, background jobs. |
| `store.py` | `RLCD_HOME`: `registry.json` (rlcd-latest + rlcd-0.1.N), joblib artifacts, `feedback.jsonl`, `decisions.jsonl`. |
| `api.py` | FastAPI: `/v1/systemone`, `/rank`, `/decide`, `/scan`, `/feedback`, `/train`, `/calibration`, `/bernoulli`, `/health`, `/models`, `/heads`; API key; sidecar errors 503/422. |
| `quant.py`, `seeds/` | Trading-only: quant sidecar client, synthetic intent phrasings. **Not carried over.** |

## 2. Reused as is

* **Wire contract** (`core/contract.py`): request bodies, question validation, limits, answer shapes, error
  body, every additive extension. A View Engine `/v1/systemone` client works unchanged: same request, same
  answer objects (`type`, `noul|choice|score`, `probabilities`, `legend`, `confidence`, `head`, `calibrated`),
  same `usage` (token counts always 0), same `warnings`, same 422 kinds (`validation`, `unknown_head`,
  `unknown_model`), same `rlcd-latest` alias and `rlcd-0.1.N` versions. Contract tests are ported.
* **Confidence formulas** (`core/confidence.py`), verbatim.
* **Metrics** (`core/metrics.py`): Brier, log loss, ECE, reliability tables, base-rate baseline, Brier skill
  score, Wilson, `calibration_report`, `summary`.
* **Bernoulli maths** (`core/bernoulli.py`): posterior, binomial test, trades (here: decisions) needed,
  Thompson draw, the JSON ledger with "each decision_id counts once" and "the ledger restarts from the new
  holdout at every train".
* **Evidence-capped policy idea**: no skill over the base rate -> the head abstains and says why; an edge that
  is not statistically established -> never more than `confirm`.
* **Training discipline**: no class weights, no resampling, holdout never used for fitting / calibrating /
  tuning, served model = the model that was scored, feedback rows up-weighted in training only, one job at a
  time, failures of one head never cost the others.
* **API surface**: same endpoints and error conventions; `RLCD_API_KEY`; models loaded only from `RLCD_HOME`.

## 3. Generalised

| reference | open-rlcd |
|---|---|
| hard-coded heads (intent, setup, patterns) | heads declared by a **domain pack** (YAML + small Python): name, type, classes / levels, feature view, model, decision mode, stakes, conformal target. |
| `CalibratedClassifierCV` pickled with joblib | Our own calibrated ensemble: one base model per fold, each with a calibrator fitted on the fold's calibration block. Calibrators are JSON (sigmoid a,b / isotonic knots / temperature T), linear weights are `safetensors`, LightGBM boosters are native text. **Nothing in a bundle is a pickle.** |
| time-ordered folds only | `core/splits.py`: **time-ordered** (expanding window, gap, label-window purge) or **group-ordered** (by document / customer / ticket; a group never straddles train / calibration / holdout). Groups are ordered by their first timestamp when there is one, so the holdout is the latest groups. |
| t-interval skill test over blocks | **Block bootstrap** of the per-row Brier gain over the base rate (contiguous time blocks or whole groups resampled), one-sided 95 % lower bound > 0 required. Pure noise -> `has_skill: false` -> abstain. |
| trading policy (breakeven, expected R, Kelly) | Stakes-tiered policy: per stakes level (`low / medium / high`) thresholds on the calibrated probability of the chosen class: `act`, `confirm`, otherwise `escalate`. Two decision modes: `argmax` (choice, score, symmetric noul) and `gate` (a noul whose `true` means "proceed", e.g. `answerable`, `grounded`). Thresholds are config and returned with every decision. |
| `verify` on trade win rates | `verify` on the holdout: no skill -> escalate; fewer than `min_verify_rows` trusted holdout rows -> confirm; act-tier accuracy significantly below its mean predicted probability (one-sided binomial p < 0.05) -> confirm; else act. |
| Bernoulli bins per (side, bin) | per (head, chosen class, probability bin); gate on `act`: posterior P(true accuracy of this bin > `min_accuracy` of the stakes level) >= 0.9. Thompson sampling opt-in (`explore: true`), labelled. |
| feedback as feature vectors | feedback stores the raw `state` (re-featurised at the next train, so it survives a feature-version change) or `{x, feature_version}`. |
| one artifact per head in `models/<version>/` | a **bundle directory** per version (section 7); `rlcd export` / `rlcd import`; optional `rlcd push --hf`. |

## 4. New

* **Weak labels + human calibration.** Every training row carries `source` (`human`, `ground_truth`,
  `judge`, `feedback`, `synthetic`) and a `trusted` flag (human-verified or ground truth). Base models fit on
  weak + trusted rows; **calibrators are fitted, and holdout metrics reported, on trusted rows only.** The
  judge's agreement with humans (accuracy, Cohen's kappa, confusion) is reported next to the head's own metrics.
* **Split-conformal abstention** (`core/calibration.py`): for gate heads with `conformal: {target_error: a}`,
  a threshold on P(true) chosen by fixed-sequence testing (Learn-then-Test, exact binomial test at `delta = 0.1`)
  on cross-fitted out-of-fold trusted development scores: the fold's base model never trained on the row, and a
  calibrator fitted on the other half of its calibration block never saw it. With probability >= 1 - delta the
  error rate among rows that pass is <= a, under exchangeability. The threshold is then checked on the untouched
  holdout and both numbers are published. If no threshold qualifies the head is capped at `confirm` and says so.
  Choice and score heads report the split-conformal prediction-set quantile (the ceil((n+1)(1-alpha))-th smallest
  score); measured fresh coverage at alpha = 0.1, 4 classes: 0.92 at n = 300, 0.90 at n = 1000-2000.
* **Feature layer** (`features/`): a frozen open multilingual encoder (default `intfloat/multilingual-e5-small`,
  MIT, about 100 languages including English, Japanese and Mongolian) run through ONNX Runtime int8 on CPU,
  embeddings cached on disk keyed by content hash (sqlite, no pickle); lexical signals (BM25, TF-IDF cosine,
  term coverage, numbers / entities overlap); retrieval signals (ANN scores, rank gaps, score entropy, source
  agreement, freshness, authority, graph path length / degree, query-term coverage of the evidence). An offline
  `hashing` encoder (signed char n-gram feature hashing) needs no download; tests and the demo bundle use it.
* **Packs**: `documents` (RAG router / gatekeeper / evaluator), `tabular` (CSV / Parquet: churn, fraud, lead
  scoring, approvals), `trading` (M8, its own pull request: a thin HTTP adapter to the View Engine quant
  sidecar's `/dataset` and `/features`; no trading logic copied).
* **Connectors**: local FAISS -> hnswlib -> exact numpy vector index and a local NetworkX entity graph today;
  Qdrant, pgvector and Neo4j (optional extras) are milestone M7. Remote hosts are refused unless explicitly
  allowed.
* **Labeling**: `rlcd label bootstrap` (LLM judge: Claude via the user's own `claude` CLI subscription or
  `ANTHROPIC_API_KEY`, or a local model via Ollama / any OpenAI-compatible server), `rlcd label audit`
  (terminal review), `rlcd label next` (uncertainty + judge/model disagreement + diversity sampling).
* **CLI** `rlcd`, **benchmark** `rlcd bench` (p50 / p95 / p99 per stage on the user's machine), **reports**
  (`MODEL_CARD.md`, an HTML reliability page with inline SVG, no external assets).

## 5. Package layout and responsibilities

```
open_rlcd/
  __init__.py            VERSION, MODEL_PREFIX = "rlcd-0.1.", LATEST = "rlcd-latest"
  core/
    contract.py          wire types, validation, limits, answers (reference-compatible)
    confidence.py        the three confidence formulas
    calibration.py       Sigmoid / Isotonic / Temperature calibrators (JSON), method choice, conformal
    metrics.py           Brier, log loss, ECE, reliability, baseline, block-bootstrap skill test, kappa
    splits.py            time / group ordering, holdout with purge, calibration folds
    head.py              HeadSpec, base models (logreg, lightgbm), CalibratedEnsemble, explanations, save/load
    dataset.py           Dataset: the rows one head is trained on
    training.py          train_head(): split -> fold ensemble -> holdout report -> verdict -> conformal -> ledger
    policy.py            stakes tiers, head decisions, holdout evaluation, verify
    bernoulli.py         Beta-Bernoulli maths and the JSON ledger
    bundle.py            on-disk bundle format: write / read / validate (no pickle)
    registry.py          RLCD_HOME: versions, rlcd-latest alias, commit, load, export, import
    feedback.py          append-only feedback.jsonl and decisions.jsonl
    config.py            rlcd.yaml project config + environment
    service.py           Service: answer questions / rank / decide / evaluate / feedback (no HTTP)
    jobs.py              Trainer: one background training job at a time
  features/  text.py  lexical.py  retrieval.py
  packs/     base.py (Pack, FeatureProvider protocols, loader, dataset_from_labels)  documents/  tabular/
             (trading/ arrives with M8)
  connectors/ base.py  security.py  local_vector.py  local_graph.py   (qdrant / pgvector / neo4j: M7)
  labeling/  judges.py  store.py  bootstrap.py  audit.py  active.py
  serve/     api.py (FastAPI app factory)
  report/    model_card.py  html.py
  bench.py
  cli.py     `rlcd`
```

## 6. Core interfaces

### HeadSpec (`core/head.py`)

```python
@dataclass(frozen=True)
class HeadSpec:
    name: str                       # question id / head name, e.g. "answerable"
    type: str                       # noul | choice | score
    classes: tuple[str, ...]        # choice options / score levels; () for a noul
    view: str                       # feature view the pack's provider builds, e.g. "query_evidence"
    model: str = "logreg"           # logreg | lightgbm | auto
    decision: str = "argmax"        # argmax | gate (noul only: true = proceed)
    stakes: str = "medium"          # low | medium | high
    conformal: float | None = None  # target error rate for gate heads
    description: str = ""
    input: dict = {}                # human description of the expected state (for /v1/heads)
    params: dict = {}               # base-model hyper-parameters
    @property labels -> ("false", "true") for a noul, else classes   # internal class order
```

### Dataset (`core/dataset.py`)

```python
@dataclass
class Dataset:
    X: np.ndarray            # (n, f) float32 features (NaN = missing)
    y: np.ndarray            # (n,) int class index into spec.labels
    feature_names: list[str]
    feature_version: str
    ids: np.ndarray          # row ids (str)
    t: np.ndarray | None     # epoch seconds; enables time ordering
    t_end: np.ndarray | None # end of the label window (purge); defaults to t
    group: np.ndarray | None # group key (document / customer / ticket); enables group ordering
    weight: np.ndarray       # sample weights (feedback up-weight only; never class weights)
    trusted: np.ndarray      # bool: human-verified or ground truth
    source: np.ndarray       # human | ground_truth | judge | feedback | synthetic
    judge_y: np.ndarray | None  # judge label (class index, -1 none) for agreement reporting
    feature_groups: dict | None # feature -> group name for reason_groups
    segment: np.ndarray | None  # optional slice key for per-segment reports
```

### Training (`core/training.py`)

`train_head(spec, ds, cfg: TrainConfig) -> TrainedHead`

1. **Holdout**: `splits.holdout_split(ds, fraction=0.2)` -> time ordering: the latest 20 % of the trusted rows,
   development rows whose label window reaches the cut purged; group ordering: the latest-starting groups (or a
   seeded subset without timestamps), groups never straddle, and there is no time cut or purge between groups
   (use time ordering when groups share time context). Only **trusted**, representatively sampled rows are scored
   there; untrusted rows of holdout groups are dropped; feedback rows never enter the holdout, and a trusted row
   that duplicates a feedback row is taken out of it.
2. **Folds** over development: time-ordered expanding window with gap and purge, or group k-fold. Each fold's
   base model trains on every development row of its train side (weak + trusted + feedback, weighted); its
   calibrator is fitted on the **trusted** rows of its calibration block only.
3. **Method**: `sigmoid` unless every class has >= `isotonic_min_per_class` (3000) trusted rows in every
   calibration block, then `isotonic`; `temperature` on request. Served model = the fold ensemble, pooled
   geometrically (the plain average of calibrated members is underconfident when they disagree; measured on
   known generators the log-linear pool lowers ECE with equal or better log loss).
4. **Holdout report**: `metrics.calibration_report` against the base-rate forecaster (trusted development class
   frequencies, per segment when segments exist), block-bootstrap skill test (the gain must be reliably positive
   both over the training base rate and over the holdout's own class frequencies, which no constant forecast can
   beat), policy evaluation per stakes
   level, `verify` -> `verdict {max_tier, reason}`, conformal threshold (dev OOF) + holdout check, judge
   agreement, Bernoulli holdout table, data card.

### Base models and the calibrated ensemble (`core/head.py`)

* `logreg`: standardise (mean / scale saved) + multinomial `LogisticRegression`; raw scores = logits.
  Explanations: `coef[c] * (x - mean) / scale` (exact for a linear model).
* `lightgbm`: `LGBMClassifier` with the reference's small, regularised trees; raw scores = `raw_score=True`;
  explanations: TreeSHAP `pred_contrib`, averaged over members. Imported lazily; on macOS it needs
  `brew install libomp`. `model: auto` uses LightGBM when importable, else logreg (recorded in the report).
* Binary heads calibrate the positive-class score; multiclass heads calibrate each class one-vs-rest on its
  logit and renormalise (the scheme `CalibratedClassifierCV` uses).

### Policy (`core/policy.py`)

```yaml
policy:
  abstain: escalate            # name of the abstaining action (the tabular pack uses "hold")
  stakes:
    low:    {act: 0.70, confirm: 0.50, min_accuracy: 0.60}
    medium: {act: 0.85, confirm: 0.65, min_accuracy: 0.75}
    high:   {act: 0.95, confirm: 0.80, min_accuracy: 0.90}
  min_verify_rows: 30
  gate_min_p_above: 0.9        # Bernoulli gate on act
```

`p` = calibrated probability of the chosen class (argmax mode) or P(true) (gate mode).
`act` if `p >= act` (and, for conformal gate heads, `p >= conformal threshold`), `confirm` if `p >= confirm`,
else the abstain action. Then: cap by the head's `verdict.max_tier`; `act` -> `confirm` when the Bernoulli
gate fails. Every decision returns `action`, `tier`, `uncapped_tier`, `p`, `thresholds`, `max_tier`, `gate`,
`reasons`, `warnings`.

### Packs (`packs/base.py`)

```python
class FeatureProvider(Protocol):
    feature_version: str
    def names(self, view: str) -> list[str]
    def transform(self, view: str, states: list[Any]) -> np.ndarray     # float32 (n, f); ContractError on bad state
    def groups(self, view: str) -> dict[str, str]
    def describe(self, view: str) -> dict                                 # input documentation
    def to_config(self) -> dict                                           # JSON, stored in the bundle

class Pack(Protocol):
    name: str
    version: str
    def heads(self, project) -> dict[str, HeadSpec]
    def provider(self, project, config: dict | None = None) -> FeatureProvider   # config from a bundle
    def datasets(self, project, provider, heads: list[str]) -> dict[str, Dataset]
    # optional hooks
    def decide(self, service, request: dict) -> dict
    def evaluate(self, service, request: dict) -> dict
    def ingest(self, project, args) -> dict
    def label_tasks(self, project) -> Iterable[LabelTask]
    def synth(self, seed: int, n: int) -> dict
```

Every head also accepts a precomputed state `{x, feature_names, feature_version}`; a mismatched
`feature_version` is a 422 (`feature_version_mismatch`), as in the reference.

## 7. Bundle format (open weights)

```
<bundle>/
  model.json                      format "open-rlcd-bundle/1", version id, created_at, pack {name, version, config},
                                  feature_version, encoder {id, sha256, dim}, policy config, heads {name: {spec,
                                  dir, files {name: sha256}, metrics, verdict, calibration, conformal}}, data_card
  heads/<dir>/ensemble.json       spec, members (kind, params, calibrator JSON), feature names, missing-input rates
  heads/<dir>/member_<k>.safetensors   logreg: coef, intercept, mean, scale, fill
  heads/<dir>/member_<k>.txt           lightgbm: native text model
  heads/<dir>/head.json           metrics, verdict, calibration, conformal, data card, notes (hashed: the policy
                                  caps are read from here, not from model.json)
  heads/<dir>/report.json         the full holdout report
  heads/<dir>/bernoulli.json      the holdout table the Bernoulli ledger restarts from
  MODEL_CARD.md
```

Loading reads only JSON, safetensors and LightGBM text: it never executes code. `registry.py` keeps
`RLCD_HOME/registry.json` (`latest`, `next`, `versions`) and one bundle per version under `models/<version>/`;
a retrain that touches some heads carries the others over. Import validates names, extensions, sizes and
refuses symlinks and path traversal.

## 8. The documents pack

Heads (all overridable in `rlcd.yaml`):

| head | type | view | decision | notes |
|---|---|---|---|---|
| `response_type` | choice: direct_answer, extract_fields, summarize, compare, list, clarify_question, escalate_human, out_of_scope | query | argmax | |
| `route` | choice: vector, graph, hybrid, none | query | argmax | |
| `needs_graph` | noul | query | argmax | multi-hop traversal needed |
| `chunk_relevant` | noul | query_chunk | argmax | via `/v1/rank` to re-rank ANN candidates |
| `answerable` | noul | query_evidence | gate, conformal 0.05 | otherwise ask / escalate / refuse |
| `grounded` | noul | query_evidence_answer | gate, conformal 0.05, stakes high | evaluator |
| `complete` | score 0-3 | query_evidence_answer | argmax | evaluator |
| `risk_tier` | score 0-3 | query_evidence_answer | argmax | evaluator |

`/v1/decide` composes: response_type + route + needs_graph on the query; `chunk_relevant` re-ranks the candidates
(given by the caller, or retrieved through the configured connectors); `answerable` on the kept evidence; the
result is a plan (`response_type`, `route`, `use_chunks`, `call_llm`) with an overall `act / confirm / escalate`
that is never stronger than its weakest gate. `/v1/evaluate` judges a drafted answer (grounded, complete,
risk_tier) and returns ship / review / reject as `act / confirm / escalate`.

Labels: an LLM judge weak-labels (query, candidates, draft answers); humans audit a sample; the synthetic
generator (fictional company corpus in English, Japanese and Mongolian, multi-hop entity links, unanswerable
and out-of-scope queries, perturbed drafts with unsupported numbers) gives ground truth for tests and the demo.

## 9. Security

* `RLCD_API_KEY`: every route but `/health` then needs `Authorization: Bearer <key>` or `x-api-key` (constant-time
  compare); request bodies are capped (`RLCD_MAX_BODY_BYTES`); 500s never echo exception text or paths.
* Bundles load only from `RLCD_HOME` and contain no pickles or code. A bundle's `pack.name` is a plain name: the
  service serves it only with the project's own pack (same name) or a built-in pack, and never imports code named
  by a bundle. Import copies without following symlinks, validates the copy (hashes, file types, sizes, path
  containment) and parses every LightGBM member in a short-lived subprocess, refusing the bundle if it fails.
  Policy caps (verdict, conformal threshold, set quantile) are read from each head's hash-checked `head.json`.
* Connectors and judges refuse non-loopback hosts unless `security.allow_remote: true` or `RLCD_ALLOW_REMOTE=1`.
  Unix sockets are only explicit paths, `unix://`-style schemes without a network host, and libpq `?host=/dir`;
  a name ending in `.sock` is a host name.
* The `claude-cli` judge runs Claude Code with every tool and MCP server disabled (`--tools "" --strict-mcp-config
  --no-session-persistence`) in an empty temporary directory; judge prompts mark item text as untrusted data.
  The worst a prompt-injected document can do is earn itself a wrong weak label, which never calibrates.
* Tabular bundles carry the most frequent values of categorical columns; `pack_config.hash_categories: true`
  stores salted hashes instead, and `rlcd export` / `rlcd push` warn. `rlcd push --private` refuses a public repo.
* Secrets come from the environment only; `.gitignore` covers `var/`, `*.npz`, `.env`, caches and bundles other
  than the demo.

## 10. What it is not

Not an LLM and not a text generator. No nanosecond claims: heads take microseconds to well under a
millisecond per row; the encoder dominates (milliseconds to tens of ms per query on CPU); `rlcd bench` prints
the real numbers. Weak labels from an LLM judge are only as good as the judge, which is why calibration and
the published metrics use human-verified rows only. Synthetic-data metrics (tests, demo bundle) are a sanity
check, not accuracy on anyone's real traffic.

## 11. Milestones

M1 core + contract + tests -> M2 tabular pack -> M3 documents pack (local FAISS / NetworkX, synthetic RAG data)
-> M4 labeling -> M5 serve API + CLI + bench -> M6 bundle export / import, demo bundle, model card -> M7 Qdrant /
pgvector / Neo4j -> M8 trading adapter -> M9 docs, examples, CI. One commit per milestone with tests green.
