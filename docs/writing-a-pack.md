# Writing a pack

A **pack** is what a business plugs into Open RLCD: the questions it wants answered (heads), how a request's
`state` becomes a feature row (a feature provider), and where labelled rows come from. Packs are YAML plus a
little Python. The core does everything else the same way for every pack: time-ordered folds with a label-window
purge or group-ordered folds (whole groups, no purge),
calibration on trusted rows, an untouched holdout, the skill test, the policy, bundles, the API.

Built-in packs to read: `open_rlcd/packs/tabular/` (simplest), `open_rlcd/packs/documents/` (largest), and
`tests/toypack.py` (a few dozen lines, for tests).

## 1. Declare the heads

```yaml
# my_pack/pack.yaml
heads:
  approve:
    type: noul                 # noul | choice | score
    view: application          # which feature view the provider builds for it
    decision: gate             # argmax (default) | gate: noul only, `true` means "proceed"
    stakes: high               # low | medium | high: which thresholds the policy uses by default
    conformal: 0.05            # gate heads: promise error <= 5% among the yeses (checked on the holdout)
    model: auto                # logreg | lightgbm | auto (LightGBM when importable)
    description: Should this application be approved without manual review?
  reason_code:
    type: choice
    classes: [income, history, fraud_signal, documents_missing, other]
    view: application
  urgency:
    type: score
    levels: ["0: whenever", "1: this week", "2: today", "3: now"]
    view: application
policy:
  abstain: escalate            # the name of the abstaining action (the tabular pack uses "hold")
```

Users override any field per head in their `rlcd.yaml` under `heads:` (and `enabled: false` removes a head).

## 2. Write the feature provider

```python
import numpy as np
from open_rlcd.core.contract import ContractError

class ApplicationFeatures:
    feature_version = "applications-v1"           # change it whenever the columns change

    def names(self, view: str) -> list[str]:
        return ["income", "debt_ratio", "age_days", "n_docs"]

    def transform(self, view: str, states: list) -> np.ndarray:
        rows = []
        for s in states:
            if not isinstance(s, dict):
                raise ContractError("state must be an application object")   # -> HTTP 422
            rows.append([s.get("income", np.nan), s.get("debt_ratio", np.nan), s.get("age_days", np.nan),
                         len(s.get("documents") or [])])
        return np.asarray(rows, np.float32)

    def groups(self, view: str) -> dict[str, str]:      # for reason_groups in decisions
        return {"income": "finance", "debt_ratio": "finance", "age_days": "history", "n_docs": "documents"}

    def describe(self, view: str) -> dict:              # shown by GET /v1/heads
        return {"income": "number", "debt_ratio": "number", "age_days": "number", "documents": "array"}

    def to_config(self) -> dict:                        # stored in the bundle; provider(config) must rebuild it
        return {"version": 1}
```

Rules that keep the probabilities honest:

* **Same features at train and serve time.** Anything learned from the training data (vocabularies, scalers,
  an encoder's identity) goes into `to_config()`, and `provider(project, config)` must rebuild exactly that.
* **Missing is NaN**, not zero, unless zero really means zero.
* **Bump `feature_version`** when columns change: the service refuses to serve a bundle with a mismatched one.

## 3. Provide labelled rows

```python
from pathlib import Path
import numpy as np
from open_rlcd.core.dataset import Dataset
from open_rlcd.packs.base import BasePack

class ApplicationsPack(BasePack):
    name = "applications"
    version = "0.1.0"
    yaml_path = Path(__file__).with_name("pack.yaml")

    def provider(self, project, config=None):
        return ApplicationFeatures()

    def datasets(self, project, provider, heads, feedback=None, feedback_weight=5.0):
        rows = load_my_labelled_rows(project.pack_config["data"])      # your code
        out = {}
        for name in heads:
            spec = self.heads(project)[name]
            X = provider.transform(spec.view, [r["state"] for r in rows])
            out[name] = Dataset(
                X=X, y=[spec.label_index(r["labels"][name]) for r in rows],
                feature_names=provider.names(spec.view), feature_version=provider.feature_version,
                group=[r["customer_id"] for r in rows],      # rows of one customer never straddle the holdout
                t=[r["decided_at_epoch"] for r in rows],     # the holdout is the latest groups
                source=["ground_truth"] * len(rows))         # trusted: calibrates and is scored
        return out
```

Label provenance (`source`) decides what a row may do:

| source | trains the base model | fits calibrators | scored in the holdout |
|---|---|---|---|
| `human`, `ground_truth`, `synthetic` (trusted) | yes | yes | yes |
| `judge` (LLM weak label) | yes | no | no |
| `feedback` (production outcome, up-weighted) | yes | no | no |

Give rows a **group** (document, customer, ticket) whenever rows can be near-duplicates; without a group but with
a **time**, the holdout is the latest period and rows whose label window (`t_end`) reaches it are purged; with
neither, rows are treated as independent and the report says so. With both a group and a time, groups never
straddle the holdout and are ordered by their first timestamp (the holdout is the latest-starting groups), but
there is no time cut or label-window purge between groups: if rows of different groups share time context (one
market, one news cycle), set `train.ordering: time` instead.

If your labels live in the label store (the LLM-judge / audit workflow), skip the hand-written loop: yield
`LabelItem`s from `label_tasks()` and build datasets with `open_rlcd.packs.base.dataset_from_labels`, as the
documents pack does. Feedback rows (`/v1/feedback` with a raw `state`) come for free that way: a feedback row
inherits the group of the labelled item it is about (pass `group_of=` to `dataset_from_labels`, else an item with
an identical state or identical features is used), and the split never scores a labelled row whose features
exactly equal a training feedback row's. Items with timestamps but no group get time ordering.

## 4. Optional hooks

| hook | used by |
|---|---|
| `decide(service, request)` | `POST /v1/decide`: compose several heads into one decision (the documents plan). Without it, the generic decide answers the requested heads and takes the weakest tier. Use `service.head_decision(...)` and `service.record(...)` so feedback can refer to the decision. |
| `evaluate(service, request)` | `POST /v1/evaluate` (evaluator mode). |
| `ingest(project, args)` | `rlcd ingest`. |
| `label_tasks(project, provider)` | `rlcd label bootstrap`: the items an LLM judge and humans label. |
| `synth(seed, n)` | synthetic data for tests and demos (with known answers). |
| `rank_head` (attribute) | the default head of `POST /v1/rank`. |

## 5. Use it

```yaml
# rlcd.yaml
pack: my_package.my_pack:ApplicationsPack      # imports your code: point it only at code you trust
pack_config: {data: ./applications.csv}
```

Then `rlcd train`, `rlcd report`, `rlcd serve` as usual. Test it like the built-in packs: a synthetic generator
whose true probabilities you know must come out calibrated, and pure noise must come out "no skill → abstain".
