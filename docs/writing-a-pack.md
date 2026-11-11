# Writing a pack

A **pack** is what a business plugs into Open RLCD: the questions it wants answered (heads), how a request's
`state` becomes a feature row (a feature provider), and where labelled rows come from. Packs are YAML plus a
little Python. The core does everything else the same way for every pack: time-ordered folds with a label-window
purge or group-ordered folds (whole groups, no purge), calibration on trusted rows, an untouched holdout, the
skill test, the policy, bundles, the API.

Write one when your decision is not a document question (`documents`) and not one row of a table (`tabular`).

This guide follows one complete pack, [examples/custom-pack](../examples/custom-pack/): support-ticket triage,
a ticket's subject and body in, a queue (one of five) and "urgent?" (yes/no) out. Every snippet below is from
that example or was run against it; its README walks the commands with their real output. Other packs to read:
`open_rlcd/packs/tabular/` (learns its schema from a file), `open_rlcd/packs/documents/` (the largest), and
`tests/toypack.py` (a few dozen lines, for tests).

## 1. The minimum: two files next to rlcd.yaml

```
my-project/
  rlcd.yaml              pack: ticketpack.pack:TicketsPack
  ticketpack/
    __init__.py          (may be empty)
    pack.yaml            the heads
    pack.py              a feature provider and the pack class
```

```yaml
# rlcd.yaml
name: ticket-triage
pack: ticketpack.pack:TicketsPack    # module:Class. This imports your code: point it only at code you trust
home: ./var
pack_config:
  data: data/tickets.jsonl
```

**Where the code is found.** `module:Class` is imported with the project folder (the folder of `rlcd.yaml`)
importable, so a pack that lives next to `rlcd.yaml` needs no install and no `PYTHONPATH`. Run `rlcd` from the
project folder (or below it), or set `RLCD_PROJECT`. The folder is added at the end of the import path, only
while the pack is imported: an installed package of the same name wins, and a file in your project can never
shadow the standard library. Make the pack a package (a folder with `__init__.py`) and import its own modules
relatively (`from .synth import write`); a pack you share between projects can be an installed package instead.

Only `rlcd.yaml` may name code to import. A bundle records the pack's `name` ("tickets") as data: the service
matches it to the project's own pack or to a built-in one and never imports anything on a bundle's word (see
[Share a bundle](#9-share-a-bundle)).

## 2. Heads: pack.yaml

```yaml
# ticketpack/pack.yaml
heads:
  queue:
    type: choice
    view: ticket
    classes: [billing, technical, account, shipping, other]
    levels:                          # what each option means, in class order (shown as the head's legend)
      - a charge, an invoice, a refund or a payment method
      - the product does not work - errors, crashes, slowness
      - sign-in, password, profile or access to the account
      - delivery of an order - late, lost, damaged, wrong address
      - anything else - opening hours, partnerships, thanks
    stakes: low
    description: Which queue should handle this support ticket?
  urgent:
    type: noul
    view: ticket
    positive: the customer is blocked or losing money now and asks for an answer today
    stakes: medium
    description: Does this ticket need an answer today?
policy:
  abstain: escalate                  # the abstaining action: a person triages the ticket
```

`rlcd check` passes this file with nothing left to warn about. What each field is, and which check asks for it:

| field | for | what it is | `rlcd check` |
|---|---|---|---|
| `type` | every head | `noul` (yes/no), `choice` (one of N), `score` (an ordered level) | |
| `view` | every head | which feature view the provider builds for it (default `default`) | |
| `description` | every head | the question, as a person would ask it | `heads.described` warns without it |
| `classes` | choice | the options, 2 or more. Include one that means "none of these" (`other`, `none`, `unknown`, `out_of_scope`, `no_match`, `needs_review`, or a name starting with `escalate` / `clarify`) | `options.no_match` warns without one |
| `levels` | choice | one description per option, in class order | `heads.described` warns without them; `options.overlap` warns when two are the same |
| `levels` | score | one description per level, lowest first; the classes become `"0"`, `"1"`, ... Describe what a reviewer can observe, not "low / medium / high" | `score.anchors` warns on levels that are only a name or a rank word |
| `positive` | noul | what `true` means, in words | `noul.positive` warns without it |
| `stakes` | every head | `low`, `medium` (default) or `high`: which thresholds the policy uses | |
| `decision` | noul | `argmax` (default) or `gate` (`true` means "proceed") | |
| `conformal` | optional | gate heads: promise at most this error rate among the yeses (checked on the holdout) | |
| `model` | optional | `logreg` (default), `lightgbm`, or `auto` (LightGBM when importable) | |

A score head, for the shape of its `levels` (`score.anchors` passes these):

```yaml
  severity:
    type: score
    view: ticket
    levels: ["0: a question, nothing is broken", "1: one user cannot do one task", "2: a whole team is blocked",
             "3: data loss or an outage for every customer"]
    description: How severe is the problem the ticket reports?
```

Users override any field per head in their `rlcd.yaml` under `heads:` (and `enabled: false` removes a head).
`policy.abstain` names the abstaining action (the tabular pack uses `hold`).

## 3. The feature provider

The provider turns states into feature rows, at training time and for every request.

```python
# ticketpack/pack.py
VIEW = "ticket"
SHAPE = ["length_chars", "exclamation_marks"]      # two plain numbers next to the text columns


def text_of(state: dict) -> str:
    return f"{state.get('subject') or ''}\n{state.get('body') or ''}"


class TicketFeatures:
    """One feature row per ticket: the text block's columns, then SHAPE."""

    def __init__(self, text: HashedNgrams | TextEmbeddings):
        self.text = text
        # the block's signature changes with its column names (n-grams) or its encoder: so does the version
        self.feature_version = f"tickets-v1:{text.signature}"

    def names(self, view: str) -> list[str]:
        return [*self.text.names(), *SHAPE]

    def transform(self, view: str, states: list) -> np.ndarray:
        for s in states:
            if not isinstance(s, dict) or not isinstance(s.get("subject") or s.get("body"), str):
                raise ContractError("state must be a ticket: {subject: text, body: text}", field="state")  # -> 422
        texts = [text_of(s) for s in states]
        shape = np.asarray([[len(t), t.count("!") + t.count("！")] for t in texts], np.float32).reshape(len(texts), 2)
        return np.hstack([self.text.transform(texts), shape])

    def groups(self, view: str) -> dict[str, str]:
        return {**self.text.groups("wording"), **dict.fromkeys(SHAPE, "shape")}

    def describe(self, view: str) -> dict:
        return {"subject": "string", "body": "string"}

    def to_config(self) -> dict:
        return {"text": self.text.to_config()}
```

The contract:

| member | what it must do |
|---|---|
| `names(view)` | The column names, in column order. They are what a decision's `reasons` show, so make them readable. |
| `transform(view, states)` | One float32 row per state, `len(names(view))` columns. Missing is NaN, not zero, unless zero really means zero. A state it cannot read raises `ContractError` (the API answers 422). |
| `groups(view)` | `{column name: group}` for `reason_groups`: how much of a decision each kind of evidence carried. |
| `describe(view)` | The shape of the state a caller sends; shown by `GET /v1/heads`. |
| `to_config()` | Plain JSON with everything learned from the training data (vocabularies, scalers, an encoder's identity). It is stored in the bundle, and `pack.provider(project, config)` must rebuild exactly the same provider from it. No pickle. |
| `feature_version` | A string that names the feature schema. |

**When to bump `feature_version`.** Whenever the same state would give a different row: a column added,
removed, renamed or reordered, a changed computation, another vocabulary or encoder. The service refuses to
score a state with a version other than the one the head was trained on, and a head is carried over into a
partial retrain (`rlcd train --heads ...`) only while the version is unchanged. Change the prefix by hand
(`tickets-v1` to `tickets-v2`) when you change the code; put a hash of what is learned from data into it (here
`text.signature`), so a retrain on new tickets changes it by itself.

**Same features at train and serve time.** When a version is served, the provider is rebuilt from the bundle's
config, not from today's data:

```python
    def provider(self, project, config=None) -> TicketFeatures:
        if config:                                   # serving a bundle: rebuild exactly what it was trained with
            return TicketFeatures(text_block(config["text"], home=project.home))
        if project.pack_config.get("text") == "embedding":
            return TicketFeatures(TextEmbeddings.for_project(project, prefix="text"))
        dim = int(project.pack_config.get("dim") or 2048)
        return TicketFeatures(HashedNgrams(dim=dim, prefix="text").fit([text_of(r) for r in self.rows(project)]))
```

Every head also accepts a precomputed state `{x, feature_names, feature_version}`; the service checks it
against the bundle, so your provider never sees it.

## 4. Text features

Most packs have a text field. `open_rlcd.features` has two blocks that turn one into columns; both have
`names()`, `transform(texts)`, `groups(group)`, `to_config()` and `signature`, and `text_block(config, home=...)`
rebuilds either from its config.

```python
from open_rlcd.features import HashedNgrams, TextEmbeddings, text_block

block = HashedNgrams(dim=2048, prefix="text").fit(training_texts)     # or TextEmbeddings.for_project(project)
X, names = block.transform(texts), block.names()
same = text_block(block.to_config(), home=project.home)               # what provider(project, config) does
```

| | `HashedNgrams` | `TextEmbeddings` |
|---|---|---|
| what | word n-grams (1 and 2 words by default; character bigrams inside Japanese / CJK runs) hashed into `dim` columns | dense vectors of a frozen encoder |
| needs | numpy; nothing to download | the encoder in `pack_config.encoder`: `{kind: hashing, dim: 384}` offline, or `{kind: onnx, model_id: ...}` (the `documents` extra and one download). No `encoder` key: the offline hashing encoder |
| column names | `text:refund`, `text:charged twice (+3)` | `text_000`, `text_001`, ... |
| reasons | name the words that moved the answer | read `reason_groups`; single columns mean nothing |
| understands | surface forms only: "car" and "automobile" are unrelated | meaning, as far as the encoder does |
| stored in the bundle | the vocabulary (words of your training texts, as text) | the encoder's identity and the hash of its model file |
| cache | none needed | `<home>/cache/embeddings`, by content hash |

`HashedNgrams.fit(texts)` learns the vocabulary: the n-grams found in at least two training texts. After it
only those count, and each column is named after its most frequent n-gram; `(+3)` means three more share the
column (raise `dim` when many do). Without `fit` the columns are `text:#0`, `text:#1`, ...

A decision from the example, trained on n-grams (`rlcd decide --state '{"subject": "URGENT: charged twice", ...}'`):

```
queue   billing  p 0.850
        reasons: text:twice, text:charged (+1), text:charged twice (+3), text:twice i, text:subscription, text:i was
        reason_groups: wording 0.9996, shape 0.0004
```

The lower-level pieces (`make_encoder`, `CachedEncoder`, `tokens`, BM25, pair features) are in the same
package when you need them.

## 5. Labels

The pack hands the trainer one `Dataset` per head. Label provenance (`source`) decides what a row may do:

| source | trains the base model | fits calibrators | scored in the holdout |
|---|---|---|---|
| `human`, `ground_truth`, `synthetic` (trusted) | yes | yes | yes |
| `judge` (LLM weak label) | yes | no | no |
| `feedback` (production outcome, up-weighted) | yes | no | no |

**Use `dataset_from_labels` unless you have a reason not to.** Give it the labelled items of one head and the
`feedback` argument as you received it:

```python
    def datasets(self, project, provider, heads, feedback=None, feedback_weight=5.0):
        rows, specs = self.rows(project), self.heads(project)
        out = {}
        for name in heads:
            labelled = [{"item": LabelItem(id=r["id"], view=VIEW, t=epoch(r.get("created_at")),
                                           state={"subject": r.get("subject"), "body": r.get("body")}),
                         "label": r[name], "source": r.get("source") or "human"}
                        for r in rows if r.get(name) is not None]
            # adds the feedback rows of this head (up-weighted, latest per decision) and says so in a note
            out[name] = dataset_from_labels(specs[name], provider, labelled, feedback, feedback_weight)
        return out
```

(`LabelItem` is `open_rlcd.labeling.store.LabelItem`; `dataset_from_labels` and `epoch` are in
`open_rlcd.packs.base`.) It does what a hand-written loop tends to forget:

* **Feedback.** Outcomes posted to `/v1/feedback` for this head become training rows of source `feedback`,
  weighted x `feedback_weight`. Only the latest outcome per decision counts. A note says how many were added.
* **Groups.** An item's `group` (customer, thread, document) keeps its rows on one side of the holdout. A
  feedback row takes the group of the labelled item it is about (pass `group_of=` to compute a group from a
  state; otherwise an item with an identical state or identical features is used), so that item is never
  scored while its outcome trains the model.
* **Time.** An item's `t` (epoch seconds; `epoch()` reads ISO-8601) gives time ordering: the holdout is the
  latest period.
* **Bad labels** (not an option of the head) are skipped and counted in a note. Judge labels only train.

Give rows a **group** whenever rows can be near-duplicates; without a group but with a **time**, the holdout is
the latest period and rows whose label window (`t_end`) reaches it are purged; with neither, rows are treated
as independent and the report says so. With both a group and a time, groups never straddle the holdout and are
ordered by their first timestamp, but there is no time cut or label-window purge between groups: if rows of
different groups share time context (one market, one news cycle), set `train.ordering: time` instead.

**Your own `datasets()`.** Build the `Dataset` yourself when rows are not items with a state: features computed
in bulk from a table (the tabular pack), label windows (`t_end`), segments, per-row weights. Then feedback is
yours to add: for each row of `feedback` whose `head` is yours, featurise `row["state"]`, append it with source
`"feedback"` and weight `row["weight"] * feedback_weight`, keep only the last row per `decision_id`, and add a
note (`_feedback_rows` in `open_rlcd/packs/tabular/pack.py` is a full version). If `datasets()` is given
feedback rows for a head and returns a dataset without a single `feedback` row and without a note about
feedback, the trainer's notes say so, as here for a pack that ignored the argument:

```
2 feedback rows for head 'good' were passed to IgnoresFeedback.datasets() but none is in the training set (no
row has source "feedback"): the pack ignores its `feedback` argument or dropped every row. Build the dataset
with open_rlcd.packs.base.dataset_from_labels, which adds them (docs/writing-a-pack.md, "Labels")
```

`rlcd train` prints dataset notes under "Data notes", and the model card lists them per head. With the
example, after one decision's outcome was posted for both heads:

```
Data notes:
  all heads: 1 feedback rows added to training (weight x 5)
  all heads: 1 feedback rows fall in the holdout window: they train the model and the holdout rows they overlap were removed
```

If your labels come from the LLM-judge / audit workflow (`rlcd label bootstrap`, `rlcd label audit`), yield
`LabelItem`s from `label_tasks()` and pass `LabelStore(project.home).labelled(head, view)` to
`dataset_from_labels`, as the documents pack does.

## 6. Let `rlcd config check` validate your pack_config

A custom pack's `pack_config` is not checked until the pack says which keys it reads:

```python
from open_rlcd.core.config_check import Key

class TicketsPack(BasePack):
    name = "tickets"
    version = "0.1.0"
    yaml_path = Path(__file__).with_name("pack.yaml")

    def config_keys(self) -> dict:
        return {"data": Key("str", default="data/tickets.jsonl"),
                "text": Key("str", default="ngrams", choices=("ngrams", "embedding")),
                "dim": Key("int", default=2048, min=8),
                "encoder": Key("map")}          # read by TextEmbeddings.for_project when text: embedding

    def check_config(self, project) -> list[dict]:
        path = self.data_path(project)
        if path.is_file():
            return []
        return [{"level": "warn", "path": "pack_config.data",
                 "message": f"{path} does not exist yet: run `python make_tickets.py`, or point it at your tickets"}]
```

| hook | what it gives |
|---|---|
| `config_keys()` | `{key: Key(...)}`: the keys of `pack_config`, with kind (`str`, `bool`, `int`, `number`, `list`, `map`, ...), `default`, `choices`, `min` / `max`. An unknown key is then an error with a suggestion, a wrong type or range an error. Returning `None` (the default) leaves `pack_config` unchecked. |
| `check_config(project)` | More findings, `[{level: "error" \| "warn", path, message}]`, for what a key list cannot say (a file that is missing, two keys that contradict). It runs after the key and type checks passed and must not need the network. |
| `head_override_keys()` | Extra fields your pack accepts under `heads.<head>` in rlcd.yaml, beyond the head fields of section 2. |

With `txt: ngram` written instead of `text: ngrams`:

```console
$ rlcd config check
error  pack_config.txt: unknown key. Did you mean 'text'?
.../rlcd.yaml: 1 error, 0 warnings
```

`rlcd train` runs the same check first and stops on an error.

## 7. Optional hooks

| hook | used by |
|---|---|
| `decide(service, request)` | `POST /v1/decide`: compose several heads into one decision (the documents plan). Without it, the generic decide answers the requested heads and takes the weakest tier. Use `service.head_decision(...)` and `service.record(...)` so feedback can refer to the decision. |
| `evaluate(service, request)` | `POST /v1/evaluate` (evaluator mode). |
| `ingest(project, args)` | `rlcd ingest`. |
| `label_tasks(project, provider)` | `rlcd label bootstrap`: the items an LLM judge and humans label. |
| `synth(seed, n)` | synthetic data for tests and demos (with known answers). |
| `rank_head` (attribute) | the default head of `POST /v1/rank`. |
| `switched_off(project, trained)` | The service (which heads of a trained version it serves), `rlcd check` and `rlcd config check` (whose route contracts, rollout modes and policy overrides they keep without comparing them with the head): the heads the project switches off, given a version's trained heads. The default is `heads.<head>.enabled: false`; extend it when your `pack_config` can switch a head off too (the tabular pack adds `pack_config.targets.<head>.enabled: false` and every trained head that is no longer a target). |

The example needs none of them to train, serve and take feedback; it has `synth` only to write its demo data.

## 8. Test a pack

A pack is tested like the built-in ones: on synthetic data whose answers you know. A head with signal must
come out with skill, and pure noise must come out "no skill, abstain". `tests/test_example_custom_pack.py`
tests the example in a few seconds; its core is:

```python
project = load_project(root, overrides={"pack_config": {"dim": 512}})
svc = Service(project, home=root / "var")
pack = svc.pack                                       # imports ticketpack from the project folder
import_module("ticketpack.synth").write(root / "data" / "tickets.jsonl", n=900, seed=3)
job = svc.trainer.start(TrainRequest(), wait=True)

assert job["heads"]["queue"]["metrics"]["has_skill"] is True
out = svc.decide(DecideRequest(state=TICKET))
assert out["heads"]["queue"]["answer"]["choice"] == "billing"
```

Worth a test each:

* **It loads** from a project folder with nothing on `PYTHONPATH` (the test enters the folder as a user would).
* **Skill where there is signal**, and readable `reasons` on a decision.
* **Feedback reaches the next training set**: post an outcome with `svc.feedback(FeedbackRequest(...))`, call
  `pack.datasets(project, pack.provider(project), heads, svc.logs.feedback(), 5.0)` and count the `feedback`
  rows (`ds.counts()["by_source"]`).
* **The bundle round-trips**: export, import into a `Service` with an empty home, and compare probabilities.
  They must be equal, not close: a difference means `to_config()` leaves something out.
* **The provider alone**: `provider(project, provider.to_config())` gives the same rows after a JSON round trip.

Before launch, run `rlcd check` on a trained project: it reads the same heads as your users will.

## 9. Share a bundle

`rlcd export <dir>` writes a version as JSON + safetensors + LightGBM text: the heads, the policy, and your
provider's `to_config()`. It holds no training rows and no code. Whatever `to_config()` returns is in it as
plain text (an n-gram vocabulary is words of your training texts), so look at it before a bundle leaves the
company.

The receiving project needs the pack's code, and its own `rlcd.yaml` must name it:

```console
$ mkdir ../triage-copy && cp -r ticketpack ../triage-copy/ && cd ../triage-copy
$ printf 'name: triage-copy\npack: ticketpack.pack:TicketsPack\n' > rlcd.yaml
$ rlcd import ../custom-pack/bundle
imported ../custom-pack/bundle as rlcd-0.1.1
```

It then answers with exactly the probabilities of the project that trained it. The bundle's pack name
("tickets", your pack's `name` attribute) is matched against the project's configured pack; it is data, never
an import path. A project configured with another pack refuses the bundle:

```
rlcd: model rlcd-0.1.1 was trained with pack 'tickets', which is neither this project's pack ('tabular') nor a
built-in pack ['documents', 'tabular']. A bundle never names code to import: set `pack:` in rlcd.yaml to the
pack it was trained with
```

So ship the pack's code the way you ship any code you trust (a repository, an internal package), and the
bundle next to it. Keep the pack's `name` stable, and bump `feature_version` rather than editing features in
place: a bundle trained on the old features is then refused with a clear message instead of answering wrongly.
