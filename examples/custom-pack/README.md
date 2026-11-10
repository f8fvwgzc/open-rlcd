# Custom pack example: support-ticket triage

A pack of your own, complete and small: a ticket's subject and body go in, and two questions come back with
calibrated probabilities.

| head | type | question |
|---|---|---|
| `queue` | choice | Which queue should handle this ticket: billing, technical, account, shipping or other? |
| `urgent` | noul | Does this ticket need an answer today? |

```
examples/custom-pack/
  rlcd.yaml              pack: ticketpack.pack:TicketsPack   (the code is found next to this file)
  ticketpack/
    pack.yaml            the two heads
    pack.py              the feature provider and the pack (110 lines)
    synth.py             synthetic tickets with known answers (English, Japanese, Mongolian)
  make_tickets.py        writes data/tickets.jsonl
```

It runs on the core install, offline. The guide that explains every part is
[docs/writing-a-pack.md](../../docs/writing-a-pack.md). The tickets are synthetic: the numbers below show that
the pack works, not how it would do on your tickets.

Run the commands from this folder (`cd examples/custom-pack`). Everything they write lands in `data/`, `var/`
and `bundle/`, which git ignores. The outputs below are real, trimmed where marked `...`.

## 1. Make the labelled tickets

```console
$ uv run python make_tickets.py
wrote 3000 tickets to .../examples/custom-pack/data/tickets.jsonl
  queues: {'account': 605, 'billing': 582, 'other': 625, 'shipping': 609, 'technical': 579}
  urgent: {False: 2244, True: 756}   languages: {'en': 2152, 'ja': 441, 'mn': 407}
```

One ticket per line:

```json
{"id": "T00000", "subject": "App crashes on start", "body": "Sync stopped working between my phone and laptop. We are losing money every hour, please help immediately.", "lang": "en", "created_at": "2026-01-05T08:00:00+00:00", "queue": "technical", "urgent": true, "source": "synthetic"}
```

About one ticket in ten says nothing specific ("I have a question about your service."), and `urgent` follows
the urgency wording with some noise, so no head can be certain.

## 2. Check the configuration

```console
$ uv run rlcd config check
.../examples/custom-pack/rlcd.yaml: no problems found
```

The pack declares its `pack_config` keys (`config_keys()` in `pack.py`), so a typo is an error, not a silent
default. With `txt: ngram` instead of `text: ngrams`:

```console
$ uv run rlcd config check
error  pack_config.txt: unknown key. Did you mean 'text'?
.../rlcd.yaml: 1 error, 0 warnings
```

## 3. Train

```console
$ uv run rlcd train
model rlcd-0.1.1 — job succeeded
head              status     holdout  logloss    base     BSS    ECE  skill verdict(medium)
queue             trained        601    0.281   1.611   0.873  0.017  yes   act
urgent            trained        601    0.274   0.588   0.629  0.033  yes   act
...
```

The holdout is the latest 601 tickets (the rows have a `created_at`, so the split is by time). `base` is the
log loss of always answering the base rate; `skill yes` means the head beat it by more than chance.

## 4. The pre-launch checklist

```console
$ uv run rlcd check
rlcd check: model rlcd-0.1.1

PASS  config.valid           rlcd.yaml is valid
PASS  heads.trained          Heads are trained
PASS  skill.queue            Head queue shows skill on its holdout
PASS  skill.urgent           Head urgent shows skill on its holdout
...
PASS  options.no_match       Choice heads have a no-match option
      every choice head can say that no option fits: queue (other)
PASS  heads.described        Heads and options are described
      all 2 heads have a description, and every choice head describes its options
PASS  noul.positive          Noul heads say what yes means
      every noul head (urgent) says what yes means
PASS  options.overlap        Options of one head are distinct
      no two options or levels share a description (queue)
...
WARN  rollback.ready         There is a version to roll back to
      rlcd-latest is rlcd-0.1.1 and there is no earlier version: `rlcd rollback` has no target until
      another version is trained or imported; ...
...
no check fails (15 pass, 1 warn, 0 fail, 15 info); the warnings are for the owners to judge before launch
```

Nothing in `pack.yaml` is left to warn about: each head has a `description`, the choice head describes its
options (`levels`) and has an `other`, the noul head says what yes means (`positive`). The one warning is about
operations (a first version has nothing to roll back to) and goes away with the retrain in step 7.

## 5. One decision

```console
$ uv run rlcd decide --state '{"subject": "URGENT: charged twice", "body": "I was charged twice for my subscription. Please answer today, our customers are waiting."}'
```

Trimmed to the answers and the reasons:

```
queue   billing  p 0.850   {billing 0.850, shipping 0.091, technical 0.022, other 0.019, account 0.017}
        reasons: text:twice, text:charged (+1), text:charged twice (+3), text:twice i, text:subscription, text:i was
        reason_groups: wording 0.9996, shape 0.0004
urgent  true     p 0.968
        reasons: text:twice i, text:urgent (+1), text:customers are, text:waiting, text:are waiting, text:please answer (+1)
        reason_groups: wording 0.992, shape 0.008
```

The reasons are n-grams of the ticket. `(+1)` means one other n-gram of the training vocabulary shares that
column (the columns are hashed; raise `pack_config.dim` to have fewer shared ones).

## 6. Serve, decide over HTTP, send the outcome back

```console
$ uv run rlcd serve --port 8195
```

In another terminal, a Japanese ticket ("the ordered item has not arrived / the parcel was damaged, please
handle this urgently"):

```console
$ curl -s http://127.0.0.1:8195/v1/decide -H 'content-type: application/json' \
    -d '{"state": {"subject": "注文した商品が届きません", "body": "荷物が破損していました。至急対応をお願いします。"}}'
```

```json
{"decision_id": "2ef59d22e2cf4c0685409cf2bafaad01", "model": "rlcd-0.1.1", "action": "confirm",
 "heads": {
  "queue": {"answer": {"type": "choice", "choice": "shipping",
                       "probabilities": {"billing": 0.096482, "technical": 0.006471, "account": 0.016873,
                                         "shipping": 0.846783, "other": 0.033391}, ...},
            "reasons": [{"feature": "text:至急 急対", "value": 0.1255, "contribution": -0.9271, "effect": "lowers P(shipping)"},
                        {"feature": "text:急対", ...},
                        {"feature": "text:破損", "value": 0.1255, "contribution": 0.4873, "effect": "raises P(shipping)"}, ...],
            "reason_groups": {"wording": 0.9994, "shape": 0.0006}, ...},
  "urgent": {"answer": {"type": "noul", "noul": 0.808257, ...}, ...}}, ...}
```

A person confirms the queue and the urgency; post what they decided, with the `decision_id`:

```console
$ curl -s http://127.0.0.1:8195/v1/feedback -H 'content-type: application/json' \
    -d '{"decision_id": "2ef59d22e2cf4c0685409cf2bafaad01", "head": "queue", "label": "shipping", "source": "agent:mika"}'
{"id":"0a9fe12de51b42ebb5a345b61d41e4bc","stored":true,"head":"queue","label":"shipping","feedback_rows":1,"bernoulli":{"updated":true,"class":"shipping","bin":8,"wins":4,"losses":0,"correct":true},"note":"used as an up-weighted training row at the next train"}
$ curl -s http://127.0.0.1:8195/v1/feedback -H 'content-type: application/json' \
    -d '{"decision_id": "2ef59d22e2cf4c0685409cf2bafaad01", "head": "urgent", "label": true, "source": "agent:mika"}'
{"id":"b98ee896586049ab8e153257d93d1c70","stored":true,"head":"urgent","label":"true","feedback_rows":1,"bernoulli":{"updated":true,"class":"true","bin":8,"wins":44,"losses":6,"correct":true},"note":"used as an up-weighted training row at the next train"}
```

The state does not have to be sent again: the service logged it with the decision.

## 7. Retrain: the feedback rows are counted

```console
$ uv run rlcd train
model rlcd-0.1.2 — job succeeded
head              status     holdout  logloss    base     BSS    ECE  skill verdict(medium)
queue             trained        601    0.280   1.611   0.874  0.015  yes   act
urgent            trained        601    0.274   0.588   0.630  0.034  yes   act

Data notes:
  all heads: 1 feedback rows added to training (weight x 5)
  all heads: 1 feedback rows fall in the holdout window: they train the model and the holdout rows they overlap were removed
...
```

The pack builds its datasets with `dataset_from_labels`, which adds each head's feedback rows (weight x 5,
the latest outcome per decision) and writes that note. A pack whose `datasets()` leaves the feedback out gets
a note saying so instead.

## 8. Export, and import somewhere else

```console
$ uv run rlcd export bundle
exported rlcd-0.1.2 to .../examples/custom-pack/bundle (JSON + safetensors + LightGBM text; no pickles). It contains no training rows, but heads learned from your data: share it only if that is allowed.
$ ls bundle bundle/heads
bundle:
MODEL_CARD.md  heads  model.json
bundle/heads:
queue  urgent
```

The bundle holds the heads and the provider's configuration (here the n-gram vocabulary: words of the training
tickets, as text in `model.json`), not the pack's code. The project that imports it needs `ticketpack/` next to
its own `rlcd.yaml`, and that file names it:

```console
$ mkdir ../triage-copy && cp -r ticketpack ../triage-copy/ && cd ../triage-copy
$ printf 'name: triage-copy\npack: ticketpack.pack:TicketsPack\n' > rlcd.yaml
$ rlcd import ../custom-pack/bundle
imported ../custom-pack/bundle as rlcd-0.1.1
$ rlcd decide --state '{"subject": "Cannot log in", "body": "My account is locked after too many attempts."}'
```

`queue` comes back as `account` with exactly the probabilities the training project gives
(`billing 0.001729, technical 0.002191, account 0.979352, shipping 0.00753, other 0.009197` in both). The copy
has no `data/tickets.jsonl`, which the pack's own config check mentions as a warning; serving does not need
it, because the provider is rebuilt from the bundle. A project
whose `rlcd.yaml` names another pack refuses the bundle instead of importing code on its word:

```
rlcd: model rlcd-0.1.1 was trained with pack 'tickets', which is neither this project's pack ('tabular') nor a
built-in pack ['documents', 'tabular']. A bundle never names code to import: set `pack:` in rlcd.yaml to the
pack it was trained with
```

## Embeddings instead of n-grams

`pack.py` builds its text columns with one of two helpers from `open_rlcd.features`, chosen in `rlcd.yaml`:

```yaml
pack_config:
  data: data/tickets.jsonl
  text: embedding
  encoder: {kind: hashing, dim: 384}      # offline; {kind: onnx, model_id: intfloat/multilingual-e5-small}
                                          # needs the documents extra and one download
```

```console
$ uv run rlcd train
model rlcd-0.1.1 — job succeeded
head              status     holdout  logloss    base     BSS    ECE  skill verdict(medium)
queue             trained        601    0.220   1.611   0.888  0.022  yes   act
urgent            trained        601    0.244   0.588   0.687  0.026  yes   act
```

The vectors are cached under `var/cache/embeddings`, so a retrain does not encode the same ticket twice. The
price is in the reasons: an embedding column has no meaning of its own (`text_231`, `text_355`), so read
`reason_groups` (`wording 0.9977, shape 0.0023`) rather than the single features.

## Use your own tickets

Point `pack_config.data` at a JSON Lines file with `id`, `subject`, `body`, `created_at` (ISO-8601 or epoch
seconds), and a label per head (`queue`, `urgent`); leave `source` out for labels a person gave (`human`), or
set it to `ground_truth` / `judge`. Change the queues in `ticketpack/pack.yaml`. A ticket without a label for a
head is skipped for that head.
