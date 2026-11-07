# Documents pack demo: a calibrated router / gatekeeper / evaluator for RAG

```bash
uv run python examples/documents/build_demo.py                    # offline, hashing encoder, about 15 s
uv run python examples/documents/build_demo.py --encoder onnx     # multilingual-e5-small (downloads once)
uv run python examples/documents/build_demo.py --docs ~/my-docs --query "What is our refund window?"
```

The script builds a project in `examples/documents/var/` (kept out of git by the repository's `var/` rule):
`rlcd.yaml`, a synthetic corpus, labelled items and a trained model. It then prints each head's holdout summary,
asks for three plans (`decide`) and judges two drafted answers (`evaluate`). Serve the result with
`cd examples/documents/var && uv run rlcd serve`.

## The synthetic data

`open_rlcd/packs/documents/world.py` invents a logistics company, *Kestrovia Logistics*. Any resemblance to a real
organisation, product or person is accidental. It has six departments, twelve teams (each with a department, an
office and a lead), eighteen products (SKU, price, warranty, weight, launch date, owning team), nine policies,
four office pages, and eight news items that name products without stating any of their facts (hard negatives).
Most pages are in English. The Osaka teams write Japanese pages and the Ulaanbaatar teams write Mongolian
(Cyrillic) pages. Product, team and code names stay in Latin script, as they usually do in real companies.

A product sheet names and links its owning team. The team page names the lead but no product. So "who leads
the team that owns the Velobox 300?" needs two documents.

`synth.py` writes 900 queries by default, in English, Japanese and Mongolian:

| family | response_type | route | example |
|---|---|---|---|
| attribute / policy lookups | direct_answer | vector | "What is the warranty period of the Velobox 300?" |
| field extraction | extract_fields | vector or graph | "Give the SKU, price and warranty of the Tessavio Dock." |
| summaries | summarize | vector | "Summarize the refund policy." |
| comparisons | compare | hybrid | "Compare the Velobox 300 and the Arcternix Hub on price." |
| lists | list | hybrid or graph | "List all products owned by the Cold Chain team." |
| multi-hop | direct_answer | graph | "Who leads the team that owns the Velobox 300?" |
| ambiguous | clarify_question | none | "What does it cost?" |
| complaints, legal, personal data | escalate_human | none | "Give me the phone number of Lena Hartmann." |
| chit-chat | out_of_scope | none | "What's the weather in Lisbon tomorrow?" |

About a quarter of the lookups are unanswerable: the product does not exist, or no document states the
attribute (battery life, country of manufacture). The candidates come from the real retriever (vector + graph +
BM25, plus documents the top hits link to). From each query the generator labels:

- **chunk_relevant**: gold chunks are true. The strongest look-alikes are false: other products' sheets, news
  naming the product, the wrong team page. A real product's sheet stays true when the question also asks for a
  field no document states, or compares it with a product that does not exist: it holds part of the answer.
- **answerable**: gold evidence is true. Evidence missing one gold chunk is false, and so is any evidence for an
  unanswerable question. About 15 % of the queries also get a row with empty evidence (false): `decide` builds
  that state when no candidate reaches `min_relevance`, so the head is calibrated on it.
- **grounded / complete / risk_tier**: on drafted answers in several wordings:
  - correct (grounded, complete 3);
  - one number, date, name or value changed (not grounded, still complete 3);
  - partial (grounded, complete 1-2 by the share of facts kept; a multi-hop answer that states only the
    owning team misses the question, complete 0, and one that states the asked lead / office / department is
    complete 3);
  - about another entity (not grounded, complete 0).

  Risk follows the topic: specs 0, internal organisation facts 1, money 2, legal / safety / medical 3.

Query times (and the `now` every state carries) spread over six years from January 2025, while the documents are
dated before 2025. `decide` measures document age from the time of the request, so a request made today falls
inside the range the heads were trained on.

Every label has source `synthetic` (trusted). The split group is the surface template, so the holdout contains
only phrasings the heads never saw. It does not contain unseen documents: entities recur across templates.

## What to expect

The script prints each head's holdout summary. With the default seed (7), 900 queries and the hashing encoder it
reads:

```
head            status   skill    acc   base    ECE  verdict (low/medium/high)
response_type   trained  yes    0.863  0.363  0.069  act/act/confirm
route           trained  yes    0.824  0.516  0.087  act/act/act
needs_graph     trained  NO     0.923  0.846  0.037  escalate/escalate/escalate
chunk_relevant  trained  yes    0.839  0.696  0.020  act/act/act
answerable      trained  yes    0.694  0.495  0.081  confirm/confirm/confirm
grounded        trained  yes    0.927  0.560  0.055  confirm/confirm/confirm
complete        trained  yes    0.842  0.756  0.087  act/act/act
risk_tier       trained  yes    0.854  0.475  0.050  confirm/confirm/confirm
```

`acc` is holdout accuracy and `base` the accuracy of the base-rate forecaster. The exact numbers move a little
with the code version; rerun the script for yours.

- `needs_graph` shows **no skill** at this seed: its accuracy beats the base rate, but the block-bootstrap skill
  test over whole holdout templates does not establish a gain (few graph templates land in the holdout). It
  abstains (`escalate`) on every decision and every plan says so in its warnings.
- `answerable` and `grounded` have skill, but no threshold keeps their error rate under 5 % with 90 %
  confidence, so neither keeps a conformal promise: both are capped at `confirm` and the report says why. An
  answer is therefore never shipped automatically on this data (`evaluate` gives at most `review`).
- `risk_tier` is capped at `confirm` by its holdout verification: its act-tier rows came true less often than
  their mean predicted probability.

These are the intended behaviours: a head that has not shown evidence never acts.

The three example plans: the warranty question and the multi-hop question both end in `next: answer` with
action `confirm` (the multi-hop plan keeps only the product sheet as evidence: the team page is retrieved and ranked
second, but chunk_relevant gives it less than 0.5); the weather question is refused (`refuse_out_of_scope`) without calling an
LLM. A plan ends in `next: retrieve_more` when no candidate reaches `min_relevance` or the answerable gate says
the evidence is not sufficient, and in `escalate_to_human` when response_type itself abstains. `evaluate` sends the
faithful price answer to review and rejects the one with the changed number: price questions are risk tier 2,
which means high stakes.

**These are sanity checks of the pipeline, not accuracy.** The heads learned a synthetic world. With `--docs` your
documents join the corpus for retrieval only; nothing is labelled for you. To get calibrated probabilities for your
own traffic, put a query log in `pack_config.queries`, label it (`rlcd label bootstrap`, then `rlcd label audit` for
the trusted rows) and retrain.

## Calling the served model

```bash
curl -s localhost:8095/v1/decide -H 'content-type: application/json' \
  -d '{"state": {"query": "Who leads the team that owns the Velobox 300?"}, "top_k": 5, "min_relevance": 0.5}'

curl -s localhost:8095/v1/evaluate -H 'content-type: application/json' -d '{"state": {
  "query": "What is the price of the Velobox 300?",
  "evidence": [{"id": "products/velobox-300#0", "text": "..."}],
  "answer": "The price of Velobox 300 is 3,900 USD."}}'

curl -s localhost:8095/v1/rank -H 'content-type: application/json' \
  -d '{"state": {"query": "Velobox 300 weight?"}, "candidates": [{"id": "a", "state": {"text": "..."}}]}'
```

Each decision returns a `decision_id`. Post the real outcome to `/v1/feedback` with
`{decision_id, head, label, source}`, naming any head that took part. The Bernoulli ledger updates immediately,
and the next train uses the row with extra weight.
