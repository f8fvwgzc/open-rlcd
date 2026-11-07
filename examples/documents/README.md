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
  naming the product, the wrong team page.
- **answerable**: gold evidence is true. Evidence missing one gold chunk is false, and so is any evidence for an
  unanswerable question.
- **grounded / complete / risk_tier**: on drafted answers in several wordings:
  - correct (grounded, complete 3);
  - one number, date, name or value changed (not grounded, still complete 3);
  - partial (grounded, complete 1-2);
  - about another entity (not grounded, complete 0).

  Risk follows the topic: specs 0, internal facts 1, money 2, legal / safety / medical 3.

Every label has source `synthetic` (trusted). The split group is the surface template, so the holdout contains
only phrasings the heads never saw. It does not contain unseen documents: entities recur across templates.

## What to expect

With the default seed and the hashing encoder, every head shows skill over the base rate on its holdout.
Typical values:

- response_type: accuracy about 0.77 against 0.50;
- chunk_relevant: about 0.90 against 0.79;
- grounded: about 0.92 against 0.52, and it keeps its 5 % conformal promise.

`answerable` has skill, but no threshold keeps its error rate under 5 % with 90 % confidence. Its verdict is
therefore capped at `confirm`, and the report says why. On some seeds `needs_graph` shows no skill, because
only a few graph templates land in the holdout; it then abstains (`escalate`) and says so. Both are the intended
behaviour: a head that has not shown evidence never acts.

Some decide plans end in `next: retrieve_more`: the gate saw that the evidence lacks the second document of a
multi-hop chain. Out-of-scope and ambiguous requests get `refuse_out_of_scope` and `ask_clarifying_question`
without calling an LLM. `evaluate` rejects the answer with the changed number and sends the faithful one to
review: price questions are risk tier 2, which means high stakes.

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
