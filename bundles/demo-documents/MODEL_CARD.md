# Model card — rlcd-0.1.1 (documents pack)

Created 2026-11-07T15:17:59Z. Format `open-rlcd-bundle/1`. Feature version `documents-v2:hashing-256-s0:0b1d19ee81e5`.

> **Trained on synthetic data only.** These numbers show that the pipeline works and that the metrics are computed honestly; they say nothing about accuracy on real traffic.

## Intended use

Routing, gatekeeping and evaluation in front of / behind an LLM over an enterprise document store: which response type, which retrieval route, which chunks matter, whether the evidence is enough, and whether a drafted answer is grounded, complete and how risky it is.

## What it is not

* Not a language model: it never writes text. It decides what, whether and from which evidence.
* Not magic: heads take microseconds per row, the text encoder milliseconds; run `rlcd bench`.
* Not better than its labels: weak labels from an LLM judge train the base models, but calibration and every number below use trusted (human-verified, ground-truth or synthetic) rows only.
* A head whose holdout shows no skill over the base rate abstains on every request.

## Data

Label sources (all heads): synthetic 12445

Text encoder: `hashing-256-s0` (dim 256, sha256 `0b1d19ee81e5e56e…`).

## Heads (trusted holdout)

| head | type | model | calibration | holdout rows | log loss / baseline | Brier skill | ECE | skill | verdict | conformal threshold |
|---|---|---|---|---|---|---|---|---|---|---|
| `response_type` | choice | logreg | sigmoid | 182 | 0.521 / 1.910 | 0.718 | 0.069 | yes | act | – |
| `route` | choice | logreg | sigmoid | 182 | 0.484 / 1.231 | 0.624 | 0.087 | yes | act | – |
| `needs_graph` | noul | logreg | sigmoid | 182 | 0.183 / 0.431 | 0.565 | 0.037 | no | escalate | – |
| `chunk_relevant` | noul | logreg | sigmoid | 744 | 0.351 / 0.617 | 0.484 | 0.020 | yes | act | – |
| `answerable` | noul | logreg | sigmoid | 281 | 0.575 / 0.698 | 0.223 | 0.081 | yes | confirm | – |
| `grounded` | noul | logreg | sigmoid | 316 | 0.198 / 0.687 | 0.781 | 0.055 | yes | confirm | – |
| `complete` | score | logreg | sigmoid | 316 | 0.533 / 0.832 | 0.313 | 0.087 | yes | act | – |
| `risk_tier` | score | logreg | sigmoid | 316 | 0.441 / 1.230 | 0.684 | 0.050 | yes | confirm | – |

### `response_type`

What kind of response does this request need? Judge the request itself, not whether the documents contain the answer: a question about a product that does not exist is still a direct_answer (the gatekeeper `answerable` catches it). Pick clarify_question only when the request does not name what it is about.

Rows 900 (trusted 900; by source {"synthetic": 900}), ordering `group`, holdout 182 trusted rows, groups 285.

Skill test: block bootstrap (56 groups, 2000 resamples): one-sided 95% lower bound of the mean Brier gain over the base rate; and the same for the gain over the holdout's own class frequencies (recomputed in every resample), which no constant forecast can beat; mean Brier gain 0.5921, lower 95% 0.4615 -> **skill**.

* low stakes: **act** — holdout: 144 act-tier rows at low stakes, accuracy 0.930556 (95% Wilson 0.876891-0.961845) against a mean prediction of 0.864257
* medium stakes: **act** — holdout: 95 act-tier rows at medium stakes, accuracy 0.989474 (95% Wilson 0.942761-0.998139) against a mean prediction of 0.9093
* high stakes: **confirm** — only 5 act-tier rows at high stakes on the holdout (need 30), so acting is not verified; decisions are capped at confirm

Reliability (top-label confidence, holdout):

| bin | mean predicted | observed | rows |
|---|---|---|---|
| 0.4–0.5 | 0.448 | 0.833 | 6 |
| 0.5–0.6 | 0.552 | 0.545 | 22 |
| 0.6–0.7 | 0.655 | 0.600 | 10 |
| 0.7–0.8 | 0.746 | 0.806 | 31 |
| 0.8–0.9 | 0.863 | 0.944 | 54 |
| 0.9–1.0 | 0.928 | 0.983 | 59 |

Notes:
* groups are ordered by their first timestamp and the holdout is the latest groups; development keeps every row of the earlier groups (also rows dated inside the holdout period) and no label-window purge applies between groups. If rows of different groups share time context (a market, a season, a policy change), set `train.ordering: time`
* the holdout's class mix differs strongly from training (clarify_question: training 0.065, holdout 0.011): its metrics describe that mix. Check the group / time split, or add trusted rows of the rare class

Classes: direct_answer, extract_fields, summarize, compare, list, clarify_question, escalate_human, out_of_scope.

### `route`

Which retrieval route should answer this request? vector for a single-document lookup, graph for multi-hop relations between entities, hybrid for comparisons and lists that gather several documents, none when no retrieval should run at all.

Rows 900 (trusted 900; by source {"synthetic": 900}), ordering `group`, holdout 182 trusted rows, groups 285.

Skill test: block bootstrap (56 groups, 2000 resamples): one-sided 95% lower bound of the mean Brier gain over the base rate; and the same for the gain over the holdout's own class frequencies (recomputed in every resample), which no constant forecast can beat; mean Brier gain 0.4102, lower 95% 0.3140 -> **skill**.

* low stakes: **act** — holdout: 125 act-tier rows at low stakes, accuracy 0.936 (95% Wilson 0.878785-0.967216) against a mean prediction of 0.896674
* medium stakes: **act** — holdout: 93 act-tier rows at medium stakes, accuracy 0.989247 (95% Wilson 0.941581-0.998099) against a mean prediction of 0.934199
* high stakes: **act** — holdout: 36 act-tier rows at high stakes, accuracy 1.0 (95% Wilson 0.903581-1.0) against a mean prediction of 0.978285

Reliability (top-label confidence, holdout):

| bin | mean predicted | observed | rows |
|---|---|---|---|
| 0.3–0.4 | 0.351 | 0.500 | 2 |
| 0.4–0.5 | 0.455 | 0.714 | 14 |
| 0.5–0.6 | 0.555 | 0.500 | 18 |
| 0.6–0.7 | 0.649 | 0.565 | 23 |
| 0.7–0.8 | 0.754 | 0.625 | 16 |
| 0.8–0.9 | 0.853 | 0.950 | 40 |
| 0.9–1.0 | 0.955 | 1.000 | 69 |

Notes:
* groups are ordered by their first timestamp and the holdout is the latest groups; development keeps every row of the earlier groups (also rows dated inside the holdout period) and no label-window purge applies between groups. If rows of different groups share time context (a market, a season, a policy change), set `train.ordering: time`

Classes: vector, graph, hybrid, none.

### `needs_graph`

Does answering this request need multi-hop traversal across linked documents or entities?

Rows 900 (trusted 900; by source {"synthetic": 900}), ordering `group`, holdout 182 trusted rows, groups 285.

Skill test: block bootstrap (56 groups, 2000 resamples): one-sided 95% lower bound of the mean Brier gain over the base rate; and the same for the gain over the holdout's own class frequencies (recomputed in every resample), which no constant forecast can beat; mean Brier gain 0.1474, lower 95% 0.0134 -> **no skill**.

* low stakes: **escalate** — the holdout shows no skill over the base rate: every decision abstains until a retrain shows skill (the gain over the holdout's own class frequencies averages 0.146652 but is not reliably positive across the 56 blocks resampled (95% lower bound -0.004235): too few independent groups or time blocks to establish it; more trusted rows from more groups would settle it)
* medium stakes: **escalate** — the holdout shows no skill over the base rate: every decision abstains until a retrain shows skill (the gain over the holdout's own class frequencies averages 0.146652 but is not reliably positive across the 56 blocks resampled (95% lower bound -0.004235): too few independent groups or time blocks to establish it; more trusted rows from more groups would settle it)
* high stakes: **escalate** — the holdout shows no skill over the base rate: every decision abstains until a retrain shows skill (the gain over the holdout's own class frequencies averages 0.146652 but is not reliably positive across the 56 blocks resampled (95% lower bound -0.004235): too few independent groups or time blocks to establish it; more trusted rows from more groups would settle it)

Reliability (P(true), holdout):

| bin | mean predicted | observed | rows |
|---|---|---|---|
| 0.0–0.1 | 0.017 | 0.007 | 135 |
| 0.1–0.2 | 0.141 | 0.273 | 11 |
| 0.2–0.3 | 0.266 | 0.333 | 3 |
| 0.3–0.4 | 0.336 | 0.167 | 6 |
| 0.4–0.5 | 0.446 | 1.000 | 3 |
| 0.5–0.6 | 0.545 | 0.667 | 3 |
| 0.6–0.7 | 0.605 | 1.000 | 2 |
| 0.7–0.8 | 0.757 | 0.889 | 9 |
| 0.8–0.9 | 0.848 | 0.750 | 8 |
| 0.9–1.0 | 0.918 | 0.500 | 2 |

Notes:
* groups are ordered by their first timestamp and the holdout is the latest groups; development keeps every row of the earlier groups (also rows dated inside the holdout period) and no label-window purge applies between groups. If rows of different groups share time context (a market, a season, a policy change), set `train.ordering: time`

### `chunk_relevant`

Does this retrieved chunk contain information needed to answer the request?

Rows 3658 (trusted 3658; by source {"synthetic": 3658}), ordering `group`, holdout 744 trusted rows, groups 262.

Skill test: block bootstrap (51 groups, 2000 resamples): one-sided 95% lower bound of the mean Brier gain over the base rate; and the same for the gain over the holdout's own class frequencies (recomputed in every resample), which no constant forecast can beat; mean Brier gain 0.2055, lower 95% 0.1766 -> **skill**.

* low stakes: **act** — holdout: 580 act-tier rows at low stakes, accuracy 0.917241 (95% Wilson 0.89198-0.937013) against a mean prediction of 0.907682
* medium stakes: **act** — holdout: 440 act-tier rows at medium stakes, accuracy 0.938636 (95% Wilson 0.912192-0.957488) against a mean prediction of 0.946326
* high stakes: **act** — holdout: 240 act-tier rows at high stakes, accuracy 0.983333 (95% Wilson 0.957938-0.9935) against a mean prediction of 0.979171

Reliability (P(true), holdout):

| bin | mean predicted | observed | rows |
|---|---|---|---|
| 0.0–0.1 | 0.037 | 0.034 | 320 |
| 0.1–0.2 | 0.147 | 0.167 | 102 |
| 0.2–0.3 | 0.243 | 0.254 | 59 |
| 0.3–0.4 | 0.351 | 0.433 | 60 |
| 0.4–0.5 | 0.443 | 0.576 | 33 |
| 0.5–0.6 | 0.546 | 0.414 | 29 |
| 0.6–0.7 | 0.661 | 0.762 | 42 |
| 0.7–0.8 | 0.757 | 1.000 | 22 |
| 0.8–0.9 | 0.855 | 0.944 | 36 |
| 0.9–1.0 | 0.959 | 0.927 | 41 |

Notes:
* groups are ordered by their first timestamp and the holdout is the latest groups; development keeps every row of the earlier groups (also rows dated inside the holdout period) and no label-window purge applies between groups. If rows of different groups share time context (a market, a season, a policy change), set `train.ordering: time`

### `answerable`

Is the evidence sufficient to answer the request fully and correctly?

Rows 1359 (trusted 1359; by source {"synthetic": 1359}), ordering `group`, holdout 281 trusted rows, groups 264.

Skill test: block bootstrap (52 groups, 2000 resamples): one-sided 95% lower bound of the mean Brier gain over the base rate; and the same for the gain over the holdout's own class frequencies (recomputed in every resample), which no constant forecast can beat; mean Brier gain 0.1126, lower 95% 0.0692 -> **skill**.

* low stakes: **confirm** — act-tier rows at low stakes came true 0.725 of the time against a mean prediction of 0.834515 (one-sided p = 0.009333): the probabilities overstate; decisions are capped at confirm
* medium stakes: **confirm** — conformal: no threshold keeps the error rate <= 0.05 at confidence 0.9 on 1078 calibration rows; decisions are capped at confirm
* high stakes: **confirm** — only 10 act-tier rows at high stakes on the holdout (need 30), so acting is not verified; decisions are capped at confirm

Reliability (P(true), holdout):

| bin | mean predicted | observed | rows |
|---|---|---|---|
| 0.0–0.1 | 0.066 | 0.000 | 16 |
| 0.1–0.2 | 0.155 | 0.195 | 41 |
| 0.2–0.3 | 0.246 | 0.290 | 31 |
| 0.3–0.4 | 0.347 | 0.280 | 25 |
| 0.4–0.5 | 0.443 | 0.688 | 32 |
| 0.5–0.6 | 0.559 | 0.586 | 29 |
| 0.6–0.7 | 0.657 | 0.778 | 27 |
| 0.7–0.8 | 0.751 | 0.629 | 35 |
| 0.8–0.9 | 0.853 | 0.708 | 24 |
| 0.9–1.0 | 0.952 | 0.905 | 21 |

Conformal promise: error <= 0.05 at confidence 0.9 for P(true) >= – (no threshold keeps the error rate <= 0.05 at confidence 0.9 on 1078 calibration rows); holdout: error – on 0 rows, coverage –.

Notes:
* groups are ordered by their first timestamp and the holdout is the latest groups; development keeps every row of the earlier groups (also rows dated inside the holdout period) and no label-window purge applies between groups. If rows of different groups share time context (a market, a season, a policy change), set `train.ordering: time`

### `grounded`

Is every factual claim in the drafted answer supported by the evidence?

Rows 1576 (trusted 1576; by source {"synthetic": 1576}), ordering `group`, holdout 316 trusted rows, groups 208.

Skill test: block bootstrap (43 groups, 2000 resamples): one-sided 95% lower bound of the mean Brier gain over the base rate; and the same for the gain over the holdout's own class frequencies (recomputed in every resample), which no constant forecast can beat; mean Brier gain 0.3852, lower 95% 0.3589 -> **skill**.

* low stakes: **confirm** — conformal: no threshold keeps the error rate <= 0.05 at confidence 0.9 on 1260 calibration rows; decisions are capped at confirm
* medium stakes: **confirm** — conformal: no threshold keeps the error rate <= 0.05 at confidence 0.9 on 1260 calibration rows; decisions are capped at confirm
* high stakes: **confirm** — conformal: no threshold keeps the error rate <= 0.05 at confidence 0.9 on 1260 calibration rows; decisions are capped at confirm

Reliability (P(true), holdout):

| bin | mean predicted | observed | rows |
|---|---|---|---|
| 0.0–0.1 | 0.017 | 0.012 | 83 |
| 0.1–0.2 | 0.148 | 0.000 | 17 |
| 0.2–0.3 | 0.253 | 0.077 | 13 |
| 0.3–0.4 | 0.364 | 0.273 | 11 |
| 0.4–0.5 | 0.455 | 0.333 | 6 |
| 0.5–0.6 | 0.546 | 0.500 | 10 |
| 0.6–0.7 | 0.643 | 0.364 | 11 |
| 0.7–0.8 | 0.756 | 0.905 | 21 |
| 0.8–0.9 | 0.846 | 0.968 | 31 |
| 0.9–1.0 | 0.964 | 0.991 | 113 |

Conformal promise: error <= 0.05 at confidence 0.9 for P(true) >= – (no threshold keeps the error rate <= 0.05 at confidence 0.9 on 1260 calibration rows); holdout: error – on 0 rows, coverage –.

Notes:
* groups are ordered by their first timestamp and the holdout is the latest groups; development keeps every row of the earlier groups (also rows dated inside the holdout period) and no label-window purge applies between groups. If rows of different groups share time context (a market, a season, a policy change), set `train.ordering: time`

### `complete`

How much of the request does the drafted answer address? Judge coverage of what was asked, not correctness (a wrong number that answers the question still counts as answering it; `grounded` judges correctness). Example: 1 of 3 requested fields -> 1, all of them -> 3. For a multi-hop question ("who leads the team that owns X?") the intermediate link alone (the owning team) misses the question -> 0, while the asked fact (the lead) answers it -> 3.

Rows 1576 (trusted 1576; by source {"synthetic": 1576}), ordering `group`, holdout 316 trusted rows, groups 208.

Skill test: block bootstrap (43 groups, 2000 resamples): one-sided 95% lower bound of the mean Brier gain over the base rate; and the same for the gain over the holdout's own class frequencies (recomputed in every resample), which no constant forecast can beat; mean Brier gain 0.1282, lower 95% 0.1009 -> **skill**.

* low stakes: **act** — holdout: 267 act-tier rows at low stakes, accuracy 0.857678 (95% Wilson 0.810688-0.894522) against a mean prediction of 0.891442
* medium stakes: **act** — holdout: 214 act-tier rows at medium stakes, accuracy 0.897196 (95% Wilson 0.849259-0.931126) against a mean prediction of 0.916262
* high stakes: **act** — holdout: 32 act-tier rows at high stakes, accuracy 0.96875 (95% Wilson 0.842557-0.994462) against a mean prediction of 0.96292

Reliability (top-label confidence, holdout):

| bin | mean predicted | observed | rows |
|---|---|---|---|
| 0.4–0.5 | 0.453 | 0.500 | 6 |
| 0.5–0.6 | 0.544 | 0.833 | 24 |
| 0.6–0.7 | 0.646 | 0.737 | 19 |
| 0.7–0.8 | 0.762 | 0.828 | 29 |
| 0.8–0.9 | 0.867 | 0.717 | 92 |
| 0.9–1.0 | 0.933 | 0.952 | 146 |

Notes:
* groups are ordered by their first timestamp and the holdout is the latest groups; development keeps every row of the earlier groups (also rows dated inside the holdout period) and no label-window purge applies between groups. If rows of different groups share time context (a market, a season, a policy change), set `train.ordering: time`

Classes: 0, 1, 2, 3.

### `risk_tier`

How much harm could a wrong answer to this request cause? Judge the topic of the request and answer: product specifications are harmless, internal organisation facts low, prices / refunds / warranties / personal data elevated, legal, medical and safety matters high.

Rows 1576 (trusted 1576; by source {"synthetic": 1576}), ordering `group`, holdout 316 trusted rows, groups 208.

Skill test: block bootstrap (43 groups, 2000 resamples): one-sided 95% lower bound of the mean Brier gain over the base rate; and the same for the gain over the holdout's own class frequencies (recomputed in every resample), which no constant forecast can beat; mean Brier gain 0.4591, lower 95% 0.3572 -> **skill**.

* low stakes: **confirm** — act-tier rows at low stakes came true 0.902527 of the time against a mean prediction of 0.948354 (one-sided p = 0.001296): the probabilities overstate; decisions are capped at confirm
* medium stakes: **confirm** — act-tier rows at medium stakes came true 0.937759 of the time against a mean prediction of 0.973057 (one-sided p = 0.002527): the probabilities overstate; decisions are capped at confirm
* high stakes: **confirm** — act-tier rows at high stakes came true 0.969231 of the time against a mean prediction of 0.989536 (one-sided p = 0.017478): the probabilities overstate; decisions are capped at confirm

Reliability (top-label confidence, holdout):

| bin | mean predicted | observed | rows |
|---|---|---|---|
| 0.3–0.4 | 0.383 | 1.000 | 1 |
| 0.4–0.5 | 0.467 | 0.385 | 13 |
| 0.5–0.6 | 0.548 | 0.571 | 14 |
| 0.6–0.7 | 0.639 | 0.545 | 11 |
| 0.7–0.8 | 0.746 | 0.632 | 19 |
| 0.8–0.9 | 0.853 | 0.763 | 38 |
| 0.9–1.0 | 0.982 | 0.950 | 220 |

Notes:
* groups are ordered by their first timestamp and the holdout is the latest groups; development keeps every row of the earlier groups (also rows dated inside the holdout period) and no label-window purge applies between groups. If rows of different groups share time context (a market, a season, a policy change), set `train.ordering: time`

Classes: 0, 1, 2, 3.

## Policy

Abstain action: `escalate`. Thresholds on the calibrated probability of the committed class:

| stakes | act | confirm | min accuracy (Bernoulli gate) |
|---|---|---|---|
| low | 0.7 | 0.5 | 0.6 |
| medium | 0.85 | 0.65 | 0.75 |
| high | 0.95 | 0.8 | 0.9 |

## Known limits

* Calibration holds for data like the training data. A shift in traffic (new products, new document types, a new language) needs fresh trusted labels and a retrain; `/v1/calibration` will show it.
* The holdout is small when trusted labels are few: read the Wilson intervals, not just the means.
* Feedback rows are biased towards decisions that were acted on; they train, they never calibrate.
* Conformal promises assume exchangeable data (no drift between calibration and use).
