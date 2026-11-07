# Model card — rlcd-0.1.1 (documents pack)

Created 2026-11-07T13:39:11Z. Format `open-rlcd-bundle/1`. Feature version `documents-v1:hashing-256-s0:0b1d19ee81e5`.

> **Trained on synthetic data only.** These numbers show that the pipeline works and that the metrics are computed honestly; they say nothing about accuracy on real traffic.

## Intended use

Routing, gatekeeping and evaluation in front of / behind an LLM over an enterprise document store: which response type, which retrieval route, which chunks matter, whether the evidence is enough, and whether a drafted answer is grounded, complete and how risky it is.

## What it is not

* Not a language model: it never writes text. It decides what, whether and from which evidence.
* Not magic: heads take microseconds per row, the text encoder milliseconds; run `rlcd bench`.
* Not better than its labels: weak labels from an LLM judge train the base models, but calibration and every number below use trusted (human-verified, ground-truth or synthetic) rows only.
* A head whose holdout shows no skill over the base rate abstains on every request.

## Data

Label sources (all heads): synthetic 12319

Text encoder: `hashing-256-s0` (dim 256, sha256 `0b1d19ee81e5e56e…`).

## Heads (trusted holdout)

| head | type | model | calibration | holdout rows | log loss / baseline | Brier skill | ECE | skill | verdict | conformal threshold |
|---|---|---|---|---|---|---|---|---|---|---|
| `response_type` | choice | logreg | sigmoid | 180 | 0.536 / 1.905 | 0.711 | 0.076 | yes | act | – |
| `route` | choice | logreg | sigmoid | 180 | 0.451 / 1.211 | 0.603 | 0.067 | yes | act | – |
| `needs_graph` | noul | logreg | sigmoid | 180 | 0.203 / 0.433 | 0.516 | 0.031 | no | escalate | – |
| `chunk_relevant` | noul | logreg | sigmoid | 737 | 0.377 / 0.619 | 0.428 | 0.047 | yes | act | – |
| `answerable` | noul | logreg | sigmoid | 249 | 0.616 / 0.691 | 0.139 | 0.073 | yes | confirm | – |
| `grounded` | noul | logreg | sigmoid | 318 | 0.196 / 0.693 | 0.781 | 0.053 | yes | confirm | – |
| `complete` | score | logreg | sigmoid | 318 | 0.382 / 0.884 | 0.567 | 0.036 | yes | act | – |
| `risk_tier` | score | logreg | sigmoid | 318 | 0.484 / 1.220 | 0.641 | 0.064 | yes | confirm | – |

### `response_type`

What kind of response does this request need? Judge the request itself, not whether the documents contain the answer: a question about a product that does not exist is still a direct_answer (the gatekeeper `answerable` catches it). Pick clarify_question only when the request does not name what it is about.

Rows 900 (trusted 900; by source {"synthetic": 900}), ordering `group`, holdout 180 trusted rows, groups 285.

Skill test: block bootstrap (58 groups, 2000 resamples): one-sided 95% lower bound of the mean Brier gain over the base rate; and the same for the gain over the holdout's own class frequencies (recomputed in every resample), which no constant forecast can beat; mean Brier gain 0.5838, lower 95% 0.4599 -> **skill**.

* low stakes: **act** — holdout: 138 act-tier rows at low stakes, accuracy 0.927536 (95% Wilson 0.871749-0.960166) against a mean prediction of 0.880986
* medium stakes: **act** — holdout: 97 act-tier rows at medium stakes, accuracy 0.958763 (95% Wilson 0.898725-0.983849) against a mean prediction of 0.914816
* high stakes: **confirm** — only 11 act-tier rows at high stakes on the holdout (need 30), so acting is not verified; decisions are capped at confirm

Reliability (top-label confidence, holdout):

| bin | mean predicted | observed | rows |
|---|---|---|---|
| 0.2–0.3 | 0.261 | 1.000 | 1 |
| 0.3–0.4 | 0.317 | 1.000 | 1 |
| 0.4–0.5 | 0.473 | 0.778 | 9 |
| 0.5–0.6 | 0.556 | 0.429 | 14 |
| 0.6–0.7 | 0.659 | 0.647 | 17 |
| 0.7–0.8 | 0.775 | 0.750 | 20 |
| 0.8–0.9 | 0.854 | 0.918 | 49 |
| 0.9–1.0 | 0.931 | 0.986 | 69 |

Notes:
* the holdout's class mix differs strongly from training (clarify_question: training 0.065, holdout 0.011): its metrics describe that mix. Check the group / time split, or add trusted rows of the rare class

Classes: direct_answer, extract_fields, summarize, compare, list, clarify_question, escalate_human, out_of_scope.

### `route`

Which retrieval route should answer this request? vector for a single-document lookup, graph for multi-hop relations between entities, hybrid for comparisons and lists that gather several documents, none when no retrieval should run at all.

Rows 900 (trusted 900; by source {"synthetic": 900}), ordering `group`, holdout 180 trusted rows, groups 285.

Skill test: block bootstrap (58 groups, 2000 resamples): one-sided 95% lower bound of the mean Brier gain over the base rate; and the same for the gain over the holdout's own class frequencies (recomputed in every resample), which no constant forecast can beat; mean Brier gain 0.3907, lower 95% 0.2875 -> **skill**.

* low stakes: **act** — holdout: 143 act-tier rows at low stakes, accuracy 0.881119 (95% Wilson 0.81786-0.924437) against a mean prediction of 0.882098
* medium stakes: **act** — holdout: 97 act-tier rows at medium stakes, accuracy 0.989691 (95% Wilson 0.943895-0.998178) against a mean prediction of 0.933355
* high stakes: **act** — holdout: 36 act-tier rows at high stakes, accuracy 1.0 (95% Wilson 0.903581-1.0) against a mean prediction of 0.97641

Reliability (top-label confidence, holdout):

| bin | mean predicted | observed | rows |
|---|---|---|---|
| 0.3–0.4 | 0.373 | 1.000 | 1 |
| 0.4–0.5 | 0.448 | 0.250 | 4 |
| 0.5–0.6 | 0.540 | 0.667 | 15 |
| 0.6–0.7 | 0.652 | 0.588 | 17 |
| 0.7–0.8 | 0.748 | 0.677 | 31 |
| 0.8–0.9 | 0.857 | 0.811 | 37 |
| 0.9–1.0 | 0.950 | 1.000 | 75 |

Classes: vector, graph, hybrid, none.

### `needs_graph`

Does answering this request need multi-hop traversal across linked documents or entities?

Rows 900 (trusted 900; by source {"synthetic": 900}), ordering `group`, holdout 180 trusted rows, groups 285.

Skill test: block bootstrap (58 groups, 2000 resamples): one-sided 95% lower bound of the mean Brier gain over the base rate; and the same for the gain over the holdout's own class frequencies (recomputed in every resample), which no constant forecast can beat; mean Brier gain 0.1358, lower 95% 0.0114 -> **no skill**.

* low stakes: **escalate** — the holdout shows no skill over the base rate: every decision abstains until a retrain shows skill
* medium stakes: **escalate** — the holdout shows no skill over the base rate: every decision abstains until a retrain shows skill
* high stakes: **escalate** — the holdout shows no skill over the base rate: every decision abstains until a retrain shows skill

Reliability (P(true), holdout):

| bin | mean predicted | observed | rows |
|---|---|---|---|
| 0.0–0.1 | 0.019 | 0.022 | 139 |
| 0.1–0.2 | 0.151 | 0.222 | 9 |
| 0.2–0.3 | 0.238 | 0.333 | 3 |
| 0.3–0.4 | 0.325 | 0.600 | 5 |
| 0.4–0.5 | 0.460 | 0.500 | 2 |
| 0.5–0.6 | 0.578 | 1.000 | 2 |
| 0.6–0.7 | 0.673 | 1.000 | 5 |
| 0.7–0.8 | 0.733 | 0.900 | 10 |
| 0.8–0.9 | 0.852 | 0.400 | 5 |

### `chunk_relevant`

Does this retrieved chunk contain information needed to answer the request?

Rows 3631 (trusted 3631; by source {"synthetic": 3631}), ordering `group`, holdout 737 trusted rows, groups 263.

Skill test: block bootstrap (51 groups, 2000 resamples): one-sided 95% lower bound of the mean Brier gain over the base rate; and the same for the gain over the holdout's own class frequencies (recomputed in every resample), which no constant forecast can beat; mean Brier gain 0.1829, lower 95% 0.1600 -> **skill**.

* low stakes: **act** — holdout: 562 act-tier rows at low stakes, accuracy 0.898577 (95% Wilson 0.87085-0.920891) against a mean prediction of 0.902064
* medium stakes: **act** — holdout: 412 act-tier rows at medium stakes, accuracy 0.946602 (95% Wilson 0.920477-0.964475) against a mean prediction of 0.94654
* high stakes: **act** — holdout: 226 act-tier rows at high stakes, accuracy 0.986726 (95% Wilson 0.961706-0.995475) against a mean prediction of 0.978351

Reliability (P(true), holdout):

| bin | mean predicted | observed | rows |
|---|---|---|---|
| 0.0–0.1 | 0.039 | 0.052 | 308 |
| 0.1–0.2 | 0.139 | 0.102 | 98 |
| 0.2–0.3 | 0.248 | 0.309 | 68 |
| 0.3–0.4 | 0.347 | 0.473 | 55 |
| 0.4–0.5 | 0.453 | 0.667 | 51 |
| 0.5–0.6 | 0.565 | 0.600 | 35 |
| 0.6–0.7 | 0.644 | 0.559 | 34 |
| 0.7–0.8 | 0.749 | 0.731 | 26 |
| 0.8–0.9 | 0.845 | 0.935 | 31 |
| 0.9–1.0 | 0.959 | 0.968 | 31 |

### `answerable`

Is the evidence sufficient to answer the request fully and correctly?

Rows 1221 (trusted 1221; by source {"synthetic": 1221}), ordering `group`, holdout 249 trusted rows, groups 262.

Skill test: block bootstrap (53 groups, 2000 resamples): one-sided 95% lower bound of the mean Brier gain over the base rate; and the same for the gain over the holdout's own class frequencies (recomputed in every resample), which no constant forecast can beat; mean Brier gain 0.0694, lower 95% 0.0306 -> **skill**.

* low stakes: **confirm** — act-tier rows at low stakes came true 0.722222 of the time against a mean prediction of 0.824277 (one-sided p = 0.021227): the probabilities overstate; decisions are capped at confirm
* medium stakes: **confirm** — only 25 act-tier rows at medium stakes on the holdout (need 30), so acting is not verified; decisions are capped at confirm
* high stakes: **confirm** — only 7 act-tier rows at high stakes on the holdout (need 30), so acting is not verified; decisions are capped at confirm

Reliability (P(true), holdout):

| bin | mean predicted | observed | rows |
|---|---|---|---|
| 0.0–0.1 | 0.068 | 0.000 | 14 |
| 0.1–0.2 | 0.157 | 0.056 | 18 |
| 0.2–0.3 | 0.254 | 0.429 | 14 |
| 0.3–0.4 | 0.345 | 0.552 | 29 |
| 0.4–0.5 | 0.455 | 0.618 | 34 |
| 0.5–0.6 | 0.545 | 0.600 | 40 |
| 0.6–0.7 | 0.640 | 0.643 | 28 |
| 0.7–0.8 | 0.757 | 0.667 | 36 |
| 0.8–0.9 | 0.846 | 0.800 | 20 |
| 0.9–1.0 | 0.949 | 0.750 | 16 |

Conformal promise: error <= 0.05 at confidence 0.9 for P(true) >= – (no threshold keeps the error rate <= 0.05 at confidence 0.9 on 972 calibration rows); holdout: error – on 0 rows, coverage –.

### `grounded`

Is every factual claim in the drafted answer supported by the evidence?

Rows 1589 (trusted 1589; by source {"synthetic": 1589}), ordering `group`, holdout 318 trusted rows, groups 208.

Skill test: block bootstrap (40 groups, 2000 resamples): one-sided 95% lower bound of the mean Brier gain over the base rate; and the same for the gain over the holdout's own class frequencies (recomputed in every resample), which no constant forecast can beat; mean Brier gain 0.3900, lower 95% 0.3545 -> **skill**.

* low stakes: **confirm** — conformal: no threshold keeps the error rate <= 0.05 at confidence 0.9 on 1271 calibration rows; decisions are capped at confirm
* medium stakes: **confirm** — conformal: no threshold keeps the error rate <= 0.05 at confidence 0.9 on 1271 calibration rows; decisions are capped at confirm
* high stakes: **confirm** — conformal: no threshold keeps the error rate <= 0.05 at confidence 0.9 on 1271 calibration rows; decisions are capped at confirm

Reliability (P(true), holdout):

| bin | mean predicted | observed | rows |
|---|---|---|---|
| 0.0–0.1 | 0.015 | 0.000 | 85 |
| 0.1–0.2 | 0.142 | 0.000 | 21 |
| 0.2–0.3 | 0.246 | 0.125 | 8 |
| 0.3–0.4 | 0.349 | 0.182 | 11 |
| 0.4–0.5 | 0.459 | 0.200 | 15 |
| 0.5–0.6 | 0.573 | 0.000 | 5 |
| 0.6–0.7 | 0.646 | 0.500 | 16 |
| 0.7–0.8 | 0.762 | 0.882 | 17 |
| 0.8–0.9 | 0.853 | 0.976 | 41 |
| 0.9–1.0 | 0.959 | 0.990 | 99 |

Conformal promise: error <= 0.05 at confidence 0.9 for P(true) >= – (no threshold keeps the error rate <= 0.05 at confidence 0.9 on 1271 calibration rows); holdout: error – on 0 rows, coverage –.

### `complete`

How much of the request does the drafted answer address? Judge coverage of what was asked, not correctness (a wrong number that answers the question still counts as answering it; `grounded` judges correctness). Example: 1 of 3 requested fields -> 1, all of them -> 3.

Rows 1589 (trusted 1589; by source {"synthetic": 1589}), ordering `group`, holdout 318 trusted rows, groups 208.

Skill test: block bootstrap (40 groups, 2000 resamples): one-sided 95% lower bound of the mean Brier gain over the base rate; and the same for the gain over the holdout's own class frequencies (recomputed in every resample), which no constant forecast can beat; mean Brier gain 0.2566, lower 95% 0.2236 -> **skill**.

* low stakes: **act** — holdout: 279 act-tier rows at low stakes, accuracy 0.913978 (95% Wilson 0.875198-0.941514) against a mean prediction of 0.907306
* medium stakes: **act** — holdout: 224 act-tier rows at medium stakes, accuracy 0.928571 (95% Wilson 0.887133-0.955558) against a mean prediction of 0.935388
* high stakes: **act** — holdout: 92 act-tier rows at high stakes, accuracy 0.956522 (95% Wilson 0.893483-0.982964) against a mean prediction of 0.96597

Reliability (top-label confidence, holdout):

| bin | mean predicted | observed | rows |
|---|---|---|---|
| 0.3–0.4 | 0.344 | 1.000 | 1 |
| 0.4–0.5 | 0.467 | 0.375 | 8 |
| 0.5–0.6 | 0.547 | 0.583 | 12 |
| 0.6–0.7 | 0.653 | 0.778 | 18 |
| 0.7–0.8 | 0.756 | 0.815 | 27 |
| 0.8–0.9 | 0.855 | 0.903 | 62 |
| 0.9–1.0 | 0.946 | 0.932 | 190 |

Classes: 0, 1, 2, 3.

### `risk_tier`

How much harm could a wrong answer to this request cause? Judge the topic of the request and answer: product specifications are harmless, internal organisation facts low, prices / refunds / warranties / personal data elevated, legal, medical and safety matters high.

Rows 1589 (trusted 1589; by source {"synthetic": 1589}), ordering `group`, holdout 318 trusted rows, groups 208.

Skill test: block bootstrap (40 groups, 2000 resamples): one-sided 95% lower bound of the mean Brier gain over the base rate; and the same for the gain over the holdout's own class frequencies (recomputed in every resample), which no constant forecast can beat; mean Brier gain 0.4281, lower 95% 0.3342 -> **skill**.

* low stakes: **act** — holdout: 243 act-tier rows at low stakes, accuracy 0.91358 (95% Wilson 0.871505-0.942782) against a mean prediction of 0.928469
* medium stakes: **confirm** — act-tier rows at medium stakes came true 0.926471 of the time against a mean prediction of 0.957074 (one-sided p = 0.030941): the probabilities overstate; decisions are capped at confirm
* high stakes: **confirm** — act-tier rows at high stakes came true 0.939394 of the time against a mean prediction of 0.986668 (one-sided p = 0.000425): the probabilities overstate; decisions are capped at confirm

Reliability (top-label confidence, holdout):

| bin | mean predicted | observed | rows |
|---|---|---|---|
| 0.3–0.4 | 0.371 | 0.500 | 4 |
| 0.4–0.5 | 0.463 | 0.417 | 12 |
| 0.5–0.6 | 0.547 | 0.593 | 27 |
| 0.6–0.7 | 0.649 | 0.719 | 32 |
| 0.7–0.8 | 0.754 | 0.840 | 25 |
| 0.8–0.9 | 0.862 | 0.941 | 51 |
| 0.9–1.0 | 0.975 | 0.916 | 167 |

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
