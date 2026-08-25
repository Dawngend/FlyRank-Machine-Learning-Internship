# Capstone Report — Freestyle: AI Referral and GEO Opportunity Scoring

- **Author:** Dawn Andrei Pamesa
- **Lane:** Freestyle, AI Referral and Generative Engine Optimization (GEO) Opportunity Scoring
- **Repo:** https://github.com/Dawngend/FlyRank-Machine-Learning-Internship
- **Date:** 2026-08-25

---

## 0. Abstract

Which pages should a content team review first, when there are more pages than review hours? This
work ranks 30,000 pseudonymized content items across 32 clients, using a 90-day performance window
from the FlyRank ML Internship starter release, against a label derived from the 30-day-over-previous-30-day
impression change. A transparent hand rule was built first, then compared against logistic regression
and a random forest under a client-grouped 5-fold split, with a leakage audit run on the final
feature set. The random forest ranked declining pages at precision@1000 of 0.726 (standard deviation
0.054) against a base rate of 0.542, ahead of the hand rule's 0.558 on all five held-out client folds;
the audit also found that two shipped columns reconstruct the label exactly, and that a naive
row-level split would have overstated the top-of-queue result by 22.7%. The output is a 1,000-page
review queue with reason codes, intended to order human review of content refresh candidates, and it
is not evidence that refreshing any page recovers traffic.

---

## 1. Problem framing

**The decision.** A content team has more pages than review capacity. Somebody has to choose which
ones a human opens first this cycle. Today that order is set by whoever asked most recently, or by
whichever client is loudest. This work replaces that ordering with a measured one.

- **Unit of analysis:** one pseudonymized content item (`content_id`), evaluated over a trailing
  90-day window.
- **Output:** a ranked queue with a probability score, a confidence band, and reason codes.
- **Action a human takes:** opens the page, checks the four review gates in section 7, and decides
  whether a refresh is the right instrument.
- **Cost of a wrong call:** a false positive costs roughly three to five hours of editorial time on a
  page that was fine. A false negative means a page with existing visibility keeps decaying until
  recovering it is expensive. The costs are not symmetric, which is why the system orders review and
  never triggers an edit.

**Why data helps here.** The base rate of declining pages in this corpus is 0.542, so an arbitrary
review order finds a declining page about half the time. That is the number any model has to beat,
and it is quoted next to every result in this report for exactly that reason.

### The scope change, stated up front

The lane was declared in Week 1 as **AI Referral and GEO Opportunity Scoring**: finding pages with
strong search demand but disproportionately low AI referral traffic. The modelled deliverable is
narrower than that, and the reason is worth stating plainly rather than hiding in a limitations
section.

The AI visibility gap is real and measurable in this corpus. 16,726 items (55.75%) have organic demand
of at least 500 impressions, and 15,033 of those (89.88%) receive zero AI referral traffic. What does
not exist is a **label** for it. Nothing in the release records whether an AI engine later cited a
page, so "AI visibility gap" can be described and ranked by rule, but it cannot be learned, validated,
or checked for leakage. Building a supervised model on it would have meant inventing a target.

The decay half of the same lane does have a label, so that is the half that carries the modelling.
Week 1 framed both as "two pressures on the same inventory," and this report delivers a validated
model for one of them and a descriptive result for the other. Treating the missing label as a finding
rather than working around it is the honest version of this project.

---

## 2. Data safety

**Source.** `data/raw/content_refresh_anonymized.csv` from the FlyRank ML Internship starter release:
30,000 content items, 44 columns, 32 pseudonymized clients, 90-day performance window. Warehouse
figures quoted in Week 1 come from the gated `FlyRank/internship-warehouse` release, partition
`month=2026-03`.

**Excluded on purpose.**

| Field | Why |
|---|---|
| `trend_direction`, `trend_pct` | The label is derived from them. Structurally dropped before any scorer sees the frame. |
| `is_declining_label` | The label itself. |
| `impressions_last_30d`, `impressions_prev_30d` | Found during the ML-09 audit: their ratio reconstructs the label exactly. See section 5. |
| `provider_used`, `model_used` | Excluded by the ML-04 data contract as out of scope for a refresh-prioritisation decision. |
| `client_id` | Used for **grouping only**, never as a feature. |

**Leakage risks considered, and what was actually found.** Week 3 excluded the label-derived fields by
name. That was correct and insufficient. The Week 6 audit reconstructed the label from first
principles and found that `impressions_last_30d` and `impressions_prev_30d`, both shipped in the
feature vector, rebuild `is_declining_label` on 30,000 of 30,000 rows. Neither column looks suspicious
alone (single-feature ROC-AUC of 0.486 and 0.621); their ratio scores 1.000. A one-feature-at-a-time
scan cannot see a leak that lives in an interaction. Neither column is in the model, and the exclusion
is now enforced by an assertion in the notebook rather than by memory.

**Client-identifying content.** None. Clients appear only as pseudonymized `client_id` values used for
fold assignment. The delivered queue CSV carries no label column and no client name. Reason codes are
generic categories.

---

## 3. Baseline

A transparent hand rule (`work/scripts/baseline_action_score.py`), written before any model, scoring
each page 0 to 100 from five stated rules with per-reason point contributions, so any row's position
is traceable to one sentence.

| Reason code | Points | Fires when |
|---|---:|---|
| `stale_visible_page` | 30 | not updated in 180+ days and 500+ impressions |
| `thin_visible_page` | 25 | under 1,200 words and 250+ impressions |
| `page_one_decay_risk` | 20 | position 1 to 10 and 180+ days old |
| `low_ctr_visible_page` | 15 | 500+ impressions, position within 20, CTR under 0.5 |
| `low_engagement_visible_page` | 10 | 30+ sessions with weak engagement or scroll |

**Why it is a fair comparison.** The rule is scored on the identical test rows of each fold, under the
identical metric, rather than against the whole-corpus figure quoted in Week 4. Comparing a
fold-evaluated model against a whole-corpus baseline would have flattered the model.

**Its numbers, on the same split as everything else:** precision@50 of 0.616 (sd 0.136), precision@1000
of 0.558 (sd 0.067), ROC-AUC 0.539, against a base rate of 0.542. The rule orders the first hundred
rows and then collapses to chance.

---

## 4. Model and analysis

**Method.** Logistic regression (readable) then random forest (stronger), plus a depth-2 decision tree
included purely so its decision surface can be printed and read. The lane asks "which first", which is
a ranking question, so every method is scored on its predicted probability at precision@K rather than
on hard labels. Accuracy appears nowhere in this project: a queue is never consumed by thresholding at
0.5, it is consumed from the top down until the reviewer runs out of time.

**Features.** 18 numeric and 8 categorical, all describing observed visibility, content shape, and
demand: impressions, clicks, sessions and AI sessions (log-scaled), days with impressions and
sessions, content age, days since last update, CTR, average position, engagement and scroll rate, AI
traffic share, word and character count, search volume, competition, CPC, and the tier encodings for
age, freshness, word count, impressions, and position.

**Left out on purpose:** every field in section 2's exclusion table.

**Target.** `is_declining_label`, true when a page's impressions in the last 30 days fell more than 20%
against the previous 30 days.

**A documentation discrepancy worth recording.** The FlyRank research paper documents this cut at 10%.
The shipped data cuts at 20%: reconstructing the label at a 10% threshold agrees on 93.3% of rows,
at 20% it agrees on 100%. Anything built on the documented figure will not reproduce. This is flagged
rather than silently corrected.

---

## 5. Evaluation

**Split.** `GroupKFold(n_splits=5)` on `client_id`. No client appears in both train and test.

**Why grouping is a correctness issue here, not a preference.** There are 32 clients across 30,000
rows, one holds 12.3% of the corpus, and the per-client declining rate runs from 0.000 to 0.937
against a corpus base rate of 0.542. Under a row-level shuffle the same client sits on both sides, so
a model can score well by inferring which client a row belongs to and predicting that client's base
rate. It would look like a content model and behave like a client lookup table.

**A time-aware split is not available.** The starter release is a single undated snapshot. The
contract's sealed test month (2026-06) is the design to run once the dated warehouse release is in
hand, and it stays in the contract as future work rather than being claimed here.

### Results on the same split

| Scorer | ROC-AUC | precision@50 | precision@1000 |
|---|---:|---:|---:|
| Base rate | 0.500 | 0.542 | 0.542 |
| ML-07 hand rule | 0.539 | 0.616 (sd 0.136) | 0.558 (sd 0.067) |
| Logistic regression | 0.660 | 0.788 (sd 0.095) | 0.710 (sd 0.084) |
| **Random forest** | **0.671** | **0.776 (sd 0.088)** | **0.726 (sd 0.054)** |
| Decision tree, depth 2 | 0.608 | 0.628 (sd 0.077) | 0.613 (sd 0.074) |

**What the split was worth.** The same models under a naive `StratifiedKFold` row shuffle:

| Scorer | precision@50, grouped | precision@50, naive | Inflation |
|---|---:|---:|---:|
| ML-07 hand rule | 0.616 | 0.556 | −9.7% |
| Decision tree, depth 2 | 0.628 | 0.640 | +1.9% |
| Logistic regression | 0.788 | 0.896 | +13.7% |
| Random forest | 0.776 | 0.952 | +22.7% |

The inflation is monotone in model capacity. The hand rule is never fitted and so cannot benefit at
all, which sets the scale for what "no leak" looks like; each step up in capacity buys more inflation.
That ordering is the signature of client identity leaking through the split rather than of a better
model. A shuffled-label null under the grouped design scores ROC-AUC 0.5016 (sd 0.0047), confirming
the harness measures the label and not the folds.

### Error analysis

The forest's ROC-AUC of 0.671 means it separates the full corpus only modestly: roughly a third of
pairwise orderings are wrong. Precision@1000 of 0.726 means it identifies the top of the corpus well.
Both are true, and only the second is used.

The errors concentrate by client. Fold metrics swing by up to 0.20 depending on which clients land in
the test fold, so performance is markedly client-dependent and there is no basis for claiming it
generalises evenly to a new client. The dominant client makes this worse, since folds containing it
are not comparable to folds that do not.

At the top of the queue the advantage over the rule is directional rather than consistent: the forest
leads the rule at precision@1000 on 5 of 5 folds, but at precision@50 on only 4 of 5. On the fifth
fold the hand rule wins, 0.76 to 0.62.

---

## 6. Interpretation

**The model ranks on sustained visibility and age, not on content shape.** The forest's top four
features (`days_with_impressions` 0.130, `log_impressions_90d` 0.130, `avg_position` 0.108,
`content_age_days` 0.084) account for roughly 45% of total importance. Word count and character count
sit well behind at 0.046 and 0.045.

**The signal the hand rule weighted highest is nearly worthless to the model.** `days_since_last_update`
carried 30 points in the rule, its largest single weight, on the reasoning that a page nobody has
touched in six months is the most urgent case. The forest ranks it 12th at 0.023 importance, below
scroll rate. Two independent lines of evidence agree that the staleness rule was a bad bet: it fired
on 17 of 30,000 rows, and the model does not use it when free to. **Recency of editing appears not to
carry information about whether a page is declining, at least in this corpus.** That is a negative
result and it is one of the more useful things this project found.

**The depth-2 tree makes the same point in one readable line:**

> `days_with_impressions > 12` and `content_age_days <= 373` → at risk

Its entire learned rule is "a page that shows up consistently and is not yet very old is the one at
risk". Nothing about word count, CTR, or edit recency. It is a weak model (precision@100 of 0.608) but
a legible one, and it agrees with the forest about where to look.

**The two scorers disagree about what matters, and measuring that is a result.** When the hand rule's
reason codes were attached to the forest's queue, they explained only **55.6%** of it. The rule fires
on thinness, staleness, and weak CTR; the forest ranks on sustained visibility and age, for which the
rule has no reason code at all. That gap is not a bug in either scorer. It is the clearest single
statement of what the model learned that the hand rule did not.

**Surprises.** Two. First, the leak described in section 2 was invisible to the correlation-based scan
used in Week 3 and only appeared when the label was rebuilt from its definition. Second, logistic
regression matches the forest at the top of the queue (0.788 against 0.776); the forest's advantage is
entirely at depth. If this queue were only ever worked 100 rows deep, the readable model would be the
better deliverable and the forest would be unnecessary complexity.

---

## 7. Recommendation

**Shipped:** the random forest, out-of-fold scored, delivering the top 1,000 of 26,612 eligible pages.
The operating assumption behind that choice is that the queue is worked 500 to 1,000 pages per cycle,
which is where the forest's advantage is consistent. If a cycle only ever reaches 100 pages, switch to
logistic regression and the queue becomes fully explainable.

**What the queue delivers.** Working the top 1,000 in this order surfaces 778 declining pages against
roughly 542 expected from an arbitrary order: about 236 fewer wasted reviews per thousand, a lift of
1.44x over the base rate.

**How an editor uses it tomorrow.** Open `work/outputs/refresh_action_queue.csv`, work top down, and
run four checks on each page before editing anything:

1. **Is the decline real, or is it the measurement?** The label is one 30-day window against the
   previous one. A seasonal dip, a SERP feature change, or a tracking gap all read as declining.
2. **Does the reason code match what you see?** The `explanation_source` column says whether the
   justification came from the hand rule or from the model. If it says `model`, the rule found nothing
   wrong with the page: read that as "the model noticed a pattern", not as a diagnosis.
3. **Is a refresh the right instrument?** A page can decline because the topic died, because a
   competitor published something better, or because another page cannibalised it. Only one of those
   is fixed by refreshing, and the model has no feature for any of them.
4. **Would you defend this edit to the client?** "It was ranked 12th" is not a justification.

**Never automated, at any confidence score:** publishing an edit; deleting, de-indexing, or
redirecting a page; client-facing performance promises; budget allocation between clients; bulk
machine regeneration of flagged pages.

**Confidence.** The `confidence` column is **not** a calibrated probability. The bands are tertiles of
this queue's own score distribution, so "high" means high relative to the other 999 rows in front of
you and nothing more. The forest's probabilities were never calibrated and no calibration is claimed.

**Monitoring.** Six triggers ship with the queue, each anchored to a measured value rather than a
round number, using the 0.054 fold spread as the scale of ordinary variation. The two that matter
most: queue precision falling below 0.670 (retrain), and any change to the 20% label threshold, the
window, or the metric (stop and rebuild, because every number in this report was computed against that
cut).

---

## 8. Reproducibility

From a fresh clone:

```bash
pip install -r requirements.txt
python scripts/run_all.py
python work/scripts/baseline_action_score.py
python work/scripts/train_refresh_model.py
python work/scripts/validation_audit.py
python work/scripts/action_playbook.py
```

**Seed:** 42, fixed in every script that samples, splits, or permutes. The hand rule is deterministic
and needs none.

**Environment:** `requirements.txt` (pandas, numpy, scikit-learn, matplotlib, reportlab, duckdb,
huggingface_hub). The random forest figure is library-version sensitive at the third decimal; the
stable claim is the lift over the baseline, not the exact value.

**Committed receipts.** Every number in this report traces to a JSON file in the repo, not to a
notebook cell that has to be re-run to be believed:

| File | What it backs |
|---|---|
| `work/outputs/data_contract.json` | field selection, windows, exclusions |
| `work/outputs/baseline_action_score_metrics.json` | section 3 |
| `work/outputs/model_comparison.json` | section 5's results table |
| `work/outputs/validation_audit.json` | the leakage findings and the split comparison |
| `work/outputs/action_playbook_metrics.json` | the queue, its quality, and the monitoring triggers |

**On the sealed evaluation.** This project does **not** claim one. The contract defines a sealed test
month (2026-06), and the starter release is undated, so the design is recorded and not executed. Every
figure here is cross-validated under client grouping, which is stated wherever a number appears rather
than described as a holdout.

**Notebooks, in order:** `w01_research_question` (ML-02), `w02_ml_task_framing` (ML-03),
`w03_data_contract` (ML-04), `w04_baseline_score` (ML-07), `w05_model` (ML-08),
`w06_validation_audit` (ML-09), `w07_action_playbook` (ML-10), `capstone` (ML-11 and ML-12).

---

## 9. Acknowledgments and data credit

Built on the FlyRank ML Internship dataset, provided by [FlyRank](https://flyrank.ai). The 30,000-row
starter release and the gated warehouse release are both FlyRank's; the analysis, the errors, and the
claims are mine. Data use follows `DATA_USE.md` in this repository: no client-identifying details, no
causal claims without a design, and no dataset files committed to version control.
