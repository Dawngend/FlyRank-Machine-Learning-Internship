"""ML-10 content action playbook: the delivered refresh queue and its guardrails.

The operating decision this file encodes, stated up front because everything
else follows from it: **the queue is worked deep, 500-1000 pages per cycle, so
the random forest ships.** ML-08 left the choice open on purpose -- logistic
regression matches the forest in the top 50 (0.788 vs 0.776) and the forest only
pulls ahead with depth (p@1000 0.726 vs 0.710). ML-09 sharpened it: at p@1000
the forest led the ML-07 rule on 5 of 5 held-out client folds; at p@50, on 4 of
5. Depth is where the advantage is consistent, so depth is what the queue is
built for.

That choice has a cost, and the playbook pays it openly: a forest cannot explain
a row. So every delivered row carries the **ML-07 rule's** reason codes as its
human-readable justification, and section 1 of the notebook discloses that the
explanation comes from a different scorer than the ranking. Coverage of that
explanation is measured here, not assumed.

Three design decisions that are not obvious:

  * **Scores are out-of-fold.** Every page is scored by a forest fitted on the
    four client folds that exclude its own client. A model fitted on all 30,000
    rows would score its own training data and the queue's quality metrics would
    be the naive-split number ML-09 retired.
  * **New and flat pages are held out of the queue, not ranked low.** A page
    with impressions_prev_30d == 0 has no prior window, so it cannot be
    declining under the label definition -- all 3,388 of them are labelled 0 by
    construction. Ranking them is not a prediction, it is an artefact.
  * **No label column reaches the delivered CSV.** The reference example export
    ships is_declining_label and trend_direction alongside the queue. For an
    evaluation artefact that is fine; for something a reviewer works from it is
    not, so the labels stay in the metrics receipt and out of the queue.

Seed: 42.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from baseline_action_score import EXCLUDED_FIELDS, LABEL_FIELDS, build
from train_refresh_model import CATEGORICAL, NUMERIC, N_SPLITS, SEED, make_models, precision_at_k

ROOT = Path(__file__).resolve().parents[2]
FEATURE_PATH = ROOT / "data" / "processed" / "refresh_feature_vector.csv"
OUT_CSV = ROOT / "work" / "outputs" / "refresh_action_queue.csv"
OUT_JSON = ROOT / "work" / "outputs" / "action_playbook_metrics.json"

SHIPPED_MODEL = "random_forest"
QUEUE_DEPTH = 1000
# Reported at the depths a cycle might actually reach, per the operating decision.
REVIEW_DEPTHS = (100, 250, 500, 1000)

# Confidence bands. Cut points are the tertiles of the delivered queue's score,
# so the labels describe position within the queue and never imply calibration --
# the forest's probabilities are not calibrated and the playbook does not claim
# they are.
CONFIDENCE_BANDS = ("high", "medium", "low")

# Columns a reviewer needs to act, and nothing else. No label, no trend field.
QUEUE_COLUMNS = [
    "queue_rank", "content_id", "client_id", "model_score", "confidence",
    "suggested_action", "explanation_source", "reason_codes", "model_reason_codes",
    "reason_count", "baseline_action_score",
    "impressions_90d", "clicks_90d", "sessions_90d", "avg_position", "ctr",
    "engagement_rate", "scroll_rate", "word_count", "content_age_days",
    "days_since_last_update", "content_type", "main_intent", "competition_level",
    "position_tier", "age_tier", "freshness_tier",
]

# The ML-07 rule explains only ~56% of what the forest ranks (measured, not
# assumed -- see explainability.rule_reason_coverage_by_depth). The rule fires on
# thinness, staleness and weak CTR; the forest ranks on sustained visibility and
# age, which the rule has no code for. Rather than ship 444 unexplained rows,
# the gap is closed with codes derived from the model's own behaviour, kept in a
# separate column so a reviewer always knows which scorer is talking.
#
# The first code is the depth-2 tree's entire learned rule, converted from
# standardised units back to readable ones (z=-1.499 -> 12 days,
# z=0.881 -> 373 days). ML-08 printed that tree precisely so it could be quoted
# here.
TREE_MIN_DAYS_WITH_IMPRESSIONS = 12
TREE_MAX_CONTENT_AGE_DAYS = 373
MODEL_HIGH_RISK_SCORE = 0.70

MODEL_REASONS = [
    (
        "consistently_visible_not_yet_old",
        lambda d: (d["days_with_impressions"] > TREE_MIN_DAYS_WITH_IMPRESSIONS)
        & (d["content_age_days"] <= TREE_MAX_CONTENT_AGE_DAYS),
    ),
    ("model_high_risk", lambda d: d["model_score"] >= MODEL_HIGH_RISK_SCORE),
    ("visible_with_demand", lambda d: d["impressions_90d"] >= 500),
]

# What to do with a row the rule had nothing to say about.
MODEL_ONLY_ACTION = "review_visibility_trend"


def out_of_fold_scores(X: pd.DataFrame, y: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Score every row with a model that never saw its client."""
    scores = np.full(len(X), np.nan)
    for train_idx, test_idx in GroupKFold(n_splits=N_SPLITS).split(X, y, groups):
        pipe = make_models()[SHIPPED_MODEL]
        pipe.fit(X.iloc[train_idx], y[train_idx])
        scores[test_idx] = pipe.predict_proba(X.iloc[test_idx])[:, 1]
    assert not np.isnan(scores).any(), "every row must be scored exactly once"
    return scores


def queue_quality(y_ranked: np.ndarray, base_rate: float) -> dict:
    """Precision at each depth a cycle might reach, against the base rate."""
    out = {}
    for depth in REVIEW_DEPTHS:
        if len(y_ranked) < depth:
            continue
        precision = float(y_ranked[:depth].mean())
        out[f"p@{depth}"] = {
            "precision": round(precision, 4),
            "lift_vs_base_rate": round(precision / base_rate, 4),
            "declining_pages_found": int(y_ranked[:depth].sum()),
            "wasted_reviews": int(depth - y_ranked[:depth].sum()),
        }
    return out


def run() -> dict:
    frame = pd.read_csv(FEATURE_PATH)
    y = frame["is_declining_label"].to_numpy()
    groups = frame["client_id"].to_numpy()
    X = frame[NUMERIC + CATEGORICAL]

    frame["model_score"] = out_of_fold_scores(X, y, groups)

    # ML-07 reason codes, built on the contract's feature view so the rule still
    # cannot see a label field. Row order is preserved for the join back.
    features_only = frame.drop(
        columns=[c for c in LABEL_FIELDS + EXCLUDED_FIELDS if c in frame.columns]
    )
    rule = (
        build(features_only.assign(_i=range(len(features_only))))
        .sort_values("_i")
        .set_index("_i")
    )
    frame["reason_codes"] = rule["reason_codes"].to_numpy()
    frame["reason_count"] = rule["reason_count"].to_numpy()
    frame["suggested_action"] = rule["suggested_action"].to_numpy()
    frame["baseline_action_score"] = rule["action_score"].to_numpy()

    # --- structural hold-out: pages with no prior window ----------------------
    no_prior_window = frame["impressions_prev_30d"] == 0
    eligible = frame.loc[~no_prior_window].copy()

    ranked = eligible.sort_values(
        ["model_score", "impressions_90d"], ascending=[False, False]
    ).reset_index(drop=True)
    ranked["queue_rank"] = ranked.index + 1

    # --- model-aligned reason codes, in their own column ----------------------
    hits = pd.concat(
        [predicate(ranked).fillna(False).rename(code) for code, predicate in MODEL_REASONS],
        axis=1,
    )
    ranked["model_reason_codes"] = hits.apply(
        lambda row: "|".join(sorted(row.index[row])), axis=1
    )
    rule_explained = ranked["reason_codes"] != "general_refresh_review"
    model_explained = ranked["model_reason_codes"] != ""
    ranked["explanation_source"] = np.select(
        [rule_explained & model_explained, rule_explained, model_explained],
        ["rule+model", "rule", "model"],
        default="none",
    )
    # A row the rule called monitor_only but the forest ranked highly still needs
    # an instruction, and it must not be the rule's instruction.
    ranked.loc[~rule_explained & model_explained, "suggested_action"] = MODEL_ONLY_ACTION

    delivered = ranked.head(QUEUE_DEPTH).copy()
    cuts = delivered["model_score"].quantile([1 / 3, 2 / 3]).to_list()
    delivered["confidence"] = pd.cut(
        delivered["model_score"],
        bins=[-np.inf, cuts[0], cuts[1], np.inf],
        labels=list(reversed(CONFIDENCE_BANDS)),
    ).astype(str)

    delivered[QUEUE_COLUMNS].to_csv(OUT_CSV, index=False)

    base_rate = float(y.mean())
    y_ranked = ranked["is_declining_label"].to_numpy()

    # --- explainability coverage: can anything justify what the forest ranked? -
    has_reason = delivered["reason_codes"] != "general_refresh_review"
    coverage_by_depth = {}
    for depth in REVIEW_DEPTHS:
        if depth > len(ranked):
            continue
        top = ranked.head(depth)
        coverage_by_depth[f"top_{depth}"] = {
            "rule_only_codes": round(
                float((top["reason_codes"] != "general_refresh_review").mean()), 4
            ),
            "any_code": round(float((top["explanation_source"] != "none").mean()), 4),
        }
    unexplained = int((delivered["explanation_source"] == "none").sum())
    assert unexplained == 0, f"{unexplained} delivered rows carry no justification at all"

    # --- concentration: is the queue really one client's problem? -------------
    client_counts = delivered["client_id"].value_counts()
    corpus_share = frame["client_id"].value_counts(normalize=True)

    return {
        "card": "ML-10",
        "seed": SEED,
        "operating_decision": {
            "shipped_model": SHIPPED_MODEL,
            "queue_depth": QUEUE_DEPTH,
            "review_cycle_depth": "500-1000 pages",
            "why": (
                "at p@1000 the forest led the ML-07 rule on 5/5 held-out client folds; "
                "at p@50 on 4/5. The advantage is consistent at depth, so the queue is "
                "built for depth"
            ),
            "cost_accepted": (
                "a forest cannot explain a row, so reason codes are supplied by the "
                "ML-07 rule -- a different scorer than the one that ranked the page"
            ),
            "scoring": "out-of-fold; every page scored by a forest that never saw its client",
        },
        "corpus": {
            "rows": int(len(frame)),
            "base_rate": round(base_rate, 4),
            "clients": int(pd.Series(groups).nunique()),
            "held_out_no_prior_window": int(no_prior_window.sum()),
            "held_out_declining_count": int(y[no_prior_window.to_numpy()].sum()),
            "eligible_for_queue": int(len(eligible)),
        },
        "queue_quality": queue_quality(y_ranked, base_rate),
        "explainability": {
            "finding": (
                "the ML-07 rule explains only about half of what the forest ranks: the "
                "rule fires on thinness, staleness and weak CTR, while the forest ranks "
                "on sustained visibility and age, which the rule has no code for"
            ),
            "coverage_by_depth": coverage_by_depth,
            "delivered_rows_with_a_rule_code": int(has_reason.sum()),
            "delivered_rows_needing_a_model_code": int((~has_reason).sum()),
            "delivered_rows_unexplained": unexplained,
            "explanation_source_mix": delivered["explanation_source"].value_counts().to_dict(),
            "action_mix": delivered["suggested_action"].value_counts().to_dict(),
            "rule_reason_mix": (
                delivered["reason_codes"].str.split("|").explode().value_counts().to_dict()
            ),
            "model_reason_mix": (
                delivered.loc[delivered["model_reason_codes"] != "", "model_reason_codes"]
                .str.split("|").explode().value_counts().to_dict()
            ),
            "confidence_mix": delivered["confidence"].value_counts().to_dict(),
            "depth_2_tree_rule_in_readable_units": (
                f"days_with_impressions > {TREE_MIN_DAYS_WITH_IMPRESSIONS} and "
                f"content_age_days <= {TREE_MAX_CONTENT_AGE_DAYS} -> at risk"
            ),
        },
        "concentration": {
            "clients_in_queue": int(delivered["client_id"].nunique()),
            "largest_client_share_of_queue": round(
                float(client_counts.iloc[0] / len(delivered)), 4
            ),
            "largest_client_share_of_corpus": round(
                float(corpus_share.loc[client_counts.index[0]]), 4
            ),
            "top_5_clients_share_of_queue": round(
                float(client_counts.head(5).sum() / len(delivered)), 4
            ),
        },
        "monitoring_triggers": monitoring_triggers(frame, delivered, base_rate),
        "exports": {
            "queue_csv": str(OUT_CSV.relative_to(ROOT)).replace("\\", "/"),
            "queue_csv_gitignored": True,
            "metrics_json": str(OUT_JSON.relative_to(ROOT)).replace("\\", "/"),
            "label_columns_in_queue": [
                c for c in LABEL_FIELDS if c in QUEUE_COLUMNS
            ],
        },
    }


def monitoring_triggers(frame: pd.DataFrame, delivered: pd.DataFrame, base_rate: float) -> dict:
    """Thresholds a person can check, with the number to compare against.

    Every threshold is anchored to something measured in ML-09 rather than to a
    round number. The fold spread is the honest scale for "has this moved":
    precision@1000 carried +/-0.054 across held-out clients, so a single cycle
    landing inside that band is noise, not decay.
    """
    return {
        "principle": (
            "ML-09 measured a fold-to-fold spread of +/-0.054 on precision@1000. "
            "Anything inside that band is fold noise; the triggers below sit outside it."
        ),
        "triggers": [
            {
                "name": "queue_precision_decay",
                "watch": "share of the delivered top-1000 that a reviewer confirms as declining",
                "baseline": round(float(delivered["is_declining_label"].mean()), 4),
                "trigger_below": round(
                    float(delivered["is_declining_label"].mean()) - 2 * 0.054, 4
                ),
                "rationale": (
                    "two fold-standard-deviations below what this queue actually "
                    "delivers. The +/-0.054 comes from ML-09's precision@1000 spread "
                    "across held-out clients and is used only as the noise scale -- the "
                    "0.726 mean it came from is NOT this number and the two are not "
                    "comparable (see the notebook's section 1 note on denominators)"
                ),
                "action": "retrain; if it does not recover, fall back to the ML-07 rule",
            },
            {
                "name": "base_rate_shift",
                "watch": "corpus-wide declining rate",
                "baseline": round(base_rate, 4),
                "trigger_outside": [0.442, 0.642],
                "rationale": (
                    "+/-0.10 around the 0.542 the model was fitted at; the queue's lift "
                    "is quoted against that base rate and stops meaning the same thing "
                    "if it moves"
                ),
                "action": "re-quote every lift figure before reusing it; retrain",
            },
            {
                "name": "label_definition_change",
                "watch": "the +/-20% cut on the 30d-vs-prev-30d impression change",
                "baseline": "down when change < -20%",
                "trigger_on": "any change to the threshold, the window, or the metric",
                "rationale": (
                    "ML-09 found the shipped cut is -20% while the paper documents -10%. "
                    "A silent change to either would move the label under the model with "
                    "no error anywhere"
                ),
                "action": "stop; re-derive the label, re-run ML-09's reconstruction check, retrain",
            },
            {
                "name": "feature_drift",
                "watch": "median of the forest's top four features",
                "baseline": {
                    column: round(float(frame[column].median()), 4)
                    for column in [
                        "days_with_impressions",
                        "log_impressions_90d",
                        "avg_position",
                        "content_age_days",
                    ]
                },
                "trigger_on": "any median moving more than 25% from the value above",
                "rationale": "these four carry ~45% of importance; the rest barely matter",
                "action": "investigate the source before retraining -- drift here is often a "
                          "collection change, not a content change",
            },
            {
                "name": "explainability_gap",
                "watch": "share of delivered rows carrying at least one ML-07 rule code",
                "baseline": round(
                    float((delivered["reason_codes"] != "general_refresh_review").mean()), 4
                ),
                "trigger_below": 0.45,
                "rationale": (
                    "the rule already explains only about half the queue, which is why "
                    "model-derived codes exist. The trigger sits below today's measured "
                    "coverage, not at an aspirational 0.80 -- a threshold already breached "
                    "on the day it ships is not a monitor. Total coverage must stay at "
                    "1.00 and is asserted in code, not monitored"
                ),
                "action": (
                    "if rule coverage falls further, the queue is drifting away from what "
                    "the hand rule was built for -- revisit the rule before the model"
                ),
            },
            {
                "name": "queue_concentration",
                "watch": "largest single client's share of the delivered queue",
                "trigger_above": 0.50,
                "rationale": (
                    "per-client declining rates run 0.000 to 0.937, so a queue dominated "
                    "by one client is measuring that client, not the portfolio"
                ),
                "action": "cap per client and re-rank within client",
            },
        ],
        "review_cadence": {
            "every_cycle": ["queue_precision_decay", "explainability_gap", "queue_concentration"],
            "monthly": ["base_rate_shift", "feature_drift"],
            "on_any_upstream_change": ["label_definition_change"],
        },
    }


if __name__ == "__main__":
    out = run()
    OUT_JSON.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k != "monitoring_triggers"}, indent=2))
    print(f"\nqueue -> {OUT_CSV}")
