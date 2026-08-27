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
    "page_archetype", "archetype_action",
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

# --- page archetypes ---------------------------------------------------------
# The queue answers "which page next". It does not answer "and then what",
# because every delivered row currently carries the same instruction: look at
# this page. The archetype layer is the "and then what": one mutually exclusive
# bucket per page, chosen from observable page condition, each implying a
# different kind of edit. It is a triage aid, not evidence -- no refresh outcome
# exists anywhere in this data, so an archetype says what to *consider*, never
# what will work.
#
# Three things to know before reading the mapping:
#
#   * Assignment is first-match-wins down ARCHETYPE_ORDER, and 8,713 of the
#     26,612 eligible pages satisfy more than one definition, so the order is
#     load-bearing rather than cosmetic. It runs cheapest-and-most-specific
#     first: a title rewrite is a smaller and more reversible act than a
#     consolidation, so where both apply the cheaper diagnosis is offered.
#   * The two buckets that need word_count sit *after* every bucket that does
#     not, so a page with a missing word count is still diagnosed on the
#     evidence that does exist, and only falls to insufficient_page_data when
#     nothing word-count-free explained it.
#   * No archetype reads a label field. They are built on the same feature view
#     the ML-07 rule gets, and the exclusion is asserted in run().
#
# CTR is compared against the page's *position peers* rather than a flat cut.
# Raw CTR falls with position, so a flat threshold would call every deep page a
# click problem. The peer baseline is the median CTR within the page's position
# band, computed only over pages visible enough for CTR to mean anything -- at
# the corpus level the top band's median CTR is 0.00, but that is because those
# rows have almost no impressions, not because page-one pages go unclicked.
VISIBILITY_FLOOR = 500       # impressions_90d below which CTR is too noisy to peer-compare
DEMAND_FLOOR = 250           # impressions_90d below which a content edit is not the lever
POSITION_BANDS = [0, 3, 5, 10, 20, 50, np.inf]
THIN_WORD_COUNT = 1200       # matches the ML-07 rule's thinness cut
STALE_DAYS = 90
DEEP_POSITION = 20

# archetype -> (action, why a reviewer is being pointed at that action)
ARCHETYPE_ACTIONS = {
    "snippet_gap": (
        "rewrite_title_and_meta",
        "ranks on page one and is seen, but clicked less than its position peers: "
        "the listing is the likeliest defect, not the body",
    ),
    "thin_with_demand": (
        "expand_depth",
        "measured thin against measured demand: add substance before anything else",
    ),
    "stale_authority": (
        "refresh_facts_and_dates",
        "a strong, visible page that has not been touched in 90+ days: defend it",
    ),
    "deep_but_substantial": (
        "reassess_intent_or_consolidate",
        "length is not the problem and it ranks deep anyway: question the target "
        "query, or merge it into a stronger page",
    ),
    "weak_engagement": (
        "review_intro_and_layout",
        "people arrive and do not engage: the defect is on the page, above the fold",
    ),
    "low_demand": (
        "no_content_edit_review_targeting",
        "not obviously defective, but the demand is not there. Editing is the wrong "
        "lever and this is where review effort is most often wasted",
    ),
    "insufficient_page_data": (
        "fix_the_record_first",
        "word count is missing, so the thin and substantial tests could not run: "
        "diagnose the data before diagnosing the page",
    ),
    "no_clear_defect": (
        "manual_diagnosis",
        "the model ranks it and none of the above explains why. A human looks, and "
        "this residual is reported rather than hidden",
    ),
}


def assign_archetypes(frame: pd.DataFrame) -> pd.DataFrame:
    """One mutually exclusive archetype per page, first match wins.

    Exhaustive by construction: no_clear_defect is the default, so every page
    gets exactly one bucket and the shares always sum to 1.
    """
    band = pd.cut(frame["avg_position"], POSITION_BANDS)
    visible = frame["impressions_90d"] >= VISIBILITY_FLOOR
    peer_ctr = frame.loc[visible].groupby(band[visible], observed=True)["ctr"].median()
    peer = band.map(peer_ctr).astype(float)

    word_count = frame["word_count"]
    on_page_one = (frame["avg_position"] > 0) & (frame["avg_position"] <= 10)
    engaged = (frame["engagement_rate"] > 0) & (frame["engagement_rate"] < 30)
    scrolled = (frame["scroll_rate"] > 0) & (frame["scroll_rate"] < 30)

    ordered = [
        ("snippet_gap", visible & on_page_one & (frame["ctr"] < peer)),
        ("thin_with_demand",
         (word_count > 0) & (word_count < THIN_WORD_COUNT)
         & (frame["impressions_90d"] >= DEMAND_FLOOR)),
        ("stale_authority",
         visible & on_page_one & (frame["days_since_last_update"] >= STALE_DAYS)),
        ("deep_but_substantial",
         (frame["avg_position"] > DEEP_POSITION) & (word_count >= THIN_WORD_COUNT)),
        ("weak_engagement", (frame["sessions_90d"] >= 30) & (engaged | scrolled)),
        ("low_demand", frame["impressions_90d"] < DEMAND_FLOOR),
        ("insufficient_page_data", word_count == 0),
    ]

    archetype = pd.Series("no_clear_defect", index=frame.index)
    claimed = pd.Series(False, index=frame.index)
    definitions_matched = pd.Series(0, index=frame.index)
    for name, condition in ordered:
        condition = condition.fillna(False)
        definitions_matched += condition.astype(int)
        archetype[condition & ~claimed] = name
        claimed |= condition

    return pd.DataFrame(
        {
            "page_archetype": archetype,
            "archetype_action": archetype.map(lambda a: ARCHETYPE_ACTIONS[a][0]),
            "archetype_definitions_matched": definitions_matched,
        },
        index=frame.index,
    )


def archetype_summary(eligible: pd.DataFrame, delivered: pd.DataFrame) -> dict:
    """Sizes, actions, and the honest caveats, for every archetype."""
    corpus_mix = eligible["page_archetype"].value_counts()
    queue_mix = delivered["page_archetype"].value_counts()

    buckets = {}
    for name, (action, rationale) in ARCHETYPE_ACTIONS.items():
        block = eligible[eligible["page_archetype"] == name]
        in_queue = delivered[delivered["page_archetype"] == name]
        buckets[name] = {
            "action": action,
            "rationale": rationale,
            "eligible_pages": int(corpus_mix.get(name, 0)),
            "eligible_share": round(float(corpus_mix.get(name, 0) / len(eligible)), 4),
            "queue_pages": int(queue_mix.get(name, 0)),
            "queue_share": round(float(queue_mix.get(name, 0) / len(delivered)), 4),
            # Observational only: the rate at which this bucket carries the
            # declining label. NOT the rate at which the action works.
            "declining_rate": round(float(block["is_declining_label"].mean()), 4)
            if len(block) else None,
            "median_impressions_90d": round(float(block["impressions_90d"].median()), 1)
            if len(block) else None,
            "median_avg_position": round(float(block["avg_position"].median()), 2)
            if len(block) else None,
            "median_word_count": round(float(block["word_count"].median()), 1)
            if len(block) else None,
        }

    return {
        "purpose": (
            "the queue says which page next; the archetype says what kind of edit to "
            "consider. Triage only -- no refresh outcome exists in this data, so no "
            "archetype carries evidence that its action recovers traffic"
        ),
        "assignment": "mutually exclusive, exhaustive, first match wins down the order below",
        "order": list(ARCHETYPE_ACTIONS),
        "order_matters_for_pages": int(
            (eligible["archetype_definitions_matched"] > 1).sum()
        ),
        "tie_break_principle": (
            "cheapest and most reversible edit first, so a page that is both a snippet "
            "problem and a consolidation candidate is offered the title rewrite"
        ),
        "eligible_pages": int(len(eligible)),
        "eligible_base_rate": round(float(eligible["is_declining_label"].mean()), 4),
        "buckets": buckets,
        "action_is_not_automated": (
            "an archetype selects the review question, never the edit. Every action "
            "in this table is performed by a human under the section 3 gate"
        ),
    }


# --- content decay -----------------------------------------------------------
# Age bands are wide enough that the smallest holds ~2,500 pages, because the
# claim this section supports is directional and a directional claim on a
# 100-page cell is not worth making.
AGE_BANDS = [89, 120, 180, 270, 365, 470, np.inf]
FRESHNESS_BANDS = [0, 30, 60, 104, 180, np.inf]


def content_decay(eligible: pd.DataFrame, delivered: pd.DataFrame) -> dict:
    """How the declining label moves with content age and with refresh recency.

    The headline is counter-intuitive and it survives the obvious confound, so
    it is reported as a finding: in this corpus *younger* pages carry the
    declining label more often, and the pattern holds inside every impression
    quartile. What it is not is evidence that content stops decaying as it ages
    -- two mechanisms predict the same curve and this data separates neither.
    """
    age_band = pd.cut(eligible["content_age_days"], AGE_BANDS)
    fresh_band = pd.cut(eligible["days_since_last_update"], FRESHNESS_BANDS)
    quartile = pd.qcut(eligible["impressions_90d"], 4, labels=["q1", "q2", "q3", "q4"])

    def profile(grouper) -> dict:
        return {
            str(key): {
                "n": int(len(block)),
                "declining_rate": round(float(block["is_declining_label"].mean()), 4),
                "median_trend_pct": round(float(block["trend_pct"].median()), 2),
                "median_impressions_90d": round(float(block["impressions_90d"].median()), 1),
                # Share of the 90-day impression total landing in each 30-day
                # window. An even split is 0.333. A page whose prior window runs
                # hot and recent window runs cold is settling, which is not the
                # same thing as dying.
                "median_prev_30d_share": round(
                    float((block["impressions_prev_30d"] / block["impressions_90d"]).median()), 4
                ),
                "median_last_30d_share": round(
                    float((block["impressions_last_30d"] / block["impressions_90d"]).median()), 4
                ),
                "median_word_count": round(float(block["word_count"].median()), 1),
            }
            for key, block in eligible.groupby(grouper, observed=True)
        }

    within_quartile = {}
    for name, block in eligible.groupby(quartile, observed=True):
        bands = pd.cut(block["content_age_days"], AGE_BANDS)
        within_quartile[str(name)] = {
            str(key): round(float(sub["is_declining_label"].mean()), 4)
            for key, sub in block.groupby(bands, observed=True)
        }

    by_age = profile(age_band)
    youngest, oldest = list(by_age)[0], list(by_age)[-1]
    holds_in_every_quartile = all(
        list(row.values())[0] > list(row.values())[-1] for row in within_quartile.values()
    )

    corpus_mix = age_band.value_counts(normalize=True)
    queue_mix = pd.cut(delivered["content_age_days"], AGE_BANDS).value_counts(normalize=True)
    skew = {
        str(band): {
            "corpus_share": round(float(corpus_mix.get(band, 0.0)), 4),
            "queue_share": round(float(queue_mix.get(band, 0.0)), 4),
            "over_representation": round(
                float(queue_mix.get(band, 0.0) / corpus_mix.get(band, 1.0)), 2
            ),
        }
        for band in sorted(corpus_mix.index)
    }

    return {
        "finding": (
            "the declining label fires more often on younger pages, not older ones: "
            f"{by_age[youngest]['declining_rate']:.3f} in the {youngest} day band against "
            f"{by_age[oldest]['declining_rate']:.3f} in the {oldest} band"
        ),
        "survives_the_visibility_confound": bool(holds_in_every_quartile),
        "by_content_age": by_age,
        "by_days_since_last_update": profile(fresh_band),
        "declining_rate_by_age_within_impression_quartile": within_quartile,
        "queue_age_skew": skew,
        "refresh_recency_is_nearly_two_valued": {
            "distinct_values": int(eligible["days_since_last_update"].nunique()),
            "top_two_values_share": round(
                float(
                    eligible["days_since_last_update"]
                    .value_counts(normalize=True).head(2).sum()
                ), 4
            ),
            "why_it_matters": (
                "refresh recency cannot be read as a continuous signal here, so this data "
                "cannot support a refresh *cadence* recommendation and none is made"
            ),
        },
        "interpretation": {
            "mechanism_1_post_launch_settling": (
                "the prior 30-day window of a young page catches more of its launch ramp. "
                "Median prev-30d share falls from "
                f"{by_age[youngest]['median_prev_30d_share']:.3f} to "
                f"{by_age[oldest]['median_prev_30d_share']:.3f} across the age range while the "
                "last-30d share rises. A page settling off a launch spike registers as "
                "declining under a label that only compares two adjacent windows"
            ),
            "mechanism_2_survivorship": (
                "ML-09 established the corpus is 100% active content. Pages that launched and "
                "died are already gone, so the old bands contain only what stabilised. Some of "
                "the attenuation with age is selection and this data cannot separate the two"
            ),
            "mechanism_3_feature_degradation": (
                "median word count is 0 in both bands past 365 days -- word count is simply "
                "missing on most old pages. The model may rank old content lower partly "
                "because it can see less about it, which is a data-quality gradient rather "
                "than a content finding, and it is the one mechanism of the three that is "
                "fixable upstream"
            ),
            "what_this_does_not_license": (
                "no claim that content stops decaying with age, and no refresh cadence. Both "
                "mechanisms predict the same curve and neither is testable without dated "
                "history and refresh outcomes"
            ),
            "what_it_does_license": (
                "read the queue within an age band. Ranked across the whole corpus, the "
                "youngest band supplies more of the queue than its share of the corpus, and "
                "some of those rows are settling rather than declining"
            ),
        },
    }


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

    # Page archetypes, on the same label-free feature view the rule gets. Asserted
    # rather than trusted: if a label field ever reaches this call it should fail
    # here, not quietly produce a better-looking table.
    assert not [c for c in LABEL_FIELDS if c in features_only.columns], \
        "archetypes must be assigned on a label-free view"
    frame[["page_archetype", "archetype_action", "archetype_definitions_matched"]] = \
        assign_archetypes(features_only)

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
        "archetypes": archetype_summary(ranked, delivered),
        "content_decay": content_decay(ranked, delivered),
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
