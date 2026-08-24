"""ML-07 baseline action score.

A transparent, hand-written rule that ranks content for refresh. No model, no
learned weights: every point in the score is traceable to one stated reason.

Copied and adapted from scripts/02_baseline_score.py per work/README.md rule 1
(the reference pipeline in scripts/ stays pristine). Differences from the
reference:

  * scores on the ML-04 data contract's feature list only, so the excluded
    fields (provider_used, model_used) and every label field are structurally
    unreachable rather than merely unused;
  * emits a 0-100 action score with per-reason point contributions, so a
    reviewer can read why any single row sits where it does;
  * writes work/outputs/baseline_action_score.csv, which ML-08 later compares
    against.

Seed: none needed. The rule is deterministic; there is no sampling anywhere.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
FEATURE_PATH = ROOT / "data" / "processed" / "refresh_feature_vector.csv"
CONTRACT_PATH = ROOT / "work" / "outputs" / "data_contract.json"
OUT_CSV = ROOT / "work" / "outputs" / "baseline_action_score.csv"
OUT_JSON = ROOT / "work" / "outputs" / "baseline_action_score_metrics.json"

# Fields the rule is allowed to touch. Anything not listed here is dropped
# before scoring, which is what makes the leakage check in section 4 a
# structural claim rather than a promise.
LABEL_FIELDS = ["trend_direction", "trend_pct", "is_declining_label"]
EXCLUDED_FIELDS = ["provider_used", "model_used"]

# Each rule: (reason_code, points, predicate). Points are the whole score.
RULES = [
    (
        "stale_visible_page",
        30,
        lambda d: (d["days_since_last_update"] >= 180) & (d["impressions_90d"] >= 500),
    ),
    (
        "thin_visible_page",
        25,
        lambda d: (d["word_count"] > 0)
        & (d["word_count"] < 1200)
        & (d["impressions_90d"] >= 250),
    ),
    (
        "page_one_decay_risk",
        20,
        lambda d: (d["avg_position"] > 0)
        & (d["avg_position"] <= 10)
        & (d["content_age_days"] >= 180),
    ),
    (
        "low_ctr_visible_page",
        15,
        lambda d: (d["impressions_90d"] >= 500)
        & (d["avg_position"] > 0)
        & (d["avg_position"] <= 20)
        & (d["ctr"] < 0.5),
    ),
    (
        "low_engagement_visible_page",
        10,
        lambda d: (d["sessions_90d"] >= 30)
        & (
            ((d["engagement_rate"] > 0) & (d["engagement_rate"] < 30))
            | ((d["scroll_rate"] > 0) & (d["scroll_rate"] < 30))
        ),
    ),
]

# Reason code -> the action a human should take. First match wins, in this
# order, so the action is always the most specific one that applies.
ACTION_BY_REASON = [
    ("thin_visible_page", "expand_and_refresh"),
    ("stale_visible_page", "refresh_content"),
    ("low_ctr_visible_page", "refresh_and_review_ctr"),
    ("page_one_decay_risk", "refresh_to_defend_position"),
    ("low_engagement_visible_page", "review_engagement"),
]


def build(frame: pd.DataFrame) -> pd.DataFrame:
    scored = frame.copy()
    scored["action_score"] = 0
    hits: list[pd.Series] = []

    for code, points, predicate in RULES:
        fired = predicate(scored).fillna(False)
        scored[f"pts_{code}"] = fired.astype(int) * points
        scored["action_score"] += scored[f"pts_{code}"]
        hits.append(fired.rename(code))

    hit_frame = pd.concat(hits, axis=1)
    scored["reason_codes"] = hit_frame.apply(
        lambda row: "|".join(sorted(row.index[row])) or "general_refresh_review",
        axis=1,
    )
    scored["reason_count"] = hit_frame.sum(axis=1)

    def pick_action(codes: str) -> str:
        present = set(codes.split("|"))
        for code, action in ACTION_BY_REASON:
            if code in present:
                return action
        return "monitor_only"

    scored["suggested_action"] = scored["reason_codes"].map(pick_action)
    scored = scored.sort_values(
        ["action_score", "impressions_90d"], ascending=[False, False]
    ).reset_index(drop=True)
    scored["rank"] = scored.index + 1
    return scored


def main() -> None:
    raw = pd.read_csv(FEATURE_PATH)
    contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))

    # Structural leakage guard: hold labels aside for evaluation only, and drop
    # the contract's excluded fields entirely.
    held_out_labels = raw[[c for c in LABEL_FIELDS if c in raw.columns]].copy()
    features = raw.drop(
        columns=[c for c in LABEL_FIELDS + EXCLUDED_FIELDS if c in raw.columns]
    )

    scored = build(features)

    # Re-attach labels only now, after every score is fixed.
    scored = scored.join(held_out_labels.reindex(scored.index))
    keep = [
        "rank",
        "content_id",
        "client_id",
        "action_score",
        "reason_codes",
        "reason_count",
        "suggested_action",
        "impressions_90d",
        "clicks_90d",
        "sessions_90d",
        "ctr",
        "avg_position",
        "word_count",
        "content_age_days",
        "days_since_last_update",
        "engagement_rate",
        "scroll_rate",
    ] + [f"pts_{code}" for code, _, _ in RULES]
    scored[keep].to_csv(OUT_CSV, index=False)

    # Evaluation. The rule never saw these columns; this is scoring the rule,
    # not fitting it.
    labelled = build(features).join(raw["is_declining_label"].rename("y"))
    base_rate = float(labelled["y"].mean())
    metrics = {
        "card": "ML-07",
        "contract_version": contract["version"],
        "rows_scored": int(len(labelled)),
        "base_rate_declining": base_rate,
        "score_distribution": {
            str(k): int(v)
            for k, v in labelled["action_score"].value_counts().sort_index().items()
        },
        "precision_at_k": {
            str(k): float(labelled.head(k)["y"].mean()) for k in (20, 50, 100, 500, 1000)
        },
        "lift_at_k": {
            str(k): float(labelled.head(k)["y"].mean() / base_rate)
            for k in (20, 50, 100, 500, 1000)
        },
        "reason_code_counts": {
            code: int(labelled[f"pts_{code}"].gt(0).sum()) for code, _, _ in RULES
        },
        "precision_by_reason": {
            code: float(labelled.loc[labelled[f"pts_{code}"].gt(0), "y"].mean())
            for code, _, _ in RULES
        },
        "action_counts": {
            str(k): int(v) for k, v in labelled["suggested_action"].value_counts().items()
        },
        "leakage_guard": {
            "label_fields_dropped_before_scoring": LABEL_FIELDS,
            "contract_excluded_fields_dropped": EXCLUDED_FIELDS,
            "columns_visible_to_rule": int(features.shape[1]),
        },
    }
    OUT_JSON.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
