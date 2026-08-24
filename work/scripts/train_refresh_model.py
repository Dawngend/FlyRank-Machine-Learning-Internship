"""ML-08 model training, evaluated the way the baseline was.

Method and split are Dawn's decisions, recorded here so the notebook narrates a
choice rather than re-deriving one:

  * methods: Logistic Regression (readable) -> Random Forest (stronger), plus a
    depth-2 decision tree kept purely so the decision surface can be printed and
    read. Per skills/training-honest-models, the lane is a ranking question, so
    every method is scored on its *probability* at precision@K rather than on
    its hard labels.
  * split: grouped by client_id, so no client appears in both train and test.
    The contract's sealed test month (2026-06) is the design that would be used
    once dated warehouse data exists; the starter CSV is a single undated
    snapshot, so it cannot be applied here.

The baseline is imported from baseline_action_score rather than reimplemented,
and is scored on the same test folds, so the comparison table is a comparison.

Seed: 42, fixed everywhere that samples.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_text

from baseline_action_score import EXCLUDED_FIELDS, LABEL_FIELDS, build

SEED = 42
N_SPLITS = 5
K_VALUES = (20, 50, 100, 500, 1000)

ROOT = Path(__file__).resolve().parents[2]
FEATURE_PATH = ROOT / "data" / "processed" / "refresh_feature_vector.csv"
OUT_JSON = ROOT / "work" / "outputs" / "model_comparison.json"

NUMERIC = [
    "search_volume", "competition", "cpc", "word_count", "char_count",
    "log_impressions_90d", "log_clicks_90d", "log_sessions_90d",
    "log_ai_sessions_90d", "days_with_impressions", "days_with_sessions",
    "content_age_days", "days_since_last_update", "ctr", "avg_position",
    "engagement_rate", "scroll_rate", "ai_traffic_pct",
]
CATEGORICAL = [
    "competition_level", "content_type", "main_intent", "age_tier",
    "freshness_tier", "word_count_tier", "impression_tier", "position_tier",
]


def precision_at_k(y_true: np.ndarray, scores: np.ndarray, k: int) -> float | None:
    """Precision among the k highest-scored rows. None if the fold is smaller than k."""
    if len(y_true) < k:
        return None
    order = np.argsort(-scores, kind="stable")
    return float(np.asarray(y_true)[order][:k].mean())


def make_models() -> dict[str, Pipeline]:
    pre = ColumnTransformer(
        [
            ("num", StandardScaler(), NUMERIC),
            ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=25), CATEGORICAL),
        ]
    )
    return {
        "logistic_regression": Pipeline(
            [("pre", pre), ("model", LogisticRegression(max_iter=2000, random_state=SEED))]
        ),
        "random_forest": Pipeline(
            [
                ("pre", pre),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=300,
                        min_samples_leaf=25,
                        n_jobs=-1,
                        random_state=SEED,
                    ),
                ),
            ]
        ),
        "decision_tree_depth2": Pipeline(
            [("pre", pre), ("model", DecisionTreeClassifier(max_depth=2, random_state=SEED))]
        ),
    }


def run() -> dict:
    frame = pd.read_csv(FEATURE_PATH)
    y = frame["is_declining_label"].to_numpy()
    groups = frame["client_id"].to_numpy()

    # The baseline scores the same rows, under the same leakage rules, so it can
    # be sliced by the same fold indices.
    features_only = frame.drop(
        columns=[c for c in LABEL_FIELDS + EXCLUDED_FIELDS if c in frame.columns]
    )
    baseline_scored = build(features_only)
    baseline_by_row = (
        baseline_scored.set_index(baseline_scored.index)["action_score"]
        .reindex(range(len(frame)))
        .to_numpy()
    )
    # build() sorts; recover per-original-row scores by re-scoring in place.
    baseline_by_row = build(features_only.assign(_i=range(len(features_only)))).sort_values("_i")[
        "action_score"
    ].to_numpy()

    X = frame[NUMERIC + CATEGORICAL]
    splitter = GroupKFold(n_splits=N_SPLITS)
    folds = list(splitter.split(X, y, groups))

    results: dict[str, dict] = {
        name: {"roc_auc": [], "average_precision": [], **{f"p@{k}": [] for k in K_VALUES}}
        for name in ["baseline_rule", *make_models().keys()]
    }
    tree_text = ""
    importances: list[pd.Series] = []

    for fold_i, (train_idx, test_idx) in enumerate(folds):
        y_test = y[test_idx]

        # --- the ML-07 rule, scored on the identical test rows ---------------
        b_scores = baseline_by_row[test_idx]
        results["baseline_rule"]["roc_auc"].append(float(roc_auc_score(y_test, b_scores)))
        results["baseline_rule"]["average_precision"].append(
            float(average_precision_score(y_test, b_scores))
        )
        for k in K_VALUES:
            results["baseline_rule"][f"p@{k}"].append(precision_at_k(y_test, b_scores, k))

        # --- the learned models ---------------------------------------------
        for name, pipe in make_models().items():
            pipe.fit(X.iloc[train_idx], y[train_idx])
            proba = pipe.predict_proba(X.iloc[test_idx])[:, 1]
            results[name]["roc_auc"].append(float(roc_auc_score(y_test, proba)))
            results[name]["average_precision"].append(
                float(average_precision_score(y_test, proba))
            )
            for k in K_VALUES:
                results[name][f"p@{k}"].append(precision_at_k(y_test, proba, k))

            if name == "decision_tree_depth2" and fold_i == 0:
                names = pipe.named_steps["pre"].get_feature_names_out()
                tree_text = export_text(
                    pipe.named_steps["model"], feature_names=list(names), decimals=3
                )
            if name == "random_forest":
                names = pipe.named_steps["pre"].get_feature_names_out()
                importances.append(
                    pd.Series(pipe.named_steps["model"].feature_importances_, index=names)
                )

    def summarise(values: list) -> dict:
        clean = [v for v in values if v is not None]
        if not clean:
            return {"mean": None, "std": None, "folds": 0}
        return {
            "mean": float(np.mean(clean)),
            "std": float(np.std(clean)),
            "folds": len(clean),
        }

    summary = {
        "card": "ML-08",
        "seed": SEED,
        "split": {
            "type": "GroupKFold on client_id",
            "n_splits": N_SPLITS,
            "n_clients": int(pd.Series(groups).nunique()),
            "rationale": "no client appears in both train and test",
            "sealed_month_design": "contract sealed test month 2026-06, not applicable to the undated starter CSV",
        },
        "base_rate": float(y.mean()),
        "metrics": {
            name: {metric: summarise(vals) for metric, vals in block.items()}
            for name, block in results.items()
        },
        "decision_tree_depth2_rules": tree_text,
        "random_forest_top_features": (
            pd.concat(importances, axis=1).mean(axis=1).sort_values(ascending=False).head(15).to_dict()
        ),
    }
    OUT_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    out = run()
    print(json.dumps({k: v for k, v in out.items() if k != "decision_tree_depth2_rules"}, indent=2))
    print("\nDepth-2 tree:\n" + out["decision_tree_depth2_rules"])
