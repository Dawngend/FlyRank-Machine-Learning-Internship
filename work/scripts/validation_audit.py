"""ML-09 validation and leakage audit.

Week 5 (ML-08) already trained under a client-grouped split. So this card is not
"go back and fix the split" -- it is "prove the split was the right one, and show
what the careless version would have claimed instead."

Four checks, in the order the notebook narrates them:

  1. label_reconstruction -- the feature vector ships impressions_last_30d and
     impressions_prev_30d. The contract's label is trend_direction == "down",
     and trend_direction is defined off the 30d-vs-prev-30d impression change.
     Those two columns therefore *are* the label. This check measures how exactly
     they rebuild it, and what a model handed them would score.
  2. single_feature_auc -- every numeric column in the feature vector scored
     alone against the label, so leakage is found by measurement rather than by
     remembering to exclude a name.
  3. split_comparison -- the ML-08 feature set and models under a naive random
     row split versus the grouped-by-client split. Same features, same seed, same
     metrics. The gap is the number this card exists to report.
  4. shuffled_label_null -- the grouped design re-run against a permuted label.
     Anything above chance here would mean the evaluation itself is broken.

Seed: 42, fixed everywhere that samples or permutes.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedKFold

from baseline_action_score import EXCLUDED_FIELDS, LABEL_FIELDS, build
from train_refresh_model import (
    CATEGORICAL,
    K_VALUES,
    NUMERIC,
    N_SPLITS,
    SEED,
    make_models,
    precision_at_k,
)

ROOT = Path(__file__).resolve().parents[2]
FEATURE_PATH = ROOT / "data" / "processed" / "refresh_feature_vector.csv"
OUT_JSON = ROOT / "work" / "outputs" / "validation_audit.json"

# The label and the two columns it is arithmetically built from.
LABEL_COL = "is_declining_label"
LABEL_SOURCE = ["trend_direction", "trend_pct"]
TREND_INPUTS = ["impressions_last_30d", "impressions_prev_30d"]

# Anything at or above this scored alone is not a feature, it is the answer.
LEAK_THRESHOLD = 0.90
# Above this is worth a sentence in the report even if it is legitimate.
WATCH_THRESHOLD = 0.65


def summarise(values: list) -> dict:
    clean = [v for v in values if v is not None]
    if not clean:
        return {"mean": None, "std": None, "folds": 0}
    return {
        "mean": float(np.mean(clean)),
        "std": float(np.std(clean)),
        "folds": len(clean),
        "per_fold": [round(float(v), 4) for v in clean],
    }


def label_reconstruction(frame: pd.DataFrame) -> dict:
    """Can the two 30-day impression columns rebuild the label on their own?

    The paper's "How to Read This Paper" page documents the trend cut at +/-10%.
    The shipped data does not agree: recovering the boundary empirically from
    trend_direction puts it at +/-20%. Both are reported so the gap is on record
    rather than silently corrected.
    """
    last = frame["impressions_last_30d"].to_numpy(dtype=float)
    prev = frame["impressions_prev_30d"].to_numpy(dtype=float)
    y = frame[LABEL_COL].to_numpy()

    with np.errstate(divide="ignore", invalid="ignore"):
        pct = np.where(prev > 0, (last - prev) / prev * 100.0, np.nan)
    scored = ~np.isnan(pct)

    # Where each documented direction actually sits on the recomputed change,
    # which is how the real boundary was found.
    observed_bands = {}
    for direction in ("down", "stable", "up"):
        band = pct[(frame["trend_direction"] == direction).to_numpy() & scored]
        if band.size:
            observed_bands[direction] = {
                "n": int(band.size),
                "min_pct": round(float(band.min()), 3),
                "max_pct": round(float(band.max()), 3),
            }

    agreement_by_threshold = {}
    for threshold in (-10.0, -20.0):
        rebuilt = np.where(prev > 0, pct < threshold, False)
        agreement_by_threshold[f"{threshold:g}%"] = round(float((rebuilt == y).mean()), 6)

    # The recovered rule, scored on every row including prev_30d == 0.
    rebuilt = np.where(prev > 0, pct < -20.0, False)
    exact = float((rebuilt == y).mean())

    # Neither column is damning alone; the ratio between them is the whole label.
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(prev > 0, last / prev, np.nan)
    usable = ~np.isnan(ratio)
    leak_auc = float(roc_auc_score(y[usable], -ratio[usable]))
    solo_auc = {
        col: round(float(roc_auc_score(y, frame[col].to_numpy(dtype=float))), 4)
        for col in TREND_INPUTS
    }

    return {
        "documented_rule": "paper p.5: down when impressions fall >10% vs prev 30d",
        "recovered_rule": "down when (imp_last_30d - imp_prev_30d) / imp_prev_30d < -20%",
        "documentation_mismatch": True,
        "observed_bands_on_recomputed_change": observed_bands,
        "rows_total": int(len(frame)),
        "rows_with_nonzero_prev_30d": int(scored.sum()),
        "rows_with_zero_prev_30d": int((prev == 0).sum()),
        "zero_prev_30d_are_new_or_flat_and_never_labelled_down": bool(
            y[(prev == 0)].sum() == 0
        ),
        "agreement_by_threshold": agreement_by_threshold,
        "exact_agreement_recovered_rule": round(exact, 6),
        "rows_reconstructed_exactly": int((rebuilt == y).sum()),
        "roc_auc_of_impression_ratio_alone": round(leak_auc, 4),
        "roc_auc_of_each_column_alone": solo_auc,
        "verdict": (
            "leak_confirmed_as_interaction" if exact > 0.999 else "no_exact_reconstruction"
        ),
        "why_a_single_feature_scan_misses_it": (
            "each column alone separates the label barely better than chance; the "
            "label lives in their ratio, which no one-feature-at-a-time scan tests"
        ),
        "in_ml08_feature_set": [c for c in TREND_INPUTS if c in NUMERIC + CATEGORICAL],
    }


def single_feature_auc(frame: pd.DataFrame) -> dict:
    """Score every numeric column alone. Leakage should be found, not remembered."""
    y = frame[LABEL_COL].to_numpy()
    rows = []
    skip = {LABEL_COL, "content_id"}
    for col in frame.columns:
        if col in skip or not pd.api.types.is_numeric_dtype(frame[col]):
            continue
        values = frame[col].to_numpy(dtype=float)
        usable = np.isfinite(values)
        if usable.sum() < 100 or len(np.unique(values[usable])) < 2:
            continue
        auc = float(roc_auc_score(y[usable], values[usable]))
        # Direction does not matter for a leakage hunt; distance from 0.5 does.
        rows.append(
            {
                "feature": col,
                "auc": round(auc, 4),
                "separation": round(abs(auc - 0.5), 4),
                "rows_scored": int(usable.sum()),
                "in_model": col in NUMERIC,
                "is_label_source": col in LABEL_SOURCE or col in TREND_INPUTS,
            }
        )
    rows.sort(key=lambda r: -r["separation"])
    return {
        "leak_threshold_auc_distance": round(LEAK_THRESHOLD - 0.5, 4),
        "watch_threshold_auc_distance": round(WATCH_THRESHOLD - 0.5, 4),
        "flagged_as_leak": [r for r in rows if r["separation"] >= LEAK_THRESHOLD - 0.5],
        "flagged_as_watch": [
            r
            for r in rows
            if WATCH_THRESHOLD - 0.5 <= r["separation"] < LEAK_THRESHOLD - 0.5
        ],
        "all_features": rows,
    }


def evaluate_split(X, y, groups, folds, shuffle_label: bool = False) -> dict:
    """Run every ML-08 model across a given fold list and collect the metrics."""
    rng = np.random.default_rng(SEED)
    y_used = rng.permutation(y) if shuffle_label else y

    results = {
        name: {"roc_auc": [], "average_precision": [], **{f"p@{k}": [] for k in K_VALUES}}
        for name in make_models()
    }
    for train_idx, test_idx in folds:
        for name, pipe in make_models().items():
            pipe.fit(X.iloc[train_idx], y_used[train_idx])
            proba = pipe.predict_proba(X.iloc[test_idx])[:, 1]
            y_test = y_used[test_idx]
            results[name]["roc_auc"].append(float(roc_auc_score(y_test, proba)))
            results[name]["average_precision"].append(
                float(average_precision_score(y_test, proba))
            )
            for k in K_VALUES:
                results[name][f"p@{k}"].append(precision_at_k(y_test, proba, k))
    return {
        name: {metric: summarise(vals) for metric, vals in block.items()}
        for name, block in results.items()
    }


def score_baseline_rule(frame: pd.DataFrame, y, folds) -> dict:
    """The ML-07 hand rule on the same folds, so section 4 can compare per fold.

    ML-08 recorded only fold means for the rule, and the claim rewrite needs the
    individual folds, so the rule is re-scored here rather than re-running ML-08.
    """
    features_only = frame.drop(
        columns=[c for c in LABEL_FIELDS + EXCLUDED_FIELDS if c in frame.columns]
    )
    scores = (
        build(features_only.assign(_i=range(len(features_only))))
        .sort_values("_i")["action_score"]
        .to_numpy()
    )
    block = {"roc_auc": [], "average_precision": [], **{f"p@{k}": [] for k in K_VALUES}}
    for _, test_idx in folds:
        y_test, s_test = y[test_idx], scores[test_idx]
        block["roc_auc"].append(float(roc_auc_score(y_test, s_test)))
        block["average_precision"].append(float(average_precision_score(y_test, s_test)))
        for k in K_VALUES:
            block[f"p@{k}"].append(precision_at_k(y_test, s_test, k))
    return {metric: summarise(vals) for metric, vals in block.items()}


def client_overlap(folds, groups) -> float:
    """Share of test clients that also appear in the same fold's training rows."""
    shares = []
    for train_idx, test_idx in folds:
        train_clients = set(groups[train_idx])
        test_clients = set(groups[test_idx])
        shares.append(len(train_clients & test_clients) / len(test_clients))
    return float(np.mean(shares))


def run() -> dict:
    frame = pd.read_csv(FEATURE_PATH)
    y = frame[LABEL_COL].to_numpy()
    groups = frame["client_id"].to_numpy()
    X = frame[NUMERIC + CATEGORICAL]

    grouped_folds = list(GroupKFold(n_splits=N_SPLITS).split(X, y, groups))
    naive_folds = list(
        StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED).split(X, y)
    )

    grouped = evaluate_split(X, y, groups, grouped_folds)
    naive = evaluate_split(X, y, groups, naive_folds)
    null = evaluate_split(X, y, groups, grouped_folds, shuffle_label=True)

    # The ML-07 rule under both designs, on the identical folds, for section 4.
    grouped["baseline_rule"] = score_baseline_rule(frame, y, grouped_folds)
    naive["baseline_rule"] = score_baseline_rule(frame, y, naive_folds)

    inflation = {}
    for name in grouped:
        inflation[name] = {}
        for metric in ("roc_auc", "average_precision", "p@50"):
            g = grouped[name][metric]["mean"]
            n = naive[name][metric]["mean"]
            if g is None or n is None:
                continue
            inflation[name][metric] = {
                "grouped": round(g, 4),
                "naive_random": round(n, 4),
                "absolute_gap": round(n - g, 4),
                "relative_inflation_pct": round((n - g) / g * 100.0, 2),
            }

    return {
        "card": "ML-09",
        "seed": SEED,
        "base_rate": float(y.mean()),
        "rows": int(len(frame)),
        "n_clients": int(pd.Series(groups).nunique()),
        "label_reconstruction": label_reconstruction(frame),
        "single_feature_auc": single_feature_auc(frame),
        "split_comparison": {
            "naive_random": {
                "type": f"StratifiedKFold(n_splits={N_SPLITS}, shuffle=True)",
                "client_overlap_share": round(client_overlap(naive_folds, groups), 4),
                "metrics": naive,
            },
            "grouped_by_client": {
                "type": f"GroupKFold(n_splits={N_SPLITS}) on client_id",
                "client_overlap_share": round(client_overlap(grouped_folds, groups), 4),
                "metrics": grouped,
            },
            "inflation": inflation,
        },
        "shuffled_label_null": {
            "design": "grouped folds, label permuted with seed 42",
            "expectation": "roc_auc ~= 0.50, precision@k ~= base rate",
            "metrics": null,
        },
        "time_aware_split": {
            "status": "not_applicable",
            "reason": (
                "the starter CSV is one undated snapshot; the contract's sealed test "
                "month (2026-06) needs the dated warehouse release to be run"
            ),
        },
    }


if __name__ == "__main__":
    out = run()
    OUT_JSON.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out["label_reconstruction"], indent=2))
    print("\nTop separating features:")
    for row in out["single_feature_auc"]["all_features"][:12]:
        print(f"  {row['feature']:<28} auc={row['auc']:.4f}  in_model={row['in_model']}")
    print("\nSplit inflation:")
    print(json.dumps(out["split_comparison"]["inflation"], indent=2))
    print("\nNull (shuffled label), random_forest:")
    print(json.dumps(out["shuffled_label_null"]["metrics"]["random_forest"]["roc_auc"], indent=2))
