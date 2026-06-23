"""
cinemastream/ml/churn/cross_validation.py

Chapter 071 -- Train/Validation/Test Split: Stratification, Time-Series
Splits, and Cross-Validation.

Every prior chapter (067-070) reported a single test_acc from one
75/25 stratified train/test split (RANDOM_SEED=42). This module asks
the obvious follow-up question: how much would that number have moved
if the split had landed differently? It re-runs the Ch070 v3 (9-feature,
includes spend_group_imputed_capped) and v2 (8-feature, Ch069 RFE set)
feature matrices through StratifiedKFold(5) cross-validation and reports
the mean +/- std across folds, alongside the original single-split number
for comparison.

Run:
    python cinemastream/ml/churn/cross_validation.py
"""

import sys
sys.path.insert(0, ".")

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

from cinemastream.ml.churn.train import build_churn_dataset, RANDOM_SEED
from cinemastream.ml.churn.data_quality import (
    inject_billing_data,
    impute_spend,
    detect_and_cap_outliers,
    prepare_features_v3,
    run_rf_model,
)


def run_cross_validation(X, y, feature_names, label, n_splits=5, random_seed=RANDOM_SEED):
    """
    Run StratifiedKFold(n_splits) cross-validation for a RandomForest on
    (X, y) and print/report fold scores, mean, and std.

    Returns the array of per-fold accuracy scores.
    """
    rf = RandomForestClassifier(n_estimators=100, random_state=random_seed)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
    scores = cross_val_score(rf, X, y, cv=skf)
    print(f"{label:38s} ({len(feature_names)} features): "
          f"fold scores={np.round(scores, 3)}, "
          f"mean={scores.mean():.3f} +/- std={scores.std():.3f}")
    return scores


def main():
    churn_df = build_churn_dataset()
    billed_df = inject_billing_data(churn_df)
    imputed_df, _ = impute_spend(billed_df)
    capped_df, _, _, _ = detect_and_cap_outliers(imputed_df)

    X_v2, X_v3, y, names_v2, names_v3 = prepare_features_v3(capped_df)

    print("--- Single 75/25 split (Ch067-070 baseline) ---")
    run_rf_model(X_v3, y, names_v3, "v3 (9 features), single split")
    print()

    print("--- StratifiedKFold(5) cross-validation ---")
    run_cross_validation(X_v2, y, names_v2, "v2 (8 features, Ch069 RFE)")
    run_cross_validation(X_v3, y, names_v3, "v3 (9 features, + spend_capped)")


if __name__ == "__main__":
    main()
