"""
cinemastream/ml/churn/data_quality.py

Chapter 070 -- Handling Missing Data and Outliers.

Simulates joining churn_df (Ch067) against a real `subscriptions` billing
table: adds monthly_spend_sgd, a column with realistic missing values
(~8%, independent of plan -- MCAR by construction) and a small cluster of
extreme outliers (a documented Premium double-charge bug). Demonstrates
detection, group-aware imputation, IQR-based outlier detection and
capping, and re-runs the Ch069 RFE-selected Random Forest with the new
feature added.

Run:
    python cinemastream/ml/churn/data_quality.py
"""

import sys
sys.path.insert(0, ".")

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from cinemastream.ml.churn.train import build_churn_dataset, RANDOM_SEED
from cinemastream.ml.churn.features import engineer_features

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

# Canonical monthly subscription prices (SGD) -- documented in
# bible_core.md plan pricing.
PLAN_BASE_SPEND = {"Free": 0.0, "Basic": 12.90, "Premium": 19.90}


def inject_billing_data(churn_df, random_seed=RANDOM_SEED):
    """
    Simulate a join against the real `subscriptions` billing table.

    Adds `monthly_spend_sgd`:
      - base spend by plan (PLAN_BASE_SPEND) + small noise
      - ~8% set to NaN, independent of plan/country -- simulating
        billing-system join failures (MCAR: Missing Completely At Random)
      - ~5% of Premium rows hit by a documented double-charge bug:
        spend is roughly doubled plus a small extra fee
    """
    rng = np.random.default_rng(random_seed + 1)  # separate stream from build_churn_dataset
    df = churn_df.copy()
    n = len(df)

    base = df["plan"].map(PLAN_BASE_SPEND).to_numpy()
    noise = rng.normal(0, 0.5, size=n)
    spend = np.round(base + noise, 2)
    spend = np.clip(spend, 0, None)

    # Double-charge bug: ~5% of Premium users billed twice + a small fee
    premium_idx = np.where(df["plan"].to_numpy() == "Premium")[0]
    n_bugged = max(1, int(len(premium_idx) * 0.05))
    bugged_idx = rng.choice(premium_idx, size=n_bugged, replace=False)
    spend[bugged_idx] = np.round(spend[bugged_idx] * 2 + rng.uniform(0, 5, size=n_bugged), 2)

    # ~8% missing, independent of plan (MCAR)
    missing_idx = rng.choice(n, size=int(n * 0.08), replace=False)
    spend[missing_idx] = np.nan

    df["monthly_spend_sgd"] = spend
    return df


def detect_missingness(df):
    """Report missing count/rate overall and by plan -- checks the MCAR assumption."""
    missing_count = df["monthly_spend_sgd"].isnull().sum()
    missing_pct = df["monthly_spend_sgd"].isnull().mean()
    print(f"monthly_spend_sgd missing: {missing_count} of {len(df)} ({missing_pct:.1%})")
    print()
    print("Missing rate by plan:")
    print(df.groupby("plan")["monthly_spend_sgd"].apply(lambda s: s.isnull().mean()).round(3))


def impute_spend(df):
    """
    Compare a naive global-median fill against a group-aware (per-plan)
    median fill for monthly_spend_sgd.
    """
    df = df.copy()
    global_median = df["monthly_spend_sgd"].median()
    df["spend_global_imputed"] = df["monthly_spend_sgd"].fillna(global_median)
    df["spend_group_imputed"] = df.groupby("plan")["monthly_spend_sgd"].transform(
        lambda s: s.fillna(s.median())
    )
    return df, global_median


def detect_and_cap_outliers(df, column="spend_group_imputed"):
    """
    IQR-based outlier detection and capping (winsorizing) on `column`.
    Returns (df_with_capped_column, lower_bound, upper_bound, n_outliers).
    """
    df = df.copy()
    Q1, Q3 = df[column].quantile([0.25, 0.75])
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    n_outliers = ((df[column] < lower) | (df[column] > upper)).sum()
    df[f"{column}_capped"] = df[column].clip(lower, upper)
    return df, lower, upper, n_outliers


# Ch069's RFE-selected 8-feature set, reproduced here as a constant so this
# module does not need to re-run RFE (avoids re-deriving a result that's
# already canonical from Ch069).
CH069_RFE_FEATURES = [
    "watch_minutes_avg", "days_since_last_watch", "tenure_months",
    "support_tickets_count", "engagement_score",
    "country_TH", "country_VN", "tenure_bucket_established",
]


def prepare_features_v3(df):
    """
    Build the Ch069 RFE-selected 8-feature matrix, then add
    spend_group_imputed_capped as a 9th feature.
    """
    engineered_df = engineer_features(df)
    features_df = pd.get_dummies(
        engineered_df.drop(columns=["user_id", "churned"]),
        columns=["country", "plan", "tenure_bucket"],
        drop_first=True,
    )
    for col in CH069_RFE_FEATURES:
        if col not in features_df.columns:
            features_df[col] = 0

    X_v2 = features_df[CH069_RFE_FEATURES].to_numpy()
    X_v3 = features_df[CH069_RFE_FEATURES + ["spend_group_imputed_capped"]].to_numpy()
    y = engineered_df["churned"].to_numpy()
    return X_v2, X_v3, y, CH069_RFE_FEATURES, CH069_RFE_FEATURES + ["spend_group_imputed_capped"]


def run_rf_model(X, y, feature_names, label):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=RANDOM_SEED, stratify=y
    )
    rf = RandomForestClassifier(n_estimators=100, random_state=RANDOM_SEED)
    rf.fit(X_train, y_train)
    train_acc = rf.score(X_train, y_train)
    test_acc = rf.score(X_test, y_test)
    print(f"{label:38s} ({len(feature_names)} features): "
          f"train_acc={train_acc:.3f}, test_acc={test_acc:.3f}")
    return rf, test_acc


def main():
    churn_df = build_churn_dataset()
    billed_df = inject_billing_data(churn_df)

    print("--- monthly_spend_sgd preview ---")
    print(billed_df[["user_id", "plan", "monthly_spend_sgd"]].head(8))
    print()

    print("--- Missingness check ---")
    detect_missingness(billed_df)
    print()

    print("--- Imputation: global median vs group (per-plan) median ---")
    imputed_df, global_median = impute_spend(billed_df)
    print(f"Global median spend (all plans): {global_median}")
    print()
    print("Per-plan MEAN of spend_global_imputed vs spend_group_imputed:")
    print(imputed_df.groupby("plan")[["spend_global_imputed", "spend_group_imputed"]].mean().round(2))
    print()

    print("--- Outlier detection + capping (IQR on spend_group_imputed) ---")
    capped_df, lower, upper, n_outliers = detect_and_cap_outliers(imputed_df)
    print(f"IQR bounds: [{lower:.2f}, {upper:.2f}]")
    print(f"Outliers detected: {n_outliers}")
    print()
    print("Premium rows flagged as outliers (before capping):")
    premium_outliers = capped_df[
        (capped_df["plan"] == "Premium") & (capped_df["spend_group_imputed"] > upper)
    ][["user_id", "plan", "spend_group_imputed", "spend_group_imputed_capped"]]
    print(premium_outliers.head(10))
    print()
    print(f"spend_group_imputed        -- mean={capped_df['spend_group_imputed'].mean():.2f}, "
          f"std={capped_df['spend_group_imputed'].std():.2f}, max={capped_df['spend_group_imputed'].max():.2f}")
    print(f"spend_group_imputed_capped -- mean={capped_df['spend_group_imputed_capped'].mean():.2f}, "
          f"std={capped_df['spend_group_imputed_capped'].std():.2f}, max={capped_df['spend_group_imputed_capped'].max():.2f}")
    print()

    print("--- RF: Ch069 RFE-8 vs Ch069 RFE-8 + spend_group_imputed_capped ---")
    X_v2, X_v3, y, names_v2, names_v3 = prepare_features_v3(capped_df)
    run_rf_model(X_v2, y, names_v2, "v2 (Ch069 RFE-selected, 8 features)")
    run_rf_model(X_v3, y, names_v3, "v3 (+ spend_group_imputed_capped)")


if __name__ == "__main__":
    main()
