"""
cinemastream/ml/churn/features.py

Chapter 069 -- Feature Engineering for the churn model.

Extends (does not modify) cinemastream/ml/churn/train.py's
build_churn_dataset() / prepare_features() (Chapter 067) per the
"Surgical Changes to Portfolio Code" rule. Adds:

  - engineer_features(): derived/binned features (tenure_bucket,
    engagement_score)
  - prepare_features_v2(): encodes the expanded feature set
  - compare_scalers(): StandardScaler vs RobustScaler on
    watch_minutes_avg
  - select_features(): SelectKBest (f_classif) and RFE (RandomForest)
    on the v2 feature set
  - run_v2_model(): retrains the Chapter 068 Random Forest on a given
    feature matrix and compares test accuracy to the 0.933 baseline

Run:
    python cinemastream/ml/churn/features.py
"""

import sys
sys.path.insert(0, ".")

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from cinemastream.ml.churn.train import build_churn_dataset, prepare_features, RANDOM_SEED

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)


def engineer_features(churn_df):
    """
    Add two derived features to churn_df:

    - tenure_bucket: bins tenure_months into named groups that mirror
      the behavioral risk multipliers used to generate the dataset
      (Chapter 067 session_log): tenure < 3 -> "new" (1.3x risk),
      tenure > 24 -> "veteran" (0.6x risk), everything between is
      "growing" / "established".
    - engagement_score: watch_minutes_avg / (days_since_last_watch + 1)
      -- a single ratio that combines "how much they watch" and "how
      recently they watched" into one number. Higher = more engaged.
      +1 in the denominator avoids division by zero for users who
      watched today (days_since_last_watch == 0).
    """
    df = churn_df.copy()

    df["tenure_bucket"] = pd.cut(
        df["tenure_months"],
        bins=[0, 3, 12, 24, 37],
        labels=["new", "growing", "established", "veteran"],
        right=False,
    )

    df["engagement_score"] = (
        df["watch_minutes_avg"] / (df["days_since_last_watch"] + 1)
    ).round(3)

    return df


def prepare_features_v2(churn_df):
    """
    Like Chapter 067's prepare_features(), but on the engineered
    DataFrame: one-hot encodes country, plan, AND tenure_bucket
    (drop_first=True for all three), and includes engagement_score
    as an additional numeric column.
    """
    engineered_df = engineer_features(churn_df)

    features_df = pd.get_dummies(
        engineered_df.drop(columns=["user_id", "churned"]),
        columns=["country", "plan", "tenure_bucket"],
        drop_first=True,
    )
    X = features_df.values
    y = engineered_df["churned"].values
    return X, y, list(features_df.columns)


def compare_scalers(churn_df):
    """
    Compare StandardScaler vs RobustScaler on watch_minutes_avg.
    Prints min/max/mean of the scaled column for each scaler, plus
    the raw column's min/max for reference.
    """
    raw = churn_df["watch_minutes_avg"].values.reshape(-1, 1)

    print("Raw watch_minutes_avg:  min={:.1f}  max={:.1f}  mean={:.1f}".format(
        raw.min(), raw.max(), raw.mean()
    ))

    for name, scaler in [("StandardScaler", StandardScaler()), ("RobustScaler", RobustScaler())]:
        scaled = scaler.fit_transform(raw)
        print(
            f"{name:15s} min={scaled.min():.3f}  max={scaled.max():.3f}  "
            f"mean={scaled.mean():.3f}"
        )


def select_features(X, y, feature_names, k=8):
    """
    Run SelectKBest (f_classif) and RFE (RandomForestClassifier) on
    the v2 feature set, each selecting the top `k` features. Prints
    both rankings and returns the RFE-selected feature names.
    """
    selector = SelectKBest(score_func=f_classif, k=k)
    selector.fit(X, y)

    scores = selector.scores_
    order = np.argsort(scores)[::-1]
    print(f"SelectKBest (f_classif) -- top {k} of {len(feature_names)} features:")
    for i in order[:k]:
        print(f"  {feature_names[i]:25s} score={scores[i]:.2f}")

    print()
    rfe = RFE(
        RandomForestClassifier(n_estimators=100, random_state=RANDOM_SEED),
        n_features_to_select=k,
    )
    rfe.fit(X, y)
    rfe_selected = [feature_names[i] for i in np.where(rfe.support_)[0]]
    print(f"RFE (RandomForest) -- top {k} of {len(feature_names)} features:")
    for name in rfe_selected:
        print(f"  {name}")

    return rfe_selected


def run_rf_model(X, y, feature_names, label):
    """
    Train a Random Forest (n_estimators=100, random_state=RANDOM_SEED)
    on the given feature matrix, using the same train/test split
    parameters as Chapter 067/068. Prints train/test accuracy and the
    majority-class baseline.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=RANDOM_SEED, stratify=y
    )

    rf = RandomForestClassifier(n_estimators=100, random_state=RANDOM_SEED)
    rf.fit(X_train, y_train)

    train_acc = rf.score(X_train, y_train)
    test_acc = rf.score(X_test, y_test)

    majority_class = int(round(y_train.mean()))
    baseline_preds = np.full_like(y_test, majority_class)
    baseline_acc = accuracy_score(y_test, baseline_preds)

    print(f"{label:32s} ({len(feature_names):2d} features): "
          f"train_acc={train_acc:.3f}, test_acc={test_acc:.3f}")

    return rf, test_acc, baseline_acc


def main():
    churn_df = build_churn_dataset()

    print("--- Chapter 069: Feature Engineering ---")
    print()
    print("engineer_features() preview:")
    engineered_df = engineer_features(churn_df)
    print(engineered_df[
        ["user_id", "tenure_months", "tenure_bucket", "watch_minutes_avg",
         "days_since_last_watch", "engagement_score"]
    ].head())
    print()
    print("tenure_bucket value counts:")
    print(engineered_df["tenure_bucket"].value_counts().sort_index())

    print()
    print("--- Scaler comparison: watch_minutes_avg ---")
    compare_scalers(churn_df)

    print()
    print("--- Feature selection (v2 feature set) ---")
    X_v1, y, feature_names_v1 = prepare_features(churn_df)
    X_v2, y, feature_names_v2 = prepare_features_v2(churn_df)
    print(f"v1 features: {len(feature_names_v1)} | v2 features: {len(feature_names_v2)}")
    print()
    rfe_selected = select_features(X_v2, y, feature_names_v2, k=8)

    print()
    print("--- Random Forest: v1 vs v2 (all) vs v2 (RFE-selected) ---")
    _, v1_acc, baseline_acc = run_rf_model(X_v1, y, feature_names_v1, "v1 (Chapter 068 baseline)")
    _, v2_acc, _ = run_rf_model(X_v2, y, feature_names_v2, "v2 (all engineered features)")

    sub_idx = [feature_names_v2.index(n) for n in rfe_selected]
    X_v2_sub = X_v2[:, sub_idx]
    _, v2_sub_acc, _ = run_rf_model(X_v2_sub, y, rfe_selected, "v2 (RFE-selected subset)")

    print(f"{'Majority-class baseline':32s} ({'--':>2s} features): test_acc={baseline_acc:.3f}")


if __name__ == "__main__":
    main()
