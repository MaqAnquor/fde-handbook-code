"""
cinemastream/ml/churn/train.py

Chapter 067 — Supervised Learning: baseline churn-prediction model.

Builds a synthetic churn dataset shaped like the real `users` /
`subscriptions` / `watch_events` / `support_tickets` joins (Chapter 069
will replace this with the real feature pipeline), trains a Logistic
Regression baseline classifier, and reports accuracy alongside the
majority-class baseline -- because (per Chapter 067 Section 3) accuracy
alone is misleading on an imbalanced dataset like churn.

Run:
    python cinemastream/ml/churn/train.py
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

RANDOM_SEED = 42


def build_churn_dataset(n=300, random_seed=RANDOM_SEED):
    """
    Generate a synthetic churn dataset with realistic per-country,
    per-plan churn rates.

    Canonical churn rates (overall / Premium), from session_log.md
    (Chapters 064-066, the "Ask Anything" churn figures):
        VN: 8.2% / 3.1%
        PH: 6.5% / 5.9%
        ID: 5.0% / 2.0%
    Other countries (MY, TH, SG, IN) use an invented baseline of
    5.5% overall / 2.5% Premium (documented in session_log.md Ch067).

    Returns a DataFrame with columns:
        user_id, country, plan, watch_minutes_avg, days_since_last_watch,
        tenure_months, support_tickets_count, churned
    """
    rng = np.random.default_rng(random_seed)

    countries = rng.choice(
        ["VN", "PH", "ID", "MY", "TH", "SG", "IN"],
        size=n,
        p=[0.20, 0.18, 0.18, 0.12, 0.12, 0.10, 0.10],
    )
    plans = rng.choice(["Free", "Basic", "Premium"], size=n, p=[0.35, 0.40, 0.25])

    overall_churn = {"VN": 0.082, "PH": 0.065, "ID": 0.050, "MY": 0.055, "TH": 0.055, "SG": 0.055, "IN": 0.055}
    premium_churn = {"VN": 0.031, "PH": 0.059, "ID": 0.020, "MY": 0.025, "TH": 0.025, "SG": 0.025, "IN": 0.025}

    def base_churn_prob(country, plan):
        if plan == "Premium":
            return premium_churn[country]
        return overall_churn[country]

    tenure_months = rng.integers(1, 37, size=n)
    watch_minutes_avg = np.clip(rng.normal(loc=85, scale=30, size=n), 5, None)
    days_since_last_watch = rng.integers(0, 60, size=n)
    support_tickets_count = rng.poisson(0.6, size=n)

    rows = []
    for i in range(n):
        base_p = base_churn_prob(countries[i], plans[i])
        # Behavioral multiplier: keeps each country/plan group centered near
        # its canonical base rate while giving the model real per-row signal.
        multiplier = 1.0
        if days_since_last_watch[i] > 21:
            multiplier *= 1.8
        if watch_minutes_avg[i] < 40:
            multiplier *= 1.6
        if support_tickets_count[i] >= 2:
            multiplier *= 1.5
        if tenure_months[i] < 3:
            multiplier *= 1.3
        if tenure_months[i] > 24:
            multiplier *= 0.6
        adj_p = np.clip(base_p * multiplier, 0.01, 0.95)
        churned = rng.random() < adj_p
        rows.append({
            "user_id": 1000 + i,
            "country": countries[i],
            "plan": plans[i],
            "watch_minutes_avg": round(float(watch_minutes_avg[i]), 1),
            "days_since_last_watch": int(days_since_last_watch[i]),
            "tenure_months": int(tenure_months[i]),
            "support_tickets_count": int(support_tickets_count[i]),
            "churned": int(churned),
        })

    return pd.DataFrame(rows)


def prepare_features(churn_df):
    """One-hot encode country/plan and split into X (features) and y (label)."""
    features_df = pd.get_dummies(
        churn_df.drop(columns=["user_id", "churned"]),
        columns=["country", "plan"],
        drop_first=True,
    )
    X = features_df.values
    y = churn_df["churned"].values
    return X, y, list(features_df.columns)


def train_baseline_classifier(X_train, y_train):
    """
    Train the Chapter 067 baseline: Logistic Regression on scaled features.

    Logistic regression is chosen over KNN as the baseline because it is
    interpretable (coefficients show direction/size of each feature's
    effect -- useful when stakeholders ask "why does the model think this
    user will churn?"), handles the mix of numeric and one-hot encoded
    categorical features well, and is the standard first model that any
    more complex algorithm (Chapter 068's trees, Chapter 102's XGBoost)
    needs to beat.
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    model = LogisticRegression(max_iter=1000, random_state=RANDOM_SEED)
    model.fit(X_train_scaled, y_train)

    return model, scaler


def train_tree_models(X_train, y_train, X_test, y_test, feature_names):
    """
    Chapter 068: train DecisionTreeClassifier and RandomForestClassifier on
    the same train/test split as the Chapter 067 baseline (no scaling needed
    -- tree-based models split on raw thresholds and are scale-invariant).

    Prints train/test accuracy for an unconstrained tree, a depth-limited
    tree, and a Random Forest, plus the Random Forest's top feature
    importances (descending).
    """
    print("\n--- Chapter 068: Decision Trees and Ensemble Methods ---")

    tree_default = DecisionTreeClassifier(random_state=RANDOM_SEED)
    tree_default.fit(X_train, y_train)
    print(
        f"Decision Tree (default, unconstrained): "
        f"train_acc={tree_default.score(X_train, y_train):.3f}, "
        f"test_acc={tree_default.score(X_test, y_test):.3f}"
    )

    tree_shallow = DecisionTreeClassifier(max_depth=3, random_state=RANDOM_SEED)
    tree_shallow.fit(X_train, y_train)
    print(
        f"Decision Tree (max_depth=3):            "
        f"train_acc={tree_shallow.score(X_train, y_train):.3f}, "
        f"test_acc={tree_shallow.score(X_test, y_test):.3f}"
    )

    rf = RandomForestClassifier(n_estimators=100, random_state=RANDOM_SEED)
    rf.fit(X_train, y_train)
    print(
        f"Random Forest (100 trees):              "
        f"train_acc={rf.score(X_train, y_train):.3f}, "
        f"test_acc={rf.score(X_test, y_test):.3f}"
    )

    print("\nRandom Forest feature importances (descending):")
    importances = rf.feature_importances_
    order = np.argsort(importances)[::-1]
    for i in order:
        print(f"  {feature_names[i]:25s} {importances[i]:.3f}")

    return tree_default, tree_shallow, rf


def main():
    churn_df = build_churn_dataset()

    print("Dataset shape:", churn_df.shape)
    print("Overall churn rate:", round(churn_df["churned"].mean(), 3))
    print()

    X, y, feature_names = prepare_features(churn_df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=RANDOM_SEED, stratify=y
    )
    print("Train churn rate:", round(y_train.mean(), 3))
    print("Test churn rate: ", round(y_test.mean(), 3))
    print()

    model, scaler = train_baseline_classifier(X_train, y_train)
    X_test_scaled = scaler.transform(X_test)

    accuracy = model.score(X_test_scaled, y_test)
    print("Logistic Regression test accuracy:", round(accuracy, 3))

    # KNN comparison (Chapter 067 Section 3)
    knn = KNeighborsClassifier(n_neighbors=5)
    X_train_scaled = scaler.transform(X_train)
    knn.fit(X_train_scaled, y_train)
    knn_accuracy = knn.score(X_test_scaled, y_test)
    print("KNN (k=5) test accuracy:        ", round(knn_accuracy, 3))

    # Majority-class baseline -- the honest comparison point on imbalanced data
    majority_class = int(round(y_train.mean()))
    baseline_preds = np.full_like(y_test, majority_class)
    baseline_accuracy = accuracy_score(y_test, baseline_preds)
    print("Majority-class baseline accuracy:", round(baseline_accuracy, 3))

    if accuracy <= baseline_accuracy:
        print()
        print("NOTE: the model did not beat the majority-class baseline on accuracy.")
        print("This is expected on a 7% churn rate -- accuracy alone cannot tell us")
        print("whether the model identifies the at-risk minority. See Chapter 072")
        print("(Metrics) for precision/recall/F1, which are the metrics that matter")
        print("for this use case.")

    # Chapter 068: Decision Trees and Ensemble Methods (no scaling needed)
    train_tree_models(X_train, y_train, X_test, y_test, feature_names)


if __name__ == "__main__":
    main()
