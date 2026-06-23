"""
cinemastream/ml/churn/model_comparison.py

Chapter 074 -- Model Comparison and the Bias-Variance Tradeoff:
under/overfitting, learning curves.

Ch073 tuned one model family (Random Forest) and left two questions open:
(1) would a different model family beat the tuned RF under the same
honest protocol, and (2) is the model data-starved -- would more churned
users tighten the recall@0.3 spread of +/-0.27 across folds?

Protocol notes (inherited from Ch073, unchanged):
  - All comparisons are scored by StratifiedKFold(5) cross-validation on
    the TRAINING rows of the canonical 75/25 split (RANDOM_SEED=42).
  - Model families are compared on ROC-AUC (threshold-free ranking
    quality): per-family decision thresholds would each need their own
    Ch073-style tuning, so comparing F1 at the default 0.5 across
    families is shown -- and then shown to be misleading.
  - The held-out test set is NOT touched in this chapter. Ch073 spent
    its one sealed look; re-opening it to compare families would turn
    it into a validation set.

Run:
    python cinemastream/ml/churn/model_comparison.py
"""

import sys
sys.path.insert(0, ".")

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, make_scorer, recall_score, roc_auc_score
from sklearn.model_selection import cross_val_score, learning_curve
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from cinemastream.ml.churn.train import RANDOM_SEED
from cinemastream.ml.churn.hyperparameter_tuning import CV, load_split

# The Ch073 winner, frozen. Any change here breaks canon.
TUNED_RF_PARAMS = dict(
    n_estimators=389, max_depth=5, min_samples_leaf=7, min_samples_split=7,
    max_features=None, class_weight="balanced", random_state=RANDOM_SEED,
)


def build_candidates():
    """One representative per model family, each configured as fairly as
    the family allows on imbalanced data (class_weight where supported,
    scaling via Pipeline where the family needs it)."""
    return {
        "Tuned Random Forest (Ch073)":
            RandomForestClassifier(**TUNED_RF_PARAMS),
        "Logistic Regression (balanced, scaled)":
            Pipeline([("scaler", StandardScaler()),
                      ("clf", LogisticRegression(class_weight="balanced",
                                                 max_iter=1000,
                                                 random_state=RANDOM_SEED))]),
        "Decision Tree (depth=3, balanced)":
            DecisionTreeClassifier(max_depth=3, class_weight="balanced",
                                   random_state=RANDOM_SEED),
        "Gradient Boosting (defaults)":
            GradientBoostingClassifier(random_state=RANDOM_SEED),
        "KNN (k=5, scaled)":
            Pipeline([("scaler", StandardScaler()),
                      ("clf", KNeighborsClassifier(n_neighbors=5))]),
    }


def compare_models(X_train, y_train):
    """Same folds, same rows, two metrics per family: threshold-free
    ROC-AUC (the fair comparison) and F1 at the default 0.5 threshold
    (shown to demonstrate why it is NOT the fair comparison)."""
    print(f"{'Model':40s} {'CV ROC-AUC':>18s} {'CV F1 @ 0.5':>18s}")
    results = {}
    for name, model in build_candidates().items():
        auc = cross_val_score(model, X_train, y_train, scoring="roc_auc", cv=CV)
        f1 = cross_val_score(model, X_train, y_train, scoring="f1", cv=CV)
        print(f"{name:40s} {auc.mean():>10.3f} +/- {auc.std():.3f} "
              f"{f1.mean():>10.3f} +/- {f1.std():.3f}")
        results[name] = (auc, f1)
    return results


def diagnose_bias_variance(X_train, y_train):
    """Train-score vs CV-score gap: the practical bias-variance read.
    A large gap = variance (memorising); both low = bias (too simple)."""
    probes = {
        "Unconstrained RF (Ch068 defaults)":
            RandomForestClassifier(n_estimators=100, random_state=RANDOM_SEED),
        "Tuned RF (Ch073, regularised)":
            RandomForestClassifier(**TUNED_RF_PARAMS),
        "Decision stump (max_depth=1, balanced)":
            DecisionTreeClassifier(max_depth=1, class_weight="balanced",
                                   random_state=RANDOM_SEED),
    }
    print(f"{'Model':40s} {'train ROC-AUC':>14s} {'CV ROC-AUC':>18s} {'gap':>7s}")
    for name, model in probes.items():
        model.fit(X_train, y_train)
        train_auc = roc_auc_score(y_train, model.predict_proba(X_train)[:, 1])
        cv_auc = cross_val_score(model, X_train, y_train, scoring="roc_auc", cv=CV)
        print(f"{name:40s} {train_auc:>14.3f} {cv_auc.mean():>10.3f} +/- {cv_auc.std():.3f}"
              f" {train_auc - cv_auc.mean():>7.3f}")


def run_learning_curve(X_full, y_full, scoring, label):
    """Learning curve for the tuned RF on all 300 rows (Ch071 protocol
    folds): does validation performance still rise with more rows?"""
    model = RandomForestClassifier(**TUNED_RF_PARAMS)
    sizes, train_scores, val_scores = learning_curve(
        model, X_full, y_full, cv=CV, scoring=scoring,
        train_sizes=np.linspace(0.3, 1.0, 6), shuffle=True,
        random_state=RANDOM_SEED,
    )
    print(f"Learning curve -- {label}:")
    print(f"{'train rows':>10s} {'train score':>12s} {'val score':>22s}")
    for n, tr, va in zip(sizes, train_scores, val_scores):
        print(f"{n:>10d} {tr.mean():>12.3f} {va.mean():>12.3f} +/- {va.std():.3f}")
    return sizes, train_scores, val_scores


def main():
    X_train, X_test, y_train, y_test, names = load_split()
    print(f"Comparison data: the {len(y_train)} training rows "
          f"({y_train.sum()} churned). Test set: untouched this chapter.")
    print()

    print("=== Model family comparison (same folds, same rows) ===")
    compare_models(X_train, y_train)
    print()

    print("=== Bias-variance read: train vs CV gap ===")
    diagnose_bias_variance(X_train, y_train)
    print()

    # Learning curves run on all 300 rows: the question is about data
    # volume, and CV's fold structure already keeps scoring honest.
    X_full = np.vstack([X_train, X_test])
    y_full = np.concatenate([y_train, y_test])

    recall_at_03 = make_scorer(
        lambda yt, yp: recall_score(yt, (yp >= 0.3).astype(int),
                                    zero_division=0),
        response_method="predict_proba")

    print("=== Is the model data-starved? (Ch073 handoff) ===")
    run_learning_curve(X_full, y_full, "roc_auc", "ROC-AUC")
    print()
    run_learning_curve(X_full, y_full, recall_at_03, "recall @ threshold 0.3")


if __name__ == "__main__":
    main()
