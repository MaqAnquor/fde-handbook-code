"""
cinemastream/ml/churn/metrics.py

Chapter 072 -- Metrics: Accuracy, Precision, Recall, F1, AUC-ROC, Log-Loss,
MAE, MSE.

Five chapters (067-071) reported test_acc=0.933 and treated it as the
headline number. This module finally asks what that number is hiding:
it computes precision, recall, F1, ROC-AUC, and log-loss for both the
majority-class baseline and the Ch070 v3 (9-feature) Random Forest on
the same RANDOM_SEED=42 single split, and runs a decision-threshold
sweep on the Random Forest's predicted probabilities.

Run:
    python cinemastream/ml/churn/metrics.py
"""

import sys
sys.path.insert(0, ".")

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    log_loss,
    classification_report,
)

from cinemastream.ml.churn.train import build_churn_dataset, RANDOM_SEED
from cinemastream.ml.churn.data_quality import (
    inject_billing_data,
    impute_spend,
    detect_and_cap_outliers,
    prepare_features_v3,
)


def evaluate_classifier(y_true, y_pred, y_proba=None, label="model"):
    """
    Print accuracy, precision, recall, F1, confusion matrix, and (if
    y_proba is given) ROC-AUC and log-loss for a binary classifier's
    predictions on the "churned" (positive) class.
    """
    print(f"--- {label} ---")
    print(f"Accuracy:  {accuracy_score(y_true, y_pred):.3f}")
    print(f"Precision: {precision_score(y_true, y_pred, zero_division=0):.3f}")
    print(f"Recall:    {recall_score(y_true, y_pred, zero_division=0):.3f}")
    print(f"F1:        {f1_score(y_true, y_pred, zero_division=0):.3f}")
    if y_proba is not None:
        print(f"ROC-AUC:   {roc_auc_score(y_true, y_proba):.3f}")
        print(f"Log-loss:  {log_loss(y_true, y_proba):.3f}")
    print(f"Confusion matrix [[TN FP] [FN TP]]:\n{confusion_matrix(y_true, y_pred)}")


def threshold_sweep(y_true, y_proba, thresholds=(0.5, 0.4, 0.3, 0.2, 0.1)):
    """
    Print precision/recall/F1/accuracy for the given probability-decision
    thresholds. Lowering the threshold trades accuracy for recall on the
    minority (churned) class.
    """
    print("Threshold sweep:")
    for t in thresholds:
        pred_t = (y_proba >= t).astype(int)
        print(f"  threshold={t:.1f}: "
              f"precision={precision_score(y_true, pred_t, zero_division=0):.3f}, "
              f"recall={recall_score(y_true, pred_t, zero_division=0):.3f}, "
              f"f1={f1_score(y_true, pred_t, zero_division=0):.3f}, "
              f"accuracy={accuracy_score(y_true, pred_t):.3f}")


def main():
    churn_df = build_churn_dataset()
    billed_df = inject_billing_data(churn_df)
    imputed_df, _ = impute_spend(billed_df)
    capped_df, _, _, _ = detect_and_cap_outliers(imputed_df)

    X_v2, X_v3, y, names_v2, names_v3 = prepare_features_v3(capped_df)

    X_train, X_test, y_train, y_test = train_test_split(
        X_v3, y, test_size=0.25, random_state=RANDOM_SEED, stratify=y
    )
    print(f"Test set: {len(y_test)} rows, {y_test.sum()} churned ({y_test.mean():.1%})")
    print()

    # Majority-class baseline
    y_pred_baseline = np.zeros_like(y_test)
    evaluate_classifier(y_test, y_pred_baseline, label="Majority-class baseline")
    print()

    # Random Forest (v3, 9 features) -- same model as Ch067-071
    rf = RandomForestClassifier(n_estimators=100, random_state=RANDOM_SEED)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    y_proba_rf = rf.predict_proba(X_test)[:, 1]
    evaluate_classifier(y_test, y_pred_rf, y_proba_rf, label="Random Forest (v3, 9 features)")
    print()

    print(classification_report(
        y_test, y_pred_rf, target_names=["not churned", "churned"], zero_division=0
    ))

    threshold_sweep(y_test, y_proba_rf)


if __name__ == "__main__":
    main()
