"""
cinemastream/ml/churn/hyperparameter_tuning.py

Chapter 073 -- Hyperparameter Tuning: Grid Search, Random Search, and
Bayesian Optimization (Optuna).

Ch072 ended with a confusion matrix identical to the majority-class
baseline's: the Ch067-071 Random Forest has never predicted a single
user as "will churn" (recall=0.000 at the default 0.5 threshold). This
module asks whether tuning -- class_weight, tree hyperparameters, and
the decision threshold itself -- can get recall on the "churned" class
above 0 without giving up the stability established in Ch071.

Selection protocol (resolves Ch071's open design decision):
  - Hyperparameters are compared via StratifiedKFold(5) cross-validation
    ON THE TRAINING SET ONLY (the same canonical 75/25 stratified split,
    RANDOM_SEED=42, that every chapter since 067 has used).
  - The decision threshold is chosen on out-of-fold training
    probabilities (cross_val_predict), never on the test set.
  - The held-out test set is touched exactly once, at the end, to report
    the final tuned model. No hyperparameter or threshold was chosen
    using it.

Run (requires `pip install optuna`):
    python cinemastream/ml/churn/hyperparameter_tuning.py
"""

import sys
sys.path.insert(0, ".")

import numpy as np
from scipy.stats import randint
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_predict,
    cross_val_score,
    train_test_split,
)

import optuna

from cinemastream.ml.churn.train import build_churn_dataset, RANDOM_SEED
from cinemastream.ml.churn.data_quality import (
    inject_billing_data,
    impute_spend,
    detect_and_cap_outliers,
    prepare_features_v3,
)
from cinemastream.ml.churn.metrics import evaluate_classifier, threshold_sweep

# One CV object, reused everywhere a hyperparameter decision is made --
# same n_splits/shuffle/random_state as Ch071's run_cross_validation().
CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

# class_weight candidates: None is the Ch067-072 status quo; "balanced"
# reweights by inverse class frequency; the dicts make the minority
# ("churned") class 5x / 10x more expensive to misclassify in training.
CLASS_WEIGHTS = [None, "balanced", {0: 1, 1: 5}, {0: 1, 1: 10}]

PARAM_GRID = {
    "class_weight": CLASS_WEIGHTS,
    "max_depth": [None, 3, 5, 8],
    "min_samples_leaf": [1, 2, 4],
}


def load_split():
    """Rebuild the canonical Ch070 v3 feature matrix and the canonical
    75/25 stratified split (RANDOM_SEED=42) used since Ch067."""
    churn_df = build_churn_dataset()
    billed_df = inject_billing_data(churn_df)
    imputed_df, _ = impute_spend(billed_df)
    capped_df, _, _, _ = detect_and_cap_outliers(imputed_df)
    X_v2, X_v3, y, names_v2, names_v3 = prepare_features_v3(capped_df)
    X_train, X_test, y_train, y_test = train_test_split(
        X_v3, y, test_size=0.25, random_state=RANDOM_SEED, stratify=y
    )
    return X_train, X_test, y_train, y_test, names_v3


def tune_grid(X_train, y_train, scoring="f1"):
    """Exhaustive grid search over PARAM_GRID (48 candidates x 5 folds)."""
    rf = RandomForestClassifier(n_estimators=100, random_state=RANDOM_SEED)
    search = GridSearchCV(rf, PARAM_GRID, scoring=scoring, cv=CV)
    search.fit(X_train, y_train)
    return search


def tune_random(X_train, y_train, n_iter=30, scoring="f1"):
    """Random search over a wider space, fixed budget of n_iter candidates."""
    space = {
        "n_estimators": randint(50, 401),
        "max_depth": [None, 3, 5, 8, 12],
        "min_samples_leaf": randint(1, 9),
        "min_samples_split": randint(2, 11),
        "max_features": ["sqrt", "log2", None],
        "class_weight": CLASS_WEIGHTS,
    }
    rf = RandomForestClassifier(random_state=RANDOM_SEED)
    search = RandomizedSearchCV(
        rf, space, n_iter=n_iter, scoring=scoring, cv=CV,
        random_state=RANDOM_SEED,
    )
    search.fit(X_train, y_train)
    return search


def tune_optuna(X_train, y_train, n_trials=40, scoring="f1"):
    """Bayesian optimization (TPE sampler) over the same space as
    tune_random(), same per-candidate cost, seeded for reproducibility.

    suggest_categorical only accepts None/bool/int/float/str, so the dict
    class weights are passed as string keys and mapped back here."""
    cw_options = {"none": None, "balanced": "balanced",
                  "w5": {0: 1, 1: 5}, "w10": {0: 1, 1: 10}}

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 400),
            "max_depth": trial.suggest_categorical("max_depth", [None, 3, 5, 8, 12]),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 8),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
        }
        cw_key = trial.suggest_categorical("class_weight", list(cw_options))
        rf = RandomForestClassifier(
            **params, class_weight=cw_options[cw_key], random_state=RANDOM_SEED
        )
        return cross_val_score(rf, X_train, y_train, scoring=scoring, cv=CV).mean()

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    sampler = optuna.samplers.TPESampler(seed=RANDOM_SEED)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials)
    return study, cw_options


def pick_threshold(model, X_train, y_train, thresholds=(0.5, 0.4, 0.3, 0.2, 0.1)):
    """Choose a decision threshold by F1 on OUT-OF-FOLD training
    probabilities. The test set plays no part in this decision."""
    proba_oof = cross_val_predict(
        model, X_train, y_train, cv=CV, method="predict_proba"
    )[:, 1]
    print("Out-of-fold (training) threshold sweep:")
    threshold_sweep(y_train, proba_oof, thresholds=thresholds)
    from sklearn.metrics import f1_score
    f1s = {t: f1_score(y_train, (proba_oof >= t).astype(int), zero_division=0)
           for t in thresholds}
    best_t = max(f1s, key=f1s.get)
    print(f"Chosen threshold (best OOF F1): {best_t}")
    return best_t


def main():
    X_train, X_test, y_train, y_test, names = load_split()
    print(f"Train: {len(y_train)} rows ({y_train.sum()} churned), "
          f"Test: {len(y_test)} rows ({y_test.sum()} churned)")
    print()

    # --- Step 1: the trap -- tuning on accuracy ---------------------------
    print("=== Grid search, scoring='accuracy' (the trap) ===")
    grid_acc = tune_grid(X_train, y_train, scoring="accuracy")
    print(f"Best params: {grid_acc.best_params_}")
    print(f"Best CV accuracy: {grid_acc.best_score_:.3f}")
    rec = cross_val_score(grid_acc.best_estimator_, X_train, y_train,
                          scoring="recall", cv=CV)
    print(f"That 'best' model's CV recall: {np.round(rec, 3)} (mean {rec.mean():.3f})")
    print()

    # --- Step 2: grid search, scoring='f1' --------------------------------
    print("=== Grid search, scoring='f1' (48 candidates x 5 folds) ===")
    grid_f1 = tune_grid(X_train, y_train, scoring="f1")
    print(f"Best params: {grid_f1.best_params_}")
    print(f"Best CV F1: {grid_f1.best_score_:.3f}")
    print()

    # --- Step 3: random search, wider space, 30-candidate budget ----------
    print("=== Random search, scoring='f1' (30 candidates x 5 folds) ===")
    rand_f1 = tune_random(X_train, y_train)
    print(f"Best params: {rand_f1.best_params_}")
    print(f"Best CV F1: {rand_f1.best_score_:.3f}")
    print()

    # --- Step 4: Bayesian optimization (Optuna TPE), 40 trials ------------
    print("=== Optuna (TPE), scoring='f1' (40 trials x 5 folds) ===")
    study, cw_options = tune_optuna(X_train, y_train)
    print(f"Best params: {study.best_params}")
    print(f"Best CV F1: {study.best_value:.3f}")
    print()

    # --- Step 5: final model = best CV F1 across all three searches -------
    candidates = {
        "grid": (grid_f1.best_score_, grid_f1.best_estimator_),
        "random": (rand_f1.best_score_, rand_f1.best_estimator_),
        "optuna": (study.best_value, None),  # estimator rebuilt below if it wins
    }
    winner = max(candidates, key=lambda k: candidates[k][0])
    if winner == "optuna":
        bp = dict(study.best_params)
        bp["class_weight"] = cw_options[bp["class_weight"]]
        final_model = RandomForestClassifier(**bp, random_state=RANDOM_SEED)
    else:
        final_model = candidates[winner][1]
    print(f"Winning search: {winner} (CV F1 {candidates[winner][0]:.3f})")
    print(f"Final model: {final_model}")
    print()

    # --- Step 6: threshold chosen on out-of-fold TRAIN probabilities ------
    best_t = pick_threshold(final_model, X_train, y_train)
    print()

    # --- Step 7: the single, final look at the test set -------------------
    final_model.fit(X_train, y_train)
    proba_test = final_model.predict_proba(X_test)[:, 1]
    print(f"=== Final evaluation on the held-out test set (threshold={best_t}) ===")
    evaluate_classifier(y_test, (proba_test >= best_t).astype(int),
                        proba_test, label=f"Tuned RF @ threshold {best_t}")
    print()
    evaluate_classifier(y_test, (proba_test >= 0.5).astype(int),
                        label="Tuned RF @ default threshold 0.5 (for comparison)")
    print()

    # --- Step 8: stability check (Ch071 framing) --------------------------
    # Same StratifiedKFold(5) protocol as Ch071's canonical 0.933 +/- 0.011,
    # run on the full dataset for comparability with that number. The
    # deployed configuration is "tuned RF at threshold 0.3", so recall/F1
    # are scored at that threshold (via predict_proba), not the 0.5 that
    # .predict() would silently use.
    from sklearn.metrics import f1_score, make_scorer, recall_score

    def _recall_at_t(y_true, y_proba, t=best_t):
        return recall_score(y_true, (y_proba >= t).astype(int), zero_division=0)

    def _f1_at_t(y_true, y_proba, t=best_t):
        return f1_score(y_true, (y_proba >= t).astype(int), zero_division=0)

    recall_scorer = make_scorer(_recall_at_t, response_method="predict_proba")
    f1_scorer = make_scorer(_f1_at_t, response_method="predict_proba")

    X_full = np.vstack([X_train, X_test])
    y_full = np.concatenate([y_train, y_test])
    acc = cross_val_score(final_model, X_full, y_full, cv=CV)
    rec = cross_val_score(final_model, X_full, y_full, scoring=recall_scorer, cv=CV)
    f1 = cross_val_score(final_model, X_full, y_full, scoring=f1_scorer, cv=CV)
    print(f"Tuned model @ threshold {best_t}, StratifiedKFold(5) on all 300 rows "
          f"(Ch071 protocol):")
    print(f"  accuracy (at 0.5, as Ch071 measured it): "
          f"fold scores={np.round(acc, 3)}, "
          f"mean={acc.mean():.3f} +/- std={acc.std():.3f}")
    print(f"  recall @ {best_t}:   fold scores={np.round(rec, 3)}, "
          f"mean={rec.mean():.3f} +/- std={rec.std():.3f}")
    print(f"  f1 @ {best_t}:       fold scores={np.round(f1, 3)}, "
          f"mean={f1.mean():.3f} +/- std={f1.std():.3f}")


if __name__ == "__main__":
    main()
