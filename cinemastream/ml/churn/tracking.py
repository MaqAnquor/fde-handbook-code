"""
cinemastream/ml/churn/tracking.py

Chapter 075 -- ML Experiment Tracking: MLflow, the model registry,
artifact management, and reproducibility.

Chapters 073-074 trained 130+ model configurations (48 grid candidates,
30 random candidates, 40 Optuna trials, 5 model families, 3 bias-variance
probes, 12 learning-curve points) and recorded none of them anywhere
queryable. This module gives the churn project a tracking home:

  - one MLflow experiment ("cinemastream-churn") on a SQLite backend
    (MLflow 3.x has put the filesystem backend into maintenance mode;
    a database URI is the supported default),
  - the Ch073/074 reference models re-run and logged (params, CV
    metrics, tags),
  - the Ch073 Optuna study re-run with every trial logged as a run --
    making "which config was trial 23?" a one-line query,
  - the tuned champion logged, registered as "cinemastream-churn",
    aliased @champion (Gradient Boosting registered as @challenger),
    and loaded back via the alias to prove byte-for-byte reproducibility.

The held-out test set's LABELS are not used anywhere in this module;
the reproducibility check compares two models' predictions to each
other on test features only.

Run (requires `pip install mlflow`):
    python cinemastream/ml/churn/tracking.py
"""

import sys
sys.path.insert(0, ".")

import logging
import os
import warnings
from pathlib import Path

import numpy as np

os.environ.setdefault("MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR", "false")
warnings.filterwarnings("ignore")
logging.getLogger("mlflow").setLevel(logging.ERROR)

import mlflow
import mlflow.sklearn
import optuna
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import cross_val_score

from cinemastream.ml.churn.train import RANDOM_SEED
from cinemastream.ml.churn.hyperparameter_tuning import CV, load_split, tune_optuna
from cinemastream.ml.churn.model_comparison import TUNED_RF_PARAMS

TRACKING_URI = "sqlite:///cinemastream/ml/mlflow.db"
EXPERIMENT = "cinemastream-churn"
REGISTERED_NAME = "cinemastream-churn"


def setup_tracking():
    """Point MLflow at the project database and experiment. The artifact
    store (model files, plots) lives in cinemastream/ml/mlruns/ -- both
    paths are gitignored (Ch017's .gitignore already covers mlruns/)."""
    mlflow.set_tracking_uri(TRACKING_URI)
    artifact_location = Path("cinemastream/ml/mlruns").absolute().as_uri()
    if mlflow.get_experiment_by_name(EXPERIMENT) is None:
        mlflow.create_experiment(EXPERIMENT, artifact_location=artifact_location)
    mlflow.set_experiment(EXPERIMENT)


def log_reference_runs(X_train, y_train):
    """Re-run and log the three models the team actually talks about:
    the Ch068-era default RF, the Ch073 tuned champion, and the Ch074
    Gradient Boosting challenger."""
    references = {
        "default-rf-ch068": (
            RandomForestClassifier(n_estimators=100, random_state=RANDOM_SEED),
            {"source_chapter": "068", "role": "historical-baseline"}),
        "tuned-rf-ch073": (
            RandomForestClassifier(**TUNED_RF_PARAMS),
            {"source_chapter": "073", "role": "champion"}),
        "gradient-boosting-ch074": (
            GradientBoostingClassifier(random_state=RANDOM_SEED),
            {"source_chapter": "074", "role": "challenger-untuned"}),
    }
    for run_name, (model, tags) in references.items():
        auc = cross_val_score(model, X_train, y_train, scoring="roc_auc", cv=CV)
        f1 = cross_val_score(model, X_train, y_train, scoring="f1", cv=CV)
        with mlflow.start_run(run_name=run_name):
            mlflow.log_params(model.get_params())
            mlflow.log_metric("cv_roc_auc", auc.mean())
            mlflow.log_metric("cv_roc_auc_std", auc.std())
            mlflow.log_metric("cv_f1", f1.mean())
            mlflow.log_metric("cv_f1_std", f1.std())
            mlflow.set_tags(tags)
        print(f"logged {run_name}: cv_roc_auc={auc.mean():.3f}, cv_f1={f1.mean():.3f}")


def log_optuna_study(X_train, y_train):
    """Re-run the Ch073 Optuna search (seeded -- identical results) and
    log every trial as its own MLflow run."""
    study, cw_options = tune_optuna(X_train, y_train)
    for trial in study.trials:
        with mlflow.start_run(run_name=f"optuna-trial-{trial.number:02d}"):
            mlflow.log_params(trial.params)
            mlflow.log_metric("cv_f1", trial.value)
            mlflow.set_tags({"search": "optuna-tpe-ch073",
                             "trial_number": str(trial.number)})
    print(f"logged {len(study.trials)} Optuna trials "
          f"(best: trial {study.best_trial.number}, cv_f1={study.best_value:.3f})")
    return study


def query_trial(trial_number):
    """The Ch074 closing question, answered: which config was trial N?"""
    runs = mlflow.search_runs(
        filter_string=f"tags.trial_number = '{trial_number}'")
    cols = ["tags.mlflow.runName", "params.class_weight", "params.max_depth",
            "params.n_estimators", "params.min_samples_leaf", "metrics.cv_f1"]
    print(runs[cols].to_string(index=False))


def register_champion_and_challenger(X_train, y_train, X_test):
    """Fit, log, and register the champion (tuned RF) and challenger
    (default GB); alias them; reload the champion via its alias and
    prove the round trip is lossless on test FEATURES (no labels)."""
    champion = RandomForestClassifier(**TUNED_RF_PARAMS).fit(X_train, y_train)
    with mlflow.start_run(run_name="champion-fit"):
        mlflow.log_params(TUNED_RF_PARAMS)
        mlflow.set_tag("decision_threshold", "0.3")
        info = mlflow.sklearn.log_model(champion, name="model",
                                        input_example=X_train[:2])
    version = mlflow.register_model(info.model_uri, REGISTERED_NAME).version

    client = mlflow.MlflowClient()
    client.set_registered_model_alias(REGISTERED_NAME, "champion", version)
    client.set_model_version_tag(REGISTERED_NAME, version,
                                 "decision_threshold", "0.3")
    print(f"registered {REGISTERED_NAME} v{version} -> @champion "
          f"(decision_threshold=0.3)")

    challenger = GradientBoostingClassifier(random_state=RANDOM_SEED).fit(
        X_train, y_train)
    with mlflow.start_run(run_name="challenger-fit"):
        mlflow.log_params(challenger.get_params())
        info_c = mlflow.sklearn.log_model(challenger, name="model",
                                          input_example=X_train[:2])
    version_c = mlflow.register_model(info_c.model_uri, REGISTERED_NAME).version
    client.set_registered_model_alias(REGISTERED_NAME, "challenger", version_c)
    print(f"registered {REGISTERED_NAME} v{version_c} -> @challenger")

    # Reproducibility round trip: load by alias, compare predictions.
    loaded = mlflow.sklearn.load_model(f"models:/{REGISTERED_NAME}@champion")
    same = np.array_equal(champion.predict_proba(X_test),
                          loaded.predict_proba(X_test))
    print(f"champion reloaded from registry, identical predict_proba "
          f"on test features: {same}")


def main():
    setup_tracking()
    X_train, X_test, y_train, y_test, names = load_split()

    print("=== Reference runs ===")
    log_reference_runs(X_train, y_train)
    print()

    print("=== Optuna study, every trial tracked ===")
    log_optuna_study(X_train, y_train)
    print()

    print("=== 'Which config was trial 23?' ===")
    query_trial(23)
    print()

    print("=== Registry: champion + challenger ===")
    register_champion_and_challenger(X_train, y_train, X_test)


if __name__ == "__main__":
    main()
