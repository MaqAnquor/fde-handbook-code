"""
cinemastream/ml/churn/serialize.py

Chapter 076 -- Model Serialization: pickle, joblib, and ONNX.

Chapter 075 logged the churn champion to the MLflow registry and noted
(twice) that scikit-learn models are stored via pickle -- a format that
executes arbitrary code on load. This module takes that warning apart:
it serializes the Ch073 tuned-RF champion three ways, compares size and
fidelity, and exports an ONNX graph that runs without scikit-learn (or
Python) at inference time -- the format you hand to a system that isn't
ours.

What this demonstrates:
  - pickle / joblib: native Python, fast, full-fidelity, but unsafe to
    load from an untrusted source and version-coupled to the libraries
    that wrote them,
  - joblib with compression: same fidelity, a fraction of the size,
  - ONNX: a portable computation graph, runnable from any language via
    onnxruntime, with predictions verified to match sklearn within
    floating-point tolerance.

The held-out test set's LABELS are not used; fidelity checks compare
each serialized model's predictions to the in-memory model's, on test
FEATURES only.

Run (requires `pip install joblib skl2onnx onnxruntime`):
    python cinemastream/ml/churn/serialize.py
"""

import sys
sys.path.insert(0, ".")

import warnings
warnings.filterwarnings("ignore")

import os
import pickle
import time
from pathlib import Path

import joblib
import numpy as np
import onnxruntime as ort
from skl2onnx import to_onnx
from sklearn.ensemble import RandomForestClassifier

from cinemastream.ml.churn.train import RANDOM_SEED
from cinemastream.ml.churn.hyperparameter_tuning import load_split
from cinemastream.ml.churn.model_comparison import TUNED_RF_PARAMS

# Where serialized champions go. Gitignored (see note in main()).
ARTIFACT_DIR = Path("cinemastream/ml/churn/artifacts")
DECISION_THRESHOLD = 0.3  # Ch073's chosen threshold; travels WITH the model.


def fit_champion(X_train, y_train):
    """The Ch073 tuned RF -- same params as the @champion registry entry."""
    return RandomForestClassifier(**TUNED_RF_PARAMS).fit(X_train, y_train)


def serialize_native(model):
    """pickle vs joblib vs compressed joblib: size and write time."""
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    paths = {}

    pkl = ARTIFACT_DIR / "champion.pkl"
    t0 = time.perf_counter()
    with open(pkl, "wb") as f:
        pickle.dump(model, f)
    paths["pickle"] = (pkl, time.perf_counter() - t0)

    jl = ARTIFACT_DIR / "champion.joblib"
    t0 = time.perf_counter()
    joblib.dump(model, jl)
    paths["joblib"] = (jl, time.perf_counter() - t0)

    jlc = ARTIFACT_DIR / "champion_compressed.joblib"
    t0 = time.perf_counter()
    joblib.dump(model, jlc, compress=3)
    paths["joblib (compress=3)"] = (jlc, time.perf_counter() - t0)

    print(f"{'format':22s} {'size':>10s} {'write':>10s}")
    for label, (path, secs) in paths.items():
        print(f"{label:22s} {os.path.getsize(path) / 1024:>7.1f} KB "
              f"{secs * 1000:>7.1f} ms")
    return paths


def verify_native(model, paths, X_test):
    """Each native format must reload to byte-identical predictions."""
    for label, (path, _) in paths.items():
        loaded = joblib.load(path) if "joblib" in label else pickle.load(
            open(path, "rb"))
        same = np.array_equal(model.predict_proba(X_test),
                              loaded.predict_proba(X_test))
        print(f"{label:22s} reload identical predict_proba: {same}")


def export_onnx(model, X_test):
    """Export to ONNX and verify the runtime matches sklearn. zipmap=False
    makes the probability output a plain array instead of a list-of-dicts."""
    onx = to_onnx(model, X_test[:1].astype(np.float32),
                  options={id(model): {"zipmap": False}})
    onnx_path = ARTIFACT_DIR / "champion.onnx"
    onnx_path.write_bytes(onx.SerializeToString())
    print(f"ONNX graph written: {os.path.getsize(onnx_path) / 1024:.1f} KB")

    sess = ort.InferenceSession(str(onnx_path),
                                providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    labels, probas = sess.run(None, {input_name: X_test.astype(np.float32)})

    sk_proba = model.predict_proba(X_test)
    print(f"ONNX labels match sklearn:          "
          f"{np.array_equal(labels, model.predict(X_test))}")
    print(f"ONNX probabilities match (atol=1e-4): "
          f"{np.allclose(probas, sk_proba, atol=1e-4)}")
    print(f"max probability difference:          "
          f"{np.abs(probas - sk_proba).max():.2e}")

    # The decision that matters for CinemaStream is "churn vs not at 0.3" --
    # confirm the threshold rule survives the ONNX round trip exactly.
    sk_flags = (sk_proba[:, 1] >= DECISION_THRESHOLD).astype(int)
    onnx_flags = (probas[:, 1] >= DECISION_THRESHOLD).astype(int)
    print(f"churn flags @ {DECISION_THRESHOLD} identical (sklearn vs ONNX): "
          f"{np.array_equal(sk_flags, onnx_flags)}")
    return onnx_path


def main():
    X_train, X_test, y_train, y_test, names = load_split()
    model = fit_champion(X_train, y_train)
    print(f"Champion fitted on {len(y_train)} rows; "
          f"serializing to {ARTIFACT_DIR}/ (gitignored).")
    print()

    print("=== Native formats: pickle vs joblib ===")
    paths = serialize_native(model)
    print()
    verify_native(model, paths, X_test)
    print()

    print("=== Portable format: ONNX (no sklearn at inference) ===")
    export_onnx(model, X_test)


if __name__ == "__main__":
    main()
