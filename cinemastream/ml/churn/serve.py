"""
cinemastream/ml/churn/serve.py

Chapter 077 -- ML Pipeline Integration: a FastAPI service that wraps the
churn champion behind a REST API, with the four production layers Part 7
has been building toward:

  1. A typed request contract (pydantic) -- the INPUT GUARDRAIL. Clients
     send raw user attributes; malformed or out-of-range requests are
     rejected with 422 before the model ever runs.
  2. Server-side feature engineering with FROZEN training constants --
     the same engagement_score formula (Ch069), tenure bucket (Ch069),
     and spend cap / per-plan imputation medians (Ch070) the model was
     trained on. Recomputing these at request time would be training-
     serving skew; we load them, we do not re-derive them.
  3. The champion loaded once at startup from a trusted artifact
     (Ch076 joblib), with the decision threshold (0.3) read from config,
     not hard-coded twice.
  4. STRUCTURED LOGGING of every prediction (request_id, latency_ms,
     inputs, probability, decision) + a /health readiness probe and a
     /metrics summary -- the observability hooks that tie this service
     back to Ch052 (alerting), Ch053 (drift), and Ch057 (incident
     response).

The held-out test set's LABELS are never used here; this is a serving
path, not an evaluation.

Run the API:
    uvicorn cinemastream.ml.churn.serve:app --reload
Then POST to http://127.0.0.1:8000/predict (see /docs for the schema).

Run the smoke test (no server needed, uses FastAPI TestClient):
    python cinemastream/ml/churn/serve.py
"""

import sys
sys.path.insert(0, ".")

import json
import logging
import time
import uuid
from collections import deque
from pathlib import Path
from typing import Literal

import joblib
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel, Field

from cinemastream.ml.churn.train import RANDOM_SEED
from cinemastream.ml.churn.hyperparameter_tuning import load_split
from cinemastream.ml.churn.model_comparison import TUNED_RF_PARAMS
from sklearn.ensemble import RandomForestClassifier

# ---- Config: everything the service is allowed to decide lives here -----
DECISION_THRESHOLD = 0.3          # Ch073's chosen cutoff; config, not literal
MODEL_VERSION = "cinemastream-churn@champion"   # Ch075 registry alias
ARTIFACT_PATH = Path("cinemastream/ml/churn/artifacts/champion_compressed.joblib")

# ---- Frozen training-time constants (Ch069/Ch070) -----------------------
# Serving MUST reuse these, never recompute them per request.
FEATURE_ORDER = ["watch_minutes_avg", "days_since_last_watch", "tenure_months",
                 "support_tickets_count", "engagement_score", "country_TH",
                 "country_VN", "tenure_bucket_established",
                 "spend_group_imputed_capped"]
SPEND_CAP = 38.1925                          # Ch070 IQR upper bound
SPEND_MEDIAN_BY_PLAN = {"Free": 0.0050, "Basic": 12.9100, "Premium": 19.9450}

# ---- Structured logging -------------------------------------------------
logger = logging.getLogger("churn-serve")
logger.setLevel(logging.INFO)
if not logger.handlers:
    _h = logging.StreamHandler(sys.stdout)
    _h.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(_h)

# In-memory rolling window of recent predictions -- the seed of the
# monitoring/drift surface (Ch053). A real deployment writes these to a
# table; here we keep the last 500 for the /metrics endpoint.
RECENT = deque(maxlen=500)


# ---- Request / response contracts (the input guardrail) -----------------
class ChurnRequest(BaseModel):
    """Raw user attributes. Ranges encode domain guardrails: a negative
    tenure or a 999-ticket user is a bad request, not a prediction."""
    country: Literal["SG", "MY", "ID", "PH", "TH", "VN", "IN"]
    plan: Literal["Free", "Basic", "Premium"]
    tenure_months: int = Field(ge=0, le=120)
    watch_minutes_avg: float = Field(ge=0, le=1000)
    days_since_last_watch: int = Field(ge=0, le=365)
    support_tickets_count: int = Field(ge=0, le=50)
    monthly_spend_sgd: float | None = Field(default=None, ge=0, le=200)


class ChurnResponse(BaseModel):
    request_id: str
    model_version: str
    churn_probability: float
    will_churn: bool
    threshold: float


# ---- Server-side feature engineering (frozen constants) -----------------
def engineer_serving_features(req: ChurnRequest) -> np.ndarray:
    """Turn raw attributes into the model's 9-feature vector, using the
    SAME formulas and frozen constants the model was trained on."""
    # engagement_score: Ch069 canonical formula
    engagement = round(req.watch_minutes_avg / (req.days_since_last_watch + 1), 3)
    # tenure bucket "established" = [12, 24) months: Ch069 bins
    established = 1.0 if 12 <= req.tenure_months < 24 else 0.0
    # spend: impute missing with the FROZEN per-plan training median,
    # then cap at the FROZEN IQR upper bound (Ch070) -- not recomputed.
    spend = (req.monthly_spend_sgd if req.monthly_spend_sgd is not None
             else SPEND_MEDIAN_BY_PLAN[req.plan])
    spend = min(spend, SPEND_CAP)
    vec = [
        req.watch_minutes_avg,
        float(req.days_since_last_watch),
        float(req.tenure_months),
        float(req.support_tickets_count),
        engagement,
        1.0 if req.country == "TH" else 0.0,
        1.0 if req.country == "VN" else 0.0,
        established,
        spend,
    ]
    return np.array([vec], dtype=object)


# ---- Model loading (once, at startup, from a trusted artifact) ----------
def load_champion():
    """Load the serialized champion. In production this is the only
    accepted source (Ch076 provenance rule); for a self-contained demo we
    regenerate the artifact from the registered params if it's missing."""
    if ARTIFACT_PATH.exists():
        return joblib.load(ARTIFACT_PATH)
    X_train, _, y_train, _, _ = load_split()
    model = RandomForestClassifier(**TUNED_RF_PARAMS).fit(X_train, y_train)
    ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, ARTIFACT_PATH, compress=3)
    return model


app = FastAPI(title="CinemaStream Churn Scoring", version="1.0.0")
MODEL = load_champion()


@app.get("/health")
def health():
    """Readiness probe: is the model loaded and answerable? (Ch052
    liveness-vs-correctness: a 200 here means 'can serve', not 'correct'.)"""
    return {"status": "ok", "model_version": MODEL_VERSION,
            "threshold": DECISION_THRESHOLD}


@app.post("/predict", response_model=ChurnResponse)
def predict(req: ChurnRequest):
    """Score one user. pydantic has already enforced the input contract;
    by the time we're here, the request is structurally valid."""
    request_id = str(uuid.uuid4())
    t0 = time.perf_counter()

    features = engineer_serving_features(req)
    proba = float(MODEL.predict_proba(features)[0, 1])
    will_churn = proba >= DECISION_THRESHOLD
    latency_ms = round((time.perf_counter() - t0) * 1000, 2)

    # Structured log line -- one JSON object per prediction, replayable.
    record = {
        "request_id": request_id,
        "model_version": MODEL_VERSION,
        "country": req.country, "plan": req.plan,
        "churn_probability": round(proba, 4),
        "will_churn": will_churn,
        "latency_ms": latency_ms,
    }
    logger.info(json.dumps(record))
    RECENT.append(record)

    return ChurnResponse(
        request_id=request_id, model_version=MODEL_VERSION,
        churn_probability=round(proba, 4), will_churn=will_churn,
        threshold=DECISION_THRESHOLD,
    )


@app.get("/metrics")
def metrics():
    """Monitoring surface (Ch053): summary stats over the recent window
    that a freshness/anomaly check (Ch052) or drift check (Ch053) reads.
    A flagged-rate far from the ~7% training base rate is the first signal
    of input drift or a broken upstream feature."""
    if not RECENT:
        return {"predictions": 0}
    probs = [r["churn_probability"] for r in RECENT]
    flagged = sum(r["will_churn"] for r in RECENT)
    return {
        "predictions": len(RECENT),
        "flagged_rate": round(flagged / len(RECENT), 4),
        "mean_churn_probability": round(float(np.mean(probs)), 4),
        "p95_latency_ms": round(float(np.percentile(
            [r["latency_ms"] for r in RECENT], 95)), 2),
        "model_version": MODEL_VERSION,
    }


def _smoke_test():
    """End-to-end check without a live server, via FastAPI's TestClient."""
    from fastapi.testclient import TestClient
    client = TestClient(app)

    print("GET /health ->", client.get("/health").json())
    print()

    # A real low-risk user from the test set (long tenure, engaged):
    # the model scores them well below the 0.3 threshold.
    loyal = {"country": "SG", "plan": "Basic", "tenure_months": 32,
             "watch_minutes_avg": 79.4, "days_since_last_watch": 36,
             "support_tickets_count": 2, "monthly_spend_sgd": 12.72}
    r1 = client.post("/predict", json=loyal).json()
    print("POST /predict (loyal user):")
    print(f"  prob={r1['churn_probability']}, will_churn={r1['will_churn']}")

    # A real churner from the test set (new, long-absent): the model
    # flags them -- one of the at-risk users Ch073's tuning made catchable.
    atrisk = {"country": "TH", "plan": "Basic", "tenure_months": 4,
              "watch_minutes_avg": 91.1, "days_since_last_watch": 55,
              "support_tickets_count": 0, "monthly_spend_sgd": 12.3}
    r2 = client.post("/predict", json=atrisk).json()
    print("POST /predict (at-risk user):")
    print(f"  prob={r2['churn_probability']}, will_churn={r2['will_churn']}")
    print()

    # Missing spend -> imputed from frozen per-plan median, still scores.
    no_spend = dict(atrisk); no_spend.pop("monthly_spend_sgd")
    r3 = client.post("/predict", json=no_spend).json()
    print(f"POST /predict (spend omitted -> imputed): "
          f"prob={r3['churn_probability']}, will_churn={r3['will_churn']}")

    # Input guardrail: negative tenure is rejected before the model runs.
    bad = dict(loyal); bad["tenure_months"] = -5
    r4 = client.post("/predict", json=bad)
    print(f"POST /predict (tenure_months=-5): HTTP {r4.status_code} "
          f"({r4.json()['detail'][0]['type']})")

    # Input guardrail: unknown country is rejected.
    bad2 = dict(loyal); bad2["country"] = "US"
    r5 = client.post("/predict", json=bad2)
    print(f"POST /predict (country='US'): HTTP {r5.status_code} "
          f"({r5.json()['detail'][0]['type']})")
    print()

    print("GET /metrics ->", client.get("/metrics").json())


if __name__ == "__main__":
    _smoke_test()
