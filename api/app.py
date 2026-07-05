"""
FastAPI Application

Loads:
1. Trained model
2. Standard Scaler

Provides:
/predict endpoint
"""

from pathlib import Path
from time import perf_counter

import joblib
import pandas as pd

from fastapi import FastAPI, Request
from fastapi.responses import Response
from pydantic import BaseModel

from prometheus_client import (
    Counter,
    Histogram,
    generate_latest,
    CONTENT_TYPE_LATEST,
)

from src.utils import get_logger


# ---------------------------------------------------------
# Project Paths
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = PROJECT_ROOT / "models" / "best_model.pkl"

SCALER_PATH = PROJECT_ROOT / "artifacts" / "scaler.pkl"


# ---------------------------------------------------------
# Load Model and Scaler
# ---------------------------------------------------------

model = joblib.load(MODEL_PATH)

scaler = joblib.load(SCALER_PATH)

logger = get_logger("FastAPI")
# ---------------------------------------------------------
# FastAPI App
# ---------------------------------------------------------
REQUEST_COUNT = Counter(
    "http_requests_total",
    "Total HTTP Requests"
)

PREDICTION_COUNT = Counter(
    "prediction_requests_total",
    "Total Prediction Requests"
)

ERROR_COUNT = Counter(
    "prediction_errors_total",
    "Prediction Errors"
)

REQUEST_LATENCY = Histogram(
    "prediction_latency_seconds",
    "Prediction Response Time"
)
app = FastAPI(

    title="HariMLOps Heart Disease Prediction API",

    description="Predict Heart Disease using Random Forest",

    version="1.0"

)

@app.middleware("http")
async def monitor_requests(request: Request, call_next):

    start = perf_counter()

    try:
        response = await call_next(request)

        REQUEST_COUNT.inc()

        elapsed = perf_counter() - start

        REQUEST_LATENCY.observe(elapsed)

        logger.info(
            f"{request.method} {request.url.path} "
            f"{response.status_code} "
            f"{elapsed:.4f} sec"
        )

        return response

    except Exception as e:

        ERROR_COUNT.inc()

        logger.exception(e)

        raise
# ---------------------------------------------------------
# Input Schema
# ---------------------------------------------------------

class PatientData(BaseModel):

    age: float

    sex: float

    cp: float

    trestbps: float

    chol: float

    fbs: float

    restecg: float

    thalach: float

    exang: float

    oldpeak: float

    slope: float

    ca: float

    thal: float


# ---------------------------------------------------------
# Home
# ---------------------------------------------------------

@app.get("/")

def home():

    return {

        "message":

        "HariMLOps Heart Disease Prediction API"

    }

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": True,
        "service": "Heart Disease Prediction API"
    }

@app.get("/metrics")

def metrics():

    return Response(

        generate_latest(),

        media_type=CONTENT_TYPE_LATEST

    )
# ---------------------------------------------------------
# Prediction Endpoint
# ---------------------------------------------------------

@app.post("/predict")
def predict(patient: PatientData):

    df = pd.DataFrame([patient.model_dump()])

    scaled = scaler.transform(df)

    scaled_df = pd.DataFrame(
        scaled,
        columns=df.columns
    )

    prediction = int(model.predict(scaled_df)[0])

    probability = float(
        model.predict_proba(scaled_df)[0][1]
    )

    label = (
        "Heart Disease"
        if prediction == 1
        else "No Heart Disease"
    )

    PREDICTION_COUNT.inc()

    logger.info(
        f"Prediction={label} "
        f"Probability={probability:.4f}"
    )

    return {
        "prediction": prediction,
        "label": label,
        "probability": round(probability, 4)
    }