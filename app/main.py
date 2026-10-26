
import json
import logging
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import numpy as np
import joblib
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field


logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,
)
logger = logging.getLogger("breast-cancer-api")

MODEL_PATH = Path(os.getenv("MODEL_PATH", "models/model.joblib"))
CARD_PATH = MODEL_PATH.parent / "model_card.json"

try:
    DECISION_THRESHOLD = float(os.getenv("DECISION_THRESHOLD", "0.5"))
except ValueError as exc:
    raise RuntimeError("DECISION_THRESHOLD must be a number.") from exc

if not 0 <= DECISION_THRESHOLD <= 1:
    raise RuntimeError("DECISION_THRESHOLD must be between 0 and 1.")


class PredictionRequest(BaseModel):
    features: list[float] = Field(min_length=30, max_length=30)


class PredictionResponse(BaseModel):
    prediction: Literal["malignant", "benign"]
    malignant_probability: float
    decision_threshold: float
    model_version: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not MODEL_PATH.is_file():
        logger.error(
            "Model artifact missing at %s. Run the training pipeline first.",
            MODEL_PATH,
        )
        raise RuntimeError(f"Missing model artifact: {MODEL_PATH}")

    # Load once at startup, not inside the prediction endpoint.
    app.state.model = joblib.load(MODEL_PATH)
    app.state.model_version = "unknown"

    if CARD_PATH.is_file():
        try:
            with open(CARD_PATH, "r", encoding="utf-8") as file:
                card = json.load(file)
            app.state.model_version = card.get("model_version", "unknown")
        except (OSError, json.JSONDecodeError):
            logger.exception("Could not read model card.")

    logger.info("Loaded model from %s", MODEL_PATH)
    logger.info("Decision threshold: %.3f", DECISION_THRESHOLD)

    yield
    logger.info("API shutting down.")


app = FastAPI(
    title="Breast Cancer Triage API",
    description="Educational coursework service, not a clinical diagnostic tool.",
    version="2.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def health(request: Request):
    model = getattr(request.app.state, "model", None)
    return {
        "status": "ok" if model is not None else "unavailable",
        "model_loaded": model is not None,
        "model_version": getattr(request.app.state, "model_version", "unknown"),
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: PredictionRequest, request: Request):
    model = getattr(request.app.state, "model", None)

    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")

    features = np.asarray(payload.features, dtype=float).reshape(1, -1)

    if not np.isfinite(features).all():
        raise HTTPException(
            status_code=422,
            detail="All feature values must be finite numbers.",
        )

    try:
        probabilities = model.predict_proba(features)[0]
        malignant_index = list(model.classes_).index(0)
        malignant_probability = float(probabilities[malignant_index])
    except Exception as exc:
        logger.exception("Prediction failed.")
        raise HTTPException(status_code=500, detail="Prediction failed.") from exc

    prediction = (
        "malignant"
        if malignant_probability >= DECISION_THRESHOLD
        else "benign"
    )

    logger.info(
        "Prediction completed; malignant_probability=%.4f",
        malignant_probability,
    )

    return PredictionResponse(
        prediction=prediction,
        malignant_probability=malignant_probability,
        decision_threshold=DECISION_THRESHOLD,
        model_version=request.app.state.model_version,
    )


@app.get("/stats")
def stats():
    return {
        "status": "placeholder",
        "message": "Statistics will be implemented in Milestone 4.",
    }