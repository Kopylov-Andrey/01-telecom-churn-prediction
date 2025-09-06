from __future__ import annotations

import os
import json
from typing import Any, Dict, List

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Пути к артефактам (можно переопределить через переменные окружения)
DEFAULT_MODEL_PATH = "models/churn.pkl"
DEFAULT_META_PATH = "models/metadata.json"

MODEL_PATH = os.getenv("MODEL_PATH", DEFAULT_MODEL_PATH)
METADATA_PATH = os.getenv("METADATA_PATH", DEFAULT_META_PATH)

app = FastAPI(title="Churn API", version="1.0")


class PredictRequest(BaseModel):
    features: List[Dict[str, Any]]


class PredictItem(BaseModel):
    churn_proba: float
    churn_label: int


class PredictResponse(BaseModel):
    predictions: List[PredictItem]


# Загрузка модели и метаданных при старте
pipeline = None
metadata = {}
FEATURES: List[str] = []
THRESHOLD: float = 0.5
MODEL_VERSION: str | None = None


def _load_artifacts():
    global pipeline, metadata, FEATURES, THRESHOLD, MODEL_VERSION
    try:
        pipeline = joblib.load(MODEL_PATH)
    except Exception as e:
        pipeline = None
        print(f"[WARN] Failed to load model from {MODEL_PATH}: {e}")

    try:
        with open(METADATA_PATH, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        FEATURES = list(metadata.get("features", []))
        # приоритет: metadata.threshold -> env PRED_THRESHOLD -> 0.5
        THRESHOLD = float(metadata.get("threshold", os.getenv("PRED_THRESHOLD", "0.5")))
        MODEL_VERSION = metadata.get("version")
    except Exception as e:
        metadata = {}
        FEATURES = []
        THRESHOLD = float(os.getenv("PRED_THRESHOLD", "0.5"))
        MODEL_VERSION = None
        print(f"[WARN] Failed to load metadata from {METADATA_PATH}: {e}")


_load_artifacts()


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": pipeline is not None,
        "features_known": bool(FEATURES),
        "threshold": THRESHOLD,
        "model_version": MODEL_VERSION,
    }


@app.get("/version")
def version():
    if not metadata:
        raise HTTPException(status_code=500, detail="Metadata not available")
    return metadata


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if pipeline is None:
        raise HTTPException(status_code=500, detail="Model is not loaded")
    if not FEATURES:
        raise HTTPException(
            status_code=500, detail="Features are unknown (metadata missing)"
        )
    if not req.features:
        raise HTTPException(status_code=422, detail="Empty features list")

    # Проверяем наличие всех необходимых ключей в каждом объекте
    for i, row in enumerate(req.features):
        missing = set(FEATURES) - set(row.keys())
        if missing:
            raise HTTPException(
                status_code=422,
                detail=f"Sample #{i} missing keys: {sorted(missing)}. Expected keys: {FEATURES}",
            )

    # Создаем DataFrame в правильном порядке колонок
    df = pd.DataFrame(req.features)
    df = df[FEATURES]  # порядок признаков из metadata

    # Предсказания
    try:
        proba = pipeline.predict_proba(df)[:, 1]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

    labels = (proba >= THRESHOLD).astype(int)

    preds = [
        {"churn_proba": float(p), "churn_label": int(lbl)}
        for p, lbl in zip(proba, labels)
    ]
    return {"predictions": preds}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api:app", host="0.0.0.0", port=8001, reload=True)
