from __future__ import annotations

from pathlib import Path
import json
from typing import Optional

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_FILE = PROJECT_ROOT / "models" / "churn.pkl"
META_FILE = PROJECT_ROOT / "models" / "metadata.json"

app = FastAPI(title="Churn Prediction API", version="0.2.0")

# Грузим модель и метаданные при старте
MODEL = None
FEATURES = None
if MODEL_FILE.exists() and META_FILE.exists():
    try:
        MODEL = joblib.load(MODEL_FILE)
        meta = json.loads(META_FILE.read_text())
        FEATURES = meta.get("features", [])
        print(f"Model loaded. Features: {FEATURES}")
    except Exception as e:
        print(f"Failed to load model: {e}")


class CustomerFeatures(BaseModel):
    # Поля делаем опциональными — если чего-то нет, imputers в пайплайне справятся.
    tenure: Optional[float] = None
    monthly_charges: Optional[float] = None
    total_charges: Optional[float] = None
    senior_citizen: Optional[int] = None

    partner: Optional[str] = None
    dependents: Optional[str] = None
    internet_service: Optional[str] = None
    contract: Optional[str] = None
    payment_method: Optional[str] = None
    gender: Optional[str] = None
    paperless_billing: Optional[str] = None
    multiple_lines: Optional[str] = None
    online_security: Optional[str] = None
    tech_support: Optional[str] = None


def to_dict(payload: CustomerFeatures) -> dict:
    # Совместимость pydantic v1/v2
    if hasattr(payload, "model_dump"):
        return payload.model_dump()
    return payload.dict()


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": MODEL is not None}


@app.post("/predict")
def predict(features: CustomerFeatures):
    if MODEL is None or FEATURES is None:
        raise HTTPException(
            status_code=500, detail="Model is not loaded. Train the model first."
        )

    data = to_dict(features)

    # Готовим ровно те фичи, на которых обучались
    row = {col: data.get(col, None) for col in FEATURES}
    X = pd.DataFrame([row])

    try:
        proba = float(MODEL.predict_proba(X)[:, 1][0])
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Inference failed: {e}")

    return {"churn_proba": round(proba, 6)}
