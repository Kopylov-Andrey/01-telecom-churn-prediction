from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Churn Prediction API", version="0.1.0")


class CustomerFeatures(BaseModel):
    tenure: float
    monthly_charges: float
    total_charges: float
    contract_type: str | None = None
    internet_service: str | None = None


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(features: CustomerFeatures):
    # TODO: загрузить препроцессор/модель из артефактов (MLflow/DVC)
    # Заглушка:
    return {"churn_proba": 0.42, "recommendation": "offer_discount"}
