from fastapi.testclient import TestClient
from src.api import app

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert "status" in r.json()


def test_predict_range():
    payload = {
        "tenure": 10,
        "monthly_charges": 45.3,
        "total_charges": 320.0,
        "contract": "month-to-month",
        "internet_service": "Fiber optic",
    }
    r = client.post("/predict", json=payload)
    assert r.status_code == 200
    proba = r.json()["churn_proba"]
    assert 0.0 <= proba <= 1.0
