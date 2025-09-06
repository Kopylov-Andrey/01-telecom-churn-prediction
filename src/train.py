# src/train.py
from __future__ import annotations

from pathlib import Path
from datetime import datetime
import json

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import joblib

# 1) Настройки путей и версия модели
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "telco.csv"
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
MODEL_FILE = MODELS_DIR / "churn.pkl"
META_FILE = MODELS_DIR / "metadata.json"
MODEL_VERSION = "0.1.0"


def load_raw_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}. Run scripts/download_data.py first."
        )
    df = pd.read_csv(path)
    return df


def rename_columns_to_snake_case(df: pd.DataFrame) -> pd.DataFrame:
    # Переименуем колонки в удобный snake_case, чтобы API и обучение совпадали
    mapping = {
        "customerID": "customer_id",
        "gender": "gender",
        "SeniorCitizen": "senior_citizen",
        "Partner": "partner",
        "Dependents": "dependents",
        "tenure": "tenure",
        "PhoneService": "phone_service",
        "MultipleLines": "multiple_lines",
        "InternetService": "internet_service",
        "OnlineSecurity": "online_security",
        "OnlineBackup": "online_backup",
        "DeviceProtection": "device_protection",
        "TechSupport": "tech_support",
        "StreamingTV": "streaming_tv",
        "StreamingMovies": "streaming_movies",
        "Contract": "contract",
        "PaperlessBilling": "paperless_billing",
        "PaymentMethod": "payment_method",
        "MonthlyCharges": "monthly_charges",
        "TotalCharges": "total_charges",
        "Churn": "churn",
    }
    return df.rename(columns=mapping)


def prepare_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    df = rename_columns_to_snake_case(df).copy()

    # Приведем total_charges к числу (в датасете бывают пустые строки)
    df["total_charges"] = pd.to_numeric(df["total_charges"], errors="coerce")

    # Целевая переменная: 1 если "Yes", иначе 0
    y = (df["churn"] == "Yes").astype(int)

    # Фичи. Выбираем компактный, но информативный набор.
    numeric_features = ["tenure", "monthly_charges", "total_charges", "senior_citizen"]
    categorical_features = [
        "partner",
        "dependents",
        "internet_service",
        "contract",
        "payment_method",
        "gender",
        "paperless_billing",
        "multiple_lines",
        "online_security",
        "tech_support",
    ]
    # Могут быть пропуски/отсутствующие колонки — убедимся, что они существуют
    missing = [
        c for c in (numeric_features + categorical_features) if c not in df.columns
    ]
    if missing:
        raise KeyError(f"Missing expected columns: {missing}")

    X = df[numeric_features + categorical_features].copy()
    return X, y


def build_pipeline(
    numeric_features: list[str], categorical_features: list[str]
) -> Pipeline:
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocess = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    model = LogisticRegression(max_iter=1000)
    pipe = Pipeline(steps=[("preprocess", preprocess), ("model", model)])
    return pipe


def evaluate(model: Pipeline, X_val: pd.DataFrame, y_val: pd.Series) -> dict:
    proba = model.predict_proba(X_val)[:, 1]
    pred = (proba >= 0.5).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y_val, proba)),
        "pr_auc": float(average_precision_score(y_val, proba)),
        "accuracy": float(accuracy_score(y_val, pred)),
    }


def save_artifacts(model: Pipeline, features: list[str]) -> None:
    joblib.dump(model, MODEL_FILE)
    meta = {
        "version": MODEL_VERSION,
        "trained_at": datetime.utcnow().isoformat() + "Z",
        "features": features,
        "model_file": MODEL_FILE.name,
    }
    META_FILE.write_text(json.dumps(meta, indent=2))
    print(f"Saved model -> {MODEL_FILE}")
    print(f"Saved metadata -> {META_FILE}")


def main():
    print(f"Loading data from {DATA_PATH} ...")
    df = load_raw_data(DATA_PATH)
    X, y = prepare_data(df)

    numeric_features = ["tenure", "monthly_charges", "total_charges", "senior_citizen"]
    categorical_features = [
        "partner",
        "dependents",
        "internet_service",
        "contract",
        "payment_method",
        "gender",
        "paperless_billing",
        "multiple_lines",
        "online_security",
        "tech_support",
    ]
    features = numeric_features + categorical_features

    print("Split train/val ...")
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("Build and fit pipeline ...")
    pipe = build_pipeline(numeric_features, categorical_features)
    pipe.fit(X_train, y_train)

    print("Evaluate ...")
    metrics = evaluate(pipe, X_val, y_val)
    print("Metrics:", metrics)

    print("Save artifacts ...")
    save_artifacts(pipe, features)


if __name__ == "__main__":
    main()
