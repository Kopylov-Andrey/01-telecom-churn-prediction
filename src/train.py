from __future__ import annotations

from pathlib import Path
from datetime import datetime, timezone
import json

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import joblib

# 1) Пути и версия
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
    # В этом проекте используем имена без подчеркиваний, чтобы совпадали с API
    mapping = {
        "customerID": "customerid",
        "gender": "gender",
        "SeniorCitizen": "seniorcitizen",
        "Partner": "partner",
        "Dependents": "dependents",
        "tenure": "tenure",
        "PhoneService": "phoneservice",
        "MultipleLines": "multiplelines",
        "InternetService": "internetservice",
        "OnlineSecurity": "onlinesecurity",
        "OnlineBackup": "onlinebackup",
        "DeviceProtection": "deviceprotection",
        "TechSupport": "techsupport",
        "StreamingTV": "streamingtv",
        "StreamingMovies": "streamingmovies",
        "Contract": "contract",
        "PaperlessBilling": "paperlessbilling",
        "PaymentMethod": "paymentmethod",
        "MonthlyCharges": "monthlycharges",
        "TotalCharges": "totalcharges",
        "Churn": "churn",
    }
    return df.rename(columns=mapping)


def prepare_data(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, list[str], list[str]]:
    df = rename_columns_to_snake_case(df).copy()

    # В данных встречаются пустые строки в totalcharges -> приводим к числовому
    df["totalcharges"] = pd.to_numeric(df["totalcharges"], errors="coerce")

    # Целевая переменная: Yes -> 1, No -> 0
    y = (df["churn"] == "Yes").astype(int)

    # Набор признаков
    numeric_features = ["tenure", "monthlycharges", "totalcharges", "seniorcitizen"]
    categorical_features = [
        "partner",
        "dependents",
        "internetservice",
        "contract",
        "paymentmethod",
        "gender",
        "paperlessbilling",
        "multiplelines",
        "onlinesecurity",
        "techsupport",
    ]

    # Проверка наличия всех колонок
    missing = [
        c for c in (numeric_features + categorical_features) if c not in df.columns
    ]
    if missing:
        raise KeyError(f"Missing expected columns: {missing}")

    X = df[numeric_features + categorical_features].copy()
    return X, y, numeric_features, categorical_features


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
    # liblinear хорошо работает с разреженными матрицами
    model = LogisticRegression(max_iter=1000, solver="liblinear")
    pipe = Pipeline(steps=[("preprocess", preprocess), ("model", model)])
    return pipe


def evaluate_model(model: Pipeline, X_val: pd.DataFrame, y_val: pd.Series) -> dict:
    proba = model.predict_proba(X_val)[:, 1]
    pred = (proba >= 0.5).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y_val, proba)),
        "pr_auc": float(average_precision_score(y_val, proba)),
        "accuracy": float(accuracy_score(y_val, pred)),
    }


def select_threshold_youden(y_true: pd.Series, proba: np.ndarray) -> float:
    fpr, tpr, thresholds = roc_curve(y_true, proba)
    youden_j = tpr - fpr
    best_idx = int(np.argmax(youden_j))
    return float(thresholds[best_idx])


def save_artifacts(
    model: Pipeline, features: list[str], metrics: dict, threshold: float
) -> None:
    joblib.dump(model, MODEL_FILE)
    meta = {
        "version": MODEL_VERSION,
        "trained_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "features": features,
        "model_file": MODEL_FILE.name,
        "metrics": {
            "roc_auc": float(metrics["roc_auc"]),
            "pr_auc": float(metrics["pr_auc"]),
            "accuracy": float(metrics["accuracy"]),
        },
        "threshold": float(threshold),
    }
    META_FILE.write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Saved model  -> {MODEL_FILE}")
    print(f"Saved meta   -> {META_FILE}")


if __name__ == "__main__":
    # Загрузка
    df = load_raw_data(DATA_PATH)

    # Подготовка
    X, y, numeric_features, categorical_features = prepare_data(df)
    feature_names = numeric_features + categorical_features

    # Разбиение
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Обучение
    model = build_pipeline(numeric_features, categorical_features)
    model.fit(X_train, y_train)

    # Метрики
    metrics = evaluate_model(model, X_val, y_val)

    # Порог по Youden J
    y_val_proba = model.predict_proba(X_val)[:, 1]
    best_threshold = select_threshold_youden(y_val, y_val_proba)

    # Сохранение
    save_artifacts(model, feature_names, metrics, best_threshold)
