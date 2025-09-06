# Название проекта

Бизнес-задача: кратко опишите цель и измеримый бизнес-результат (например, экономия ₽/мес).

Данные: источник, ключевые поля, ограничения, разрешения.

Метрики:
- ML: ROC-AUC/PR-AUC/MAE/Recall@K и т.д.
- Бизнес: экономический эффект/риск (cost@threshold, uplift).

Технологии: Python, scikit-learn, XGBoost/LightGBM, MLflow, DVC, Hydra, Great Expectations, Evidently, FastAPI, Docker.

Как воспроизвести:
1) Установка
   - python -m venv .venv && source .venv/bin/activate
   - pip install -r requirements.txt -r requirements-dev.txt
   - pre-commit install
2) (Если используется DVC) dvc pull
3) Тренировка/оценка: см. scripts/ и configs/
4) Сервис: uvicorn src.api:app --reload

Результаты:
- Ключевые метрики и сравнение бейзлайна vs прод-модели
- Скриншоты/отчёты (напр., Evidently)
