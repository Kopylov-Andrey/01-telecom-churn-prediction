.PHONY: init lint test fmt

init:
    python -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt -r requirements-dev.txt && pre-commit install

lint:
    pre-commit run --all-files

fmt:
    black . && ruff --fix .

test:
    pytest -q
