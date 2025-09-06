.PHONY: init lint test fmt run

init:
	python -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt -r requirements-dev.txt && pre-commit install

lint:
	pre-commit run --all-files

fmt:
	black .
	ruff check --fix .
	ruff format .

test:
	pytest -q

run:
	uvicorn src.api:app --reload
