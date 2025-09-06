#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source venv/bin/activate
pre-commit run --all-files
pytest -q
uvicorn src.api:app --host 127.0.0.1 --port 8001 --reload
