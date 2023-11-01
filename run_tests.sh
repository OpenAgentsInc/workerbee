#!/bin/bash -e

CI=1 poetry run python -u -mcoverage run --source ai_worker -m pytest -v --run-onnx tests/
coverage html
coverage report --fail-under 55  --omit=quantize_main.py
