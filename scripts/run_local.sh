#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/../app/backend"
export PYTHONPATH=.
uvicorn src.main:app --reload --host 0.0.0.0 --port "${PORT:-8000}"
