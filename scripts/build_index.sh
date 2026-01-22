#!/usr/bin/env bash
set -e

cd app/backend
python -c "from app.backend.src.core.config import PRIVATE_DATA_DIR; from app.backend.src.rag.index import build_index; build_index(PRIVATE_DATA_DIR); print('✅ Index built')"
