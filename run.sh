#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "📦  Installing dependencies with uv..."
uv sync

# Ensure dataset exists
if [ ! -f "backend/data/email_reply_dataset.jsonl" ]; then
    echo "📝  Building dataset..."
    uv run backend/scripts/build_dataset.py
fi

echo "🚀  Starting server at http://localhost:8000"
uv run uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000
