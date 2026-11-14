#!/usr/bin/env bash
set -euo pipefail

uvicorn spectrumbloom.main:app --host 127.0.0.1 --port 8000 &
API_PID=$!

# Wait briefly for FastAPI health endpoint
for _ in $(seq 1 30); do
  if python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')" 2>/dev/null; then
    break
  fi
  sleep 0.5
done

trap 'kill "$API_PID" 2>/dev/null || true' EXIT

exec streamlit run src/spectrumbloom/ui/app.py \
  --server.address=0.0.0.0 \
  --server.port=7860 \
  --server.headless=true
