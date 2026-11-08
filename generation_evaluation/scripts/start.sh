#!/usr/bin/env bash
# Start the Generation Evaluation API in the background on 127.0.0.1:8013 (logs in logs/api.log).
set -euo pipefail
cd "$(dirname "$0")/.."
PORT="${PORT:-8013}"
if ss -ltn | grep -q "127.0.0.1:${PORT} "; then
  echo "Something is already listening on ${PORT}."; exit 1
fi
mkdir -p logs
# On a small machine, let this process be the first one the OOM killer stops, not the user's apps.
echo 1000 > /proc/self/oom_score_adj 2>/dev/null || true
nohup uv run --no-sync uvicorn app.main:app --app-dir services/fastapi_service \
  --host 127.0.0.1 --port "${PORT}" > logs/api.log 2>&1 &
for _ in $(seq 1 30); do
  sleep 1
  if curl -sf "http://127.0.0.1:${PORT}/metrics" > /dev/null; then
    echo "Generation Evaluation API running at http://127.0.0.1:${PORT}/docs"; exit 0
  fi
done
echo "The API did not start; see logs/api.log"; exit 1
