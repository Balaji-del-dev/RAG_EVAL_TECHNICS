#!/usr/bin/env bash
# Start the dev server in the background on 127.0.0.1:8011 (logs in logs/runserver.log).
set -euo pipefail
cd "$(dirname "$0")/.."
PORT="${PORT:-8011}"
if ss -ltn | grep -q "127.0.0.1:${PORT} "; then
  echo "Something is already listening on ${PORT}."; exit 1
fi
mkdir -p logs
venv/bin/python manage.py migrate --noinput >/dev/null
nohup venv/bin/python manage.py runserver "127.0.0.1:${PORT}" > logs/runserver.log 2>&1 &
sleep 2
echo "Retrieval Evaluation running at http://127.0.0.1:${PORT}/"
