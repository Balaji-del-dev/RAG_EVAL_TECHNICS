#!/usr/bin/env bash
# Stop the API started by scripts/start.sh (only this project's uvicorn on this port).
set -euo pipefail
cd "$(dirname "$0")/.."
PORT="${PORT:-8013}"
for pid in $(ss -ltnpH "sport = :${PORT}" | grep -o 'pid=[0-9]*' | cut -d= -f2 | sort -u); do
  if tr '\0' ' ' < "/proc/$pid/cmdline" | grep -q "uvicorn app.main:app"; then kill "$pid" && echo "Stopped $pid"; fi
done
