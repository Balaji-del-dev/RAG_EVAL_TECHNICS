#!/usr/bin/env bash
# Stop the dev server started by scripts/start.sh (matches this project's runserver only).
set -euo pipefail
cd "$(dirname "$0")/.."
PORT="${PORT:-8011}"
PIDS=$(ps -eo pid,args | awk -v dir="$(pwd)" -v port="$PORT" '$0 ~ "manage.py runserver 127.0.0.1:" port && $0 !~ /awk/ {print $1}')
for pid in $PIDS; do
  if [ "$(readlink -f /proc/$pid/cwd)" = "$(pwd)" ]; then kill "$pid" && echo "Stopped $pid"; fi
done
