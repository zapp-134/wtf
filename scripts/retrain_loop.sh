#!/usr/bin/env bash
set -euo pipefail

THRESHOLD=${THRESHOLD:-5}
INTERVAL=${INTERVAL:-300}

echo "[trainer] starting feedback watcher (threshold=$THRESHOLD, interval=$INTERVAL)"
while true; do
  if python ml/retrain.py --threshold "$THRESHOLD"; then
    echo "[trainer] retrain script exited successfully"
  else
    echo "[trainer] retrain script exited with error" >&2
  fi
  sleep "$INTERVAL"
done
