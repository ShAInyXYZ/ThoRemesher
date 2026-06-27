#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

# ensure deps
python3 -m pip install -q -r requirements.txt 2>/dev/null || true

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
echo ">> Curvature-Aware Remesher on http://${HOST}:${PORT}"
exec python3 -m uvicorn app:app --host "$HOST" --port "$PORT"
