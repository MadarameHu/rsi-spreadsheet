#!/usr/bin/env bash
# Local read-only viewer: build the index, then start uvicorn on 127.0.0.1.
#
#   ./viewer/serve.sh [port]
#
# The process is detached from this shell so closing the terminal does not
# kill it. Stop with ./viewer/serve.sh stop.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"
PORT="${2:-${PORT:-8791}}"
PIDFILE="$HERE/server.pid"
LOG="$HERE/server.log"
BATCH="${VIEWER_BATCH:-$REPO/batch}"

if [[ "${1:-start}" == "stop" ]]; then
  if [[ -f "$PIDFILE" ]] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
    kill "$(cat "$PIDFILE")" && echo "stopped $(cat "$PIDFILE")"
  else
    echo "not running"
  fi
  rm -f "$PIDFILE"
  exit 0
fi

mkdir -p "$REPO/runs" "$BATCH"
python3 "$HERE/indexer/build_index.py" \
  --runs "$REPO/runs" --batch "$BATCH" --out "$HERE/data/index.json"

VIEWER_RUNS="$REPO/runs" \
VIEWER_BATCH="$BATCH" \
VIEWER_INDEX="$HERE/data/index.json" \
VIEWER_WEB="$HERE/web" \
python3 - "$PORT" "$PIDFILE" "$LOG" "$REPO" <<'PY'
import os, subprocess, sys
port, pidfile, log, repo = sys.argv[1:5]
with open(log, 'w') as fh:
    proc = subprocess.Popen(
        [sys.executable, '-m', 'uvicorn', 'viewer.server.main:app',
         '--host', '127.0.0.1', '--port', port],
        cwd=repo, stdout=fh, stderr=subprocess.STDOUT,
        start_new_session=True, env=dict(os.environ),
    )
open(pidfile, 'w').write(str(proc.pid))
print(f'viewer pid {proc.pid} on http://127.0.0.1:{port}')
PY
