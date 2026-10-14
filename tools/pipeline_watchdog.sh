#!/bin/bash
# Restart a tools/pipeline.py supervisor if it died before finishing. Meant for cron, e.g.
#   */10 * * * * /path/to/repo/tools/pipeline_watchdog.sh /path/to/pipeline.json
# The config's state_dir holds the supervisor's PID and logs; once pipeline.log says
# "pipeline finished" this does nothing, so the cron line can stay until you clean it up.
export PATH=/usr/local/bin:/usr/bin:/bin
CFG="$(realpath "$1")"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
S="$("$REPO/.venv/bin/python" -I -c 'import json,sys; print(json.load(open(sys.argv[1]))["state_dir"])' "$CFG")"
mkdir -p "$S"
grep -q "pipeline finished" "$S/pipeline.log" 2>/dev/null && exit 0
pid=$(cat "$S/supervisor.pid" 2>/dev/null)
[ -n "$pid" ] && kill -0 "$pid" 2>/dev/null && exit 0
echo "$(date '+%F %T') cron watchdog: supervisor not running, (re)starting" >> "$S/pipeline.log"
cd "$REPO" && setsid nohup .venv/bin/python tools/pipeline.py "$CFG" >> "$S/supervisor.out" 2>&1 < /dev/null &
