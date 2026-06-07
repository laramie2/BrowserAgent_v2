#!/usr/bin/env bash
set -u

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
STATE_DIR="${STATE_DIR:-$SCRIPT_DIR/run}"
LOG_DIR="$STATE_DIR/logs"
LB_PORT="${LB_PORT:-22015}"
TEST_PATH="${TEST_PATH:-/wikipedia_en_all_maxi_2022-05/A/User:The_other_Kiwix_guy/Landing}"

echo "===== native wiki cluster pids ====="
if compgen -G "$STATE_DIR/*.pid" >/dev/null; then
    for pid_file in "$STATE_DIR"/*.pid; do
        name="$(basename "$pid_file" .pid)"
        pid="$(cat "$pid_file" 2>/dev/null || true)"
        if [[ -n "$pid" ]] && kill -0 "$pid" >/dev/null 2>&1; then
            echo "$name pid=$pid status=running"
        else
            echo "$name pid=${pid:-unknown} status=stopped"
        fi
    done
else
    echo "No pid files found in $STATE_DIR"
fi

echo
echo "===== backend health ====="
if [[ -f "$STATE_DIR/backends.txt" ]]; then
    IFS=, read -r -a backend_ports <"$STATE_DIR/backends.txt"
else
    backend_ports=(22115 22116 22117 22118 22119 22120 22121 22122)
fi

for port in "${backend_ports[@]}"; do
    echo -n "checking backend $port ... "
    curl -I -s --max-time 5 "http://localhost:${port}${TEST_PATH}" | head -n 1 || echo "FAILED"
done

echo
echo "===== load balancer health ====="
curl -I -s --max-time 5 "http://localhost:${LB_PORT}${TEST_PATH}" | head -n 1 || echo "FAILED"

echo
echo "===== recent logs ====="
if [[ -f "$LOG_DIR/wiki-lb.log" ]]; then
    echo "--- wiki-lb ---"
    tail -n 20 "$LOG_DIR/wiki-lb.log"
fi

for log_file in "$LOG_DIR"/wikipedia-*.log; do
    [[ -f "$log_file" ]] || continue
    echo
    echo "--- $(basename "$log_file" .log) ---"
    tail -n 5 "$log_file"
done
