#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

DEFAULT_ZIM_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)/webarena/webarena_zim"
BUNDLED_KIWIX_SERVE="$SCRIPT_DIR/tools/kiwix-tools_linux-x86_64-3.3.0/kiwix-serve"
ZIM_ROOT="${ZIM_ROOT:-$DEFAULT_ZIM_ROOT}"
ZIM_NAME="${ZIM_NAME:-wikipedia_en_all_maxi_2022-05.zim}"
ZIM_COPIES="${ZIM_COPIES:-4}"
WORKERS_PER_ZIM="${WORKERS_PER_ZIM:-2}"
if [[ -x "$BUNDLED_KIWIX_SERVE" ]]; then
    KIWIX_SERVE_BIN="${KIWIX_SERVE_BIN:-$BUNDLED_KIWIX_SERVE}"
else
    KIWIX_SERVE_BIN="${KIWIX_SERVE_BIN:-kiwix-serve}"
fi
PORT_START="${PORT_START:-22115}"
LB_HOST="${LB_HOST:-0.0.0.0}"
LB_PORT="${LB_PORT:-22015}"
STATE_DIR="${STATE_DIR:-$SCRIPT_DIR/run}"
LOG_DIR="$STATE_DIR/logs"
BACKEND_STATE_FILE="$STATE_DIR/backends.tsv"
WATCHDOG_INTERVAL="${WATCHDOG_INTERVAL:-15}"
WATCHDOG_HEALTH_TIMEOUT="${WATCHDOG_HEALTH_TIMEOUT:-12}"
WATCHDOG_MAX_FAILURES="${WATCHDOG_MAX_FAILURES:-3}"

zim_paths=()
if [[ -n "${ZIM_PATHS:-}" ]]; then
    IFS=, read -r -a zim_paths <<<"$ZIM_PATHS"
elif [[ -n "${ZIM_PATH:-}" ]]; then
    zim_paths=("$ZIM_PATH")
else
    for ((copy = 1; copy <= ZIM_COPIES; copy++)); do
        zim_paths+=("$ZIM_ROOT/$copy/$ZIM_NAME")
    done
fi

for zim_path in "${zim_paths[@]}"; do
    if [[ ! -f "$zim_path" ]]; then
        echo "ZIM file not found: $zim_path" >&2
        echo "Set ZIM_ROOT=/path/to/webarena_zim or ZIM_PATHS=/path/1.zim,/path/2.zim and retry." >&2
        exit 1
    fi
done

if [[ "$KIWIX_SERVE_BIN" == */* ]]; then
    if [[ ! -x "$KIWIX_SERVE_BIN" ]]; then
        echo "kiwix-serve binary is not executable: $KIWIX_SERVE_BIN" >&2
        exit 1
    fi
else
    if ! command -v "$KIWIX_SERVE_BIN" >/dev/null 2>&1; then
        cat >&2 <<EOF
Cannot find kiwix-serve.

Download the official Linux x86_64 archive into wiki_cluster/tools, or point KIWIX_SERVE_BIN to a binary:
  KIWIX_SERVE_BIN=/path/to/kiwix-serve ./start.sh
EOF
        exit 1
    fi
fi

mkdir -p "$LOG_DIR"

echo "[1/4] Stop existing native wiki cluster if present..."
"$SCRIPT_DIR/stop.sh" >/dev/null 2>&1 || true
: >"$BACKEND_STATE_FILE"

total_workers=$((${#zim_paths[@]} * WORKERS_PER_ZIM))
echo "[2/4] Start $total_workers kiwix-serve backends from ${#zim_paths[@]} ZIM copies..."
backend_ports=()
worker_index=0
for zim_index in "${!zim_paths[@]}"; do
    zim_path="${zim_paths[$zim_index]}"
    for ((copy_worker = 1; copy_worker <= WORKERS_PER_ZIM; copy_worker++)); do
        port=$((PORT_START + worker_index))
        name="wikipedia-$((worker_index + 1))"
        log_file="$LOG_DIR/${name}.log"
        setsid "$KIWIX_SERVE_BIN" --port="$port" "$zim_path" >"$log_file" 2>&1 &
        pid=$!
        echo "$pid" >"$STATE_DIR/${name}.pid"
        printf '%s\t%s\t%s\t%s\n' "$name" "$port" "$zim_path" "$log_file" >>"$BACKEND_STATE_FILE"
        backend_ports+=("$port")
        echo "  $name pid=$pid port=$port zim_copy=$((zim_index + 1)) log=$log_file"
        worker_index=$((worker_index + 1))
    done
done

backends="$(IFS=,; echo "${backend_ports[*]}")"
echo "$backends" >"$STATE_DIR/backends.txt"

echo "[3/4] Start Python load balancer on ${LB_HOST}:${LB_PORT}..."
setsid python3 "$SCRIPT_DIR/wiki_lb.py" \
    --listen-host "$LB_HOST" \
    --listen-port "$LB_PORT" \
    --backends "$backends" \
    >"$LOG_DIR/wiki-lb.log" 2>&1 &
echo "$!" >"$STATE_DIR/wiki-lb.pid"

echo "[4/4] Start watchdog and probe services..."
: >"$LOG_DIR/wiki-watchdog.log"
setsid env \
    STATE_DIR="$STATE_DIR" \
    KIWIX_SERVE_BIN="$KIWIX_SERVE_BIN" \
    BACKEND_STATE_FILE="$BACKEND_STATE_FILE" \
    WATCHDOG_INTERVAL="$WATCHDOG_INTERVAL" \
    WATCHDOG_HEALTH_TIMEOUT="$WATCHDOG_HEALTH_TIMEOUT" \
    WATCHDOG_MAX_FAILURES="$WATCHDOG_MAX_FAILURES" \
    bash "$SCRIPT_DIR/wiki_watchdog.sh" \
    >/dev/null 2>&1 &
echo "$!" >"$STATE_DIR/wiki-watchdog.pid"

sleep 2
"$SCRIPT_DIR/check.sh"

echo
echo "Wiki entry URL:"
echo "http://localhost:${LB_PORT}/wikipedia_en_all_maxi_2022-05/A/User:The_other_Kiwix_guy/Landing"
