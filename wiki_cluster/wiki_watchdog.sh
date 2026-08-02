#!/usr/bin/env bash
set -u

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
STATE_DIR="${STATE_DIR:-$SCRIPT_DIR/run}"
LOG_DIR="$STATE_DIR/logs"
BACKEND_STATE_FILE="${BACKEND_STATE_FILE:-$STATE_DIR/backends.tsv}"
WATCHDOG_LOG="$LOG_DIR/wiki-watchdog.log"
TEST_PATH="${TEST_PATH:-/wikipedia_en_all_maxi_2022-05/A/User:The_other_Kiwix_guy/Landing}"
WATCHDOG_INTERVAL="${WATCHDOG_INTERVAL:-15}"
WATCHDOG_HEALTH_TIMEOUT="${WATCHDOG_HEALTH_TIMEOUT:-12}"
WATCHDOG_MAX_FAILURES="${WATCHDOG_MAX_FAILURES:-3}"

if [[ -z "${KIWIX_SERVE_BIN:-}" ]]; then
    echo "KIWIX_SERVE_BIN is required for wiki watchdog" >&2
    exit 1
fi

mkdir -p "$LOG_DIR"

log() {
    printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" >>"$WATCHDOG_LOG"
}

pid_is_backend() {
    local pid="$1"
    local cmdline
    [[ -n "$pid" ]] || return 1
    kill -0 "$pid" >/dev/null 2>&1 || return 1
    cmdline="$(ps -p "$pid" -o args= 2>/dev/null || true)"
    [[ "$cmdline" == *"kiwix-serve"* ]]
}

stop_backend_pid() {
    local pid="$1"
    local name="$2"
    [[ -n "$pid" ]] || return 0
    if ! pid_is_backend "$pid"; then
        return 0
    fi

    log "$name stopping unhealthy pid=$pid"
    kill "$pid" >/dev/null 2>&1 || true
    sleep 2
    if pid_is_backend "$pid"; then
        log "$name force stopping pid=$pid"
        kill -9 "$pid" >/dev/null 2>&1 || true
    fi
}

start_backend() {
    local name="$1"
    local port="$2"
    local zim_path="$3"
    local log_file="$4"
    local pid

    setsid "$KIWIX_SERVE_BIN" --port="$port" "$zim_path" >>"$log_file" 2>&1 &
    pid=$!
    echo "$pid" >"$STATE_DIR/${name}.pid"
    log "$name restarted pid=$pid port=$port zim=$zim_path"
}

backend_healthy() {
    local port="$1"
    curl -I -s --max-time "$WATCHDOG_HEALTH_TIMEOUT" \
        "http://127.0.0.1:${port}${TEST_PATH}" | head -n 1 | grep -qE '^HTTP/[0-9.]+ [23][0-9][0-9]'
}

declare -A failures

log "watchdog started interval=${WATCHDOG_INTERVAL}s timeout=${WATCHDOG_HEALTH_TIMEOUT}s max_failures=${WATCHDOG_MAX_FAILURES}"

while true; do
    if [[ ! -f "$BACKEND_STATE_FILE" ]]; then
        log "backend state file missing: $BACKEND_STATE_FILE"
        sleep "$WATCHDOG_INTERVAL"
        continue
    fi

    while IFS=$'\t' read -r name port zim_path log_file; do
        [[ -n "${name:-}" ]] || continue
        [[ "$name" == \#* ]] && continue

        pid="$(cat "$STATE_DIR/${name}.pid" 2>/dev/null || true)"
        if ! pid_is_backend "$pid"; then
            log "$name pid=${pid:-missing} is not running"
            start_backend "$name" "$port" "$zim_path" "$log_file"
            failures["$name"]=0
            continue
        fi

        if backend_healthy "$port"; then
            failures["$name"]=0
            continue
        fi

        current_failures="${failures[$name]:-0}"
        current_failures=$((current_failures + 1))
        failures["$name"]="$current_failures"
        log "$name health check failed count=$current_failures pid=$pid port=$port"

        if (( current_failures >= WATCHDOG_MAX_FAILURES )); then
            stop_backend_pid "$pid" "$name"
            start_backend "$name" "$port" "$zim_path" "$log_file"
            failures["$name"]=0
        fi
    done <"$BACKEND_STATE_FILE"

    sleep "$WATCHDOG_INTERVAL"
done
