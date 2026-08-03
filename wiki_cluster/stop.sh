#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
STATE_DIR="${STATE_DIR:-$SCRIPT_DIR/run}"

if [[ ! -d "$STATE_DIR" ]]; then
    echo "No native wiki cluster state found: $STATE_DIR"
    exit 0
fi

shopt -s nullglob
pid_files=("$STATE_DIR"/*.pid)

if (( ${#pid_files[@]} == 0 )); then
    echo "No native wiki cluster pid files found."
    exit 0
fi

watchdog_pid_file="$STATE_DIR/wiki-watchdog.pid"
if [[ -f "$watchdog_pid_file" ]]; then
    watchdog_pid="$(cat "$watchdog_pid_file" 2>/dev/null || true)"
    if [[ -n "$watchdog_pid" ]] && kill -0 "$watchdog_pid" >/dev/null 2>&1; then
        cmdline="$(ps -p "$watchdog_pid" -o args= 2>/dev/null || true)"
        if [[ "$cmdline" == *"wiki_watchdog.sh"* ]]; then
            echo "Stopping wiki-watchdog pid=$watchdog_pid"
            kill "$watchdog_pid" >/dev/null 2>&1 || true
        fi
    fi
    sleep 1
fi

for pid_file in "${pid_files[@]}"; do
    name="$(basename "$pid_file" .pid)"
    pid="$(cat "$pid_file" 2>/dev/null || true)"
    if [[ -z "$pid" ]]; then
        rm -f "$pid_file"
        continue
    fi

    if kill -0 "$pid" >/dev/null 2>&1; then
        cmdline="$(ps -p "$pid" -o args= 2>/dev/null || true)"
        if [[ "$cmdline" == *"kiwix-serve"* || "$cmdline" == *"load_balancer.py"* || "$cmdline" == *"wiki_watchdog.sh"* ]]; then
            echo "Stopping $name pid=$pid"
            kill "$pid" >/dev/null 2>&1 || true
        else
            echo "Skip $name pid=$pid because it does not look like this wiki cluster: $cmdline"
        fi
    else
        echo "$name pid=$pid is not running"
    fi
done

sleep 1

for pid_file in "${pid_files[@]}"; do
    pid="$(cat "$pid_file" 2>/dev/null || true)"
    if [[ -n "$pid" ]] && kill -0 "$pid" >/dev/null 2>&1; then
        cmdline="$(ps -p "$pid" -o args= 2>/dev/null || true)"
        if [[ "$cmdline" == *"kiwix-serve"* || "$cmdline" == *"load_balancer.py"* || "$cmdline" == *"wiki_watchdog.sh"* ]]; then
            echo "Force stopping pid=$pid"
            kill -9 "$pid" >/dev/null 2>&1 || true
        fi
    fi
    rm -f "$pid_file"
done
