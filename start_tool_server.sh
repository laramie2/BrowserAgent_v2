#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${TOOL_SERVER_PYTHON:-python}"
TOOL_SERVER_HOST="${TOOL_SERVER_HOST:-127.0.0.1}"
TOOL_SERVER_PORT="${TOOL_SERVER_PORT:-5000}"

export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/verl-tool${PYTHONPATH:+:${PYTHONPATH}}"
export NO_PROXY="localhost,127.0.0.1,::1${NO_PROXY:+,${NO_PROXY}}"
export no_proxy="${NO_PROXY}"

export TEXT_BROWSER_RAY_NUM_CPUS="${TEXT_BROWSER_RAY_NUM_CPUS:-4}"
export TEXT_BROWSER_MAX_ACTIVE_ACTORS="${TEXT_BROWSER_MAX_ACTIVE_ACTORS:-16}"
export TEXT_BROWSER_IDLE_POOL_SIZE="${TEXT_BROWSER_IDLE_POOL_SIZE:-2}"
export TEXT_BROWSER_ACTION_TIMEOUT_SEC="${TEXT_BROWSER_ACTION_TIMEOUT_SEC:-120}"
export TEXT_BROWSER_ENV_RPC_TIMEOUT_SEC="${TEXT_BROWSER_ENV_RPC_TIMEOUT_SEC:-110}"
export VT_HEALTH_CHECK_TIMEOUT="${VT_HEALTH_CHECK_TIMEOUT:-180}"

cd "${PROJECT_ROOT}"
exec "${PYTHON_BIN}" -m verl_tool.servers.serve \
    --tool_type text_browser \
    --host "${TOOL_SERVER_HOST}" \
    --port "${TOOL_SERVER_PORT}" \
    --uvi_workers "${TOOL_SERVER_UVI_WORKERS:-1}" \
    --router_workers "${TOOL_SERVER_ROUTER_WORKERS:-1}" \
    --workers_per_tool "${TOOL_SERVER_WORKERS_PER_TOOL:-16}" \
    --max_concurrent_requests "${TOOL_SERVER_MAX_CONCURRENT_REQUESTS:-16}" \
    --thread_pool_size "${TOOL_SERVER_THREAD_POOL_SIZE:-64}" \
    --request_timeout "${TOOL_SERVER_REQUEST_TIMEOUT:-120}"
