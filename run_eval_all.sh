#!/usr/bin/env bash
set -Eeuo pipefail

export RAY_TMPDIR="${RAY_TMPDIR_OVERRIDE:-/home/nvidia/yutao/lzt/tmp}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_PATH="${ROOT_DIR}/$(basename "${BASH_SOURCE[0]}")"
LOG_DIR="${ROOT_DIR}/logs"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
RUN_LOG="${LOG_DIR}/eval_runner_${RUN_ID}.log"

DEFAULT_MODEL_PATH="${ROOT_DIR}/sft/output/Qwen2.5-VL-7B-Instruct-task-opsrc-hotpot5459-nq6318-sft-5e-5lr-freeze_false-2epoch-merged"
DEFAULT_OUTPUT_DIR="${ROOT_DIR}/gen_seq/results/Qwen2.5-VL-7B-Instruct-task-opsrc-hotpot5459-nq6318-sft-5e-5lr-freeze_false-2epoch-merged"
DEFAULT_PROMPT_PATH="${ROOT_DIR}/prompt/system_prompt_with_history_info.txt"

# 单跳数据集
DEFAULT_NQ_DATA_PATH="${ROOT_DIR}/benchmark/nq/test-00000-of-00001.parquet"
DEFAULT_TRIVIAQA_DATA_PATH="${ROOT_DIR}/benchmark/triviaqa/test-00000-of-00001.parquet"
DEFAULT_POPQA_DATA_PATH="${ROOT_DIR}/benchmark/popqa/test-00000-of-00001.parquet"

# 多跳数据集
DEFAULT_HOTPOT_DATA_PATH="${ROOT_DIR}/benchmark/hotpot/validation-00000-of-00001.parquet"
DEFAULT_2WIKI_DATA_PATH="${ROOT_DIR}/benchmark/2wiki/validation-00000-of-00001.parquet"
DEFAULT_MUSIQUE_DATA_PATH="${ROOT_DIR}/benchmark/musique/validation-00000-of-00001.parquet"
DEFAULT_BAMBOOGLE_DATA_PATH="${ROOT_DIR}/benchmark/bamboogle/test-00000-of-00001.parquet"


VLLM_MODEL_PATH="${VLLM_MODEL_PATH:-${DEFAULT_MODEL_PATH}}"
VLLM_PYTHON="${VLLM_PYTHON:-/home/nvidia/anaconda3/envs/browseragent-v2/bin/python}"
VLLM_CUDA_DEVICES="${VLLM_CUDA_DEVICES:-}"
VLLM_HOST="${VLLM_HOST:-0.0.0.0}"
VLLM_PORT="${VLLM_PORT:-8008}"
VLLM_SERVED_MODEL_NAME="${VLLM_SERVED_MODEL_NAME:-custom-llm-1}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-16384}"
VLLM_TENSOR_PARALLEL_SIZE="${VLLM_TENSOR_PARALLEL_SIZE:-2}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-}"
VLLM_READY_TIMEOUT="${VLLM_READY_TIMEOUT:-900}"
VLLM_HEALTH_URL="${VLLM_HEALTH_URL:-}"
KILL_EXISTING_VLLM="${KILL_EXISTING_VLLM:-1}"
CLEAN_TRITON_CACHE="${CLEAN_TRITON_CACHE:-1}"
START_VLLM="${START_VLLM:-1}"

TOOL_SERVER_PYTHON="${TOOL_SERVER_PYTHON:-python}"
TOOL_SERVER_PORT="${TOOL_SERVER_PORT:-5000}"
TOOL_SERVER_READY_TIMEOUT="${TOOL_SERVER_READY_TIMEOUT:-300}"
TOOL_SERVER_HEALTH_URL="${TOOL_SERVER_HEALTH_URL:-http://127.0.0.1:${TOOL_SERVER_PORT}/health}"
TOOL_SERVER_WORKERS_PER_TOOL="${TOOL_SERVER_WORKERS_PER_TOOL:-32}"
TOOL_SERVER_MAX_CONCURRENT_REQUESTS="${TOOL_SERVER_MAX_CONCURRENT_REQUESTS:-32}"
TOOL_SERVER_THREAD_POOL_SIZE="${TOOL_SERVER_THREAD_POOL_SIZE:-64}"
TOOL_SERVER_REQUEST_TIMEOUT="${TOOL_SERVER_REQUEST_TIMEOUT:-120}"
TOOL_SERVER_UVI_WORKERS="${TOOL_SERVER_UVI_WORKERS:-1}"
TOOL_SERVER_ROUTER_WORKERS="${TOOL_SERVER_ROUTER_WORKERS:-1}"
TEXT_BROWSER_RAY_NUM_CPUS="${TEXT_BROWSER_RAY_NUM_CPUS:-4}"
TEXT_BROWSER_ACTION_TIMEOUT_SEC="${TEXT_BROWSER_ACTION_TIMEOUT_SEC:-120}"
TEXT_BROWSER_ENV_RPC_TIMEOUT_SEC="${TEXT_BROWSER_ENV_RPC_TIMEOUT_SEC:-110}"
VT_HEALTH_CHECK_TIMEOUT="${VT_HEALTH_CHECK_TIMEOUT:-180}"
START_TOOL_SERVER="${START_TOOL_SERVER:-1}"

PIPELINE_PYTHON="${PIPELINE_PYTHON:-python}"
OUTPUT_DIR="${OUTPUT_DIR:-${DEFAULT_OUTPUT_DIR}}"
PROMPT_PATH="${PROMPT_PATH:-${DEFAULT_PROMPT_PATH}}"
NQ_DATA_PATH="${NQ_DATA_PATH:-${DEFAULT_NQ_DATA_PATH}}"
TRIVIAQA_DATA_PATH="${TRIVIAQA_DATA_PATH:-${DEFAULT_TRIVIAQA_DATA_PATH}}"
POPQA_DATA_PATH="${POPQA_DATA_PATH:-${DEFAULT_POPQA_DATA_PATH}}"
HOTPOT_DATA_PATH="${HOTPOT_DATA_PATH:-${DEFAULT_HOTPOT_DATA_PATH}}"
TWOWIKI_DATA_PATH="${TWOWIKI_DATA_PATH:-${DEFAULT_2WIKI_DATA_PATH}}"
MUSIQUE_DATA_PATH="${MUSIQUE_DATA_PATH:-${DEFAULT_MUSIQUE_DATA_PATH}}"
BAMBOOGLE_DATA_PATH="${BAMBOOGLE_DATA_PATH:-${DEFAULT_BAMBOOGLE_DATA_PATH}}"
MAX_SAMPLES="${MAX_SAMPLES:-1000}"
SAMPLE_SEED="${SAMPLE_SEED:-42}"
NUM_TRIALS="${NUM_TRIALS:-1}"
NQ_NUM_TRIALS="${NQ_NUM_TRIALS:-1}"
TRIVIAQA_NUM_TRIALS="${TRIVIAQA_NUM_TRIALS:-1}"
POPQA_NUM_TRIALS="${POPQA_NUM_TRIALS:-1}"
HOTPOT_NUM_TRIALS="${HOTPOT_NUM_TRIALS:-1}"
TWOWIKI_NUM_TRIALS="${TWOWIKI_NUM_TRIALS:-1}"
MUSIQUE_NUM_TRIALS="${MUSIQUE_NUM_TRIALS:-1}"
BAMBOOGLE_NUM_TRIALS="${BAMBOOGLE_NUM_TRIALS:-1}"
NUM_WORKERS="${NUM_WORKERS:-16}"
NQ_NUM_WORKERS="${NQ_NUM_WORKERS:-16}"
TRIVIAQA_NUM_WORKERS="${TRIVIAQA_NUM_WORKERS:-16}"
POPQA_NUM_WORKERS="${POPQA_NUM_WORKERS:-16}"
HOTPOT_NUM_WORKERS="${HOTPOT_NUM_WORKERS:-16}"
TWOWIKI_NUM_WORKERS="${TWOWIKI_NUM_WORKERS:-16}"
MUSIQUE_NUM_WORKERS="${MUSIQUE_NUM_WORKERS:-16}"
BAMBOOGLE_NUM_WORKERS="${BAMBOOGLE_NUM_WORKERS:-16}"
VLLM_BASE_URL="${VLLM_BASE_URL:-}"
USE_VLM="${USE_VLM:-1}"
KEEP_SERVICES="${KEEP_SERVICES:-0}"
DRY_RUN="${DRY_RUN:-0}"

VLLM_PID=""
TOOL_SERVER_PID=""
BENCHMARKS=()

usage() {
    cat <<'EOF'
Usage:
  ./run_eval_all.sh [nq|popqa|hotpot|2wiki|musique|bamboogle|triviaqa|all ...] [options]
  ./run_eval_all.sh --benchmarks hotpot,nq,popqa --model-path /path/to/model --num-workers 32

Common options:
  -b, --benchmarks LIST           Benchmarks: nq,popqa,hotpot,2wiki,musique,bamboogle,triviaqa,all.
      --output-dir DIR            Directory for jsonl results and observation images.
      --prompt-path PATH          System prompt path.
      --max-samples N             Max samples per benchmark.
      --sample-seed SEED          Random sample seed. Empty or 0 keeps sequential order.
      --num-trials N              Override trial count for all benchmarks.
      --nq-num-trials N           Trial count for nq. Default: 4.
      --popqa-num-trials N        Trial count for popqa. Default: 4.
      --hotpot-num-trials N       Trial count for hotpot. Default: 4.
      --2wiki-num-trials N        Trial count for 2wiki. Default: 4.
      --musique-num-trials N      Trial count for musique. Default: 4.
      --bamboogle-num-trials N    Trial count for bamboogle. Default: 4.
      --triviaqa-num-trials N     Trial count for triviaqa. Default: 1.
      --num-workers N             Override pipeline parallelism for all benchmarks.
      --nq-num-workers N          Pipeline workers for nq. Default: 16.
      --popqa-num-workers N       Pipeline workers for popqa. Default: 16.
      --hotpot-num-workers N      Pipeline workers for hotpot. Default: 16.
      --2wiki-num-workers N       Pipeline workers for 2wiki. Default: 16.
      --musique-num-workers N     Pipeline workers for musique. Default: 16.
      --bamboogle-num-workers N   Pipeline workers for bamboogle. Default: 16.
      --triviaqa-num-workers N    Pipeline workers for triviaqa. Default: 16.
      --nq-data-path PATH         NQ parquet path.
      --triviaqa-data-path PATH    TriviaQA parquet path.
      --popqa-data-path PATH      PopQA parquet path.
      --hotpot-data-path PATH     HotpotQA parquet path.
      --2wiki-data-path PATH      2Wiki parquet path.
      --musique-data-path PATH    MuSiQue parquet path.
      --bamboogle-data-path PATH  Bamboogle parquet path.
      --base-url URL              OpenAI-compatible vLLM API base URL.
      --pipeline-python PATH      Python executable for gen_seq.pipeline.
      --no-use-vlm                Do not pass --use_vlm to gen_seq.pipeline.

vLLM options:
      --model-path DIR            Model directory served by vLLM.
      --vllm-python PATH          Python executable for vLLM.
      --vllm-cuda-devices LIST    CUDA_VISIBLE_DEVICES. Default: 0,1,2,3.
      --vllm-port PORT            vLLM API port. Default: 8008.
      --served-model-name NAME    Served model name. Default: custom-llm-1.
      --vllm-max-model-len N      vLLM --max-model-len. Default: 16384.
      --vllm-tensor-parallel-size N
      --vllm-gpu-memory-utilization VALUE
      --vllm-ready-timeout SEC
      --vllm-health-url URL
      --skip-vllm                 Use an already running vLLM service.
      --no-kill-existing-vllm     Do not pkill an existing vLLM process on the selected port.
      --no-clean-triton-cache     Do not remove ~/.triton/cache before starting vLLM.

Tool-server options:
      --skip-tool-server          Use an already running tool-server on port 5000.
      --tool-ready-timeout SEC
      --tool-server-python PATH

Other options:
      --keep-services             Leave vLLM/tool-server running after evaluation.
      --dry-run                   Print commands without starting services or running eval.
  -h, --help                      Show this help.

Environment variables with the same uppercase names can also be used, for example:
  VLLM_MODEL_PATH=/path/to/model NUM_WORKERS=32 ./run_eval_all.sh all
EOF
}

log() {
    local message="$*"
    mkdir -p "${LOG_DIR}"
    printf '[%(%Y-%m-%d %H:%M:%S)T] %s\n' -1 "${message}" | tee -a "${RUN_LOG}"
}

die() {
    log "ERROR: $*"
    exit 1
}

quote_cmd() {
    printf '%q ' "$@"
}

add_benchmark() {
    local candidate="$1"
    local existing

    for existing in "${BENCHMARKS[@]}"; do
        [[ "${existing}" == "${candidate}" ]] && return 0
    done
    BENCHMARKS+=("${candidate}")
}

normalize_benchmarks() {
    local raw=("$@")
    local item

    if [[ ${#raw[@]} -eq 0 ]]; then
        raw=("all")
    fi

    BENCHMARKS=()
    for item in "${raw[@]}"; do
        item="${item//,/ }"
        # shellcheck disable=SC2206
        local parts=(${item})
        local part
        for part in "${parts[@]}"; do
            case "${part}" in
                all)
                    add_benchmark "nq"
                    add_benchmark "triviaqa"
                    add_benchmark "popqa"
                    add_benchmark "hotpot"
                    add_benchmark "2wiki"
                    add_benchmark "musique"
                    add_benchmark "bamboogle"
                    ;;
                nq|triviaqa|popqa|hotpot|2wiki|musique|bamboogle)
                    add_benchmark "${part}"
                    ;;
                "")
                    ;;
                *)
                    die "Unknown benchmark '${part}'. Supported: nq, triviaqa, popqa, hotpot, 2wiki, musique, bamboogle, all."
                    ;;
            esac
        done
    done

    [[ ${#BENCHMARKS[@]} -gt 0 ]] || die "No benchmark selected."
}

parse_args() {
    local selected=()

    while [[ $# -gt 0 ]]; do
        case "$1" in
            -h|--help)
                usage
                exit 0
                ;;
            -b|--benchmarks|--benchmark)
                [[ $# -ge 2 ]] || die "$1 requires a value."
                selected+=("$2")
                shift 2
                ;;
            --output-dir)
                [[ $# -ge 2 ]] || die "$1 requires a value."
                OUTPUT_DIR="$2"
                shift 2
                ;;
            --prompt-path)
                [[ $# -ge 2 ]] || die "$1 requires a value."
                PROMPT_PATH="$2"
                shift 2
                ;;
            --max-samples)
                [[ $# -ge 2 ]] || die "$1 requires a value."
                MAX_SAMPLES="$2"
                shift 2
                ;;
            --sample-seed|--sample_seed)
                [[ $# -ge 2 ]] || die "$1 requires a value."
                SAMPLE_SEED="$2"
                shift 2
                ;;
            --num-trials)
                [[ $# -ge 2 ]] || die "$1 requires a value."
                NUM_TRIALS="$2"
                shift 2
                ;;
            --nq-num-trials)
                [[ $# -ge 2 ]] || die "$1 requires a value."
                NQ_NUM_TRIALS="$2"
                shift 2
                ;;
            --triviaqa-num-trials)
                [[ $# -ge 2 ]] || die "$1 requires a value."
                TRIVIAQA_NUM_TRIALS="$2"
                shift 2
                ;;
            --popqa-num-trials)
                [[ $# -ge 2 ]] || die "$1 requires a value."
                POPQA_NUM_TRIALS="$2"
                shift 2
                ;;
            --hotpot-num-trials)
                [[ $# -ge 2 ]] || die "$1 requires a value."
                HOTPOT_NUM_TRIALS="$2"
                shift 2
                ;;
            --2wiki-num-trials|--twowiki-num-trials)
                [[ $# -ge 2 ]] || die "$1 requires a value."
                TWOWIKI_NUM_TRIALS="$2"
                shift 2
                ;;
            --musique-num-trials)
                [[ $# -ge 2 ]] || die "$1 requires a value."
                MUSIQUE_NUM_TRIALS="$2"
                shift 2
                ;;
            --bamboogle-num-trials)
                [[ $# -ge 2 ]] || die "$1 requires a value."
                BAMBOOGLE_NUM_TRIALS="$2"
                shift 2
                ;;
            --num-workers)
                [[ $# -ge 2 ]] || die "$1 requires a value."
                NUM_WORKERS="$2"
                shift 2
                ;;
            --nq-num-workers)
                [[ $# -ge 2 ]] || die "$1 requires a value."
                NQ_NUM_WORKERS="$2"
                shift 2
                ;;
            --triviaqa-num-workers)
                [[ $# -ge 2 ]] || die "$1 requires a value."
                TRIVIAQA_NUM_WORKERS="$2"
                shift 2
                ;;
            --popqa-num-workers)
                [[ $# -ge 2 ]] || die "$1 requires a value."
                POPQA_NUM_WORKERS="$2"
                shift 2
                ;;
            --hotpot-num-workers)
                [[ $# -ge 2 ]] || die "$1 requires a value."
                HOTPOT_NUM_WORKERS="$2"
                shift 2
                ;;
            --2wiki-num-workers|--twowiki-num-workers)
                [[ $# -ge 2 ]] || die "$1 requires a value."
                TWOWIKI_NUM_WORKERS="$2"
                shift 2
                ;;
            --musique-num-workers)
                [[ $# -ge 2 ]] || die "$1 requires a value."
                MUSIQUE_NUM_WORKERS="$2"
                shift 2
                ;;
            --bamboogle-num-workers)
                [[ $# -ge 2 ]] || die "$1 requires a value."
                BAMBOOGLE_NUM_WORKERS="$2"
                shift 2
                ;;
            --nq-data-path)
                [[ $# -ge 2 ]] || die "$1 requires a value."
                NQ_DATA_PATH="$2"
                shift 2
                ;;
            --triviaqa-data-path)
                [[ $# -ge 2 ]] || die "$1 requires a value."
                TRIVIAQA_DATA_PATH="$2"
                shift 2
                ;;
            --popqa-data-path)
                [[ $# -ge 2 ]] || die "$1 requires a value."
                POPQA_DATA_PATH="$2"
                shift 2
                ;;
            --hotpot-data-path)
                [[ $# -ge 2 ]] || die "$1 requires a value."
                HOTPOT_DATA_PATH="$2"
                shift 2
                ;;
            --2wiki-data-path|--twowiki-data-path)
                [[ $# -ge 2 ]] || die "$1 requires a value."
                TWOWIKI_DATA_PATH="$2"
                shift 2
                ;;
            --musique-data-path)
                [[ $# -ge 2 ]] || die "$1 requires a value."
                MUSIQUE_DATA_PATH="$2"
                shift 2
                ;;
            --bamboogle-data-path)
                [[ $# -ge 2 ]] || die "$1 requires a value."
                BAMBOOGLE_DATA_PATH="$2"
                shift 2
                ;;
            --base-url|--vllm-base-url)
                [[ $# -ge 2 ]] || die "$1 requires a value."
                VLLM_BASE_URL="$2"
                shift 2
                ;;
            --pipeline-python)
                [[ $# -ge 2 ]] || die "$1 requires a value."
                PIPELINE_PYTHON="$2"
                shift 2
                ;;
            --no-use-vlm)
                USE_VLM=0
                shift
                ;;
            --model-path|--model-dir)
                [[ $# -ge 2 ]] || die "$1 requires a value."
                VLLM_MODEL_PATH="$2"
                shift 2
                ;;
            --vllm-python)
                [[ $# -ge 2 ]] || die "$1 requires a value."
                VLLM_PYTHON="$2"
                shift 2
                ;;
            --vllm-cuda-devices)
                [[ $# -ge 2 ]] || die "$1 requires a value."
                VLLM_CUDA_DEVICES="$2"
                shift 2
                ;;
            --vllm-host)
                [[ $# -ge 2 ]] || die "$1 requires a value."
                VLLM_HOST="$2"
                shift 2
                ;;
            --vllm-port)
                [[ $# -ge 2 ]] || die "$1 requires a value."
                VLLM_PORT="$2"
                shift 2
                ;;
            --served-model-name|--model)
                [[ $# -ge 2 ]] || die "$1 requires a value."
                VLLM_SERVED_MODEL_NAME="$2"
                shift 2
                ;;
            --vllm-max-model-len)
                [[ $# -ge 2 ]] || die "$1 requires a value."
                VLLM_MAX_MODEL_LEN="$2"
                shift 2
                ;;
            --vllm-tensor-parallel-size)
                [[ $# -ge 2 ]] || die "$1 requires a value."
                VLLM_TENSOR_PARALLEL_SIZE="$2"
                shift 2
                ;;
            --vllm-gpu-memory-utilization)
                [[ $# -ge 2 ]] || die "$1 requires a value."
                VLLM_GPU_MEMORY_UTILIZATION="$2"
                shift 2
                ;;
            --vllm-ready-timeout)
                [[ $# -ge 2 ]] || die "$1 requires a value."
                VLLM_READY_TIMEOUT="$2"
                shift 2
                ;;
            --vllm-health-url)
                [[ $# -ge 2 ]] || die "$1 requires a value."
                VLLM_HEALTH_URL="$2"
                shift 2
                ;;
            --skip-vllm)
                START_VLLM=0
                KILL_EXISTING_VLLM=0
                CLEAN_TRITON_CACHE=0
                shift
                ;;
            --no-kill-existing-vllm)
                KILL_EXISTING_VLLM=0
                shift
                ;;
            --no-clean-triton-cache)
                CLEAN_TRITON_CACHE=0
                shift
                ;;
            --skip-tool-server)
                START_TOOL_SERVER=0
                shift
                ;;
            --tool-ready-timeout|--tool-server-ready-timeout)
                [[ $# -ge 2 ]] || die "$1 requires a value."
                TOOL_SERVER_READY_TIMEOUT="$2"
                shift 2
                ;;
            --tool-server-python)
                [[ $# -ge 2 ]] || die "$1 requires a value."
                TOOL_SERVER_PYTHON="$2"
                shift 2
                ;;
            --keep-services)
                KEEP_SERVICES=1
                shift
                ;;
            --dry-run)
                DRY_RUN=1
                shift
                ;;
            --)
                shift
                selected+=("$@")
                break
                ;;
            -*)
                die "Unknown option '$1'."
                ;;
            *)
                selected+=("$1")
                shift
                ;;
        esac
    done

    normalize_benchmarks "${selected[@]}"

    if [[ -z "${VLLM_BASE_URL}" ]]; then
        VLLM_BASE_URL="http://localhost:${VLLM_PORT}/v1/"
    fi
    if [[ -z "${VLLM_HEALTH_URL}" ]]; then
        VLLM_HEALTH_URL="http://127.0.0.1:${VLLM_PORT}/v1/models"
    fi
    TOOL_SERVER_HEALTH_URL="http://127.0.0.1:${TOOL_SERVER_PORT}/health"
}

validate_config() {
    if [[ "${TOOL_SERVER_PORT}" != "5000" ]]; then
        die "gen_seq.pipeline currently hardcodes http://localhost:5000/get_observation; keep TOOL_SERVER_PORT=5000."
    fi

    if [[ "${DRY_RUN}" != "1" ]]; then
        [[ -f "${PROMPT_PATH}" ]] || die "Missing prompt file: ${PROMPT_PATH}"

        local bench
        local data_path
        for bench in "${BENCHMARKS[@]}"; do
            data_path="$(benchmark_data_path "${bench}")"
            [[ -f "${data_path}" ]] || die "Missing ${bench} data file: ${data_path}"
        done
    fi

    if [[ "${START_VLLM}" == "1" && "${DRY_RUN}" != "1" ]]; then
        [[ -x "${VLLM_PYTHON}" || -n "$(command -v "${VLLM_PYTHON}" 2>/dev/null)" ]] || die "vLLM python not found: ${VLLM_PYTHON}"
        [[ -d "${VLLM_MODEL_PATH}" ]] || die "Missing vLLM model directory: ${VLLM_MODEL_PATH}"
    fi

    [[ -x "${PIPELINE_PYTHON}" || -n "$(command -v "${PIPELINE_PYTHON}" 2>/dev/null)" ]] || die "Pipeline python not found: ${PIPELINE_PYTHON}"
    if [[ "${START_TOOL_SERVER}" == "1" ]]; then
        [[ -x "${TOOL_SERVER_PYTHON}" || -n "$(command -v "${TOOL_SERVER_PYTHON}" 2>/dev/null)" ]] || die "Tool-server python not found: ${TOOL_SERVER_PYTHON}"
    fi
}

http_ok() {
    local url="$1"
    curl -fsS --max-time 5 "${url}" >/dev/null 2>&1
}

wait_for_http() {
    local name="$1"
    local url="$2"
    local timeout="$3"
    local pid="${4:-}"
    local start_ts
    local now_ts
    local elapsed

    start_ts="$(date +%s)"
    while true; do
        if http_ok "${url}"; then
            log "${name} is ready: ${url}"
            return 0
        fi

        if [[ -n "${pid}" ]] && ! kill -0 "${pid}" 2>/dev/null; then
            die "${name} process exited before becoming ready. Check ${RUN_LOG} and service logs under ${LOG_DIR}."
        fi

        now_ts="$(date +%s)"
        elapsed=$((now_ts - start_ts))
        if (( elapsed >= timeout )); then
            die "Timed out waiting for ${name} after ${timeout}s: ${url}"
        fi

        if (( elapsed > 0 && elapsed % 30 == 0 )); then
            log "Waiting for ${name}... ${elapsed}s elapsed"
        fi
        sleep 2
    done
}

stop_process_group() {
    local name="$1"
    local pid="$2"

    [[ -n "${pid}" ]] || return 0
    if ! kill -0 "${pid}" 2>/dev/null; then
        return 0
    fi

    log "Stopping ${name} process group pid=${pid}"
    kill -TERM "-${pid}" 2>/dev/null || kill -TERM "${pid}" 2>/dev/null || true

    local i
    for i in {1..20}; do
        if ! kill -0 "${pid}" 2>/dev/null; then
            log "${name} stopped"
            return 0
        fi
        sleep 1
    done

    log "${name} did not stop after SIGTERM; sending SIGKILL"
    kill -KILL "-${pid}" 2>/dev/null || kill -KILL "${pid}" 2>/dev/null || true
}

cleanup() {
    local exit_code=$?
    trap - EXIT INT TERM

    if [[ "${KEEP_SERVICES}" != "1" ]]; then
        stop_process_group "tool-server" "${TOOL_SERVER_PID}"
        stop_process_group "vLLM" "${VLLM_PID}"
    else
        log "Keeping services alive by request."
    fi

    if [[ ${exit_code} -eq 0 ]]; then
        log "Selected benchmarks finished successfully."
    else
        log "Evaluation runner exited with code ${exit_code}."
    fi

    exit "${exit_code}"
}

export_runtime_config() {
    export ROOT_DIR LOG_DIR RUN_ID
    export VLLM_MODEL_PATH VLLM_PYTHON VLLM_CUDA_DEVICES VLLM_HOST VLLM_PORT
    export VLLM_SERVED_MODEL_NAME VLLM_MAX_MODEL_LEN VLLM_TENSOR_PARALLEL_SIZE
    export VLLM_GPU_MEMORY_UTILIZATION KILL_EXISTING_VLLM CLEAN_TRITON_CACHE
    export TOOL_SERVER_PYTHON TOOL_SERVER_PORT TOOL_SERVER_WORKERS_PER_TOOL
    export TOOL_SERVER_MAX_CONCURRENT_REQUESTS TOOL_SERVER_THREAD_POOL_SIZE
    export TOOL_SERVER_REQUEST_TIMEOUT TOOL_SERVER_UVI_WORKERS TOOL_SERVER_ROUTER_WORKERS
    export TEXT_BROWSER_RAY_NUM_CPUS TEXT_BROWSER_ACTION_TIMEOUT_SEC
    export TEXT_BROWSER_ENV_RPC_TIMEOUT_SEC VT_HEALTH_CHECK_TIMEOUT
}

start_vllm() {
    local service_log="${LOG_DIR}/vllm_${RUN_ID}.log"

    if [[ "${DRY_RUN}" == "1" ]]; then
        log "DRY-RUN vLLM: ${SCRIPT_PATH} __run_vllm"
        return 0
    fi

    export_runtime_config
    log "Starting vLLM on port ${VLLM_PORT}; log=${service_log}"
    setsid bash "${SCRIPT_PATH}" __run_vllm >"${service_log}" 2>&1 &
    VLLM_PID=$!
    log "vLLM started with process-group pid=${VLLM_PID}"
}

start_tool_server() {
    local service_log="${LOG_DIR}/tool_server_${RUN_ID}.log"

    if [[ "${DRY_RUN}" == "1" ]]; then
        log "DRY-RUN tool-server: ${SCRIPT_PATH} __run_tool_server"
        return 0
    fi

    export_runtime_config
    log "Starting tool-server on port ${TOOL_SERVER_PORT}; log=${service_log}"
    setsid bash "${SCRIPT_PATH}" __run_tool_server >"${service_log}" 2>&1 &
    TOOL_SERVER_PID=$!
    log "tool-server started with process-group pid=${TOOL_SERVER_PID}"
}

benchmark_data_path() {
    case "$1" in
        nq) printf '%s\n' "${NQ_DATA_PATH}" ;;
        triviaqa) printf '%s\n' "${TRIVIAQA_DATA_PATH}" ;;
        popqa) printf '%s\n' "${POPQA_DATA_PATH}" ;;
        hotpot) printf '%s\n' "${HOTPOT_DATA_PATH}" ;;
        2wiki) printf '%s\n' "${TWOWIKI_DATA_PATH}" ;;
        musique) printf '%s\n' "${MUSIQUE_DATA_PATH}" ;;
        bamboogle) printf '%s\n' "${BAMBOOGLE_DATA_PATH}" ;;
        *) die "Unsupported benchmark: $1" ;;
    esac
}

benchmark_trials() {
    case "$1" in
        nq) printf '%s\n' "${NUM_TRIALS:-${NQ_NUM_TRIALS}}" ;;
        triviaqa) printf '%s\n' "${NUM_TRIALS:-${TRIVIAQA_NUM_TRIALS}}" ;;
        popqa) printf '%s\n' "${NUM_TRIALS:-${POPQA_NUM_TRIALS}}" ;;
        hotpot) printf '%s\n' "${NUM_TRIALS:-${HOTPOT_NUM_TRIALS}}" ;;
        2wiki) printf '%s\n' "${NUM_TRIALS:-${TWOWIKI_NUM_TRIALS}}" ;;
        musique) printf '%s\n' "${NUM_TRIALS:-${MUSIQUE_NUM_TRIALS}}" ;;
        bamboogle) printf '%s\n' "${NUM_TRIALS:-${BAMBOOGLE_NUM_TRIALS}}" ;;
        *) die "Unsupported benchmark: $1" ;;
    esac
}

benchmark_workers() {
    case "$1" in
        nq) printf '%s\n' "${NUM_WORKERS:-${NQ_NUM_WORKERS}}" ;;
        triviaqa) printf '%s\n' "${NUM_WORKERS:-${TRIVIAQA_NUM_WORKERS}}" ;;
        popqa) printf '%s\n' "${NUM_WORKERS:-${POPQA_NUM_WORKERS}}" ;;
        hotpot) printf '%s\n' "${NUM_WORKERS:-${HOTPOT_NUM_WORKERS}}" ;;
        2wiki) printf '%s\n' "${NUM_WORKERS:-${TWOWIKI_NUM_WORKERS}}" ;;
        musique) printf '%s\n' "${NUM_WORKERS:-${MUSIQUE_NUM_WORKERS}}" ;;
        bamboogle) printf '%s\n' "${NUM_WORKERS:-${BAMBOOGLE_NUM_WORKERS}}" ;;
        *) die "Unsupported benchmark: $1" ;;
    esac
}

run_benchmark() {
    local bench="$1"
    local data_path
    local trials
    local workers
    local output_file
    local image_output_dir
    local cmd

    data_path="$(benchmark_data_path "${bench}")"
    trials="$(benchmark_trials "${bench}")"
    workers="$(benchmark_workers "${bench}")"
    output_file="${OUTPUT_DIR}/${bench}_test_results.jsonl"
    image_output_dir="${OUTPUT_DIR}/${bench}_obs_images"

    mkdir -p "${OUTPUT_DIR}" "${image_output_dir}"

    cmd=(
        "${PIPELINE_PYTHON}" -m gen_seq.pipeline
        --output_file "${output_file}"
        --data_path "${data_path}"
        --system_prompt "${PROMPT_PATH}"
        --max_samples "${MAX_SAMPLES}"
        --sample_seed "${SAMPLE_SEED}"
        --num_trials "${trials}"
        --base_url "${VLLM_BASE_URL}"
        --model "${VLLM_SERVED_MODEL_NAME}"
        --image_output_dir "${image_output_dir}"
        --num_workers "${workers}"
    )
    if [[ "${USE_VLM}" == "1" ]]; then
        cmd+=(--use_vlm)
    fi

    log "Running benchmark=${bench}, trials=${trials}, workers=${workers}, max_samples=${MAX_SAMPLES}, sample_seed=${SAMPLE_SEED:-sequential}"
    log "Pipeline command: $(quote_cmd "${cmd[@]}")"
    if [[ "${DRY_RUN}" == "1" ]]; then
        return 0
    fi

    (
        cd "${ROOT_DIR}"
        "${cmd[@]}"
    ) 2>&1 | tee -a "${RUN_LOG}"
    log "Benchmark finished: ${bench}"
}

internal_run_vllm() {
    cd "${ROOT_DIR}"
    mkdir -p "${LOG_DIR}"

    if [[ "${KILL_EXISTING_VLLM}" == "1" ]]; then
        pkill -9 -f "VLLM.*--port ${VLLM_PORT}" 2>/dev/null || true
    fi
    if [[ "${CLEAN_TRITON_CACHE}" == "1" ]]; then
        rm -rf "${HOME}/.triton/cache"
    fi

    "${VLLM_PYTHON}" - <<'PYVLLM'
import jinja2
if not hasattr(jinja2, "pass_eval_context"):
    raise SystemExit(
        f"Jinja2 {jinja2.__version__} lacks pass_eval_context; "
        "install jinja2>=3.1.4 in the vLLM server env"
    )
PYVLLM

    local cmd=(
        "${VLLM_PYTHON}" -m vllm.entrypoints.openai.api_server
        --model "${VLLM_MODEL_PATH}"
        --served-model-name "${VLLM_SERVED_MODEL_NAME}"
        --host "${VLLM_HOST}"
        --port "${VLLM_PORT}"
        --trust-remote-code
        --max-model-len "${VLLM_MAX_MODEL_LEN}"
        --tensor-parallel-size "${VLLM_TENSOR_PARALLEL_SIZE}"
    )
    if [[ -n "${VLLM_GPU_MEMORY_UTILIZATION}" ]]; then
        cmd+=(--gpu-memory-utilization "${VLLM_GPU_MEMORY_UTILIZATION}")
    fi

    exec env CUDA_VISIBLE_DEVICES="${VLLM_CUDA_DEVICES}" "${cmd[@]}"
}

internal_run_tool_server() {
    cd "${ROOT_DIR}"

    export PYTHONPATH="${PYTHONPATH:-}:/home/nvidia/yutao/lzt/BrowserAgent_v2:${ROOT_DIR}:${ROOT_DIR}/verl-tool"
    unset http_proxy https_proxy all_proxy

    export TEXT_BROWSER_RAY_NUM_CPUS
    export TEXT_BROWSER_ACTION_TIMEOUT_SEC
    export TEXT_BROWSER_ENV_RPC_TIMEOUT_SEC
    export VT_HEALTH_CHECK_TIMEOUT

    exec "${TOOL_SERVER_PYTHON}" -m verl_tool.servers.serve \
        --tool_type text_browser \
        --host 0.0.0.0 \
        --port "${TOOL_SERVER_PORT}" \
        --uvi_workers "${TOOL_SERVER_UVI_WORKERS}" \
        --router_workers "${TOOL_SERVER_ROUTER_WORKERS}" \
        --workers_per_tool "${TOOL_SERVER_WORKERS_PER_TOOL}" \
        --max_concurrent_requests "${TOOL_SERVER_MAX_CONCURRENT_REQUESTS}" \
        --thread_pool_size "${TOOL_SERVER_THREAD_POOL_SIZE}" \
        --request_timeout "${TOOL_SERVER_REQUEST_TIMEOUT}"
}

main() {
    mkdir -p "${LOG_DIR}"
    parse_args "$@"
    validate_config

    trap cleanup EXIT INT TERM

    log "Selected benchmarks: ${BENCHMARKS[*]}"
    log "Output dir: ${OUTPUT_DIR}"
    log "vLLM model path: ${VLLM_MODEL_PATH}"

    if [[ "${START_VLLM}" == "1" ]]; then
        start_vllm
    else
        log "Skipping vLLM startup; waiting for existing service."
    fi
    if [[ "${DRY_RUN}" != "1" ]]; then
        wait_for_http "vLLM" "${VLLM_HEALTH_URL}" "${VLLM_READY_TIMEOUT}" "${VLLM_PID}"
    fi

    if [[ "${START_TOOL_SERVER}" == "1" ]]; then
        start_tool_server
    else
        log "Skipping tool-server startup; waiting for existing service."
    fi
    if [[ "${DRY_RUN}" != "1" ]]; then
        wait_for_http "tool-server" "${TOOL_SERVER_HEALTH_URL}" "${TOOL_SERVER_READY_TIMEOUT}" "${TOOL_SERVER_PID}"
    fi

    local bench
    for bench in "${BENCHMARKS[@]}"; do
        run_benchmark "${bench}"
    done
}

case "${1:-}" in
    __run_vllm)
        shift
        internal_run_vllm "$@"
        ;;
    __run_tool_server)
        shift
        internal_run_tool_server "$@"
        ;;
    *)
        main "$@"
        ;;
esac
