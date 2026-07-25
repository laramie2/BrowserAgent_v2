#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
export RAY_TMPDIR="${RAY_TMPDIR_OVERRIDE:-${RAY_TMPDIR:-/tmp/ba-ray-$(id -u)-${RUN_ID}}}"
SCRIPT_PATH="${ROOT_DIR}/$(basename "${BASH_SOURCE[0]}")"
LOG_DIR="${ROOT_DIR}/logs"
RUN_LOG="${LOG_DIR}/eval_runner_${RUN_ID}.log"

DEFAULT_MODEL_PATH="${ROOT_DIR}/sft/output/Qwen2.5-VL-7B-Instruct-task-opsrc-hotpot5459-nq6318-sft-5e-5lr-freeze_false-2epoch-merged"
DEFAULT_OUTPUT_DIR="${ROOT_DIR}/gen_seq/results/$(basename "${DEFAULT_MODEL_PATH}")"
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
VLLM_PYTHON="${VLLM_PYTHON:-python}"
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
TOOL_SERVER_WORKERS_PER_TOOL="${TOOL_SERVER_WORKERS_PER_TOOL:-128}"
TOOL_SERVER_MAX_CONCURRENT_REQUESTS="${TOOL_SERVER_MAX_CONCURRENT_REQUESTS:-128}"
TOOL_SERVER_THREAD_POOL_SIZE="${TOOL_SERVER_THREAD_POOL_SIZE:-128}"
TOOL_SERVER_REQUEST_TIMEOUT="${TOOL_SERVER_REQUEST_TIMEOUT:-300}"
TOOL_SERVER_UVI_WORKERS="${TOOL_SERVER_UVI_WORKERS:-1}"
TOOL_SERVER_ROUTER_WORKERS="${TOOL_SERVER_ROUTER_WORKERS:-1}"
TEXT_BROWSER_RAY_NUM_CPUS="${TEXT_BROWSER_RAY_NUM_CPUS:-128}"
TEXT_BROWSER_MAX_ACTIVE_ACTORS="${TEXT_BROWSER_MAX_ACTIVE_ACTORS:-128}"
TEXT_BROWSER_IDLE_POOL_SIZE="${TEXT_BROWSER_IDLE_POOL_SIZE:-16}"
TEXT_BROWSER_ACTOR_CPUS="${TEXT_BROWSER_ACTOR_CPUS:-1}"
TEXT_BROWSER_ACTION_TIMEOUT_SEC="${TEXT_BROWSER_ACTION_TIMEOUT_SEC:-240}"
TEXT_BROWSER_ENV_RPC_TIMEOUT_SEC="${TEXT_BROWSER_ENV_RPC_TIMEOUT_SEC:-220}"
VT_HEALTH_CHECK_TIMEOUT="${VT_HEALTH_CHECK_TIMEOUT:-300}"
MINI_WEB_ARENA_PROMPT_MODEL="${MINI_WEB_ARENA_PROMPT_MODEL:-${ROOT_DIR}/models/Qwen2.5-14B-Instruct}"
MINI_WEB_ARENA_TOKENIZER_LOCAL_ONLY="${MINI_WEB_ARENA_TOKENIZER_LOCAL_ONLY:-1}"
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
NUM_WORKERS="${NUM_WORKERS:-128}"
NQ_NUM_WORKERS="${NQ_NUM_WORKERS:-128}"
TRIVIAQA_NUM_WORKERS="${TRIVIAQA_NUM_WORKERS:-128}"
POPQA_NUM_WORKERS="${POPQA_NUM_WORKERS:-128}"
HOTPOT_NUM_WORKERS="${HOTPOT_NUM_WORKERS:-128}"
TWOWIKI_NUM_WORKERS="${TWOWIKI_NUM_WORKERS:-128}"
MUSIQUE_NUM_WORKERS="${MUSIQUE_NUM_WORKERS:-128}"
BAMBOOGLE_NUM_WORKERS="${BAMBOOGLE_NUM_WORKERS:-128}"
VLLM_BASE_URL="${VLLM_BASE_URL:-}"
TOOL_SERVER_BASE_URL="${TOOL_SERVER_BASE_URL:-}"
ENV_URL="${ENV_URL:-}"
BROWSER_URL="${BROWSER_URL:-http://localhost:22015/wikipedia_en_all_maxi_2022-05/A/User:The_other_Kiwix_guy/Landing}"
COMPRESSION_FACTOR="${COMPRESSION_FACTOR:-1.2}"
IMAGE_MAX_WIDTH="${IMAGE_MAX_WIDTH:-2048}"
IMAGE_MAX_HEIGHT="${IMAGE_MAX_HEIGHT:-2048}"
MAX_STEPS="${MAX_STEPS:-30}"
MAX_TOKENS="${MAX_TOKENS:-1024}"
LLM_REQUEST_TIMEOUT="${LLM_REQUEST_TIMEOUT:-300}"
PIPELINE_ENV_REQUEST_TIMEOUT="${PIPELINE_ENV_REQUEST_TIMEOUT:-300}"
TEMPERATURE="${TEMPERATURE:-0.3}"
RUN_TOKEN_STATS="${RUN_TOKEN_STATS:-1}"
TOKEN_STATS_MODEL_PATH="${TOKEN_STATS_MODEL_PATH:-}"
USE_VLM="${USE_VLM:-1}"
KEEP_SERVICES="${KEEP_SERVICES:-0}"
DRY_RUN="${DRY_RUN:-0}"
RESUME="${RESUME:-1}"
BENCHMARK_MAX_RETRIES="${BENCHMARK_MAX_RETRIES:-5}"
BENCHMARK_RETRY_DELAY_SEC="${BENCHMARK_RETRY_DELAY_SEC:-5}"

VLLM_PID=""
TOOL_SERVER_PID=""
TOOL_SERVER_START_COUNT=0
BENCHMARKS=()
EVAL_TOTAL_SECONDS=0
EVAL_TOTAL_COMPLETED=0
EVAL_BENCHMARK_COUNT=0

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
      --nq-num-workers N          Pipeline workers for nq. Default: 128.
      --popqa-num-workers N       Pipeline workers for popqa. Default: 128.
      --hotpot-num-workers N      Pipeline workers for hotpot. Default: 128.
      --2wiki-num-workers N       Pipeline workers for 2wiki. Default: 128.
      --musique-num-workers N     Pipeline workers for musique. Default: 128.
      --bamboogle-num-workers N   Pipeline workers for bamboogle. Default: 128.
      --triviaqa-num-workers N    Pipeline workers for triviaqa. Default: 128.
      --nq-data-path PATH         NQ parquet path.
      --triviaqa-data-path PATH    TriviaQA parquet path.
      --popqa-data-path PATH      PopQA parquet path.
      --hotpot-data-path PATH     HotpotQA parquet path.
      --2wiki-data-path PATH      2Wiki parquet path.
      --musique-data-path PATH    MuSiQue parquet path.
      --bamboogle-data-path PATH  Bamboogle parquet path.
      --base-url URL              OpenAI-compatible vLLM API base URL.
      --pipeline-python PATH      Python executable for gen_seq.pipeline.
      --compression-factor VALUE VTC image compression ratio. Default: 1.2.
      --max-steps N               Maximum browser steps per trajectory. Default: 30.
      --max-tokens N              Maximum generated tokens per step. Default: 1024.
      --env-url URL               Full tool observation endpoint. Derived from tool port by default.
      --browser-url URL           Kiwix landing URL passed to the browser environment.
      --llm-request-timeout SEC   Pipeline-to-vLLM timeout. Default: 300.
      --env-request-timeout SEC   Pipeline-to-tool-server timeout. Default: 300.
      --token-stats-model-path DIR  Local tokenizer/processor used for token counting.
      --no-token-stats            Skip post-evaluation compressed/raw token counting.
      --no-use-vlm                Do not pass --use_vlm to gen_seq.pipeline.
      --no-resume                 Re-run all samples instead of skipping existing results.
      --benchmark-max-retries N    Resume a benchmark after exit code 2. Default: 5.
      --benchmark-retry-delay SEC  Delay before an automatic retry. Default: 5.

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
      --tool-server-port PORT     Tool server port; pipeline endpoint follows it.
      --tool-workers N            Workers per tool. Default: 128.
      --tool-max-requests N       Maximum concurrent tool requests. Default: 128.
      --tool-thread-pool-size N   Tool server thread pool size. Default: 128.
      --browser-max-actors N      Maximum active Ray browser actors. Default: 128.
      --browser-idle-pool N       Warm idle browser actors. Default: 16.
      --browser-ray-cpus N        Ray CPU budget for text browser actors.
      --skip-tool-server          Use an already running tool-server.
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
            --compression-factor)
                [[ $# -ge 2 ]] || die "$1 requires a value."
                COMPRESSION_FACTOR="$2"
                shift 2
                ;;
            --max-steps)
                [[ $# -ge 2 ]] || die "$1 requires a value."
                MAX_STEPS="$2"
                shift 2
                ;;
            --max-tokens)
                [[ $# -ge 2 ]] || die "$1 requires a value."
                MAX_TOKENS="$2"
                shift 2
                ;;
            --env-url)
                [[ $# -ge 2 ]] || die "$1 requires a value."
                ENV_URL="$2"
                shift 2
                ;;
            --browser-url)
                [[ $# -ge 2 ]] || die "$1 requires a value."
                BROWSER_URL="$2"
                shift 2
                ;;
            --llm-request-timeout)
                [[ $# -ge 2 ]] || die "$1 requires a value."
                LLM_REQUEST_TIMEOUT="$2"
                shift 2
                ;;
            --env-request-timeout)
                [[ $# -ge 2 ]] || die "$1 requires a value."
                PIPELINE_ENV_REQUEST_TIMEOUT="$2"
                shift 2
                ;;
            --token-stats-model-path)
                [[ $# -ge 2 ]] || die "$1 requires a value."
                TOKEN_STATS_MODEL_PATH="$2"
                shift 2
                ;;
            --no-token-stats)
                RUN_TOKEN_STATS=0
                shift
                ;;
            --no-use-vlm)
                USE_VLM=0
                shift
                ;;
            --no-resume)
                RESUME=0
                shift
                ;;
            --benchmark-max-retries)
                [[ $# -ge 2 ]] || die "$1 requires a value."
                BENCHMARK_MAX_RETRIES="$2"
                shift 2
                ;;
            --benchmark-retry-delay)
                [[ $# -ge 2 ]] || die "$1 requires a value."
                BENCHMARK_RETRY_DELAY_SEC="$2"
                shift 2
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
            --tool-server-port)
                [[ $# -ge 2 ]] || die "$1 requires a value."
                TOOL_SERVER_PORT="$2"
                shift 2
                ;;
            --tool-workers)
                [[ $# -ge 2 ]] || die "$1 requires a value."
                TOOL_SERVER_WORKERS_PER_TOOL="$2"
                shift 2
                ;;
            --tool-max-requests)
                [[ $# -ge 2 ]] || die "$1 requires a value."
                TOOL_SERVER_MAX_CONCURRENT_REQUESTS="$2"
                shift 2
                ;;
            --tool-thread-pool-size)
                [[ $# -ge 2 ]] || die "$1 requires a value."
                TOOL_SERVER_THREAD_POOL_SIZE="$2"
                shift 2
                ;;
            --browser-max-actors)
                [[ $# -ge 2 ]] || die "$1 requires a value."
                TEXT_BROWSER_MAX_ACTIVE_ACTORS="$2"
                shift 2
                ;;
            --browser-idle-pool)
                [[ $# -ge 2 ]] || die "$1 requires a value."
                TEXT_BROWSER_IDLE_POOL_SIZE="$2"
                shift 2
                ;;
            --browser-ray-cpus)
                [[ $# -ge 2 ]] || die "$1 requires a value."
                TEXT_BROWSER_RAY_NUM_CPUS="$2"
                shift 2
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
    if [[ -z "${TOOL_SERVER_BASE_URL}" ]]; then
        TOOL_SERVER_BASE_URL="http://127.0.0.1:${TOOL_SERVER_PORT}"
    fi
    if [[ -z "${ENV_URL}" ]]; then
        ENV_URL="${TOOL_SERVER_BASE_URL%/}/get_observation"
    fi
    if [[ -z "${TOKEN_STATS_MODEL_PATH}" ]]; then
        TOKEN_STATS_MODEL_PATH="${VLLM_MODEL_PATH}"
    fi
}

validate_config() {
    [[ "${BENCHMARK_MAX_RETRIES}" =~ ^[0-9]+$ ]] || die "--benchmark-max-retries must be a non-negative integer."
    [[ "${BENCHMARK_RETRY_DELAY_SEC}" =~ ^[0-9]+$ ]] || die "--benchmark-retry-delay must be a non-negative integer."
    if [[ "${RESUME}" != "1" && "${BENCHMARK_MAX_RETRIES}" != "0" ]]; then
        log "Automatic benchmark retries disabled because resume is disabled"
        BENCHMARK_MAX_RETRIES=0
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
        if [[ "${DRY_RUN}" != "1" && "${MINI_WEB_ARENA_TOKENIZER_LOCAL_ONLY}" == "1" ]]; then
            [[ -f "${MINI_WEB_ARENA_PROMPT_MODEL}/tokenizer.json" ]] || die "Missing local prompt tokenizer: ${MINI_WEB_ARENA_PROMPT_MODEL}/tokenizer.json"
        fi
    fi
    if [[ "${RUN_TOKEN_STATS}" == "1" && "${DRY_RUN}" != "1" ]]; then
        [[ -d "${TOKEN_STATS_MODEL_PATH}" ]] || die "Missing token-stats model directory: ${TOKEN_STATS_MODEL_PATH}"
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
        # Check the process before the endpoint. Otherwise an old service on the
        # same port can be mistaken for the process that was just launched.
        if [[ -n "${pid}" ]] && ! kill -0 "${pid}" 2>/dev/null; then
            die "${name} process exited before becoming ready. Check ${RUN_LOG} and service logs under ${LOG_DIR}."
        fi

        if http_ok "${url}"; then
            log "${name} is ready: ${url}"
            return 0
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

process_group_exists() {
    local pid="$1"
    ps -eo pgid=,stat= | awk -v target="${pid}" '
        $1 == target && $2 !~ /^Z/ { found = 1 }
        END { exit(found ? 0 : 1) }
    '
}

stop_process_group() {
    local name="$1"
    local pid="$2"

    [[ -n "${pid}" ]] || return 0
    process_group_exists "${pid}" || return 0

    log "Stopping ${name} process group pid=${pid}"
    kill -TERM "-${pid}" 2>/dev/null || kill -TERM "${pid}" 2>/dev/null || true

    local i
    for i in {1..20}; do
        if ! process_group_exists "${pid}"; then
            wait "${pid}" 2>/dev/null || true
            log "${name} stopped"
            return 0
        fi
        sleep 1
    done

    log "${name} did not stop after SIGTERM; sending SIGKILL"
    kill -KILL "-${pid}" 2>/dev/null || kill -KILL "${pid}" 2>/dev/null || true
    for i in {1..10}; do
        if ! process_group_exists "${pid}"; then
            wait "${pid}" 2>/dev/null || true
            return 0
        fi
        sleep 1
    done
    log "WARNING: ${name} process group ${pid} still appears alive after SIGKILL"
}

stop_existing_vllm() {
    [[ "${DRY_RUN}" != "1" ]] || return 0
    http_ok "${VLLM_HEALTH_URL}" || return 0
    if [[ "${KILL_EXISTING_VLLM}" != "1" ]]; then
        die "A vLLM service is already responding on port ${VLLM_PORT}. Use --skip-vllm to reuse it or allow the runner to replace it."
    fi

    local pattern="vllm\.entrypoints\.openai\.api_server.*--port([=[:space:]])${VLLM_PORT}([[:space:]]|$)"
    if ! pgrep -f "${pattern}" >/dev/null 2>&1; then
        die "Port ${VLLM_PORT} already serves vLLM, but its process could not be identified safely. Stop it explicitly or use --skip-vllm."
    fi

    log "Stopping existing vLLM service on port ${VLLM_PORT} before model startup"
    pkill -TERM -f "${pattern}" 2>/dev/null || true
    local i
    for i in {1..30}; do
        http_ok "${VLLM_HEALTH_URL}" || return 0
        sleep 1
    done
    log "Existing vLLM did not stop after SIGTERM; sending SIGKILL"
    pkill -KILL -f "${pattern}" 2>/dev/null || true
    for i in {1..10}; do
        http_ok "${VLLM_HEALTH_URL}" || return 0
        sleep 1
    done
    die "Existing vLLM is still responding on port ${VLLM_PORT}."
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
    export TEXT_BROWSER_RAY_NUM_CPUS TEXT_BROWSER_MAX_ACTIVE_ACTORS
    export TEXT_BROWSER_IDLE_POOL_SIZE TEXT_BROWSER_ACTOR_CPUS
    export TEXT_BROWSER_ACTION_TIMEOUT_SEC
    export TEXT_BROWSER_ENV_RPC_TIMEOUT_SEC VT_HEALTH_CHECK_TIMEOUT
    export MINI_WEB_ARENA_PROMPT_MODEL MINI_WEB_ARENA_TOKENIZER_LOCAL_ONLY
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
    TOOL_SERVER_START_COUNT=$((TOOL_SERVER_START_COUNT + 1))
    local suffix=""
    if (( TOOL_SERVER_START_COUNT > 1 )); then
        suffix="_restart$((TOOL_SERVER_START_COUNT - 1))"
    fi
    local service_log="${LOG_DIR}/tool_server_${RUN_ID}${suffix}.log"

    if [[ "${DRY_RUN}" == "1" ]]; then
        log "DRY-RUN tool-server: ${SCRIPT_PATH} __run_tool_server"
        return 0
    fi
    if http_ok "${TOOL_SERVER_HEALTH_URL}"; then
        die "A tool-server is already responding on port ${TOOL_SERVER_PORT}. Use --skip-tool-server to reuse it or stop it before this run."
    fi

    export_runtime_config
    log "Starting tool-server on port ${TOOL_SERVER_PORT}; log=${service_log}"
    setsid bash "${SCRIPT_PATH}" __run_tool_server >"${service_log}" 2>&1 &
    TOOL_SERVER_PID=$!
    log "tool-server started with process-group pid=${TOOL_SERVER_PID}"
}

restart_tool_server_for_retry() {
    if [[ "${START_TOOL_SERVER}" != "1" ]]; then
        log "Tool-server is externally managed; retrying without restarting it"
        return 0
    fi

    stop_process_group "tool-server" "${TOOL_SERVER_PID}"
    TOOL_SERVER_PID=""
    if http_ok "${TOOL_SERVER_HEALTH_URL}"; then
        die "Owned tool-server still responds after shutdown; refusing to start a duplicate on port ${TOOL_SERVER_PORT}."
    fi
    start_tool_server
    wait_for_http "tool-server" "${TOOL_SERVER_HEALTH_URL}" "${TOOL_SERVER_READY_TIMEOUT}" "${TOOL_SERVER_PID}"
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

count_jsonl_records() {
    local file_path="$1"
    if [[ ! -f "${file_path}" ]]; then
        printf '0\n'
        return 0
    fi
    awk 'NF { count++ } END { print count + 0 }' "${file_path}"
}

format_seconds() {
    local total_seconds="$1"
    printf '%02d:%02d:%02d' \
        "$((total_seconds / 3600))" \
        "$(((total_seconds % 3600) / 60))" \
        "$((total_seconds % 60))"
}

run_benchmark_once() {
    local bench="$1"
    local attempt="$2"
    local data_path
    local trials
    local workers
    local output_file
    local image_output_dir
    local cmd
    local records_before
    local records_after
    local completed_records
    local start_ts
    local end_ts
    local elapsed_seconds
    local duration
    local samples_per_second
    local seconds_per_sample
    local pipeline_status

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
        --env-url "${ENV_URL}"
        --browser-url "${BROWSER_URL}"
        --compression-factor "${COMPRESSION_FACTOR}"
        --image-max-width "${IMAGE_MAX_WIDTH}"
        --image-max-height "${IMAGE_MAX_HEIGHT}"
        --max-steps "${MAX_STEPS}"
        --max-tokens "${MAX_TOKENS}"
        --temperature "${TEMPERATURE}"
        --llm-request-timeout "${LLM_REQUEST_TIMEOUT}"
        --env-request-timeout "${PIPELINE_ENV_REQUEST_TIMEOUT}"
        --image_output_dir "${image_output_dir}"
        --num_workers "${workers}"
    )
    if [[ "${USE_VLM}" == "1" ]]; then
        cmd+=(--use_vlm)
    fi
    if [[ "${RESUME}" == "1" ]]; then
        cmd+=(--resume)
    fi

    log "Running benchmark=${bench}, attempt=${attempt}, trials=${trials}, workers=${workers}, max_samples=${MAX_SAMPLES}, sample_seed=${SAMPLE_SEED:-sequential}, compression_factor=${COMPRESSION_FACTOR}"
    log "Pipeline command: $(quote_cmd "${cmd[@]}")"
    if [[ "${DRY_RUN}" == "1" ]]; then
        return 0
    fi

    records_before="$(count_jsonl_records "${output_file}")"
    start_ts="$(date +%s)"
    if (
        cd "${ROOT_DIR}"
        "${cmd[@]}"
    ) 2>&1 | tee -a "${RUN_LOG}"; then
        pipeline_status=0
    else
        pipeline_status=$?
    fi
    end_ts="$(date +%s)"
    elapsed_seconds=$((end_ts - start_ts))
    records_after="$(count_jsonl_records "${output_file}")"
    completed_records=$((records_after - records_before))
    if (( completed_records < 0 )); then
        completed_records=0
    fi
    duration="$(format_seconds "${elapsed_seconds}")"

    if (( elapsed_seconds > 0 )); then
        samples_per_second="$(awk -v count="${completed_records}" -v seconds="${elapsed_seconds}" 'BEGIN { printf "%.4f", count / seconds }')"
    else
        samples_per_second="0.0000"
    fi
    if (( completed_records > 0 )); then
        seconds_per_sample="$(awk -v count="${completed_records}" -v seconds="${elapsed_seconds}" 'BEGIN { printf "%.4f", seconds / count }')"
    else
        seconds_per_sample="n/a"
    fi

    EVAL_TOTAL_SECONDS=$((EVAL_TOTAL_SECONDS + elapsed_seconds))
    EVAL_TOTAL_COMPLETED=$((EVAL_TOTAL_COMPLETED + completed_records))
    log "BENCHMARK_TIMING benchmark=${bench} attempt=${attempt} duration=${duration} elapsed_seconds=${elapsed_seconds} completed_sample_evals=${completed_records} avg_sample_evals_per_second=${samples_per_second} avg_seconds_per_sample_eval=${seconds_per_sample}"

    if (( pipeline_status != 0 )); then
        log "Benchmark attempt failed: benchmark=${bench}, attempt=${attempt}, exit_code=${pipeline_status}"
        return "${pipeline_status}"
    fi
    log "Benchmark finished: ${bench}, attempts=${attempt}"
}

run_benchmark() {
    local bench="$1"
    local attempt=1
    local max_attempts=$((BENCHMARK_MAX_RETRIES + 1))
    local status

    while true; do
        if run_benchmark_once "${bench}" "${attempt}"; then
            EVAL_BENCHMARK_COUNT=$((EVAL_BENCHMARK_COUNT + 1))
            return 0
        else
            status=$?
        fi

        if (( status != 2 )); then
            log "Benchmark ${bench} failed with non-retryable exit code ${status}"
            return "${status}"
        fi
        if [[ "${RESUME}" != "1" ]]; then
            log "Benchmark ${bench} cannot retry because resume is disabled"
            return "${status}"
        fi
        if (( attempt >= max_attempts )); then
            log "Benchmark ${bench} exhausted ${BENCHMARK_MAX_RETRIES} automatic retries"
            return "${status}"
        fi

        log "AUTO_RETRY benchmark=${bench} failed_attempt=${attempt} next_attempt=$((attempt + 1)) max_attempts=${max_attempts} delay_seconds=${BENCHMARK_RETRY_DELAY_SEC}; preserving vLLM and resuming saved JSONL"
        if (( BENCHMARK_RETRY_DELAY_SEC > 0 )); then
            sleep "${BENCHMARK_RETRY_DELAY_SEC}"
        fi
        restart_tool_server_for_retry
        attempt=$((attempt + 1))
    done
}

internal_run_vllm() {
    cd "${ROOT_DIR}"
    mkdir -p "${LOG_DIR}"

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

    export PYTHONPATH="${PYTHONPATH:-}:${ROOT_DIR}:${ROOT_DIR}/verl-tool"
    unset http_proxy https_proxy all_proxy RAY_ADDRESS

    export TEXT_BROWSER_RAY_NUM_CPUS
    export TEXT_BROWSER_MAX_ACTIVE_ACTORS
    export TEXT_BROWSER_IDLE_POOL_SIZE
    export TEXT_BROWSER_ACTOR_CPUS
    export TEXT_BROWSER_ACTION_TIMEOUT_SEC
    export TEXT_BROWSER_ENV_RPC_TIMEOUT_SEC
    export VT_HEALTH_CHECK_TIMEOUT
    export MINI_WEB_ARENA_PROMPT_MODEL
    export MINI_WEB_ARENA_TOKENIZER_LOCAL_ONLY

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

run_token_stats() {
    local output_json="${OUTPUT_DIR}/token_usage_summary.json"
    local output_csv="${OUTPUT_DIR}/token_usage_by_benchmark.csv"
    local cmd=(
        "${PIPELINE_PYTHON}" -m gen_seq.token_stats
        --input "${OUTPUT_DIR}"
        --glob "*_test_results.jsonl"
        --model_path "${TOKEN_STATS_MODEL_PATH}"
        --system_prompt "${PROMPT_PATH}"
        --output_json "${output_json}"
        --output_csv "${output_csv}"
        --vtc_compression_factor "${COMPRESSION_FACTOR}"
    )

    log "Token stats command: $(quote_cmd "${cmd[@]}")"
    if [[ "${DRY_RUN}" == "1" ]]; then
        return 0
    fi
    (
        cd "${ROOT_DIR}"
        "${cmd[@]}"
    ) 2>&1 | tee -a "${RUN_LOG}"
    log "Token usage summary: ${output_json}"
}

main() {
    mkdir -p "${LOG_DIR}"
    parse_args "$@"
    validate_config

    trap cleanup EXIT INT TERM

    log "Selected benchmarks: ${BENCHMARKS[*]}"
    log "Output dir: ${OUTPUT_DIR}"
    log "vLLM model path: ${VLLM_MODEL_PATH}"
    log "Pipeline config: env_url=${ENV_URL}, vllm_url=${VLLM_BASE_URL}, compression_factor=${COMPRESSION_FACTOR}, max_steps=${MAX_STEPS}, max_tokens=${MAX_TOKENS}, token_stats=${RUN_TOKEN_STATS}"
    log "Concurrency: pipeline_workers=${NUM_WORKERS}, tool_workers=${TOOL_SERVER_WORKERS_PER_TOOL}, tool_max_requests=${TOOL_SERVER_MAX_CONCURRENT_REQUESTS}, tool_thread_pool=${TOOL_SERVER_THREAD_POOL_SIZE}, browser_max_actors=${TEXT_BROWSER_MAX_ACTIVE_ACTORS}, browser_idle_pool=${TEXT_BROWSER_IDLE_POOL_SIZE}, browser_actor_cpus=${TEXT_BROWSER_ACTOR_CPUS}"
    log "Recovery: benchmark_max_retries=${BENCHMARK_MAX_RETRIES}, retry_delay=${BENCHMARK_RETRY_DELAY_SEC}s, resume=${RESUME}, ray_tmpdir=${RAY_TMPDIR}"
    log "Timeouts: env_rpc=${TEXT_BROWSER_ENV_RPC_TIMEOUT_SEC}s, browser_action=${TEXT_BROWSER_ACTION_TIMEOUT_SEC}s, tool_request=${TOOL_SERVER_REQUEST_TIMEOUT}s; prompt_tokenizer=${MINI_WEB_ARENA_PROMPT_MODEL}, local_only=${MINI_WEB_ARENA_TOKENIZER_LOCAL_ONLY}"

    if [[ "${START_VLLM}" == "1" ]]; then
        stop_existing_vllm
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

    local total_duration
    local overall_samples_per_second
    local overall_seconds_per_sample
    total_duration="$(format_seconds "${EVAL_TOTAL_SECONDS}")"
    if (( EVAL_TOTAL_SECONDS > 0 )); then
        overall_samples_per_second="$(awk -v count="${EVAL_TOTAL_COMPLETED}" -v seconds="${EVAL_TOTAL_SECONDS}" 'BEGIN { printf "%.4f", count / seconds }')"
    else
        overall_samples_per_second="0.0000"
    fi
    if (( EVAL_TOTAL_COMPLETED > 0 )); then
        overall_seconds_per_sample="$(awk -v count="${EVAL_TOTAL_COMPLETED}" -v seconds="${EVAL_TOTAL_SECONDS}" 'BEGIN { printf "%.4f", seconds / count }')"
    else
        overall_seconds_per_sample="n/a"
    fi
    log "EVALUATION_TIMING_SUMMARY benchmarks=${EVAL_BENCHMARK_COUNT} duration=${total_duration} elapsed_seconds=${EVAL_TOTAL_SECONDS} completed_sample_evals=${EVAL_TOTAL_COMPLETED} avg_sample_evals_per_second=${overall_samples_per_second} avg_seconds_per_sample_eval=${overall_seconds_per_sample}"

    if [[ "${RUN_TOKEN_STATS}" == "1" ]]; then
        run_token_stats
    else
        log "Skipping token statistics by request."
    fi
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
