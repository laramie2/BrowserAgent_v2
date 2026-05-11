#!/usr/bin/env bash
set -euo pipefail

# 默认从 RL/configs/train.yaml 读取配置；已有 .success 的实验会跳过。
# 推荐入口：
#   bash RL/scripts/auto_train.sh
# 兼容入口：
#   bash RL/auto_train.sh
#
# 使用同一训练脚本跑 DAPO 网格/成组实验，日志结构相同但落在 logs/dapo/。
#   TRAIN_ALGO=dapo bash RL/scripts/auto_train.sh

# 跳过当前实验
# Ctrl + C
# 或
# touch 对应 GRID_LOG_ROOT 下的 SKIP_CURRENT

# Ctrl + C
# Ctrl + C
# 或
# touch 对应 GRID_LOG_ROOT 下的 STOP_ALL

# 强制全部重跑
#   FORCE_RERUN=1 bash RL/scripts/auto_train.sh

# 跑指定的几组完整配置（设置后会跳过网格搜索）
#   EXPERIMENT_CONFIGS="n6_b64_lr3e-7_t0.8_kl0.01_warmup10 n6_b64_lr5e-7_t0.7_e4_kl0.005_warmup20" bash RL/scripts/auto_train.sh
# 或修改 RL/configs/train.yaml 里的 auto.grid.EXPERIMENT_CONFIG_LIST
#
# 配置格式：
#   n{rollout_n}_b{batch}_lr{lr}_t{temperature}_e{epoch}_kl{kl_loss_coef}_warmup{lr_warmup_steps}_turns{max_turns}
# 其中 e{epoch}、turns{max_turns} 可省略；e 省略时使用 EPOCH_LIST 的第一个值，turns 省略时使用 train.sh 默认值。

# 跳过失败实验
#   SKIP_FAILED=1 bash RL/scripts/auto_train.sh

# 失败就停止
#   CONTINUE_ON_FAIL=0 bash RL/scripts/auto_train.sh

# 查看结果
# cat 对应 GRID_LOG_ROOT 下的 summary.tsv

# 查看最优实验（按 metric 排序）
# awk -F'\t' 'NR>1 && $3!="NA"{print $0}' 对应 GRID_LOG_ROOT 下的 summary.tsv \
#   | sort -t$'\t' -k3,3gr \
#   | head

# 查看某个实验日志
# # 外层日志
# tail -f 对应 GRID_LOG_ROOT 下的 *.outer.log

# # 内层训练日志（关键）
# tail -f 对应 TRAIN_LOG_ROOT 下的 *.log

# 日志结构
# logs/
# └── <algo>/
#     ├── grid_search/
#     │   └── <sft_model>/<train_val_data>/
#     │       ├── *.outer.log    # 外层日志（控制+状态）
#     │       ├── *.success
#     │       ├── *.failed
#     │       ├── summary.tsv    # 汇总（最重要）
#     │
#     └── train/
#         └── <sft_model>/<train_val_data>/
#             ├── *.log          # 训练日志（early stop 用这个）

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RL_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PROJECT_DIR="$(cd "$RL_DIR/.." && pwd)"
TRAIN_CONFIG="${TRAIN_CONFIG:-$RL_DIR/configs/train.yaml}"

if [ -f "$TRAIN_CONFIG" ]; then
  eval "$(python3 "$SCRIPT_DIR/yaml_env.py" auto "$TRAIN_CONFIG")"
fi

cd "$RL_DIR"

TRAIN_ALGO="${TRAIN_ALGO:-dapo}"

case "$TRAIN_ALGO" in
  mt_grpo|grpo|ppo|default)
    TRAIN_SCRIPT="${TRAIN_SCRIPT_OVERRIDE:-$SCRIPT_DIR/train.sh}"
    LOG_ROOT="${LOG_ROOT_OVERRIDE:-$RL_DIR/logs/mt_grpo}"
    BASE_RAY_TMP_DEFAULT="/DATA/disk0/yjb/yutao/ray_tmp"
    ;;
  dapo)
    TRAIN_SCRIPT="${TRAIN_SCRIPT_OVERRIDE:-$SCRIPT_DIR/train.sh}"
    LOG_ROOT="${LOG_ROOT_OVERRIDE:-$RL_DIR/logs/dapo}"
    BASE_RAY_TMP_DEFAULT="/tmp/raydapo_grid"
    ;;
  *)
    echo "[CONFIG ERROR] unsupported TRAIN_ALGO='$TRAIN_ALGO'. Use mt_grpo or dapo." >&2
    exit 1
    ;;
esac

sanitize_run_tag() {
  printf '%s' "$1" | tr -c 'A-Za-z0-9._-' '-'
}

SFT_MODEL_NAME="${SFT_MODEL_NAME_OVERRIDE:-task-opsrc_12619-sft-5e-5lr-freeze_false-2epoch}"
BENCHMARK="${BENCHMARK_OVERRIDE:-train_hotpot500_nq500}"
VAL_DATASET_NAME="${VAL_DATASET_NAME_OVERRIDE:-test_20}"

MODEL_LOG_COMPONENT="$(sanitize_run_tag "$SFT_MODEL_NAME")"
DATA_LOG_COMPONENT="$(sanitize_run_tag "${BENCHMARK}_val-${VAL_DATASET_NAME}")"
LOG_CONTEXT_DIR="${LOG_CONTEXT_DIR_OVERRIDE:-${MODEL_LOG_COMPONENT}/${DATA_LOG_COMPONENT}}"

GRID_LOG_ROOT="$LOG_ROOT/grid_search/$LOG_CONTEXT_DIR"
TRAIN_LOG_ROOT="$LOG_ROOT/train/$LOG_CONTEXT_DIR"
mkdir -p "$GRID_LOG_ROOT"
mkdir -p "$TRAIN_LOG_ROOT"

SUMMARY_FILE="$GRID_LOG_ROOT/summary.tsv"

export SFT_MODEL_NAME_OVERRIDE="$SFT_MODEL_NAME"
export BENCHMARK_OVERRIDE="$BENCHMARK"
export VAL_DATASET_NAME_OVERRIDE="$VAL_DATASET_NAME"
export LOG_CONTEXT_DIR_OVERRIDE="$LOG_CONTEXT_DIR"
export LOG_DIR_OVERRIDE="$LOG_ROOT"

# ===== 搜索空间 =====
declare -p N_LIST >/dev/null 2>&1 || N_LIST=(6)
declare -p BATCH_LIST >/dev/null 2>&1 || BATCH_LIST=(64)
declare -p LR_LIST >/dev/null 2>&1 || LR_LIST=("1e-7" "3e-7" "5e-7")
declare -p TEMP_LIST >/dev/null 2>&1 || TEMP_LIST=("0.7" "1.0")
declare -p EPOCH_LIST >/dev/null 2>&1 || EPOCH_LIST=(4)
declare -p KL_LOSS_COEF_LIST >/dev/null 2>&1 || KL_LOSS_COEF_LIST=("0.005" "0.001")
declare -p LR_WARMUP_STEPS_LIST >/dev/null 2>&1 || LR_WARMUP_STEPS_LIST=(10)
# 留空表示网格搜索不覆盖 MAX_TURNS_OVERRIDE，使用 train.sh/YAML 默认值。
declare -p MAX_TURNS_LIST >/dev/null 2>&1 || MAX_TURNS_LIST=()

# ===== 统一训练参数 =====
# 这些变量会导出给 scripts/train.sh；根目录兼容入口也使用同一套
# override 名称，避免自动训练和手动训练的配置漂移。
RL_ALG_VALUE="${RL_ALG_OVERRIDE:-}"
PPO_MINI_BATCH_SIZE_VALUE="${PPO_MINI_BATCH_SIZE_OVERRIDE:-}"
USE_KL_LOSS_VALUE="${USE_KL_LOSS_OVERRIDE:-}"
KL_COEF_VALUE="${KL_COEF_OVERRIDE:-0}"
ENTROPY_COEFF_VALUE="${ENTROPY_COEFF_OVERRIDE:-0}"

ENABLE_DAPO_FILTER_GROUPS_VALUE="${ENABLE_DAPO_FILTER_GROUPS_OVERRIDE:-}"
DAPO_FILTER_GROUPS_METRIC_VALUE="${DAPO_FILTER_GROUPS_METRIC_OVERRIDE:-seq_final_reward}"
DAPO_MAX_NUM_GEN_BATCHES_VALUE="${DAPO_MAX_NUM_GEN_BATCHES_OVERRIDE:-0}"
NORM_ADV_BY_STD_IN_GRPO_VALUE="${NORM_ADV_BY_STD_IN_GRPO_OVERRIDE:-True}"
MASK_OVERLONG_LOSS_VALUE="${MASK_OVERLONG_LOSS_OVERRIDE:-True}"
CLIP_RATIO_LOW_VALUE="${CLIP_RATIO_LOW_OVERRIDE:-0.2}"
CLIP_RATIO_HIGH_VALUE="${CLIP_RATIO_HIGH_OVERRIDE:-0.3}"
CLIP_RATIO_C_VALUE="${CLIP_RATIO_C_OVERRIDE:-10.0}"
LOSS_AGG_MODE_VALUE="${LOSS_AGG_MODE_OVERRIDE:-token-mean}"

ENABLE_STRATIFIED_SAMPLER_VALUE="${ENABLE_STRATIFIED_SAMPLER_OVERRIDE:-False}"
STRATIFIED_SOURCE_LABELS_VALUE="${STRATIFIED_SOURCE_LABELS_OVERRIDE:-[hotpot,nq]}"
STRATIFIED_SOURCE_RATIOS_VALUE="${STRATIFIED_SOURCE_RATIOS_OVERRIDE:-[1,1]}"
ENABLE_DIFFICULTY_BUCKET_SAMPLER_VALUE="${ENABLE_DIFFICULTY_BUCKET_SAMPLER_OVERRIDE:-False}"
DIFFICULTY_BUCKET_FILE_VALUE="${DIFFICULTY_BUCKET_FILE_OVERRIDE:-}"
DIFFICULTY_BUCKET_LABELS_VALUE="${DIFFICULTY_BUCKET_LABELS_OVERRIDE:-[bucket_0,bucket_1,bucket_2,bucket_3,bucket_4]}"
DIFFICULTY_BUCKET_RATIOS_VALUE="${DIFFICULTY_BUCKET_RATIOS_OVERRIDE:-[24,16,12,8,4]}"

REWARD_ENABLE_VALUE="${REWARD_ENABLE_OVERRIDE:-True}"
REWARD_ANSWER_WEIGHT_VALUE="${REWARD_ANSWER_WEIGHT_OVERRIDE:-0.9}"
REWARD_FORMAT_WEIGHT_VALUE="${REWARD_FORMAT_WEIGHT_OVERRIDE:-0.1}"
REWARD_ENABLE_PROCESS_VALUE="${REWARD_ENABLE_PROCESS_OVERRIDE:-True}"
REWARD_ACTION_CORRECTNESS_WEIGHT_VALUE="${REWARD_ACTION_CORRECTNESS_WEIGHT_OVERRIDE:-0.1}"
REWARD_HALLUCINATED_ID_PENALTY_WEIGHT_VALUE="${REWARD_HALLUCINATED_ID_PENALTY_WEIGHT_OVERRIDE:-0.0}"
REWARD_TOOL_INVALID_PENALTY_WEIGHT_VALUE="${REWARD_TOOL_INVALID_PENALTY_WEIGHT_OVERRIDE:-0.0}"
if [ "$REWARD_ENABLE_VALUE" != "True" ]; then
  REWARD_ANSWER_WEIGHT_VALUE="0.0"
  REWARD_FORMAT_WEIGHT_VALUE="0.0"
  REWARD_ENABLE_PROCESS_VALUE="False"
  REWARD_ACTION_CORRECTNESS_WEIGHT_VALUE="0.0"
  REWARD_HALLUCINATED_ID_PENALTY_WEIGHT_VALUE="0.0"
  REWARD_TOOL_INVALID_PENALTY_WEIGHT_VALUE="0.0"
fi

AGENT_LOOP_NUM_WORKERS_VALUE="${AGENT_LOOP_NUM_WORKERS:-4}"
MAX_CONCURRENT_TRAJECTORIES_VALUE="${MAX_CONCURRENT_TRAJECTORIES:-64}"
TOOL_SERVER_MAX_CONCURRENT_REQUESTS_VALUE="${TOOL_SERVER_MAX_CONCURRENT_REQUESTS:-64}"
TEXT_BROWSER_MAX_ACTIVE_ACTORS_VALUE="${TEXT_BROWSER_MAX_ACTIVE_ACTORS_OVERRIDE:-64}"
TEXT_BROWSER_IDLE_POOL_SIZE_VALUE="${TEXT_BROWSER_IDLE_POOL_SIZE_OVERRIDE:-4}"
TEXT_BROWSER_ACTOR_CPUS_VALUE="${TEXT_BROWSER_ACTOR_CPUS_OVERRIDE:-1}"
PASS_EXTRA_FIELDS_TO_REWARD_VALUE="${PASS_EXTRA_FIELDS_TO_REWARD:-True}"
REWARD_FN_ASYNC_VALUE="${REWARD_FN_ASYNC:-False}"
COMPACT_TOOL_INTERACT_INFO_VALUE="${COMPACT_TOOL_INTERACT_INFO:-True}"
ROLLOUT_MAX_NUM_SEQS_VALUE="${ROLLOUT_MAX_NUM_SEQS_OVERRIDE:-64}"
ROLLOUT_MAX_BATCHED_TOKENS_VALUE="${ROLLOUT_MAX_BATCHED_TOKENS_OVERRIDE:-16384}"
LOG_PROB_MBS_VALUE="${LOG_PROB_MBS_OVERRIDE:-4}"
GPU_MEMORY_UTILIZATION_VALUE="${GPU_MEMORY_UTILIZATION_OVERRIDE:-}"
RUN_NAME_POSTFIX_VALUE="${RUN_NAME_POSTFIX:-}"
RUN_NAME_POSTFIX_PATH="$(sanitize_run_tag "$RUN_NAME_POSTFIX_VALUE")"

SAMPLER_MODE="default"
if [ "$ENABLE_DIFFICULTY_BUCKET_SAMPLER_VALUE" = "True" ]; then
  SAMPLER_MODE="difficulty_bucket"
elif [ "$ENABLE_STRATIFIED_SAMPLER_VALUE" = "True" ]; then
  SAMPLER_MODE="stratified_source"
fi

# ===== 指定实验列表 =====
# 非空时跳过网格搜索，按列表顺序只跑这些配置。
if ! declare -p EXPERIMENT_CONFIG_LIST >/dev/null 2>&1; then
  EXPERIMENT_CONFIG_LIST=(
    # "n6_b64_lr3e-7_t0.8_kl0.01_warmup10_turns10"
    "n6_b64_lr4e-7_t0.8_kl0.00001_warmup10_turns15"
    "n6_b64_lr5e-7_t0.8_kl0.00001_warmup10_turns15"
    "n6_b64_lr6e-7_t0.8_kl0.00001_warmup10_turns15"
  )
fi

if [ -n "${EXPERIMENT_CONFIGS:-}" ]; then
  EXPERIMENT_CONFIGS_NORMALIZED="${EXPERIMENT_CONFIGS//,/ }"
  # shellcheck disable=SC2206
  EXPERIMENT_CONFIG_LIST=($EXPERIMENT_CONFIGS_NORMALIZED)
fi

CONTINUE_ON_FAIL="${CONTINUE_ON_FAIL:-1}"
SLEEP_BETWEEN_RUNS="${SLEEP_BETWEEN_RUNS:-20}"
BASE_RAY_PORT="${BASE_RAY_PORT:-6380}"
BASE_RAY_TMP="${BASE_RAY_TMP:-$BASE_RAY_TMP_DEFAULT}"

FORCE_RERUN="${FORCE_RERUN:-0}"
SKIP_FAILED="${SKIP_FAILED:-0}"

# ===== Early Stop =====
ENABLE_EARLY_STOP="${ENABLE_EARLY_STOP:-0}"
METRIC_KEY="${METRIC_KEY:-critic/rewards/mean}"
# 使用最近多少个 step 的 reward_mean 滑动平均判断早停
EARLY_STOP_REWARD_WINDOW="${EARLY_STOP_REWARD_WINDOW:-8}"
# 当前滑动平均超过历史最佳滑动平均多少才算提升
EARLY_STOP_MIN_DELTA="${EARLY_STOP_MIN_DELTA:-0.01}"
# 连续多少个完整窗口没有达到 min_delta 提升后，才允许早停
EARLY_STOP_PATIENCE_WINDOWS="${EARLY_STOP_PATIENCE_WINDOWS:-5}"
# 只有策略更新幅度也很小时，才触发 reward 滑动平均未提升早停
EARLY_STOP_PPO_KL_KEY="${EARLY_STOP_PPO_KL_KEY:-actor/ppo_kl}"
EARLY_STOP_PPO_KL_MAX="${EARLY_STOP_PPO_KL_MAX:-0.001}"
EARLY_STOP_PG_CLIPFRAC_KEY="${EARLY_STOP_PG_CLIPFRAC_KEY:-actor/pg_clipfrac}"
EARLY_STOP_PG_CLIPFRAC_MAX="${EARLY_STOP_PG_CLIPFRAC_MAX:-0.01}"
# 每隔多久检查一次日志
EARLY_STOP_CHECK_INTERVAL="${EARLY_STOP_CHECK_INTERVAL:-120}"

SKIP_FILE="$GRID_LOG_ROOT/SKIP_CURRENT"
STOP_FILE="$GRID_LOG_ROOT/STOP_ALL"

if [ -n "$RL_ALG_VALUE" ]; then
  RL_ALG_EFFECTIVE="$RL_ALG_VALUE"
else
  case "$TRAIN_ALGO" in
    dapo) RL_ALG_EFFECTIVE="grpo" ;;
    grpo) RL_ALG_EFFECTIVE="grpo" ;;
    ppo) RL_ALG_EFFECTIVE="ppo" ;;
    *) RL_ALG_EFFECTIVE="mt_grpo" ;;
  esac
fi

if [ -n "$USE_KL_LOSS_VALUE" ]; then
  USE_KL_LOSS_EFFECTIVE="$USE_KL_LOSS_VALUE"
elif [ "$TRAIN_ALGO" = "dapo" ]; then
  USE_KL_LOSS_EFFECTIVE="False"
else
  USE_KL_LOSS_EFFECTIVE="True"
fi

if [ -n "$ENABLE_DAPO_FILTER_GROUPS_VALUE" ]; then
  ENABLE_DAPO_FILTER_GROUPS_EFFECTIVE="$ENABLE_DAPO_FILTER_GROUPS_VALUE"
elif [ "$TRAIN_ALGO" = "dapo" ]; then
  ENABLE_DAPO_FILTER_GROUPS_EFFECTIVE="True"
else
  ENABLE_DAPO_FILTER_GROUPS_EFFECTIVE="False"
fi

skip_current=0
stop_all=0
early_stopped=0
train_pid=""
WANDB_RUN_NAME_CURRENT=""

SUMMARY_HEADER="run_tag\tstatus\tbest_metric\texit_code\tn\tbatch\tlr\ttemp\tepoch\tkl_loss_coef\tlr_warmup_steps\tmax_turns\trun_name_postfix\trl_alg\tuse_kl_loss\tdapo_filter_groups\tnorm_adv_std\tclip_low\tclip_high\tloss_agg_mode\tsampler_mode\treward_enable\treward_answer\treward_format\treward_process\treward_action_correctness\treward_hallucinated_id\treward_tool_invalid\tpass_extra_fields\tcompact_tool_info\tagent_workers\tmax_concurrent_traj\trollout_max_num_seqs\trollout_max_batched_tokens\twandb_run\tinner_log\touter_log"

if [ ! -f "$SUMMARY_FILE" ]; then
  echo -e "$SUMMARY_HEADER" > "$SUMMARY_FILE"
elif ! head -n 1 "$SUMMARY_FILE" | grep -q "run_name_postfix"; then
  mv "$SUMMARY_FILE" "${SUMMARY_FILE}.bak.$(date +%F_%H%M%S)"
  echo -e "$SUMMARY_HEADER" > "$SUMMARY_FILE"
fi

kill_train_group() {
  if [ -n "${train_pid:-}" ] && kill -0 "$train_pid" 2>/dev/null; then
    echo "[INTERRUPT] killing train process group pid=$train_pid"
    kill -TERM "-$train_pid" 2>/dev/null || true
    sleep 5
    kill -KILL "-$train_pid" 2>/dev/null || true
  fi
}

cleanup_after_run() {
  echo "[CLEANUP] stopping training/tool/ray processes..."

  ray stop --force >/dev/null 2>&1 || true

  pkill -u "$(id -u)" -f "verl_tool.servers.serve" || true
  pkill -u "$(id -u)" -f "verl_tool.trainer.main_ppo" || true
  pkill -u "$(id -u)" -f "main_ppo.py" || true
  pkill -u "$(id -u)" -f "$TRAIN_SCRIPT" || true

  pkill -u "$(id -u)" -f "raylet" || true
  pkill -u "$(id -u)" -f "plasma_store" || true
  pkill -u "$(id -u)" -f "gcs_server" || true
  pkill -u "$(id -u)" -f "dashboard" || true
  pkill -u "$(id -u)" -f "runtime_env_agent" || true
  pkill -u "$(id -u)" -f "log_monitor" || true
  pkill -u "$(id -u)" -f "monitor.py" || true

  sleep 3

  pkill -9 -u "$(id -u)" -f "raylet" || true
  pkill -9 -u "$(id -u)" -f "plasma_store" || true
  pkill -9 -u "$(id -u)" -f "gcs_server" || true
  pkill -9 -u "$(id -u)" -f "dashboard" || true
  pkill -9 -u "$(id -u)" -f "runtime_env_agent" || true
  pkill -9 -u "$(id -u)" -f "log_monitor" || true
  pkill -9 -u "$(id -u)" -f "monitor.py" || true

  if [ -n "${RAY_TMPDIR_OVERRIDE:-}" ] && [ -d "$RAY_TMPDIR_OVERRIDE" ]; then
    rm -rf "${RAY_TMPDIR_OVERRIDE:?}/ray" || true
    rm -rf "${RAY_TMPDIR_OVERRIDE:?}/session_"* || true
  fi

  echo "[CLEANUP] done."
}

handle_int() {
  if [ "$skip_current" -eq 0 ]; then
    echo ""
    echo "⚠️  Ctrl+C received: skip current run and continue next..."
    skip_current=1
    kill_train_group
  else
    echo ""
    echo "🛑 Ctrl+C received again: stop all grid search..."
    stop_all=1
    skip_current=1
    kill_train_group
  fi
}

trap handle_int INT

extract_latest_step_metric() {
  local log_file="$1"

  if [ ! -f "$log_file" ]; then
    echo "NA NA"
    return
  fi

  awk -v key="$METRIC_KEY" '
    $0 ~ "training/global_step:" && index($0, key ":") > 0 {
      step="NA"; metric="NA";

      if (match($0, /training\/global_step:[0-9]+/)) {
        s=substr($0, RSTART, RLENGTH);
        sub("training/global_step:", "", s);
        step=s;
      }

      pos=index($0, key ":");
      if (pos > 0) {
        rest=substr($0, pos + length(key) + 1);

        # value 到下一个空格或 " - " 前为止
        split(rest, arr, " ");
        metric=arr[1];

        gsub(/np.float64\(/, "", metric);
        gsub(/np.int32\(/, "", metric);
        gsub(/\)/, "", metric);
      }

      latest_step=step;
      latest_metric=metric;
    }
    END {
      if (latest_step == "" || latest_metric == "") print "NA NA";
      else print latest_step, latest_metric;
    }
  ' "$log_file"
}

extract_sliding_metrics_after() {
  local log_file="$1"
  local last_seen_step="${2:-0}"

  if [ ! -f "$log_file" ]; then
    return
  fi

  awk -v reward_key="$METRIC_KEY" \
      -v ppo_kl_key="$EARLY_STOP_PPO_KL_KEY" \
      -v pg_clipfrac_key="$EARLY_STOP_PG_CLIPFRAC_KEY" \
      -v window="$EARLY_STOP_REWARD_WINDOW" \
      -v min_delta="$EARLY_STOP_MIN_DELTA" \
      -v last_seen_step="$last_seen_step" '
    function clean_metric(value) {
      gsub(/np.float64\(/, "", value);
      gsub(/np.int32\(/, "", value);
      gsub(/\)/, "", value);
      return value;
    }

    function is_numeric(value) {
      return value ~ /^[-+]?([0-9]+([.][0-9]*)?|[.][0-9]+)([eE][-+]?[0-9]+)?$/;
    }

    function metric_value(key, pos, rest, arr, value) {
      pos=index($0, key ":");
      if (pos <= 0) return "NA";

      rest=substr($0, pos + length(key) + 1);
      split(rest, arr, " ");
      value=arr[1];
      return clean_metric(value);
    }

    $0 ~ "training/global_step:" && index($0, reward_key ":") > 0 {
      step="NA";

      if (match($0, /training\/global_step:[0-9]+/)) {
        s=substr($0, RSTART, RLENGTH);
        sub("training/global_step:", "", s);
        step=s;
      }

      reward=metric_value(reward_key);
      if (!is_numeric(reward)) next;

      metric_count += 1;
      window_index=((metric_count - 1) % window) + 1;

      if (metric_count > window) rolling_sum -= reward_window[window_index];
      reward_window[window_index]=reward + 0;
      rolling_sum += reward + 0;

      latest_window_step=step;
      latest_reward=reward + 0;
      latest_ppo_kl=metric_value(ppo_kl_key);
      latest_pg_clipfrac=metric_value(pg_clipfrac_key);
      latest_window_avg="NA";
      printable_best_step="NA";
      printable_best_avg="NA";
      printable_stale_windows="NA";

      if (metric_count >= window) {
        full_window_count += 1;
        latest_window_avg=rolling_sum / window;

        if (best_window_avg == "" || latest_window_avg > best_window_avg + min_delta) {
          best_window_avg=latest_window_avg;
          best_window_step=latest_window_step;
          stale_windows=0;
        } else {
          stale_windows += 1;
        }

        printable_best_step=best_window_step;
        printable_best_avg=best_window_avg;
        printable_stale_windows=stale_windows;
      }

      if (is_numeric(step) && step + 0 > last_seen_step + 0) {
        print latest_window_step, latest_reward, latest_window_avg, latest_ppo_kl, latest_pg_clipfrac, metric_count, printable_best_step, printable_best_avg, printable_stale_windows, full_window_count + 0;
      } else {
        next;
      }
    }
  ' "$log_file"
}

is_number_less_than() {
  local value="$1"
  local threshold="$2"

  awk -v value="$value" -v threshold="$threshold" '
    BEGIN {
      numeric = "^[-+]?([0-9]+([.][0-9]*)?|[.][0-9]+)([eE][-+]?[0-9]+)?$";
      print (value ~ numeric && value + 0 < threshold + 0) ? 1 : 0;
    }
  '
}

format_fixed() {
  local value="$1"
  local digits="${2:-4}"

  awk -v value="$value" -v digits="$digits" '
    BEGIN {
      numeric = "^[-+]?([0-9]+([.][0-9]*)?|[.][0-9]+)([eE][-+]?[0-9]+)?$";
      if (value ~ numeric) printf "%.*f", digits, value + 0;
      else printf "%s", value;
    }
  '
}

format_scientific() {
  local value="$1"

  awk -v value="$value" '
    BEGIN {
      numeric = "^[-+]?([0-9]+([.][0-9]*)?|[.][0-9]+)([eE][-+]?[0-9]+)?$";
      if (value ~ numeric) printf "%.2e", value + 0;
      else printf "%s", value;
    }
  '
}

extract_best_metric() {
  local log_file="$1"

  if [ ! -f "$log_file" ]; then
    echo "NA"
    return
  fi

  awk -v key="$METRIC_KEY" '
    $0 ~ "training/global_step:" && index($0, key ":") > 0 {
      pos=index($0, key ":");
      rest=substr($0, pos + length(key) + 1);
      split(rest, arr, " ");
      metric=arr[1];

      gsub(/np.float64\(/, "", metric);
      gsub(/np.int32\(/, "", metric);
      gsub(/\)/, "", metric);

      if (metric != "nan" && metric != "NA" && (best == "" || metric > best)) {
        best=metric;
      }
    }
    END {
      if (best == "") print "NA";
      else print best;
    }
  ' "$log_file"
}

append_summary() {
  local run_tag="$1"
  local status="$2"
  local best_metric="$3"
  local exit_code="$4"
  local n="$5"
  local batch="$6"
  local lr="$7"
  local temp="$8"
  local epoch="$9"
  local kl_loss_coef="${10}"
  local lr_warmup_steps="${11}"
  local max_turns="${12}"
  local inner_log="${13}"
  local outer_log="${14}"

  echo -e "${run_tag}\t${status}\t${best_metric}\t${exit_code}\t${n}\t${batch}\t${lr}\t${temp}\t${epoch}\t${kl_loss_coef}\t${lr_warmup_steps}\t${max_turns}\t${RUN_NAME_POSTFIX_VALUE}\t${RL_ALG_EFFECTIVE}\t${USE_KL_LOSS_EFFECTIVE}\t${ENABLE_DAPO_FILTER_GROUPS_EFFECTIVE}\t${NORM_ADV_BY_STD_IN_GRPO_VALUE}\t${CLIP_RATIO_LOW_VALUE}\t${CLIP_RATIO_HIGH_VALUE}\t${LOSS_AGG_MODE_VALUE}\t${SAMPLER_MODE}\t${REWARD_ENABLE_VALUE}\t${REWARD_ANSWER_WEIGHT_VALUE}\t${REWARD_FORMAT_WEIGHT_VALUE}\t${REWARD_ENABLE_PROCESS_VALUE}\t${REWARD_ACTION_CORRECTNESS_WEIGHT_VALUE}\t${REWARD_HALLUCINATED_ID_PENALTY_WEIGHT_VALUE}\t${REWARD_TOOL_INVALID_PENALTY_WEIGHT_VALUE}\t${PASS_EXTRA_FIELDS_TO_REWARD_VALUE}\t${COMPACT_TOOL_INTERACT_INFO_VALUE}\t${AGENT_LOOP_NUM_WORKERS_VALUE}\t${MAX_CONCURRENT_TRAJECTORIES_VALUE}\t${ROLLOUT_MAX_NUM_SEQS_VALUE}\t${ROLLOUT_MAX_BATCHED_TOKENS_VALUE}\t${WANDB_RUN_NAME_CURRENT}\t${inner_log}\t${outer_log}" >> "$SUMMARY_FILE"
}

parse_experiment_config() {
  local config="$1"
  local n=""
  local batch=""
  local lr=""
  local temp=""
  local epoch="${EPOCH_LIST[0]}"
  local kl_loss_coef=""
  local lr_warmup_steps=""
  local max_turns=""
  local token=""
  local parts=()
  local missing=()

  IFS='_' read -r -a parts <<< "$config"
  for token in "${parts[@]}"; do
    case "$token" in
      warmup*) lr_warmup_steps="${token#warmup}" ;;
      maxturns*) max_turns="${token#maxturns}" ;;
      turns*) max_turns="${token#turns}" ;;
      mt*) max_turns="${token#mt}" ;;
      kl*) kl_loss_coef="${token#kl}" ;;
      lr*) lr="${token#lr}" ;;
      n*) n="${token#n}" ;;
      b*) batch="${token#b}" ;;
      t*) temp="${token#t}" ;;
      e*) epoch="${token#e}" ;;
      *)
        echo "[CONFIG ERROR] unknown token '$token' in '$config'" >&2
        return 1
        ;;
    esac
  done

  [ -n "$n" ] || missing+=("n")
  [ -n "$batch" ] || missing+=("b")
  [ -n "$lr" ] || missing+=("lr")
  [ -n "$temp" ] || missing+=("t")
  [ -n "$kl_loss_coef" ] || missing+=("kl")
  [ -n "$lr_warmup_steps" ] || missing+=("warmup")

  if [ "${#missing[@]}" -gt 0 ]; then
    echo "[CONFIG ERROR] '$config' missing required fields: ${missing[*]}" >&2
    return 1
  fi

  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "$n" "$batch" "$lr" "$temp" "$epoch" "$kl_loss_coef" "$lr_warmup_steps" "$max_turns"
}

CONFIG_MODE="grid"
CONFIG_ROWS=()
GRID_MAX_TURNS_VALUES=("")

if [ "${#MAX_TURNS_LIST[@]}" -gt 0 ]; then
  GRID_MAX_TURNS_VALUES=("${MAX_TURNS_LIST[@]}")
fi

if [ "${#EXPERIMENT_CONFIG_LIST[@]}" -gt 0 ]; then
  CONFIG_MODE="explicit"
  for config in "${EXPERIMENT_CONFIG_LIST[@]}"; do
    parsed_config="$(parse_experiment_config "$config")"
    CONFIG_ROWS+=("$config"$'\t'"$parsed_config")
  done
else
  for n in "${N_LIST[@]}"; do
    for batch in "${BATCH_LIST[@]}"; do
      for lr in "${LR_LIST[@]}"; do
        for temp in "${TEMP_LIST[@]}"; do
          for epoch in "${EPOCH_LIST[@]}"; do
            for kl_loss_coef in "${KL_LOSS_COEF_LIST[@]}"; do
              for lr_warmup_steps in "${LR_WARMUP_STEPS_LIST[@]}"; do
                for max_turns in "${GRID_MAX_TURNS_VALUES[@]}"; do
                  CONFIG_ROWS+=("grid"$'\t'"$n"$'\t'"$batch"$'\t'"$lr"$'\t'"$temp"$'\t'"$epoch"$'\t'"$kl_loss_coef"$'\t'"$lr_warmup_steps"$'\t'"$max_turns")
                done
              done
            done
          done
        done
      done
    done
  done
fi

run_idx=0

rm -f "$SKIP_FILE" "$STOP_FILE"

if [ "$CONFIG_MODE" = "explicit" ]; then
  echo "========== Explicit Experiments Start =========="
else
  echo "========== Grid Search Start =========="
fi
echo "TRAIN_SCRIPT=$TRAIN_SCRIPT"
echo "TRAIN_ALGO=$TRAIN_ALGO"
echo "SFT_MODEL_NAME=$SFT_MODEL_NAME"
echo "BENCHMARK=$BENCHMARK"
echo "VAL_DATASET_NAME=$VAL_DATASET_NAME"
echo "LOG_CONTEXT_DIR=$LOG_CONTEXT_DIR"
echo "LOG_ROOT=$LOG_ROOT"
echo "CONFIG_MODE=$CONFIG_MODE"
echo "CONFIG_COUNT=${#CONFIG_ROWS[@]}"
echo "GRID_LOG_ROOT=$GRID_LOG_ROOT"
echo "TRAIN_LOG_ROOT=$TRAIN_LOG_ROOT"
echo "SUMMARY_FILE=$SUMMARY_FILE"
echo "FORCE_RERUN=$FORCE_RERUN"
echo "SKIP_FAILED=$SKIP_FAILED"
echo "RUN_NAME_POSTFIX=$RUN_NAME_POSTFIX_VALUE"
echo "RL_ALG=$RL_ALG_EFFECTIVE"
echo "USE_KL_LOSS=$USE_KL_LOSS_EFFECTIVE"
echo "DAPO_FILTER_GROUPS=$ENABLE_DAPO_FILTER_GROUPS_EFFECTIVE metric=$DAPO_FILTER_GROUPS_METRIC_VALUE max_gen_batches=$DAPO_MAX_NUM_GEN_BATCHES_VALUE"
echo "DAPO_CLIP low=$CLIP_RATIO_LOW_VALUE high=$CLIP_RATIO_HIGH_VALUE c=$CLIP_RATIO_C_VALUE loss_agg=$LOSS_AGG_MODE_VALUE norm_adv_std=$NORM_ADV_BY_STD_IN_GRPO_VALUE mask_overlong=$MASK_OVERLONG_LOSS_VALUE"
echo "SAMPLER_MODE=$SAMPLER_MODE stratified_labels=$STRATIFIED_SOURCE_LABELS_VALUE stratified_ratios=$STRATIFIED_SOURCE_RATIOS_VALUE difficulty_file=${DIFFICULTY_BUCKET_FILE_VALUE:-NA}"
echo "REWARD enable=$REWARD_ENABLE_VALUE answer=$REWARD_ANSWER_WEIGHT_VALUE format=$REWARD_FORMAT_WEIGHT_VALUE process=$REWARD_ENABLE_PROCESS_VALUE action=$REWARD_ACTION_CORRECTNESS_WEIGHT_VALUE hallucinated=$REWARD_HALLUCINATED_ID_PENALTY_WEIGHT_VALUE tool_invalid=$REWARD_TOOL_INVALID_PENALTY_WEIGHT_VALUE"
echo "CONCURRENCY agent_workers=$AGENT_LOOP_NUM_WORKERS_VALUE max_traj=$MAX_CONCURRENT_TRAJECTORIES_VALUE tool_requests=$TOOL_SERVER_MAX_CONCURRENT_REQUESTS_VALUE browser_actors=$TEXT_BROWSER_MAX_ACTIVE_ACTORS_VALUE"
echo "ENABLE_EARLY_STOP=$ENABLE_EARLY_STOP"
echo "EARLY_STOP_METRIC=$METRIC_KEY"
echo "EARLY_STOP_REWARD_WINDOW=$EARLY_STOP_REWARD_WINDOW"
echo "EARLY_STOP_MIN_DELTA=$EARLY_STOP_MIN_DELTA"
echo "EARLY_STOP_PATIENCE_WINDOWS=$EARLY_STOP_PATIENCE_WINDOWS"
echo "EARLY_STOP_PPO_KL=${EARLY_STOP_PPO_KL_KEY}<${EARLY_STOP_PPO_KL_MAX}"
echo "EARLY_STOP_PG_CLIPFRAC=${EARLY_STOP_PG_CLIPFRAC_KEY}<${EARLY_STOP_PG_CLIPFRAC_MAX}"
echo ""
echo "Control:"
echo "  Ctrl+C once  -> skip current run and continue"
echo "  Ctrl+C twice -> stop all"
echo "  touch $SKIP_FILE -> skip current run"
echo "  touch $STOP_FILE -> stop all"
echo "======================================="

for config_row in "${CONFIG_ROWS[@]}"; do
  IFS=$'\t' read -r config_name n batch lr temp epoch kl_loss_coef lr_warmup_steps max_turns <<< "$config_row"

  if [ "$stop_all" -eq 1 ] || [ -f "$STOP_FILE" ]; then
    echo "$CONFIG_MODE stopped by user."
    cleanup_after_run
    exit 130
  fi

  run_idx=$((run_idx + 1))

  if [ "$CONFIG_MODE" = "explicit" ]; then
    RUN_TAG_BASE="exp_${run_idx}_$(sanitize_run_tag "$config_name")"
  else
    RUN_TAG_BASE="grid_${run_idx}_n${n}_b${batch}_lr${lr}_t${temp}_e${epoch}_kl${kl_loss_coef}_warmup${lr_warmup_steps}"
    if [ -n "$max_turns" ]; then
      RUN_TAG_BASE="${RUN_TAG_BASE}_turns${max_turns}"
    fi
  fi
  RUN_TAG="${RUN_TAG_BASE}${RUN_NAME_POSTFIX_PATH}"

  OUTER_LOG="$GRID_LOG_ROOT/${RUN_TAG}.outer.log"
  INNER_LOG="$TRAIN_LOG_ROOT/${RUN_TAG}.log"
  AGENT_LOOP_LOG_DIR="$LOG_ROOT/verltool_agent_loop/$LOG_CONTEXT_DIR/$RUN_TAG"

  SUCCESS_MARK="$GRID_LOG_ROOT/${RUN_TAG}.success"
  FAILED_MARK="$GRID_LOG_ROOT/${RUN_TAG}.failed"
  RUNNING_MARK="$GRID_LOG_ROOT/${RUN_TAG}.running"

  if [ "$FORCE_RERUN" != "1" ]; then
    if [ -f "$SUCCESS_MARK" ]; then
      echo "[SKIP SUCCESS] $RUN_TAG"
      continue
    fi

    if [ "$SKIP_FAILED" = "1" ] && [ -f "$FAILED_MARK" ]; then
      echo "[SKIP FAILED] $RUN_TAG"
      continue
    fi
  else
    rm -f "$SUCCESS_MARK" "$FAILED_MARK" "$RUNNING_MARK"
  fi

  export RUN_TAG="$RUN_TAG"
  export RUN_NAME_POSTFIX="$RUN_NAME_POSTFIX_VALUE"
  export RUN_LOG_TAG_OVERRIDE="$RUN_TAG"
  export TRAIN_LOG_NAME_OVERRIDE="$RUN_TAG"
  if [ "$TRAIN_ALGO" = "dapo" ]; then
    export TRAIN_PRESET=dapo
  else
    export TRAIN_PRESET=mt_grpo
  fi
  WANDB_RUN_NAME_CURRENT="${TRAIN_ALGO}-${MODEL_LOG_COMPONENT}-${DATA_LOG_COMPONENT}-${RUN_TAG}"
  export WANDB_RUN_NAME_OVERRIDE="$WANDB_RUN_NAME_CURRENT"
  export VERLTOOL_AGENT_LOOP_LOG_DIR="$AGENT_LOOP_LOG_DIR"
  mkdir -p "$VERLTOOL_AGENT_LOOP_LOG_DIR"

  export RL_ALG_OVERRIDE="$RL_ALG_EFFECTIVE"
  if [ -n "$PPO_MINI_BATCH_SIZE_VALUE" ]; then
    export PPO_MINI_BATCH_SIZE_OVERRIDE="$PPO_MINI_BATCH_SIZE_VALUE"
  else
    unset PPO_MINI_BATCH_SIZE_OVERRIDE
  fi
  export USE_KL_LOSS_OVERRIDE="$USE_KL_LOSS_EFFECTIVE"
  export KL_COEF_OVERRIDE="$KL_COEF_VALUE"
  export ENTROPY_COEFF_OVERRIDE="$ENTROPY_COEFF_VALUE"
  export ENABLE_DAPO_FILTER_GROUPS_OVERRIDE="$ENABLE_DAPO_FILTER_GROUPS_EFFECTIVE"
  export DAPO_FILTER_GROUPS_METRIC_OVERRIDE="$DAPO_FILTER_GROUPS_METRIC_VALUE"
  export DAPO_MAX_NUM_GEN_BATCHES_OVERRIDE="$DAPO_MAX_NUM_GEN_BATCHES_VALUE"
  export NORM_ADV_BY_STD_IN_GRPO_OVERRIDE="$NORM_ADV_BY_STD_IN_GRPO_VALUE"
  export MASK_OVERLONG_LOSS_OVERRIDE="$MASK_OVERLONG_LOSS_VALUE"
  export CLIP_RATIO_LOW_OVERRIDE="$CLIP_RATIO_LOW_VALUE"
  export CLIP_RATIO_HIGH_OVERRIDE="$CLIP_RATIO_HIGH_VALUE"
  export CLIP_RATIO_C_OVERRIDE="$CLIP_RATIO_C_VALUE"
  export LOSS_AGG_MODE_OVERRIDE="$LOSS_AGG_MODE_VALUE"

  export ENABLE_STRATIFIED_SAMPLER_OVERRIDE="$ENABLE_STRATIFIED_SAMPLER_VALUE"
  export STRATIFIED_SOURCE_LABELS_OVERRIDE="$STRATIFIED_SOURCE_LABELS_VALUE"
  export STRATIFIED_SOURCE_RATIOS_OVERRIDE="$STRATIFIED_SOURCE_RATIOS_VALUE"
  export ENABLE_DIFFICULTY_BUCKET_SAMPLER_OVERRIDE="$ENABLE_DIFFICULTY_BUCKET_SAMPLER_VALUE"
  export DIFFICULTY_BUCKET_FILE_OVERRIDE="$DIFFICULTY_BUCKET_FILE_VALUE"
  export DIFFICULTY_BUCKET_LABELS_OVERRIDE="$DIFFICULTY_BUCKET_LABELS_VALUE"
  export DIFFICULTY_BUCKET_RATIOS_OVERRIDE="$DIFFICULTY_BUCKET_RATIOS_VALUE"

  export REWARD_ENABLE_OVERRIDE="$REWARD_ENABLE_VALUE"
  export REWARD_ANSWER_WEIGHT_OVERRIDE="$REWARD_ANSWER_WEIGHT_VALUE"
  export REWARD_FORMAT_WEIGHT_OVERRIDE="$REWARD_FORMAT_WEIGHT_VALUE"
  export REWARD_ENABLE_PROCESS_OVERRIDE="$REWARD_ENABLE_PROCESS_VALUE"
  export REWARD_ACTION_CORRECTNESS_WEIGHT_OVERRIDE="$REWARD_ACTION_CORRECTNESS_WEIGHT_VALUE"
  export REWARD_HALLUCINATED_ID_PENALTY_WEIGHT_OVERRIDE="$REWARD_HALLUCINATED_ID_PENALTY_WEIGHT_VALUE"
  export REWARD_TOOL_INVALID_PENALTY_WEIGHT_OVERRIDE="$REWARD_TOOL_INVALID_PENALTY_WEIGHT_VALUE"

  export AGENT_LOOP_NUM_WORKERS="$AGENT_LOOP_NUM_WORKERS_VALUE"
  export MAX_CONCURRENT_TRAJECTORIES="$MAX_CONCURRENT_TRAJECTORIES_VALUE"
  export TOOL_SERVER_MAX_CONCURRENT_REQUESTS="$TOOL_SERVER_MAX_CONCURRENT_REQUESTS_VALUE"
  export TEXT_BROWSER_MAX_ACTIVE_ACTORS_OVERRIDE="$TEXT_BROWSER_MAX_ACTIVE_ACTORS_VALUE"
  export TEXT_BROWSER_IDLE_POOL_SIZE_OVERRIDE="$TEXT_BROWSER_IDLE_POOL_SIZE_VALUE"
  export TEXT_BROWSER_ACTOR_CPUS_OVERRIDE="$TEXT_BROWSER_ACTOR_CPUS_VALUE"
  export PASS_EXTRA_FIELDS_TO_REWARD="$PASS_EXTRA_FIELDS_TO_REWARD_VALUE"
  export REWARD_FN_ASYNC="$REWARD_FN_ASYNC_VALUE"
  export COMPACT_TOOL_INTERACT_INFO="$COMPACT_TOOL_INTERACT_INFO_VALUE"
  export ROLLOUT_MAX_NUM_SEQS_OVERRIDE="$ROLLOUT_MAX_NUM_SEQS_VALUE"
  export ROLLOUT_MAX_BATCHED_TOKENS_OVERRIDE="$ROLLOUT_MAX_BATCHED_TOKENS_VALUE"
  export LOG_PROB_MBS_OVERRIDE="$LOG_PROB_MBS_VALUE"
  if [ -n "$GPU_MEMORY_UTILIZATION_VALUE" ]; then
    export GPU_MEMORY_UTILIZATION_OVERRIDE="$GPU_MEMORY_UTILIZATION_VALUE"
  fi

  export ROLLOUT_N_OVERRIDE="$n"
  export TRAIN_BATCH_SIZE_OVERRIDE="$batch"
  export LR_OVERRIDE="$lr"
  export TEMPERATURE_OVERRIDE="$temp"
  export EPOCH_OVERRIDE="$epoch"
  export KL_LOSS_COEF_OVERRIDE="$kl_loss_coef"
  export LR_WARMUP_STEPS_OVERRIDE="$lr_warmup_steps"
  if [ -n "$max_turns" ]; then
    export MAX_TURNS_OVERRIDE="$max_turns"
  else
    unset MAX_TURNS_OVERRIDE
  fi

  export RAY_PORT_OVERRIDE=$((BASE_RAY_PORT + run_idx))
  export RAY_TMPDIR_OVERRIDE="${BASE_RAY_TMP}/r${run_idx}"
  mkdir -p "$RAY_TMPDIR_OVERRIDE"

  rm -f "$SUCCESS_MARK" "$FAILED_MARK" "$SKIP_FILE"
  echo "started_at=$(date '+%F %T')" > "$RUNNING_MARK"

  echo "" | tee -a "$OUTER_LOG"
  echo "========== Run $run_idx/${#CONFIG_ROWS[@]} ==========" | tee -a "$OUTER_LOG"
  echo "RUN_TAG=$RUN_TAG" | tee -a "$OUTER_LOG"
  echo "config_name=$config_name" | tee -a "$OUTER_LOG"
  echo "run_name_postfix=$RUN_NAME_POSTFIX_VALUE" | tee -a "$OUTER_LOG"
  echo "n=$n batch=$batch lr=$lr temp=$temp epoch=$epoch kl_loss_coef=$kl_loss_coef lr_warmup_steps=$lr_warmup_steps max_turns=${max_turns:-default}" | tee -a "$OUTER_LOG"
  echo "rl_alg=$RL_ALG_EFFECTIVE use_kl_loss=$USE_KL_LOSS_EFFECTIVE kl_coef=$KL_COEF_VALUE entropy_coeff=$ENTROPY_COEFF_VALUE" | tee -a "$OUTER_LOG"
  echo "dapo_filter_groups=$ENABLE_DAPO_FILTER_GROUPS_EFFECTIVE metric=$DAPO_FILTER_GROUPS_METRIC_VALUE max_gen_batches=$DAPO_MAX_NUM_GEN_BATCHES_VALUE norm_adv_std=$NORM_ADV_BY_STD_IN_GRPO_VALUE mask_overlong=$MASK_OVERLONG_LOSS_VALUE" | tee -a "$OUTER_LOG"
  echo "clip_low=$CLIP_RATIO_LOW_VALUE clip_high=$CLIP_RATIO_HIGH_VALUE clip_c=$CLIP_RATIO_C_VALUE loss_agg_mode=$LOSS_AGG_MODE_VALUE" | tee -a "$OUTER_LOG"
  echo "sampler_mode=$SAMPLER_MODE stratified_labels=$STRATIFIED_SOURCE_LABELS_VALUE stratified_ratios=$STRATIFIED_SOURCE_RATIOS_VALUE difficulty_file=${DIFFICULTY_BUCKET_FILE_VALUE:-NA}" | tee -a "$OUTER_LOG"
  echo "reward_enable=$REWARD_ENABLE_VALUE reward_answer=$REWARD_ANSWER_WEIGHT_VALUE reward_format=$REWARD_FORMAT_WEIGHT_VALUE reward_process=$REWARD_ENABLE_PROCESS_VALUE reward_action=$REWARD_ACTION_CORRECTNESS_WEIGHT_VALUE reward_hallucinated=$REWARD_HALLUCINATED_ID_PENALTY_WEIGHT_VALUE reward_tool_invalid=$REWARD_TOOL_INVALID_PENALTY_WEIGHT_VALUE" | tee -a "$OUTER_LOG"
  echo "agent_workers=$AGENT_LOOP_NUM_WORKERS_VALUE max_concurrent_trajectories=$MAX_CONCURRENT_TRAJECTORIES_VALUE tool_server_max_requests=$TOOL_SERVER_MAX_CONCURRENT_REQUESTS_VALUE compact_tool_interact_info=$COMPACT_TOOL_INTERACT_INFO_VALUE pass_extra_fields_to_reward=$PASS_EXTRA_FIELDS_TO_REWARD_VALUE" | tee -a "$OUTER_LOG"
  echo "RAY_PORT_OVERRIDE=$RAY_PORT_OVERRIDE" | tee -a "$OUTER_LOG"
  echo "RAY_TMPDIR_OVERRIDE=$RAY_TMPDIR_OVERRIDE" | tee -a "$OUTER_LOG"
  echo "WANDB_RUN_NAME_OVERRIDE=$WANDB_RUN_NAME_OVERRIDE" | tee -a "$OUTER_LOG"
  echo "INNER_LOG=$INNER_LOG" | tee -a "$OUTER_LOG"
  echo "VERLTOOL_AGENT_LOOP_LOG_DIR=$VERLTOOL_AGENT_LOOP_LOG_DIR" | tee -a "$OUTER_LOG"
  echo "start_time=$(date '+%F %T')" | tee -a "$OUTER_LOG"

  skip_current=0
  early_stopped=0
  train_pid=""

  set +e

  setsid bash "$TRAIN_SCRIPT" >> "$OUTER_LOG" 2>&1 &
  train_pid=$!

  echo "train_pid=$train_pid" | tee -a "$OUTER_LOG"

  last_seen_step=0
  missing_metrics_logged=0
  last_check_ts=$(date +%s)

  while kill -0 "$train_pid" 2>/dev/null; do
    now_ts=$(date +%s)

    if [ -f "$STOP_FILE" ]; then
      echo "[CONTROL] STOP_ALL file detected." | tee -a "$OUTER_LOG"
      stop_all=1
      skip_current=1
      kill_train_group
      break
    fi

    if [ -f "$SKIP_FILE" ]; then
      echo "[CONTROL] SKIP_CURRENT file detected." | tee -a "$OUTER_LOG"
      skip_current=1
      kill_train_group
      break
    fi

    if [ "$ENABLE_EARLY_STOP" = "1" ] && [ $((now_ts - last_check_ts)) -ge "$EARLY_STOP_CHECK_INTERVAL" ]; then
      saw_metric=0

      while read -r latest_step latest_reward latest_metric latest_ppo_kl latest_pg_clipfrac reward_metric_count history_best_step history_best_metric stale_windows full_window_count; do
        saw_metric=1
        last_seen_step="$latest_step"
        missing_metrics_logged=0

        latest_reward_fmt=$(format_fixed "$latest_reward" 4)
        latest_ppo_kl_fmt=$(format_scientific "$latest_ppo_kl")
        latest_pg_clipfrac_fmt=$(format_fixed "$latest_pg_clipfrac" 4)

        if [ "$latest_ppo_kl" = "NA" ] || [ "$latest_pg_clipfrac" = "NA" ]; then
          echo "[EARLY_STOP] step=$latest_step reward=$latest_reward_fmt window=$reward_metric_count/$EARLY_STOP_REWARD_WINDOW ppo_kl=$latest_ppo_kl_fmt pg_clipfrac=$latest_pg_clipfrac_fmt action=wait_policy_metrics" | tee -a "$OUTER_LOG"
        elif [ "$latest_metric" = "NA" ]; then
          echo "[EARLY_STOP] step=$latest_step reward=$latest_reward_fmt window=$reward_metric_count/$EARLY_STOP_REWARD_WINDOW ppo_kl=$latest_ppo_kl_fmt pg_clipfrac=$latest_pg_clipfrac_fmt action=wait_window" | tee -a "$OUTER_LOG"
        else
          low_ppo_kl=$(is_number_less_than "$latest_ppo_kl" "$EARLY_STOP_PPO_KL_MAX")
          low_pg_clipfrac=$(is_number_less_than "$latest_pg_clipfrac" "$EARLY_STOP_PG_CLIPFRAC_MAX")
          latest_metric_fmt=$(format_fixed "$latest_metric" 4)
          history_best_metric_fmt=$(format_fixed "$history_best_metric" 4)
          min_delta_fmt=$(format_fixed "$EARLY_STOP_MIN_DELTA" 4)

          if [ "$full_window_count" -le 1 ]; then
            echo "[EARLY_STOP] step=$latest_step reward=$latest_reward_fmt window_avg=$latest_metric_fmt best@${history_best_step}=$history_best_metric_fmt delta=$min_delta_fmt patience=${stale_windows}/$EARLY_STOP_PATIENCE_WINDOWS ppo_kl=$latest_ppo_kl_fmt pg_clipfrac=$latest_pg_clipfrac_fmt action=init" | tee -a "$OUTER_LOG"
          elif [ "$stale_windows" = "0" ]; then
            echo "[EARLY_STOP] step=$latest_step reward=$latest_reward_fmt window_avg=$latest_metric_fmt best@${history_best_step}=$history_best_metric_fmt delta=$min_delta_fmt patience=${stale_windows}/$EARLY_STOP_PATIENCE_WINDOWS improved=yes ppo_kl=$latest_ppo_kl_fmt pg_clipfrac=$latest_pg_clipfrac_fmt action=continue" | tee -a "$OUTER_LOG"
          elif [ "$stale_windows" -ge "$EARLY_STOP_PATIENCE_WINDOWS" ] && [ "$low_ppo_kl" -eq 1 ] && [ "$low_pg_clipfrac" -eq 1 ]; then
            echo "[EARLY_STOP] step=$latest_step reward=$latest_reward_fmt window_avg=$latest_metric_fmt best@${history_best_step}=$history_best_metric_fmt delta=$min_delta_fmt patience=${stale_windows}/$EARLY_STOP_PATIENCE_WINDOWS improved=no ppo_kl=$latest_ppo_kl_fmt<$EARLY_STOP_PPO_KL_MAX pg_clipfrac=$latest_pg_clipfrac_fmt<$EARLY_STOP_PG_CLIPFRAC_MAX action=stop" | tee -a "$OUTER_LOG"
            skip_current=1
            early_stopped=1
            kill_train_group
            break
          elif [ "$stale_windows" -ge "$EARLY_STOP_PATIENCE_WINDOWS" ]; then
            echo "[EARLY_STOP] step=$latest_step reward=$latest_reward_fmt window_avg=$latest_metric_fmt best@${history_best_step}=$history_best_metric_fmt delta=$min_delta_fmt patience=${stale_windows}/$EARLY_STOP_PATIENCE_WINDOWS improved=no ppo_kl=$latest_ppo_kl_fmt pg_clipfrac=$latest_pg_clipfrac_fmt action=continue_policy_moving" | tee -a "$OUTER_LOG"
          else
            echo "[EARLY_STOP] step=$latest_step reward=$latest_reward_fmt window_avg=$latest_metric_fmt best@${history_best_step}=$history_best_metric_fmt delta=$min_delta_fmt patience=${stale_windows}/$EARLY_STOP_PATIENCE_WINDOWS improved=no ppo_kl=$latest_ppo_kl_fmt pg_clipfrac=$latest_pg_clipfrac_fmt action=continue_patience" | tee -a "$OUTER_LOG"
          fi
        fi
      done < <(extract_sliding_metrics_after "$INNER_LOG" "$last_seen_step")

      if [ "$early_stopped" -eq 1 ]; then
        last_check_ts="$now_ts"
        break
      fi

      if [ "$saw_metric" -eq 0 ]; then
        if [ "$missing_metrics_logged" -eq 0 ]; then
          echo "[EARLY_STOP] waiting for metrics INNER_LOG=$INNER_LOG" | tee -a "$OUTER_LOG"
          missing_metrics_logged=1
        fi
      fi

      last_check_ts="$now_ts"
    fi

    sleep 5
  done

  wait "$train_pid"
  exit_code=$?

  set -e

  cleanup_after_run

  final_best_metric=$(extract_best_metric "$INNER_LOG")

  echo "end_time=$(date '+%F %T')" | tee -a "$OUTER_LOG"
  echo "exit_code=$exit_code" | tee -a "$OUTER_LOG"
  echo "best_metric=$final_best_metric" | tee -a "$OUTER_LOG"

  rm -f "$RUNNING_MARK"

  if [ "$early_stopped" -eq 1 ]; then
    echo "[EARLY STOPPED] $RUN_TAG" | tee -a "$OUTER_LOG"
    echo "early_stopped_at=$(date '+%F %T')" > "$FAILED_MARK"
    echo "exit_code=130" >> "$FAILED_MARK"
    append_summary "$RUN_TAG" "early_stopped" "$final_best_metric" "130" "$n" "$batch" "$lr" "$temp" "$epoch" "$kl_loss_coef" "$lr_warmup_steps" "$max_turns" "$INNER_LOG" "$OUTER_LOG"

    sleep "$SLEEP_BETWEEN_RUNS"
    continue
  fi

  if [ "$skip_current" -eq 1 ]; then
    echo "[MANUAL SKIP] $RUN_TAG" | tee -a "$OUTER_LOG"
    echo "manual_skip_at=$(date '+%F %T')" > "$FAILED_MARK"
    echo "exit_code=130" >> "$FAILED_MARK"
    append_summary "$RUN_TAG" "manual_skip" "$final_best_metric" "130" "$n" "$batch" "$lr" "$temp" "$epoch" "$kl_loss_coef" "$lr_warmup_steps" "$max_turns" "$INNER_LOG" "$OUTER_LOG"

    rm -f "$SKIP_FILE"

    if [ "$stop_all" -eq 1 ] || [ -f "$STOP_FILE" ]; then
      echo "$CONFIG_MODE stopped by user." | tee -a "$OUTER_LOG"
      exit 130
    fi

    sleep "$SLEEP_BETWEEN_RUNS"
    continue
  fi

  if [ "$exit_code" -ne 0 ]; then
    echo "[FAILED] $RUN_TAG" | tee -a "$OUTER_LOG"
    echo "failed_at=$(date '+%F %T')" > "$FAILED_MARK"
    echo "exit_code=$exit_code" >> "$FAILED_MARK"
    append_summary "$RUN_TAG" "failed" "$final_best_metric" "$exit_code" "$n" "$batch" "$lr" "$temp" "$epoch" "$kl_loss_coef" "$lr_warmup_steps" "$max_turns" "$INNER_LOG" "$OUTER_LOG"

    if [ "$CONTINUE_ON_FAIL" != "1" ]; then
      exit "$exit_code"
    fi
  else
    echo "[SUCCESS] $RUN_TAG" | tee -a "$OUTER_LOG"
    echo "success_at=$(date '+%F %T')" > "$SUCCESS_MARK"
    echo "exit_code=0" >> "$SUCCESS_MARK"
    append_summary "$RUN_TAG" "success" "$final_best_metric" "0" "$n" "$batch" "$lr" "$temp" "$epoch" "$kl_loss_coef" "$lr_warmup_steps" "$max_turns" "$INNER_LOG" "$OUTER_LOG"
  fi

  sleep "$SLEEP_BETWEEN_RUNS"
done

if [ "$CONFIG_MODE" = "explicit" ]; then
  echo "========== Explicit Experiments Finished =========="
else
  echo "========== Grid Search Finished =========="
fi
echo "Summary: $SUMMARY_FILE"
