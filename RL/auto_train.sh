#!/usr/bin/env bash

# # 正常跑：已有 .success 的实验会跳过
# bash auto_train.sh
# 或
# chmod +x auto_train.sh
# ./auto_train.sh
#
# 使用 DAPO 脚本跑同样的网格/成组实验，日志结构相同但落在 logs/dapo/
# TRAIN_ALGO=dapo bash auto_train.sh

# 跳过当前实验
# Ctrl + C
# 或
# touch 对应 GRID_LOG_ROOT 下的 SKIP_CURRENT

# Ctrl + C
# Ctrl + C
# 或
# touch 对应 GRID_LOG_ROOT 下的 STOP_ALL

# 重跑（.success的实验自动跳过）
# bash auto_train.sh

# 强制全部重跑
# FORCE_RERUN=1 bash auto_train.sh

# 跑指定的几组完整配置（设置后会跳过网格搜索）
# EXPERIMENT_CONFIGS="n6_b64_lr3e-7_t0.8_kl0.01_warmup10 n6_b64_lr5e-7_t0.7_e4_kl0.005_warmup20" bash auto_train.sh
# 或修改下面 EXPERIMENT_CONFIG_LIST 数组
#
# 配置格式：
#   n{rollout_n}_b{batch}_lr{lr}_t{temperature}_e{epoch}_kl{kl_loss_coef}_warmup{lr_warmup_steps}_turns{max_turns}
# 其中 e{epoch}、turns{max_turns} 可省略；e 省略时使用 EPOCH_LIST 的第一个值，turns 省略时使用 rl_train_1.sh 默认值。

# 跳过失败实验
# SKIP_FAILED=1 bash auto_train.sh

# 失败就停止
# CONTINUE_ON_FAIL=0 bash auto_train.sh

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

#!/usr/bin/env bash
set -euo pipefail

TRAIN_ALGO="${TRAIN_ALGO:-dapo}"

case "$TRAIN_ALGO" in
  mt_grpo|grpo|ppo|default)
    TRAIN_SCRIPT="${TRAIN_SCRIPT_OVERRIDE:-/DATA/disk0/yjb/yutao/lzt/BrowserAgent_v2/RL/rl_train_1.sh}"
    LOG_ROOT="${LOG_ROOT_OVERRIDE:-$(pwd)/logs/mt_grpo}"
    BASE_RAY_TMP_DEFAULT="/DATA/disk0/yjb/yutao/ray_tmp"
    ;;
  dapo)
    TRAIN_SCRIPT="${TRAIN_SCRIPT_OVERRIDE:-/DATA/disk0/yjb/yutao/lzt/BrowserAgent_v2/RL/rl_train_dapo.sh}"
    LOG_ROOT="${LOG_ROOT_OVERRIDE:-$(pwd)/logs/dapo}"
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
N_LIST=(6)
BATCH_LIST=(64)
LR_LIST=("1e-7" "3e-7" "5e-7")
TEMP_LIST=("0.7" "1.0")
EPOCH_LIST=(4)
KL_LOSS_COEF_LIST=("0.005" "0.001")
LR_WARMUP_STEPS_LIST=(10)
# 留空表示网格搜索不覆盖 MAX_TURNS_OVERRIDE，使用 rl_train_1.sh 默认值。
MAX_TURNS_LIST=()

# ===== 指定实验列表 =====
# 非空时跳过网格搜索，按列表顺序只跑这些配置。
EXPERIMENT_CONFIG_LIST=(
  # "n6_b64_lr3e-7_t0.8_kl0.01_warmup10_turns10"
  "n6_b64_lr4e-7_t0.8_kl0.01_warmup10_turns15"
  "n6_b64_lr4e-7_t0.8_kl0.005_warmup10_turns15"
  "n6_b64_lr5e-7_t0.8_kl0.01_warmup10_turns15"
)

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
ENABLE_EARLY_STOP="${ENABLE_EARLY_STOP:-1}"
METRIC_KEY="${METRIC_KEY:-critic/rewards/mean}"
# 使用最近多少个 step 的 reward_mean 滑动平均判断早停
EARLY_STOP_REWARD_WINDOW="${EARLY_STOP_REWARD_WINDOW:-8}"
# 当前滑动平均超过历史最佳滑动平均多少才算提升
EARLY_STOP_MIN_DELTA="${EARLY_STOP_MIN_DELTA:-0.01}"
# 只有策略更新幅度也很小时，才触发 reward 滑动平均未提升早停
EARLY_STOP_PPO_KL_KEY="${EARLY_STOP_PPO_KL_KEY:-actor/ppo_kl}"
EARLY_STOP_PPO_KL_MAX="${EARLY_STOP_PPO_KL_MAX:-0.001}"
EARLY_STOP_PG_CLIPFRAC_KEY="${EARLY_STOP_PG_CLIPFRAC_KEY:-actor/pg_clipfrac}"
EARLY_STOP_PG_CLIPFRAC_MAX="${EARLY_STOP_PG_CLIPFRAC_MAX:-0.01}"
# 每隔多久检查一次日志
EARLY_STOP_CHECK_INTERVAL="${EARLY_STOP_CHECK_INTERVAL:-120}"

SKIP_FILE="$GRID_LOG_ROOT/SKIP_CURRENT"
STOP_FILE="$GRID_LOG_ROOT/STOP_ALL"

skip_current=0
stop_all=0
early_stopped=0
train_pid=""

SUMMARY_HEADER="run_tag\tstatus\tbest_metric\texit_code\tn\tbatch\tlr\ttemp\tepoch\tkl_loss_coef\tlr_warmup_steps\tmax_turns\tinner_log\touter_log"

if [ ! -f "$SUMMARY_FILE" ]; then
  echo -e "$SUMMARY_HEADER" > "$SUMMARY_FILE"
elif ! head -n 1 "$SUMMARY_FILE" | grep -q "max_turns"; then
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

extract_latest_sliding_metrics() {
  local log_file="$1"

  if [ ! -f "$log_file" ]; then
    echo "NA NA NA NA 0 NA NA"
    return
  fi

  awk -v reward_key="$METRIC_KEY" \
      -v ppo_kl_key="$EARLY_STOP_PPO_KL_KEY" \
      -v pg_clipfrac_key="$EARLY_STOP_PG_CLIPFRAC_KEY" \
      -v window="$EARLY_STOP_REWARD_WINDOW" '
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

      if (metric_count >= window) {
        if (latest_window_avg != "") {
          if (best_window_avg == "" || latest_window_avg > best_window_avg) {
            best_window_avg=latest_window_avg;
            best_window_step=latest_window_step;
          }
        }

        latest_window_step=step;
        latest_window_avg=rolling_sum / window;
        latest_ppo_kl=metric_value(ppo_kl_key);
        latest_pg_clipfrac=metric_value(pg_clipfrac_key);
      } else {
        latest_window_step=step;
        latest_ppo_kl=metric_value(ppo_kl_key);
        latest_pg_clipfrac=metric_value(pg_clipfrac_key);
      }
    }
    END {
      if (latest_window_step == "") {
        print "NA NA NA NA 0 NA NA";
      } else if (metric_count < window) {
        print latest_window_step, "NA", latest_ppo_kl, latest_pg_clipfrac, metric_count, "NA", "NA";
      } else if (best_window_avg == "") {
        print latest_window_step, latest_window_avg, latest_ppo_kl, latest_pg_clipfrac, metric_count, "NA", "NA";
      } else {
        print latest_window_step, latest_window_avg, latest_ppo_kl, latest_pg_clipfrac, metric_count, best_window_step, best_window_avg;
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

  echo -e "${run_tag}\t${status}\t${best_metric}\t${exit_code}\t${n}\t${batch}\t${lr}\t${temp}\t${epoch}\t${kl_loss_coef}\t${lr_warmup_steps}\t${max_turns}\t${inner_log}\t${outer_log}" >> "$SUMMARY_FILE"
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
echo "ENABLE_EARLY_STOP=$ENABLE_EARLY_STOP"
echo "EARLY_STOP_METRIC=$METRIC_KEY"
echo "EARLY_STOP_REWARD_WINDOW=$EARLY_STOP_REWARD_WINDOW"
echo "EARLY_STOP_MIN_DELTA=$EARLY_STOP_MIN_DELTA"
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
    RUN_TAG="exp_${run_idx}_$(sanitize_run_tag "$config_name")"
  else
    RUN_TAG="grid_${run_idx}_n${n}_b${batch}_lr${lr}_t${temp}_e${epoch}_kl${kl_loss_coef}_warmup${lr_warmup_steps}"
    if [ -n "$max_turns" ]; then
      RUN_TAG="${RUN_TAG}_turns${max_turns}"
    fi
  fi

  OUTER_LOG="$GRID_LOG_ROOT/${RUN_TAG}.outer.log"
  INNER_LOG="$TRAIN_LOG_ROOT/${RUN_TAG}.log"

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
  export RUN_NAME_POSTFIX="-${RUN_TAG}"
  export TRAIN_LOG_NAME_OVERRIDE="$RUN_TAG"
  export WANDB_RUN_NAME_OVERRIDE="${TRAIN_ALGO}-${MODEL_LOG_COMPONENT}-${DATA_LOG_COMPONENT}-${RUN_TAG}"

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
  echo "n=$n batch=$batch lr=$lr temp=$temp epoch=$epoch kl_loss_coef=$kl_loss_coef lr_warmup_steps=$lr_warmup_steps max_turns=${max_turns:-default}" | tee -a "$OUTER_LOG"
  echo "RAY_PORT_OVERRIDE=$RAY_PORT_OVERRIDE" | tee -a "$OUTER_LOG"
  echo "RAY_TMPDIR_OVERRIDE=$RAY_TMPDIR_OVERRIDE" | tee -a "$OUTER_LOG"
  echo "INNER_LOG=$INNER_LOG" | tee -a "$OUTER_LOG"
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
      read latest_step latest_metric latest_ppo_kl latest_pg_clipfrac reward_metric_count history_best_step history_best_metric < <(extract_latest_sliding_metrics "$INNER_LOG")

      if [ "$latest_step" = "NA" ] || [ "$latest_ppo_kl" = "NA" ] || [ "$latest_pg_clipfrac" = "NA" ]; then
        if [ "$missing_metrics_logged" -eq 0 ]; then
          echo "[EARLY_STOP] waiting for metrics INNER_LOG=$INNER_LOG" | tee -a "$OUTER_LOG"
          missing_metrics_logged=1
        fi
      elif [ "$latest_step" -gt "$last_seen_step" ]; then
        last_seen_step="$latest_step"
        missing_metrics_logged=0

        latest_ppo_kl_fmt=$(format_scientific "$latest_ppo_kl")
        latest_pg_clipfrac_fmt=$(format_fixed "$latest_pg_clipfrac" 4)

        if [ "$latest_metric" = "NA" ]; then
          echo "[EARLY_STOP] step=$latest_step window=$reward_metric_count/$EARLY_STOP_REWARD_WINDOW ppo_kl=$latest_ppo_kl_fmt pg_clipfrac=$latest_pg_clipfrac_fmt action=wait_window" | tee -a "$OUTER_LOG"
        elif [ "$history_best_metric" = "NA" ]; then
          latest_metric_fmt=$(format_fixed "$latest_metric" 4)
          echo "[EARLY_STOP] step=$latest_step window_avg=$latest_metric_fmt best=NA ppo_kl=$latest_ppo_kl_fmt pg_clipfrac=$latest_pg_clipfrac_fmt action=init" | tee -a "$OUTER_LOG"
        else
          improved=$(awk -v cur="$latest_metric" -v best="$history_best_metric" -v delta="$EARLY_STOP_MIN_DELTA" 'BEGIN{print (cur > best + delta) ? 1 : 0}')
          low_ppo_kl=$(is_number_less_than "$latest_ppo_kl" "$EARLY_STOP_PPO_KL_MAX")
          low_pg_clipfrac=$(is_number_less_than "$latest_pg_clipfrac" "$EARLY_STOP_PG_CLIPFRAC_MAX")
          latest_metric_fmt=$(format_fixed "$latest_metric" 4)
          history_best_metric_fmt=$(format_fixed "$history_best_metric" 4)
          min_delta_fmt=$(format_fixed "$EARLY_STOP_MIN_DELTA" 4)

          if [ "$improved" -eq 1 ]; then
            echo "[EARLY_STOP] step=$latest_step window_avg=$latest_metric_fmt best@${history_best_step}=$history_best_metric_fmt delta=$min_delta_fmt improved=yes ppo_kl=$latest_ppo_kl_fmt pg_clipfrac=$latest_pg_clipfrac_fmt action=continue" | tee -a "$OUTER_LOG"
          elif [ "$low_ppo_kl" -eq 1 ] && [ "$low_pg_clipfrac" -eq 1 ]; then
            echo "[EARLY_STOP] step=$latest_step window_avg=$latest_metric_fmt best@${history_best_step}=$history_best_metric_fmt delta=$min_delta_fmt improved=no ppo_kl=$latest_ppo_kl_fmt<$EARLY_STOP_PPO_KL_MAX pg_clipfrac=$latest_pg_clipfrac_fmt<$EARLY_STOP_PG_CLIPFRAC_MAX action=stop" | tee -a "$OUTER_LOG"
            skip_current=1
            early_stopped=1
            kill_train_group
            break
          else
            echo "[EARLY_STOP] step=$latest_step window_avg=$latest_metric_fmt best@${history_best_step}=$history_best_metric_fmt delta=$min_delta_fmt improved=no ppo_kl=$latest_ppo_kl_fmt pg_clipfrac=$latest_pg_clipfrac_fmt action=continue_policy_moving" | tee -a "$OUTER_LOG"
          fi
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
