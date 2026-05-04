#!/usr/bin/env bash

# # 正常跑：已有 .success 的实验会跳过
# bash auto_train.sh
# 或
# chmod +x auto_train.sh
# ./auto_train.sh

# 跳过当前实验
# Ctrl + C
# 或
# touch logs/grid_search/SKIP_CURRENT

# Ctrl + C
# Ctrl + C
# 或
# touch logs/grid_search/STOP_ALL

# 重跑（.success的实验自动跳过）
# bash auto_train.sh

# 强制全部重跑
# FORCE_RERUN=1 bash auto_train.sh

# 跳过失败实验
# SKIP_FAILED=1 bash auto_train.sh

# 失败就停止
# CONTINUE_ON_FAIL=0 bash auto_train.sh

# 查看结果
# cat logs/grid_search/summary.tsv

# 查看最优实验（按 metric 排序）
# awk -F'\t' 'NR>1 && $3!="NA"{print $0}' logs/grid_search/summary.tsv \
#   | sort -t$'\t' -k3,3gr \
#   | head

# 查看某个实验日志
# # 外层日志
# tail -f logs/grid_search/*.outer.log

# # 内层训练日志（关键）
# tail -f logs/train/*.log

# 日志结构
# logs/
# ├── grid_search/
# │   ├── *.outer.log        # 外层日志（控制+状态）
# │   ├── *.success
# │   ├── *.failed
# │   ├── summary.tsv        # 汇总（最重要）
# │
# └── train/
#     ├── *.log              # 训练日志（early stop 用这个）

#!/usr/bin/env bash
set -euo pipefail

TRAIN_SCRIPT="/DATA/disk0/yjb/yutao/lzt/BrowserAgent_v2/RL/rl_train_1.sh"

GRID_LOG_ROOT="$(pwd)/logs/grid_search"
mkdir -p "$GRID_LOG_ROOT"

SUMMARY_FILE="$GRID_LOG_ROOT/summary.tsv"

# ===== 搜索空间 =====
N_LIST=(6)
BATCH_LIST=(64)
LR_LIST=("1e-7" "3e-7" "5e-7")
TEMP_LIST=("0.7" "1.0")
EPOCH_LIST=(4)
KL_LOSS_COEF_LIST=("0.01" "0.03" "0.05")

CONTINUE_ON_FAIL="${CONTINUE_ON_FAIL:-1}"
SLEEP_BETWEEN_RUNS="${SLEEP_BETWEEN_RUNS:-20}"
BASE_RAY_PORT="${BASE_RAY_PORT:-6380}"
BASE_RAY_TMP="/DATA/disk0/yjb/yutao/ray_tmp"

FORCE_RERUN="${FORCE_RERUN:-0}"
SKIP_FAILED="${SKIP_FAILED:-0}"

# ===== Early Stop =====
ENABLE_EARLY_STOP="${ENABLE_EARLY_STOP:-1}"
METRIC_KEY="${METRIC_KEY:-critic/score/mean}"
# 至少跑多少个有效 step 后才允许早停
EARLY_STOP_MIN_STEPS="${EARLY_STOP_MIN_STEPS:-8}"
# 连续多少个 step 没提升就早停
EARLY_STOP_PATIENCE_STEPS="${EARLY_STOP_PATIENCE_STEPS:-15}"
# 最小提升幅度
EARLY_STOP_MIN_DELTA="${EARLY_STOP_MIN_DELTA:-0.001}"
# 每隔多久检查一次日志
EARLY_STOP_CHECK_INTERVAL="${EARLY_STOP_CHECK_INTERVAL:-120}"

# 默认匹配 reward / score / val_score / test_score 后面的数字
METRIC_KEY="${METRIC_KEY:-critic/score/mean}"

SKIP_FILE="$GRID_LOG_ROOT/SKIP_CURRENT"
STOP_FILE="$GRID_LOG_ROOT/STOP_ALL"

skip_current=0
stop_all=0
early_stopped=0
train_pid=""

if [ ! -f "$SUMMARY_FILE" ]; then
  echo -e "run_tag\tstatus\tbest_metric\texit_code\tn\tbatch\tlr\ttemp\tepoch\tkl_loss_coef\tinner_log\touter_log" > "$SUMMARY_FILE"
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
  local inner_log="${11}"
  local outer_log="${12}"

  echo -e "${run_tag}\t${status}\t${best_metric}\t${exit_code}\t${n}\t${batch}\t${lr}\t${temp}\t${epoch}\t${kl_loss_coef}\t${inner_log}\t${outer_log}" >> "$SUMMARY_FILE"
}

run_idx=0

rm -f "$SKIP_FILE" "$STOP_FILE"

echo "========== Grid Search Start =========="
echo "TRAIN_SCRIPT=$TRAIN_SCRIPT"
echo "GRID_LOG_ROOT=$GRID_LOG_ROOT"
echo "SUMMARY_FILE=$SUMMARY_FILE"
echo "FORCE_RERUN=$FORCE_RERUN"
echo "SKIP_FAILED=$SKIP_FAILED"
echo "ENABLE_EARLY_STOP=$ENABLE_EARLY_STOP"
echo ""
echo "Control:"
echo "  Ctrl+C once  -> skip current run and continue"
echo "  Ctrl+C twice -> stop all"
echo "  touch $SKIP_FILE -> skip current run"
echo "  touch $STOP_FILE -> stop all"
echo "======================================="

for n in "${N_LIST[@]}"; do
  for batch in "${BATCH_LIST[@]}"; do
    for lr in "${LR_LIST[@]}"; do
      for temp in "${TEMP_LIST[@]}"; do
        for epoch in "${EPOCH_LIST[@]}"; do
          for kl_loss_coef in "${KL_LOSS_COEF_LIST[@]}"; do

            if [ "$stop_all" -eq 1 ] || [ -f "$STOP_FILE" ]; then
              echo "Grid search stopped by user."
              cleanup_after_run
              exit 130
            fi

            run_idx=$((run_idx + 1))
            RUN_TAG="grid_${run_idx}_n${n}_b${batch}_lr${lr}_t${temp}_e${epoch}_kl${kl_loss_coef}"

            OUTER_LOG="$GRID_LOG_ROOT/${RUN_TAG}.outer.log"
            INNER_LOG="$(pwd)/logs/train/${RUN_TAG}-n${n}-b${batch}-t${temp}-lr${lr}-e${epoch}.log"

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

            export ROLLOUT_N_OVERRIDE="$n"
            export TRAIN_BATCH_SIZE_OVERRIDE="$batch"
            export LR_OVERRIDE="$lr"
            export TEMPERATURE_OVERRIDE="$temp"
            export EPOCH_OVERRIDE="$epoch"
            export KL_LOSS_COEF_OVERRIDE="$kl_loss_coef"

            export RAY_PORT_OVERRIDE=$((BASE_RAY_PORT + run_idx))
            export RAY_TMPDIR_OVERRIDE="${BASE_RAY_TMP}/r${run_idx}"
            mkdir -p "$RAY_TMPDIR_OVERRIDE"

            rm -f "$SUCCESS_MARK" "$FAILED_MARK" "$SKIP_FILE"
            echo "started_at=$(date '+%F %T')" > "$RUNNING_MARK"

            echo "" | tee -a "$OUTER_LOG"
            echo "========== Run $run_idx ==========" | tee -a "$OUTER_LOG"
            echo "RUN_TAG=$RUN_TAG" | tee -a "$OUTER_LOG"
            echo "n=$n batch=$batch lr=$lr temp=$temp epoch=$epoch kl_loss_coef=$kl_loss_coef" | tee -a "$OUTER_LOG"
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

            best_metric="NA"
            best_step=0
            last_seen_step=0
            no_improve_steps=0
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
                read latest_step latest_metric < <(extract_latest_step_metric "$INNER_LOG")

                if [ "$latest_step" = "NA" ] || [ "$latest_metric" = "NA" ]; then
                echo "[EARLY_STOP] metric not found yet in INNER_LOG=$INNER_LOG" | tee -a "$OUTER_LOG"
                elif [ "$latest_step" -gt "$last_seen_step" ]; then
                last_seen_step="$latest_step"

                if [ "$best_metric" = "NA" ]; then
                    best_metric="$latest_metric"
                    best_step="$latest_step"
                    no_improve_steps=0
                    echo "[EARLY_STOP] init step=$latest_step metric=$latest_metric" | tee -a "$OUTER_LOG"
                else
                    improved=$(awk -v cur="$latest_metric" -v best="$best_metric" -v delta="$EARLY_STOP_MIN_DELTA" 'BEGIN{print (cur > best + delta) ? 1 : 0}')

                    if [ "$improved" -eq 1 ]; then
                    best_metric="$latest_metric"
                    best_step="$latest_step"
                    no_improve_steps=0
                    echo "[EARLY_STOP] improved step=$latest_step best_metric=$best_metric" | tee -a "$OUTER_LOG"
                    else
                    no_improve_steps=$((no_improve_steps + 1))
                    echo "[EARLY_STOP] no improve step=$latest_step metric=$latest_metric best=$best_metric no_improve_steps=$no_improve_steps" | tee -a "$OUTER_LOG"
                    fi
                fi

                if [ "$latest_step" -ge "$EARLY_STOP_MIN_STEPS" ] && [ "$no_improve_steps" -ge "$EARLY_STOP_PATIENCE_STEPS" ]; then
                    echo "[EARLY_STOP] stop current run: step=$latest_step best_step=$best_step best_metric=$best_metric no_improve_steps=$no_improve_steps" | tee -a "$OUTER_LOG"
                    skip_current=1
                    early_stopped=1
                    kill_train_group
                    break
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
              append_summary "$RUN_TAG" "early_stopped" "$final_best_metric" "130" "$n" "$batch" "$lr" "$temp" "$epoch" "$kl_loss_coef" "$INNER_LOG" "$OUTER_LOG"

              sleep "$SLEEP_BETWEEN_RUNS"
              continue
            fi

            if [ "$skip_current" -eq 1 ]; then
              echo "[MANUAL SKIP] $RUN_TAG" | tee -a "$OUTER_LOG"
              echo "manual_skip_at=$(date '+%F %T')" > "$FAILED_MARK"
              echo "exit_code=130" >> "$FAILED_MARK"
              append_summary "$RUN_TAG" "manual_skip" "$final_best_metric" "130" "$n" "$batch" "$lr" "$temp" "$epoch" "$kl_loss_coef" "$INNER_LOG" "$OUTER_LOG"

              rm -f "$SKIP_FILE"

              if [ "$stop_all" -eq 1 ] || [ -f "$STOP_FILE" ]; then
                echo "Grid search stopped by user." | tee -a "$OUTER_LOG"
                exit 130
              fi

              sleep "$SLEEP_BETWEEN_RUNS"
              continue
            fi

            if [ "$exit_code" -ne 0 ]; then
              echo "[FAILED] $RUN_TAG" | tee -a "$OUTER_LOG"
              echo "failed_at=$(date '+%F %T')" > "$FAILED_MARK"
              echo "exit_code=$exit_code" >> "$FAILED_MARK"
              append_summary "$RUN_TAG" "failed" "$final_best_metric" "$exit_code" "$n" "$batch" "$lr" "$temp" "$epoch" "$kl_loss_coef" "$INNER_LOG" "$OUTER_LOG"

              if [ "$CONTINUE_ON_FAIL" != "1" ]; then
                exit "$exit_code"
              fi
            else
              echo "[SUCCESS] $RUN_TAG" | tee -a "$OUTER_LOG"
              echo "success_at=$(date '+%F %T')" > "$SUCCESS_MARK"
              echo "exit_code=0" >> "$SUCCESS_MARK"
              append_summary "$RUN_TAG" "success" "$final_best_metric" "0" "$n" "$batch" "$lr" "$temp" "$epoch" "$kl_loss_coef" "$INNER_LOG" "$OUTER_LOG"
            fi

            sleep "$SLEEP_BETWEEN_RUNS"

          done
        done
      done
    done
  done
done

echo "========== Grid Search Finished =========="
echo "Summary: $SUMMARY_FILE"