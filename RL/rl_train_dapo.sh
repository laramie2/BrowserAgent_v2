#!/usr/bin/env bash
set -x

# 本机运行配置参数
export PYTHONPATH=$PYTHONPATH:$(pwd)/..:$(pwd)/../..:$(pwd)/../verl-tool:$(pwd)/../mini_webarena
rm -rf ~/.triton/cache
ln -sf /usr/lib/x86_64-linux-gnu/libcuda.so.1 $CONDA_PREFIX/lib/libcuda.so
export LIBRARY_PATH=$CONDA_PREFIX/lib:$LIBRARY_PATH

export RUN_TAG="${RUN_TAG:-run1-dapo}"
# Ray creates AF_UNIX socket files under RAY_TMPDIR; keep this path short
# or plasma_store can exceed Linux's 107-byte socket path limit.
export RAY_TMPDIR="${RAY_TMPDIR_OVERRIDE:-/tmp/raydapo_${RUN_TAG:0:12}}"
mkdir -p "$RAY_TMPDIR"

export RAY_PORT="${RAY_PORT_OVERRIDE:-6378}"

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export WANDB_API_KEY=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

export MASTER_PORT="${MASTER_PORT_OVERRIDE:-29501}"

sft_model_name="${SFT_MODEL_NAME_OVERRIDE:-task-opsrc_12619-sft-5e-5lr-freeze_false-2epoch}"
model_name=$(pwd)/models/Qwen2.5-VL-7B-Instruct-${sft_model_name}-merged
export MINI_WEB_ARENA_PROMPT_MODEL="$model_name"
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

benchmark="${BENCHMARK_OVERRIDE:-train_hotpot500_nq500}"
val_dataset_name="${VAL_DATASET_NAME_OVERRIDE:-test_20}"
train_data=$(pwd)/dataset/${benchmark}/data.parquet
val_data=$(pwd)/dataset/${val_dataset_name}/data.parquet

sanitize_path_component() {
    printf '%s' "$1" | tr -c 'A-Za-z0-9._-' '-'
}

model_log_component="$(sanitize_path_component "$sft_model_name")"
data_log_component="$(sanitize_path_component "${benchmark}_val-${val_dataset_name}")"
LOG_CONTEXT_DIR="${LOG_CONTEXT_DIR_OVERRIDE:-${model_log_component}/${data_log_component}}"

DAPO_ARTIFACT_SUFFIX="${DAPO_ARTIFACT_SUFFIX:-dapo}"
LOG_DIR="${LOG_DIR_OVERRIDE:-$(pwd)/logs/${DAPO_ARTIFACT_SUFFIX}}"
TRAIN_LOG_DIR="$LOG_DIR/train/$LOG_CONTEXT_DIR"
mkdir -p "$LOG_DIR"
mkdir -p "$TRAIN_LOG_DIR"

TRACE_DIR="$LOG_DIR/pid_trace/$LOG_CONTEXT_DIR"
mkdir -p "$TRACE_DIR"
TRACE_FILE="$TRACE_DIR/trace_${RUN_TAG}_$(date +%F_%H%M%S).log"

log_trace() {
    echo "[$(date '+%F %T')] $*" | tee -a "$TRACE_FILE"
}

log_trace "train_script_start run_tag=$RUN_TAG shell_pid=$$ ppid=$PPID user=$(whoami) host=$(hostname)"
log_trace "log_context_dir=$LOG_CONTEXT_DIR sft_model_name=$sft_model_name benchmark=$benchmark val_dataset_name=$val_dataset_name"

# 添加分层抽样的标签范围参数
enable_stratified_sampler=False
stratified_hotpot_count=500
stratified_nq_count=500
stratified_total_count=$((stratified_hotpot_count + stratified_nq_count))
stratified_label_ranges="[{label:hotpot,start:0,end:${stratified_hotpot_count}},{label:nq,start:${stratified_hotpot_count},end:${stratified_total_count}}]"
sampler_args=()
if [ "$enable_stratified_sampler" = "True" ]; then
    sampler_args+=(
        data.dataloader_num_workers=0
        data.sampler.class_path=pkg://verl_tool.trainer.stratified_sampler
        data.sampler.class_name=StratifiedSourceSampler
        data.sampler.label_ranges="$stratified_label_ranges"
    )
fi

# DAPO in verl-tool is configured as GRPO plus dynamic filtering,
# decoupled clipping, token-level loss aggregation, and overlong masking.
rl_alg=grpo
n_gpus_per_node=8
n_nodes=1

n=${ROLLOUT_N_OVERRIDE:-8}
train_batch_size=${TRAIN_BATCH_SIZE_OVERRIDE:-64}
val_batch_size=2
ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE_OVERRIDE:-$train_batch_size}

max_prompt_length=2048
max_response_length=8192
max_action_length=2048
max_obs_length=2048

temperature=${TEMPERATURE_OVERRIDE:-0.5}
top_p=${TOP_P_OVERRIDE:-1.0}

enable_agent=True
strategy="fsdp"

action_stop_tokens='<dummy_stop_token_never_generate>'
action_extract_tokens='```'

max_turns=${MAX_TURNS_OVERRIDE:-8}

use_kl_loss=${USE_KL_LOSS_OVERRIDE:-False}
kl_loss_coef=${KL_LOSS_COEF_OVERRIDE:-0}
kl_coef=${KL_COEF_OVERRIDE:-0}
entropy_coeff=${ENTROPY_COEFF_OVERRIDE:-0}
kl_loss_type=low_var_kl

enable_dapo_filter_groups=${ENABLE_DAPO_FILTER_GROUPS_OVERRIDE:-True}
dapo_filter_groups_metric=${DAPO_FILTER_GROUPS_METRIC_OVERRIDE:-seq_final_reward}
dapo_max_num_gen_batches=${DAPO_MAX_NUM_GEN_BATCHES_OVERRIDE:-0}
norm_adv_by_std_in_grpo=${NORM_ADV_BY_STD_IN_GRPO_OVERRIDE:-True}
mask_overlong_loss=${MASK_OVERLONG_LOSS_OVERRIDE:-True}
clip_ratio_low=${CLIP_RATIO_LOW_OVERRIDE:-0.2}
clip_ratio_high=${CLIP_RATIO_HIGH_OVERRIDE:-0.3}
clip_ratio_c=${CLIP_RATIO_C_OVERRIDE:-10.0}
loss_agg_mode=${LOSS_AGG_MODE_OVERRIDE:-token-mean}

lr=${LR_OVERRIDE:-5e-7}
lr_warmup_steps=${LR_WARMUP_STEPS_OVERRIDE:-10}
epoch=${EPOCH_OVERRIDE:-8}

reward_manager=BrowserAgent
ppo_micro_batch_size_per_gpu=1
log_prob_micro_batch_size_per_gpu=${LOG_PROB_MBS_OVERRIDE:-4}
tensor_model_parallel_size=1
gpu_memory_utilization=${GPU_MEMORY_UTILIZATION_OVERRIDE:-0.3}

do_offload=False
critic_param_offload=${CRITIC_PARAM_OFFLOAD_OVERRIDE:-False}
critic_optimizer_offload=${CRITIC_OPTIMIZER_OFFLOAD_OVERRIDE:-False}

use_dynamic_bsz=True
ulysses_sequence_parallel_size=1
fsdp_size=-1

additional_eos_token_ids=[151645]
mask_observations=True
enable_mtrl=True

model_pretty_name="${MODEL_PRETTY_NAME_OVERRIDE:-$sft_model_name}"
model_pretty_name=$(echo "$model_pretty_name" | tr '/' '_' | tr '[:upper:]' '[:lower:]')
data_pretty_name="${DATA_PRETTY_NAME_OVERRIDE:-${benchmark}_val-${val_dataset_name}}"
data_pretty_name=$(echo "$data_pretty_name" | tr '/' '_' | tr '[:upper:]' '[:lower:]')
run_name_postfix="${RUN_NAME_POSTFIX:--dapo}"

if [ "$enable_agent" = "True" ]; then
    run_name="${reward_manager}-${strategy}-agent-${model_pretty_name}-${data_pretty_name}-${rl_alg}-n${n}-b${train_batch_size}-t${temperature}-lr${lr}${run_name_postfix}"
else
    run_name="${reward_manager}-${strategy}-${model_pretty_name}-${data_pretty_name}-${rl_alg}-n${n}-b${train_batch_size}-t${temperature}-lr${lr}${run_name_postfix}"
fi
run_name="${WANDB_RUN_NAME_OVERRIDE:-$run_name}"

export VERL_RUN_ID="$run_name"
export NCCL_DEBUG=INFO
export VLLM_USE_V1=1

rollout_mode='async'

# ===== 内层并发参数：保持默认不变 =====
agent_loop_num_workers=${AGENT_LOOP_NUM_WORKERS:-4}
max_concurrent_trajectories=${MAX_CONCURRENT_TRAJECTORIES:-64}
tool_server_max_concurrent_requests=${TOOL_SERVER_MAX_CONCURRENT_REQUESTS:-64}
text_browser_max_active_actors=${TEXT_BROWSER_MAX_ACTIVE_ACTORS_OVERRIDE:-64}
text_browser_idle_pool_size=${TEXT_BROWSER_IDLE_POOL_SIZE_OVERRIDE:-4}
text_browser_actor_cpus=${TEXT_BROWSER_ACTOR_CPUS_OVERRIDE:-1}
pass_extra_fields_to_reward=${PASS_EXTRA_FIELDS_TO_REWARD:-False}
reward_fn_async=${REWARD_FN_ASYNC:-False}
compact_tool_interact_info=${COMPACT_TOOL_INTERACT_INFO:-True}
rollout_max_num_seqs=${ROLLOUT_MAX_NUM_SEQS_OVERRIDE:-64}
rollout_max_num_batched_tokens=${ROLLOUT_MAX_BATCHED_TOKENS_OVERRIDE:-16384}

action_stop_tokens_file="$(pwd)$(mktemp)"
mkdir -p "$(dirname "$action_stop_tokens_file")"
echo -e -n "$action_stop_tokens" | tee "$action_stop_tokens_file"
echo "action_stop_tokens_file=$action_stop_tokens_file"

checkpoint_path="${CHECKPOINT_PATH_OVERRIDE:-$(pwd)/checkpoints_${DAPO_ARTIFACT_SUFFIX}/${sft_model_name}-${benchmark}/${RUN_TAG}-n${n}-b${train_batch_size}-t${temperature}-lr${lr}-e${epoch}/}"
mkdir -p "$checkpoint_path"

action_extract_tokens_file="$(pwd)/$(mktemp)"
echo -e -n "$action_extract_tokens" | tee "$action_extract_tokens_file"

export TEXT_BROWSER_MAX_ACTIVE_ACTORS=$text_browser_max_active_actors
export TEXT_BROWSER_IDLE_POOL_SIZE=$text_browser_idle_pool_size
export TEXT_BROWSER_ACTOR_CPUS=$text_browser_actor_cpus

export TEXT_BROWSER_ENV_RPC_TIMEOUT_SEC=240
export TEXT_BROWSER_ACTION_TIMEOUT_SEC=240
export TEXT_BROWSER_STEP_RETRIES=1
export TEXT_BROWSER_INIT_RETRIES=1

host=$(hostname -i | awk '{print $1}')
port=$(shuf -i 30000-31000 -n 1)
tool_server_url=http://$host:$port/get_observation

python -m verl_tool.servers.serve \
    --host "$host" \
    --port "$port" \
    --tool_type "'text_browser'" \
    --use_ray True \
    --max_concurrent_requests "$tool_server_max_concurrent_requests" \
    --request_timeout 300 \
    --uvi_workers 4 \
    --router_workers 4 \
    --log_directory "$LOG_DIR" > /dev/null 2>&1 &

server_pid=$!

echo "Server pid=$server_pid started at $tool_server_url"
log_trace "tool_server_started pid=$server_pid tool_url=$tool_server_url"

cleanup_server() {
    if [ -n "${server_pid:-}" ] && kill -0 "$server_pid" 2>/dev/null; then
        child_pids=$(pgrep -P "$server_pid" 2>/dev/null || true)
        kill "$server_pid" 2>/dev/null || true
        if [ -n "$child_pids" ]; then
            kill $child_pids 2>/dev/null || true
        fi
        sleep 2
        kill -9 "$server_pid" 2>/dev/null || true
        if [ -n "$child_pids" ]; then
            kill -9 $child_pids 2>/dev/null || true
        fi
    fi
}

on_signal_cleanup() {
    sig="$1"
    log_trace "signal_received sig=$sig shell_pid=$$ ppid=$PPID"
    cleanup_server
    exit 128
}

trap 'on_signal_cleanup SIGINT' INT
trap 'on_signal_cleanup SIGTERM' TERM
trap 'on_signal_cleanup SIGHUP' HUP

TRAIN_START_TS="$(date '+%F %T')"
log_trace "train_start ts=$TRAIN_START_TS run_name=$run_name"

default_train_log_name="${RUN_TAG}-n${n}-b${train_batch_size}-t${temperature}-lr${lr}-e${epoch}"
train_log_name="${TRAIN_LOG_NAME_OVERRIDE:-$default_train_log_name}"

PYTHONUNBUFFERED=1 python3 -m verl_tool.trainer.main_ppo \
    algorithm.adv_estimator=$rl_alg \
    +algorithm.filter_groups.enable=$enable_dapo_filter_groups \
    +algorithm.filter_groups.metric=$dapo_filter_groups_metric \
    +algorithm.filter_groups.max_num_gen_batches=$dapo_max_num_gen_batches \
    algorithm.norm_adv_by_std_in_grpo=$norm_adv_by_std_in_grpo \
    data.train_files=$train_data \
    data.val_files=$val_data \
    data.train_batch_size=$train_batch_size \
    data.val_batch_size=$val_batch_size \
    data.max_prompt_length=$max_prompt_length \
    data.max_response_length=$max_response_length \
    data.truncation='right' \
    "${sampler_args[@]}" \
    reward_model.reward_manager=$reward_manager \
    reward_model.launch_reward_fn_async=$reward_fn_async \
    actor_rollout_ref.model.path=$model_name \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=$lr \
    actor_rollout_ref.actor.optim.lr_warmup_steps=$lr_warmup_steps \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.trust_remote_code=True \
    actor_rollout_ref.actor.checkpoint.save_contents=['model','optimizer','extra','hf_model'] \
    actor_rollout_ref.actor.ppo_mini_batch_size=$ppo_mini_batch_size \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$ppo_micro_batch_size_per_gpu \
    actor_rollout_ref.actor.use_dynamic_bsz=$use_dynamic_bsz \
    actor_rollout_ref.actor.use_kl_loss=$use_kl_loss \
    actor_rollout_ref.actor.strategy=$strategy \
    actor_rollout_ref.actor.kl_loss_coef=$kl_loss_coef \
    actor_rollout_ref.actor.kl_loss_type=$kl_loss_type \
    actor_rollout_ref.actor.entropy_coeff=$entropy_coeff \
    actor_rollout_ref.actor.fsdp_config.param_offload=$do_offload \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=$do_offload \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=$fsdp_size \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=$ulysses_sequence_parallel_size \
    actor_rollout_ref.actor.clip_ratio_high=$clip_ratio_high \
    actor_rollout_ref.actor.clip_ratio_low=$clip_ratio_low \
    actor_rollout_ref.actor.clip_ratio_c=$clip_ratio_c \
    actor_rollout_ref.actor.loss_agg_mode=$loss_agg_mode \
    actor_rollout_ref.agent.enable_agent=$enable_agent \
    actor_rollout_ref.agent.tool_server_url=$tool_server_url \
    actor_rollout_ref.agent.max_prompt_length=$max_prompt_length \
    actor_rollout_ref.agent.max_response_length=$max_response_length \
    actor_rollout_ref.agent.max_start_length=$max_prompt_length \
    actor_rollout_ref.agent.max_obs_length=$max_obs_length \
    actor_rollout_ref.agent.max_turns=$max_turns \
    actor_rollout_ref.agent.additional_eos_token_ids=$additional_eos_token_ids \
    actor_rollout_ref.agent.mask_observations=$mask_observations \
    actor_rollout_ref.agent.mask_overlong_loss=$mask_overlong_loss \
    actor_rollout_ref.agent.action_stop_tokens=$action_stop_tokens_file \
    +actor_rollout_ref.agent.action_extract_tokens=$action_extract_tokens_file \
    actor_rollout_ref.agent.tool_call_timeout=300 \
    +actor_rollout_ref.agent.tool_call_max_retries=0 \
    actor_rollout_ref.agent.enable_mtrl=$enable_mtrl \
    actor_rollout_ref.agent.max_action_length=$max_action_length \
    actor_rollout_ref.agent.max_concurrent_trajectories=$max_concurrent_trajectories \
    +actor_rollout_ref.agent.pass_extra_fields_to_reward=$pass_extra_fields_to_reward \
    +actor_rollout_ref.agent.compact_tool_interact_info=$compact_tool_interact_info \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$tensor_model_parallel_size \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$log_prob_micro_batch_size_per_gpu \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=$gpu_memory_utilization \
    actor_rollout_ref.rollout.temperature=$temperature \
    actor_rollout_ref.rollout.top_p=$top_p \
    actor_rollout_ref.rollout.top_k=-1 \
    actor_rollout_ref.rollout.n=$n \
    actor_rollout_ref.rollout.agent.num_workers=$agent_loop_num_workers \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=$use_dynamic_bsz \
    actor_rollout_ref.rollout.max_num_seqs=$rollout_max_num_seqs \
    actor_rollout_ref.rollout.mode=$rollout_mode \
    actor_rollout_ref.rollout.max_num_batched_tokens=$rollout_max_num_batched_tokens \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.mm_processor_cache_gb=0 \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=$use_dynamic_bsz \
    actor_rollout_ref.ref.fsdp_config.param_offload=$do_offload \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$log_prob_micro_batch_size_per_gpu \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=$ulysses_sequence_parallel_size \
    critic.optim.lr=1e-5 \
    critic.strategy=$strategy \
    critic.model.path=$model_name \
    critic.model.fsdp_config.param_offload=$critic_param_offload \
    critic.model.fsdp_config.optimizer_offload=$critic_optimizer_offload \
    critic.model.fsdp_config.fsdp_size=$fsdp_size \
    critic.ppo_micro_batch_size_per_gpu=$ppo_micro_batch_size_per_gpu \
    critic.ulysses_sequence_parallel_size=$ulysses_sequence_parallel_size \
    algorithm.kl_ctrl.kl_coef=$kl_coef \
    trainer.logger=['console','wandb'] \
    trainer.experiment_name=$run_name \
    trainer.val_before_train=False \
    trainer.default_hdfs_dir=null \
    trainer.n_gpus_per_node=$n_gpus_per_node \
    trainer.nnodes=$n_nodes \
    +trainer.remove_previous_ckpt_in_save=True \
    trainer.default_local_dir=$checkpoint_path \
    trainer.save_freq=10 \
    trainer.test_freq=1000 \
    trainer.total_epochs=$epoch \
    +trainer.save_last=True \
    > "$TRAIN_LOG_DIR/${train_log_name}.log" 2>&1

train_exit_code=$?
TRAIN_END_TS="$(date '+%F %T')"
RAYLET_PID=$(pgrep -f "raylet" | head -n1 || true)
DRIVER_PIDS=$(pgrep -f "python-core-driver|main_ppo|verl_tool.trainer" | tr '\n' ',' || true)

log_trace "train_end ts=$TRAIN_END_TS exit_code=$train_exit_code"
log_trace "runtime_pids raylet_pid=${RAYLET_PID:-NA} driver_pids=${DRIVER_PIDS:-NA}"

cleanup_server
exit $train_exit_code
