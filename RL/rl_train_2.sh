# bash ./examples/train/wikiRL/train_wikiRL_VL_3B.sh > ./logs/train_wiki_VL_3B.log 2>&1 &

set -x

# 本机运行配置参数
export PYTHONPATH=$PYTHONPATH:$(pwd)/..:$(pwd)/../..:$(pwd)/../verl-tool:$(pwd)/../mini_webarena
rm -rf ~/.triton/cache
ln -sf /usr/lib/x86_64-linux-gnu/libcuda.so.1 $CONDA_PREFIX/lib/libcuda.so
export LIBRARY_PATH=$CONDA_PREFIX/lib:$LIBRARY_PATH
export RAY_TMPDIR="/DATA/disk0/yjb/yutao/ray_tmp/run2"
mkdir -p $RAY_TMPDIR
export RAY_PORT=6379

export CUDA_VISIBLE_DEVICES=1,2,3,4
export WANDB_API_KEY=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

export MASTER_PORT=29502

# export export WANDB_MODE="offline"


LOG_DIR="$(pwd)/logs"
mkdir -p $LOG_DIR

# ===== TRACE BLOCK START =====
TRACE_DIR="$LOG_DIR/pid_trace"
mkdir -p "$TRACE_DIR"
TRACE_FILE="$TRACE_DIR/trace_$(date +%F_%H%M%S).log"

log_trace() {
    echo "[$(date '+%F %T')] $*" | tee -a "$TRACE_FILE"
}

log_trace "train_script_start shell_pid=$$ ppid=$PPID user=$(whoami) host=$(hostname)"
# ===== TRACE BLOCK END =====

sft_model_name=task-opsrc-new_add2400-sft-5e-5lr-freeze_false-2epoch
model_name=$(pwd)/models/Qwen2.5-VL-7B-Instruct-${sft_model_name}-merged

benchmark=train_hotpot500_nq500
val_dataset_name=test_20
train_data=$(pwd)/dataset/${benchmark}/data.parquet
# val_data=$(pwd)/dataset/${benchmark}/data.parquet
val_data=$(pwd)/dataset/${val_dataset_name}/data.parquet
rl_alg=mt_grpo # gae(ppo) or grpo, if grpo, then better set n>1 otherwise the group norm can not be effective
n_gpus_per_node=4
n_nodes=1
n=${ROLLOUT_N_OVERRIDE:-6}
train_batch_size=${TRAIN_BATCH_SIZE_OVERRIDE:-64}
val_batch_size=2
ppo_mini_batch_size=4
max_prompt_length=2048
max_response_length=8192
max_action_length=2048
max_obs_length=2048
temperature=0.5
top_p=1.0
enable_agent=True # enable agent for tool use
strategy="fsdp"
# action_stop_tokens='</action>'
action_stop_tokens='<dummy_stop_token_never_generate>'
action_extract_tokens='```'
max_turns=15
kl_loss_coef=0.005
kl_coef=0
entropy_coeff=0
kl_loss_type=low_var_kl
lr=1e-7
epoch=8
reward_manager=BrowserAgent
ppo_micro_batch_size_per_gpu=1
log_prob_micro_batch_size_per_gpu=${LOG_PROB_MBS_OVERRIDE:-4}
tensor_model_parallel_size=1
gpu_memory_utilization=0.4 # higher gpu_memory_utilization will likely cause the vllm to OOM and get stuck, so set it to a lower value like 0.4 or 0.5
default_do_offload=True
do_offload=${DO_OFFLOAD:-$default_do_offload} # 4-GPU runs keep CPU offload by default, but you can override with DO_OFFLOAD=False.
use_dynamic_bsz=True # faster
ulysses_sequence_parallel_size=1 # set to 1 for normal verl behavior, otherwise it will cause OOM
fsdp_size=-1
additional_eos_token_ids=[151645] # <|im_end|> token id
mask_observations=True # mask observations for kl loss and gradient descent
enable_mtrl=True # enable multi-turn training
model_pretty_name=$(echo $model_name | tr '/' '_' | tr '[:upper:]' '[:lower:]')
run_name_postfix=""
if [ "$enable_agent" = "True" ]; then
    run_name="${reward_manager}-${strategy}-agent-${model_pretty_name}-${benchmark}-${rl_alg}-n${n}-b${train_batch_size}-t${temperature}-lr${lr}${run_name_postfix}"
else
    run_name="${reward_manager}-${strategy}-${model_pretty_name}-${benchmark}-${rl_alg}-n${n}-b${train_batch_size}-t${temperature}-lr${lr}${run_name_postfix}"
fi
export VERL_RUN_ID=$run_name
export NCCL_DEBUG=INFO
export VLLM_USE_V1=1
rollout_mode='async'
# rollout_mode='sync'

# Real rollout concurrency is roughly train_batch_size * n before agent-side
# throttling. Cap it explicitly to avoid CPU/Ray OOM in multi-turn runs.
agent_loop_num_workers=${AGENT_LOOP_NUM_WORKERS:-2}
max_concurrent_trajectories=${MAX_CONCURRENT_TRAJECTORIES:-16}
tool_server_max_concurrent_requests=${TOOL_SERVER_MAX_CONCURRENT_REQUESTS:-16}
text_browser_max_active_actors=${TEXT_BROWSER_MAX_ACTIVE_ACTORS_OVERRIDE:-16}
text_browser_idle_pool_size=${TEXT_BROWSER_IDLE_POOL_SIZE_OVERRIDE:-4}
text_browser_actor_cpus=${TEXT_BROWSER_ACTOR_CPUS_OVERRIDE:-1}
pass_extra_fields_to_reward=${PASS_EXTRA_FIELDS_TO_REWARD:-False}
reward_fn_async=${REWARD_FN_ASYNC:-False}
compact_tool_interact_info=${COMPACT_TOOL_INTERACT_INFO:-True}
rollout_max_num_seqs=${ROLLOUT_MAX_NUM_SEQS_OVERRIDE:-128}
rollout_max_num_batched_tokens=${ROLLOUT_MAX_BATCHED_TOKENS_OVERRIDE:-16384}

# temp file for action tokens as verl cannot pass special strs as params
action_stop_tokens_file="$(pwd)$(mktemp)"
mkdir -p $(dirname $action_stop_tokens_file)
echo -e -n "$action_stop_tokens" | tee $action_stop_tokens_file
echo "action_stop_tokens_file=$action_stop_tokens_file"

checkpoint_path=$(pwd)/checkpoints/${sft_model_name}-n${n}-b${train_batch_size}-t${temperature}-lr${lr}-e${epoch}/
mkdir -p ${checkpoint_path}

# temp file for action extract tokens
action_extract_tokens_file="$(pwd)/$(mktemp)"
echo -e -n "$action_extract_tokens" | tee $action_extract_tokens_file

# export TEXT_BROWSER_ENV_RPC_TIMEOUT_SEC=70
# export TEXT_BROWSER_ACTION_TIMEOUT_SEC=75

export TEXT_BROWSER_MAX_ACTIVE_ACTORS=$text_browser_max_active_actors
export TEXT_BROWSER_IDLE_POOL_SIZE=$text_browser_idle_pool_size
export TEXT_BROWSER_ACTOR_CPUS=$text_browser_actor_cpus

export TEXT_BROWSER_ENV_RPC_TIMEOUT_SEC=160
export TEXT_BROWSER_ACTION_TIMEOUT_SEC=160
export TEXT_BROWSER_STEP_RETRIES=1
export TEXT_BROWSER_INIT_RETRIES=1

host=$(hostname -i | awk '{print $1}')
port=$(shuf -i 30000-31000 -n 1)
tool_server_url=http://$host:$port/get_observation

python -m verl_tool.servers.serve \
    --host $host \
    --port $port \
    --tool_type "'text_browser'" \
    --use_ray True \
    --max_concurrent_requests $tool_server_max_concurrent_requests \
    --request_timeout 180 \
    --uvi_workers 4 \
    --router_workers 4 \
    --log_directory "$LOG_DIR" > /dev/null 2>&1 &

server_pid=$!

echo "Server (pid=$server_pid) started at $tool_server_url"
log_trace "tool_server_started pid=$server_pid tool_url=$tool_server_url"

cleanup_server() {
    if [ -n "${server_pid:-}" ] && kill -0 "$server_pid" 2>/dev/null; then
        # Only cleanup the tool server process tree started by this script.
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

# ===== TRACE BLOCK START =====
on_signal_cleanup() {
    sig="$1"
    log_trace "signal_received sig=$sig shell_pid=$$ ppid=$PPID"
    cleanup_server
    exit 128
}

trap 'on_signal_cleanup SIGINT' INT
trap 'on_signal_cleanup SIGTERM' TERM
trap 'on_signal_cleanup SIGHUP' HUP
# ===== TRACE BLOCK END =====

TRAIN_START_TS="$(date '+%F %T')"
log_trace "train_start ts=$TRAIN_START_TS"
PYTHONUNBUFFERED=1 python3 -m verl_tool.trainer.main_ppo \
    algorithm.adv_estimator=$rl_alg \
    data.train_files=$train_data \
    data.val_files=$val_data \
    data.train_batch_size=$train_batch_size \
    data.val_batch_size=$val_batch_size \
    data.max_prompt_length=$max_prompt_length \
    data.max_response_length=$max_response_length \
    data.truncation='right' \
    reward_model.reward_manager=$reward_manager \
    reward_model.launch_reward_fn_async=$reward_fn_async \
    actor_rollout_ref.model.path=$model_name \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=$lr \
    actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.trust_remote_code=True \
    actor_rollout_ref.actor.checkpoint.save_contents=['model','optimizer','extra','hf_model'] \
    actor_rollout_ref.actor.ppo_mini_batch_size=$ppo_mini_batch_size \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$ppo_micro_batch_size_per_gpu \
    actor_rollout_ref.actor.use_dynamic_bsz=$use_dynamic_bsz \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.strategy=$strategy \
    actor_rollout_ref.actor.kl_loss_coef=$kl_loss_coef \
    actor_rollout_ref.actor.kl_loss_type=$kl_loss_type \
    actor_rollout_ref.actor.entropy_coeff=$entropy_coeff \
    actor_rollout_ref.actor.fsdp_config.param_offload=$do_offload \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=$do_offload \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=$fsdp_size \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=$ulysses_sequence_parallel_size \
    actor_rollout_ref.agent.enable_agent=$enable_agent \
    actor_rollout_ref.agent.tool_server_url=$tool_server_url \
    actor_rollout_ref.agent.max_prompt_length=$max_prompt_length \
    actor_rollout_ref.agent.max_response_length=$max_response_length \
    actor_rollout_ref.agent.max_start_length=$max_prompt_length \
    actor_rollout_ref.agent.max_obs_length=$max_obs_length \
    actor_rollout_ref.agent.max_turns=$max_turns \
    actor_rollout_ref.agent.additional_eos_token_ids=$additional_eos_token_ids \
    actor_rollout_ref.agent.mask_observations=$mask_observations \
    actor_rollout_ref.agent.action_stop_tokens=$action_stop_tokens_file \
    +actor_rollout_ref.agent.action_extract_tokens=$action_extract_tokens_file \
    actor_rollout_ref.agent.tool_call_timeout=180 \
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
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=$use_dynamic_bsz \
    actor_rollout_ref.ref.fsdp_config.param_offload=$do_offload \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$log_prob_micro_batch_size_per_gpu \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=$ulysses_sequence_parallel_size \
    critic.optim.lr=1e-5 \
    critic.strategy=$strategy \
    critic.model.path=$model_name \
    critic.model.fsdp_config.fsdp_size=$fsdp_size \
    critic.ppo_micro_batch_size_per_gpu=$ppo_micro_batch_size_per_gpu \
    critic.ulysses_sequence_parallel_size=$ulysses_sequence_parallel_size \
    algorithm.kl_ctrl.kl_coef=$kl_coef \
    trainer.logger=['console','wandb'] \
    trainer.project_name=wikiRL \
    trainer.experiment_name=$run_name \
    trainer.val_before_train=False \
    trainer.default_hdfs_dir=null \
    trainer.n_gpus_per_node=$n_gpus_per_node \
    trainer.nnodes=$n_nodes \
    +trainer.remove_previous_ckpt_in_save=True \
    trainer.default_local_dir=$checkpoint_path \
    trainer.save_freq=15 \
    trainer.test_freq=1000 \
    trainer.total_epochs=$epoch \
    +trainer.save_last=True \
    > "$LOG_DIR/train/${sft_model_name}-n${n}-b${train_batch_size}-t${temperature}-lr${lr}-e${epoch}.log" 2>&1

train_exit_code=$?
TRAIN_END_TS="$(date '+%F %T')"
RAYLET_PID=$(pgrep -f "raylet" | head -n1 || true)
DRIVER_PIDS=$(pgrep -f "python-core-driver|main_ppo|verl_tool.trainer" | tr '\n' ',' || true)
log_trace "train_end ts=$TRAIN_END_TS exit_code=$train_exit_code"
log_trace "runtime_pids raylet_pid=${RAYLET_PID:-NA} driver_pids=${DRIVER_PIDS:-NA}"
cleanup_server
exit $train_exit_code
