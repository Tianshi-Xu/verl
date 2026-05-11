#!/usr/bin/env bash
# ALFWorld validation | multi-turn agent loop | optional skill injection

set -euo pipefail

source /mnt/data/xts/setup_env.sh
activate_torch290

export TMPDIR="${RUNTIME_TMPDIR:-/tmp/xts-runtime-tmp}"
mkdir -p "$TMPDIR"

verl_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
skillrl_root="$verl_root/examples/skillrl_agent_loop"
skill_data_root="$skillrl_root/memory_data"

export VERL_ROOT="$verl_root"
export SKILLRL_ROOT="$skillrl_root"
export PYTHONPATH="$verl_root:$skillrl_root:${PYTHONPATH:-}"
export HYDRA_FULL_ERROR=1
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export RAY_DISABLE_GPU_METRICS=1
export RAY_DISABLE_DASHBOARD=1
export RAY_DEDUP_LOGS=1

cd "$verl_root"

if [[ -n "${WANDB_API_KEY:-}" ]]; then
    wandb login --relogin "$WANDB_API_KEY"
fi

########################### user settings ###########################
project_name=skill-v1
experiment_name=4b-base

model_path=/mnt/data/xts/models/Qwen3-4B
data_root=/mnt/data/xts/data/skillrl-agent-loop
alfworld_data=/mnt/data/xts/data/alfworld

train_data_size=16
val_data_size=-1

max_steps=50
max_prompt_length=4096
max_response_length=27000
max_tool_response_length=null
max_model_len=$((max_prompt_length + max_response_length + 1))

val_temperature=0.4
val_do_sample=true
val_n=4
val_shuffle=true

ngpus_per_node=8
nnodes=1
agent_num_workers="$ngpus_per_node"
val_only=true

alfworld_eval_dataset=eval_in_distribution
inject_skills=false
skill_json_path="$skill_data_root/alfworld/toolcall_skills_v2.json"
skill_top_k=6
task_specific_top_k=null
mistakes_top_k=5

rollout_gpu_memory_utilization=0.85
rollout_tensor_parallel_size=1

logger='["console","wandb"]'

########################### submission overrides ###########################

model_path="${MODEL_PATH:-$model_path}"
data_root="${DATA_ROOT:-$data_root}"
alfworld_data="${ALFWORLD_DATA:-$alfworld_data}"
train_data_size="${TRAIN_SIZE:-$train_data_size}"
val_data_size="${VAL_SIZE:-$val_data_size}"
max_steps="${MAX_STEPS:-$max_steps}"
max_prompt_length="${MAX_PROMPT_LENGTH:-$max_prompt_length}"
max_response_length="${MAX_RESPONSE_LENGTH:-$max_response_length}"
if [[ -n "${MAX_MODEL_LEN:-}" ]]; then
    max_model_len="$MAX_MODEL_LEN"
else
    max_model_len=$((max_prompt_length + max_response_length + 1))
fi
val_temperature="${VAL_TEMPERATURE:-$val_temperature}"
val_do_sample="${VAL_DO_SAMPLE:-$val_do_sample}"
val_n="${VAL_N:-$val_n}"
val_shuffle="${VAL_SHUFFLE:-$val_shuffle}"
ngpus_per_node="${NGPUS_PER_NODE:-$ngpus_per_node}"
nnodes="${NNODES:-$nnodes}"
agent_num_workers="${AGENT_NUM_WORKERS:-$ngpus_per_node}"
val_only="${VAL_ONLY:-$val_only}"
alfworld_eval_dataset="${ALFWORLD_EVAL_DATASET:-$alfworld_eval_dataset}"
inject_skills="${INJECT_SKILLS:-$inject_skills}"
skill_json_path="${SKILL_JSON_PATH:-$skill_json_path}"
skill_top_k="${SKILL_TOP_K:-$skill_top_k}"
task_specific_top_k="${TASK_SPECIFIC_TOP_K:-$task_specific_top_k}"
mistakes_top_k="${MISTAKES_TOP_K:-$mistakes_top_k}"
rollout_gpu_memory_utilization="${ROLLOUT_GPU_MEMORY_UTILIZATION:-$rollout_gpu_memory_utilization}"
rollout_tensor_parallel_size="${ROLLOUT_TENSOR_PARALLEL_SIZE:-$rollout_tensor_parallel_size}"
project_name="${PROJECT_NAME:-$project_name}"
experiment_name="${EXPERIMENT_NAME:-$experiment_name}"
logger="${LOGGER:-$logger}"

for arg in "$@"; do
    case "$arg" in
        trainer.experiment_name=*)
            experiment_name="${arg#trainer.experiment_name=}"
            ;;
    esac
done

########################### derived paths ###########################

dump_root="${DUMP_ROOT:-$verl_root/recipe/skillrl_agent_loop/$experiment_name}"
rollout_data_dir="${ROLLOUT_DATA_DIR:-$dump_root/rollout}"
validation_data_dir="${VALIDATION_DATA_DIR:-$dump_root/validation}"

train_file="$data_root/alfworld/train.parquet"
val_file="$data_root/alfworld/test.parquet"
alfworld_config_path="$skillrl_root/agent_system/environments/env_package/alfworld/configs/config_tw.yaml"
tool_config_path="$verl_root/examples/skillrl_agent_loop/config/alfworld_tool.yaml"
agent_loop_config_path="$verl_root/examples/skillrl_agent_loop/config/agent_loop.yaml"

export SKILLRL_ALFWORLD_MAX_STEPS="$max_steps"
export SKILLRL_ALFWORLD_EVAL_DATASET="$alfworld_eval_dataset"
export SKILLRL_ALFWORLD_INJECT_SKILLS="$inject_skills"
export SKILLRL_ALFWORLD_SKILL_JSON_PATH="$skill_json_path"
export SKILLRL_ALFWORLD_SKILL_TOP_K="$skill_top_k"
export SKILLRL_ALFWORLD_TASK_SPECIFIC_TOP_K="$task_specific_top_k"
export SKILLRL_ALFWORLD_MISTAKES_TOP_K="$mistakes_top_k"
export ALFWORLD_DATA="$alfworld_data"

mkdir -p "$rollout_data_dir" "$validation_data_dir"

echo "ALFWorld model: $model_path"
echo "ALFWorld data: $ALFWORLD_DATA"
echo "ALFWorld eval dataset: $alfworld_eval_dataset"
echo "ALFWorld inject skills: $inject_skills"
echo "ALFWorld skill file: $skill_json_path"
echo "ALFWorld skill top-k: skill=$skill_top_k task_specific=$task_specific_top_k mistakes=$mistakes_top_k"
echo "ALFWorld validation sampling: temperature=$val_temperature do_sample=$val_do_sample n=$val_n shuffle=$val_shuffle"
echo "ALFWorld rollout dump dir: $rollout_data_dir"
echo "ALFWorld validation dump dir: $validation_data_dir"

########################### prepare data ###########################

python examples/skillrl_agent_loop/prepare_agent_data.py \
    --data_source alfworld \
    --local_dir "$data_root" \
    --train_data_size "$train_data_size" \
    --val_data_size "$val_data_size" \
    --alfworld_config_path "$alfworld_config_path" \
    --alfworld_eval_dataset "$alfworld_eval_dataset"

########################### parameter arrays ###########################

data=(
    algorithm.adv_estimator=grpo
    algorithm.use_kl_in_reward=False
    data.train_files="$train_file"
    data.val_files="$val_file"
    data.train_batch_size="$train_data_size"
    data.max_prompt_length="$max_prompt_length"
    data.max_response_length="$max_response_length"
    data.filter_overlong_prompts=True
    data.truncation=error
    data.return_raw_chat=True
)

model=(
    actor_rollout_ref.model.path="$model_path"
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.model.enable_gradient_checkpointing=True
)

actor=(
    actor_rollout_ref.actor.optim.lr=1e-6
    actor_rollout_ref.actor.ppo_mini_batch_size="$train_data_size"
    actor_rollout_ref.actor.use_dynamic_bsz=True
    actor_rollout_ref.actor.use_kl_loss=False
    actor_rollout_ref.actor.fsdp_config.dtype=float16
    actor_rollout_ref.actor.fsdp_config.param_offload=True
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True
)

rollout=(
    actor_rollout_ref.rollout.name=vllm
    actor_rollout_ref.rollout.mode=async
    actor_rollout_ref.rollout.dtype=float16
    actor_rollout_ref.rollout.n=1
    actor_rollout_ref.rollout.tensor_model_parallel_size="$rollout_tensor_parallel_size"
    actor_rollout_ref.rollout.gpu_memory_utilization="$rollout_gpu_memory_utilization"
    actor_rollout_ref.rollout.max_model_len="$max_model_len"
    actor_rollout_ref.rollout.val_kwargs.temperature="$val_temperature"
    actor_rollout_ref.rollout.val_kwargs.do_sample="$val_do_sample"
    actor_rollout_ref.rollout.val_kwargs.n="$val_n"
    actor_rollout_ref.rollout.val_shuffle="$val_shuffle"
)

agent_loop=(
    actor_rollout_ref.rollout.multi_turn.enable=True
    actor_rollout_ref.rollout.multi_turn.tool_config_path="$tool_config_path"
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=$((max_steps + 1))
    actor_rollout_ref.rollout.multi_turn.max_tool_response_length="$max_tool_response_length"
    actor_rollout_ref.rollout.agent.default_agent_loop=skillrl_env_agent
    actor_rollout_ref.rollout.agent.num_workers="$agent_num_workers"
    actor_rollout_ref.rollout.agent.agent_loop_config_path="$agent_loop_config_path"
)

trainer=(
    reward_model.enable=False
    trainer.logger="$logger"
    trainer.project_name="$project_name"
    trainer.experiment_name="$experiment_name"
    trainer.n_gpus_per_node="$ngpus_per_node"
    trainer.nnodes="$nnodes"
    trainer.val_before_train=True
    trainer.val_only="$val_only"
    trainer.rollout_data_dir="$rollout_data_dir"
    trainer.save_freq=20
    trainer.test_freq=5
    trainer.total_epochs=1
    # trainer.validation_data_dir="$validation_data_dir"
)

ray=(
    +ray_kwargs.ray_init.include_dashboard=False
)

########################### launch ###########################

python -m verl.trainer.main_ppo \
    "${data[@]}" \
    "${model[@]}" \
    "${actor[@]}" \
    "${rollout[@]}" \
    "${agent_loop[@]}" \
    "${trainer[@]}" \
    "${ray[@]}" \
    "$@"
