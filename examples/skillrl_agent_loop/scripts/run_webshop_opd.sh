#!/usr/bin/env bash
# WebShop on-policy distillation | multi-turn agent loop | vLLM rollout | FSDP student

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
export NCCL_DEBUG=WARN

cd "$verl_root"

if [[ -n "${WANDB_API_KEY:-}" ]]; then
    wandb login --relogin "$WANDB_API_KEY"
fi

########################### user settings ###########################

project_name=skill-v1-webshop
experiment_name=8b_opd

student_model=/mnt/data/xts/models/Qwen3-8B
teacher_model=/mnt/data/xts/models/Qwen3-8B
data_root=/mnt/data/xts/data/skillrl-agent-loop
webshop_data_root=/mnt/data/xts/data/webshop

resume_mode=disable
resume_dir=/mnt/data/xts/skill/verl/checkpoints/skill-v1-webshop/webshop_4b_opd/global_step_20

student_world_size=6
teacher_world_size=2
nnodes=1

train_data_size=-1
val_data_size=-1
train_batch_size=30

max_steps=20
max_tokens_per_step=1024
max_prompt_length=4096
max_response_length=27000
max_model_len=$((max_prompt_length + max_response_length + 1))
max_token_len_per_gpu=32768
# Use null to disable tool-response character truncation.
max_tool_response_length=null

# WebShop official goal split:
#   test:     indexes 0..499
#   test_128: 128 evenly spaced indexes sampled from test 0..499
#   valid:    indexes 500..1499 (called eval in the original WebShop code)
#   train:    indexes 1500..end
webshop_eval_dataset=test_128

student_inject_skills=false
student_skill_json_path=

teacher_prompt_mode=skill_injected
teacher_prompt_template_path=
teacher_skill_json_path="$skill_data_root/webshop/claude_style_skills_v2.json"
teacher_skill_top_k=6
teacher_task_specific_top_k=null
teacher_mistakes_top_k=5

distillation_loss_mode=backward_kl_topk
distillation_topk=128
distillation_use_policy_gradient=false
distillation_loss_max_clamp=10.0
distillation_log_prob_min_clamp=-10.0
distillation_use_response_mask_for_topk_kl=true

actor_lr=1e-6
actor_calculate_entropy=true
rollout_gpu_memory_utilization=0.85
teacher_gpu_memory_utilization=0.5
rollout_tensor_parallel_size=1
teacher_tensor_parallel_size=1

val_temperature=0.4
val_do_sample=true
val_n=4
val_over_sample_rate=0.0
val_shuffle=true

total_epochs=2
save_freq=20
test_freq=20
log_val_generations=10

logger='["console","wandb"]'

########################### submission overrides ###########################

project_name="${PROJECT_NAME:-$project_name}"
experiment_name="${EXPERIMENT_NAME:-$experiment_name}"
student_model="${STUDENT_MODEL:-$student_model}"
teacher_model="${TEACHER_MODEL:-$teacher_model}"
data_root="${DATA_ROOT:-$data_root}"
webshop_data_root="${WEBSHOP_DATA_ROOT:-$webshop_data_root}"
resume_mode="${RESUME_MODE:-$resume_mode}"
resume_dir="${RESUME_DIR:-$resume_dir}"
student_world_size="${STUDENT_WORLD_SIZE:-$student_world_size}"
teacher_world_size="${TEACHER_WORLD_SIZE:-$teacher_world_size}"
nnodes="${NNODES:-$nnodes}"
train_data_size="${TRAIN_SIZE:-$train_data_size}"
val_data_size="${VAL_SIZE:-$val_data_size}"
train_batch_size="${TRAIN_BATCH_SIZE:-$train_batch_size}"
max_steps="${MAX_STEPS:-$max_steps}"
max_tokens_per_step="${MAX_TOKENS_PER_STEP:-$max_tokens_per_step}"
max_prompt_length="${MAX_PROMPT_LENGTH:-$max_prompt_length}"
max_response_length="${MAX_RESPONSE_LENGTH:-$max_response_length}"
if [[ -n "${MAX_MODEL_LEN:-}" ]]; then
    max_model_len="$MAX_MODEL_LEN"
else
    max_model_len=$((max_prompt_length + max_response_length + 1))
fi
max_token_len_per_gpu="${MAX_TOKEN_LEN_PER_GPU:-$max_token_len_per_gpu}"
max_tool_response_length="${MAX_TOOL_RESPONSE_LENGTH:-$max_tool_response_length}"
webshop_eval_dataset="${WEBSHOP_EVAL_DATASET:-$webshop_eval_dataset}"
student_inject_skills="${STUDENT_INJECT_SKILLS:-$student_inject_skills}"
student_skill_json_path="${STUDENT_SKILL_JSON_PATH:-$student_skill_json_path}"
teacher_prompt_mode="${TEACHER_PROMPT_MODE:-$teacher_prompt_mode}"
teacher_prompt_template_path="${TEACHER_PROMPT_TEMPLATE_PATH:-$teacher_prompt_template_path}"
teacher_skill_json_path="${TEACHER_SKILL_JSON_PATH:-$teacher_skill_json_path}"
teacher_skill_top_k="${TEACHER_SKILL_TOP_K:-$teacher_skill_top_k}"
teacher_task_specific_top_k="${TEACHER_TASK_SPECIFIC_TOP_K:-$teacher_task_specific_top_k}"
teacher_mistakes_top_k="${TEACHER_MISTAKES_TOP_K:-$teacher_mistakes_top_k}"
distillation_loss_mode="${DISTILLATION_LOSS_MODE:-$distillation_loss_mode}"
distillation_topk="${DISTILLATION_TOPK:-$distillation_topk}"
distillation_use_policy_gradient="${DISTILLATION_USE_POLICY_GRADIENT:-$distillation_use_policy_gradient}"
distillation_loss_max_clamp="${DISTILLATION_LOSS_MAX_CLAMP:-$distillation_loss_max_clamp}"
distillation_log_prob_min_clamp="${DISTILLATION_LOG_PROB_MIN_CLAMP:-$distillation_log_prob_min_clamp}"
distillation_use_response_mask_for_topk_kl="${DISTILLATION_USE_RESPONSE_MASK_FOR_TOPK_KL:-$distillation_use_response_mask_for_topk_kl}"
actor_lr="${ACTOR_LR:-$actor_lr}"
actor_calculate_entropy="${ACTOR_CALCULATE_ENTROPY:-$actor_calculate_entropy}"
rollout_gpu_memory_utilization="${ROLLOUT_GPU_MEMORY_UTILIZATION:-$rollout_gpu_memory_utilization}"
teacher_gpu_memory_utilization="${TEACHER_GPU_MEMORY_UTILIZATION:-$teacher_gpu_memory_utilization}"
rollout_tensor_parallel_size="${ROLLOUT_TENSOR_PARALLEL_SIZE:-$rollout_tensor_parallel_size}"
teacher_tensor_parallel_size="${TEACHER_TENSOR_PARALLEL_SIZE:-$teacher_tensor_parallel_size}"
val_temperature="${VAL_TEMPERATURE:-$val_temperature}"
val_do_sample="${VAL_DO_SAMPLE:-$val_do_sample}"
val_n="${VAL_N:-$val_n}"
val_over_sample_rate="${VAL_OVER_SAMPLE_RATE:-$val_over_sample_rate}"
val_shuffle="${VAL_SHUFFLE:-$val_shuffle}"
total_epochs="${TOTAL_EPOCHS:-$total_epochs}"
save_freq="${SAVE_FREQ:-$save_freq}"
test_freq="${TEST_FREQ:-$test_freq}"
log_val_generations="${LOG_VAL_GENERATIONS:-$log_val_generations}"
logger="${LOGGER:-$logger}"

for arg in "$@"; do
    case "$arg" in
        trainer.experiment_name=*)
            experiment_name="${arg#trainer.experiment_name=}"
            ;;
    esac
done

########################### derived paths ###########################

train_file="$data_root/webshop/train.parquet"
val_file="$data_root/webshop/test.parquet"
webshop_package_root="$skillrl_root/agent_system/environments/env_package/webshop/webshop"
tool_config_path="$skillrl_root/config/webshop_tool.yaml"
agent_loop_config_path="$skillrl_root/config/agent_loop.yaml"

webshop_file_path="$webshop_data_root/data/items_shuffle_1000.json"
webshop_attr_path="$webshop_data_root/data/items_ins_v2_1000.json"
webshop_human_attr_path="$webshop_data_root/data/items_human_ins.json"
webshop_search_engine_root="$webshop_data_root/search_engine"

export SKILLRL_WEBSHOP_MAX_STEPS="$max_steps"
export SKILLRL_MAX_TOKENS_PER_STEP="$max_tokens_per_step"
export SKILLRL_WEBSHOP_INJECT_SKILLS="$student_inject_skills"
export SKILLRL_WEBSHOP_SKILL_JSON_PATH="$student_skill_json_path"
export WEBSHOP_FILE_PATH="$webshop_file_path"
export WEBSHOP_ATTR_PATH="$webshop_attr_path"
export WEBSHOP_HUMAN_ATTR_PATH="$webshop_human_attr_path"
export WEBSHOP_SEARCH_ENGINE_ROOT="$webshop_search_engine_root"

echo "WebShop OPD student model: $student_model"
echo "WebShop OPD teacher model: $teacher_model"
echo "WebShop package root: $webshop_package_root"
echo "WebShop data root: $webshop_data_root"
echo "WebShop search engine root: $webshop_search_engine_root"
echo "WebShop eval dataset: $webshop_eval_dataset"
echo "WebShop student inject skills: $student_inject_skills"
echo "WebShop teacher prompt mode: $teacher_prompt_mode"
echo "WebShop teacher skill file: $teacher_skill_json_path"
echo "WebShop teacher skill top-k: skill=$teacher_skill_top_k task_specific=$teacher_task_specific_top_k mistakes=$teacher_mistakes_top_k"
echo "WebShop train/val size: $train_data_size/$val_data_size batch=$train_batch_size"
echo "WebShop max steps: $max_steps"
echo "WebShop max tokens per step: $max_tokens_per_step"
echo "WebShop max tool response length: $max_tool_response_length"
echo "WebShop validation sampling: temperature=$val_temperature do_sample=$val_do_sample n=$val_n shuffle=$val_shuffle oversample=$val_over_sample_rate"

########################### prepare data ###########################

python examples/skillrl_agent_loop/prepare_agent_data.py \
    --data_source webshop \
    --local_dir "$data_root" \
    --train_data_size "$train_data_size" \
    --val_data_size "$val_data_size" \
    --webshop_root "$webshop_data_root" \
    --webshop_eval_dataset "$webshop_eval_dataset"

########################### parameter arrays ###########################

data=(
    algorithm.adv_estimator=grpo
    algorithm.rollout_correction.bypass_mode=True
    algorithm.use_kl_in_reward=False
    data.train_files="$train_file"
    data.val_files="$val_file"
    data.train_batch_size="$train_batch_size"
    data.max_prompt_length="$max_prompt_length"
    data.max_response_length="$max_response_length"
    data.filter_overlong_prompts=True
    data.truncation=error
    data.return_raw_chat=True
)

model=(
    actor_rollout_ref.model.path="$student_model"
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.model.enable_gradient_checkpointing=True
)

actor=(
    actor_rollout_ref.actor.optim.lr="$actor_lr"
    actor_rollout_ref.actor.ppo_mini_batch_size="$train_batch_size"
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu="$max_token_len_per_gpu"
    actor_rollout_ref.actor.use_dynamic_bsz=True
    actor_rollout_ref.actor.use_kl_loss=False
    actor_rollout_ref.actor.calculate_entropy="$actor_calculate_entropy"
    actor_rollout_ref.actor.fsdp_config.dtype=float16
    actor_rollout_ref.actor.fsdp_config.param_offload=True
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True
)

rollout=(
    actor_rollout_ref.rollout.name=vllm
    actor_rollout_ref.rollout.mode=async
    actor_rollout_ref.rollout.dtype=float16
    actor_rollout_ref.rollout.n=1
    actor_rollout_ref.rollout.val_kwargs.n="$val_n"
    actor_rollout_ref.rollout.val_kwargs.temperature="$val_temperature"
    actor_rollout_ref.rollout.val_kwargs.do_sample="$val_do_sample"
    actor_rollout_ref.rollout.val_over_sample_rate="$val_over_sample_rate"
    actor_rollout_ref.rollout.val_shuffle="$val_shuffle"
    actor_rollout_ref.rollout.gpu_memory_utilization="$rollout_gpu_memory_utilization"
    actor_rollout_ref.rollout.calculate_log_probs=True
    actor_rollout_ref.rollout.tensor_model_parallel_size="$rollout_tensor_parallel_size"
    actor_rollout_ref.rollout.max_model_len="$max_model_len"
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu="$max_token_len_per_gpu"
)

agent_loop=(
    actor_rollout_ref.rollout.multi_turn.enable=True
    actor_rollout_ref.rollout.multi_turn.tool_config_path="$tool_config_path"
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=$((max_steps + 1))
    actor_rollout_ref.rollout.multi_turn.max_tool_response_length="$max_tool_response_length"
    actor_rollout_ref.rollout.agent.default_agent_loop=skillrl_env_agent
    actor_rollout_ref.rollout.agent.num_workers="$student_world_size"
    actor_rollout_ref.rollout.agent.agent_loop_config_path="$agent_loop_config_path"
)

distillation=(
    reward_model.enable=False
    distillation.enabled=True
    distillation.n_gpus_per_node="$teacher_world_size"
    distillation.nnodes="$nnodes"
    distillation.teacher_models.teacher_model.model_path="$teacher_model"
    distillation.teacher_models.teacher_model.inference.name=vllm
    distillation.teacher_models.teacher_model.inference.tensor_model_parallel_size="$teacher_tensor_parallel_size"
    distillation.teacher_models.teacher_model.inference.gpu_memory_utilization="$teacher_gpu_memory_utilization"
    distillation.teacher_models.teacher_model.inference.max_model_len="$max_model_len"
    distillation.teacher_prompt.mode="$teacher_prompt_mode"
    distillation.teacher_prompt.template_path="$teacher_prompt_template_path"
    distillation.teacher_prompt.skill_json_path="$teacher_skill_json_path"
    distillation.teacher_prompt.skill_top_k="$teacher_skill_top_k"
    distillation.teacher_prompt.task_specific_top_k="$teacher_task_specific_top_k"
    distillation.teacher_prompt.mistakes_top_k="$teacher_mistakes_top_k"
    distillation.distillation_loss.loss_mode="$distillation_loss_mode"
    distillation.distillation_loss.topk="$distillation_topk"
    distillation.distillation_loss.use_task_rewards=False
    distillation.distillation_loss.use_policy_gradient="$distillation_use_policy_gradient"
    distillation.distillation_loss.loss_max_clamp="$distillation_loss_max_clamp"
    distillation.distillation_loss.log_prob_min_clamp="$distillation_log_prob_min_clamp"
    distillation.distillation_loss.use_response_mask_for_topk_kl="$distillation_use_response_mask_for_topk_kl"
)

trainer=(
    trainer.logger="$logger"
    trainer.project_name="$project_name"
    trainer.experiment_name="$experiment_name"
    trainer.n_gpus_per_node="$student_world_size"
    trainer.nnodes="$nnodes"
    trainer.val_before_train=False
    trainer.log_val_generations="$log_val_generations"
    trainer.save_freq="$save_freq"
    trainer.test_freq="$test_freq"
    trainer.total_epochs="$total_epochs"
    trainer.resume_mode="$resume_mode"
    # trainer.resume_from_path=$resume_dir
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
    "${distillation[@]}" \
    "${trainer[@]}" \
    "${ray[@]}" \
    "$@"
