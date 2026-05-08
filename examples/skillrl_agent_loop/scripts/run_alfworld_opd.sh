#!/usr/bin/env bash
# ALFWorld on-policy distillation | multi-turn agent loop | vLLM rollout | FSDP student

set -euo pipefail

# source /mnt/workspace/xts/setup_env.sh
# activate_torch290

verl_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
skillrl_root="$(cd "$verl_root/.." && pwd)/SkillRL"

export VERL_ROOT="$verl_root"
export SKILLRL_ROOT="$skillrl_root"
export PYTHONPATH="$verl_root:$skillrl_root"
export HYDRA_FULL_ERROR=1
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export RAY_DISABLE_GPU_METRICS=1
export RAY_DISABLE_DASHBOARD=1
export RAY_DEDUP_LOGS=1

cd "$verl_root"

########################### user settings ###########################

student_model=/mnt/workspace/xts/models/Qwen3-4B
teacher_model=/mnt/workspace/xts/models/Qwen3-4B
data_root=/mnt/workspace/xts/data/skillrl-agent-loop

student_world_size=7
teacher_world_size=1
nnodes=1

train_data_size=-1
val_data_size=-1
train_batch_size=28

max_steps=50
max_prompt_length=4096
max_response_length=27000
max_model_len=$((max_prompt_length + max_response_length + 1))
max_token_len_per_gpu=32768

student_inject_skills=false
student_skill_json_path=

teacher_prompt_mode=skill_injected
teacher_prompt_template_path=
teacher_skill_json_path=/mnt/workspace/xts/skill/SkillRL/memory_data/alfworld/toolcall_skills_v2.json
teacher_skill_top_k=1
teacher_task_specific_top_k=2
teacher_mistakes_top_k=2

distillation_loss_mode=backward_kl_topk
distillation_topk=128
distillation_use_policy_gradient=false
distillation_loss_max_clamp=10.0
distillation_log_prob_min_clamp=-10.0
distillation_use_response_mask_for_topk_kl=true

actor_lr=1e-6
rollout_gpu_memory_utilization=0.80
teacher_gpu_memory_utilization=0.3
rollout_tensor_parallel_size=1
teacher_tensor_parallel_size=1

total_epochs=2
save_freq=20
test_freq=10
log_val_generations=10

project_name=skillrl_agent_loop
experiment_name=alfworld_env_agent_opd_backward_kl

########################### derived paths ###########################

train_file="$data_root/alfworld/train.parquet"
val_file="$data_root/alfworld/test.parquet"
alfworld_config_path="$skillrl_root/agent_system/environments/env_package/alfworld/configs/config_tw.yaml"
tool_config_path="$verl_root/examples/skillrl_agent_loop/config/alfworld_tool.yaml"
agent_loop_config_path="$verl_root/examples/skillrl_agent_loop/config/agent_loop.yaml"

export SKILLRL_ALFWORLD_MAX_STEPS="$max_steps"
export SKILLRL_ALFWORLD_INJECT_SKILLS="$student_inject_skills"
export SKILLRL_ALFWORLD_SKILL_JSON_PATH="$student_skill_json_path"

########################### prepare data ###########################

python examples/skillrl_agent_loop/prepare_agent_data.py \
    --data_source alfworld \
    --local_dir "$data_root" \
    --train_data_size "$train_data_size" \
    --val_data_size "$val_data_size" \
    --alfworld_config_path "$alfworld_config_path"

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
    actor_rollout_ref.actor.ppo_mini_batch_size=28
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu="$max_token_len_per_gpu"
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
    actor_rollout_ref.rollout.val_kwargs.n=1
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
    trainer.logger='["console","wandb"]'
    trainer.project_name="$project_name"
    trainer.experiment_name="$experiment_name"
    trainer.n_gpus_per_node="$student_world_size"
    trainer.nnodes="$nnodes"
    trainer.val_before_train=False
    trainer.log_val_generations="$log_val_generations"
    trainer.save_freq="$save_freq"
    trainer.test_freq="$test_freq"
    trainer.total_epochs="$total_epochs"
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
