#!/usr/bin/env bash
set -euo pipefail

source /mnt/workspace/xts/setup_env.sh
activate_torch290

export VERL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
export SKILLRL_ROOT="$(cd "$VERL_ROOT/.." && pwd)/SkillRL"
export PYTHONPATH="$VERL_ROOT:$SKILLRL_ROOT:${PYTHONPATH:-}"
export HYDRA_FULL_ERROR=1
export RAY_DISABLE_GPU_METRICS=1
export RAY_DISABLE_DASHBOARD=1
export RAY_DEDUP_LOGS=1

cd "$VERL_ROOT"

MODEL_PATH=${MODEL_PATH:-/mnt/workspace/xts/models/Qwen3-4B}
DATA_ROOT=${DATA_ROOT:-/mnt/workspace/xts/data/skillrl-agent-loop}
TRAIN_SIZE=${TRAIN_SIZE:-16}
VAL_SIZE=${VAL_SIZE:-16}
MAX_STEPS=${MAX_STEPS:-15}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-4096}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-8192}
MAX_MODEL_LEN=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH + 1))
export SKILLRL_WEBSHOP_MAX_STEPS="$MAX_STEPS"

python examples/skillrl_agent_loop/prepare_agent_data.py \
  --data_source webshop \
  --local_dir "$DATA_ROOT" \
  --train_data_size "$TRAIN_SIZE" \
  --val_data_size "$VAL_SIZE"

python -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  algorithm.use_kl_in_reward=False \
  data.train_files="$DATA_ROOT/webshop/train.parquet" \
  data.val_files="$DATA_ROOT/webshop/test.parquet" \
  data.train_batch_size="$TRAIN_SIZE" \
  data.val_batch_size="$VAL_SIZE" \
  data.max_prompt_length="$MAX_PROMPT_LENGTH" \
  data.max_response_length="$MAX_RESPONSE_LENGTH" \
  data.filter_overlong_prompts=True \
  data.truncation=error \
  data.return_raw_chat=True \
  actor_rollout_ref.model.path="$MODEL_PATH" \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.ppo_mini_batch_size="$TRAIN_SIZE" \
  actor_rollout_ref.actor.use_dynamic_bsz=True \
  actor_rollout_ref.actor.use_kl_loss=False \
  actor_rollout_ref.actor.fsdp_config.param_offload=True \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.mode=async \
  actor_rollout_ref.rollout.n=1 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.max_model_len="$MAX_MODEL_LEN" \
  actor_rollout_ref.rollout.multi_turn.enable=True \
  actor_rollout_ref.rollout.multi_turn.tool_config_path="$VERL_ROOT/examples/skillrl_agent_loop/config/webshop_tool.yaml" \
  actor_rollout_ref.rollout.multi_turn.max_assistant_turns=$((MAX_STEPS + 1)) \
  actor_rollout_ref.rollout.agent.default_agent_loop=skillrl_env_agent \
  actor_rollout_ref.rollout.agent.agent_loop_config_path="$VERL_ROOT/examples/skillrl_agent_loop/config/agent_loop.yaml" \
  reward_model.enable=False \
  trainer.logger='["console"]' \
  trainer.project_name=skillrl_agent_loop \
  trainer.experiment_name=webshop_env_agent \
  trainer.n_gpus_per_node="${NGPUS_PER_NODE:-1}" \
  trainer.nnodes=1 \
  trainer.val_before_train=True \
  trainer.save_freq=20 \
  trainer.test_freq=5 \
  trainer.total_epochs=1 \
  +ray_kwargs.ray_init.include_dashboard=False \
  "$@"

