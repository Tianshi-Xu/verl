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
MAX_STEPS=${MAX_STEPS:-50}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-4096}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-27000}
MAX_MODEL_LEN=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH + 1))
VAL_TEMPERATURE=${VAL_TEMPERATURE:-0.4}
VAL_DO_SAMPLE=${VAL_DO_SAMPLE:-true}
VAL_N=${VAL_N:-4}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-8}
VAL_ONLY=${VAL_ONLY:-true}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-alfworld_env_agent}
for arg in "$@"; do
  case "$arg" in
    trainer.experiment_name=*)
      EXPERIMENT_NAME="${arg#trainer.experiment_name=}"
      ;;
  esac
done
DUMP_ROOT=${DUMP_ROOT:-"$VERL_ROOT/recipe/skillrl_agent_loop/$EXPERIMENT_NAME"}
ROLLOUT_DATA_DIR=${ROLLOUT_DATA_DIR:-"$DUMP_ROOT/rollout"}
VALIDATION_DATA_DIR=${VALIDATION_DATA_DIR:-"$DUMP_ROOT/validation"}
export SKILLRL_ALFWORLD_MAX_STEPS="$MAX_STEPS"

mkdir -p "$ROLLOUT_DATA_DIR" "$VALIDATION_DATA_DIR"
echo "ALFWorld rollout dump dir: $ROLLOUT_DATA_DIR"
echo "ALFWorld validation dump dir: $VALIDATION_DATA_DIR"

python examples/skillrl_agent_loop/prepare_agent_data.py \
  --data_source alfworld \
  --local_dir "$DATA_ROOT" \
  --train_data_size "$TRAIN_SIZE" \
  --val_data_size "$VAL_SIZE" \
  --alfworld_config_path "$SKILLRL_ROOT/agent_system/environments/env_package/alfworld/configs/config_tw.yaml"

python -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  algorithm.use_kl_in_reward=False \
  data.train_files="$DATA_ROOT/alfworld/train.parquet" \
  data.val_files="$DATA_ROOT/alfworld/test.parquet" \
  data.train_batch_size="$TRAIN_SIZE" \
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
  actor_rollout_ref.actor.fsdp_config.dtype=float16 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.85 \
  actor_rollout_ref.rollout.dtype=float16 \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.mode=async \
  actor_rollout_ref.rollout.n=1 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.max_model_len="$MAX_MODEL_LEN" \
  actor_rollout_ref.rollout.val_kwargs.temperature="$VAL_TEMPERATURE" \
  actor_rollout_ref.rollout.val_kwargs.do_sample="$VAL_DO_SAMPLE" \
  actor_rollout_ref.rollout.val_kwargs.n="$VAL_N" \
  actor_rollout_ref.rollout.multi_turn.enable=True \
  actor_rollout_ref.rollout.multi_turn.tool_config_path="$VERL_ROOT/examples/skillrl_agent_loop/config/alfworld_tool.yaml" \
  actor_rollout_ref.rollout.multi_turn.max_assistant_turns=$((MAX_STEPS + 1)) \
  actor_rollout_ref.rollout.agent.default_agent_loop=skillrl_env_agent \
  actor_rollout_ref.rollout.agent.agent_loop_config_path="$VERL_ROOT/examples/skillrl_agent_loop/config/agent_loop.yaml" \
  reward_model.enable=False \
  trainer.logger='["console"]' \
  trainer.project_name=skillrl_agent_loop \
  trainer.experiment_name="$EXPERIMENT_NAME" \
  trainer.n_gpus_per_node="$NGPUS_PER_NODE" \
  trainer.nnodes=1 \
  trainer.val_before_train=True \
  trainer.val_only="$VAL_ONLY" \
  trainer.rollout_data_dir="$ROLLOUT_DATA_DIR" \
  trainer.validation_data_dir="$VALIDATION_DATA_DIR" \
  trainer.save_freq=20 \
  trainer.test_freq=5 \
  trainer.total_epochs=1 \
  +ray_kwargs.ray_init.include_dashboard=False \
  "$@"
