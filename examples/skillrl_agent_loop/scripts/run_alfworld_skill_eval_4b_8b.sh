#!/usr/bin/env bash
# Run full ALFWorld skill-injected validation for Qwen3-4B and Qwen3-8B sequentially.

set -euo pipefail

source /mnt/data/xts/setup_env.sh
activate_torch290
export ALFWORLD_DATA=/mnt/data/xts/data/alfworld
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
verl_root="$(cd "$script_dir/../../.." && pwd)"
skillrl_root="$verl_root/examples/skillrl_agent_loop"
skill_data_root="$skillrl_root/memory_data"
child_script_src="$script_dir/run_alfworld_agent_loop.sh"
child_script_tmp="$(mktemp "$script_dir/.run_alfworld_agent_loop.XXXXXX.sh")"

cp "$child_script_src" "$child_script_tmp"
bash -n "$child_script_tmp"

cleanup_ray() {
    ray stop --force >/dev/null 2>&1 || true
}

cleanup_all() {
    cleanup_ray
    rm -f "$child_script_tmp"
}

trap cleanup_all EXIT

run_eval() {
    local model_name="$1"
    local model_path="$2"
    local experiment_name="$3"

    echo "Starting ALFWorld skill validation for $model_name"
    cleanup_ray

    MODEL_PATH="$model_path" \
    EXPERIMENT_NAME="$experiment_name" \
    DUMP_ROOT="$verl_root/recipe/skillrl_agent_loop/$experiment_name" \
    DATA_ROOT="${DATA_ROOT:-/mnt/data/xts/data/skillrl-agent-loop}" \
    ALFWORLD_DATA="${ALFWORLD_DATA:-/mnt/data/xts/data/alfworld}" \
    TRAIN_SIZE="${TRAIN_SIZE:-16}" \
    VAL_SIZE="${VAL_SIZE:--1}" \
    MAX_STEPS="${MAX_STEPS:-50}" \
    MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-4096}" \
    MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-27000}" \
    VAL_TEMPERATURE="${VAL_TEMPERATURE:-0.4}" \
    VAL_DO_SAMPLE="${VAL_DO_SAMPLE:-true}" \
    VAL_N="${VAL_N:-4}" \
    NGPUS_PER_NODE="${NGPUS_PER_NODE:-8}" \
    AGENT_NUM_WORKERS="${AGENT_NUM_WORKERS:-${NGPUS_PER_NODE:-8}}" \
    ALFWORLD_EVAL_DATASET="${ALFWORLD_EVAL_DATASET:-eval_out_of_distribution}" \
    INJECT_SKILLS="${INJECT_SKILLS:-false}" \
    SKILL_JSON_PATH="${SKILL_JSON_PATH:-$skill_data_root/alfworld/toolcall_skills_v2.json}" \
    SKILL_TOP_K="${SKILL_TOP_K:-1}" \
    TASK_SPECIFIC_TOP_K="${TASK_SPECIFIC_TOP_K:-2}" \
    MISTAKES_TOP_K="${MISTAKES_TOP_K:-2}" \
    PROJECT_NAME="${PROJECT_NAME:-skillrl_agent_loop}" \
    LOGGER="${LOGGER:-[\"console\",\"wandb\"]}" \
    bash "$child_script_tmp"

    cleanup_ray
    echo "Finished ALFWorld skill validation for $model_name"
}

run_eval "Qwen3-4B" "/mnt/data/xts/models/Qwen3-4B" "4b_base_out_of_distribution"
# run_eval "Qwen3-8B" "/mnt/data/xts/models/Qwen3-8B" "8b_skills_out_of_distribution"
