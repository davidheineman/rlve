#!/usr/bin/env bash
# One policy, with independent Skywork replicas for chat.
set -euo pipefail
cd "${RLVE_REPO:?}"
source scripts/slurm/common.sh
export RLVE_REPO RLVE_SCRATCH RLVE_IMAGE RLVE_WANDB_TEAM
export PYTHONPATH="$RLVE_REPO:/root/Megatron-LM"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=6
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-garbage_collection_threshold:0.7,max_split_size_mb:128}"
export HF_HOME="${MODULAR_RL_HF_HOME:?}"
export HF_HUB_CACHE="$HF_HOME/hub" HUGGINGFACE_HUB_CACHE="$HF_HOME/hub" HF_DATASETS_CACHE="$HF_HOME/datasets"
export RLVE_WANDB_PROJECT="${MODULAR_RL_WANDB_PROJECT:?}" WANDB_PROJECT="$MODULAR_RL_WANDB_PROJECT"
export WANDB_GROUP="${MODULAR_RL_GROUP:-rl-v1}" RLVE_WANDB_RUN_NAME="${MODULAR_RL_RUN_NAME:?}"
if [[ "${1:-}" == --check-environment ]]; then
    bash -c 'set -eu; : "${RLVE_REPO:?}" "${RLVE_SCRATCH:?}" "${RLVE_IMAGE:?}" "${RLVE_WANDB_TEAM:?}" "${RLVE_WANDB_PROJECT:?}"; [[ -d "$RLVE_REPO" && -d "$RLVE_SCRATCH" ]]'
    echo "Verified container environment: $MODULAR_RL_RUN_NAME"
    exit 0
fi
IFS=, read -r -a suite_gpu_ids <<< "${CUDA_VISIBLE_DEVICES:?}"
task="${MODULAR_RL_TASKS:?}"
policy_gpus="${MODULAR_RL_POLICY_GPUS:-1}"
reward_gpus="${MODULAR_RL_REWARD_GPUS:-1}"
[[ "$policy_gpus" == 1 || "$policy_gpus" == 2 || "$policy_gpus" == 4 ]]
[[ "$reward_gpus" == 1 || "$reward_gpus" == 2 ]]
expected_cards="$policy_gpus"
[[ "$task" != chat ]] || expected_cards=$((policy_gpus + reward_gpus))
[[ "${#suite_gpu_ids[@]}" == "$expected_cards" ]]
mapfile -t initializer_paths < <(python3 -m modular_rl.initializers \
    "$MODULAR_RL_INITIALIZERS" "$MODULAR_RL_INITIALIZER" \
    --expected-sha256 "$MODULAR_RL_INITIALIZERS_SHA256" --paths)
[[ "${#initializer_paths[@]}" == 2 ]]
export RLVE_HF_CKPT="${initializer_paths[0]}" RLVE_REF_LOAD="${initializer_paths[1]}"
export RLVE_MODEL_ARGS_SCRIPT=scripts/models/qwen3-0.6B-Base.sh
export RLVE_RUN_DIR="${MODULAR_RL_OUTPUT:?}/policy" MODULAR_RL_RUN_DIR="$MODULAR_RL_OUTPUT/policy"
mkdir -p "$RLVE_RUN_DIR"
export MODULAR_RL_REWARD_ENDPOINT="$MODULAR_RL_OUTPUT/reward-endpoint.json"
reward_pid=''
cleanup() { [[ -z "$reward_pid" ]] || kill "$reward_pid" 2>/dev/null || true; }
trap cleanup EXIT INT TERM
if [[ "$task" == chat ]]; then
    rm -f -- "$MODULAR_RL_REWARD_ENDPOINT"
    reward_devices="$(IFS=,; echo "${suite_gpu_ids[*]:policy_gpus:reward_gpus}")"
    CUDA_VISIBLE_DEVICES="$reward_devices" python3 -u -m modular_rl.skywork \
        --replicas "$reward_gpus" --endpoint-file "$MODULAR_RL_REWARD_ENDPOINT" > "$MODULAR_RL_OUTPUT/skywork.log" 2>&1 &
    reward_pid=$!
    for attempt in $(seq 1 180); do
        [[ ! -e "$MODULAR_RL_REWARD_ENDPOINT" ]] || break
        kill -0 "$reward_pid" 2>/dev/null || { cat "$MODULAR_RL_OUTPUT/skywork.log"; exit 1; }
        sleep 2
    done
    [[ -s "$MODULAR_RL_REWARD_ENDPOINT" ]]
fi
export CUDA_VISIBLE_DEVICES
CUDA_VISIBLE_DEVICES="$(IFS=,; echo "${suite_gpu_ids[*]:0:policy_gpus}")"
export RUN_NUM_CPUS=$((6 * policy_gpus)) RLVE_RUN_NUM_GPUS="$policy_gpus" RLVE_EXPORT_TRAIN_ROLLOUTS=1
export RLVE_CONFIG=modular_rl/v2/config.yaml RLVE_FINAL_WEIGHTS_ONLY=0
export RLVE_RESUMABLE_RUN=1 RLVE_CHECKPOINT_INTERVAL="${MODULAR_RL_CHECKPOINT_INTERVAL:-50}" RLVE_CHECKPOINT_KEEP=1
export RLVE_PRUNE_COMPLETED_CHECKPOINTS=1 RLVE_TRAIN_ENTRYPOINT=modular_rl/v2/driver.py
export RLVE_FIXED_PROMPT_POOL="$MODULAR_RL_SELECTION/$task/train.jsonl"
export RLVE_EXTRA_ARGS="${MODULAR_RL_EXTRA_ARGS:?}" SWEEP_DIR="$MODULAR_RL_OUTPUT"
export RUN_TIMEOUT=56400
bash scripts/slurm/run_one.sh 0 "$task" "$MODULAR_RL_SELECTION/$task/val.jsonl" \
    >> "$RLVE_RUN_DIR/train.log" 2>&1
python3 -m modular_rl.v2.report --root "$MODULAR_RL_OUTPUT"
