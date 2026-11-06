#!/usr/bin/env bash
# Configurable student, frozen-teacher, and Skywork GPU roles.
set -euo pipefail
if [[ "${RLVE_MOPD_LAUNCHER_SNAPSHOT:-}" != "${BASH_SOURCE[0]}" ]]; then
    launcher_snapshot="$(mktemp /tmp/rlve-mopd.XXXXXXXX.sh)"
    cp -- "${BASH_SOURCE[0]}" "$launcher_snapshot"
    export RLVE_MOPD_LAUNCHER_SNAPSHOT="$launcher_snapshot"
    exec bash "$launcher_snapshot" "$@"
fi
cd "${RLVE_REPO:?}"
source scripts/slurm/common.sh
export PYTHONPATH="$RLVE_REPO:/root/Megatron-LM" PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=6
export HF_HOME="${MODULAR_RL_HF_HOME:?}"
export HF_HUB_CACHE="$HF_HOME/hub" HUGGINGFACE_HUB_CACHE="$HF_HOME/hub" HF_DATASETS_CACHE="$HF_HOME/datasets"
export RLVE_WANDB_PROJECT="${MODULAR_RL_WANDB_PROJECT:?}" WANDB_PROJECT="$MODULAR_RL_WANDB_PROJECT"
export WANDB_GROUP=modular-mopd RLVE_WANDB_RUN_NAME="${MODULAR_RL_RUN_NAME:?}"
if [[ "${1:-}" == --check-environment ]]; then
    [[ -d "$RLVE_REPO" && -d "$RLVE_SCRATCH" && -f "$MODULAR_RL_TEACHER_BANK/bank.json" ]]
    python3 -m modular_rl.initializers "$MODULAR_RL_INITIALIZERS" "${MODULAR_RL_INITIALIZER:-base}" \
        --expected-sha256 "$MODULAR_RL_INITIALIZERS_SHA256" --paths >/dev/null
    echo "Verified MOPD container environment: $MODULAR_RL_RUN_NAME"
    exit 0
fi
IFS=, read -r -a mopd_devices <<< "${CUDA_VISIBLE_DEVICES:?}"
reward_gpus="${MODULAR_RL_REWARD_GPUS:-2}"
policy_gpus="${MODULAR_RL_POLICY_GPUS:-4}"
teacher_gpus="${MODULAR_RL_TEACHER_GPUS:-2}"
[[ "$reward_gpus" == 1 || "$reward_gpus" == 2 ]]
[[ "$policy_gpus" == 2 || "$policy_gpus" == 4 ]]
[[ "$teacher_gpus" == 1 || "$teacher_gpus" == 2 ]]
main_gpus=$((policy_gpus + teacher_gpus))
[[ "${#mopd_devices[@]}" == $((main_gpus + reward_gpus)) ]]
mapfile -t initializer_paths < <(python3 -m modular_rl.initializers \
    "$MODULAR_RL_INITIALIZERS" "${MODULAR_RL_INITIALIZER:-base}" --expected-sha256 "$MODULAR_RL_INITIALIZERS_SHA256" --paths)
[[ "${#initializer_paths[@]}" == 2 ]]
export RLVE_HF_CKPT="${initializer_paths[0]}" RLVE_REF_LOAD="${initializer_paths[1]}"
export RLVE_MODEL_ARGS_SCRIPT=scripts/models/qwen3-0.6B-Base.sh
export RLVE_RUN_DIR="$MODULAR_RL_OUTPUT/policy" MODULAR_RL_RUN_DIR="$MODULAR_RL_OUTPUT/policy"
mkdir -p "$RLVE_RUN_DIR"
export MODULAR_RL_REWARD_ENDPOINT="$MODULAR_RL_OUTPUT/reward-endpoint.json"
rm -f -- "$MODULAR_RL_REWARD_ENDPOINT"
reward_devices="$(IFS=,; echo "${mopd_devices[*]:main_gpus:reward_gpus}")"
CUDA_VISIBLE_DEVICES="$reward_devices" python3 -u -m modular_rl.skywork \
    --replicas "$reward_gpus" --endpoint-file "$MODULAR_RL_REWARD_ENDPOINT" >> "$MODULAR_RL_OUTPUT/skywork.log" 2>&1 &
reward_pid=$!
cleanup() { kill "$reward_pid" 2>/dev/null || true; }
trap cleanup EXIT INT TERM
for attempt in $(seq 1 180); do
    [[ ! -e "$MODULAR_RL_REWARD_ENDPOINT" ]] || break
    kill -0 "$reward_pid" 2>/dev/null || { tail -n 80 "$MODULAR_RL_OUTPUT/skywork.log"; exit 1; }
    sleep 2
done
[[ -s "$MODULAR_RL_REWARD_ENDPOINT" ]]
export CUDA_VISIBLE_DEVICES
CUDA_VISIBLE_DEVICES="$(IFS=,; echo "${mopd_devices[*]:0:main_gpus}")"
export RUN_NUM_CPUS=$((6 * main_gpus)) RLVE_RUN_NUM_GPUS="$main_gpus" RLVE_EXPORT_TRAIN_ROLLOUTS=1
export RLVE_CONFIG=modular_rl/mopd-config.yaml RLVE_FINAL_WEIGHTS_ONLY=0
export RLVE_RESUMABLE_RUN=1 RLVE_CHECKPOINT_INTERVAL="${MODULAR_RL_CHECKPOINT_INTERVAL:-64}" RLVE_CHECKPOINT_KEEP=1
export RLVE_PRUNE_COMPLETED_CHECKPOINTS=1 RLVE_TRAIN_ENTRYPOINT=modular_rl/mopd_runtime.py
export RLVE_FIXED_PROMPT_POOL="$MODULAR_RL_SELECTION/mixed-train.jsonl"
export RLVE_EXTRA_ARGS="${MODULAR_RL_EXTRA_ARGS:?}" SWEEP_DIR="$MODULAR_RL_OUTPUT" RUN_TIMEOUT=84600
bash scripts/slurm/run_one.sh 0 mixed "$MODULAR_RL_SELECTION/mixed-val.jsonl" >> "$RLVE_RUN_DIR/train.log" 2>&1
