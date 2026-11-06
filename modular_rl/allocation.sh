#!/usr/bin/env bash
# Runs inside the existing Slime container. Three policy GPUs plus one reward GPU.
set -euo pipefail
if [[ "${MODULAR_RL_ALLOCATION_SNAPSHOT:-}" != "${BASH_SOURCE[0]}" ]]; then
    allocation_snapshot="$(mktemp /tmp/modular-rl-allocation.XXXXXXXX.sh)"
    cp -- "${BASH_SOURCE[0]}" "$allocation_snapshot"
    export MODULAR_RL_ALLOCATION_SNAPSHOT="$allocation_snapshot"
    exec bash "$allocation_snapshot" "$@"
fi
cd "${RLVE_REPO:?}"
source scripts/slurm/common.sh
profile="${MODULAR_RL_PROFILE:-v1}"
[[ "$profile" == v1 || "$profile" == v2 ]] || { echo "Unknown profile $profile" >&2; exit 2; }
output="${1:?run directory}"
mkdir -p "$output"
exec {allocation_lock}>"$output/allocation.lock"
flock -n "$allocation_lock" || { echo "Allocation already owns $output" >&2; exit 1; }
export PYTHONPATH="$RLVE_REPO:/root/Megatron-LM"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=6
export HF_HOME="${MODULAR_RL_HF_HOME:?}"
export HF_HUB_CACHE="$HF_HOME/hub" HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export WANDB_MODE=online
export RLVE_WANDB_PROJECT=modular-mopd-rl-initial
export RLVE_WANDB_TEAM
default_group="modular-smoke-${SLURM_JOB_ID:-local}"
[[ "$profile" != v2 ]] || default_group="modular-v2-${SLURM_JOB_ID:-local}"
export WANDB_GROUP="${MODULAR_RL_GROUP:-$default_group}"
export MODULAR_RL_REWARD_ENDPOINT="$output/reward-endpoint.json"
[[ -n "${WANDB_API_KEY:-}" ]] || { echo 'WANDB_API_KEY is required' >&2; exit 1; }
[[ -f "${MODULAR_RL_DATA:?}/ready.json" ]] || { echo 'Prepare task pools first' >&2; exit 1; }
IFS=, read -r -a gpu_ids <<< "${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
read -r -a tasks <<< "${MODULAR_RL_TASKS:-math code chat}"
reward_gpu=0
for task in "${tasks[@]}"; do
    case "$task" in math|code|if) ;; chat) reward_gpu=1 ;; *) echo "Invalid task $task" >&2; exit 2;; esac
done
share_reward_gpu="${MODULAR_RL_SHARE_REWARD_GPU:-0}"
[[ "$share_reward_gpu" == 0 || "$share_reward_gpu" == 1 ]] || { echo 'Invalid reward GPU sharing mode' >&2; exit 2; }
reward_index="${#tasks[@]}"
extra_reward_gpu="$reward_gpu"
if (( reward_gpu && share_reward_gpu )); then
    for index in "${!tasks[@]}"; do
        [[ "${tasks[index]}" != chat ]] || reward_index="$index"
    done
    extra_reward_gpu=0
fi
(( ${#gpu_ids[@]} == ${#tasks[@]} + extra_reward_gpu )) || {
    echo 'Need one GPU per policy task, plus a reward GPU when chat is included' >&2; exit 1;
}

# Convert each requested model once. Serialize concurrent suite conversions with flock.
for model in ${MODULAR_RL_MODELS:-Qwen3-0.6B Qwen3-1.7B}; do
    case "$model" in Qwen3-0.6B|Qwen3-1.7B|Qwen3-0.6B-Base|Qwen3-1.7B-Base) ;; *) echo "Unsupported model $model" >&2; exit 2;; esac
    ref="$RLVE_SCRATCH/models/${model}_torch_dist"
    exec {conversion_lock}>"$RLVE_SCRATCH/models/${model}.conversion.lock"
    flock "$conversion_lock"
    if [[ ! -f "$ref/latest_checkpointed_iteration.txt" ]]; then
        source "scripts/models/qwen3-${model#Qwen3-}.sh"
        temporary="${ref}.modular-${SLURM_JOB_ID:-local}"
        CUDA_VISIBLE_DEVICES="${gpu_ids[0]}" MASTER_ADDR=127.0.0.1 MASTER_PORT=29597 \
            python3 -u tools/convert_hf_to_torch_dist.py "${MODEL_ARGS[@]}" \
            --hf-checkpoint "$RLVE_SCRATCH/models/$model" --save "$temporary" \
            > "$output/convert-$model.log" 2>&1
        [[ -s "$temporary/latest_checkpointed_iteration.txt" ]]
        [[ ! -e "$ref" ]] || { echo "Incomplete existing checkpoint: $ref" >&2; exit 1; }
        mv "$temporary" "$ref"
    fi
    flock -u "$conversion_lock"
done

service_pid=''
worker_pids=()
cleanup() {
    local pid
    for pid in "${worker_pids[@]}"; do kill "$pid" 2>/dev/null || true; done
    if [[ -n "$service_pid" ]]; then kill "$service_pid" 2>/dev/null || true; fi
}
trap cleanup EXIT INT TERM
# Refuse an old endpoint: the private output directory belongs to this allocation.
if [[ "$profile" == v2 ]]; then
    # A requeued allocation replaces the endpoint of its terminated reward process.
    rm -f -- "$MODULAR_RL_REWARD_ENDPOINT"
else
    [[ ! -e "$MODULAR_RL_REWARD_ENDPOINT" ]] || { echo 'Use a fresh run directory' >&2; exit 1; }
fi
if (( reward_gpu )); then
    CUDA_VISIBLE_DEVICES="${gpu_ids[reward_index]}" python3 -u -m modular_rl.skywork \
        --endpoint-file "$MODULAR_RL_REWARD_ENDPOINT" > "$output/skywork.log" 2>&1 &
    service_pid=$!
    for attempt in $(seq 1 180); do
        [[ ! -e "$MODULAR_RL_REWARD_ENDPOINT" ]] || break
        kill -0 "$service_pid" 2>/dev/null || { cat "$output/skywork.log"; exit 1; }
        sleep 2
    done
    [[ -e "$MODULAR_RL_REWARD_ENDPOINT" ]] || { echo 'Skywork readiness timeout' >&2; exit 1; }
fi

run_lane() (
    index="$1" task="$2"
    export CUDA_VISIBLE_DEVICES="${gpu_ids[index]}"
    export RUN_NUM_CPUS=6 RLVE_RUN_NUM_GPUS=1 RLVE_EXPORT_TRAIN_ROLLOUTS=1
    if [[ "$profile" == v2 ]]; then
        export RLVE_CONFIG=modular_rl/v2/config.yaml RLVE_FINAL_WEIGHTS_ONLY=0
        export RLVE_RESUMABLE_RUN=1 RLVE_CHECKPOINT_INTERVAL="${MODULAR_RL_CHECKPOINT_INTERVAL:-50}" RLVE_CHECKPOINT_KEEP="${MODULAR_RL_CHECKPOINT_KEEP:-2}"
        export RLVE_PRUNE_COMPLETED_CHECKPOINTS=1 RLVE_TRAIN_ENTRYPOINT=modular_rl/v2/driver.py
        export MODULAR_RL_RUNTIME="${MODULAR_RL_RUNTIME:?Pinned math grader runtime is required}"
        export RUN_TIMEOUT="${MODULAR_RL_TIMEOUT:-40000}"
        selection="${MODULAR_RL_SELECTION:?Prepare v2 selections before launch}/$task"
        export RLVE_FIXED_PROMPT_POOL="$selection/train.jsonl"
        export RLVE_EXTRA_ARGS="--num-rollout ${MODULAR_RL_STEPS:-1000} --eval-interval ${MODULAR_RL_EVAL_INTERVAL:-50}"
        response_tokens="${MODULAR_RL_MAX_RESPONSE_TOKENS:-1024}"
        answer_tokens="${MODULAR_RL_ANSWER_FORCING_TOKENS:-128}"
        export RLVE_EXTRA_ARGS="$RLVE_EXTRA_ARGS --rollout-max-response-len $response_tokens --eval-max-response-len $response_tokens --answer-forcing-extra-tokens $answer_tokens"
        export RLVE_EXTRA_ARGS="$RLVE_EXTRA_ARGS ${MODULAR_RL_EXTRA_ARGS:-}"
        if [[ "${MODULAR_RL_EVAL_ONLY:-0}" == 1 ]]; then
            export RLVE_RESUMABLE_RUN=0 RLVE_PRUNE_COMPLETED_CHECKPOINTS=0
            export RLVE_TRAIN_ENTRYPOINT=modular_rl/v2/endpoint.py
            unset RLVE_WANDB_RUN_ID
        fi
    else
        export RLVE_CONFIG=modular_rl/smoke.yaml RLVE_FINAL_WEIGHTS_ONLY=1 RLVE_FINAL_FRESH_RUN=1
        export RUN_TIMEOUT="${MODULAR_RL_TIMEOUT:-2400}"
        python3 -m modular_rl.select --data "$MODULAR_RL_DATA" --output "$output/$task-selection" \
            --task "$task" --model "$RLVE_SCRATCH/models/Qwen3-1.7B" \
            --train-prompts "${MODULAR_RL_TRAIN_PROMPTS:-256}" --val-prompts "${MODULAR_RL_VAL_PROMPTS:-16}"
        selection="$output/$task-selection"
        export RLVE_FIXED_PROMPT_POOL="$selection/train.jsonl"
        export RLVE_EXTRA_ARGS="--num-rollout ${MODULAR_RL_STEPS:-3}"
    fi
    for model in ${MODULAR_RL_MODELS:-Qwen3-0.6B Qwen3-1.7B}; do
        export RLVE_HF_CKPT="$RLVE_SCRATCH/models/$model"
        export RLVE_REF_LOAD="$RLVE_SCRATCH/models/${model}_torch_dist"
        export RLVE_MODEL_ARGS_SCRIPT="scripts/models/qwen3-${model#Qwen3-}.sh"
        export RLVE_RUN_DIR="$output/$model-$task" MODULAR_RL_RUN_DIR="$output/$model-$task"
        run_suffix=''
        [[ "$profile" != v2 ]] || run_suffix=-v2
        export RLVE_WANDB_RUN_NAME="$model-$task$run_suffix" SWEEP_DIR="$output"
        mkdir -p "$RLVE_RUN_DIR"
        echo "Starting $model $task on allocated GPU $index"
        bash scripts/slurm/run_one.sh "$index" "$task" \
            "$selection/val.jsonl" \
            >> "$RLVE_RUN_DIR/train.log" 2>&1
        echo "Completed $model $task"
    done
)

for index in "${!tasks[@]}"; do
    run_lane "$index" "${tasks[index]}" &
    worker_pids+=("$!")
done
result=0
for pid in "${worker_pids[@]}"; do
    if wait "$pid"; then :; else result=1; fi
done
(( result == 0 )) || { echo "A policy lane failed; see $output/*/train.log" >&2; exit "$result"; }
if [[ "${MODULAR_RL_EVAL_ONLY:-0}" == 1 ]]; then
    python3 -m modular_rl.v2.endpoint --report "$output"
elif [[ "$profile" == v2 ]]; then
    python3 -m modular_rl.v2.report --root "$output"
else
    python3 -m modular_rl.report --root "$output"
fi
