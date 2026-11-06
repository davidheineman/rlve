#!/bin/bash
# One-time setup for the RLVE launchers: container image, model weights (HF + megatron),
# and the in-loop eval sets. Safe to re-run; every step is skipped if its output exists.
#
#   bash scripts/setup/prepare.sh
#
# The weight conversion needs a GPU. On a login node this script grabs one with srun;
# inside an existing allocation it just runs there.
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/slurm/common.sh"
cd "$RLVE_REPO"

mkdir -p "$RLVE_SCRATCH"/{images,models,cache,runs,apptainer_cache,tmp} "$RLVE_EVAL_DIR"

# 1. Container image (~25 GB; takes a while the first time). Unpacking wants ~60 GB of
#    temp space, and mksquashfs takes a quarter of physical RAM, so this step needs a
#    roomy node - on a small login-node cgroup it gets OOM-killed part way through.
if [[ ! -f "$RLVE_IMAGE" ]]; then
    echo "==> pulling $RLVE_DOCKER_URI"
    if [[ -z "${APPTAINER_TMPDIR:-}" ]]; then
        # Node-local disk is several times faster than NFS here; fall back to scratch.
        local_tmp="/tmp/$USER/apptainer-tmp"
        mkdir -p "$local_tmp"
        avail=$(df --output=avail -BG "$local_tmp" | tail -1 | tr -dc '0-9')
        if (( avail > 120 )); then APPTAINER_TMPDIR="$local_tmp"; else APPTAINER_TMPDIR="$RLVE_SCRATCH/tmp"; fi
    fi
    echo "    unpacking in $APPTAINER_TMPDIR"
    APPTAINER_CACHEDIR="${APPTAINER_CACHEDIR:-$RLVE_SCRATCH/apptainer_cache}" \
    APPTAINER_TMPDIR="$APPTAINER_TMPDIR" \
        apptainer pull "$RLVE_IMAGE" "$RLVE_DOCKER_URI"
fi

# 2. HF weights - read by sglang and by every tokenizer call.
if [[ ! -f "$RLVE_HF_CKPT/config.json" ]]; then
    echo "==> downloading $RLVE_MODEL_REPO"
    HF_HOME="$RLVE_SCRATCH/cache/hf" rlve_apptainer_cpu python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('$RLVE_MODEL_REPO', local_dir='$RLVE_HF_CKPT')"
fi

# 3. Megatron weights. A failed conversion does not raise: training would silently start
#    from random weights, so check the tracker file afterwards.
if [[ ! -f "$RLVE_REF_LOAD/latest_checkpointed_iteration.txt" ]]; then
    echo "==> converting to megatron torch_dist format"
    convert="source $RLVE_MODEL_ARGS_SCRIPT && \
        PYTHONPATH=/root/Megatron-LM:$RLVE_REPO python3 tools/convert_hf_to_torch_dist.py \
        \"\${MODEL_ARGS[@]}\" --hf-checkpoint $RLVE_HF_CKPT --save $RLVE_REF_LOAD"
    if command -v nvidia-smi >/dev/null && nvidia-smi -L >/dev/null 2>&1; then
        rlve_apptainer bash -c "$convert"
    else
        srun --account="${RLVE_ACCOUNT:-nlp}" --partition="${RLVE_PARTITION:-jag-standard}" \
             --gres=gpu:1 --cpus-per-task=8 --mem=64G --time=00:30:00 \
             bash -c "source scripts/slurm/common.sh && rlve_apptainer bash -c '$convert'"
    fi
    [[ -f "$RLVE_REF_LOAD/latest_checkpointed_iteration.txt" ]] || {
        echo "conversion failed: no latest_checkpointed_iteration.txt in $RLVE_REF_LOAD"; exit 1; }
fi

# 4. Eval sets: one spread of held-out environments (generalization) plus one frozen set
#    per training environment (same environment the run trains on, fixed seeds).
if [[ ! -f "$RLVE_EVAL_HELDOUT" ]]; then
    echo "==> building $RLVE_EVAL_HELDOUT"
    rlve_apptainer_cpu python3 tools/make_rlve_eval_set.py --output "$RLVE_EVAL_HELDOUT" subsample \
        --num-environments "$RLVE_EVAL_HELDOUT_NUM_ENVS" \
        --num-problems-per-environment 1 \
        --max-prompt-chars 1200
fi

read -r -a ENVIRONMENTS <<< "$RLVE_ENVIRONMENTS"
for environment in "${ENVIRONMENTS[@]}"; do
    rlve_build_env_eval_set "$environment" >/dev/null
done
echo "eval sets in $RLVE_EVAL_DIR:"
ls "$RLVE_EVAL_DIR"

echo
echo "ready. launch with:  sbatch scripts/slurm/8way-50step.sbatch"
