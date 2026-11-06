#!/usr/bin/env bash
#SBATCH --job-name=modular-rl-v2
#SBATCH --account=miso
#SBATCH --partition=jag-standard,miso,sphinx,miso-lo,sphinx-lo,jag-lo,sc-loprio
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --constraint=40G|48G|80G|141G
#SBATCH --cpus-per-task=24
#SBATCH --mem=416G
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --open-mode=append

set -euo pipefail
# Keep caches off the Slurm job /tmp mount. A unique directory per attempt also
# prevents cleanup from an earlier allocation from removing a retry's files.
export RLVE_CONTAINER_TMP
RLVE_CONTAINER_TMP="$(mktemp -d "/var/tmp/rlve-${USER}-${SLURM_JOB_ID:-local}.XXXXXXXX")"
echo "Private container /tmp: $RLVE_CONTAINER_TMP"
source "${RLVE_REPO:-${SLURM_SUBMIT_DIR:-$PWD}}/scripts/slurm/common.sh"
cd "$RLVE_REPO"
export RLVE_REPO RLVE_SCRATCH RLVE_IMAGE
export MODULAR_RL_PROFILE=v2
export MODULAR_RL_DATA="${MODULAR_RL_DATA:-$RLVE_SCRATCH/modular-mopd-rl/v2/data}"
export MODULAR_RL_SELECTION="${MODULAR_RL_SELECTION:-$RLVE_SCRATCH/modular-mopd-rl/v2/selection}"
export MODULAR_RL_RUNTIME="${MODULAR_RL_RUNTIME:-$RLVE_SCRATCH/modular-mopd-rl/v2/runtime}"
export MODULAR_RL_HF_HOME="${MODULAR_RL_HF_HOME:-$RLVE_SCRATCH/tinker-mopd/hf}"
export MODULAR_RL_OUTPUT="${MODULAR_RL_OUTPUT:-$RLVE_SCRATCH/modular-mopd-rl/v2/runs/$SLURM_JOB_ID}"
mkdir -p "$MODULAR_RL_OUTPUT"
echo "V2 task(s): ${MODULAR_RL_TASKS:-math code chat}; output: $MODULAR_RL_OUTPUT"
export APPTAINERENV_RLVE_REPO="$RLVE_REPO" APPTAINERENV_RLVE_SCRATCH="$RLVE_SCRATCH"
export APPTAINERENV_MODULAR_RL_PROFILE=v2
export APPTAINERENV_MODULAR_RL_DATA="$(readlink -f "$MODULAR_RL_DATA")"
export APPTAINERENV_MODULAR_RL_SELECTION="$(readlink -f "$MODULAR_RL_SELECTION")"
export APPTAINERENV_MODULAR_RL_RUNTIME="$(readlink -f "$MODULAR_RL_RUNTIME")"
export APPTAINERENV_MODULAR_RL_HF_HOME="$(readlink -f "$MODULAR_RL_HF_HOME")"
export APPTAINERENV_WANDB_API_KEY="${WANDB_API_KEY:?WANDB_API_KEY is required}"
rlve_apptainer bash modular_rl/allocation.sh "$(readlink -f "$MODULAR_RL_OUTPUT")"
