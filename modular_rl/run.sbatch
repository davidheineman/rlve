#!/usr/bin/env bash
#SBATCH --job-name=modular-rl-smoke
#SBATCH --account=miso
#SBATCH --partition=jag-standard,miso,sphinx,miso-lo,sphinx-lo,jag-lo,sc-loprio
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --constraint=48G|80G|141G
#SBATCH --cpus-per-task=24
#SBATCH --mem=160G
#SBATCH --time=01:30:00
#SBATCH --open-mode=append
#SBATCH --output=/nlp/scr/%u/runs/modular-rl-%j.out

set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$PWD}/scripts/slurm/common.sh"
cd "$RLVE_REPO"
export RLVE_REPO RLVE_SCRATCH RLVE_IMAGE
export MODULAR_RL_DATA="${MODULAR_RL_DATA:-$RLVE_SCRATCH/modular-mopd-rl/data}"
export MODULAR_RL_HF_HOME="${MODULAR_RL_HF_HOME:-$RLVE_SCRATCH/tinker-mopd/hf}"
output="${MODULAR_RL_OUTPUT:-$RLVE_SCRATCH/modular-mopd-rl/runs/$SLURM_JOB_ID}"
mkdir -p "$output"
echo "Run directory: $output"
echo "Node: $(hostname); allocation GPUs: ${CUDA_VISIBLE_DEVICES:-unset}"
export APPTAINERENV_RLVE_REPO="$RLVE_REPO" APPTAINERENV_RLVE_SCRATCH="$RLVE_SCRATCH"
export APPTAINERENV_MODULAR_RL_DATA="$(readlink -f "$MODULAR_RL_DATA")"
export APPTAINERENV_MODULAR_RL_HF_HOME="$(readlink -f "$MODULAR_RL_HF_HOME")"
export APPTAINERENV_WANDB_API_KEY="${WANDB_API_KEY:?WANDB_API_KEY is required}"
rlve_apptainer bash modular_rl/allocation.sh "$(readlink -f "$output")"
