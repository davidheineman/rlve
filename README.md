# RLVE

Run commands from this directory. See [AGENTS.md](AGENTS.md) for experiment
recipes, findings, cluster settings and recovery guidance.

```bash
bash scripts/setup/prepare.sh
RLVE_ENVIRONMENTS=Multiplication \
  RLVE_EXTRA_ARGS="--num-rollout 4 --eval-interval 2" \
  sbatch --gres=gpu:1 --cpus-per-task=8 --mem=64G --time=00:40:00 \
  scripts/slurm/8way-50step.sbatch
```

Setup prepares the pinned Slime container, default Qwen2.5-1.5B-Instruct weights
and evaluation sets; conversion may allocate a GPU. Export `WANDB_API_KEY` for logging.
