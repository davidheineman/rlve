#!/bin/bash
#SBATCH --job-name=rlve-1gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=08:00:00
#SBATCH --output=rlve-%j.out

CONFIG=${1:-configs/1gpu-smoke.yaml}

srun --container-image=slimerl/slime:v0.5.0rc0-cu126 \
     --container-mounts=/scratch/$USER:/scratch/$USER \
     --container-workdir=/scratch/$USER/RLVE \
     bash -lc "
set -ex
pip install -e . --no-deps
export PYTHONBUFFERED=16 TOKENIZERS_PARALLELISM=false

source scripts/models/deepseek-r1-distill-qwen-1.5B.sh
mapfile -t CFG_ARGS < <(python3 tools/yaml_to_args.py '$CONFIG')

ray start --head --node-ip-address 127.0.0.1 --num-gpus 1 \
    --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

ray job submit --address='http://127.0.0.1:8265' \
  --runtime-env-json='{\"env_vars\":{\"PYTHONPATH\":\"/root/Megatron-LM/\",\"CUDA_DEVICE_MAX_CONNECTIONS\":\"1\",\"NCCL_NVLS_ENABLE\":\"0\",\"PYTORCH_CUDA_ALLOC_CONF\":\"max_split_size_mb:1024\"}}' \
  -- python3 train.py \${MODEL_ARGS[@]} \"\${CFG_ARGS[@]}\"
"
