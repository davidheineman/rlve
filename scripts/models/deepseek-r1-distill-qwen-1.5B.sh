source "$(dirname "${BASH_SOURCE[0]}")/qwen2.5-1.5B.sh"
MODEL_ARGS+=(
   --rotary-base 10000
   --untie-embeddings-and-output-weights
)
