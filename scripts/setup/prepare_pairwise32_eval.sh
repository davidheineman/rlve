#!/bin/bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/slurm/pairwise-ood-32x32.sh"
cd "$RLVE_REPO"

mapfile -t batches < "$RLVE_ENV_FILE"
(( ${#batches[@]} == 4 )) || { echo "expected four batches in $RLVE_ENV_FILE" >&2; exit 1; }
environments=()
for batch in "${batches[@]}"; do
    read -r -a batch_environments <<< "$batch"
    (( ${#batch_environments[@]} == 8 )) || { echo "expected eight environments per batch" >&2; exit 1; }
    environments+=("${batch_environments[@]}")
done
(( $(printf '%s\n' "${environments[@]}" | sort -u | wc -l) == 32 )) || {
    echo "expected 32 distinct environments" >&2
    exit 1
}

rlve_apptainer_cpu python3 tools/make_rlve_eval_set.py --output "$RLVE_EVAL_HELDOUT" \
    --param-seed 1 generate --environments "${environments[@]}" \
    --difficulties $RLVE_EVAL_ENV_DIFFICULTIES \
    --num-problems-per-environment 12 --seed-start 2000000

for environment in "${environments[@]}"; do
    rlve_build_env_eval_set "$environment"
done
