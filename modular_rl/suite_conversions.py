"""Prepare the four native initializers without submitting RL training."""

import argparse
import json
import subprocess
from pathlib import Path

from .initializers import load_registry
from .prepare import write_json
from .suite import QUEUES, shell_exports
from .v2.base_models import sha256


def prepare(experiment, submit=False, queues=None):
    plan = json.loads((experiment / 'launch-plan.json').read_text())
    registry, source = Path(plan['initializers']), Path(plan['source'])
    directory = experiment / 'conversions'
    directory.mkdir(exist_ok=True)
    jobs = []
    for alias, spec in load_registry(registry, plan['initializers_sha256']).items():
        if alias == 'base':
            continue
        env = {'RLVE_REPO': source, 'MODULAR_RL_PROFILE': 'v2',
               'MODULAR_RL_INITIALIZERS_SHA256': plan['initializers_sha256'],
               'CONVERT_REGISTRY': registry, 'CONVERT_ALIAS': alias, 'CONVERT_TARGET': spec['native']}
        body = ('#!/usr/bin/env bash\n#SBATCH --account=miso\n'
                f'#SBATCH --partition={queues or QUEUES}\n#SBATCH --nodes=1\n#SBATCH --ntasks=1\n'
                '#SBATCH --requeue\n#SBATCH --open-mode=append\nset -euo pipefail\n')
        body += shell_exports(env)
        body += '''source "$RLVE_REPO/scripts/slurm/common.sh"
cd "$RLVE_REPO"
export PYTHONPATH="$RLVE_REPO:/root/Megatron-LM" OMP_NUM_THREADS=6 TOKENIZERS_PARALLELISM=false
mkdir -p "$(dirname "$CONVERT_TARGET")"
exec {conversion_lock}>"$CONVERT_TARGET.lock"
flock "$conversion_lock"
if [[ -e "$CONVERT_TARGET" ]]; then
    /nlp/scr/dhei/tinker-mopd/venv-rl/bin/python -m modular_rl.initializers "$CONVERT_REGISTRY" "$CONVERT_ALIAS"
    exit 0
fi
export CONVERT_TMP
CONVERT_TMP="$(mktemp -d "${CONVERT_TARGET}.partial.XXXXXXXX")"
rlve_apptainer bash -c '
    set -euo pipefail
    source "$RLVE_REPO/scripts/models/qwen3-0.6B-Base.sh"
    python3 -u -m modular_rl.initializer_convert "$CONVERT_REGISTRY" "$CONVERT_ALIAS" "$CONVERT_TMP" --architecture "${MODEL_ARGS[@]}"
'
[[ ! -e "$CONVERT_TARGET" ]]
mv -- "$CONVERT_TMP" "$CONVERT_TARGET"
'''
        script = directory / (alias + '.sbatch')
        script.write_text(body)
        command = ['sbatch', '--parsable', '--job-name=prepare-mid-' + alias, '--gres=gpu:1',
                   '--constraint=40G|48G|80G|141G', '--cpus-per-task=6', '--mem=32G', '--time=01:00:00',
                   '--exclude=viscam10,jagupard33,jagupard34', f'--chdir={source}',
                   f'--output={directory}/{alias}-%j.log', str(script)]
        job = subprocess.check_output(command, text=True).strip().split(';')[0] if submit else None
        jobs.append({'initializer': alias, 'job_id': job, 'command': command,
                     'script_sha256': sha256(script), 'rl_training': False})
        write_json(experiment / 'conversion-jobs.json', jobs)
    print(json.dumps(jobs), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('experiment', type=Path)
    parser.add_argument('--submit', action='store_true')
    args = parser.parse_args()
    prepare(args.experiment.resolve(), args.submit)
