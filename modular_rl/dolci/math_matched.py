"""Matched eight-prompt, single-pass GRPO/MaxRL/MetaRL math pilot."""

import argparse
import ast
import fcntl
import json
import math
import os
import re
import shlex
import shutil
import subprocess
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

from .config import atomic_json
from .math_billion import PARENT
from .math_billion_watch import retry
from .math_exploration import CONTROL, install, read, sha
from .report import events, scores, finalize_wandb
from .watch import states, TERMINAL_FAILURES

QUEUES = 'miso-lo,sphinx-lo,jag-lo,sc-loprio,miso,sphinx'
METHODS = ('grpo', 'maxrl', 'metarl')
RESPONSES = 64
STEPS = 125
EVAL_INTERVAL = 6


def replace_argument(arguments, flag, value):
    if arguments.count(flag) > 1:
        raise ValueError('Duplicate argument: ' + flag)
    if flag in arguments:
        arguments[arguments.index(flag) + 1] = str(value)
    else:
        arguments += [flag, str(value)]


def prepare(root):
    from modular_rl.suite import prepare as prepare_suite
    from .followup import safe_command
    root = Path(root).resolve()
    if (root / 'experiment.json').exists():
        return
    if (root / 'rl').exists():
        raise ValueError('Inspect incomplete matched experiment before preparing again')
    repo = Path(__file__).resolve().parents[2]
    suite = root / 'rl'
    root.mkdir(parents=True, exist_ok=True)
    prepare_suite(repo, Path('/nlp/scr/dhei/modular-mopd-rl/v2'), suite,
                  CONTROL / 'sft', PARENT / 'rl/base/selection', policy_gpus=4,
                  initializers_from=CONTROL / 'rl/math/initializers.json',
                  selection_from=PARENT / 'rl/base/selection',
                  runs=[('math-batch8-' + method, 'math', 'math') for method in METHODS],
                  maxrl_runs=['math-batch8-maxrl'], queues=QUEUES,
                  group='math-refreshed-matched-batch8', frozen_from=PARENT / 'rl/base',
                  data_from=PARENT / 'rl-data', resize_frozen_policy=True,
                  train_steps=STEPS, evaluation_interval=EVAL_INTERVAL, time_limit='08:00:00')
    plan = read(suite / 'launch-plan.json')
    source = Path(plan['source'])
    install(source, repo)
    # Only fresh snapshots receive the new controller. Existing suites are immutable.
    for module in ('math_matched.py', 'report.py'):
        shutil.copy2(repo / 'modular_rl/dolci' / module, source / 'modular_rl/dolci' / module)
    for row in plan['runs']:
        method = row['name'].removeprefix('math-batch8-')
        row.update(algorithm=method, prompts_per_update=8, objective_passes=1)
        row['command'] = safe_command(row['command'])
        row['command'] = [
            '--constraint=ampere|ada|hopper' if v.startswith('--constraint=') else
            '--export=ALL,WANDB_SERVICE=,NCCL_P2P_DISABLE=1,NCCL_CUMEM_HOST_ENABLE=0'
            if v.startswith('--export=') else v for v in row['command']]
        script = Path(row['command'][-1])
        lines = script.read_text().splitlines()
        at = lines.index('set -euo pipefail') + 1
        lines[at:at] = [f'export MODULAR_RL_EXPLORATION_METHOD={method}',
                       f'export APPTAINERENV_MODULAR_RL_EXPLORATION_METHOD={method}']
        for i, line in enumerate(lines):
            if line.startswith(('export MODULAR_RL_PROMPTS_PER_UPDATE=',
                                'export APPTAINERENV_MODULAR_RL_PROMPTS_PER_UPDATE=')):
                lines[i] = line.split('=', 1)[0] + '=8'
            if line.startswith(('export MODULAR_RL_EXTRA_ARGS=',
                                'export APPTAINERENV_MODULAR_RL_EXTRA_ARGS=')):
                prefix, value = line.split('=', 1)
                arguments = shlex.split(shlex.split(value)[0])
                for flag, value in (
                    ('--train-reuse-epochs', 1), ('--rollout-batch-size', 8),
                    ('--over-sampling-batch-size', 8), ('--global-batch-size', 64),
                    ('--custom-loss-function-path', 'modular_rl.adaptive_entropy.policy_loss'),
                    ('--eval-temperature', .7), ('--eval-top-p', .95),
                    ('--seed', 1234), ('--clip-grad', 1.),
                    ('--custom-megatron-before-train-step-hook-path', 'modular_rl.adaptive_entropy.before_step')):
                    replace_argument(arguments, flag, value)
                if method == 'metarl':
                    replace_argument(arguments, '--custom-reward-post-process-path', 'modular_rl.metarl.post_process')
                    replace_argument(arguments, '--optimize-pass-k', 4)
                lines[i] = prefix + '=' + shlex.quote(shlex.join(arguments))
        script.write_text('\n'.join(lines) + '\n')
        subprocess.run(['bash', '-n', str(script)], check=True)
        row['script_sha256'] = sha(script)
    plan.update(prompts_per_step=8, samples_per_prompt=8, train_rollouts=8000,
                objective_passes=1, enable_thinking=False, seed=1234,
                evaluation_temperature=.7, evaluation_top_p=.95,
                gradient_norm_clip=1., global_loss_normalization='global valid response token mean',
                comparisons_note='All three methods share initialization, batches, loss reduction, '
                                 'LR, clipping, single-pass budget, and evaluation. Eight prompts '
                                 'per batch gives 125 optimizer opportunities at 8K responses.',
                previous_runs_untouched=True)
    plan['source_sha256'] = {str(p.relative_to(source)): sha(p) for p in sorted(source.rglob('*'))
                             if p.is_file() and '__pycache__' not in p.parts and p.suffix != '.pyc'}
    atomic_json(suite / 'launch-plan.json', plan)
    atomic_json(root / 'experiment.json', {k:v for k,v in plan.items() if k not in ('source_sha256','selection')})


def curve(policy, keys):
    values = []
    paths = sorted((policy / 'rollouts/eval').glob('*.jsonl'))
    for path in paths:
        try:
            values.append({'rollout_id': int(path.stem), **scores(events(path), keys)})
        except (ValueError, json.JSONDecodeError):
            if path != paths[-1] or (policy / 'completed.json').exists():
                raise
    return values


def startup_validation(row):
    """Check live 64-response grouping and objective advantages, not just CLI flags."""
    policy = Path(row['output'])/'policy'
    required = ['optimizer-decisions.jsonl','exploration.jsonl','update-probes.jsonl',
                'rollouts/train/0000.jsonl','train.log']
    if not all((policy/p).exists() for p in required):
        return None
    try:
        decisions = events(policy/'optimizer-decisions.jsonl')
        exploration = events(policy/'exploration.jsonl')
        probes = events(policy/'update-probes.jsonl')
        rows = events(policy/'rollouts/train/0000.jsonl')
    except json.JSONDecodeError:
        return None
    active = sum(bool(v['optimizer_update_applied']) for v in decisions)
    if active < 3 or not probes or len(rows) != 64:
        return None
    assert all(v['optimizer_steps'] == int(v['optimizer_update_applied']) for v in decisions)
    assert all(v['david/exploration/pass'] == 0 for v in exploration)
    keys = [json.loads(v['metadata_json'])['key'] for v in rows]
    assert len(set(keys)) == 8
    assert all(len(set(keys[i:i+8])) == 1 for i in range(0,64,8))
    probe = next(v for v in probes if v['rollout_id'] == 0)
    samples = sorted(probe['samples'],key=lambda v:v['sample'])
    assert [v['sample'] for v in samples] == list(range(64))
    weights, probabilities = [], []
    for start in range(0,64,8):
        group = samples[start:start+8]
        p = sum(v['reward'] for v in group)/8
        if row['algorithm']=='maxrl':weight=1/(p+1e-6)
        elif row['algorithm']=='metarl':weight=sum((1-p)**j for j in range(4))/sum(1/k for k in range(1,5))
        else:weight=1/(math.sqrt(p*(1-p)*8/7)+1e-6)
        for v in group:
            assert abs(v['advantage']-weight*(v['reward']-p)) < 1e-5
        probabilities.append(p);weights.append(weight)
    metrics=[]
    for line in (policy/'train.log').open():
        if "'train/normalization_mass'" not in line:
            continue
        match=re.search(r'step (\d+): (\{.*\})',line)
        if match:metrics.append(ast.literal_eval(match[2]))
    if len(metrics)<3:return None
    assert all(math.isfinite(v['train/grad_norm']) for v in metrics)
    assert all(abs(v['train/normalization_mass']-1)<1e-5 for v in metrics)
    return {'passed':True,'utc':datetime.now(timezone.utc).isoformat(),
            'job_id':row['job_id'],'algorithm':row['algorithm'],'optimizer_updates_observed':active,
            'prompt_groups_per_batch':8,'responses_per_batch':64,'objective_passes':1,
            'probe_rollout_id':0,'group_pass_rates':probabilities,'group_weights':weights,
            'all_64_live_advantages_match':True,'finite_gradient_norms':True,
            'global_token_normalization_mass_one':True,
            'wandb_run_id':(policy/'wandb_run_id.txt').read_text().strip()}


def audit(row, suite):
    policy = Path(row['output']) / 'policy'
    completed = read(policy / 'completed.json')
    assert completed['completed'] and completed['max_train_rollouts'] == 8000
    assert completed['prompts_per_update'] == 8 and completed['samples_per_prompt'] == 8
    assert completed['sampled_token_budget'] == 2048 and completed['optimizer_updates'] > 0
    count = completed['updates']
    assert completed['train_rollouts'] == count * RESPONSES
    assert completed['reason'] == 'validation_accuracy' or count == STEPS
    selection = read(suite / 'selection/math/selection.json')
    keys, val_keys = [set(selection['keys'][s]) for s in ('train', 'val')]
    assert not keys & val_keys
    paths = sorted((policy / 'rollouts/train').glob('*.jsonl'))
    assert [int(p.stem) for p in paths] == list(range(count))
    for path in paths:
        rows = events(path)
        assert len(rows) == RESPONSES
        group_keys = [json.loads(v['metadata_json'])['key'] for v in rows]
        assert len(set(group_keys)) == 8 and set(group_keys) <= keys
        assert all(len(set(group_keys[i:i+8])) == 1 for i in range(0,64,8))
        for v in rows:
            prefix = json.loads(v['metadata_json']).get('_modular_v2_forcing', {}).get('prefix_tokens', 0)
            assert 0 < v['response_tokens'] - prefix <= 2048
    decisions = {v['rollout_id']:v for v in events(policy/'optimizer-decisions.jsonl')}
    assert set(decisions) == set(range(count))
    assert all(v['optimizer_steps'] == int(v['optimizer_update_applied']) for v in decisions.values())
    state = read(policy / f'ckpt/iter_{count-1:07d}/exploration-state.json')
    assert state['method'] == row['algorithm']
    assert state['recipe_config'] == {'prompts_per_update':8,'samples_per_prompt':8,'objective_passes':1}
    assert state['controller']['steps'] == sum(v['optimizer_steps'] for v in decisions.values())
    if row['algorithm'] == 'metarl':
        assert state['method_config'] == {'optimize_pass_k':4}
    values = curve(policy, val_keys)
    assert [p['rollout_id'] for p in values] == sorted({0,count-1,*range(EVAL_INTERVAL-1,count,EVAL_INTERVAL)})
    return {'passed':True,'name':row['name'],'completed':completed,'curve':values,'state':state}


def plot(root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter
    suite = root / 'rl'
    keys = set(read(suite/'selection/math/selection.json')['keys']['val'])
    fig, ax = plt.subplots(figsize=(7,4))
    for row in read(suite/'jobs.json',[]):
        values = curve(Path(row['output'])/'policy',keys)
        if values:
            label,color = {'grpo':('GRPO','tab:green'),'maxrl':('MaxRL','tab:red'),
                           'metarl':('MetaRL (K=4)','tab:blue')}[row['algorithm']]
            ax.plot([0 if p['rollout_id']==0 else (p['rollout_id']+1)*RESPONSES for p in values],
                    [p['pass@1'] for p in values],label=label,color=color)
    ax.set(title='Math: 8 prompts per batch, 1 pass',xlabel='train rollouts',ylabel='pass@1',
           xlim=(0,8000),ylim=(0,1))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda n,_:'0' if n==0 else f'{n/1000:g}K'))
    if ax.lines: ax.legend(fontsize=9)
    ax.grid(alpha=.2);fig.tight_layout()
    output=Path('/juice5b/u/dhei/tatsu/rlve/plot/mopd-modular-v2/math_matched.png')
    fig.savefig(output,dpi=180);fig.savefig(output.with_suffix('.pdf'));plt.close(fig)


def watch(root):
    root = Path(root).resolve()
    suite = root/'rl'
    with (root/'watch.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        errors=0
        while True:
            try:
                jobs=read(suite/'jobs.json',[])
                slurm=states(jobs)
                status={'utc':datetime.now(timezone.utc).isoformat(),'runs':[],'complete':bool(jobs)}
                for row in jobs:
                    state=slurm.get(row['job_id'],{'state':'UNKNOWN'})
                    policy=Path(row['output'])/'policy'
                    status['runs'].append({'name':row['name'],'job_id':row['job_id'],**state,
                                           'completed':read(policy/'completed.json')})
                    validation_path=root/(row['algorithm']+'-startup-validation.json')
                    if not validation_path.exists():
                        validation=startup_validation(row)
                        if validation:atomic_json(validation_path,validation)
                    status['runs'][-1]['startup_validated']=validation_path.exists()
                    if state['state'] in TERMINAL_FAILURES:
                        retry(root,suite/'jobs.json',row,state)
                    if state['state']=='COMPLETED':
                        destination=root/(row['algorithm']+'-completion-audit.json')
                        if not destination.exists():atomic_json(destination,audit(row,suite))
                        status['complete'] &= bool(finalize_wandb(row))
                    else:status['complete']=False
                plot(root)
                atomic_json(root/'monitor.json',status)
                print(json.dumps(status),flush=True)
                errors=0
                if status['complete']:return
            except Exception:
                errors+=1
                atomic_json(root/'watch-error.json',{'consecutive_errors':errors,'traceback':traceback.format_exc()})
                traceback.print_exc()
                if errors>=8:raise
            time.sleep(45)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',required=True,type=Path)
    parser.add_argument('--submit',action='store_true')
    parser.add_argument('--watch',action='store_true')
    args=parser.parse_args()
    if args.watch:
        watch(args.root)
    else:
        prepare(args.root)
        if args.submit:
            from modular_rl.suite import submit
            submit(args.root/'rl')
