"""Launch and monitor MaxRL, MAI, and MetaRL from refreshed 1B math SFT."""

import argparse
import fcntl
import hashlib
import json
import math
import os
import shlex
import shutil
import subprocess
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

from .config import atomic_json
from .math_billion import PARENT, QUEUES
from .report import events, scores, finalize_wandb
from .watch import states, TERMINAL_FAILURES
from .math_billion_watch import retry

CONTROL = Path('/nlp/scr/dhei/math-large-20261101')


def read(path, default=None):
    return json.loads(Path(path).read_text()) if Path(path).exists() else default


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError('Frozen trainer differs at insertion point: ' + old[:100])
    return text.replace(old, new)


def install(source, repo):
    """Patch only this new immutable experiment's trainer lifecycle."""
    shutil.copy2(repo / 'modular_rl/adaptive_entropy.py', source / 'modular_rl/adaptive_entropy.py')
    shutil.copy2(repo / 'modular_rl/metarl.py', source / 'modular_rl/metarl.py')
    shutil.copy2(__file__, source / 'modular_rl/dolci/math_exploration.py')
    arguments = source / 'slime/utils/arguments.py'
    anchor = '            parser.add_argument(\n                "--advantage-estimator",\n'
    arguments.write_text(replace_once(arguments.read_text(), anchor,
        '            parser.add_argument("--optimize-pass-k", type=int, default=1,\n'
        '                                help="MetaRL harmonic pass@1..K reweighting; one disables reweighting")\n' + anchor))
    actor = source / 'slime/backends/megatron_utils/actor.py'
    text = actor.read_text()
    anchor = '        start_rollout_id = loaded_rollout_id + 1\n'
    text = replace_once(text, anchor, anchor +
        '        from modular_rl.adaptive_entropy import initialize\n        initialize(args, loaded_rollout_id)\n')
    anchor = '            if self.args.train_reuse_epochs > 1:\n'
    text = replace_once(text, anchor,
        '            from modular_rl.adaptive_entropy import prepare_rollout\n'
        '            prepare_rollout(self.args, rollout_data, rollout_id)\n\n' + anchor)
    anchor = '            save(iteration, self.model, None, None)\n'
    text = replace_once(text, anchor, anchor +
        '        from modular_rl.adaptive_entropy import save_state\n        save_state(self.args, iteration)\n')
    actor.write_text(text)
    model = source / 'slime/backends/megatron_utils/model.py'
    text = model.read_text()
    anchor = '        opt_param_scheduler.step(increment=args.global_batch_size)\n'
    text = replace_once(text, anchor, anchor +
        '        from modular_rl.adaptive_entropy import after_step\n        after_step(args, rollout_id, step_id)\n')
    model.write_text(text)
    rollout = source / 'modular_rl/v2/rollout.py'
    rollout.write_text(replace_once(rollout.read_text(), 'enable_thinking=True,', 'enable_thinking=False,'))


def prepare(root, methods=('maxrl', 'mai')):
    from modular_rl.suite import prepare as prepare_suite
    from .followup import safe_command
    root = Path(root).resolve()
    if not methods or len(methods) != len(set(methods)) or set(methods) - {'maxrl', 'mai', 'metarl'}:
        raise ValueError('Select unique supported exploration methods')
    suite = root / 'rl'
    if (root / 'experiment.json').exists():
        return
    if suite.exists():
        raise ValueError('Inspect the incomplete experiment directory before retrying preparation')
    root.mkdir(parents=True, exist_ok=True)
    repo = Path(__file__).resolve().parents[2]
    result = read(CONTROL / 'sft/math/results.json')
    assert result['train_tokens'] == 999817216
    assert result['config']['model']['repo'] == 'Qwen/Qwen3-0.6B-Base'
    registry = CONTROL / 'rl/math/initializers.json'
    prepare_suite(repo, Path('/nlp/scr/dhei/modular-mopd-rl/v2'), suite,
        CONTROL / 'sft', PARENT / 'rl/base/selection', policy_gpus=4,
        initializers_from=registry, selection_from=PARENT / 'rl/base/selection',
        runs=[('math-refreshed-' + method, 'math', 'math') for method in methods],
        maxrl_runs=['math-refreshed-maxrl'] if 'maxrl' in methods else [],
        queues=QUEUES, group='math-refreshed-exploration',
        frozen_from=PARENT / 'rl/base', data_from=PARENT / 'rl-data',
        resize_frozen_policy=True, time_limit='08:00:00')
    plan = read(suite / 'launch-plan.json')
    source = Path(plan['source'])
    install(source, repo)
    for row in plan['runs']:
        row['command'] = safe_command(row['command'])
        method = row['name'].removeprefix('math-refreshed-')
        row['algorithm'] = method
        script = Path(row['command'][-1])
        lines = script.read_text().splitlines()
        for i, line in enumerate(lines):
            if line.startswith(('export MODULAR_RL_EXTRA_ARGS=', 'export APPTAINERENV_MODULAR_RL_EXTRA_ARGS=')):
                prefix, value = line.split('=', 1)
                arguments = shlex.split(shlex.split(value)[0])
                for flag, target in (('--train-reuse-epochs', '4'),
                                     ('--custom-loss-function-path', 'modular_rl.adaptive_entropy.policy_loss')):
                    assert arguments.count(flag) == 1
                    arguments[arguments.index(flag) + 1] = target
                arguments += ['--eval-temperature', '.7', '--eval-top-p', '.95',
                              '--custom-megatron-before-train-step-hook-path', 'modular_rl.adaptive_entropy.before_step']
                if method == 'metarl':
                    arguments += ['--custom-reward-post-process-path', 'modular_rl.metarl.post_process',
                                  '--optimize-pass-k', '4']
                lines[i] = prefix + '=' + shlex.quote(shlex.join(arguments))
        at = lines.index('set -euo pipefail') + 1
        lines[at:at] = [f'export MODULAR_RL_EXPLORATION_METHOD={method}',
                       f'export APPTAINERENV_MODULAR_RL_EXPLORATION_METHOD={method}']
        script.write_text('\n'.join(lines) + '\n')
        subprocess.run(['bash', '-n', str(script)], check=True)
        row['script_sha256'] = sha(script)
    plan.update(objective_passes=4, enable_thinking=False, evaluation_temperature=.7,
                evaluation_top_p=.95, training_temperature=1., training_top_p=1.,
                global_loss_normalization='sum of unmasked losses / global unmasked response tokens',
                checkpointed_controller=True,
                metarl={'optimize_pass_k': 4, 'harmonic_normalizer': sum(1/k for k in range(1,5)),
                        'weight': 'sum((1-p)**j for j in range(K))/H_K',
                        'advantages': 'weighted_reward - prompt_mean(weighted_reward)',
                        'dataset_reward_scale': 1., 'per_group_std_normalization': False,
                        'dataset_scale_note': 'Fixed common scale. Estimating a dataset std from one prompt would cancel the objective weight.'},
                mai={'epsilon': .2, 'initial_k': 0., 'target_entropy': .3, 'delta': .25,
                     'kmax': 2.5, 'outer_ratio_max': 50.,
                     'estimator': 'importance-weighted sampled surprisal, equation 7',
                     'epsilon_note': 'Retain existing small-model epsilon=.2; paper large-model default is .6.'},
                reference_grpo=str(CONTROL / 'rl/math'),
                comparison_note='Both new methods use four objective passes and global token normalization; previous GRPO used one pass and a fixed token denominator.')
    plan['source_sha256'] = {str(p.relative_to(source)): sha(p) for p in sorted(source.rglob('*'))
                             if p.is_file() and '__pycache__' not in p.parts and p.suffix != '.pyc'}
    atomic_json(suite / 'launch-plan.json', plan)
    atomic_json(root / 'experiment.json', {k:v for k,v in plan.items() if k not in ('source_sha256', 'selection')})


def audit(row, suite):
    policy = Path(row['output']) / 'policy'
    completed = read(policy / 'completed.json')
    assert completed['completed'] and completed['train_rollouts'] == 8000
    assert completed['sampled_token_budget'] == 2048 and completed['optimizer_updates'] > 0
    selection = read(suite / 'selection/math/selection.json')
    curve = [{'rollout_id': int(p.stem), **scores(events(p), set(selection['keys']['val']))}
             for p in sorted((policy / 'rollouts/eval').glob('*.jsonl'))]
    assert [p['rollout_id'] for p in curve] == [0, *range(49, 1000, 50)]
    decisions = {v['rollout_id']:v for v in events(policy / 'optimizer-decisions.jsonl')}
    assert set(decisions) == set(range(1000))
    assert all(v['optimizer_steps'] == 4 * v['optimizer_update_applied'] for v in decisions.values())
    state = read(policy / 'ckpt/iter_0000999/exploration-state.json')
    assert state['method'] == row['algorithm']
    if row['algorithm'] == 'metarl':
        assert state['method_config'] == {'optimize_pass_k': 4}
    assert state['controller']['steps'] == sum(v['optimizer_steps'] for v in decisions.values())
    assert 0 <= state['controller']['k'] <= state['controller']['maximum']
    run_id = (policy / 'wandb_run_id.txt').read_text().strip()
    return {'passed': True, 'name': row['name'], 'completed': completed, 'curve': curve,
            'controller': state, 'wandb_run_id': run_id,
            'wandb_url': 'https://forge.coreweave.com/wandb/david-heineman/modular-mopd-rl-v1/runs/' + run_id}


def tracked_jobs(root):
    suites = [root / 'rl'] + [Path(p) for p in read(root / 'extra-suites.json', [])]
    jobs = [{**row, '_suite': str(suite), '_record': str(suite / 'jobs.json')}
            for suite in suites for row in read(suite / 'jobs.json', [])]
    if len({row['name'] for row in jobs}) != len(jobs):
        raise ValueError('Duplicate run names in exploration monitor')
    return jobs


def plot(root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter
    series = [('GRPO (1 pass)', read(CONTROL / 'rl/math/completion-audit.json')['curve'], 'tab:green')]
    suite = root / 'rl'
    keys = set(read(suite / 'selection/math/selection.json')['keys']['val'])
    for row in tracked_jobs(root):
        policy = Path(row['output']) / 'policy'
        curve = []
        paths = sorted((policy / 'rollouts/eval').glob('*.jsonl'))
        for p in paths:
            try:
                curve.append({'rollout_id':int(p.stem), **scores(events(p), keys)})
            except (ValueError, json.JSONDecodeError):
                if p != paths[-1] or (policy / 'completed.json').exists(): raise
        if curve:
            label, color = {'maxrl': ('MaxRL (4 passes)', 'tab:red'),
                            'mai': ('MAI entropy control (4 passes)', 'tab:purple'),
                            'metarl': ('MetaRL (K=4, 4 passes)', 'tab:blue')}[row['algorithm']]
            series.append((label, curve, color))
    fig, ax = plt.subplots(figsize=(7,4))
    for label, curve, color in series:
        ax.plot([0 if p['rollout_id']==0 else (p['rollout_id']+1)*8 for p in curve],
                [p['pass@1'] for p in curve], label=label, color=color)
    ax.set(title='Math',xlabel='train rollouts',ylabel='pass@1',xlim=(0,8000),ylim=(0,1))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda n,_:'0' if n==0 else f'{n/1000:g}K'))
    ax.legend(fontsize=8);ax.grid(alpha=.2);fig.tight_layout()
    output=Path('/juice5b/u/dhei/tatsu/rlve/plot/mopd-modular-v2/math_exploration.png')
    fig.savefig(output,dpi=180);fig.savefig(output.with_suffix('.pdf'));plt.close(fig)


def watch(root):
    root = Path(root).resolve()
    suite = root / 'rl'
    with (root / 'watch.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        errors = 0
        while True:
            try:
                jobs = tracked_jobs(root)
                slurm = states(jobs)
                status = {'utc': datetime.now(timezone.utc).isoformat(), 'runs': [], 'complete': True}
                for row in jobs:
                    state = slurm.get(row['job_id'], {'state':'UNKNOWN'})
                    policy = Path(row['output']) / 'policy'
                    info = {'name':row['name'],'job_id':row['job_id'],**state,
                            'completed':read(policy/'completed.json')}
                    status['runs'].append(info)
                    if state['state'] in TERMINAL_FAILURES:
                        retry(root,Path(row['_record']),row,state)
                    if state['state']=='COMPLETED':
                        destination=root/(row['algorithm']+'-completion-audit.json')
                        if not destination.exists():atomic_json(destination,audit(row,Path(row['_suite'])))
                        info['wandb_finalized'] = finalize_wandb(row)
                        status['complete'] &= bool(info['wandb_finalized'])
                    else:status['complete']=False
                plot(root)
                atomic_json(root/'monitor.json',status)
                print(json.dumps(status),flush=True)
                errors = 0
                if status['complete']:return
            except Exception:
                errors += 1
                atomic_json(root/'watch-error.json',{'consecutive_errors':errors,'traceback':traceback.format_exc()})
                traceback.print_exc()
                if errors >= 8:raise
            time.sleep(45)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--submit',action='store_true')
    parser.add_argument('--watch',action='store_true')
    parser.add_argument('--meta-only',action='store_true')
    args=parser.parse_args()
    if args.watch:
        watch(args.root)
    else:
        prepare(args.root, methods=('metarl',) if args.meta_only else ('maxrl', 'mai'))
        if args.submit:
            from modular_rl.suite import submit
            submit(args.root/'rl')
