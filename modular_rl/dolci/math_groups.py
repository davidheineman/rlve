"""Original single-prompt, fixed-cap GRPO with groups of 16 and 32."""

import argparse
import ast
import fcntl
import json
import math
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
from .math_exploration import CONTROL, read, replace_once, sha
from .math_matched import QUEUES, replace_argument
from .report import events, scores, finalize_wandb
from .watch import states, TERMINAL_FAILURES

GROUPS = (16, 32)
RESPONSE_BUDGET = 8000


def prepare(root):
    from modular_rl.suite import prepare as prepare_suite
    from .followup import safe_command
    root = Path(root).resolve()
    if (root/'experiment.json').exists():return
    if (root/'rl').exists():raise ValueError('Inspect incomplete group-size experiment before preparing again')
    repo = Path(__file__).resolve().parents[2]
    suite = root/'rl'
    root.mkdir(parents=True,exist_ok=True)
    prepare_suite(repo,Path('/nlp/scr/dhei/modular-mopd-rl/v2'),suite,
        CONTROL/'sft',PARENT/'rl/base/selection',policy_gpus=4,
        initializers_from=CONTROL/'rl/math/initializers.json',selection_from=PARENT/'rl/base/selection',
        runs=[(f'math-grpo-g{g}','math','math') for g in GROUPS],
        queues=QUEUES,group='math-refreshed-grpo-group-size',frozen_from=PARENT/'rl/base',
        data_from=PARENT/'rl-data',resize_frozen_policy=True,time_limit='08:00:00')
    plan=read(suite/'launch-plan.json')
    source=Path(plan['source'])
    # Native actor, optimizer, advantage normalization, probes, and fixed-cap
    # loss are unchanged. The driver/reporter distinguish train and eval G.
    for module in ('driver.py','report.py'):
        shutil.copy2(repo/'modular_rl/v2'/module,source/'modular_rl/v2'/module)
    rollout=source/'modular_rl/v2/rollout.py'
    rollout.write_text(replace_once(rollout.read_text(),'enable_thinking=True,','enable_thinking=False,'))
    for row in plan['runs']:
        group=int(row['name'].rsplit('g',1)[1])
        steps=RESPONSE_BUDGET//group
        interval={16:25,32:12}[group]
        row.update(samples_per_prompt=group,train_steps=steps,evaluation_interval=interval,
                   prompts_per_update=1,objective_passes=1,train_rollouts=RESPONSE_BUDGET)
        row['command']=safe_command(row['command'])
        row['command']=[
            '--constraint=ampere|ada|hopper' if v.startswith('--constraint=') else
            '--export=ALL,WANDB_SERVICE=,NCCL_P2P_DISABLE=1,NCCL_CUMEM_HOST_ENABLE=0'
            if v.startswith('--export=') else v for v in row['command']]
        script=Path(row['command'][-1])
        lines=script.read_text().splitlines()
        at=lines.index('set -euo pipefail')+1
        lines[at:at]=[f'export MODULAR_RL_SAMPLES_PER_PROMPT={group}',
                       f'export APPTAINERENV_MODULAR_RL_SAMPLES_PER_PROMPT={group}']
        for i,line in enumerate(lines):
            if line.startswith(('export MODULAR_RL_CHECKPOINT_INTERVAL=',
                                'export APPTAINERENV_MODULAR_RL_CHECKPOINT_INTERVAL=')):
                lines[i]=line.split('=',1)[0]+'='+str(interval)
            if line.startswith(('export MODULAR_RL_EXTRA_ARGS=',
                                'export APPTAINERENV_MODULAR_RL_EXTRA_ARGS=')):
                prefix,value=line.split('=',1)
                arguments=shlex.split(shlex.split(value)[0])
                for flag,value in (
                    ('--n-samples-per-prompt',group),('--n-samples-per-eval-prompt',8),
                    ('--rollout-batch-size',1),('--over-sampling-batch-size',1),
                    ('--global-batch-size',group),('--num-rollout',steps),
                    ('--eval-interval',interval),('--eval-temperature',.7),('--eval-top-p',.95),
                    ('--seed',1234),('--clip-grad',1.)):
                    replace_argument(arguments,flag,value)
                assert arguments[arguments.index('--train-reuse-epochs')+1]=='1'
                assert arguments[arguments.index('--custom-loss-function-path')+1]=='modular_rl.v2.losses.fixed_token_policy_loss'
                assert '--custom-reward-post-process-path' not in arguments
                lines[i]=prefix+'='+shlex.quote(shlex.join(arguments))
        script.write_text('\n'.join(lines)+'\n')
        subprocess.run(['bash','-n',str(script)],check=True)
        row['script_sha256']=sha(script)
    for key in ('train_steps','samples_per_prompt','evaluation_interval','maximum_evaluation_rollouts_per_run'):
        plan.pop(key,None)
    plan.update(train_rollouts=RESPONSE_BUDGET,groups=list(GROUPS),objective_passes=1,
        enable_thinking=False,seed=1234,evaluation_temperature=.7,evaluation_top_p=.95,
        validation_samples_per_prompt=8,loss_normalization='original fixed 2048-token denominator',
        reference_group8=str(CONTROL/'rl/math'),gradient_norm_clip=1.,previous_runs_untouched=True,
        comparison_note='One prompt per batch. Equal 8K response budget gives G8:1000, '
                        'G16:500, G32:250 optimizer opportunities; fewer distinct prompts at larger G.')
    plan['source_sha256']={str(p.relative_to(source)):sha(p) for p in sorted(source.rglob('*'))
                          if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc'}
    atomic_json(suite/'launch-plan.json',plan)
    atomic_json(root/'experiment.json',{k:v for k,v in plan.items() if k not in ('source_sha256','selection')})


def curve(policy,keys):
    values=[]
    paths=sorted((policy/'rollouts/eval').glob('*.jsonl'))
    for path in paths:
        try:values.append({'rollout_id':int(path.stem),**scores(events(path),keys)})
        except (ValueError,json.JSONDecodeError):
            if path!=paths[-1] or (policy/'completed.json').exists():raise
    return values


def startup_validation(row):
    policy=Path(row['output'])/'policy'
    required=('optimizer-decisions.jsonl','update-probes.jsonl','rollouts/train/0000.jsonl','train.log')
    if not all((policy/p).exists() for p in required):return None
    try:
        decisions=events(policy/'optimizer-decisions.jsonl')
        probes=events(policy/'update-probes.jsonl')
        rows=events(policy/'rollouts/train/0000.jsonl')
    except json.JSONDecodeError:return None
    active=sum(bool(v['optimizer_update_applied']) for v in decisions)
    if active<3 or not probes:return None
    group=row['samples_per_prompt']
    assert len(rows)==group and len({json.loads(v['metadata_json'])['key'] for v in rows})==1
    assert all(v['optimizer_steps']==int(v['optimizer_update_applied']) for v in decisions)
    probe=probes[0]
    samples=sorted(probe['samples'],key=lambda v:v['sample'])
    assert [v['sample'] for v in samples]==list(range(group))
    p=sum(v['reward'] for v in samples)/group
    std=math.sqrt(p*(1-p)*group/(group-1))
    assert all(abs(v['advantage']-(v['reward']-p)/(std+1e-6))<1e-5 for v in samples)
    metrics=[]
    for line in (policy/'train.log').open():
        if "'train/normalization_mass'" not in line:continue
        match=re.search(r'step (\d+): (\{.*\})',line)
        if match:metrics.append(ast.literal_eval(match[2]))
    if len(metrics)<3:return None
    assert all(math.isfinite(v['train/grad_norm']) and 0<v['train/normalization_mass']<=1 for v in metrics)
    initial=events(policy/'rollouts/eval/0000.jsonl')
    assert len(initial)==512
    return {'passed':True,'utc':datetime.now(timezone.utc).isoformat(),'job_id':row['job_id'],
        'group_size':group,'optimizer_updates_observed':active,'objective_passes':1,
        'probe_group_pass_rate':p,'all_live_advantages_match_standard_grpo':True,
        'finite_gradient_norms':True,'fixed_token_loss_preserved':True,
        'validation_responses':512,'wandb_run_id':(policy/'wandb_run_id.txt').read_text().strip()}


def audit(row,suite):
    policy=Path(row['output'])/'policy'
    done=read(policy/'completed.json')
    group=row['samples_per_prompt'];count=done['updates']
    assert done['completed'] and done['max_train_rollouts']==RESPONSE_BUDGET
    assert done['prompts_per_update']==1 and done['samples_per_prompt']==group
    assert done['eval_samples_per_prompt']==8 and done['sampled_token_budget']==2048
    assert done['train_rollouts']==count*group and done['optimizer_updates']>0
    assert done['reason']=='validation_accuracy' or count==row['train_steps']
    selection=read(suite/'selection/math/selection.json')
    train_keys,val_keys=[set(selection['keys'][s]) for s in ('train','val')]
    assert not train_keys&val_keys
    paths=sorted((policy/'rollouts/train').glob('*.jsonl'))
    assert [int(p.stem) for p in paths]==list(range(count))
    for path in paths:
        rows=events(path)
        assert len(rows)==group
        keys={json.loads(v['metadata_json'])['key'] for v in rows}
        assert len(keys)==1 and keys<=train_keys
        for v in rows:
            prefix=json.loads(v['metadata_json']).get('_modular_v2_forcing',{}).get('prefix_tokens',0)
            assert 0<v['response_tokens']-prefix<=2048
    decisions={v['rollout_id']:v for v in events(policy/'optimizer-decisions.jsonl')}
    assert set(decisions)==set(range(count))
    assert all(v['optimizer_steps']==int(v['optimizer_update_applied']) for v in decisions.values())
    values=curve(policy,val_keys)
    assert [v['rollout_id'] for v in values]==sorted({0,count-1,*range(row['evaluation_interval']-1,count,row['evaluation_interval'])})
    checkpoint=read(policy/'ckpt/rlve_checkpoint_state.json')
    assert checkpoint['completed'] and checkpoint['rollout_id']==count-1
    return {'passed':True,'name':row['name'],'completed':done,'curve':values}


def plot(root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter
    suite=root/'rl'
    keys=set(read(suite/'selection/math/selection.json')['keys']['val'])
    reference=read(CONTROL/'rl/math/completion-audit.json')['curve']
    fig,ax=plt.subplots(figsize=(7,4))
    ax.plot([0 if v['rollout_id']==0 else (v['rollout_id']+1)*8 for v in reference],
            [v['pass@1'] for v in reference],label='GRPO group 8',color='tab:green')
    for row in read(suite/'jobs.json',[]):
        values=curve(Path(row['output'])/'policy',keys)
        if values:
            group=row['samples_per_prompt']
            ax.plot([0 if v['rollout_id']==0 else (v['rollout_id']+1)*group for v in values],
                    [v['pass@1'] for v in values],label=f'GRPO group {group}',color={16:'tab:blue',32:'tab:purple'}[group])
    ax.set(title='Math: original GRPO, larger groups',xlabel='train rollouts',ylabel='pass@1',
           xlim=(0,RESPONSE_BUDGET),ylim=(0,1))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda n,_:'0' if n==0 else f'{n/1000:g}K'))
    ax.legend(fontsize=9);ax.grid(alpha=.2);fig.tight_layout()
    output=Path('/juice5b/u/dhei/tatsu/rlve/plot/mopd-modular-v2/math_group_size.png')
    fig.savefig(output,dpi=180);fig.savefig(output.with_suffix('.pdf'));plt.close(fig)


def watch(root):
    root=Path(root).resolve();suite=root/'rl'
    with (root/'watch.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        errors=0
        while True:
            try:
                jobs=read(suite/'jobs.json',[]);slurm=states(jobs)
                status={'utc':datetime.now(timezone.utc).isoformat(),'runs':[],'complete':bool(jobs)}
                for row in jobs:
                    state=slurm.get(row['job_id'],{'state':'UNKNOWN'});policy=Path(row['output'])/'policy'
                    validation_path=root/f"g{row['samples_per_prompt']}-startup-validation.json"
                    if not validation_path.exists():
                        validation=startup_validation(row)
                        if validation:atomic_json(validation_path,validation)
                    status['runs'].append({'name':row['name'],'job_id':row['job_id'],**state,
                        'startup_validated':validation_path.exists(),'completed':read(policy/'completed.json')})
                    if state['state'] in TERMINAL_FAILURES:retry(root,suite/'jobs.json',row,state)
                    if state['state']=='COMPLETED':
                        destination=root/f"g{row['samples_per_prompt']}-completion-audit.json"
                        if not destination.exists():atomic_json(destination,audit(row,suite))
                        status['complete'] &= bool(finalize_wandb(row))
                    else:status['complete']=False
                plot(root);atomic_json(root/'monitor.json',status);print(json.dumps(status),flush=True)
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
    parser.add_argument('--submit',action='store_true');parser.add_argument('--watch',action='store_true')
    args=parser.parse_args()
    if args.watch:watch(args.root)
    else:
        prepare(args.root)
        if args.submit:
            from modular_rl.suite import submit
            submit(args.root/'rl')
