"""Prepare, launch and supervise twelve independent 16K-response GRPO runs."""

import argparse
import fcntl
import hashlib
import json
import math
import os
import shlex
import shutil
import subprocess
import sys
import time
import traceback
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import yaml

from modular_rl.initializers import verify
from .config import atomic_json
from .report import scores, finalize_wandb
from .watch import states, TERMINAL_FAILURES

REPO = Path(__file__).resolve().parents[2]
PARENT = Path('/nlp/scr/dhei/dolci-mopd-sftinit-4096-eval32-20261103')
CONTROL = Path('/nlp/scr/dhei/math-groups-20261102/rl')
QUEUES = 'miso-lo,sphinx-lo,jag-lo,sc-loprio,miso,sphinx,jag-standard'


def read(path, default=None):
    return json.loads(Path(path).read_text()) if Path(path).exists() else default


def events(path):
    return [json.loads(line) for line in Path(path).open() if line.strip()]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def shell_exports(values):
    return ''.join(f'export {key}={shlex.quote(str(value))}\n' for key,value in values.items())


def prepare(root):
    if (root/'launch-plan.json').exists():
        return
    if (root/'source').exists():
        raise ValueError('Inspect incomplete preparation before retrying')
    plan = read(root/'plan.json')
    atomic_json(root/'estimate-plan-v1.json', plan)
    discarded = [v for v in plan['runs'] if v['task']=='math' and v['group_size']==16]
    plan['runs'] = [v for v in plan['runs'] if v not in discarded]
    assert len(plan['runs']) == 12
    for v in discarded:
        target=root/'superseded-configs';target.mkdir(exist_ok=True)
        shutil.move(str(root/'configs'/(v['name']+'.json')), str(target/(v['name']+'.json')))
    plan.update(status='preparing-not-submitted',run_count=12,total_training_responses=192000,
                total_validation_responses=202752,gpu_hours_estimate=[156,252],
                sweep_hours_with_24_available_gpus=[8,14],sweep_hours_with_12_available_gpus=[16,24])
    source = root/'source'
    control = read(CONTROL/'launch-plan.json')
    for relative, expected in control['source_sha256'].items():
        if sha(Path(control['source'])/relative) != expected:
            raise ValueError('Proven GRPO source changed: '+relative)
    ignore=shutil.ignore_patterns('__pycache__','*.pyc')
    shutil.copytree(CONTROL/'source',source,ignore=ignore)
    shutil.copytree(PARENT/'students/runtime',root/'runtime',ignore=ignore)
    (root/'selection').mkdir()
    for task in ('math','code','if','chat'):
        shutil.copytree(PARENT/'students/selection'/task,root/'selection'/task,ignore=ignore)
    # The four-GPU group16/group32 GRPO backend stays pinned. Bring in the
    # current prompt, grader, EOS, and reward adapters from the audited MOPD tree.
    fixes = ('modular_rl/v2/rollout.py','modular_rl/v2/if_rollout.py',
             'modular_rl/v2/math_reward.py','modular_rl/graders.py',
             'modular_rl/code_worker.py','modular_rl/v2/mbpp_spec.py',
             'modular_rl/if_grader.py','modular_rl/skywork.py',
             'modular_rl/dolci/chat_reward.py')
    for relative in fixes:
        target=source/relative;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(PARENT/'students/source'/relative,target)
    shutil.rmtree(source/'modular_rl/vendor')
    shutil.copytree(PARENT/'students/source/modular_rl/vendor',source/'modular_rl/vendor',ignore=ignore)
    for relative in ('modular_rl/v2/driver.py','modular_rl/v2/report.py',
                     'modular_rl/v2/recovery.py','modular_rl/mopd_recovery.py',
                     'modular_rl/suite_allocation.sh','modular_rl/dolci/rl16k_runtime.py'):
        target=source/relative;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(REPO/relative,target)
    config_path=source/'modular_rl/v2/config.yaml';config=yaml.safe_load(config_path.read_text())
    config.pop('early-stop-val-accuracy',None)
    config.update({'rollout-max-response-len':2048,'eval-max-response-len':2048,
                   'eval-temperature':.7,'eval-top-p':.95})
    config_path.write_text(yaml.safe_dump(config,sort_keys=False))
    allocation=source/'modular_rl/suite_allocation.sh';text=allocation.read_text()
    anchor='mkdir -p "$RLVE_RUN_DIR"'
    assert text.count(anchor)==1
    text=text.replace(anchor,anchor+'\npython3 -m modular_rl.mopd_recovery --run "$RLVE_RUN_DIR"')
    text=text.replace('RLVE_TRAIN_ENTRYPOINT=modular_rl/v2/driver.py',
                      'RLVE_TRAIN_ENTRYPOINT=modular_rl/dolci/rl16k_runtime.py')
    text=text.replace('RUN_TIMEOUT=56400','RUN_TIMEOUT=42600')
    allocation.write_text(text)
    registry=root/'initializers.json';atomic_json(registry,plan['models'])
    args_line=next(l for l in (CONTROL/'runs/math-grpo-g32/run.sbatch').read_text().splitlines()
                   if l.startswith('export MODULAR_RL_EXTRA_ARGS='))
    base_args=shlex.split(shlex.split(args_line.split('=',1)[1])[0])
    def replace(arguments,flag,value):
        if arguments.count(flag)>1:raise ValueError('Duplicate '+flag)
        if flag in arguments:arguments[arguments.index(flag)+1]=str(value)
        else:arguments.extend([flag,str(value)])
    for row in plan['runs']:
        args=list(base_args);task=row['task'];group=row['group_size']
        for flag,value in (('--num-rollout',row['training_batch_steps']),
            ('--eval-interval',row['eval_interval_batches']),('--global-batch-size',group),
            ('--n-samples-per-prompt',group),('--answer-forcing-extra-tokens',max(128,row['answer_forcing_reserve'])),
            ('--actor-num-gpus-per-node',row['policy_gpus']),('--num-gpus-per-node',row['policy_gpus'])):
            replace(args,flag,value)
        if task=='if':
            args.extend(['--custom-generate-function-path','modular_rl.v2.if_rollout.generate',
                         '--custom-rm-path','modular_rl.if_grader.reward'])
        args.extend(['--rollout-stop-token-ids','151643','151645'])
        output=root/'runs'/row['name'];output.mkdir(parents=True)
        env={'RLVE_REPO':source,'RLVE_SCRATCH':Path('/nlp/scr/dhei').resolve(),
            'RLVE_IMAGE':'/juice5b/scr5b/dhei/images/slime-v0.5.0rc0-cu126.sif',
            'RLVE_WANDB_TEAM':'david-heineman','WANDB_BASE_URL':'https://api.forge.coreweave.com',
            'MODULAR_RL_PROFILE':'v2','MODULAR_RL_TASKS':task,'MODULAR_RL_INITIALIZER':row['initializer_alias'],
            'MODULAR_RL_INITIALIZERS':registry,'MODULAR_RL_INITIALIZERS_SHA256':sha(registry),
            'MODULAR_RL_RUN_NAME':row['name'],'MODULAR_RL_WANDB_PROJECT':plan['project'],
            'MODULAR_RL_RUNTIME':root/'runtime','MODULAR_RL_SELECTION':root/'selection',
            'MODULAR_RL_DATA':'/juice5b/scr5b/dhei/dolci-mix-20261029/rl-data',
            'MODULAR_RL_OUTPUT':output,'MODULAR_RL_HF_HOME':'/juice5b/scr5b/dhei/tinker-mopd/hf',
            'MODULAR_RL_VALIDATION_PROMPTS':64,'MODULAR_RL_PROMPTS_PER_UPDATE':1,
            'MODULAR_RL_SAMPLES_PER_PROMPT':group,'MODULAR_RL_CHECKPOINT_INTERVAL':row['checkpoint_interval_batches'],
            'MODULAR_RL_MAX_RESPONSE_TOKENS':2048,'MODULAR_RL_ANSWER_FORCING_TOKENS':max(128,row['answer_forcing_reserve']),
            'MODULAR_RL_BASE_TOKENIZER':1,'MODULAR_RL_SKIP_ZERO_ADVANTAGE':1,
            'MODULAR_RL_POLICY_GPUS':row['policy_gpus'],'MODULAR_RL_REWARD_GPUS':max(1,row['reward_gpus']),
            'MODULAR_RL_GROUP':'dolci-rl16k-g32-math-g16-other',
            'MODULAR_RL_EVAL_ONLY':0,'MODULAR_RL_FILTER_UNIFORM_REFERENCE_BATCHES':0,
            'MODULAR_RL_ALLOW_OBJECTIVE_REUSE':1,'MODULAR_RL_CUDA_CACHE_CLEAR_INTERVAL':1,
            'MODULAR_RL_UPDATE_PROBE_INTERVAL':10,'MODULAR_RL_CODE_FORCING_FALLBACK':1,
            'MODULAR_RL_RUN_CONFIG':root/'configs'/(row['name']+'.json'),
            'MODULAR_RL_EXTRA_ARGS':shlex.join(args)}
        script=output/'run.sbatch'
        script.write_text('#!/usr/bin/env bash\n#SBATCH --account=miso\n#SBATCH --partition='+QUEUES+
            '\n#SBATCH --nodes=1\n#SBATCH --ntasks=1\n#SBATCH --requeue\n#SBATCH --open-mode=append\n'+
            'set -euo pipefail\n'+shell_exports(env)+shell_exports({'APPTAINERENV_'+k:v for k,v in env.items()})+
            'export APPTAINERENV_WANDB_API_KEY="${WANDB_API_KEY:?}"\nsource "$RLVE_REPO/scripts/slurm/common.sh"\n'+
            'rlve_apptainer bash modular_rl/suite_allocation.sh "$@"\n')
        row.update(output=str(output),script_sha256=sha(script),command=['sbatch','--parsable',
            '--job-name='+row['name'],'--gres=gpu:4','--constraint=ampere|ada|hopper',
            '--cpus-per-task=28','--mem=128G','--time=12:00:00',
            '--exclude=viscam10,jagupard33,jagupard34,pasteur6','--chdir='+str(source),
            '--output='+str(output/'slurm-%j.out'),
            '--export=ALL,WANDB_SERVICE=,NCCL_P2P_DISABLE=1,NCCL_CUMEM_HOST_ENABLE=0',str(script)])
        atomic_json(root/'configs'/(row['name']+'.json'),row)
    plan.update(status='prepared-not-submitted',source=str(source),initializers=str(registry),
                initializers_sha256=sha(registry),source_sha256={str(p.relative_to(source)):sha(p)
                    for p in sorted(source.rglob('*')) if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc'})
    atomic_json(root/'plan.json',plan);atomic_json(root/'launch-plan.json',plan)
    # Keep reporting and recovery imports self-contained for the CPU watcher.
    shutil.copytree(source/'modular_rl',root/'controller-source/modular_rl',ignore=ignore)
    for name in ('rl16k.py','report.py','watch.py','config.py'):
        shutil.copy2(REPO/'modular_rl/dolci'/name,root/'controller-source/modular_rl/dolci'/name)
    watch=root/'watch.sbatch'
    watch.write_text('#!/usr/bin/env bash\nset -euo pipefail\nunset WANDB_SERVICE\n'+
        'export PYTHONPATH='+shlex.quote(str(root/'controller-source'))+'\n'+
        'export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1\nexec '+
        '/juice5b/u/dhei/tatsu/nano-omni/.venv-midtraining/bin/python -u -m modular_rl.dolci.rl16k watch --root '+
        shlex.quote(str(root))+'\n')
    (root/'submit-all.sh').write_text('#!/usr/bin/env bash\nset -euo pipefail\nunset WANDB_SERVICE\n'+
        'export PYTHONPATH='+shlex.quote(str(root/'controller-source'))+'\nexec '+
        '/nlp/scr/dhei/tinker-mopd/venv-rl/bin/python -u -m modular_rl.dolci.rl16k submit --root '+shlex.quote(str(root))+'\n')
    preflight(root)


def preflight(root):
    plan=read(root/'launch-plan.json');checks=[]
    assert len(plan['runs'])==12 and all(v['group_size']==(32 if v['task']=='math' else 16) for v in plan['runs'])
    for spec in plan['models'].values():verify(spec)
    for rel,digest in plan['source_sha256'].items():
        if sha(Path(plan['source'])/rel)!=digest:raise ValueError('Frozen source changed: '+rel)
    for task in ('math','code','if','chat'):
        d=root/'selection'/task;info=read(d/'selection.json')
        for split in ('train','val'):
            assert sha(d/(split+'.jsonl'))==info['sha256'][split]
        assert len(info['keys']['val'])==64 and not set(info['keys']['val'])&set(info['keys']['train'])
    import tempfile
    with tempfile.TemporaryDirectory(prefix='rl16k-preflight-') as temporary:
        for row in plan['runs']:
            script=Path(row['command'][-1]);assert sha(script)==row['script_sha256']
            subprocess.run(['bash','-n',str(script)],check=True)
            text=script.read_text().replace('rlve_apptainer bash modular_rl/suite_allocation.sh',
                                          'rlve_apptainer_cpu bash modular_rl/suite_allocation.sh')
            path=Path(temporary)/row['name'];path.write_text(text)
            proc=subprocess.run(['bash',str(path),'--check-environment'],capture_output=True,text=True)
            checks.append({'name':row['name'],'returncode':proc.returncode,'output':proc.stdout+proc.stderr})
            atomic_json(root/'container-preflight.json',checks)
            if proc.returncode:raise ValueError('Container preflight failed: '+row['name'])
    atomic_json(root/'preflight.json',{'passed':True,'runs':12,'groups':{'math':32,'other':16},
        'training_responses_each':16000,'gpus_each':4,'models_verified':list(plan['models']),
        'project':plan['project'],'container_checks':len(checks),'no_training_submitted':not (root/'jobs.json').exists()})


def submit(root):
    if (root/'HALTED_BY_USER.json').exists():raise ValueError('Experiment halted')
    preflight(root);plan=read(root/'launch-plan.json')
    required=len(plan['runs'])*2*8347048481+8*1024**3
    free=shutil.disk_usage(root).free
    atomic_json(root/'storage-capacity.json',{'free_bytes':free,'required_bytes':required,
        'all_twelve_runs_simultaneously_resumable':free>=required})
    if free<required:raise RuntimeError('Insufficient checkpoint reserve for twelve simultaneous runs')
    jobs=read(root/'jobs.json',[])
    for row in plan['runs']:
        if any(j['name']==row['name'] for j in jobs):continue
        job=subprocess.check_output(row['command'],text=True).strip().split(';')[0]
        jobs.append({**row,'job_id':job});atomic_json(root/'jobs.json',jobs)
        print('SUBMITTED',row['name'],job,flush=True)
    if not (root/'watch-job.json').exists():
        command=['sbatch','--parsable','--account=miso','--partition='+QUEUES,
            '--nodes=1','--ntasks=1','--cpus-per-task=1','--mem=4G','--time=3-00:00:00',
            '--requeue','--open-mode=append','--job-name=rl16k-watch',
            '--output='+str(root/'watch-%j.log'),str(root/'watch.sbatch')]
        job=subprocess.check_output(command,text=True).strip().split(';')[0]
        atomic_json(root/'watch-job.json',{'job_id':job,'command':command})
    plan.update(status='submitted',no_jobs_submitted=False,prelaunch_remaining=[])
    atomic_json(root/'launch-plan.json',plan);atomic_json(root/'plan.json',plan)


def panel_curve(root,row,cache):
    policy=Path(row['output'])/'policy';selection=read(root/'selection'/row['task']/'selection.json')
    curve=[];paths=sorted((policy/'rollouts/eval').glob('*.jsonl'))
    for path in paths:
        signature=(path.stat().st_size,path.stat().st_mtime_ns);key=str(path)
        if key not in cache or cache[key][0]!=signature:
            try:values=scores(events(path),set(selection['keys']['val']))
            except (ValueError,json.JSONDecodeError):
                if path==paths[-1] and not (policy/'completed.json').exists():continue
                raise
            cache[key]=(signature,{'rollout_id':int(path.stem),**values})
        curve.append(cache[key][1])
    return curve


def startup(row):
    policy=Path(row['output'])/'policy'
    paths=[policy/n for n in ('optimizer-decisions.jsonl','update-probes.jsonl','training-protocol.jsonl')]
    if not all(p.exists() for p in paths):return None
    try:decisions,probes,protocol=[events(p) for p in paths]
    except json.JSONDecodeError:return None
    if sum(v['optimizer_update_applied'] for v in decisions)<3 or not probes:return None
    import statistics
    informative=[v for v in probes if len({s['reward'] for s in v['samples']})>1]
    if not informative:return None
    group=row['group_size'];sample=informative[0]['samples']
    assert sorted(v['sample'] for v in sample)==list(range(group))
    rewards=[v['reward'] for v in sample];mean=statistics.mean(rewards);std=statistics.stdev(rewards)
    assert all(abs(v['advantage']-(v['reward']-mean)/(std+1e-6))<1e-4 for v in sample)
    first=events(policy/'rollouts/train/0000.jsonl')
    assert len(first)==group and len({json.loads(v['metadata_json'])['key'] for v in first})==1
    return {'passed':True,'group_size':group,'live_advantages_match_grpo':True,
        'optimizer_updates_observed':sum(v['optimizer_update_applied'] for v in decisions),
        'protocol':protocol[-1],'wandb_run_id':(policy/'wandb_run_id.txt').read_text().strip()}


def audit(root,row,curve):
    policy=Path(row['output'])/'policy';done=read(policy/'completed.json');steps=row['training_batch_steps']
    assert done['completed'] and done['train_rollouts']==done['max_train_rollouts']==16000
    assert done['updates']==steps and done['samples_per_prompt']==row['group_size']
    assert done['prompts_per_update']==1 and done['sampled_token_budget']==2048
    assert done['reason']=='rollout_budget' and done['optimizer_updates']>0
    state=read(policy/'ckpt/rlve_checkpoint_state.json')
    assert state['completed'] and state['rollout_id']==steps-1
    paths=sorted((policy/'rollouts/train').glob('*.jsonl'))
    assert [int(p.stem) for p in paths]==list(range(steps))
    train_keys=set(read(root/'selection'/row['task']/'selection.json')['keys']['train'])
    for p in paths:
        values=events(p);assert len(values)==row['group_size']
        keys={json.loads(v['metadata_json'])['key'] for v in values}
        assert len(keys)==1 and keys<=train_keys
        for v in values:
            assert v['source_environment']==row['task']
            prefix=json.loads(v['metadata_json']).get('_modular_v2_forcing',{}).get('prefix_tokens',0)
            assert 0<=v['response_tokens']-prefix<=2048
    expected=sorted({0,steps-1,*range(row['eval_interval_batches']-1,steps,row['eval_interval_batches'])})
    assert [v['rollout_id'] for v in curve]==expected
    return {'passed':True,'name':row['name'],'completed':done,'curve':curve}


def plot(root,curves):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter
    fig,axes=plt.subplots(2,2,figsize=(8.8,5.6))
    tasks=('math','code','if','chat');colors={'base':'tab:blue','baseline-sft':'tab:orange','task-sft':'tab:green'}
    labels={'base':'RL on base','baseline-sft':'RL on baseline SFT','task-sft':'RL on task SFT'}
    for row in read(root/'jobs.json',[]):
        ax=axes.flat[tasks.index(row['task'])];values=curves.get(row['name'],[])
        if values:ax.plot([0 if v['rollout_id']==0 else (v['rollout_id']+1)*row['group_size'] for v in values],
                         [v['pass@1'] for v in values],color=colors[row['bank']],label=labels[row['bank']])
    for task,ax in zip(tasks,axes.flat):
        ax.set(title={'math':'Math','code':'Code','if':'IF','chat':'Chat'}[task],xlabel='train rollouts',ylabel='pass@1',xlim=(0,16000))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda n,_:'0' if n==0 else f'{n/1000:g}K'))
        ax.grid(alpha=.2)
        if ax.lines:ax.legend(fontsize=8)
    fig.tight_layout();dest=Path('/juice5b/u/dhei/tatsu/rlve/plot/mopd-modular-v2/rl16k_training_curves.png')
    fig.savefig(dest,dpi=180);fig.savefig(dest.with_suffix('.pdf'));plt.close(fig)


def retry(root,row,state):
    history=read(root/'retry-history.json',[])
    if sum(v['name']==row['name'] for v in history)>=6:raise RuntimeError('Retry limit reached: '+row['name'])
    command=list(row['command'])
    if state['state']=='OUT_OF_MEMORY':
        command=['--constraint=40G|48G|80G|141G' if v.startswith('--constraint=') else v for v in command]
    job=subprocess.check_output(command,text=True).strip().split(';')[0]
    history.append({'name':row['name'],'previous_job':row['job_id'],'job_id':job,'previous_state':state,
                    'utc':datetime.now(timezone.utc).isoformat()})
    atomic_json(root/'retry-history.json',history)
    jobs=read(root/'jobs.json')
    for v in jobs:
        if v['name']==row['name']:v.update(job_id=job,command=command)
    atomic_json(root/'jobs.json',jobs);print('RETRIED',row['name'],job,flush=True)


def watch(root):
    cache={};last_curves=None
    with (root/'watch.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        while not (root/'HALTED_BY_USER.json').exists():
            jobs=read(root/'jobs.json',[]);slurm=states(jobs);curves={}
            status={'utc':datetime.now(timezone.utc).isoformat(),'complete':bool(jobs),'runs':[]}
            for row in jobs:
                try:
                    state=slurm.get(row['job_id'],{'state':'UNKNOWN'})
                    state['state']=state['state'].split()[0].rstrip('+')
                    policy=Path(row['output'])/'policy'
                    curve=panel_curve(root,row,cache);curves[row['name']]=curve
                    receipt=root/'startup-audits'/(row['name']+'.json')
                    if not receipt.exists():
                        proof=startup(row)
                        if proof:atomic_json(receipt,proof)
                    entry={'name':row['name'],'job_id':row['job_id'],**state,
                           'startup_validated':receipt.exists(),'evaluations':len(curve)}
                    progress=policy/'progress.jsonl'
                    if progress.exists():
                        last=[v for v in events(progress) if v['phase']=='train']
                        if last:entry.update(completed_batches=last[-1]['completed_updates'],train_rollouts=last[-1]['train_rollouts'])
                    finished=False
                    if state['state'] in TERMINAL_FAILURES:retry(root,row,state)
                    if state['state']=='COMPLETED':
                        dest=root/'completion-audits'/(row['name']+'.json')
                        if not dest.exists():atomic_json(dest,audit(root,row,curve))
                        finished=bool(finalize_wandb(row))
                    status['runs'].append(entry);status['complete'] &= finished
                except Exception as error:
                    status['complete']=False
                    status['runs'].append({'name':row['name'],'job_id':row['job_id'],'error':repr(error)})
                    traceback.print_exc()
            atomic_json(root/'curves.json',curves)
            if curves!=last_curves:
                plot(root,curves);last_curves=curves
            atomic_json(root/'monitor.json',status);print(json.dumps(status),flush=True)
            if status['complete']:return
            time.sleep(45)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('prepare','preflight','submit','watch'))
    parser.add_argument('--root',type=Path,required=True)
    args=parser.parse_args();globals()[args.action](args.root.resolve())
