"""Prepare and monitor MOPD students from the plotted RL teacher banks."""

import argparse
import fcntl
import json
import math
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import yaml

from modular_rl.initializers import verify, load_registry
from modular_rl.mopd_export import audit as audit_hf, digest
from modular_rl.mopd_suite import mixed_data
from modular_rl.suite import shell_exports
from .config import atomic_json
from .math_billion_watch import retry
from .watch import states, TERMINAL_FAILURES

TASKS=('math','code','if','chat')
ALIASES={'base':dict.fromkeys(TASKS,'base'),
         'baseline-sft':dict.fromkeys(TASKS,'baseline'),
         'task-sft':{'math':'math','code':'code','if':'if20','chat':'chat'}}
PROVEN=Path('/nlp/scr/dhei/modular-mopd-rl/v1/mopd-followup-4k-20261029/students')
QUEUES='miso-lo,sphinx-lo,jag-lo,sc-loprio,miso,sphinx'
STEPS=4000
INTERVAL=256
PROJECT='modular-mopd-rl-v1'


def read(path):return json.loads(Path(path).read_text())


def configure(root):
    global STEPS,INTERVAL,PROJECT
    plan=read(root/'students/launch-plan.json')
    STEPS=plan['train_steps'];INTERVAL=plan['evaluation_interval']
    PROJECT=plan.get('wandb_project','modular-mopd-rl-v1')
    return plan


def select_teachers(snapshot):
    by_key={(r['initializer'],r['task']):r for r in snapshot['runs']}
    selected={}
    for bank,aliases in ALIASES.items():
        selected[bank]={}
        for task,alias in aliases.items():
            row=by_key[alias,task]
            if row['status']!='completed' or row.get('method')!='rl' or row.get('algorithm','grpo')!='grpo':
                raise ValueError('Require completed plotted GRPO teacher: '+row['name'])
            selected[bank][task]=row
    return selected


def teacher_plan(root,snapshot_path):
    snapshot=read(snapshot_path)
    selected=select_teachers(snapshot)
    teachers={}
    for bank,rows in selected.items():
        for task,row in rows.items():
            policy=Path(row['run_directory']).resolve()
            completed=read(policy/'completed.json')
            checkpoint=policy/'ckpt'
            step=int((checkpoint/'latest_checkpointed_iteration.txt').read_text())
            state=read(checkpoint/'rlve_checkpoint_state.json')
            if not (completed['completed'] and completed['train_rollouts']==8000
                    and completed['updates']==1000 and step==999
                    and state['completed'] and state['rollout_id']==999):
                raise ValueError('Plotted teacher lacks its final8K checkpoint: '+row['name'])
            identity=read(policy/'initial-model-identity.json')['initializer']
            hf=Path(identity['hf']).resolve()
            teachers[row['name']]={'name':row['name'],'task':task,'bank':bank,
                'plot_initializer':row['initializer'],'plot_label':row['label'],
                'wandb_run_id':row['wandb_run_id'],'teacher_run_directory':str(policy),
                'native_checkpoint':str(checkpoint),'checkpoint_step':step,'train_rollouts':8000,
                'common_sha256':digest(checkpoint/f'iter_{step:07d}/common.pt'),
                'original_hf':str(hf),'selection':str(policy.parents[2]/'selection'/task),
                'training_source':str(policy.parents[2]/'source'),
                'initializer_identity':identity,'final_plot_metrics':row['points'][-1]}
    if len(teachers)!=12:raise ValueError('Expected twelve distinct plotted teachers')
    plan={'snapshot_sha256':digest(snapshot_path),'snapshot_source':str(snapshot_path),
          'banks':{bank:{task:rows[task]['name'] for task in TASKS} for bank,rows in selected.items()},
          'teachers':teachers,'student_steps':STEPS,'student_rollouts':STEPS*8}
    atomic_json(root/'teacher-plan.json',plan)
    shutil.copy2(snapshot_path,root/'plot-snapshot.json')
    return plan


def export_teachers(root,plan,repo):
    tool=root/'export-source/tools/convert_torch_dist_to_hf.py'
    for relative in ('tools/convert_torch_dist_to_hf.py','slime/backends/megatron_utils/megatron_to_hf/qwen2.py'):
        dst=root/'export-source'/relative;dst.parent.mkdir(parents=True,exist_ok=True)
        if not dst.exists():shutil.copy2(repo/relative,dst)
    # The saved arguments contain unused TransformerEngine recipe classes.
    # CPU conversion needs their attributes, not the CUDA extension itself.
    text=tool.read_text()
    text=text.replace('mod_name.startswith("glm"):',
                      'mod_name.startswith("glm") or mod_name.startswith("transformer_engine"):')
    if 'from torch.distributed.checkpoint.metadata import StorageMeta' not in text:
        text=text.replace('import torch.distributed.checkpoint as dist_cp\n',
            'import torch.distributed.checkpoint as dist_cp\nfrom torch.distributed.checkpoint.metadata import StorageMeta\n')
    tool.write_text(text)
    atomic_json(root/'export-source/compatibility.json',{'cpu_only':True,
        'unused_transformer_engine_argument_types_stubbed':True,'storage_meta_imported':True,'converter_sha256':digest(tool)})
    for name,row in plan['teachers'].items():
        slot=root/'teachers/exports'/name;slot.mkdir(parents=True,exist_ok=True)
        marker=slot/'weights_only_complete.json'
        if marker.exists():
            receipt=read(marker)
            if receipt['common_sha256']!=row['common_sha256']:raise ValueError('Export identity changed: '+name)
            continue
        temporary=slot/'hf-exporting'
        target=slot/'hf-final'
        if target.exists():raise ValueError('Inspect an uncommitted teacher export: '+str(target))
        if not temporary.exists():
            command=[sys.executable,str(tool),'--input-dir',row['native_checkpoint']+'/iter_0000999',
                     '--output-dir',str(temporary),'--origin-hf-dir',row['original_hf']]
            env={**os.environ,'OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'1','TOKENIZERS_PARALLELISM':'false','PYTHONUNBUFFERED':'1'}
            print('Exporting teacher on CPU:',name,flush=True)
            with (slot/'export.log').open('w') as log:subprocess.run(command,check=True,env=env,stdout=log,stderr=subprocess.STDOUT)
        verified=audit_hf(temporary,Path(row['original_hf']))
        (temporary/'pretrained_model_identity.json').unlink(missing_ok=True)
        receipt={**row,'format':'hf-safetensors','path':'hf-final',**verified}
        atomic_json(temporary/'rl_teacher_identity.json',receipt)
        temporary.rename(target)
        atomic_json(marker,receipt)
        print('Verified teacher:',name,verified['parameter_count'],'parameters',flush=True)
    for bank,mapping in plan['banks'].items():
        directory=root/'teachers/banks'/bank;directory.mkdir(parents=True,exist_ok=True)
        for i,(task,name) in enumerate(mapping.items()):
            slot=directory/f'{i:02d}-{task}';slot.mkdir(exist_ok=True)
            export=root/'teachers/exports'/name
            for asset in ('hf-final','weights_only_complete.json'):
                link=slot/asset
                if not link.exists():link.symlink_to(export/asset,target_is_directory=asset=='hf-final')
        atomic_json(directory/'bank.json',mapping)


def replace_arg(args,flag,value):
    if args.count(flag)!=1:raise ValueError('Expected one flag: '+flag)
    args[args.index(flag)+1]=str(value)


def prepare_students(root,teacher_plan,repo):
    experiment=root/'students';source=experiment/'source'
    if (experiment/'launch-plan.json').exists():return read(experiment/'launch-plan.json')
    if (experiment/'jobs.json').exists():raise ValueError('Never modify a submitted experiment')
    old=read(PROVEN/'launch-plan.json')
    for relative,expected in old['source_sha256'].items():
        src=Path(old['source'])/relative;dst=source/relative;dst.parent.mkdir(parents=True,exist_ok=True)
        if digest(src)!=expected:raise ValueError('Proven MOPD source changed: '+relative)
        shutil.copy2(src,dst)
    shutil.copytree(PROVEN/'runtime',experiment/'runtime',dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    base_identity=teacher_plan['teachers'][teacher_plan['banks']['base']['math']]['initializer_identity']
    if base_identity['kind']!='pretrained-base' or base_identity['id']!='base':
        raise ValueError('Student must initialize from the pretrained Base release')
    verify(base_identity)
    registry=experiment/'initializers.json';atomic_json(registry,{'base':base_identity})
    selections={}
    for task in TASKS:
        teacher=teacher_plan['teachers'][teacher_plan['banks']['task-sft'][task]]
        source_selection=Path(teacher['selection'])
        canonical={split:[json.loads(line) for line in (source_selection/f'{split}.jsonl').open()]
                   for split in ('train','val')}
        # Teacher policy differences must not change student training/eval data.
        for bank in ALIASES:
            selection=Path(teacher_plan['teachers'][teacher_plan['banks'][bank][task]]['selection'])
            for split,rows in canonical.items():
                other=[json.loads(line) for line in (selection/f'{split}.jsonl').open()]
                keyed=lambda values:{v['metadata']['key']:v for v in values}
                if keyed(other)!=keyed(rows):raise ValueError('Teacher task pools differ: '+bank+'/'+task+'/'+split)
        keys={split:{r['metadata']['key'] for r in rows} for split,rows in canonical.items()}
        if keys['train']&keys['val'] or len(keys['val'])!=64:raise ValueError('Invalid task split: '+task)
        destination=experiment/'selection'/task
        shutil.copytree(source_selection,destination,dirs_exist_ok=True)
        selections[task]={'source':str(source_selection),'sha256':{s:digest(destination/f'{s}.jsonl') for s in canonical}}
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(base_identity['hf'],local_files_only=True)
    mixed_data(experiment/'selection',STEPS,tokenizer,enable_thinking=False)
    # Keep the completed MOPD trainer, and use the exact current task verifiers.
    assets={'math':['modular_rl/v2/math_reward.py'],
            'code':['modular_rl/graders.py','modular_rl/code_worker.py','modular_rl/v2/mbpp_spec.py'],
            'if':['modular_rl/if_grader.py'],
            'chat':['modular_rl/skywork.py','modular_rl/dolci/chat_reward.py']}
    changes={}
    for task,files in assets.items():
        teacher=teacher_plan['teachers'][teacher_plan['banks']['task-sft'][task]]
        for relative in files:
            src=Path(teacher['training_source'])/relative;dst=source/relative
            dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
            changes[relative]={'source':str(src),'sha256':digest(dst)}
    for relative in ('modular_rl/v2/rollout.py','modular_rl/v2/if_rollout.py'):
        rollout=source/relative;text=rollout.read_text()
        if text.count('enable_thinking=True')!=1:raise ValueError('Inspect proven prompt formatting: '+relative)
        rollout.write_text(text.replace('enable_thinking=True','enable_thinking=False'))
    config_path=source/'modular_rl/mopd-config.yaml';config=yaml.safe_load(config_path.read_text())
    config.update({'num-rollout':STEPS,'eval-interval':INTERVAL,'eval-temperature':.7,'eval-top-p':.95})
    config.pop('early-stop-val-accuracy',None)
    config_path.write_text(yaml.safe_dump(config,sort_keys=False))
    allocation=source/'modular_rl/mopd_allocation.sh'
    text=allocation.read_text()
    if text.count('WANDB_GROUP=modular-mopd')!=1:raise ValueError('Inspect MOPD run grouping')
    text=text.replace('WANDB_GROUP=modular-mopd','WANDB_GROUP="${MODULAR_RL_GROUP:-modular-mopd}"')
    anchor='mkdir -p "$RLVE_RUN_DIR"\n'
    if text.count(anchor)!=1:raise ValueError('Inspect MOPD recovery insertion point')
    allocation.write_text(text.replace(anchor,anchor+'python3 -m modular_rl.mopd_recovery --run "$RLVE_RUN_DIR"\n'))
    for relative in ('modular_rl/mopd_recovery.py','modular_rl/v2/recovery.py','modular_rl/initializers.py'):
        shutil.copy2(repo/relative,source/relative)
    original_script=Path(old['runs'][0]['command'][-1])
    line=next(l for l in original_script.read_text().splitlines() if l.startswith('export MODULAR_RL_EXTRA_ARGS='))
    original_args=shlex.split(shlex.split(line.split('=',1)[1])[0])
    jobs=[]
    for bank,mapping in teacher_plan['banks'].items():
        name='mopd-'+bank+'-4k';output=experiment/'runs'/name;output.mkdir(parents=True,exist_ok=True)
        args=list(original_args)
        for flag,value in (('--num-rollout',STEPS),('--eval-interval',INTERVAL),
                           ('--opd-teacher-dir',root/'teachers/banks'/bank)):
            replace_arg(args,flag,value)
        args+=['--eval-temperature','.7','--eval-top-p','.95']
        env={'RLVE_REPO':source,'RLVE_SCRATCH':root.parent,
            'RLVE_IMAGE':root.parent/'images/slime-v0.5.0rc0-cu126.sif','RLVE_WANDB_TEAM':'david-heineman',
            'WANDB_BASE_URL':'https://api.forge.coreweave.com','MODULAR_RL_PROFILE':'v2',
            'MODULAR_RL_TASKS':','.join(TASKS),'MODULAR_RL_INITIALIZER':'base',
            'MODULAR_RL_INITIALIZERS':registry,'MODULAR_RL_INITIALIZERS_SHA256':digest(registry),
            'MODULAR_RL_RUN_NAME':name,'MODULAR_RL_WANDB_PROJECT':'modular-mopd-rl-v1',
            'MODULAR_RL_RUNTIME':experiment/'runtime','MODULAR_RL_SELECTION':experiment/'selection',
            'MODULAR_RL_DATA':root.parent/'modular-mopd-rl/v2/data','MODULAR_RL_OUTPUT':output,
            'MODULAR_RL_HF_HOME':root.parent/'tinker-mopd/hf','MODULAR_RL_VALIDATION_PROMPTS':64,
            'MODULAR_RL_PROMPTS_PER_UPDATE':8,'MODULAR_RL_CHECKPOINT_INTERVAL':64,
            'MODULAR_RL_MAX_RESPONSE_TOKENS':2048,'MODULAR_RL_BASE_TOKENIZER':1,
            'MODULAR_RL_SKIP_ZERO_ADVANTAGE':0,'MODULAR_RL_GROUP':'mopd-modular-v2-4k',
            'MODULAR_RL_TEACHER_BANK':root/'teachers/banks'/bank,'MODULAR_RL_CUDA_CACHE_CLEAR_INTERVAL':1,
            'MODULAR_RL_POLICY_GPUS':4,'MODULAR_RL_TEACHER_GPUS':2,'MODULAR_RL_REWARD_GPUS':2,
            'MODULAR_RL_UPDATE_PROBE_INTERVAL':10,'MODULAR_RL_EXTRA_ARGS':shlex.join(args)}
        script=output/'run.sbatch'
        script.write_text('#!/usr/bin/env bash\n#SBATCH --account=miso\n#SBATCH --partition='+QUEUES+
            '\n#SBATCH --nodes=1\n#SBATCH --ntasks=1\n#SBATCH --requeue\n#SBATCH --open-mode=append\nset -euo pipefail\n'+
            shell_exports(env)+shell_exports({'APPTAINERENV_'+k:v for k,v in env.items()})+
            'export APPTAINERENV_WANDB_API_KEY="${WANDB_API_KEY:?}"\nsource "$RLVE_REPO/scripts/slurm/common.sh"\n'+
            'rlve_apptainer bash modular_rl/mopd_allocation.sh "$@"\n')
        subprocess.run(['bash','-n',str(script)],check=True)
        command=['sbatch','--parsable','--job-name='+name,'--gres=gpu:8','--constraint=ampere|ada|hopper',
            '--cpus-per-task=52','--mem=192G','--time=23:50:00','--exclude=viscam10,jagupard33,jagupard34,pasteur6',
            '--chdir='+str(source),'--output='+str(output/'slurm-%j.out'),
            '--export=ALL,WANDB_SERVICE=,NCCL_P2P_DISABLE=1,NCCL_CUMEM_HOST_ENABLE=0',str(script)]
        jobs.append({'name':name,'bank':bank,'teachers':mapping,'teacher_bank':str(root/'teachers/banks'/bank),
            'gpus':8,'student_gpus':4,'teacher_gpus':2,'reward_gpus':2,'output':str(output),
            'command':command,'script_sha256':digest(script)})
    result={'status':'prepared-not-submitted','runs':jobs,'source':str(source),'initializers':str(registry),
        'student_initializer':'Qwen/Qwen3-0.6B-Base','initializers_sha256':digest(registry),
        'train_steps':STEPS,'train_rollouts':STEPS*8,'prompts_per_step':8,'samples_per_prompt':1,
        'prompts_per_task_per_step':2,'objective_passes':1,'max_response_tokens':2048,
        'lr':2e-6,'weight_decay':.01,'enable_thinking':False,'checkpoint_interval':64,
        'evaluation_interval':INTERVAL,'validation_prompts_per_task':64,'validation_samples_per_prompt':8,
        'evaluation_temperature':.7,'evaluation_top_p':.95,'early_stop_disabled':True,
        'simultaneous_gpus':24,'dependencies':[],'objective':'native sampled reverse-KL MOPD; rewards used only for metrics',
        'proven_source':str(PROVEN),'verifier_sources':changes,'task_selections':selections,
        'selection_sha256':{n:digest(experiment/'selection'/n) for n in ('mixed-train.jsonl','mixed-val.jsonl','mixed-selection.json')},
        'source_sha256':{str(p.relative_to(source)):digest(p) for p in sorted(source.rglob('*'))
                         if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc'}}
    atomic_json(experiment/'launch-plan.json',result)
    return result


def validate_tensor_shapes(model,actual):
    expected={k:list(v.shape) for k,v in model.state_dict().items()}
    aliases=defaultdict(list)
    for name,parameter in model.named_parameters(remove_duplicate=False):aliases[id(parameter)].append(name)
    omitted=[]
    for names in aliases.values():
        if any(name in actual for name in names):
            for name in names:
                if name not in actual:
                    omitted.append(name);expected.pop(name)
    if actual!=expected:raise ValueError('HF model key/shape mismatch')
    return omitted


def model_contracts(root):
    """Check all exported names/shapes against a storage-free HF model."""
    import torch
    from safetensors import safe_open
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    base=read(root/'students/initializers.json')['base']['hf']
    student_tokenizer=AutoTokenizer.from_pretrained(base,local_files_only=True)
    vocabulary=student_tokenizer.get_vocab()
    receipts={}
    for name in read(root/'teacher-plan.json')['teachers']:
        directory=root/'teachers/exports'/name/'hf-final'
        config=AutoConfig.from_pretrained(directory,local_files_only=True)
        tokenizer=AutoTokenizer.from_pretrained(directory,local_files_only=True)
        if (tokenizer.get_vocab()!=vocabulary or tokenizer.chat_template!=student_tokenizer.chat_template
                or tokenizer.eos_token_id!=student_tokenizer.eos_token_id):
            raise ValueError('Teacher/student tokenizer mismatch: '+name)
        with torch.device('meta'):
            model=AutoModelForCausalLM.from_config(config)
        index=read(directory/'model.safetensors.index.json')['weight_map']
        actual={}
        for filename in set(index.values()):
            with safe_open(directory/filename,framework='pt',device='cpu') as weights:
                actual.update({key:weights.get_slice(key).get_shape() for key in weights.keys()})
        omitted=validate_tensor_shapes(model,actual)
        receipts[name]={'passed':True,'tensor_count':len(actual),'architecture':type(model).__name__,
                        'tokenizer_matches_student':True,'omitted_tied_parameters':omitted}
        del model,tokenizer
    atomic_json(root/'teacher-model-contracts.json',receipts)
    first=next(iter(receipts))
    model,loading=AutoModelForCausalLM.from_pretrained(root/'teachers/exports'/first/'hf-final',
        torch_dtype=torch.bfloat16,local_files_only=True,low_cpu_mem_usage=True,output_loading_info=True)
    if any(loading.get(k) for k in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs')):
        raise ValueError('Representative HF model failed to load cleanly')
    with torch.no_grad():logits=model(**student_tokenizer('2+2',return_tensors='pt')).logits
    if not torch.isfinite(logits).all():raise ValueError('Nonfinite teacher forward pass')
    atomic_json(root/'cpu-model-load-check.json',{'passed':True,'teacher':first,
        'loading_info':{k:sorted(loading.get(k,[])) for k in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs')},
        'finite_logits':True,'logits_shape':list(logits.shape),'device':'cpu'})


def preflight(root):
    plan=configure(root)
    alias=plan.get('student_initializer_alias','base')
    identity=verify(load_registry(plan['initializers'],plan['initializers_sha256'])[alias])
    if identity['id']!=alias:raise ValueError('Student initializer alias and identity disagree')
    if plan.get('student_initializer_hf') and Path(identity['hf']).resolve()!=Path(plan['student_initializer_hf']).resolve():
        raise ValueError('Student initializer model differs from the experiment plan')
    source=Path(plan['source'])
    config=yaml.safe_load((source/'modular_rl/mopd-config.yaml').read_text())
    if (config['num-rollout']!=STEPS or config['eval-interval']!=INTERVAL
            or 'early-stop-val-accuracy' in config):raise ValueError('Incorrect step/evaluation/stop budget')
    if 'python3 -m modular_rl.mopd_recovery --run' not in (source/'modular_rl/mopd_allocation.sh').read_text():
        raise ValueError('Missing preemption recovery hook')
    for relative in ('modular_rl/v2/rollout.py','modular_rl/v2/if_rollout.py'):
        text=(source/relative).read_text()
        if 'enable_thinking=True' in text or 'enable_thinking=False' not in text:
            raise ValueError('Inconsistent prompt mode: '+relative)
    data=events(root/'students/selection/mixed-train.jsonl')
    if len(data)!=STEPS*8:raise ValueError(f'Expected {STEPS*8} training prompts')
    validation=events(root/'students/selection/mixed-val.jsonl')
    if len(validation)!=4*64 or {v['metadata']['key'] for v in data}&{v['metadata']['key'] for v in validation}:
        raise ValueError('Invalid validation count or cross-task train/validation overlap')
    for start in range(0,len(data),8):
        if Counter(row['metadata']['task'] for row in data[start:start+8])!=Counter(dict.fromkeys(TASKS,2)):
            raise ValueError('Incorrect task mix in prepared batch')
    for relative,sha in plan['source_sha256'].items():
        if digest(Path(plan['source'])/relative)!=sha:raise ValueError('Frozen source changed: '+relative)
    for name,sha in plan['selection_sha256'].items():
        if digest(root/'students/selection'/name)!=sha:raise ValueError('Prepared selection changed: '+name)
    checks=[]
    with tempfile.TemporaryDirectory(prefix='mopd-curve-preflight-') as temporary:
        for row in plan['runs']:
            if digest(Path(row['command'][-1]))!=row['script_sha256']:raise ValueError('Launcher changed')
            exports=validate_gpu_roles(row,config,PROJECT)
            if exports.get('MODULAR_RL_INITIALIZER')!=alias:
                raise ValueError('Launcher initializes a different student')
            line=next(v for v in Path(row['command'][-1]).read_text().splitlines() if v.startswith('export MODULAR_RL_EXTRA_ARGS='))
            args=shlex.split(shlex.split(line.split('=',1)[1])[0])
            last=lambda flag:args[max(i for i,v in enumerate(args) if v==flag)+1]
            if plan.get('eos_alignment'):
                validate_eos_launch(args,exports,
                    student_eos=row.get('student_eos_token_id',151643),
                    teacher_eos=row.get('teacher_eos_token_id',151645))
            if (last('--num-rollout')!=str(STEPS) or last('--eval-interval')!=str(INTERVAL)
                    or last('--train-reuse-epochs')!='1'
                    or last('--opd-teacher-gpus')!=str(row['teacher_gpus'])
                    or last('--custom-loss-function-path')!='slime.backends.megatron_utils.opd_loss.opd_loss_function'):
                raise ValueError('Incorrect MOPD learning objective or step budget')
            if read(Path(row['teacher_bank'])/'bank.json')!=row['teachers']:raise ValueError('Wrong teacher bank')
            for i,(task,name) in enumerate(row['teachers'].items()):
                marker=read(Path(row['teacher_bank'])/f'{i:02d}-{task}/weights_only_complete.json')
                if marker['name']!=name or marker['checkpoint_step']!=999 or marker['train_rollouts']!=8000:
                    raise ValueError('Wrong teacher identity')
            text=Path(row['command'][-1]).read_text().replace('rlve_apptainer bash modular_rl/mopd_allocation.sh',
                'rlve_apptainer_cpu bash modular_rl/mopd_allocation.sh')
            script=Path(temporary)/row['name'];script.write_text(text)
            process=subprocess.run(['bash',str(script),'--check-environment'],capture_output=True,text=True)
            checks.append({'name':row['name'],'returncode':process.returncode,'output':process.stdout+process.stderr})
            atomic_json(root/'container-preflight.json',checks)
            if process.returncode:raise ValueError('Container preflight failed: '+row['name'])
    atomic_json(root/'preflight.json',{'passed':True,'teacher_exports':len(read(root/'teacher-plan.json')['teachers']),'student_steps':STEPS,
        'student_rollouts_per_run':STEPS*8,'gpus_per_run':plan['runs'][0]['gpus'],
        'simultaneous_gpus':sum(row['gpus'] for row in plan['runs']),'wandb_project':PROJECT,
        'shell_checks_passed':len(checks),'balanced_training_data_verified':True,
        'eos_alignment_enabled':bool(plan.get('eos_alignment')),
        'student_initializer_alias':alias,'student_initializer_hf':identity['hf'],
        'validation_questions_globally_disjoint_from_training':True,
        'native_training_not_launched':not (root/'students/jobs.json').exists()})


def validate_gpu_roles(row,config,project):
    if sum(row[k] for k in ('student_gpus','teacher_gpus','reward_gpus'))!=row['gpus']:
        raise ValueError('GPU role counts do not match allocation')
    if (config['actor-num-gpus-per-node']!=row['student_gpus']
            or config['num-gpus-per-node']!=row['student_gpus']
            or '--gres=gpu:'+str(row['gpus']) not in row['command']):
        raise ValueError('Native GPU topology does not match allocation')
    exports={}
    for line in Path(row['command'][-1]).read_text().splitlines():
        if line.startswith('export ') and not line.startswith('export APPTAINERENV_'):
            key,value=line[7:].split('=',1);exports[key]=shlex.split(value)[0]
    expected={'MODULAR_RL_WANDB_PROJECT':project,'MODULAR_RL_POLICY_GPUS':row['student_gpus'],
              'MODULAR_RL_TEACHER_GPUS':row['teacher_gpus'],'MODULAR_RL_REWARD_GPUS':row['reward_gpus']}
    if any(exports.get(k)!=str(v) for k,v in expected.items()):
        raise ValueError('Launcher project or GPU roles do not match plan')
    return exports


def validate_eos_launch(args,exports,*,student_eos=151643,teacher_eos=151645):
    def value(flag):
        if args.count(flag)!=1:raise ValueError('Expected one v2 flag: '+flag)
        return args[args.index(flag)+1]
    if (value('--opd-student-eos-token-id')!=str(student_eos) or value('--opd-teacher-eos-token-id')!=str(teacher_eos)
            or exports.get('MODULAR_RL_STUDENT_EOS_TOKEN_ID',str(student_eos))!=str(student_eos)
            or exports.get('MODULAR_RL_TEACHER_EOS_TOKEN_ID',str(teacher_eos))!=str(teacher_eos)
            or exports.get('MODULAR_RL_CODE_FORCING_FALLBACK')!='1'
            or exports.get('MODULAR_RL_EOS_ALIGNMENT_REQUIRED')!='1'):
        raise ValueError('Missing v2 EOS alignment or code fallback')
    value('--rollout-stop-token-ids')
    start=args.index('--rollout-stop-token-ids')+1;ids=[]
    for item in args[start:]:
        if item.startswith('--'):break
        ids.append(item)
    if set(ids)!={'151643','151645'}:raise ValueError('Both EOS aliases must stop generation')


def prepare(root,snapshot,repo):
    root.mkdir(parents=True,exist_ok=True)
    teacher_path=root/'teacher-plan.json'
    plan=read(teacher_path) if teacher_path.exists() else teacher_plan(root,snapshot)
    if digest(snapshot)!=plan['snapshot_sha256']:raise ValueError('Plot snapshot changed during preparation')
    export_teachers(root,plan,repo)
    prepare_students(root,plan,repo)
    model_contracts(root)
    preflight(root)
    write_launch_files(root,repo)
    print('Prepared all three4,000-step runs; no GPU training jobs submitted.',flush=True)


def resize_prepared(previous,root,repo,project):
    """Reuse verified teachers/data for the requested 4-GPU, 8K-response jobs."""
    if root.exists():raise ValueError('Resize requires a new experiment directory')
    if (previous/'students/jobs.json').exists():raise ValueError('Previous plan was already submitted')
    old=read(previous/'students/launch-plan.json')
    for relative,sha in old['source_sha256'].items():
        if digest(Path(old['source'])/relative)!=sha:raise ValueError('Prepared source changed: '+relative)
    root.mkdir(parents=True)
    for name in ('teacher-plan.json','plot-snapshot.json','teacher-model-contracts.json',
                 'cpu-model-load-check.json','test-results.json'):
        shutil.copy2(previous/name,root/name)
    teachers=read(root/'teacher-plan.json')
    teachers.update(student_steps=1000,student_rollouts=8000,reused_verified_exports=str(previous),wandb_project=project)
    atomic_json(root/'teacher-plan.json',teachers)
    # Hard links preserve verified weights without consuming another 14 GB.
    shutil.copytree(previous/'teachers/exports',root/'teachers/exports',copy_function=os.link)
    for bank,mapping in teachers['banks'].items():
        directory=root/'teachers/banks'/bank;directory.mkdir(parents=True)
        for i,(task,name) in enumerate(mapping.items()):
            slot=directory/f'{i:02d}-{task}';slot.mkdir()
            for asset in ('hf-final','weights_only_complete.json'):
                (slot/asset).symlink_to(root/'teachers/exports'/name/asset,target_is_directory=asset=='hf-final')
        atomic_json(directory/'bank.json',mapping)
    experiment=root/'students';source=experiment/'source'
    for name in ('source','runtime','selection'):
        shutil.copytree(previous/'students'/name,experiment/name,
                        ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    registry=experiment/'initializers.json';shutil.copy2(previous/'students/initializers.json',registry)
    shutil.copy2(repo/'modular_rl/initializers.py',source/'modular_rl/initializers.py')
    data=experiment/'selection/mixed-train.jsonl'
    with (previous/'students/selection/mixed-train.jsonl').open() as src,data.open('w') as dst:
        for i,line in enumerate(src):
            if i==8000:break
            dst.write(line)
    selection=read(experiment/'selection/mixed-selection.json');selection['train_rollouts']=8000
    for counts in selection['counts'].values():counts['train']=2000
    selection['sha256']['train']=digest(data)
    atomic_json(experiment/'selection/mixed-selection.json',selection)
    config_path=source/'modular_rl/mopd-config.yaml';config=yaml.safe_load(config_path.read_text())
    config.update({'actor-num-gpus-per-node':2,'num-gpus-per-node':2,'num-rollout':1000})
    config_path.write_text(yaml.safe_dump(config,sort_keys=False))
    allocation=source/'modular_rl/mopd_allocation.sh'
    text=(repo/'modular_rl/mopd_allocation.sh').read_text()
    text=text.replace('WANDB_GROUP=modular-mopd','WANDB_GROUP="${MODULAR_RL_GROUP:-modular-mopd}"')
    anchor='mkdir -p "$RLVE_RUN_DIR"\n'
    if text.count(anchor)!=1:raise ValueError('Inspect allocation recovery insertion')
    allocation.write_text(text.replace(anchor,anchor+'python3 -m modular_rl.mopd_recovery --run "$RLVE_RUN_DIR"\n'))
    jobs=[]
    for row in old['runs']:
        env={}
        for line in Path(row['command'][-1]).read_text().splitlines():
            if line.startswith('export ') and not line.startswith('export APPTAINERENV_'):
                key,value=line[7:].split('=',1);env[key]=shlex.split(value)[0]
        bank=row['bank'];name='mopd-'+bank+'-8k';output=experiment/'runs'/name;output.mkdir(parents=True)
        env={k:v.replace(str(previous),str(root)) for k,v in env.items()}
        args=shlex.split(env['MODULAR_RL_EXTRA_ARGS'])
        replace_arg(args,'--num-rollout',1000);replace_arg(args,'--opd-teacher-gpus',1)
        env.update(RLVE_REPO=str(source),MODULAR_RL_INITIALIZERS=str(registry),
            MODULAR_RL_RUN_NAME=name,MODULAR_RL_OUTPUT=str(output),MODULAR_RL_WANDB_PROJECT=project,
            MODULAR_RL_GROUP='mopd-modular-v2-8k',MODULAR_RL_POLICY_GPUS=2,
            MODULAR_RL_TEACHER_GPUS=1,MODULAR_RL_REWARD_GPUS=1,MODULAR_RL_EXTRA_ARGS=shlex.join(args))
        script=output/'run.sbatch'
        script.write_text('#!/usr/bin/env bash\n#SBATCH --account=miso\n#SBATCH --partition='+QUEUES+
            '\n#SBATCH --nodes=1\n#SBATCH --ntasks=1\n#SBATCH --requeue\n#SBATCH --open-mode=append\nset -euo pipefail\n'+
            shell_exports(env)+shell_exports({'APPTAINERENV_'+k:v for k,v in env.items()})+
            'export APPTAINERENV_WANDB_API_KEY="${WANDB_API_KEY:?}"\nsource "$RLVE_REPO/scripts/slurm/common.sh"\n'+
            'rlve_apptainer bash modular_rl/mopd_allocation.sh "$@"\n')
        subprocess.run(['bash','-n',str(script)],check=True)
        command=['sbatch','--parsable','--job-name='+name,'--gres=gpu:4','--constraint=ampere|ada|hopper',
            '--cpus-per-task=28','--mem=128G','--time=12:00:00','--exclude=viscam10,jagupard33,jagupard34,pasteur6',
            '--chdir='+str(source),'--output='+str(output/'slurm-%j.out'),
            '--export=ALL,WANDB_SERVICE=,NCCL_P2P_DISABLE=1,NCCL_CUMEM_HOST_ENABLE=0',str(script)]
        jobs.append({**row,'name':name,'gpus':4,'student_gpus':2,'teacher_gpus':1,'reward_gpus':1,
            'teacher_bank':str(root/'teachers/banks'/bank),'output':str(output),'command':command,
            'script_sha256':digest(script)})
    plan={**old,'runs':jobs,'source':str(source),'initializers':str(registry),'wandb_project':project,
        'train_steps':1000,'train_rollouts':8000,'simultaneous_gpus':12,'estimated_wall_hours':[3,6],
        'resized_from':str(previous),'selection_sha256':{n:digest(experiment/'selection'/n)
            for n in ('mixed-train.jsonl','mixed-val.jsonl','mixed-selection.json')},
        'source_sha256':{str(p.relative_to(source)):digest(p) for p in sorted(source.rglob('*'))
            if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc'}}
    atomic_json(experiment/'launch-plan.json',plan)
    preflight(root);write_launch_files(root,repo)
    atomic_json(previous/'superseded.json',{'replacement':str(root),'reason':'User requested 4 GPUs and 8K rollouts'})
    (previous/'submit-all.sh').write_text('#!/usr/bin/env bash\nset -euo pipefail\n'+
        'echo '+shlex.quote('Superseded; use '+str(root/'submit-all.sh'))+' >&2\nexit 1\n')


def write_launch_files(root,repo):
    configure(root)
    controller=root/'controller-source'
    if not controller.exists():
        shutil.copytree(root.parent/'math-groups-20261102/controller-source',controller,
                        ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    for relative in ('modular_rl/dolci/mopd_from_curves.py','modular_rl/mopd_suite.py'):
        shutil.copy2(repo/relative,controller/relative)
    python='/nlp/scr/dhei/tinker-mopd/venv-rl/bin/python'
    (root/'submit-all.sh').write_text('#!/usr/bin/env bash\nset -euo pipefail\n'+
        'export PYTHONPATH='+shlex.quote(str(controller))+'\nunset WANDB_SERVICE\nexec '+python+
        ' -u -m modular_rl.dolci.mopd_from_curves submit --root '+shlex.quote(str(root))+'\n')
    (root/'watch.sbatch').write_text('#!/usr/bin/env bash\nset -euo pipefail\n'+
        'export PYTHONPATH='+shlex.quote(str(controller))+'\nexport PYTHONUNBUFFERED=1\n'+
        'export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1\nunset WANDB_SERVICE\n'+
        'exec /juice5b/u/dhei/tatsu/nano-omni/.venv-midtraining/bin/python -u '+
        '-m modular_rl.dolci.mopd_from_curves watch --root '+shlex.quote(str(root))+'\n')
    plan=read(root/'students/launch-plan.json');teachers=read(root/'teacher-plan.json')
    if plan['train_steps']!=4000 or plan['runs'][0]['gpus']!=8:
        write_run_description(root,repo,plan,teachers)
        return
    lines=['# MOPD from the three training-curve teacher banks','',
        'Prepared only. No MOPD training jobs have been submitted. The group-16 and group-32 RL jobs were left unchanged.','',
        '| Student | Teachers selected from training_curves.png | Steps | Student responses | GPUs |',
        '| --- | --- | ---: | ---: | ---: |']
    for row in plan['runs']:
        label={'base':'Blue: RL on base','baseline-sft':'Orange: RL on baseline SFT',
               'task-sft':'Dark green: RL on task SFT'}[row['bank']]
        lines.append(f"| {row['name']} | {label} | 4,000 | 32,000 | 8 |")
    lines+=['','Every student starts from the same pinned Qwen/Qwen3-0.6B-Base release. '
        'Each step samples eight prompts, exactly two each from math/code/IF/chat, with one response per prompt. '
        'The native sampled reverse-KL MOPD objective uses frozen per-task teacher log probabilities; '
        'environment rewards are metrics and do not enter the loss. All student prompt sequences and '
        'evaluation panels are identical across the three runs.','',
        'One optimizer pass per step, LR 2e-6, weight decay 0.01, and a 2,048-token generation cap. '
        'All four tasks use the same empty-thinking prompt mode across banks. Math/code/chat answer forcing '
        'stays inside the cap; IF adds no forcing text. Validation uses 64 questions × 8 samples per task '
        'every 256 steps, plus initial and final evaluations. This schedules 34,816 evaluation responses '
        'against 32,000 training responses per student. Save resumable checkpoints every 64 steps; '
        'keep only the latest committed checkpoint. Accuracy early stopping is disabled to honor 4,000 steps.','',
        'Each independent job requests 4 student + 2 teacher + 2 Skywork GPUs, with no dependencies. '
        'All three together request 24 GPUs across low-priority queues. W&B project: modular-mopd-rl-v1; '
        'group: mopd-modular-v2-4k. The completed A100 MOPD runs took roughly 10–11 seconds per training step; '
        'budget approximately 12–20 hours per student, or 288–480 GPU hours total, excluding queue time. '
        'This estimate depends on hardware and generated response lengths.','',
        '## Exact final 8K RL teachers','',
        '| Task | Blue bank | Orange bank | Dark green bank |','| --- | --- | --- | --- |']
    for task in TASKS:
        names=[teachers['banks'][bank][task] for bank in ALIASES]
        lines.append('| '+task+' | '+' | '.join(names)+' |')
    lines+=['','The dark-green IF teacher is IF20, not the older chat-SFT IF run. '
        'The code teachers are the paired GRPO runs on repaired MBPP contracts. Dark-green code uses '
        'the original code-heavy SFT initializer at dolci-mix-20261029/sft/code/model; the separate '
        'fresh-Python SFT model is not plotted and is not substituted into this bank. '
        'The dark-green math teacher is the refreshed 1B-token SFT + original GRPO checkpoint actually drawn '
        'in training_curves.png; the group-16 and group-32 math runs are separate experiments. '
        'Every teacher is the final checkpoint at step 999 after 8,000 RL responses.','',
        'The blue math curve retains its historical decoding protocol; all new students share one '
        'decoding protocol. Teacher identity records preserve the plotted provenance.','',
        '## Validation and launch','',
        'teacher-plan.json and plot-snapshot.json pin the plotted teachers. teachers/exports contains '
        'CPU-converted HF models with finite tensors and 596,049,920 parameters each. '
        'All twelve model layouts and tokenizers match the student, and a representative CPU load/forward check passed. '
        'students/launch-plan.json pins scripts, source checksums, data checksums, and GPU roles. '
        'container-preflight.json records each verified native container environment. '
        'GPU learning steps have not been tested in this prepare-only stage.','',
        'The frozen trainer is cloned from the completed MOPD experiment at '+str(PROVEN)+'. '
        'Task verifiers come from the corresponding plotted task-specific RL snapshots. '
        'The controller retries terminal failures at most 4 times, audits all 4,000 steps and teacher routes, '
        'finalizes W&B summaries, and writes unsmoothed mopd_training_curves.png/.pdf in '
        'rlve/plot/mopd-modular-v2/.','',
        'Run the prepared submission script to launch all three independent students and their monitor:','',
        '```bash','bash '+str(root/'submit-all.sh'),'```','']
    document='\n'.join(lines)
    (root/'README.md').write_text(document)
    output=repo/'plot/mopd-modular-v2/mopd_experiment.md';output.write_text(document)
    (root/'AGENT_STATE.md').write_text('# Prepared MOPD experiment, not launched\n\n'+
        'Read README.md, teacher-plan.json, students/launch-plan.json and preflight.json. '
        'Three student jobs are configured for 4,000 steps / 32,000 rollouts each, starting from '
        'Qwen/Qwen3-0.6B-Base, using the exact plotted blue/orange/dark-green RL teachers. '
        'Do not substitute group-32 math automatically. No training jobs submitted during preparation. '
        'submit-all.sh launches students and their independent CPU monitor; no dependencies.\n')


def write_run_description(root,repo,plan,teachers):
    submitted=(root/'students/jobs.json').exists()
    status='Submitted' if submitted else 'Prepared, not submitted'
    total=sum(v['gpus'] for v in plan['runs']);roles=plan['runs'][0]
    hours=plan.get('estimated_wall_hours',[3,6])
    evaluations=len({0,STEPS-1,*range(INTERVAL-1,STEPS,INTERVAL)})
    lines=['# MOPD: plotted RL teacher banks','',status+'.','',
        '| Student | Teachers | Steps | Student rollouts | GPUs |',
        '| --- | --- | ---: | ---: | ---: |']
    labels={'base':'Blue: RL on base','baseline-sft':'Orange: RL on baseline SFT','task-sft':'Dark green: RL on task SFT'}
    for row in plan['runs']:
        lines.append(f"| {row['name']} | {labels[row['bank']]} | {STEPS:,} | {STEPS*8:,} | {row['gpus']} |")
    student=plan.get('student_initializer','Qwen/Qwen3-0.6B-Base')
    lines+=['',f"W&B project: **{PROJECT}**. All students initialize from the same pinned **{student}**.",
        '',f"Each run uses {roles['student_gpus']} student GPUs, {roles['teacher_gpus']} frozen-teacher GPU, "
        f"and {roles['reward_gpus']} Skywork GPU: {roles['gpus']} GPUs per job, {total} concurrently. "
        'The jobs submit independently across low-priority queues with no dependencies.',
        '',f"Budget: {STEPS:,} optimizer steps × 8 prompts × 1 response = {STEPS*8:,} student rollouts. "
        'Every batch contains two prompts each from math, code, IF, and chat. All banks use the same prompt order. '
        'The native sampled reverse-KL MOPD objective uses teacher log probabilities; environment rewards are metrics only. '
        'One optimizer pass, LR 2e-6, weight decay 0.01, a 2,048-token generation cap, and empty thinking prefill.',
        '',f"Validation: 64 questions × 8 samples per task every {INTERVAL} steps, plus initial/final evaluations "
        f"({evaluations} panels, {evaluations*2048:,} evaluation responses). Evaluation temperature 0.7, top-p 0.95. "
        'Resumable checkpoints save every 64 steps; uncommitted traces are archived before recovery. '
        'Accuracy early stopping is disabled to honor the full rollout budget.',
        '',f"Runtime estimate: {hours[0]}–{hours[1]} hours per run, excluding queue time; "
        f"approximately {total*hours[0]}–{total*hours[1]} GPU hours total. Based on the completed four-GPU MOPD run; "
        'hardware and generated response lengths affect wall time.',
        '', '## Exact teacher checkpoints','',
        '| Task | '+' | '.join(teachers['banks'])+' |','| --- | '+' | '.join('---' for _ in teachers['banks'])+' |']
    for task in TASKS:lines.append('| '+task+' | '+' | '.join(mapping[task] for mapping in teachers['banks'].values())+' |')
    lines+=['','These are the exact final 8K RL checkpoints drawn in training_curves.png. '
        'The task bank uses IF20 and the original plotted code-SFT model on repaired MBPP contracts. '
        'Its math teacher is the refreshed math-SFT + original GRPO checkpoint; group-16/group-32 runs are separate. '
        'The previously verified exports are reused without reconversion.',
        '', 'Teacher tensors, vocabulary, model layouts, representative CPU forward, balanced data, '
        'source checksums and container environments are checked in the JSON receipts. Native GPU startup '
        'is checked after submission. The monitor retries failures at most four times, audits completions, '
        'finalizes summaries in the requested W&B project, and updates `mopd_training_curves.png` without smoothing.',
        '', '## Launch','', '```bash','bash '+str(root/'submit-all.sh'),'```','']
    if plan.get('eos_alignment'):
        lines+=['## V2 termination fixes','',
            'Both 151643 (`<|endoftext|>`) and 151645 (`<|im_end|>`) stop all train/eval generation. '
            'Teacher EOS output labels align with the selected student initializer; the per-run plan records the mapping. '
            'Code uses a syntax-only fallback when the forced final implementation is malformed. '
            'Each student starts fresh from the selected initializer. Frozen teachers, LR and task balance are pinned in the plan. '
            'Per-task token lengths, truncation, forcing and stop-token fractions are logged, along with terminal '
            'teacher/student log probabilities. Historical runs keep their original source and metrics.','']
    doc='\n'.join(lines)
    (root/'README.md').write_text(doc)
    (repo/'plot/mopd-modular-v2'/plan.get('description_filename','mopd_experiment.md')).write_text(doc)
    (root/'AGENT_STATE.md').write_text(f'# {status}: MOPD\n\nRead README.md and students/launch-plan.json. '
        f"{len(plan['runs'])} students, {STEPS*8:,} rollouts/{STEPS:,} steps each, {total} GPUs total, project {PROJECT}. "
        'Exact plotted teacher banks are pinned; do not replace math with group32 automatically. '
        'Keep native snapshots immutable after submission. Monitor records are in watch-job.json and monitor.json.\n')


def submit(root):
    if (root/'superseded.json').exists():raise ValueError('This launch plan has been superseded')
    preflight(root)
    plan=read(root/'students/launch-plan.json')
    jobs=read(root/'students/jobs.json') if (root/'students/jobs.json').exists() else []
    submitted={row['name'] for row in jobs}
    for row in plan['runs']:
        if row['name'] in submitted:continue
        job=subprocess.check_output(row['command'],text=True).strip().split(';')[0]
        jobs.append({**row,'job_id':job});atomic_json(root/'students/jobs.json',jobs)
        print('Submitted',row['name'],job,flush=True)
    if not (root/'watch-job.json').exists():
        command=['sbatch','--parsable','--account=miso','--partition='+QUEUES+',jag-standard',
            '--nodes=1','--ntasks=1','--cpus-per-task=1','--mem=4G','--time='+plan.get('watch_time','2-00:00:00'),
            '--requeue','--open-mode=append','--job-name='+plan.get('watch_name','mopd-v2-watch'),
            '--output='+str(root/'watch-%j.log'),str(root/'watch.sbatch')]
        job=subprocess.check_output(command,text=True).strip().split(';')[0]
        atomic_json(root/'watch-job.json',{'job_id':job,'command':command})
    plan['status']='submitted';atomic_json(root/'students/launch-plan.json',plan)
    write_run_description(root,Path(__file__).resolve().parents[2] if not (root/'controller-source').exists()
                          else Path('/juice5b/u/dhei/tatsu/rlve'),plan,read(root/'teacher-plan.json'))


def events(path):
    with Path(path).open() as stream:return [json.loads(line) for line in stream if line.strip()]


def scores(rows):
    grouped=defaultdict(list)
    for row in rows:
        meta=json.loads(row['metadata_json']);grouped[meta['task'],meta['key']].append(bool(row['correct']))
    if not grouped or any(len(v)!=8 for v in grouped.values()):raise ValueError('Incomplete eight-sample validation panel')
    return {'pass@1':sum(map(sum,grouped.values()))/len(rows),'pass@8':sum(any(v) for v in grouped.values())/len(grouped)}


def panels(policy):
    values=[];paths=sorted((policy/'rollouts/eval').glob('*.jsonl'))
    for path in paths:
        try:
            rows=events(path);by_task={t:[v for v in rows if v['source_environment']==t] for t in TASKS}
            if any(len(v)!=512 for v in by_task.values()):raise ValueError('Incomplete four-task evaluation')
            values.append({'rollout_id':int(path.stem),'aggregate':scores(rows),
                           'tasks':{t:scores(v) for t,v in by_task.items()}})
        except (json.JSONDecodeError,ValueError):
            if path!=paths[-1] or (policy/'completed.json').exists():raise
    return values


def audit_student(root,row):
    configure(root)
    policy=Path(row['output'])/'policy';done=read(policy/'completed.json')
    if not (done['completed'] and done['updates']==STEPS and done['train_rollouts']==STEPS*8):
        raise ValueError('Incorrect MOPD step/response budget')
    state=read(policy/'ckpt/rlve_checkpoint_state.json')
    if not state['completed'] or state['rollout_id']!=STEPS-1:raise ValueError('Incorrect final checkpoint')
    iteration=policy/f'ckpt/iter_{STEPS-1:07d}'
    if not (iteration/'common.pt').exists() or not (iteration/'.metadata').exists():
        raise ValueError('Missing committed model/optimizer checkpoint')
    checkpoint_bytes=sum(p.stat().st_size for p in iteration.rglob('*') if p.is_file())
    if checkpoint_bytes<7_000_000_000:raise ValueError('Expected the full resumable optimizer checkpoint')
    train=sorted((policy/'rollouts/train').glob('*.jsonl'))
    if [int(p.stem) for p in train]!=list(range(STEPS)):raise ValueError('Missing training batches')
    for path in train:
        rows=events(path)
        if Counter(v['source_environment'] for v in rows)!=Counter(dict.fromkeys(TASKS,2)):
            raise ValueError('Unbalanced student batch')
        for v in rows:
            prefix=json.loads(v['metadata_json']).get('_modular_v2_forcing',{}).get('prefix_tokens',0)
            if not 0<v['response_tokens']-prefix<=2048:raise ValueError('Generation cap exceeded')
    routes=events(policy/'teacher-scores.jsonl')
    if [v['rollout_id'] for v in routes]!=list(range(STEPS)):raise ValueError('Missing teacher routing records')
    for v in routes:
        if len(v['samples'])!=8:raise ValueError('Missing teacher samples')
        for s in v['samples']:
            if s['teacher']!=row['teachers'][s['task']] or not math.isfinite(s['mean_teacher_minus_student_logp']):
                raise ValueError('Wrong teacher route or nonfinite log probabilities')
    curve=panels(policy)
    if [v['rollout_id'] for v in curve]!=sorted({0,STEPS-1,*range(INTERVAL-1,STEPS,INTERVAL)}):
        raise ValueError('Missing validation panel')
    for path in (policy/'rollouts/eval').glob('*.jsonl'):
        rows=events(path)
        for task in TASKS:
            expected={v['metadata']['key'] for v in events(root/'students/selection'/task/'val.jsonl')}
            actual={json.loads(v['metadata_json'])['key'] for v in rows if v['source_environment']==task}
            if actual!=expected:raise ValueError('Changed validation question set')
    probes=events(policy/'update-probes.jsonl')
    if [v['rollout_id'] for v in probes]!=list(range(0,STEPS,10)):raise ValueError('Missing objective probes')
    return {'passed':True,'name':row['name'],'completed':done,'curve':curve,
            'teacher_scored_responses':sum(len(v['samples']) for v in routes),'checkpoint_bytes':checkpoint_bytes}


def finalize(row,audit):
    policy=Path(row['output'])/'policy';marker=policy/'wandb-final-summary.json'
    if marker.exists():return True
    import wandb
    run_id=(policy/'wandb_run_id.txt').read_text().strip()
    api=wandb.Api(timeout=60,overrides={'base_url':'https://api.forge.coreweave.com'})
    remote=api.run('david-heineman/'+PROJECT+'/'+run_id)
    if remote.state!='finished':return False
    last=audit['curve'][-1]
    values={'david/step':STEPS-1,'david/rollouts':STEPS*8}
    values.update({'david/eval/val_envs/'+k:v for k,v in last['aggregate'].items()})
    for task,metrics in last['tasks'].items():
        values.update({f'david/eval/val_envs/{task}/{k}':v for k,v in metrics.items()})
    remote.summary.update(values);remote.load(force=True)
    if any(remote.summary.get(k)!=v for k,v in values.items()):raise ValueError('W&B summary readback failed')
    atomic_json(marker,{'run_id':run_id,'summary':values,'verified':True})
    return True


def plot(root):
    plan=configure(root)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter
    jobs=plan.get('comparison_jobs',[])+read(root/'students/jobs.json')
    fig,axes=plt.subplots(2,2,figsize=(8.8,5.6))
    names={'base':'Base RL teachers','baseline-sft':'Baseline SFT RL teachers','task-sft':'Task SFT RL teachers'}
    colors={'base':'tab:blue','baseline-sft':'tab:orange','task-sft':'tab:green'}
    for row in jobs:
        curve=panels(Path(row['output'])/'policy')
        for task,ax in zip(TASKS,axes.flat):
            if curve:ax.plot([0 if v['rollout_id']==0 else (v['rollout_id']+1)*8 for v in curve],
                [v['tasks'][task]['pass@1'] for v in curve],label=names[row['bank']],color=colors[row['bank']])
    for task,ax in zip(TASKS,axes.flat):
        ax.set(title={'math':'Math','code':'Code','if':'IF','chat':'Chat'}[task],
               xlabel='train rollouts',ylabel='pass@1',xlim=(0,STEPS*8))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x,_:'0' if x==0 else f'{x/1000:g}K'))
        ax.grid(alpha=.2)
        if ax.lines:ax.legend(fontsize=8)
    fig.tight_layout()
    output=Path('/juice5b/u/dhei/tatsu/rlve/plot/mopd-modular-v2')/plan.get('plot_filename','mopd_training_curves.png')
    fig.savefig(output,dpi=180);fig.savefig(output.with_suffix('.pdf'));plt.close(fig)


def watch(root):
    configure(root)
    with (root/'watch.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);errors=0
        while not (root/'HALTED_BY_USER.json').exists():
            try:
                jobs=read(root/'students/jobs.json');slurm=states(jobs)
                status={'utc':datetime.now(timezone.utc).isoformat(),'runs':[],'complete':bool(jobs)}
                for row in jobs:
                    state=slurm.get(row['job_id'],{'state':'UNKNOWN'})
                    status['runs'].append({'name':row['name'],'job_id':row['job_id'],**state})
                    if state['state'] in TERMINAL_FAILURES:retry(root,root/'students/jobs.json',row,state)
                    if state['state']=='COMPLETED':
                        destination=root/(row['bank']+'-completion-audit.json')
                        if not destination.exists():atomic_json(destination,audit_student(root,row))
                        status['complete'] &= bool(finalize(row,read(destination)))
                    else:status['complete']=False
                plot(root);atomic_json(root/'monitor.json',status);print(json.dumps(status),flush=True)
                errors=0
                if status['complete']:return
            except Exception as error:
                errors+=1
                atomic_json(root/'watch-error.json',{'consecutive_errors':errors,'error':repr(error)})
                if errors>=8:raise
            time.sleep(45)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('prepare','resize','submit','watch'))
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--snapshot',type=Path)
    parser.add_argument('--previous',type=Path)
    parser.add_argument('--project',default='modular-mopd-r1-v1')
    args=parser.parse_args();repo=Path(__file__).resolve().parents[2]
    if args.action=='prepare':prepare(args.root.resolve(),args.snapshot.resolve(),repo)
    elif args.action=='resize':resize_prepared(args.previous.resolve(),args.root.resolve(),repo,args.project)
    elif args.action=='submit':submit(args.root.resolve())
    else:watch(args.root.resolve())
