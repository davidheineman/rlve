"""Prepare the three 4,096-step MOPD comparisons from one baseline-SFT student."""

import argparse
import json
import os
import shlex
import shutil
import subprocess
from pathlib import Path

import yaml

from modular_rl.initializers import verify
from modular_rl.mopd_export import digest
from modular_rl.mopd_suite import mixed_data
from modular_rl.suite import shell_exports
from .config import atomic_json
from .mopd_from_curves import preflight, read, replace_arg, write_launch_files


STEPS=4096
INTERVAL=256
BANKS=('base','baseline-sft','task-sft')


def eos_pair(bank):
    # The common SFT student emits im_end. Raw-Base RL teachers emit endoftext;
    # SFT RL teachers already share the student's termination convention.
    if bank not in BANKS:raise ValueError('Unknown teacher bank')
    return 151645,151643 if bank=='base' else 151645


def allocation_for_initializer(text):
    anchor='"$MODULAR_RL_INITIALIZERS" base'
    if text.count(anchor)!=2:raise ValueError('Inspect the proven allocation initializer calls')
    return text.replace(anchor,'"$MODULAR_RL_INITIALIZERS" "${MODULAR_RL_INITIALIZER:-base}"')


def prepare(root,previous,teacher_root,repo):
    if root.exists():raise ValueError('Use a new experiment directory')
    old=read(previous/'students/launch-plan.json')
    for relative,sha in old['source_sha256'].items():
        if digest(Path(old['source'])/relative)!=sha:raise ValueError('Proven v2 source changed: '+relative)
    teachers=read(teacher_root/'teacher-plan.json')
    baseline=teachers['teachers'][teachers['banks']['baseline-sft']['math']]['initializer_identity']
    if baseline['id']!='baseline' or baseline['kind']!='dolci-sft-base':
        raise ValueError('Require the original shared baseline-SFT initializer')
    verify(baseline)
    root.mkdir(parents=True)
    teachers.update(student_steps=STEPS,student_rollouts=STEPS*8,student_initializer='baseline',
                    reused_verified_exports=str(teacher_root),rerun_revision=3)
    atomic_json(root/'teacher-plan.json',teachers)
    for name in ('plot-snapshot.json','teacher-model-contracts.json','cpu-model-load-check.json'):
        shutil.copy2(teacher_root/name,root/name)
    shutil.copytree(teacher_root/'teachers/exports',root/'teachers/exports',copy_function=os.link)
    for bank,mapping in teachers['banks'].items():
        directory=root/'teachers/banks'/bank;directory.mkdir(parents=True)
        for index,(task,name) in enumerate(mapping.items()):
            slot=directory/f'{index:02d}-{task}';slot.mkdir()
            for asset in ('hf-final','weights_only_complete.json'):
                (slot/asset).symlink_to(root/'teachers/exports'/name/asset,target_is_directory=asset=='hf-final')
        atomic_json(directory/'bank.json',mapping)
    experiment=root/'students';source=experiment/'source'
    for name in ('source','runtime','selection'):
        shutil.copytree(previous/'students'/name,experiment/name,
                        ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    initializers=read(previous/'students/initializers.json');initializers['baseline']=baseline
    registry=experiment/'initializers.json';atomic_json(registry,initializers)
    # Keep the completed v2 trainer; change only student selection and protocol logging.
    shutil.copy2(repo/'modular_rl/mopd_runtime.py',source/'modular_rl/mopd_runtime.py')
    allocation=source/'modular_rl/mopd_allocation.sh'
    allocation.write_text(allocation_for_initializer(allocation.read_text()))
    config_path=source/'modular_rl/mopd-config.yaml';config=yaml.safe_load(config_path.read_text())
    config.update({'num-rollout':STEPS,'eval-interval':INTERVAL})
    config_path.write_text(yaml.safe_dump(config,sort_keys=False))
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(baseline['hf'],local_files_only=True)
    mixed_data(experiment/'selection',STEPS,tokenizer,enable_thinking=False)
    # Preserve exactly the previous 8K prompt prefix, while extending the budget.
    with (previous/'students/selection/mixed-train.jsonl').open() as old_data, \
         (experiment/'selection/mixed-train.jsonl').open() as new_data:
        for old_line in old_data:
            if next(new_data,None)!=old_line:raise ValueError('The existing student curriculum changed')
    if digest(experiment/'selection/mixed-val.jsonl')!=old['selection_sha256']['mixed-val.jsonl']:
        raise ValueError('Validation questions changed')
    jobs=[]
    short={'base':'base','baseline-sft':'baseline','task-sft':'task'}
    template=Path(old['runs'][0]['command'][-1]).read_text()
    for bank in BANKS:
        student_eos,teacher_eos=eos_pair(bank)
        name='mopd-sftinit-'+short[bank]+'-v3';output=experiment/'runs'/name;output.mkdir(parents=True)
        env={}
        for line in template.splitlines():
            if line.startswith('export ') and not line.startswith('export APPTAINERENV_'):
                key,value=line[7:].split('=',1);env[key]=shlex.split(value)[0].replace(str(previous),str(root))
        args=shlex.split(env['MODULAR_RL_EXTRA_ARGS'])
        for flag,value in [('--num-rollout',STEPS),('--eval-interval',INTERVAL),
                           ('--opd-teacher-dir',root/'teachers/banks'/bank),
                           ('--opd-student-eos-token-id',student_eos),('--opd-teacher-eos-token-id',teacher_eos)]:
            replace_arg(args,flag,value)
        env.update(MODULAR_RL_RUN_NAME=name,MODULAR_RL_OUTPUT=str(output),MODULAR_RL_INITIALIZER='baseline',
            MODULAR_RL_INITIALIZERS_SHA256=digest(registry),MODULAR_RL_GROUP='mopd-sftinit-4096-v3',
            MODULAR_RL_TEACHER_BANK=str(root/'teachers/banks'/bank),MODULAR_RL_EXTRA_ARGS=shlex.join(args),
            MODULAR_RL_STUDENT_EOS_TOKEN_ID=str(student_eos),MODULAR_RL_TEACHER_EOS_TOKEN_ID=str(teacher_eos))
        script=output/'run.sbatch'
        script.write_text('#!/usr/bin/env bash\n#SBATCH --account=miso\n#SBATCH --partition='+
            'miso-lo,sphinx-lo,jag-lo,sc-loprio,miso,sphinx\n#SBATCH --nodes=1\n#SBATCH --ntasks=1\n'+
            '#SBATCH --requeue\n#SBATCH --open-mode=append\nset -euo pipefail\n'+shell_exports(env)+
            shell_exports({'APPTAINERENV_'+k:v for k,v in env.items()})+
            'export APPTAINERENV_WANDB_API_KEY="${WANDB_API_KEY:?}"\nsource "$RLVE_REPO/scripts/slurm/common.sh"\n'+
            'rlve_apptainer bash modular_rl/mopd_allocation.sh "$@"\n')
        subprocess.run(['bash','-n',str(script)],check=True)
        command=['sbatch','--parsable','--job-name='+name,'--gres=gpu:4','--constraint=ampere|ada|hopper',
            '--cpus-per-task=28','--mem=128G','--time=23:50:00','--exclude=viscam10,jagupard33,jagupard34,pasteur6',
            '--chdir='+str(source),'--output='+str(output/'slurm-%j.out'),
            '--export=ALL,WANDB_SERVICE=,NCCL_P2P_DISABLE=1,NCCL_CUMEM_HOST_ENABLE=0',str(script)]
        jobs.append({'name':name,'bank':bank,'teachers':teachers['banks'][bank],
            'teacher_bank':str(root/'teachers/banks'/bank),'gpus':4,'student_gpus':2,'teacher_gpus':1,'reward_gpus':1,
            'student_eos_token_id':student_eos,'teacher_eos_token_id':teacher_eos,
            'output':str(output),'command':command,'script_sha256':digest(script)})
    plan={**old,'status':'prepared-not-submitted','runs':jobs,'source':str(source),'initializers':str(registry),
        'initializers_sha256':digest(registry),'student_initializer':'baseline SFT (Dolci 1B, Qwen3-0.6B-Base ancestry)',
        'student_initializer_alias':'baseline','student_initializer_hf':baseline['hf'],
        'train_steps':STEPS,'train_rollouts':STEPS*8,'evaluation_interval':INTERVAL,'simultaneous_gpus':12,
        'estimated_wall_hours':[6,14],'rerun_revision':3,'rerun_of':str(previous),'comparison_jobs':[],
        'plot_filename':'mopd_training_curves_sftinit.png','description_filename':'mopd_experiment_sftinit.md',
        'student_eos_token_id':151645,'teacher_eos_token_id':None,
        'selection_sha256':{name:digest(experiment/'selection'/name)
            for name in ('mixed-train.jsonl','mixed-val.jsonl','mixed-selection.json')},
        'source_sha256':{str(p.relative_to(source)):digest(p) for p in sorted(source.rglob('*'))
            if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc'}}
    atomic_json(experiment/'launch-plan.json',plan)
    atomic_json(root/'student-initializer.json',baseline)
    atomic_json(root/'rerun-provenance.json',{'trainer_parent':str(previous),'teacher_parent':str(teacher_root),
        'fresh_baseline_sft_students':True,'same_first_8000_prompts':True,'validation_unchanged':True,
        'teacher_weights_unchanged':True,'steps':STEPS,'student_rollouts':STEPS*8,'eval_interval':INTERVAL})
    preflight(root);write_launch_files(root,repo)
    print('Prepared three baseline-SFT students:',json.dumps([job['name'] for job in jobs]),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--previous',type=Path,required=True)
    parser.add_argument('--teachers',type=Path,required=True)
    args=parser.parse_args()
    prepare(args.root.resolve(),args.previous.resolve(),args.teachers.resolve(),Path(__file__).resolve().parents[2])
