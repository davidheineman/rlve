"""Restart the baseline-SFT MOPD comparisons with evaluation every 32 updates."""

import argparse
import json
import os
import shlex
import shutil
import subprocess
from pathlib import Path

import yaml

from modular_rl.mopd_export import digest
from modular_rl.suite import shell_exports
from .config import atomic_json
from .mopd_from_curves import preflight, read, replace_arg, write_launch_files


def prepare(root,previous,repo):
    if root.exists():raise ValueError('Use a fresh experiment directory')
    if not (previous/'HALTED_BY_USER.json').exists():raise ValueError('Stop the superseded runs first')
    old=read(previous/'students/launch-plan.json')
    if old['train_steps']!=4096 or old['student_initializer_alias']!='baseline':
        raise ValueError('Require the baseline-SFT 4096-step parent')
    for relative,sha in old['source_sha256'].items():
        if digest(Path(old['source'])/relative)!=sha:raise ValueError('Parent source changed: '+relative)
    root.mkdir(parents=True)
    for name in ('teacher-plan.json','plot-snapshot.json','teacher-model-contracts.json',
                 'cpu-model-load-check.json','student-initializer.json','test-results.json'):
        shutil.copy2(previous/name,root/name)
    teachers=read(root/'teacher-plan.json')
    shutil.copytree(previous/'teachers/exports',root/'teachers/exports',copy_function=os.link)
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
    registry=experiment/'initializers.json';shutil.copy2(previous/'students/initializers.json',registry)
    path=source/'modular_rl/mopd-config.yaml';config=yaml.safe_load(path.read_text())
    config['eval-interval']=32;path.write_text(yaml.safe_dump(config,sort_keys=False))
    path=source/'modular_rl/mopd_allocation.sh';text=path.read_text()
    if text.count('RUN_TIMEOUT=84600')!=1:raise ValueError('Inspect native timeout setting')
    path.write_text(text.replace('RUN_TIMEOUT=84600','RUN_TIMEOUT=169200'))
    path=source/'modular_rl/mopd_runtime.py';text=path.read_text()
    anchor="'train_steps': args.num_rollout,"
    if text.count(anchor)!=1:raise ValueError('Inspect protocol logging')
    path.write_text(text.replace(anchor,anchor+" 'eval_interval': args.eval_interval,"))
    jobs=[]
    for row in old['runs']:
        name=row['name']+'-eval32';output=experiment/'runs'/name;output.mkdir(parents=True)
        env={}
        for line in Path(row['command'][-1]).read_text().splitlines():
            if line.startswith('export ') and not line.startswith('export APPTAINERENV_'):
                key,value=line[7:].split('=',1);env[key]=shlex.split(value)[0].replace(str(previous),str(root))
        args=shlex.split(env['MODULAR_RL_EXTRA_ARGS']);replace_arg(args,'--eval-interval',32)
        env.update(MODULAR_RL_RUN_NAME=name,MODULAR_RL_OUTPUT=str(output),
                   MODULAR_RL_GROUP='mopd-sftinit-4096-v3-eval32',MODULAR_RL_EXTRA_ARGS=shlex.join(args))
        script=output/'run.sbatch'
        script.write_text('#!/usr/bin/env bash\n#SBATCH --account=miso\n#SBATCH --partition='+
            'miso-lo,sphinx-lo,jag-lo,sc-loprio,miso,sphinx\n#SBATCH --nodes=1\n#SBATCH --ntasks=1\n'+
            '#SBATCH --requeue\n#SBATCH --open-mode=append\nset -euo pipefail\n'+shell_exports(env)+
            shell_exports({'APPTAINERENV_'+k:v for k,v in env.items()})+
            'export APPTAINERENV_WANDB_API_KEY="${WANDB_API_KEY:?}"\nsource "$RLVE_REPO/scripts/slurm/common.sh"\n'+
            'rlve_apptainer bash modular_rl/mopd_allocation.sh "$@"\n')
        subprocess.run(['bash','-n',str(script)],check=True)
        command=[v.replace(str(previous),str(root)).replace(row['name'],name) for v in row['command']]
        command=['--time=1-23:50:00' if v.startswith('--time=') else v for v in command]
        jobs.append({**row,'name':name,'output':str(output),'teacher_bank':str(root/'teachers/banks'/row['bank']),
                     'command':command,'script_sha256':digest(script)})
    plan={**old,'status':'prepared-not-submitted','runs':jobs,'source':str(source),'initializers':str(registry),
        'evaluation_interval':32,'estimated_wall_hours':[16,24],
        'timing_basis':'4096 training updates plus 129 full validation panels; actual generation lengths/hardware affect time.',
        'watch_time':'3-00:00:00','watch_name':'mopd-v3-eval32-watch','rerun_of':str(previous),
        'plot_filename':'mopd_training_curves_sftinit_eval32.png',
        'description_filename':'mopd_experiment_sftinit_eval32.md',
        'source_sha256':{str(p.relative_to(source)):digest(p) for p in sorted(source.rglob('*'))
            if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc'}}
    for relative,sha in old['selection_sha256'].items():
        if digest(experiment/'selection'/relative)!=sha:raise ValueError('Student data changed')
    atomic_json(experiment/'launch-plan.json',plan)
    atomic_json(root/'rerun-provenance.json',{'parent':str(previous),'fresh_baseline_sft_students':True,
        'train_data_unchanged':True,'validation_questions_unchanged':True,'teacher_weights_unchanged':True,
        'train_steps':4096,'train_rollouts':32768,'eval_interval_optimizer_updates':32,
        'eval_interval_train_rollouts':256,'validation_panels':129,'validation_responses':129*2048})
    preflight(root);write_launch_files(root,repo)
    atomic_json(previous/'superseded.json',{'replacement':str(root),'reason':'User requested evaluation every 32 optimizer updates'})
    print('Prepared three fresh eval32 replacements:',json.dumps([j['name'] for j in jobs]),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--previous',type=Path,required=True)
    args=parser.parse_args()
    prepare(args.root.resolve(),args.previous.resolve(),Path(__file__).resolve().parents[2])
