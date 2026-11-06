"""Prepare fresh 8K MOPD v2 students using the verified SFT teacher banks."""

import argparse
import ast
import json
import os
import shlex
import shutil
import subprocess
from pathlib import Path

from modular_rl.mopd_export import digest
from modular_rl.suite import shell_exports
from .config import atomic_json
from .mopd_from_curves import preflight, read, write_launch_files


def copy_reward_adapter(source,repo):
    """Keep the proven prompt/generation code; overlay only the opt-in fallback."""
    original=source/'modular_rl/v2/rollout.py'
    text=original.read_text();fresh=(repo/'modular_rl/v2/rollout.py').read_text()
    tree=ast.parse(fresh)
    definitions={node.name:ast.get_source_segment(fresh,node) for node in tree.body
                 if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef))}
    start=text.index('async def reward(');end=text.index('def validation_prompt_count(',start)
    original.write_text(text[:start]+definitions['reward']+'\n\n\n'+definitions['code_submission']+'\n\n\n'+text[end:])


def prepare(root,previous,repo):
    if root.exists():raise ValueError('Use a new experiment directory; never rewrite launched sources')
    old=read(previous/'students/launch-plan.json')
    for relative,sha in old['source_sha256'].items():
        if digest(Path(old['source'])/relative)!=sha:raise ValueError('Original source changed: '+relative)
    root.mkdir(parents=True)
    teachers=read(previous/'teacher-plan.json');banks=('baseline-sft','task-sft')
    teachers['banks']={b:teachers['banks'][b] for b in banks}
    selected={name for mapping in teachers['banks'].values() for name in mapping.values()}
    teachers['teachers']={name:value for name,value in teachers['teachers'].items() if name in selected}
    teachers.update(reused_verified_exports=str(previous),rerun_revision=2)
    atomic_json(root/'teacher-plan.json',teachers)
    for name in ('plot-snapshot.json','cpu-model-load-check.json'):
        shutil.copy2(previous/name,root/name)
    contracts=read(previous/'teacher-model-contracts.json')
    atomic_json(root/'teacher-model-contracts.json',{name:contracts[name] for name in selected})
    for name in selected:
        shutil.copytree(previous/'teachers/exports'/name,root/'teachers/exports'/name,copy_function=os.link)
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
    # These three library files differ from v1 only by the tested EOS alignment.
    overlays=('slime/utils/opd.py','slime/ray/opd_teachers.py','slime/utils/arguments.py','modular_rl/mopd_runtime.py')
    for relative in overlays:shutil.copy2(repo/relative,source/relative)
    copy_reward_adapter(source,repo)
    changed={relative:{'before':old['source_sha256'][relative],'after':digest(source/relative)}
             for relative in (*overlays,'modular_rl/v2/rollout.py')}
    jobs=[]
    for row in old['runs']:
        if row['bank'] not in banks:continue
        name=row['name']+'-v2';output=experiment/'runs'/name;output.mkdir(parents=True)
        env={}
        for line in Path(row['command'][-1]).read_text().splitlines():
            if line.startswith('export ') and not line.startswith('export APPTAINERENV_'):
                key,value=line[7:].split('=',1);env[key]=shlex.split(value)[0].replace(str(previous),str(root))
        args=shlex.split(env['MODULAR_RL_EXTRA_ARGS'])
        args+=['--opd-student-eos-token-id','151643','--opd-teacher-eos-token-id','151645',
               '--rollout-stop-token-ids','151643','151645']
        env.update(MODULAR_RL_RUN_NAME=name,MODULAR_RL_OUTPUT=str(output),
                   MODULAR_RL_GROUP='mopd-modular-v2-eos',MODULAR_RL_EXTRA_ARGS=shlex.join(args),
                   MODULAR_RL_CODE_FORCING_FALLBACK='1',MODULAR_RL_EOS_ALIGNMENT_REQUIRED='1')
        script=output/'run.sbatch'
        script.write_text('#!/usr/bin/env bash\n#SBATCH --account=miso\n#SBATCH --partition='+
            'miso-lo,sphinx-lo,jag-lo,sc-loprio,miso,sphinx\n#SBATCH --nodes=1\n#SBATCH --ntasks=1\n'+
            '#SBATCH --requeue\n#SBATCH --open-mode=append\nset -euo pipefail\n'+shell_exports(env)+
            shell_exports({'APPTAINERENV_'+k:v for k,v in env.items()})+
            'export APPTAINERENV_WANDB_API_KEY="${WANDB_API_KEY:?}"\nsource "$RLVE_REPO/scripts/slurm/common.sh"\n'+
            'rlve_apptainer bash modular_rl/mopd_allocation.sh "$@"\n')
        subprocess.run(['bash','-n',str(script)],check=True)
        command=[value.replace(str(previous),str(root)).replace(row['name'],name) for value in row['command']]
        jobs.append({**row,'name':name,'output':str(output),'teacher_bank':str(root/'teachers/banks'/row['bank']),
                     'command':command,'script_sha256':digest(script)})
    plan={**old,'status':'prepared-not-submitted','runs':jobs,'source':str(source),'initializers':str(registry),
          'simultaneous_gpus':8,'rerun_revision':2,'rerun_of':str(previous),'eos_alignment':True,
          'student_eos_token_id':151643,'teacher_eos_token_id':151645,'rollout_stop_token_ids':[151643,151645],
          'code_forcing_fallback':True,'source_changes':changed,
          'comparison_jobs':[j for j in read(previous/'students/jobs.json') if j['bank']=='base'],
          'plot_filename':'mopd_training_curves_v2.png','description_filename':'mopd_experiment_v2.md',
          'source_sha256':{str(p.relative_to(source)):digest(p) for p in sorted(source.rglob('*'))
                          if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc'}}
    for relative,expected in old['selection_sha256'].items():
        if digest(experiment/'selection'/relative)!=expected:raise ValueError('Comparison data changed: '+relative)
    atomic_json(experiment/'launch-plan.json',plan)
    atomic_json(root/'rerun-provenance.json',{'parent':str(previous),'parent_run_names':[r['name'] for r in old['runs']],
                'fresh_base_students':True,'training_data_unchanged':True,'validation_data_unchanged':True,
                'teacher_weights_reused':True,'source_changes':changed})
    preflight(root);write_launch_files(root,repo)
    print('Prepared two independent v2 students:',json.dumps([j['name'] for j in jobs]),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--previous',type=Path,required=True)
    args=parser.parse_args()
    prepare(args.root.resolve(),args.previous.resolve(),Path(__file__).resolve().parents[2])
