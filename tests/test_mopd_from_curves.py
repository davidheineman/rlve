import json
from types import SimpleNamespace

import pytest

from modular_rl.dolci.mopd_from_curves import ALIASES, TASKS, scores, select_teachers, validate_tensor_shapes, validate_gpu_roles
from modular_rl.mopd_suite import mixed_data


def snapshot():
    keys={(alias,task) for mapping in ALIASES.values() for task,alias in mapping.items()}
    keys.add(('chat','if'))
    return {'runs':[{'initializer':alias,'task':task,'name':alias+'-'+task,
                     'method':'rl','status':'completed','algorithm':'grpo'} for alias,task in keys]}


def test_plot_teacher_banks_use_if20_and_keep_three_comparisons_separate():
    banks=select_teachers(snapshot())
    assert set(banks)=={'base','baseline-sft','task-sft'}
    assert banks['task-sft']['if']['name']=='if20-if'
    assert banks['task-sft']['chat']['name']=='chat-chat'
    assert all(banks['base'][task]['initializer']=='base' for task in TASKS)
    assert all(banks['baseline-sft'][task]['initializer']=='baseline' for task in TASKS)
    assert len({row['name'] for bank in banks.values() for row in bank.values()})==12


@pytest.mark.parametrize('change',[{'status':'in_progress'},{'algorithm':'maxrl'}])
def test_teacher_selection_rejects_incomplete_or_wrong_objective(change):
    value=snapshot()
    next(r for r in value['runs'] if r['initializer']=='code' and r['task']=='code').update(change)
    with pytest.raises(ValueError,match='completed plotted GRPO'):select_teachers(value)


def test_generation_template_cap_is_filtered_before_balanced_interleaving(tmp_path):
    class Tokenizer:
        def apply_chat_template(self,messages,*args,**kwargs):
            return messages[0]['content']+('x'*10 if kwargs.get('enable_thinking') is False else '')
        def __call__(self,prompt):return {'input_ids':list(range(len(prompt)))}
    for task in TASKS:
        directory=tmp_path/task;directory.mkdir()
        rows=[{'user_prompt':text,'metadata':{'task':task,'key':key}}
              for text,key in [('short','keep'),('x'*2040,'generation-too-long')]]
        (directory/'train.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
        (directory/'val.jsonl').write_text(json.dumps(rows[0])+'\n')
    mixed_data(tmp_path,3,Tokenizer(),enable_thinking=False)
    result=json.loads((tmp_path/'mixed-selection.json').read_text())
    assert result['eligible_pool_counts']==dict.fromkeys(TASKS,1)
    rows=[json.loads(line) for line in (tmp_path/'mixed-train.jsonl').open()]
    assert len(rows)==24 and all(r['metadata']['key']=='keep' for r in rows)


def test_evaluation_grouping_keeps_same_key_in_different_tasks_separate():
    rows=[]
    for task,correct in zip(TASKS,[0,4,8,2]):
        rows.extend({'metadata_json':json.dumps({'task':task,'key':'shared'}),'correct':i<correct}
                    for i in range(8))
    assert scores(rows)=={'pass@1':14/32,'pass@8':3/4}
    with pytest.raises(ValueError,match='Incomplete'):scores(rows[:-1])


def test_four_gpu_launch_requires_matching_roles_native_topology_and_exact_project(tmp_path):
    script=tmp_path/'run.sbatch'
    script.write_text('export MODULAR_RL_WANDB_PROJECT=modular-mopd-r1-v1\n'
        'export MODULAR_RL_POLICY_GPUS=2\nexport MODULAR_RL_TEACHER_GPUS=1\nexport MODULAR_RL_REWARD_GPUS=1\n')
    row={'gpus':4,'student_gpus':2,'teacher_gpus':1,'reward_gpus':1,
         'command':['sbatch','--gres=gpu:4',str(script)]}
    config={'actor-num-gpus-per-node':2,'num-gpus-per-node':2}
    validate_gpu_roles(row,config,'modular-mopd-r1-v1')
    with pytest.raises(ValueError,match='project'):validate_gpu_roles(row,config,'modular-mopd-rl-v1')
    with pytest.raises(ValueError,match='topology'):validate_gpu_roles(row,{**config,'actor-num-gpus-per-node':4},'modular-mopd-r1-v1')
    with pytest.raises(ValueError,match='role counts'):validate_gpu_roles({**row,'teacher_gpus':2},config,'modular-mopd-r1-v1')


def test_model_contract_allows_omitted_tied_weights_but_requires_other_parameters():
    import torch
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding=torch.nn.Embedding(5,2)
            self.output=torch.nn.Linear(2,5,bias=False)
            self.output.weight=self.embedding.weight
            self.other=torch.nn.Linear(2,2,bias=False)
    model=Model()
    actual={'embedding.weight':[5,2],'other.weight':[2,2]}
    assert validate_tensor_shapes(model,actual)==['output.weight']
    with pytest.raises(ValueError,match='key/shape'):validate_tensor_shapes(model,{'embedding.weight':[5,2]})
