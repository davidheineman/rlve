from pathlib import Path
from types import SimpleNamespace

import pytest

from modular_rl.dolci.mopd_eos_rerun import copy_reward_adapter
from modular_rl.dolci.mopd_from_curves import validate_eos_launch
from modular_rl.mopd_runtime import generation_health


ARGS=['--opd-student-eos-token-id','151643','--opd-teacher-eos-token-id','151645',
      '--rollout-stop-token-ids','151643','151645']
ENV={'MODULAR_RL_CODE_FORCING_FALLBACK':'1','MODULAR_RL_EOS_ALIGNMENT_REQUIRED':'1'}


def test_rerun_requires_both_sampler_stops_and_teacher_alignment():
    validate_eos_launch(ARGS,ENV)
    with pytest.raises(ValueError,match='Both EOS'):validate_eos_launch(ARGS[:-1],ENV)
    with pytest.raises(ValueError,match='fallback'):
        validate_eos_launch(ARGS,{**ENV,'MODULAR_RL_CODE_FORCING_FALLBACK':'0'})
    with pytest.raises(ValueError,match='alignment'):
        validate_eos_launch(['wrong' if v=='151645' else v for v in ARGS],ENV)


def test_overlay_does_not_revert_the_frozen_generation_prompt_mode(tmp_path):
    source=tmp_path/'source';path=source/'modular_rl/v2/rollout.py';path.parent.mkdir(parents=True)
    path.write_text('async def generate():\n    return dict(enable_thinking=False)\n\n'
                    'async def reward(args,sample,**kwargs):\n    return 0\n\n'
                    'def validation_prompt_count():\n    return 64\n')
    copy_reward_adapter(source,Path(__file__).resolve().parents[1])
    result=path.read_text()
    assert 'return dict(enable_thinking=False)' in result
    assert 'MODULAR_RL_CODE_FORCING_FALLBACK' in result and 'def code_submission(' in result


def test_generation_health_tracks_truncation_and_each_stop_alias_by_task():
    samples=[SimpleNamespace(metadata={'task':task},tokens=[1,last])
             for task,last in [('code',151643),('if',50),('code',151645)]]
    counts=[{'sampled_tokens':n,'truncated':truncated,'forced':forced}
            for n,truncated,forced in [(10,0,0),(2048,1,0),(30,0,1)]]
    result=generation_health(samples,counts)
    assert result['code']=={'responses':2,'mean_tokens':20,'truncated_fraction':0,'forced_fraction':.5,
                            'endoftext_fraction':.5,'im_end_fraction':.5}
    assert result['if']['truncated_fraction']==1 and result['if']['mean_tokens']==2048
    with pytest.raises(ValueError):generation_health(samples,counts[:-1])
