import pytest

from modular_rl.dolci.mopd_from_curves import validate_eos_launch
from modular_rl.dolci.mopd_sftinit import allocation_for_initializer, eos_pair


@pytest.mark.parametrize('bank,expected',[('base',(151645,151643)),
    ('baseline-sft',(151645,151645)),('task-sft',(151645,151645))])
def test_sft_student_uses_its_own_stop_convention_for_each_teacher_bank(bank,expected):
    assert eos_pair(bank)==expected
    student,teacher=expected
    args=['--opd-student-eos-token-id',str(student),'--opd-teacher-eos-token-id',str(teacher),
          '--rollout-stop-token-ids','151643','151645']
    env={'MODULAR_RL_CODE_FORCING_FALLBACK':'1','MODULAR_RL_EOS_ALIGNMENT_REQUIRED':'1',
         'MODULAR_RL_STUDENT_EOS_TOKEN_ID':str(student),'MODULAR_RL_TEACHER_EOS_TOKEN_ID':str(teacher)}
    validate_eos_launch(args,env,student_eos=student,teacher_eos=teacher)
    with pytest.raises(ValueError,match='alignment'):
        validate_eos_launch(args,{**env,'MODULAR_RL_STUDENT_EOS_TOKEN_ID':'151643'},
                            student_eos=student,teacher_eos=teacher)


def test_launcher_uses_selected_initializer_in_preflight_and_actual_path_loading():
    text='check "$MODULAR_RL_INITIALIZERS" base --paths\nload "$MODULAR_RL_INITIALIZERS" base --paths\n'
    actual=allocation_for_initializer(text)
    assert '"$MODULAR_RL_INITIALIZERS" base' not in actual
    assert actual.count('"${MODULAR_RL_INITIALIZER:-base}"')==2
    with pytest.raises(ValueError,match='initializer calls'):allocation_for_initializer('load base')
