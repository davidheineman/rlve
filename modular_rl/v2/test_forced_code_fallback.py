from modular_rl.graders import grade_code
from modular_rl.v2.rollout import code_submission


TARGET={'setup':'','tests':['assert increment(4) == 5']}
CORRECT='def increment(x):\n    return x + 1\n'
WRONG='def increment(x):\n    return x - 1\n'


def submission(earlier,final,*,fallback=True):
    return code_submission(earlier+final,{'answer_char_start':len(earlier)},fallback=fallback)


def test_opt_in_fallback_preserves_complete_earlier_code_when_forced_fence_is_malformed():
    malformed='\n\n```python\n# Repeated explanation; no completed fence or solution'
    earlier=CORRECT+'\n# Repeated but valid comments\n'*10
    assert grade_code(submission(earlier,malformed),TARGET)['accuracy']==1
    assert grade_code(submission(earlier,malformed,fallback=False),TARGET)['accuracy']==0


def test_valid_final_implementation_wins_even_when_earlier_solution_would_pass():
    final='\n```python\n'+WRONG+'```'
    assert submission(CORRECT,final)==final
    assert grade_code(submission(CORRECT,final),TARGET)['accuracy']==0


def test_incomplete_earlier_fence_does_not_override_complete_forced_solution():
    final='\n```python\n'+CORRECT+'```'
    assert submission('```python\ndef broken(',final)==final
    assert grade_code(submission('```python\ndef broken(',final),TARGET)['accuracy']==1


def test_fallback_does_not_change_unit_tests_or_repair_incorrect_code():
    assert grade_code(submission(WRONG,'\n```python\n'),TARGET)['accuracy']==0
    assert submission('def broken(', '\n```python\n')=='\n```python\n'
