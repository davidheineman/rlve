import pytest

from modular_rl.graders import grade_code


@pytest.mark.parametrize("mutation", [
    "builtins.exec = lambda *a, **k: None",
    "fake = compile('pass', '<fake>', 'exec')\nbuiltins.compile = lambda *a, **k: fake",
])
def test_generated_code_cannot_disable_grader_by_replacing_executor(mutation):
    response = "import builtins\ndef add_one(x): return 0\n" + mutation
    assert grade_code(response, {"tests": ["assert add_one(5) == 6"], "setup": ""})["accuracy"] == 0


def test_valid_code_still_passes_precompiled_tests_with_setup():
    result = grade_code("def root(x): return math.sqrt(x)", {
        "setup": "import math", "tests": ["assert root(4) == 2", "assert root(9) == 3"]})
    assert result["accuracy"] == 1 and result["passes"] == [True, True]


def test_late_worker_cpu_limit_is_a_failed_sample_not_a_failed_training_job(monkeypatch):
    from types import SimpleNamespace

    process = SimpleNamespace(returncode=1, communicate=lambda *a, **k: (
        '', 'Traceback: json.dumps\nTimeoutError: CPU limit\n'))
    monkeypatch.setattr('modular_rl.graders.subprocess.Popen', lambda *a, **k: process)
    result = grade_code('def f(): pass', {'setup': '', 'tests': ['assert f() == 1']})
    assert result['reward'] == 0 and result['accuracy'] == 0 and result['error'] == 'worker_resource_limit'


def test_unrelated_executor_infrastructure_error_still_fails_loudly(monkeypatch):
    from types import SimpleNamespace

    process = SimpleNamespace(returncode=1, communicate=lambda *a, **k: (
        '', 'ModuleNotFoundError: missing grader runtime'))
    monkeypatch.setattr('modular_rl.graders.subprocess.Popen', lambda *a, **k: process)
    with pytest.raises(RuntimeError, match='executor failed'):
        grade_code('def f(): pass', {'setup': '', 'tests': ['assert f() == 1']})
