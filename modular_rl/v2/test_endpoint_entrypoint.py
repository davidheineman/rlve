import json
import runpy
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize('task', ['code', 'if'])
def test_endpoint_executes_as_standalone_script_without_training(tmp_path, monkeypatch, task):
    """Slurm launches a file path, so __main__ has no package for relative imports."""
    repo, checkpoint, output = (tmp_path / name for name in ("repo", "checkpoint", "output"))
    for path in (repo, checkpoint, output):
        path.mkdir()
    (checkpoint / "latest_checkpointed_iteration.txt").write_text("release")
    (repo / "train.py").write_text('''
import os
from types import SimpleNamespace

ray = SimpleNamespace(get=lambda value: value)

def parse_args():
    return SimpleNamespace(load=os.environ["TEST_ENDPOINT_CHECKPOINT"], use_wandb=False, answer_forcing=True)

def evaluate_student(args, actor, manager, rollout_id):
    return 0.75

def train(args):
    assert args.start_rollout_id == 0 and args.num_rollout == 1
    assert args.no_load_optim and args.no_load_rng
    assert not args.transactional_checkpoints and args.save is None
    if os.environ['MODULAR_RL_TASKS'] == 'if':
        assert not args.answer_forcing
    manager = SimpleNamespace(controller=SimpleNamespace(
        finish_wandb=SimpleNamespace(remote=lambda: None)))
    evaluate_student(args, None, manager, 0)
    raise AssertionError("Evaluation must terminate before any optimizer update")
''')
    entrypoint = Path(__file__).with_name("endpoint.py")
    monkeypatch.setenv("RLVE_REPO", str(repo))
    monkeypatch.setenv("MODULAR_RL_RUN_DIR", str(output))
    monkeypatch.setenv("MODULAR_RL_ENDPOINT_TRACKER", "release")
    monkeypatch.setenv("MODULAR_RL_ENDPOINT_PROMPTS", "32")
    monkeypatch.setenv("MODULAR_RL_ENDPOINT_NAME", "script-mode CPU regression")
    monkeypatch.setenv("TEST_ENDPOINT_CHECKPOINT", str(checkpoint))
    monkeypatch.setenv("MODULAR_RL_TASKS", task)
    monkeypatch.setitem(sys.modules, "rlve_endpoint_native", None)
    monkeypatch.setattr(sys, "argv", [str(entrypoint)])
    runpy.run_path(str(entrypoint), run_name="__main__")
    result = json.loads((output / "evaluated.json").read_text())
    assert result["evaluation_only"] and result["optimizer_updates"] == 0
    assert result["pass@1"] == .75
