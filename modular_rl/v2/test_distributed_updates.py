"""Real two-process DDP checks for collective update/skip and probe accounting."""

import copy
import json
import os
import sys
from datetime import timedelta
from pathlib import Path
from types import ModuleType, SimpleNamespace

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import pytest

from .test_updates import options
from .update_probe import probe
from .updates import maybe_train


def worker(rank, directory, world=2):
    directory = Path(directory)
    os.environ.update(MODULAR_RL_SKIP_ZERO_ADVANTAGE='1', MODULAR_RL_RUN_DIR=str(directory),
                      MODULAR_RL_UPDATE_PROBE_INTERVAL='1')
    dist.init_process_group('gloo', init_method='file://' + str(directory / 'rendezvous'),
                            rank=rank, world_size=world, timeout=timedelta(seconds=40))
    try:
        torch.manual_seed(17)
        model = torch.nn.Linear(1, 1, bias=False)
        initial = copy.deepcopy(model.state_dict())
        ddp = torch.nn.parallel.DistributedDataParallel(model)
        optimizer = torch.optim.Adam(model.parameters(), lr=.01, betas=(.9, .98))
        args = options(actor_num_gpus_per_node=world)
        x = torch.arange(1., 9.).reshape(-1, 1)
        # Rank zero's shard is entirely zero; rank one still requires backward on both ranks.
        advantages = torch.tensor([0., -2., 0., -1., 0., 1., 0., 2.])
        local = advantages[rank::world]
        calls = []

        def train(rollout_id):
            calls.append(rollout_id)
            optimizer.zero_grad()
            loss = -(ddp(x[rank::world]).flatten() * local).mean()
            loss.backward()
            optimizer.step()

        maybe_train(args, {'advantages': [local]}, train, 0)
        assert calls == [0]
        reference = torch.nn.Linear(1, 1, bias=False)
        reference.load_state_dict(initial)
        ref_optimizer = torch.optim.Adam(reference.parameters(), lr=.01, betas=(.9, .98))
        (-(reference(x).flatten() * advantages).mean()).backward()
        ref_optimizer.step()
        torch.testing.assert_close(model.weight, reference.weight, atol=1e-7, rtol=0)
        before, moments = model.weight.detach().clone(), copy.deepcopy(optimizer.state[model.weight])
        maybe_train(args, {'advantages': [torch.zeros(4)]}, train, 1)
        assert calls == [0]
        torch.testing.assert_close(model.weight, before, rtol=0, atol=0)
        for key, value in moments.items():
            torch.testing.assert_close(optimizer.state[model.weight][key], value, rtol=0, atol=0)
        maybe_train(args, {'raw_reward': [1/3]*8, 'advantages': [torch.full((4,), .029802322)]}, train, 2)
        assert calls == [0]
        torch.testing.assert_close(model.weight, before, rtol=0, atol=0)
        for key, value in moments.items():
            torch.testing.assert_close(optimizer.state[model.weight][key], value, rtol=0, atol=0)
        try:
            maybe_train(args, {'advantages': [torch.tensor([float('nan') if rank else 0.])]}, train, 3)
        except ValueError as exc:
            assert 'finite' in str(exc)
        else:
            raise AssertionError('Nonfinite advantages must fail on both ranks')
        # Exercise actual distributed gathering of all eight response diagnostics.
        backend = ModuleType('slime.backends.megatron_utils.model')
        backend.forward_only = lambda *a: {'log_probs': [torch.tensor([-.4])] * (8 // world)}
        sys.modules[backend.__name__] = backend
        megatron = ModuleType('megatron.core')
        megatron.mpu = SimpleNamespace(get_data_parallel_group_gloo=lambda **kw: dist.group.WORLD)
        sys.modules['megatron.core'] = megatron
        actor = SimpleNamespace(model=None, args=SimpleNamespace(
            tensor_model_parallel_size=1, pipeline_model_parallel_size=1, context_parallel_size=1,
            balance_data=False, train_reuse_epochs=1, rollout_max_response_len=2048,
            eps_clip=.2, eps_clip_high=.28, use_wandb=False))
        probe(actor, 0, {'raw_reward': list(range(8)), 'rewards': list(range(4)),
                        'log_probs': [torch.tensor([-.5])] * (8 // world), 'loss_masks': [[1]] * (8 // world),
                        'advantages': [torch.tensor([float(v)]) for v in local]}, None, None)
        dist.barrier()
        if rank == 0:
            decisions = [json.loads(line) for line in (directory / 'optimizer-decisions.jsonl').open()]
            assert [(r['rollout_id'], r['optimizer_update_applied']) for r in decisions] == [(0, True), (1, False), (2, False)]
            assert all(r['policy_world_size'] == world for r in decisions)
            rows = [json.loads(line) for line in (directory / 'update-probes.jsonl').open()]
            assert len(rows) == 1
            assert [r['sample'] for r in rows[0]['samples']] == list(range(8))
            assert [r['reward'] for r in rows[0]['samples']] == list(range(8))
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize('world', [2, 4])
def test_distributed_update_matches_global_objective_and_skips_collectively(tmp_path, world):
    mp.spawn(worker, args=(str(tmp_path), world), nprocs=world, join=True)
