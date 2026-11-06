"""Opt-in token-normalized MaxRL, MetaRL, and MAI adaptive entropy clipping.

MAI-Thinking-1, equations 5--9: the controller observes an importance-weighted
sampled surprisal, not the full-vocabulary entropy or a length-scaled logger.
The controller advances once per optimizer step, never once per microbatch.
Only experiment-local frozen trainer copies install the lifecycle hooks.
"""

import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import torch


@dataclass
class Controller:
    target: float = .3
    delta: float = .25
    maximum: float = 2.5
    k: float = 0.
    steps: int = 0

    def __post_init__(self):
        if (not all(math.isfinite(x) for x in (self.target, self.delta, self.maximum, self.k))
                or self.target <= 0 or self.delta <= 0 or self.maximum < 0
                or not 0 <= self.k <= self.maximum or self.steps < 0):
            raise ValueError('Invalid adaptive entropy controller state')

    def update(self, observed):
        if not math.isfinite(observed):
            raise ValueError('Nonfinite adaptive entropy estimate')
        direction = int(observed < self.target) - int(observed > self.target)
        self.k = min(self.maximum, max(0., self.k + self.delta * direction))
        self.steps += 1


def clipped_surrogate(log_ratio, advantages, epsilon, upper, outer=None):
    """Return the minimized PPO surrogate with finite hard outer clipping."""
    if not 0 < epsilon < 1 or upper < 1 or (outer is not None and outer < upper):
        raise ValueError('Invalid trust-region or outer bounds')
    bounded_log = log_ratio if outer is None else log_ratio.clamp(max=math.log(outer))
    ratio = bounded_log.exp()
    trust = ratio.clamp(min=1 - epsilon, max=upper)
    return torch.minimum(ratio * advantages, trust * advantages)


def weighted_surprisal(log_probs, sampling_log_probs, mask):
    """Equation 7 numerator and denominator; detached controller statistics."""
    with torch.no_grad():
        logp, old, keep = [x.detach().double() for x in (log_probs, sampling_log_probs, mask)]
        ratio = (logp - old).exp()
        return torch.stack(((-logp * ratio * keep).sum(), keep.sum()))


_runtime = None


def initialize(args, loaded_iteration):
    global _runtime
    mode = os.environ['MODULAR_RL_EXPLORATION_METHOD']
    if mode not in ('grpo', 'maxrl', 'mai', 'metarl'):
        raise ValueError('Expected GRPO, MaxRL, MetaRL, or MAI exploration method')
    if (args.actor_num_nodes * args.actor_num_gpus_per_node not in (1, 2, 4)
            or args.tensor_model_parallel_size != 1 or args.pipeline_model_parallel_size != 1
            or args.context_parallel_size != 1 or args.calculate_per_token_loss
            or args.use_kl_loss or args.kl_coef != 0 or args.entropy_coef != 0
            or args.train_reuse_epochs not in (1, 4)
            or (mode == 'mai' and args.train_reuse_epochs != 4)):
        raise ValueError('Exploration comparison requires supported pure data parallelism and one/four passes; MAI needs four')
    config = {'target': float(os.environ.get('MODULAR_RL_ENTROPY_TARGET', '.3')),
              'delta': float(os.environ.get('MODULAR_RL_ENTROPY_DELTA', '.25')),
              'maximum': float(os.environ.get('MODULAR_RL_ENTROPY_KMAX', '2.5'))}
    controller = Controller(**config)
    method_config = {'optimize_pass_k': args.optimize_pass_k} if mode == 'metarl' else {}
    recipe_config = {'objective_passes': args.train_reuse_epochs,
                     'prompts_per_update': getattr(args, 'rollout_batch_size', 1),
                     'samples_per_prompt': getattr(args, 'n_samples_per_prompt', 8)}
    if Path(args.load).resolve() == Path(args.save).resolve() and loaded_iteration >= 0:
        state = json.loads((Path(args.load) / f'iter_{loaded_iteration:07d}' / 'exploration-state.json').read_text())
        if state['method'] != mode or state['rollout_id'] != loaded_iteration:
            raise ValueError('Exploration state belongs to another checkpoint')
        if state.get('method_config', {}) != method_config:
            raise ValueError('Exploration objective settings changed on resume')
        if state.get('recipe_config', recipe_config) != recipe_config:
            raise ValueError('Exploration batch or objective passes changed on resume')
        if any(state['controller'][k] != v for k, v in config.items()):
            raise ValueError('Entropy controller settings changed on resume')
        controller = Controller(**state['controller'])
    _runtime = {'method': mode, 'method_config': method_config, 'recipe_config': recipe_config,
                'controller': controller, 'tokens': None, 'stats': None,
                'rollout_id': None, 'pass_id': None}


def prepare_rollout(args, rollout_data, rollout_id):
    """Count each original unmasked response token once across all DP ranks."""
    if _runtime is None:
        raise RuntimeError('Exploration state was not initialized')
    masks = rollout_data['loss_masks']
    count = torch.stack([m.detach().double().sum() for m in masks]).sum()
    if torch.distributed.is_initialized():
        torch.distributed.all_reduce(count)
    if not math.isfinite(count.item()) or count.item() <= 0:
        raise ValueError('No valid response tokens in the global batch')
    _runtime.update(tokens=count.item(), rollout_id=rollout_id)


def before_step(args, rollout_id, step_id, *unused):
    if _runtime is None or _runtime['rollout_id'] != rollout_id or not _runtime['tokens']:
        raise RuntimeError('Missing global token denominator for optimizer step')
    _runtime.update(stats=torch.zeros(4, dtype=torch.float64, device=torch.cuda.current_device()),
                    pass_id=step_id)


def policy_loss(args, batch, logits, sum_of_sample_mean):
    from slime.backends.megatron_utils.cp_utils import get_sum_of_sample_mean
    from slime.backends.megatron_utils.loss import get_log_probs_and_entropy

    del sum_of_sample_mean
    if _runtime is None or _runtime['stats'] is None:
        raise RuntimeError('Loss called outside an exploration optimizer step')
    sums = get_sum_of_sample_mean(batch['total_lengths'], batch['response_lengths'],
                                  batch['loss_masks'], calculate_per_token_loss=True)
    # Native loss_function divides by global_batch_size, then Megatron cancels
    # the microbatch and DP factors. This yields SUM(valid loss)/GLOBAL(tokens).
    reduce = lambda values: sums(values) * args.global_batch_size / _runtime['tokens']
    values = get_log_probs_and_entropy(logits, args=args, unconcat_tokens=batch['unconcat_tokens'],
                                      total_lengths=batch['total_lengths'],
                                      response_lengths=batch['response_lengths'], with_entropy=True)
    logp = torch.cat(values['log_probs'])
    old = torch.cat(batch['log_probs']).detach()
    sampling = torch.cat(batch['rollout_log_probs']).detach()
    advantages = torch.cat(batch['advantages']).detach()
    mask = torch.cat(batch['loss_masks']).to(logp)
    log_ratio = logp - old
    controller = _runtime['controller']
    mai = _runtime['method'] == 'mai'
    upper = 1 / (1 - args.eps_clip) + controller.k if mai else 1 + args.eps_clip_high
    objective = clipped_surrogate(log_ratio, advantages, args.eps_clip, upper,
                                  outer=50. if mai else None)
    # Keep the proven trainer/inference numerical importance correction.
    tis = (old - sampling).exp()
    correction = tis.clamp(min=args.tis_clip_low, max=args.tis_clip) if args.use_tis else torch.ones_like(tis)
    loss = -reduce(objective * correction)
    with torch.no_grad():
        estimate = weighted_surprisal(logp, sampling, mask)
        affected = ((advantages > 0) & (log_ratio > math.log(1 / (1 - args.eps_clip)))
                    & (log_ratio < math.log(upper))) if mai else torch.zeros_like(mask, dtype=torch.bool)
        _runtime['stats'] += torch.cat((estimate, torch.stack(((affected * mask).double().sum(),
                                                             (log_ratio.detach().abs() * mask).double().sum()))))
    ratio = log_ratio.detach().exp()
    clip = ((advantages > 0) & (ratio > upper)) | ((advantages < 0) & (ratio < 1 - args.eps_clip))
    reported = {'loss': loss.detach(), 'pg_loss': loss.detach(),
                'entropy_loss': reduce(torch.cat(values['entropy'])).detach(),
                'pg_clipfrac': reduce(clip.float()).detach(), 'ppo_kl': reduce(-log_ratio).detach(),
                'tis': reduce(tis).detach(), 'ois': reduce(ratio).detach(),
                'tis_clipfrac': reduce((tis != correction).float()).detach(),
                'normalization_mass': reduce(torch.ones_like(logp)).detach()}
    return loss, reported


def after_step(args, rollout_id, step_id):
    """All-reduce statistics and update k once after a successful Adam step."""
    stats = _runtime['stats']
    if stats is None or _runtime['pass_id'] != step_id or _runtime['rollout_id'] != rollout_id:
        raise RuntimeError('Entropy controller step mismatch')
    if torch.distributed.is_initialized():
        torch.distributed.all_reduce(stats)
    numerator, count, affected, delta_logp = stats.tolist()
    if abs(count - _runtime['tokens']) > .5:
        raise RuntimeError('Microbatches duplicated or omitted global response tokens')
    observed = numerator / count
    controller = _runtime['controller']
    previous = controller.k
    if _runtime['method'] == 'mai':
        controller.update(observed)
    else:
        controller.steps += 1
    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        metrics = {'david/step': rollout_id, 'david/exploration/pass': step_id,
                   'david/exploration/optimizer_steps': controller.steps,
                   'david/exploration/importance_weighted_entropy': observed,
                   'david/exploration/k_used': previous, 'david/exploration/k_next': controller.k,
                   'david/exploration/upper_clip': (1 / (1 - args.eps_clip) + previous
                                                  if _runtime['method'] == 'mai' else 1 + args.eps_clip_high),
                   'david/exploration/adaptive_gradient_fraction': affected / count,
                   'david/exploration/mean_abs_log_ratio': delta_logp / count}
        if not all(math.isfinite(v) for v in metrics.values()):
            raise ValueError('Nonfinite exploration diagnostics')
        path = Path(os.environ['MODULAR_RL_RUN_DIR']) / 'exploration.jsonl'
        with path.open('a') as stream:
            stream.write(json.dumps(metrics) + '\n')
        print('EXPLORATION', json.dumps(metrics), flush=True)
        if args.use_wandb:
            import wandb
            wandb.log(metrics)
    _runtime['stats'] = None


def save_state(args, iteration):
    """The controller sidecar is inside the same committed model checkpoint."""
    from slime.utils.checkpoint_state import atomic_write
    from slime.utils.distributed_utils import get_gloo_group
    if torch.distributed.get_rank() == 0:
        state = {'method': _runtime['method'], 'rollout_id': iteration,
                 'controller': asdict(_runtime['controller']), 'method_config': _runtime['method_config'],
                 'recipe_config': _runtime['recipe_config']}
        path = Path(args.save) / f'iter_{iteration:07d}' / 'exploration-state.json'
        atomic_write(path, json.dumps(state, sort_keys=True) + '\n')
    torch.distributed.barrier(group=get_gloo_group())
