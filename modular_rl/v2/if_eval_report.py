"""Replay Dolci IF grades, report strict/constraint metrics, and publish responses."""

import argparse
import hashlib
import json
import os
from collections import Counter
from pathlib import Path

from modular_rl.if_grader import grade_if
from modular_rl.prepare import write_json

from .analyze_endpoint import analyze as analyze_endpoint
from .base_models import BASE_MODELS
from .inspect_rollouts import read_events


def analyze(review):
    plan = json.loads((review / 'if-eval-experiments.json').read_text())
    if len(plan['experiments']) != len(plan['models']) * len(plan['caps']):
        raise ValueError('IF submission is incomplete')
    outcomes, examples = [], []
    for name in plan['experiments']:
        experiment = Path(name)
        os.environ['MODULAR_RL_RUNTIME'] = str(experiment / 'runtime')
        endpoint = json.loads((experiment / 'plan.json').read_text())
        result = analyze_endpoint(experiment)
        assert len(result['policies']) == 1
        policy = result['policies'][0]
        assert policy['evaluation_only'] and policy['optimizer_updates'] == policy['training_responses'] == 0
        directory = Path(policy['directory'])
        identity = json.loads((directory / 'initial-model-identity.json').read_text())
        model = endpoint['model'].split('/')[-1]
        assert identity['repo_id'] == endpoint['model']
        assert all(identity[k] == v for k, v in BASE_MODELS[model].items())
        assert Path(identity['actor_load_for_this_attempt']).resolve() == Path(policy['checkpoint']).resolve()
        assert not list((directory / 'rollouts/train').glob('*.jsonl'))
        panel = experiment / 'selection/if/val.jsonl'
        assert hashlib.sha256(panel.read_bytes()).hexdigest() == plan['panel_sha256']
        canonical = {r['metadata']['key']: r['metadata'] for r in read_events(panel)}
        rows = read_events(directory / 'rollouts/eval/0000.jsonl')
        assert len(rows) == plan['validation_prompts'] * plan['samples_per_prompt']
        counts, constraints, passed = Counter(), Counter(), Counter()
        rewards, lengths = [], []
        for index, row in enumerate(rows):
            meta, original = json.loads(row['metadata_json']), json.loads(row['reward_json'])
            assert all(meta[k] == canonical[meta['key']][k]
                       for k in ('messages', 'target', 'split_group', 'task', 'environment'))
            assert '_modular_v2_forcing' not in meta
            assert row['response_tokens'] <= endpoint['max_sampled_tokens']
            replay = grade_if(row['response'], meta['target'])
            assert replay == original, 'Saved IF grade differs from replay'
            assert bool(row['correct']) == bool(replay['accuracy'])
            counts[meta['key']] += 1
            rewards.append(replay['reward'])
            lengths.append(row['response_tokens'])
            for instruction, success in zip(replay['constraints'], replay['constraint_passes'], strict=True):
                constraints[instruction] += 1
                passed[instruction] += success
            examples.append({'model': endpoint['model'], 'cap': endpoint['max_sampled_tokens'],
                             'key': meta['key'], 'row': index, 'prompt': meta['messages'],
                             'response': row['response'], 'target': meta['target'],
                             'accuracy': replay['accuracy'], 'constraint_fraction': replay['reward']})
        assert set(counts) == set(canonical) and set(counts.values()) == {plan['samples_per_prompt']}
        outcomes.append({'model': endpoint['model'], 'cap': endpoint['max_sampled_tokens'],
                         'experiment': name, 'wandb_id': policy['wandb_id'],
                         'pass@1': policy['pass@1'], 'pass@8': policy['pass@8'],
                         'mean_constraint_fraction': sum(rewards) / len(rewards),
                         'constraint_micro_accuracy': sum(passed.values()) / sum(constraints.values()),
                         'mean_response_tokens': sum(lengths) / len(lengths),
                         'truncated_fraction': policy['behavior']['truncated_fraction'],
                         'responses': len(rows), 'optimizer_updates': 0, 'answer_forcing': False,
                         'per_constraint': {k: {'passed': passed[k], 'total': constraints[k]}
                                            for k in sorted(constraints)},
                         'panel_sha256': hashlib.sha256(panel.read_bytes()).hexdigest()})
    value = {'dataset': plan['dataset'], 'panel_sha256': plan['panel_sha256'],
             'all_saved_grades_replayed': True, 'outcomes': outcomes,
             'responses_replayed': len(examples), 'evaluation_only': True}
    write_json(review / 'if-eval-results.json', value)
    write_json(review / 'if-eval-complete-responses.json', examples)
    print(json.dumps(value), flush=True)
    return value, examples


def publish(review, audit_review):
    import tempfile

    import wandb

    value, examples = analyze(review)
    api = wandb.Api(timeout=30)
    runtime = []
    for outcome in value['outcomes']:
        remote = api.run('david-heineman/modular-mopd-rl-initial/' + outcome['wandb_id'])
        assert remote.state == 'finished', 'Wait for native W&B writers to finish before repairing summaries'
        expected = {'answer_forcing': False, 'eval_reward_key': 'accuracy',
                    'custom_generate_function_path': 'modular_rl.v2.if_rollout.generate',
                    'custom_rm_path': 'modular_rl.if_grader.reward',
                    'eval_max_response_len': outcome['cap'], 'n_samples_per_eval_prompt': 8,
                    'no_load_optim': True, 'no_load_rng': True}
        actual = {k: remote.config.get(k) for k in expected}
        assert actual == expected, f'IF runtime differs from declared settings: {actual}'
        remote.summary.update({'david/eval/val_envs/pass@1': outcome['pass@1'],
                               'david/eval/val_envs/pass@8': outcome['pass@8'],
                               'david/eval/if/mean_constraint_fraction': outcome['mean_constraint_fraction'],
                               'david/rollouts': 0, 'david/step': 0})
        remote.summary.update()
        runtime.append({'wandb_id': outcome['wandb_id'], 'actual_runtime': actual})
    write_json(review / 'if-eval-runtime-verification.json', runtime)
    columns = ['model', 'cap', 'pass@1', 'pass@8', 'mean_constraint_fraction',
               'constraint_micro_accuracy', 'mean_response_tokens', 'truncated_fraction', 'wandb_id']
    with tempfile.TemporaryDirectory(prefix='dolci-if-report-') as local:
        run_id = json.loads((audit_review / 'base-plan-publication.json').read_text())['run_id']
        run = wandb.init(entity='david-heineman', project='modular-mopd-rl-initial', id=run_id,
                         resume='must', dir=local)
        run.log({'if_eval/outcomes': wandb.Table(columns=columns,
                    data=[[o[k] for k in columns] for o in value['outcomes']]),
                 'if_eval/complete_responses': wandb.Table(columns=[
                     'model', 'cap', 'key', 'prompt', 'response', 'target', 'accuracy', 'constraint_fraction'],
                    data=[[r[k] if k not in ('prompt', 'target') else json.dumps(r[k], ensure_ascii=False)
                           for k in ('model', 'cap', 'key', 'prompt', 'response', 'target', 'accuracy',
                                     'constraint_fraction')] for r in examples])})
        artifact = wandb.Artifact('dolci-if-evaluation-and-complete-responses', type='rl-diagnostics')
        for path in sorted(review.glob('*.json')):
            artifact.add_file(str(path), name=path.name)
        for name in ('if_grader.py', 'if_controls.py', 'requirements-if.lock.txt'):
            path = Path(__file__).parents[1] / name
            artifact.add_file(str(path), name=name)
        for outcome in value['outcomes']:
            experiment = Path(outcome['experiment'])
            prefix = outcome['model'].split('/')[-1] + '-gen' + str(outcome['cap'])
            for name in ('plan.json', 'jobs.json'):
                artifact.add_file(str(experiment / name), name=prefix + '/' + name)
            artifact.add_dir(str(experiment / 'selection/if'), name=prefix + '/selection')
            model = outcome['model'].split('/')[-1]
            directory = experiment / 'if/base' / (model + '-if')
            for name in ('evaluated.json', 'initial-model-identity.json', 'generation.jsonl',
                         'rollouts/eval/0000.jsonl'):
                artifact.add_file(str(directory / name), name=prefix + '/' + name)
        logged = run.log_artifact(artifact)
        logged.wait()
        status = ('Code confirmed on both Base models; Dolci IF evaluation verified'
                  if run.summary.get('code_confirmed') else 'Dolci IF evaluation verified')
        run.summary.update({'IFEval_started': True, 'IFEval_working': True, 'status': status,
                            'if_eval/responses_replayed': len(examples)})
        receipt = {'run_url': run.url, 'artifact_url': logged.url}
        run.finish()
    write_json(review / 'if-eval-publication.json', receipt)
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('review', type=Path)
    parser.add_argument('--publish', action='store_true')
    parser.add_argument('--audit-review', type=Path)
    args = parser.parse_args()
    if args.publish:
        if args.audit_review is None:
            parser.error('--publish requires --audit-review')
        publish(args.review, args.audit_review)
    else:
        analyze(args.review)
