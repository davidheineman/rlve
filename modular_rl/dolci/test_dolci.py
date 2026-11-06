import ast
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from .config import ACTUAL_TOKENS, SEQ_LEN, STEPS, TOKENS_PER_STEP, apportion, family, mixtures
from .data import MixedBlocks
from .training import trainer_source
from .watch import TASKS, select_web


def test_mixture_weights_preserve_empirical_tokens_and_conditional_ratios():
    sources = {str(i): {'tokens': tokens, 'domain': domain} for i, (tokens, domain) in enumerate(
        [(20, 'Math'), (10, 'Math'), (15, 'Coding'), (5, 'Chat'), (40, 'Science'), (10, 'Safety')])}
    mixes = mixtures(sources)
    assert mixes['baseline']['0'] == .2
    for mix in ('math', 'code', 'chat', 'web'):
        weights = mixes[mix]
        assert sum(weights.values()) == pytest.approx(1)
        assert sum(w for n, w in weights.items() if family(sources[n]['domain']) == mix) == pytest.approx(.8)
        for category in (True, False):
            ratios = [weights[n] / mixes['baseline'][n] for n in weights if (family(sources[n]['domain']) == mix) == category]
            assert max(ratios) == pytest.approx(min(ratios))
        counts = apportion(ACTUAL_TOKENS // SEQ_LEN, weights)
        assert sum(counts.values()) * SEQ_LEN == STEPS * TOKENS_PER_STEP == 999817216


def test_virtual_repetition_is_deterministic_and_masks_shift_with_targets(tmp_path):
    (tmp_path / 'sources').mkdir()
    tokens = np.arange(15, dtype='<u4').reshape(3, 5)
    mask = np.asarray([[0, 0, 1, 1, 1], [0, 1, 0, 0, 1], [1, 1, 1, 0, 0]], dtype='u1')
    tokens.tofile(tmp_path / 'sources/s00.bin')
    mask.tofile(tmp_path / 'sources/s00.mask')
    manifest = {'seq_len': 4, 'seed': 42, 'objective': 'assistant', 'sources': {'s00': {'packed_tokens': 15}},
                'quotas': {'baseline': {'s00': 8}}}
    (tmp_path / 'manifest.json').write_text(json.dumps(manifest))
    data = MixedBlocks(tmp_path, 'baseline')
    duplicate = MixedBlocks(tmp_path, 'baseline')
    np.testing.assert_array_equal(data.indices, duplicate.indices)
    for start in (0, 3):
        assert set(data.indices[start:start + 3]) == {0, 1, 2}
    x, y = data.batch(0, 8, 'cpu')
    assert data.loss_tokens(0, 8) == int(y.ne(-100).sum())
    np.testing.assert_array_equal(x.numpy(), tokens[data.indices, :-1])
    expected = tokens[data.indices, 1:].astype('int64')
    expected[~mask[data.indices, 1:].astype(bool)] = -100
    np.testing.assert_array_equal(y.numpy(), expected)


def test_masked_chunked_loss_and_global_token_normalization_match_dense_gradients():
    reference = Path('/juice5b/u/dhei/tatsu/nano-omni/experiments/midtraining/train.py')
    if not reference.exists():
        pytest.skip('Requires the sibling reference trainer')
    parsed = ast.parse(trainer_source(reference.read_text()))
    definitions = [n for n in parsed.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in ('linear_ce', 'CausalLoss')]
    scope = {'torch': torch, 'F': F, 'nn': nn, 'checkpoint': checkpoint}
    exec(compile(ast.Module(body=definitions, type_ignores=[]), 'frozen-sft-loss', 'exec'), scope)

    class Backbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(17, 5)

        def forward(self, input_ids, use_cache=False):
            return SimpleNamespace(last_hidden_state=self.embedding(input_ids))

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = Backbone()
            self.lm_head = nn.Linear(5, 17, bias=False)

    torch.manual_seed(42)
    model = Model()
    dense = copy.deepcopy(model)
    x = torch.randint(17, (3, 4))
    y = torch.randint(17, (3, 4))
    y[0, :3] = -100
    y[1, :] = -100  # A prompt-only microbatch must have finite zero loss.
    y[2, :1] = -100
    total = int(y.ne(-100).sum())
    wrapped = scope['CausalLoss'](model, chunk=2)
    losses = [wrapped(x[i:i + 1], y[i:i + 1], normalization=total / 3) / 3 for i in range(3)]
    loss = sum(losses)
    dense_loss = F.cross_entropy(dense.lm_head(dense.model(x).last_hidden_state).flatten(0, 1), y.flatten())
    torch.testing.assert_close(loss, dense_loss)
    loss.backward()
    dense_loss.backward()
    for actual, expected in zip(model.parameters(), dense.parameters(), strict=True):
        torch.testing.assert_close(actual.grad, expected.grad)


def test_assistant_masks_match_unmodified_qwen_template_and_preserve_tools():
    from . import prepare as p
    model = Path('/nlp/scr/dhei/models/Qwen3-0.6B-Base')
    if not model.exists():
        pytest.skip('Pinned tokenizer not installed')
    p.initialize(model, [])
    messages = [{'role': 'system', 'content': 'Use functions.', 'functions': '[{"name":"weather"}]'},
                {'role': 'user', 'content': 'PRIVATE USER REQUEST'},
                {'role': 'assistant', 'content': None, 'function_calls': 'weather(city="Paris")'},
                {'role': 'environment', 'content': 'PRIVATE TOOL RESULT'},
                {'role': 'assistant', 'content': 'The weather is sunny.'}]
    ids, mask, clean, rendered = p.encode(messages)
    assert rendered == p.TOKENIZER.apply_chat_template(clean, tokenize=False, add_generation_prompt=False)
    targets = p.TOKENIZER.decode([i for i, m in zip(ids, mask) if m])
    assert 'weather(city="Paris")' in targets and 'The weather is sunny.' in targets
    assert 'PRIVATE USER REQUEST' not in targets and 'PRIVATE TOOL RESULT' not in targets
    assert '<functions>' in rendered and '<function_calls>' in rendered
    assert '<|im_end|>' in targets
    multilingual = [{'role': 'user', 'content': '秘密问题: 你好，世界。'},
                    {'role': 'assistant', 'content': '答案是四。 উত্তর চার।'}]
    ids, mask, clean, rendered = p.encode(multilingual)
    targets = p.TOKENIZER.decode([i for i, m in zip(ids, mask) if m])
    assert '答案是四。' in targets and 'উত্তর চার।' in targets and '秘密问题' not in targets


def test_fourteen_rl_runs_match_requested_task_mapping():
    assert sum(map(len, TASKS.values())) == 14
    assert set(TASKS['base']) == set(TASKS['baseline']) == {'math', 'code', 'chat', 'if'}
    assert TASKS['math'] == ('math',) and TASKS['code'] == ('code',)
    assert set(TASKS['web']) == set(TASKS['chat']) == {'chat', 'if'}


def test_both_natural_definitions_and_full_c4_model_selection(tmp_path):
    sources = {'math': {'tokens': 20, 'domain': 'Math'}, 'code': {'tokens': 10, 'domain': 'Coding'},
               'chat': {'tokens': 20, 'domain': 'Chat'}, 'science': {'tokens': 20, 'domain': 'Science'},
               'other': {'tokens': 10, 'domain': 'Other'}, 'safety': {'tokens': 20, 'domain': 'Safety'}}
    broad, knowledge = (mixtures(sources, natural=n)['web'] for n in ('complement', 'knowledge'))
    assert broad['science'] + broad['other'] + broad['safety'] == pytest.approx(.8)
    assert knowledge['science'] + knowledge['other'] == pytest.approx(.8)
    assert knowledge['safety'] < broad['safety']
    jobs, slurm = [], {}
    for i, (mix, bpb) in enumerate((('web', 1.2), ('web-knowledge', 1.1))):
        output = tmp_path / mix
        output.mkdir()
        (output / 'results.json').write_text(json.dumps({'train_tokens': ACTUAL_TOKENS,
            'eval': {'paloma_c4_en': {'bpb': bpb, 'n_docs': 512, 'n_tokens': 12345}}}))
        jobs.append({'kind': 'sft', 'mix': mix, 'job_id': str(i), 'output': str(output)})
        slurm[str(i)] = {'state': 'COMPLETED'}
    assert not select_web(tmp_path, jobs, {'0': slurm['0'], '1': {'state': 'RUNNING'}})
    assert not (tmp_path / 'web-selection.json').exists()
    assert select_web(tmp_path, jobs, slurm)
    assert json.loads((tmp_path / 'web-selection.json').read_text())['chosen_mix'] == 'web-knowledge'


def test_tokenization_rowgroup_units_cover_each_row_once_and_resume(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from . import prepare as p
    model = Path('/nlp/scr/dhei/models/Qwen3-0.6B-Base')
    if not model.exists():
        pytest.skip('Pinned tokenizer not installed')
    p.initialize(model, [])
    rows = [{'id': str(i), 'source_dataset': 'toy', 'domain': 'Math',
             'messages': [{'role': 'user', 'content': f'What is {i}+1?'},
                          {'role': 'assistant', 'content': str(i + 1)}]} for i in range(5)]
    path = tmp_path / 'toy.parquet'
    pq.write_table(pa.Table.from_pylist(rows), path, row_group_size=2)
    parts = tmp_path / 'parts'
    pairs = {('toy', 'Math'): 's00'}
    counts = []
    for groups in ([0, 1], [2]):
        arguments = (path, str(parts), pairs, groups)
        first = p.tokenize_shard(arguments)
        assert p.tokenize_shard(arguments) == first
        binary = parts / first['shard'] / 's00.bin'
        original = binary.read_bytes()
        binary.unlink()
        assert p.tokenize_shard(arguments) == first
        assert binary.read_bytes() == original
        counts.append(first['counts']['s00'])
    assert sum(c['rows'] for c in counts) == sum(c['original_rows'] for c in counts) == 5
    assert sum(x.stat().st_size for x in parts.glob('*/s00.bin')) == sum(c['tokens'] for c in counts) * 4


def test_completed_optimizer_retirement_requires_exact_preserved_weights(tmp_path, monkeypatch):
    from safetensors.torch import save_file
    from . import storage
    (tmp_path / 'model').mkdir()
    model = {'model.weight': torch.arange(4, dtype=torch.float32)}
    state_path = tmp_path / 'train_state.pt'
    torch.save({'next_step': STEPS, 'model': model}, state_path)
    (tmp_path / 'results.json').write_text(json.dumps({'train_tokens': ACTUAL_TOKENS,
        'config': {'objective': 'assistant', 'tokens_per_step': ACTUAL_TOKENS // STEPS}}))
    weight = tmp_path / 'model/model.safetensors'
    save_file({'model.weight': model['model.weight'] + 1}, weight)
    monkeypatch.setattr(storage, 'EXPECTED_PARAMETERS', 4)
    with pytest.raises(ValueError, match='differs'):
        storage.retire(tmp_path)
    assert state_path.exists() and weight.exists()
    save_file(model, weight)
    result = storage.retire(tmp_path)
    assert result['all_final_model_weights_preserved'] and result['parameter_elements_compared'] == 4
    assert not state_path.exists() and weight.exists()
    assert storage.retire(tmp_path) is None


def test_sft_export_stops_on_supervised_turn_end_and_preserves_architecture(tmp_path):
    from .export import ENDINGS, finalize_generation
    model = tmp_path / 'model'
    model.mkdir()
    (tmp_path / 'results.json').write_text(json.dumps({'train_tokens': ACTUAL_TOKENS, 'config': {'objective': 'assistant'}}))
    config = model / 'config.json'
    config.write_text(json.dumps({'eos_token_id': 151643}))
    original_config = config.read_bytes()
    (model / 'tokenizer_config.json').write_text(json.dumps({'added_tokens_decoder': {
        '151643': {'content': '<|endoftext|>'}, '151645': {'content': '<|im_end|>'}}}))
    generation = model / 'generation_config.json'
    generation.write_text(json.dumps({'eos_token_id': 151643, 'do_sample': False, 'max_new_tokens': 2048}))
    proof = finalize_generation(tmp_path)
    assert proof['eos_token_id'] == ENDINGS
    assert config.read_bytes() == original_config
    updated = json.loads(generation.read_text())
    assert updated['eos_token_id'] == ENDINGS and not updated['do_sample'] and updated['max_new_tokens'] == 2048
    before = generation.stat().st_mtime_ns
    assert finalize_generation(tmp_path) == json.loads((tmp_path / 'generation-stop-tokens.json').read_text())
    assert generation.stat().st_mtime_ns == before


def test_rollout_audit_preserves_unicode_separators_inside_json_strings(tmp_path):
    from .report import events
    rows = [{'response': 'one\u2028two\u2029three\u0085four'}, {'response': 'next'}]
    path = tmp_path / 'rollouts.jsonl'
    path.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows))
    assert events(path) == rows


def test_chat_reward_normalizes_assistant_endings_without_mutating_rollouts():
    from .chat_reward import normalize_conversation
    messages = [{'role': 'user', 'content': 'Explain <|im_end|>'},
                {'role': 'assistant', 'content': 'Answer.\n<|im_end|> <|endoftext|>\n'},
                {'role': 'assistant', 'content': 'Plain answer.  '}]
    original = json.loads(json.dumps(messages))
    clean = normalize_conversation(messages)
    assert clean[0] == messages[0] and clean[1]['content'] == 'Answer.'
    assert clean[2] == messages[2] and messages == original
    for ending in ('', '<|endoftext|>', '<|im_end|>'):
        assert normalize_conversation([{'role': 'assistant', 'content': 'Answer.' + ending}]) == [
            {'role': 'assistant', 'content': 'Answer.'}]


def test_chat_reward_hook_normalizes_before_cache_and_tokenization(tmp_path):
    from .chat_reward import install
    root = Path(__file__).resolve().parents[1]
    (tmp_path / 'modular_rl').mkdir()
    path = tmp_path / 'modular_rl/skywork.py'
    path.write_text((root / 'skywork.py').read_text())
    proof = install(tmp_path, Path(__file__).with_name('chat_reward.py'))
    text = path.read_text()
    assert proof['before_sha256'] != proof['after_sha256']
    assert text.index('messages = normalize_conversation') < text.index('key = digest(messages)')
    assert text.index('messages = normalize_conversation') < text.index('ids = self.tokenizer.apply_chat_template')
    with pytest.raises(ValueError, match='Inspect'):
        install(tmp_path, Path(__file__).with_name('chat_reward.py'))


def test_finished_wandb_summary_waits_for_writers_and_repairs_from_raw_panel(tmp_path):
    from types import SimpleNamespace
    from .report import finalize_wandb
    directory = tmp_path / 'runs/example/policy'
    (directory / 'rollouts/eval').mkdir(parents=True)
    (directory / 'completed.json').write_text(json.dumps({'completed': True, 'updates': 1000, 'train_rollouts': 8000}))
    (directory / 'wandb_run_id.txt').write_text('example-id')
    selection = tmp_path / 'selection/chat'
    selection.mkdir(parents=True)
    keys = [str(i) for i in range(64)]
    (selection / 'selection.json').write_text(json.dumps({'keys': {'val': keys}}))
    rows = [{'metadata_json': json.dumps({'key': key}), 'correct': int(key) % 2 == 0}
            for key in keys for _ in range(8)]
    (directory / 'rollouts/eval/0999.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows))
    job = {'output': str(directory.parent), 'name': 'example', 'task': 'chat',
           'command': [str(directory.parent / 'run.sbatch')]}
    remote = SimpleNamespace(state='running', summary={}, load=lambda **kwargs: None)
    calls = []
    api = SimpleNamespace(run=lambda name: (calls.append(name), remote)[1])
    assert not finalize_wandb(job, api)
    assert not remote.summary and not (directory / 'wandb-final-summary.json').exists()
    remote.state = 'finished'
    assert finalize_wandb(job, api)
    assert remote.summary['david/eval/val_envs/pass@1'] == .5
    assert remote.summary['david/eval/val_envs/pass@8'] == .5
    assert remote.summary['david/rollouts'] == 8000 and remote.summary['david/step'] == 999
    assert finalize_wandb(job, api) and len(calls) == 2
