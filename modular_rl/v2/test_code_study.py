from .base_sweep import Cell, validate_cells
from .code_study import cells, split_examples


def test_confirmation_is_disjoint_by_problem_group_even_for_distinct_api_keys():
    rows = [{'metadata': {'key': str(i), 'split_group': str(i)}} for i in range(10)]
    rows += [{'metadata': {'key': 'other-api', 'split_group': '0'}}]
    parts = split_examples(rows, monitor=2, confirmation=3)
    assert {k: len(v) for k, v in parts.items()} == {'train': 6, 'val': 2, 'confirmation': 3}
    groups = {k: {r['metadata']['split_group'] for r in v} for k, v in parts.items()}
    assert '0' in groups['train'] and '0' not in groups['val'] | groups['confirmation']
    assert not groups['train'] & groups['val'] and not groups['train'] & groups['confirmation']
    assert not groups['val'] & groups['confirmation']
    assert split_examples(list(reversed(rows)), monitor=2, confirmation=3) == parts
    assert all(r['metadata']['environment'] == 'code' for split in parts.values() for r in split)
    assert all('environment' not in r['metadata'] for r in rows)


def test_code_grid_only_has_base_models_and_one_objective_pass():
    grid = validate_cells([Cell(**r) for r in cells()])
    assert len(grid) == 24 and {c.task for c in grid} == {'code'}
    assert {c.passes for c in grid} == {1} and {c.responses for c in grid} == {8000}
    assert {c.reward_variant for c in grid} == {'binary', 'partial'}
    assert {c.answer_reserve for c in grid} == {512, 1024}


def test_optional_prefix_conditions_are_full_raw_base_runs():
    grid = validate_cells([Cell(**r) for r in cells(include_prefix=True)])
    assert len(grid) == 26
    prefix = [c for c in grid if c.code_format == 'prefix']
    assert len(prefix) == 2 and {c.model for c in prefix} == {'Qwen3-0.6B-Base', 'Qwen3-1.7B-Base'}
    assert {(c.cap, c.responses, c.passes, c.reward_variant, c.reference_kl) for c in prefix} == {
        (2048, 8000, 1, 'binary', .01)}


def test_prepared_metadata_is_accepted_by_native_eval_rollout_tables():
    from types import SimpleNamespace

    from slime.utils.eval_rollouts import select_eval_rollouts

    rows = [{'metadata': {'key': str(i), 'split_group': str(i)}} for i in range(4)]
    parts = split_examples(rows, monitor=1, confirmation=1)
    sample = SimpleNamespace(metadata=parts['val'][0]['metadata'], index=0, prompt='Writecode',
                             response='def f(): return1', label=None, reward={'accuracy': 1},
                             status=SimpleNamespace(value='completed'), response_length=8)
    examples = select_eval_rollouts([sample], samples_per_prompt=1, reward_key='accuracy')
    assert len(examples) == 1 and examples[0]['environment'] == 'code' and examples[0]['score'] == 1
