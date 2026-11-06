import json

import numpy as np
import pytest

from modular_rl.dolci.data import MixedBlocks
from modular_rl.dolci.fresh_data import question_key


def test_question_dedup_ignores_response_and_followup_variants():
    one = [{'role': 'user', 'content': '  Write A\nFunction. '},
           {'role': 'assistant', 'content': 'solution one'}]
    two = [{'role': 'user', 'content': 'write a function.'},
           {'role': 'assistant', 'content': 'solution two'},
           {'role': 'user', 'content': 'try again'}]
    assert question_key(one) == question_key(two)
    assert question_key([{'role': 'assistant', 'content': 'answer'}]) is None


def test_fresh_pool_cannot_cycle_and_each_selected_block_is_unique(tmp_path):
    source = tmp_path / 'sources'; source.mkdir()
    np.arange(15, dtype='<u4').tofile(source / 'code.bin')
    np.ones(15, dtype='u1').tofile(source / 'code.mask')
    manifest = {'seq_len': 4, 'seed': 12, 'objective': 'assistant', 'require_unique_blocks': True,
                'sources': {'code': {'packed_tokens': 15}}, 'quotas': {'code': {'code': 3}}}
    p = tmp_path / 'manifest.json'; p.write_text(json.dumps(manifest))
    stream = MixedBlocks(tmp_path, 'code')
    assert len(set(stream.indices.tolist())) == 3
    assert stream.loss_tokens(0, 3) == 12
    manifest['quotas']['code']['code'] = 4
    p.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='repeat blocks'):
        MixedBlocks(tmp_path, 'code')
