"""Exercise deletion boundaries and restore verification without network calls."""

import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from tools import archive_storage_review_20261026 as archive


class StorageArchiveTest(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec('huggingface_hub'), 'Run with the archive virtualenv')
    def test_cached_hash_operations_keep_real_file_content_without_rescanning(self):
        from huggingface_hub.lfs import UploadInfo
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'weights.bin'
            data = b'checkpoint-data' * 100000
            path.write_bytes(data)
            prior = {'files': {'weights': {'path': str(path), 'identity': archive.identity(path),
                'size': len(data), 'sha256': hashlib.sha256(data).hexdigest()}}}
            with patch.object(UploadInfo, 'from_path', side_effect=AssertionError('Must reuse immutable cached hash')):
                operation = archive.commit_operations({'weights': path}, prior)[0]
            self.assertEqual(operation.upload_info.sha256.hex(), hashlib.sha256(data).hexdigest())
            self.assertEqual(operation.upload_info.sample, data[:512])
            with operation.as_file() as stream:
                self.assertEqual(stream.read(), data)

    def test_existing_inventory_repairs_invalid_slug_without_changing_source(self):
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary)
            (out / 'inventory.json').write_text(json.dumps({'items': [{
                'id': 'fixed-id', 'repo_id': 'davidheineman/archive-group--student',
                'source': '/original/model', 'payload': '/original/model/hf-final'}]}))
            with patch.object(archive, 'OUT', out):
                value = archive.prepare()
            item = value['items'][0]
            self.assertEqual(item['repo_id'], 'davidheineman/archive-group-student')
            self.assertEqual(item['source'], '/original/model')
            self.assertTrue((out / 'repository-name-repairs.json').exists())

    def fixture(self, root, kind='hf-safetensors'):
        source = root / 'runs/old-teachers/group'
        payload = source / ('hf-final' if kind == 'hf-safetensors' else 'ckpt/iter_0000001')
        payload.mkdir(parents=True)
        weight = payload / 'weights.bin'
        weight.write_bytes(b'preserve the only model until verification')
        (source / 'training.log').write_text('retained training provenance')
        item = {'source': str(source), 'payload': str(payload), 'repo_id': 'davidheineman/test',
                'kind': kind, 'checkpoint_step': 1}
        receipt = {'status': 'evicting', 'files': {'weights.bin': {
            'path': str(weight), 'identity': archive.identity(weight), 'size': weight.stat().st_size,
            'sha256': hashlib.sha256(weight.read_bytes()).hexdigest(), 'git_sha1': archive.git_sha(weight)}}}
        return item, receipt, weight

    def test_eviction_removes_only_verified_payload_and_preserves_logs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            item, receipt, weight = self.fixture(root)
            with patch.object(archive, 'SCRATCH', root):
                archive.evict_payload(item, receipt)
            self.assertFalse(weight.exists())
            self.assertFalse(Path(item['payload']).exists())
            self.assertEqual((Path(item['source']) / 'training.log').read_text(), 'retained training provenance')

    def test_new_unrecorded_file_blocks_every_deletion(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            item, receipt, weight = self.fixture(root)
            (weight.parent / 'new-model.bin').write_bytes(b'new data')
            with patch.object(archive, 'SCRATCH', root), self.assertRaisesRegex(RuntimeError, 'Unrecorded'):
                archive.evict_payload(item, receipt)
            self.assertTrue(weight.exists())

    def test_changed_file_identity_blocks_eviction(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            item, receipt, weight = self.fixture(root)
            weight.write_bytes(b'changed after upload')
            with patch.object(archive, 'SCRATCH', root), self.assertRaisesRegex(RuntimeError, 'changed'):
                archive.evict_payload(item, receipt)
            self.assertTrue(weight.exists())

    def test_overlaid_original_card_is_retained_while_verified_weights_are_evicted(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            item, receipt, weight = self.fixture(root)
            original = weight.parent / 'README.md'
            original.write_text('original training metadata')
            staged = root / 'staged/README.md'
            staged.parent.mkdir()
            staged.write_text('generated archive metadata')
            item['initial_files'] = {'README.md': {'path': str(original), 'identity': archive.identity(original)}}
            receipt['files']['README.md'] = {'path': str(staged), 'identity': archive.identity(staged)}
            with patch.object(archive, 'SCRATCH', root):
                archive.evict_payload(item, receipt)
            self.assertFalse(weight.exists())
            self.assertEqual(original.read_text(), 'original training metadata')
            self.assertTrue(staged.exists())

    def test_changed_overlaid_original_card_blocks_every_deletion(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            item, receipt, weight = self.fixture(root)
            original = weight.parent / 'README.md'
            original.write_text('original')
            item['initial_files'] = {'README.md': {'path': str(original), 'identity': archive.identity(original)}}
            receipt['files']['README.md'] = {'path': str(root / 'staged/README.md')}
            original.write_text('modified with different bytes and length')
            with patch.object(archive, 'SCRATCH', root), self.assertRaisesRegex(RuntimeError, 'metadata changed'):
                archive.evict_payload(item, receipt)
            self.assertTrue(weight.exists())

    def test_uninventoried_readme_still_blocks_every_deletion(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            item, receipt, weight = self.fixture(root)
            (weight.parent / 'README.md').write_text('new unrecorded metadata')
            with patch.object(archive, 'SCRATCH', root), self.assertRaisesRegex(RuntimeError, 'Unrecorded'):
                archive.evict_payload(item, receipt)
            self.assertTrue(weight.exists())

    def test_live_reference_blocks_eviction_authorization(self):
        with tempfile.TemporaryDirectory() as temporary:
            item, receipt, weight = self.fixture(Path(temporary))
            for snapshot in [
                {'details': ['Command=' + item['source'] + '/train.sh'], 'documents': []},
                {'details': [], 'documents': [('plan.json', json.dumps({'teacher': item['payload']}))]},
            ]:
                with self.assertRaisesRegex(RuntimeError, 'Live'):
                    archive.assert_unused(item, snapshot)
            self.assertTrue(weight.exists())

    def test_async_runs_are_always_out_of_scope(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / 'runs/data-scaling-async-n16-original'
            payload = source / 'hf-final'
            payload.mkdir(parents=True)
            with patch.object(archive, 'SCRATCH', root), self.assertRaisesRegex(ValueError, 'scope'):
                archive.check_scope({'source': str(source), 'payload': str(payload), 'repo_id': 'davidheineman/test'})

    def test_nano_retains_json_results_and_restore_marker(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / 'nano-omni-multiteacher/runs-inloop-300000000-2000000/math'
            source.mkdir(parents=True)
            weight = source / 'model.pt';weight.write_bytes(b'weights')
            result = source / 'results.json';result.write_text('{}')
            marker = source / 'storage_archive_restore.json';marker.write_text('{}')
            (source / 'archive-storage.lock').touch()
            receipt = {'status': 'evicting', 'files': {str(i): {'path': str(f), 'identity': archive.identity(f)}
                                                     for i, f in enumerate([weight, result])}}
            item = {'source': str(source), 'payload': str(source), 'repo_id': 'davidheineman/test', 'kind': 'pytorch-checkpoint'}
            with patch.object(archive, 'SCRATCH', root):
                archive.evict_payload(item, receipt)
            self.assertFalse(weight.exists())
            self.assertTrue(result.exists())
            self.assertTrue(marker.exists())

    def test_remote_digest_mismatch_cannot_be_verified(self):
        api = SimpleNamespace(model_info=lambda *a, **kw: SimpleNamespace(sha='commit', siblings=[
            SimpleNamespace(rfilename='weights', size=5, lfs=SimpleNamespace(sha256='bad'), blob_id=None)]))
        receipt = {'revision': 'commit', 'files': {'weights': {'size': 5, 'sha256': 'expected', 'git_sha1': None}}}
        with self.assertRaisesRegex(ValueError, 'SHA256'):
            archive.verify_remote(api, {'repo_id': 'davidheineman/test'}, receipt)

    def test_live_inputs_expands_arrays_and_tolerates_only_completed_job_races(self):
        with tempfile.TemporaryDirectory() as temporary:
            with patch.object(archive, 'SCRATCH', Path(temporary)), \
                    patch.object(archive.subprocess, 'check_output', side_effect=['123_0\n', '']) as queue, \
                    patch.object(archive.subprocess, 'run', return_value=SimpleNamespace(
                        returncode=1, stdout='', stderr='Invalid job id specified')):
                snapshot = archive.live_inputs()
            self.assertIn('--array', queue.call_args_list[0].args[0])
            self.assertEqual(snapshot['details'], [])

    def test_live_inputs_refuses_to_ignore_an_unreadable_live_job(self):
        with tempfile.TemporaryDirectory() as temporary:
            with patch.object(archive, 'SCRATCH', Path(temporary)), \
                    patch.object(archive.subprocess, 'check_output', return_value='123_0\n'), \
                    patch.object(archive.subprocess, 'run', return_value=SimpleNamespace(
                        returncode=1, stdout='', stderr='controller unavailable')), \
                    self.assertRaisesRegex(RuntimeError, 'Unable to audit live job'):
                archive.live_inputs()


if __name__ == '__main__':
    unittest.main()
