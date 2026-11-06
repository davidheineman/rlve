"""Guard cancelled-run deletion boundaries and test an actual provenance restore."""

import hashlib
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import patch

from tools import archive_cancelled_async_20261027 as archive
from tools import restore_cancelled_async_archive as restore


class CancelledAsyncArchiveTest(unittest.TestCase):
    def fixture(self, root):
        source = root / 'runs' / archive.SOURCES['n32']['name']
        source.mkdir(parents=True)
        path = source / 'model.bin'
        path.write_bytes(b'checkpoint bytes')
        record = {'path': 'model.bin', 'identity': archive.shared.identity(path)}
        receipt = {'status': 'verified', 'revision': 'pinned-commit', 'files': {'run/model.bin': {
            'path': str(path), 'identity': record['identity'], 'size': path.stat().st_size}}}
        return source, path, record, receipt

    def test_only_explicitly_named_cancelled_runs_are_allowed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with patch.object(archive.shared, 'SCRATCH', root):
                for name in ['async-lora-n16-20261025-095504', 'async-lora-n32-20261025-141924',
                             'data-scaling-async-n16-20261023-171338', 'unrelated']:
                    source = root / 'runs' / name
                    source.mkdir(parents=True)
                    with self.assertRaisesRegex(ValueError, 'scope'):
                        archive.allowed_source(source)

    def test_unverified_receipt_cannot_delete_a_file(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, path, record, receipt = self.fixture(root)
            receipt['status'] = 'hashed'
            with patch.object(archive.shared, 'SCRATCH', root), self.assertRaisesRegex(ValueError, 'verified'):
                archive.evict_records(source, [record], receipt)
            self.assertTrue(path.exists())

    def test_changed_batch_member_blocks_every_deletion(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, path, record, receipt = self.fixture(root)
            second = source / 'second.bin'
            second.write_bytes(b'original data')
            other = {'path': second.name, 'identity': archive.shared.identity(second)}
            receipt['files']['run/second.bin'] = {'path': str(second), 'identity': other['identity'],
                                                'size': second.stat().st_size}
            second.write_bytes(b'changed after upload, preserve both files')
            with patch.object(archive.shared, 'SCRATCH', root), self.assertRaisesRegex(RuntimeError, 'changed'):
                archive.evict_records(source, [record, other], receipt)
            self.assertTrue(path.exists())
            self.assertTrue(second.exists())

    def test_eviction_preserves_unrecorded_files_and_other_runs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, path, record, receipt = self.fixture(root)
            other = source / 'new-weight.bin'
            other.write_bytes(b'not inventoried or uploaded')
            with patch.object(archive.shared, 'SCRATCH', root):
                archive.evict_records(source, [record], receipt)
            self.assertFalse(path.exists())
            self.assertTrue(other.exists())

    def test_provenance_round_trip_preserves_state_replay_and_links(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, _, _, _ = self.fixture(root)
            state = source / 'state.json'
            state.write_text(json.dumps({'student_rollouts': 6800, 'optimizer': '/saved/state', 'rng': [1, 2, 3]}))
            inventory = {'root': str(source), 'files': [{'path': state.name,
                          'identity': archive.shared.identity(state)}]}
            out = root / 'staging'
            with patch.object(archive, 'OUT', out):
                package = archive.package_provenance({'key': 'test'}, inventory)
            destination = root / 'restored'
            destination.mkdir()
            manifest = {'source': str(source), 'files': package['files'],
                        'symlinks': [{'path': 'teacher-bank/model', 'target': str(source / 'state.json'),
                                      'directory': False}]}
            restore.extract_provenance(out / 'test/provenance.tar.gz', manifest, destination)
            restore.restore_links(manifest, destination)
            self.assertEqual(state.read_bytes(), (destination / 'state.json').read_bytes())
            self.assertEqual((destination / 'teacher-bank/model').resolve(), destination / 'state.json')

    def test_restore_rejects_path_traversal_and_external_links(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name in ['/outside', '../outside']:
                with self.assertRaisesRegex(ValueError, 'Unsafe'):
                    restore.safe_path(root, name)
            (root / 'link').symlink_to('/tmp', target_is_directory=True)
            with self.assertRaisesRegex(ValueError, 'Symlink'):
                restore.safe_path(root, 'link/model.bin')
            manifest = {'source': '/original', 'symlinks': [
                {'path': 'teacher-bank/model', 'target': '/unrelated/checkpoint', 'directory': False}]}
            with self.assertRaisesRegex(ValueError, 'external'):
                restore.restore_links(manifest, root)

    def test_bad_provenance_checksum_leaves_existing_state_unchanged(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            existing = root / 'state.json'
            existing.write_bytes(b'preserve existing state')
            package = root / 'bad.tar.gz'
            data = b'bad replacement'
            with tarfile.open(package, 'w:gz') as output:
                info = tarfile.TarInfo('state.json')
                info.size = len(data)
                output.addfile(info, io.BytesIO(data))
            manifest = {'files': {'state.json': {'size': len(data), 'sha256': hashlib.sha256(b'correct bytes').hexdigest(),
                                                'storage': 'provenance'}}}
            with self.assertRaisesRegex(ValueError, 'checksum'):
                restore.extract_provenance(package, manifest, root)
            self.assertEqual(existing.read_bytes(), b'preserve existing state')


if __name__ == '__main__':
    unittest.main()
