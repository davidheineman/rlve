import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

source = Path(__file__).parents[2] / "tools/archive_scaling_opd_20261025.py"
spec = importlib.util.spec_from_file_location("scaling_archive_20261025_test", source)
archive = importlib.util.module_from_spec(spec)
spec.loader.exec_module(archive)


class ScalingArchiveTest(unittest.TestCase):
    def fixture(self, root):
        group = root / "experiment"
        leaf = group / "teacher" / "00-example"
        export = leaf / "hf-final"
        export.mkdir(parents=True)
        weight = export / "model.safetensors"
        weight.write_bytes(b"weights")
        marker = leaf / "weights_only_complete.json"
        marker.write_text('{}')
        item = {"id": "id", "source": str(leaf), "repo_id": "owner/model",
                "run": {"label": leaf.name}}
        receipt = {"status": "verified", "revision": "commit", "collection": "owner/collection",
                   "files": {weight.name: {"size": weight.stat().st_size,
                   "sha256": archive.digest(weight), "git_sha1": archive.git_sha(weight),
                   "path": str(weight), "fingerprint": archive.fingerprint(weight)}}}
        path = root / "receipt.json"
        archive.atomic(path, receipt)
        sibling = SimpleNamespace(rfilename=weight.name, size=weight.stat().st_size,
                                  lfs=SimpleNamespace(sha256=archive.digest(weight)))
        info = SimpleNamespace(sha="commit", siblings=[sibling])
        api = SimpleNamespace(model_info=lambda *a, **k: info)
        return group, leaf, item, receipt, path, info, api

    def test_corrupt_remote_never_deletes_local_weights(self):
        with tempfile.TemporaryDirectory() as d:
            group, leaf, item, receipt, path, info, api = self.fixture(Path(d))
            info.siblings[0].lfs.sha256 = 'wrong'
            with patch.object(archive, 'SOURCES', (group,)), patch.object(archive, 'receipt_path', return_value=path):
                with self.assertRaisesRegex(ValueError, 'SHA-256 mismatch'):
                    archive.evict_one(api, item)
            self.assertTrue((leaf / 'hf-final/model.safetensors').exists())
            self.assertTrue((leaf / 'weights_only_complete.json').exists())

    def test_changed_local_source_never_deletes_it(self):
        with tempfile.TemporaryDirectory() as d:
            group, leaf, item, receipt, path, info, api = self.fixture(Path(d))
            (leaf / 'hf-final/model.safetensors').write_bytes(b'changed weights')
            with patch.object(archive, 'SOURCES', (group,)), patch.object(archive, 'receipt_path', return_value=path):
                with self.assertRaisesRegex(RuntimeError, 'Local file changed'):
                    archive.evict_one(api, item)
            self.assertTrue((leaf / 'hf-final/model.safetensors').exists())

    def test_remote_revision_change_never_deletes_weights(self):
        with tempfile.TemporaryDirectory() as d:
            group, leaf, item, receipt, path, info, api = self.fixture(Path(d))
            info.sha = 'another-commit'
            with patch.object(archive, 'SOURCES', (group,)), patch.object(archive, 'receipt_path', return_value=path):
                with self.assertRaisesRegex(RuntimeError, 'Remote HEAD changed'):
                    archive.evict_one(api, item)
            self.assertTrue((leaf / 'hf-final/model.safetensors').exists())

    def test_verified_eviction_keeps_restore_receipt_and_is_idempotent(self):
        with tempfile.TemporaryDirectory() as d:
            group, leaf, item, receipt, path, info, api = self.fixture(Path(d))
            with patch.object(archive, 'SOURCES', (group,)), patch.object(archive, 'receipt_path', return_value=path):
                archive.evict_one(api, item)
                archive.evict_one(api, item)
            self.assertFalse((leaf / 'hf-final').exists())
            self.assertTrue((leaf / 'weights_only_complete.uploaded.json').exists())
            self.assertEqual(archive.read(leaf / 'uploaded_to_hf.json')['revision'], 'commit')
            self.assertEqual(archive.read(path)['status'], 'evicted')

    def test_partial_eviction_resumes_only_after_remote_reverification(self):
        with tempfile.TemporaryDirectory() as d:
            group, leaf, item, receipt, path, info, api = self.fixture(Path(d))
            receipt['status'] = 'evicting'
            archive.atomic(path, receipt)
            (leaf / 'hf-final/model.safetensors').unlink()
            with patch.object(archive, 'SOURCES', (group,)), patch.object(archive, 'receipt_path', return_value=path):
                archive.evict_one(api, item)
            self.assertFalse((leaf / 'hf-final').exists())
            self.assertEqual(archive.read(path)['status'], 'evicted')

    def test_collection_omission_prevents_archive_release(self):
        api = SimpleNamespace(get_collection=lambda slug: SimpleNamespace(items=[]))
        with self.assertRaisesRegex(ValueError, 'Collection mismatch'):
            archive.verify_collections(api, [{'repo_id': 'owner/model', 'role': 'teacher'}],
                                       {'teacher': {'slug': 'owner/teachers'}})

    def test_metadata_only_checkpoint_is_allowed_but_training_state_is_not(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            checkpoint = root / 'ckpt'
            checkpoint.mkdir()
            (checkpoint / 'rlve_checkpoint_state.json').write_text('{}')
            self.assertTrue(archive.checkpoint_metadata_only(root))
            (checkpoint / 'iter_0000010').mkdir()
            self.assertFalse(archive.checkpoint_metadata_only(root))

    def test_receipt_accepts_same_nfs_file_on_another_node(self):
        with tempfile.TemporaryDirectory() as d:
            group, leaf, item, receipt, path, info, api = self.fixture(Path(d))
            receipt['files']['model.safetensors']['fingerprint'][0] += 1000
            archive.atomic(path, receipt)
            with patch.object(archive, 'SOURCES', (group,)), patch.object(archive, 'receipt_path', return_value=path):
                archive.evict_one(api, item)
            self.assertEqual(archive.read(path)['status'], 'evicted')

    def test_missing_collection_entries_are_repaired_before_release(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            group, leaf, item, receipt, path, info, original_api = self.fixture(root)
            item.update(role='teacher', wandb_run_id='wandb-id')
            item['run']['kind'] = 'teacher'
            members = []
            def add(slug, repo, kind, **kwargs):
                members.append(SimpleNamespace(item_id=repo, item_type=kind))
            api = SimpleNamespace(model_info=original_api.model_info,
                                  get_collection=lambda slug: SimpleNamespace(items=list(members)),
                                  add_collection_item=add)
            with patch.object(archive, 'ARCHIVE', root), patch.object(archive, 'receipt_path', return_value=path):
                result = archive.reconcile_collections(api, [item], {'teacher': {'slug': 'owner/teachers'}})
            self.assertEqual(result, ['owner/model'])
            self.assertEqual([x.item_id for x in members], ['owner/model'])


if __name__ == '__main__':
    unittest.main()
