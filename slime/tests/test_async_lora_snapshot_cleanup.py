"""Busy NFS mappings defer reclamation without hiding unrelated storage errors."""

import errno
import mmap
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from async_lora.snapshot_cleanup import remove_retired, retire_snapshot


class SnapshotCleanupTest(unittest.TestCase):
    def test_only_the_obsolete_version_is_retired(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            old = root / "model/sampler_weights/old"
            current = old.with_name("current")
            for path in (old, current):
                path.mkdir(parents=True)
                (path / "adapter.safetensors").write_bytes(path.name.encode())
            retire_snapshot(root, old)
            self.assertFalse(old.exists())
            self.assertEqual((current / "adapter.safetensors").read_bytes(), b"current")
            self.assertFalse(list((root / ".retired/model").iterdir()))

    def test_busy_nfs_files_are_retried_after_the_mapping_is_released(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "retired"
            path.mkdir()
            (path / ".nfs012345").write_bytes(b"mapped weights")
            for code in (errno.EBUSY, errno.ENOTEMPTY):
                with patch("async_lora.snapshot_cleanup.shutil.rmtree", side_effect=OSError(code, "busy NFS file")):
                    self.assertFalse(remove_retired(path))
                self.assertEqual((path / ".nfs012345").read_bytes(), b"mapped weights")
            self.assertTrue(remove_retired(path))
            self.assertFalse(path.exists())

    def test_permission_and_unexpected_remaining_files_are_not_suppressed(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)
            (path / "unexpected-new-adapter").touch()
            for code in (errno.EACCES, errno.EIO, errno.ENOTEMPTY):
                with (
                    patch("async_lora.snapshot_cleanup.shutil.rmtree", side_effect=OSError(code, "storage error")),
                    self.assertRaises(OSError),
                ):
                    remove_retired(path)

    def test_paths_outside_a_version_directory_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for source in (root, root / "model", root / "model/sampler_weights", root.parent / "outside"):
                with self.assertRaises(ValueError):
                    retire_snapshot(root, source)

    @unittest.skipUnless(os.environ.get("LORA_NFS_TEST_ROOT"), "requires shared NFS scratch")
    def test_real_nfs_open_mapping_survives_retirement_and_is_eventually_collected(self):
        with tempfile.TemporaryDirectory(dir=os.environ["LORA_NFS_TEST_ROOT"]) as temporary:
            root = Path(temporary)
            source = root / "model/sampler_weights/old"
            source.mkdir(parents=True)
            adapter = source / "adapter.safetensors"
            adapter.write_bytes(b"immutable adapter weights")
            with adapter.open("rb") as stream, mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ) as mapping:
                retire_snapshot(root, source)
                self.assertEqual(mapping[:], b"immutable adapter weights")
                self.assertFalse(source.exists())
                retired = list((root / ".retired/model").iterdir())
                self.assertEqual(len(retired), 1)
                self.assertTrue(any(path.name.startswith(".nfs") for path in retired[0].iterdir()))
            # A subsequent export also reaps earlier deferred versions.
            next_source = source.with_name("next-old")
            next_source.mkdir()
            retire_snapshot(root, next_source)
            self.assertFalse(list((root / ".retired/model").iterdir()))


if __name__ == "__main__":
    unittest.main()
