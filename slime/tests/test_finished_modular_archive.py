"""Archive eviction remains closed until remote proof and unchanged local files exist."""

import hashlib
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from archive_finished_modular_checkpoints_20261103 import (
    evict_tensors,
    identity,
    verify_remote,
)


class ArchiveProofTests(unittest.TestCase):
    def fixture(self, root):
        checkpoint = root / "ckpt/iter_0000999"
        checkpoint.mkdir(parents=True)
        tensor, metadata = checkpoint / "weights.distcp", checkpoint / ".metadata"
        tensor.write_bytes(b"model optimizer RNG bytes")
        metadata.write_bytes(b"metadata")
        files = {
            str(path.relative_to(root)): {
                "local_path": str(path),
                "identity": identity(path),
                "size": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "git_blob_sha1": "blob",
            }
            for path in [tensor, metadata]
        }
        return checkpoint, tensor, metadata, {"status": "verified", "files": files}

    def test_unverified_or_changed_source_is_never_removed(self):
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint, tensor, metadata, receipt = self.fixture(Path(temporary))
            receipt["status"] = "uploading"
            with self.assertRaisesRegex(ValueError, "unverified"):
                evict_tensors(checkpoint, receipt)
            receipt["status"] = "verified"
            tensor.write_bytes(b"different checkpoint")
            with self.assertRaisesRegex(ValueError, "changed"):
                evict_tensors(checkpoint, receipt)
            self.assertTrue(tensor.exists())
            self.assertTrue(metadata.exists())

    def test_only_verified_tensor_files_are_evicted(self):
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint, tensor, metadata, receipt = self.fixture(Path(temporary))
            evict_tensors(checkpoint, receipt)
            self.assertFalse(tensor.exists())
            self.assertEqual(metadata.read_bytes(), b"metadata")

    def test_new_local_file_blocks_eviction(self):
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint, tensor, metadata, receipt = self.fixture(Path(temporary))
            (checkpoint / "new-training-step").touch()
            with self.assertRaisesRegex(ValueError, "new files"):
                evict_tensors(checkpoint, receipt)
            self.assertTrue(tensor.exists())

    def test_eviction_can_resume_after_a_tensor_was_removed(self):
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint, tensor, metadata, receipt = self.fixture(Path(temporary))
            receipt["status"] = "evicting"
            tensor.unlink()
            evict_tensors(checkpoint, receipt)
            self.assertTrue(metadata.exists())

    def test_remote_checksum_mismatch_blocks_verification(self):
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint, tensor, metadata, receipt = self.fixture(Path(temporary))
            remote = [
                SimpleNamespace(
                    rfilename=name,
                    size=value["size"],
                    lfs=SimpleNamespace(sha256="incorrect"),
                    blob_id=None,
                )
                for name, value in receipt["files"].items()
            ]
            api = SimpleNamespace(
                model_info=lambda *args, **kwargs: SimpleNamespace(
                    private=True, siblings=remote
                )
            )
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                verify_remote(api, "repo", "revision", receipt["files"])
            self.assertTrue(tensor.exists())


if __name__ == "__main__":
    unittest.main()
